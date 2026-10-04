// Finite-difference gradient check for the SkePU DNN library.
// Compares analytic gradients (from each layer's backward pass) with
// central finite differences of the loss, for small models.
//
// Usage: dnn_gradcheck [epsilon]   (default 3e-3; finite differences get noisy for large steps and near ReLU kinks)
// Build (sequential, no precompiler):
//   clang++ -std=c++20 -O1 -I ../skepu-headers/src -Wno-attributes dnn_gradcheck.cpp -o dnn_gradcheck
#include <iostream>
#include <random>
#include <cmath>
#include <functional>

#define SKEPU_ENABLE_EXCEPTIONS

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/ml.hpp>

using namespace skepu::dnn;

static std::mt19937 rng(1234);
static float g_eps = 3e-3f; // finite-difference step (first command-line argument)

static void fill_random(skepu::Tensor4<float> &t, float lo, float hi)
{
	std::uniform_real_distribution<float> d(lo, hi);
	for (size_t i = 0; i < t.size_i(); ++i)
		for (size_t j = 0; j < t.size_j(); ++j)
			for (size_t k = 0; k < t.size_k(); ++k)
				for (size_t l = 0; l < t.size_l(); ++l)
					t(i, j, k, l) = d(rng);
}

// One-hot (or binary) labels
static void fill_labels(skepu::Tensor4<float> &y)
{
	for (size_t i = 0; i < y.size_i(); ++i)
	{
		size_t c = rng() % y.size_l();
		for (size_t l = 0; l < y.size_l(); ++l)
			y(i, 0, 0, l) = (y.size_l() == 1) ? (float)(rng() % 2) : (l == c ? 1.f : 0.f);
	}
}

template <typename Model>
static float loss_of(Model &model, skepu::dnn::impl::LossFunction<float> &lf, skepu::Tensor4<float> &x, skepu::Tensor4<float> &y)
{
	auto &pred = model.forward(x, false);
	return (float)lf.compute_loss(y, pred, x.size_i());
}

// Accessors for a layer's parameters and gradients, whichever its type
struct ParamRef
{
	std::function<float&(size_t)> w, b;   // flat parameter access
	std::function<float&(size_t)> dw, db; // flat gradient access
	size_t nw, nb;
};

static bool get_params(skepu::dnn::impl::Layer<float> &layer, ParamRef &r)
{
	using namespace skepu::dnn::impl;
	if (auto *c = dynamic_cast<skepu::dnn::impl::Conv2D<float>*>(&layer))
	{
		auto &W = c->get_weights(); auto &dW = c->get_gradients();
		auto &B = c->get_biases();  auto &dB = c->get_biases_gradients();
		W.updateHost(); dW.updateHost(); B.updateHost(); dB.updateHost();
		r.nw = W.size(); r.nb = B.size();
		r.w  = [&W](size_t i) -> float& { return W[i]; };
		r.dw = [&dW](size_t i) -> float& { return dW[i]; };
		r.b  = [&B](size_t i) -> float& { return B[i]; };
		r.db = [&dB](size_t i) -> float& { return dB[i]; };
		return true;
	}
	if (auto *d = dynamic_cast<skepu::dnn::impl::Dense<float>*>(&layer))
	{
		auto &W = d->get_d_weights(); auto &dW = d->get_d_gradients();
		auto &B = d->get_biases();    auto &dB = d->get_biases_gradients();
		W.updateHost(); dW.updateHost(); B.updateHost(); dB.updateHost();
		r.nw = W.size(); r.nb = B.size();
		r.w  = [&W](size_t i) -> float& { return W[i]; };
		r.dw = [&dW](size_t i) -> float& { return dW[i]; };
		r.b  = [&B](size_t i) -> float& { return B[i]; };
		r.db = [&dB](size_t i) -> float& { return dB[i]; };
		return true;
	}
	return false;
}

template <typename Model>
static bool gradcheck(const std::string &name, Model &model, Loss loss, skepu::Tensor4<float> &x, skepu::Tensor4<float> &y, float bias_shift = 0)
{
	std::cout << "=== " << name << " ===\n";
	model.init();
	model.init_parameters();
	// Optionally shift all biases: keeps ReLU units away from their kink at 0, where
	// finite differences are unreliable
	if (bias_shift != 0)
		for (size_t idx = 0; idx < model.layerCount(); ++idx)
		{
			ParamRef r;
			if (get_params(model.getLayer(idx), r))
				for (size_t i = 0; i < r.nb; ++i) r.b(i) += bias_shift;
		}
	skepu::dnn::impl::LossFunction<float> lf(loss);

	// Analytic gradients (backward pass without parameter updates)
	auto &pred = model.forward(x, false);
	auto grad = lf.compute_gradient(pred, y);
	skepu::Tensor4<float> *g = &grad;
	size_t n = model.layerCount();
	for (size_t idx = n; idx-- > 0;)
	{
		auto &layer = model.getLayer(idx);
		skepu::Tensor4<float> *in = (idx > 0) ? &model.getLayer(idx - 1).m_vals : &x;
		g = layer.backward(in, g, &y);
	}

	bool ok = true;
	const float eps = g_eps;
	for (size_t idx = 0; idx < n; ++idx)
	{
		ParamRef r;
		if (!get_params(model.getLayer(idx), r)) continue;

		// Snapshot analytic gradients (forward passes below don't change them, but be safe)
		std::vector<float> adw(r.nw), adb(r.nb);
		for (size_t i = 0; i < r.nw; ++i) adw[i] = r.dw(i);
		for (size_t i = 0; i < r.nb; ++i) adb[i] = r.db(i);

		double max_err = 0; size_t checked = 0;
		auto check = [&](float &p, float analytic, const char *kind, size_t i)
		{
			float old = p;
			p = old + eps; float lp = loss_of(model, lf, x, y);
			p = old - eps; float lm = loss_of(model, lf, x, y);
			p = old;
			float numeric = (lp - lm) / (2 * eps);
			double err = std::fabs(numeric - analytic) / std::max(1e-2, (double)(std::fabs(numeric) + std::fabs(analytic)));
			max_err = std::max(max_err, err);
			++checked;
			if (err > 0.05)
				std::cout << "  layer " << idx << " " << kind << "[" << i << "]: analytic " << analytic << " numeric " << numeric << "\n";
		};
		size_t stepw = std::max<size_t>(1, r.nw / 40);
		for (size_t i = 0; i < r.nw; i += stepw) check(r.w(i), adw[i], "W", i);
		for (size_t i = 0; i < r.nb; ++i)        check(r.b(i), adb[i], "b", i);
		bool layer_ok = max_err < 0.05;
		ok = ok && layer_ok;
		std::cout << "layer " << idx << ": " << checked << " params checked, max rel err " << max_err << (layer_ok ? "  OK" : "  FAIL") << "\n";
	}
	return ok;
}

// Adam optimizer step against a straightforward host implementation of the Adam update rule
template <typename Model>
static bool adam_check(const std::string &name, Model &model, size_t layer_idx)
{
	std::cout << "=== " << name << " ===\n";
	model.init();
	model.init_parameters();
	auto &layer = model.getLayer(layer_idx);
	ParamRef r;
	get_params(layer, r);

	const float lr = 0.01f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
	layer.m_learning_rate = lr;
	skepu::dnn::impl::Optimizer<float> opt(Opt::Adam, b1, b2, eps);

	std::vector<double> w(r.nw), b(r.nb), mw(r.nw, 0), vw(r.nw, 0), mb(r.nb, 0), vb(r.nb, 0);
	for (size_t i = 0; i < r.nw; ++i) w[i] = r.w(i);
	for (size_t i = 0; i < r.nb; ++i) b[i] = r.b(i);

	std::uniform_real_distribution<float> d(-1, 1);
	auto host_adam = [&](double &p, double g, double &m, double &v, int t)
	{
		m = b1 * m + (1 - b1) * g;
		v = b2 * v + (1 - b2) * g * g;
		p -= lr * (m / (1 - std::pow((double)b1, t))) / (std::sqrt(v / (1 - std::pow((double)b2, t))) + eps);
	};

	double max_diff = 0;
	for (int t = 1; t <= 5; ++t)
	{
		std::vector<float> gw(r.nw), gb(r.nb);
		for (size_t i = 0; i < r.nw; ++i) { gw[i] = d(rng); r.dw(i) = gw[i]; }
		for (size_t i = 0; i < r.nb; ++i) { gb[i] = d(rng); r.db(i) = gb[i]; }
		opt.step(layer);
		get_params(layer, r); // refresh host copies after the skeleton calls
		for (size_t i = 0; i < r.nw; ++i) host_adam(w[i], gw[i], mw[i], vw[i], t);
		for (size_t i = 0; i < r.nb; ++i) host_adam(b[i], gb[i], mb[i], vb[i], t);
		for (size_t i = 0; i < r.nw; ++i) max_diff = std::max(max_diff, std::fabs(w[i] - r.w(i)));
		for (size_t i = 0; i < r.nb; ++i) max_diff = std::max(max_diff, std::fabs(b[i] - r.b(i)));
	}
	bool ok = max_diff < 1e-5;
	std::cout << "max |host - skeleton| over 5 Adam steps: " << max_diff << (ok ? "  OK" : "  FAIL") << "\n";
	return ok;
}

int main(int argc, char *argv[])
{
	if (argc > 1) g_eps = atof(argv[1]);
	skepu::BackendSpec spec{"cpu"};
	skepu::setGlobalBackendSpec(spec);
	bool ok = true;
	const size_t B = 4;

	{ // MLP, tanh hidden, softmax output, categorical cross-entropy
		auto model = SequentialModel(Dimensions{B, 1, 1, 8})
			<< Dense(6).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
			<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::SoftMax);
		skepu::Tensor4<float> x(B, 1, 1, 8), y(B, 1, 1, 3);
		fill_random(x, -1, 1); fill_labels(y);
		ok &= gradcheck("MLP tanh+softmax / CCE", model, Loss::CategoricalCrossEntropy, x, y);
	}
	{ // MLP, sigmoid output, binary cross-entropy
		auto model = SequentialModel(Dimensions{B, 1, 1, 8})
			<< Dense(5).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::Sigmoid)
			<< Dense(1).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::Sigmoid);
		skepu::Tensor4<float> x(B, 1, 1, 8), y(B, 1, 1, 1);
		fill_random(x, -1, 1); fill_labels(y);
		ok &= gradcheck("MLP sigmoid / BCE", model, Loss::BinaryCrossEntropy, x, y);
	}
	{ // CNN: conv + tanh, maxpool, flatten, dense + sigmoid
		auto model = SequentialModel(Dimensions{B, 8, 8, 1})
			<< Conv2D(3).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
			<< MaxPooling2D().kernel(2, 2)
			<< Flatten()
			<< Dense(1).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::Sigmoid);
		skepu::Tensor4<float> x(B, 8, 8, 1), y(B, 1, 1, 1);
		fill_random(x, 0, 1); fill_labels(y);
		ok &= gradcheck("CNN tanh+maxpool+sigmoid / BCE", model, Loss::BinaryCrossEntropy, x, y);
	}
	{ // CNN with two conv layers, multi-channel input, ReLU, softmax output
		auto model = SequentialModel(Dimensions{B, 8, 8, 2})
			<< Conv2D(3).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::ReLU)
			<< Conv2D(2).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::ReLU)
			<< Flatten()
			<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::SoftMax);
		skepu::Tensor4<float> x(B, 8, 8, 2), y(B, 1, 1, 3);
		fill_random(x, 0, 1); fill_labels(y);
		ok &= gradcheck("CNN 2xconv ReLU+softmax / CCE", model, Loss::CategoricalCrossEntropy, x, y, 0.5f);
	}

	{ // 1D CNN: conv1d + tanh, maxpool1d, flatten, dense + softmax (sequence input: batch x 1 x length x channels)
		auto model = SequentialModel(Dimensions{B, 1, 12, 2})
			<< Conv1D(3).kernel(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
			<< MaxPooling1D().kernel(2)
			<< Conv1D(2).kernel(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
			<< Flatten()
			<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::SoftMax);
		skepu::Tensor4<float> x(B, 1, 12, 2), y(B, 1, 1, 3);
		fill_random(x, -1, 1); fill_labels(y);
		ok &= gradcheck("CNN1D 2xConv1D tanh+maxpool1d+softmax / CCE", model, Loss::CategoricalCrossEntropy, x, y);
	}
	{ // 1D CNN with ReLU and an odd kernel size 5, single channel
		auto model = SequentialModel(Dimensions{B, 1, 16, 1})
			<< Conv1D(3).kernel(5).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::ReLU)
			<< Flatten()
			<< Dense(1).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::Sigmoid);
		skepu::Tensor4<float> x(B, 1, 16, 1), y(B, 1, 1, 1);
		fill_random(x, 0, 1); fill_labels(y);
		ok &= gradcheck("CNN1D Conv1D k5 ReLU / BCE", model, Loss::BinaryCrossEntropy, x, y, 0.5f);
	}
	{ // Adam step on a Dense layer (Matrix weights) and a Conv2D layer (Tensor4 weights)
		auto dense_model = SequentialModel(Dimensions{B, 1, 1, 6}) << Dense(5).weights(Init::Xavier_uniform).biases(Init::Uniform);
		ok &= adam_check("Adam step, Dense", dense_model, 0);
		auto conv_model = SequentialModel(Dimensions{B, 6, 6, 2}) << Conv2D(3).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Uniform);
		ok &= adam_check("Adam step, Conv2D", conv_model, 0);
	}

	std::cout << (ok ? "\nALL GRADIENT CHECKS PASSED\n" : "\nSOME GRADIENT CHECKS FAILED\n");
	return ok ? 0 : 1;
}
