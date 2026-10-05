// Tests of the SkePU-DNN library (skepu-lib/dnn), sequential implementation.
//
//  * finite-difference gradient checks of every layer type (Dense, Conv2D, Conv1D, MaxPooling2D/1D,
//    Flatten, activations) through the full backward pass,
//  * the Adam optimizer step against a host reference,
//  * predict() and evaluate() on datasets that are not a multiple of the batch size,
//  * a small synthetic end-to-end training run (Conv1D) that must reach a minimum accuracy.
//
// None of these need data files. For regression tests on MNIST and CIFAR-10 logs, see dnn_regression/.
#include "../../external/catch2/catch.hpp"

#define SKEPU_ENABLE_EXCEPTIONS
#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/ml.hpp>

using namespace skepu::dnn;

static std::mt19937 rng(1234);

// The tests run on the sequential implementation by default. When built with the precompiler, the
// backend can be selected with the environment variable SKEPU_TEST_BACKEND (CPU, OpenMP, OpenCL, ...).
static void use_sequential_cpu()
{
	const char *backend = std::getenv("SKEPU_TEST_BACKEND");
	static skepu::BackendSpec spec{backend ? backend : "cpu"};
	skepu::setGlobalBackendSpec(spec);
	rng.seed(1234);
}
static const float g_eps = 3e-3f; // finite-difference step; large steps are noisy near ReLU kinks

// std::uniform_real_distribution and std::normal_distribution are not specified exactly and give different
// numbers with different standard libraries, std::mt19937 is. Building the distributions on its raw output
// makes the data (and the logs of the regression test) the same with every compiler.
static float uniform01(std::mt19937 &rng) { return (float)(rng() >> 8) * (1.0f / 16777216.0f); } // [0, 1)
static float uniform(std::mt19937 &rng, float lo, float hi) { return lo + (hi - lo) * uniform01(rng); }
static float normal(std::mt19937 &rng, float sigma) // Box-Muller
{
	const float u1 = 1.0f - uniform01(rng), u2 = uniform01(rng);
	return sigma * std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}

static void fill_random(skepu::Tensor4<float> &t, float lo, float hi)
{
	for (size_t i = 0; i < t.size_i(); ++i)
		for (size_t j = 0; j < t.size_j(); ++j)
			for (size_t k = 0; k < t.size_k(); ++k)
				for (size_t l = 0; l < t.size_l(); ++l)
					t(i, j, k, l) = uniform(rng, lo, hi);
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
	std::function<void()> touch;           // call after writing through w/b/dw/db (skepu::external write): marks the device copies as outdated
	size_t nw, nb;
};

static bool get_params(skepu::dnn::impl::Layer<float> &layer, ParamRef &r)
{
	using namespace skepu::dnn::impl;
	if (auto *c = dynamic_cast<skepu::dnn::impl::Conv2D<float>*>(&layer))
	{
		auto &W = c->get_weights(); auto &dW = c->get_gradients();
		auto &B = c->get_biases();  auto &dB = c->get_biases_gradients();
		skepu::external(skepu::read(W, dW, B, dB), [] {});
		r.nw = W.size(); r.nb = B.size();
		r.w  = [&W](size_t i) -> float& { return W[i]; };
		r.dw = [&dW](size_t i) -> float& { return dW[i]; };
		r.b  = [&B](size_t i) -> float& { return B[i]; };
		r.db = [&dB](size_t i) -> float& { return dB[i]; };
		r.touch = [&W, &dW, &B, &dB] { skepu::external([] {}, skepu::write(W, dW, B, dB)); };
		return true;
	}
	if (auto *d = dynamic_cast<skepu::dnn::impl::Dense<float>*>(&layer))
	{
		auto &W = d->get_d_weights(); auto &dW = d->get_d_gradients();
		auto &B = d->get_biases();    auto &dB = d->get_biases_gradients();
		skepu::external(skepu::read(W, dW, B, dB), [] {});
		r.nw = W.size(); r.nb = B.size();
		r.w  = [&W](size_t i) -> float& { return W[i]; };
		r.dw = [&dW](size_t i) -> float& { return dW[i]; };
		r.b  = [&B](size_t i) -> float& { return B[i]; };
		r.db = [&dB](size_t i) -> float& { return dB[i]; };
		r.touch = [&W, &dW, &B, &dB] { skepu::external([] {}, skepu::write(W, dW, B, dB)); };
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
				{
					for (size_t i = 0; i < r.nb; ++i) r.b(i) += bias_shift;
					r.touch();
				}
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
			p = old + eps; r.touch(); float lp = loss_of(model, lf, x, y);
			p = old - eps; r.touch(); float lm = loss_of(model, lf, x, y);
			p = old; r.touch();
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
	skepu::dnn::impl::Optimizer<float> opt(Opt::Adam, b1, b2, eps);

	std::vector<double> w(r.nw), b(r.nb), mw(r.nw, 0), vw(r.nw, 0), mb(r.nb, 0), vb(r.nb, 0);
	for (size_t i = 0; i < r.nw; ++i) w[i] = r.w(i);
	for (size_t i = 0; i < r.nb; ++i) b[i] = r.b(i);

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
		for (size_t i = 0; i < r.nw; ++i) { gw[i] = uniform(rng, -1, 1); r.dw(i) = gw[i]; }
		for (size_t i = 0; i < r.nb; ++i) { gb[i] = uniform(rng, -1, 1); r.db(i) = gb[i]; }
		r.touch();
		opt.step(layer, lr);
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

// predict() on a dataset whose size is not a multiple of the batch size must equal the batch-wise forward pass
static bool predict_check()
{
	std::cout << "=== predict() with a remainder batch ===\n";
	const size_t B = 4, N = 11; // 2 full batches + remainder 3
	auto model = SequentialModel(Dimensions{B, 1, 1, 6})
		<< Dense(5).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
		<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::SoftMax);
	model.init();
	model.init_parameters();

	skepu::Tensor4<float> x(N, 1, 1, 6), y(N, 1, 1, 3);
	fill_random(x, -1, 1); fill_labels(y);
	skepu::Tensor4<float> pred = model.predict(x);

	double max_diff = 0;
	bool shape_ok = pred.size_i() == N && pred.size_l() == 3;
	for (size_t start : {(size_t)0, (size_t)4, N - B}) // full batches and the overlapping last batch
	{
		skepu::Tensor4<float> bx(B, 1, 1, 6);
		for (size_t i = 0; i < B; ++i)
			for (size_t l = 0; l < 6; ++l)
				bx(i, 0, 0, l) = x(start + i, 0, 0, l);
		auto &out = model.forward(bx, false);
		skepu::external(skepu::read(out), [] {});
		for (size_t i = 0; i < B; ++i)
			for (size_t l = 0; l < 3; ++l)
				max_diff = std::max(max_diff, (double)std::fabs(out(i, 0, 0, l) - pred(start + i, 0, 0, l)));
	}
	bool ok = shape_ok && max_diff < 1e-6;
	std::cout << "shape " << pred.size_i() << " x " << pred.size_l() << ", max |predict - forward| = " << max_diff << (ok ? "  OK" : "  FAIL") << "\n";
	return ok;
}


// Per-sample synthetic signals: four waveform classes with random period, phase and amplitude plus noise
static void generate_signals(skepu::Tensor4<float> &x, skepu::Tensor4<float> &y, float noise)
{
	const size_t length = x.size_k(), classes = y.size_l();
	const float two_pi = 6.28318530718f;
	for (size_t i = 0; i < x.size_i(); ++i)
	{
		const size_t cls = i % classes;
		const float period = uniform(rng, 8.f, 20.f), phase = uniform(rng, 0.f, 1.f), amp = uniform(rng, 0.7f, 1.3f);
		for (size_t t = 0; t < length; ++t)
		{
			float u = (float)t / period + phase;
			u -= std::floor(u);
			float v = 0;
			switch (cls)
			{
			case 0: v = std::sin(two_pi * u); break;
			case 1: v = (u < 0.5f) ? 1.f : -1.f; break;
			case 2: v = 2.f * u - 1.f; break;
			case 3: v = 1.f - 4.f * std::fabs(u - 0.5f); break;
			}
			x(i, 0, t, 0) = amp * v + normal(rng, noise);
		}
		for (size_t c = 0; c < classes; ++c)
			y(i, 0, 0, c) = (c == cls) ? 1.f : 0.f;
	}
}


TEST_CASE("DNN gradients: MLP, tanh + softmax, categorical cross-entropy")
{
	use_sequential_cpu();
	const size_t B = 4;
	auto model = SequentialModel(Dimensions{B, 1, 1, 8})
		<< Dense(6).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
		<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::SoftMax);
	skepu::Tensor4<float> x(B, 1, 1, 8), y(B, 1, 1, 3);
	fill_random(x, -1, 1); fill_labels(y);
	REQUIRE(gradcheck("MLP tanh+softmax / CCE", model, Loss::CategoricalCrossEntropy, x, y));
}

TEST_CASE("DNN gradients: MLP, sigmoid, binary cross-entropy")
{
	use_sequential_cpu();
	const size_t B = 4;
	auto model = SequentialModel(Dimensions{B, 1, 1, 8})
		<< Dense(5).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::Sigmoid)
		<< Dense(1).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::Sigmoid);
	skepu::Tensor4<float> x(B, 1, 1, 8), y(B, 1, 1, 1);
	fill_random(x, -1, 1); fill_labels(y);
	REQUIRE(gradcheck("MLP sigmoid / BCE", model, Loss::BinaryCrossEntropy, x, y));
}

TEST_CASE("DNN gradients: CNN with max pooling")
{
	use_sequential_cpu();
	const size_t B = 4;
	auto model = SequentialModel(Dimensions{B, 8, 8, 1})
		<< Conv2D(3).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
		<< MaxPooling2D().kernel(2, 2)
		<< Flatten()
		<< Dense(1).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::Sigmoid);
	skepu::Tensor4<float> x(B, 8, 8, 1), y(B, 1, 1, 1);
	fill_random(x, 0, 1); fill_labels(y);
	REQUIRE(gradcheck("CNN tanh+maxpool+sigmoid / BCE", model, Loss::BinaryCrossEntropy, x, y));
}

TEST_CASE("DNN gradients: two conv layers, multiple channels, ReLU")
{
	use_sequential_cpu();
	const size_t B = 4;
	auto model = SequentialModel(Dimensions{B, 8, 8, 2})
		<< Conv2D(3).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::ReLU)
		<< Conv2D(2).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::ReLU)
		<< Flatten()
		<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::SoftMax);
	skepu::Tensor4<float> x(B, 8, 8, 2), y(B, 1, 1, 3);
	fill_random(x, 0, 1); fill_labels(y);
	// Biases are shifted by +0.5 to keep ReLU units away from their kink at 0
	REQUIRE(gradcheck("CNN 2xconv ReLU+softmax / CCE", model, Loss::CategoricalCrossEntropy, x, y, 0.5f));
}

TEST_CASE("DNN gradients: Conv1D and MaxPooling1D")
{
	use_sequential_cpu();
	const size_t B = 4;
	{
		auto model = SequentialModel(Dimensions{B, 1, 12, 2})
			<< Conv1D(3).kernel(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
			<< MaxPooling1D().kernel(2)
			<< Conv1D(2).kernel(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
			<< Flatten()
			<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::SoftMax);
		skepu::Tensor4<float> x(B, 1, 12, 2), y(B, 1, 1, 3);
		fill_random(x, -1, 1); fill_labels(y);
		REQUIRE(gradcheck("CNN1D 2xConv1D tanh+maxpool1d+softmax / CCE", model, Loss::CategoricalCrossEntropy, x, y));
	}
	{
		auto model = SequentialModel(Dimensions{B, 1, 16, 1})
			<< Conv1D(3).kernel(5).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::ReLU)
			<< Flatten()
			<< Dense(1).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::Sigmoid);
		skepu::Tensor4<float> x(B, 1, 16, 1), y(B, 1, 1, 1);
		fill_random(x, 0, 1); fill_labels(y);
		REQUIRE(gradcheck("CNN1D Conv1D k5 ReLU / BCE", model, Loss::BinaryCrossEntropy, x, y, 0.5f));
	}
}

TEST_CASE("DNN Adam optimizer matches a host reference")
{
	use_sequential_cpu();
	const size_t B = 4;
	auto dense_model = SequentialModel(Dimensions{B, 1, 1, 6}) << Dense(5).weights(Init::Xavier_uniform).biases(Init::Uniform);
	REQUIRE(adam_check("Adam step, Dense", dense_model, 0));
	auto conv_model = SequentialModel(Dimensions{B, 6, 6, 2}) << Conv2D(3).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Uniform);
	REQUIRE(adam_check("Adam step, Conv2D", conv_model, 0));
}

TEST_CASE("DNN predict() on a dataset that is not a multiple of the batch size")
{
	use_sequential_cpu();
	REQUIRE(predict_check());
}

TEST_CASE("DNN evaluate() counts every sample exactly once")
{
	use_sequential_cpu();
	const size_t B = 4, N = 11; // 2 full batches + remainder 3
	auto model = SequentialModel(Dimensions{B, 1, 1, 6})
		<< Dense(5).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
		<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::SoftMax);
	model.init();
	model.init_parameters();
	model.configure(Loss::CategoricalCrossEntropy, Opt::SGD);

	skepu::Tensor4<float> x(N, 1, 1, 6), y(N, 1, 1, 3);
	fill_random(x, -1, 1); fill_labels(y);
	skepu::Matrix<int> confusion(3, 3, "confusion", 0);
	auto score = model.evaluate(x, y, &confusion);

	int total = 0, correct = 0;
	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; ++j)
		{
			total += confusion(i, j);
			if (i == j) correct += confusion(i, j);
		}
	REQUIRE(total == (int)N);
	REQUIRE((float)std::get<1>(score) == Approx((float)correct / N).margin(1e-6));
	REQUIRE((float)std::get<0>(score) > 0.f);
}

TEST_CASE("DNN reports misuse with clear errors")
{
	use_sequential_cpu();
	auto model = SequentialModel(Dimensions{4, 1, 1, 6}) << Dense(3).activation(Act::SoftMax);
	skepu::Tensor4<float> x(8, 1, 1, 6, "x", 0.5f), y(8, 1, 1, 3, "y", 0.f);
	skepu::Tensor4<float> small_x(2, 1, 1, 6, "small x", 0.f), small_y(2, 1, 1, 3, "small y", 0.f);

	model.init();
	model.init(); // idempotent

	// evaluate()/fit() before configure()
	REQUIRE_THROWS(model.evaluate(x, y));
	REQUIRE_THROWS(model.fit(x, y, 1, 0.1f, 0.0f, "unused", false));

	// fewer samples than one batch
	model.configure(Loss::CategoricalCrossEntropy, Opt::SGD);
	REQUIRE_THROWS(model.evaluate(small_x, small_y));
	REQUIRE_THROWS(model.predict(small_x));
}

TEST_CASE("DNN trains a Conv1D model on synthetic signals")
{
	use_sequential_cpu();
	const size_t length = 64, classes = 4, B = 32;
	skepu::Tensor4<float> train_x(960, 1, length, 1), train_y(960, 1, 1, classes);
	skepu::Tensor4<float> test_x(256, 1, length, 1), test_y(256, 1, 1, classes);
	generate_signals(train_x, train_y, 0.3f);
	generate_signals(test_x, test_y, 0.3f);

	auto model = SequentialModel(Dimensions{B, 1, length, 1})
		<< Conv1D(8).kernel(5).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU)
		<< MaxPooling1D().kernel(2)
		<< Conv1D(16).kernel(5).weights(Init::He_uniform).biases(Init::Zero).activation(Act::ReLU)
		<< MaxPooling1D().kernel(2)
		<< Flatten()
		<< Dense(classes).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	model.init();
	model.configure(Loss::CategoricalCrossEntropy, Opt::Adam);
	model.fit(train_x, train_y, 8, 0.003f, 0.1f, "signals", false);

	auto score = model.evaluate(test_x, test_y);
	const float loss = std::get<0>(score), accuracy = std::get<1>(score);
	std::cout << "synthetic signals: test loss " << loss << ", test accuracy " << accuracy << "\n";
	// Chance level is 0.25; the network reaches about 0.8 or more here
	REQUIRE(accuracy > 0.6f);
	REQUIRE(loss < std::log(4.f));
}


static std::string read_file(const std::string &path)
{
	std::ifstream in(path);
	std::stringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

TEST_CASE("DNN model files: save and load round trip")
{
	use_sequential_cpu();
	const size_t B = 4, N = 8;
	auto model = SequentialModel(Dimensions{B, 8, 8, 2})
		<< Conv2D(3).kernel(3, 3).weights(Init::He_uniform).biases(Init::Uniform).activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Flatten()
		<< Dropout(0.25f)
		<< Dense(5).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::TanH)
		<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::SoftMax);
	model.init();
	model.init_parameters();

	skepu::Tensor4<float> x(N, 8, 8, 2);
	fill_random(x, 0, 1);
	skepu::Tensor4<float> expected = model.predict(x);

	const std::string path = (std::filesystem::temp_directory_path() / "skepu_dnn_test_model.txt").string();
	model.save(path);

	for (size_t batch : {B, (size_t)3}) // the batch size is not part of the model file
	{
		auto loaded = load_model(path, batch);
		skepu::Tensor4<float> got = loaded.predict(x);
		REQUIRE(got.size() == expected.size());
		double max_diff = 0;
		for (size_t i = 0; i < N; ++i)
			for (size_t l = 0; l < 3; ++l)
				max_diff = std::max(max_diff, (double)std::fabs(got(i, 0, 0, l) - expected(i, 0, 0, l)));
		REQUIRE(max_diff == 0.0); // parameters are written with enough digits to reproduce floats exactly
	}

	// Saving the loaded model gives the same file
	auto loaded = load_model(path, B);
	const std::string path2 = path + ".2";
	loaded.save(path2);
	REQUIRE(read_file(path) == read_file(path2));

	std::filesystem::remove(path);
	std::filesystem::remove(path2);
}

TEST_CASE("DNN model files: description without parameters, and errors")
{
	use_sequential_cpu();
	const std::string dir = std::filesystem::temp_directory_path().string();
	auto write = [&](const std::string &name, const std::string &text)
	{
		std::string path = dir + "/" + name;
		std::ofstream(path) << text;
		return path;
	};

	// A hand-written description: initialized from the layers' init attributes, 1D layers, comments
	std::string ok = write("skepu_dnn_desc.txt",
		"skepu-dnn-model 1\n"
		"input 1 32 1   # sequence of 32 samples\n"
		"layer Conv1D features=4 kernel=5 weights=He_uniform biases=Zero activation=ReLU\n"
		"layer MaxPooling1D kernel=2\n"
		"layer Flatten\n"
		"layer Dense units=3 activation=SoftMax\n");
	auto model = load_model(ok, 4);
	skepu::Tensor4<float> x(4, 1, 32, 1);
	fill_random(x, -1, 1);
	skepu::Tensor4<float> out = model.predict(x);
	REQUIRE(out.size_l() == 3);
	float sum = 0;
	for (size_t l = 0; l < 3; ++l) sum += out(0, 0, 0, l);
	REQUIRE(sum == Approx(1.f).margin(1e-5)); // softmax output

	REQUIRE_THROWS(load_model(dir + "/skepu_dnn_does_not_exist.txt", 4));
	REQUIRE_THROWS(load_model(write("skepu_dnn_bad1.txt", "not a model\n"), 4));
	REQUIRE_THROWS(load_model(write("skepu_dnn_bad2.txt", "skepu-dnn-model 1\ninput 1 1 4\nlayer Banana\n"), 4));
	REQUIRE_THROWS(load_model(write("skepu_dnn_bad3.txt", "skepu-dnn-model 1\ninput 1 1 4\nlayer Dense\n"), 4)); // units missing
	REQUIRE_THROWS(load_model(write("skepu_dnn_bad4.txt", "skepu-dnn-model 1\ninput 1 1 4\nlayer Dense units=2 activation=Swish\n"), 4));
	REQUIRE_THROWS(load_model(write("skepu_dnn_bad5.txt", "skepu-dnn-model 1\ninput 1 1 4\nlayer Dense units=2\nparams\nblock 0\n3\n1 2 3\n"), 4)); // wrong count

	for (auto name : {"skepu_dnn_desc.txt", "skepu_dnn_bad1.txt", "skepu_dnn_bad2.txt", "skepu_dnn_bad3.txt", "skepu_dnn_bad4.txt", "skepu_dnn_bad5.txt"})
		std::filesystem::remove(dir + "/" + name);
}

TEST_CASE("DNN rejects unsupported strides and padding")
{
	use_sequential_cpu();

	// Supported values are accepted
	REQUIRE_NOTHROW(Conv2D(2).kernel(3, 3).stride(1, 1).padding(false));
	REQUIRE_NOTHROW(MaxPooling2D().kernel(2, 2).stride(2, 2).padding(false));

	// Everything else is an error instead of being ignored
	REQUIRE_THROWS(Conv2D(2).kernel(3, 3).stride(2, 2));
	REQUIRE_THROWS(Conv2D(2).kernel(3, 3).padding(true));
	REQUIRE_THROWS(MaxPooling2D().kernel(2, 2).padding(true));

	// The pooling stride must equal the pool size; this is checked when the model is set up
	auto model = SequentialModel(Dimensions{4, 8, 8, 1}) << MaxPooling2D().kernel(2, 2).stride(1, 1);
	REQUIRE_THROWS(model.init());
}

TEST_CASE("DNN gradients: leaky ReLU and ELU activations")
{
	use_sequential_cpu();
	const size_t B = 4;
	for (Act act : {Act::LeakyReLU, Act::ELU})
	{
		auto model = SequentialModel(Dimensions{B, 1, 1, 8})
			<< Dense(6).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(act)
			<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::SoftMax);
		skepu::Tensor4<float> x(B, 1, 1, 8), y(B, 1, 1, 3);
		fill_random(x, -1, 1); fill_labels(y);
		// Biases shifted by +0.5 to keep the units away from the kink at 0
		REQUIRE(gradcheck(std::string("MLP ") + to_string(act) + " + softmax / CCE", model, Loss::CategoricalCrossEntropy, x, y, 0.5f));
	}
}

TEST_CASE("DNN gradients: average pooling")
{
	use_sequential_cpu();
	const size_t B = 4;
	{
		auto model = SequentialModel(Dimensions{B, 9, 9, 2})   // 9 is not divisible by 2: the last row/column is not pooled
			<< Conv2D(3).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
			<< AvgPooling2D().kernel(2, 2)
			<< Flatten()
			<< Dense(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::SoftMax);
		skepu::Tensor4<float> x(B, 9, 9, 2), y(B, 1, 1, 3);
		fill_random(x, 0, 1); fill_labels(y);
		REQUIRE(gradcheck("CNN conv + avgpool2d + softmax / CCE", model, Loss::CategoricalCrossEntropy, x, y));
	}
	{
		auto model = SequentialModel(Dimensions{B, 1, 14, 2})
			<< Conv1D(3).kernel(3).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::TanH)
			<< AvgPooling1D().kernel(3)
			<< Flatten()
			<< Dense(1).weights(Init::Xavier_uniform).biases(Init::Uniform).activation(Act::Sigmoid);
		skepu::Tensor4<float> x(B, 1, 14, 2), y(B, 1, 1, 1);
		fill_random(x, -1, 1); fill_labels(y);
		REQUIRE(gradcheck("CNN1D conv1d + avgpool1d + sigmoid / BCE", model, Loss::BinaryCrossEntropy, x, y));
	}
}

TEST_CASE("DNN model files: the new activations and average pooling")
{
	use_sequential_cpu();
	const std::string path = (std::filesystem::temp_directory_path() / "skepu_dnn_test_new_layers.txt").string();
	std::ofstream(path) <<
		"skepu-dnn-model 1\n"
		"input 1 16 1\n"
		"layer Conv1D features=4 kernel=3 activation=LeakyReLU\n"
		"layer AvgPooling1D kernel=2\n"
		"layer Conv1D features=4 kernel=3 activation=ELU\n"
		"layer Flatten\n"
		"layer Dense units=3 activation=SoftMax\n";
	auto model = load_model(path, 4);
	skepu::Tensor4<float> x(4, 1, 16, 1);
	fill_random(x, -1, 1);
	skepu::Tensor4<float> out = model.predict(x);
	REQUIRE(out.size_l() == 3);

	// Saved and loaded again, the same model
	const std::string path2 = path + ".2";
	model.save(path2);
	auto loaded = load_model(path2, 4);
	skepu::Tensor4<float> out2 = loaded.predict(x);
	for (size_t l = 0; l < 3; ++l)
		REQUIRE(out2(0, 0, 0, l) == out(0, 0, 0, l));
	std::filesystem::remove(path);
	std::filesystem::remove(path2);
}
