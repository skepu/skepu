#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_DNN_Support_HPP;


float cross_entropy_batched_uf(
		skepu::Index1D index,
		skepu::Ten4<float> y,
		skepu::Ten4<float> s)
{
	size_t item = index.i;
	float epsilon = 1e-9;
	float res = 0;
	for (size_t i = 0; i < y.size_l; ++i)
		res += y(item, 0, 0, i) * log(s(item, 0, 0, i) + epsilon);
	return res;
}

int accuracy_batched_uf(
		skepu::Index1D index,
		skepu::Ten4<float> y,
		skepu::Ten4<float> s)
{
	size_t item = index.i;

	// Single output (binary classification): compare thresholded prediction
	if (y.size_l == 1)
	{
		int label = (y(item, 0, 0, 0) > 0.5f) ? 1 : 0;
		int pred  = (s(item, 0, 0, 0) > 0.5f) ? 1 : 0;
		return (label == pred) ? 1 : 0;
	}

	float max_y = -1e30f;
	float max_s = -1e30f;
	size_t argmax_y = 0;
	size_t argmax_s = 0;

	for (size_t i = 0; i < y.size_l; ++i)
	{
		float this_y = y(item, 0, 0, i);
		float this_s = s(item, 0, 0, i);
		if (this_y > max_y)
		{
			max_y = this_y;
			argmax_y = i;
		}
		if (this_s > max_s)
		{
			max_s = this_s;
			argmax_s = i;
		}
	}
	return (argmax_y == argmax_s) ? 1 : 0;
}


// Gradient of categorical cross-entropy w.r.t. the softmax output (dL/ds).
// The softmax layer's backward pass then applies the softmax Jacobian.
float cce_grad_uf(float pred, float gold) { return -gold / (pred + 1e-9f); }

// Works regardless of weights shape, i.e., on dense weight matrices,
// convolutional weight tensors, bias vectors
float update_params_uf(float old_weight, float gradient, float learning_rate)
{
	float val = old_weight - learning_rate * gradient;
#ifdef SKEPU_DNN_WEIGHT_CAP
	// Optional clipping of parameters (disabled by default)
	const float cap = SKEPU_DNN_WEIGHT_CAP;
	if (val > cap) val = cap;
	if (val < -cap) val = -cap;
#endif
	return val;
}

// Adam update of one parameter. m and v are the running first and second moment estimates of the
// gradient; corr1 = 1 - beta1^t and corr2 = 1 - beta2^t correct their initialization bias.
// Returns the new (parameter, m, v).
skepu::multiple<float, float, float>
adam_update_uf(float w, float g, float m, float v,
	float lr, float beta1, float beta2, float epsilon, float corr1, float corr2)
{
	float m_new = beta1 * m + (1.f - beta1) * g;
	float v_new = beta2 * v + (1.f - beta2) * g * g;
	float m_hat = m_new / corr1;
	float v_hat = v_new / corr2;
	float w_new = w - lr * m_hat / (sqrt(v_hat) + epsilon);
	return skepu::ret(w_new, m_new, v_new);
}

template <typename T> T sum_uf(T a, T b) { return a + b; }


// Initialization
float init_zero_uf() { return 0; }

template <typename T>
T init_random_uniform_uf(skepu::Random<1> &rand, T min, T max)
{
	T val = rand.getNormalized() * (max - min) + min;
	return val;
}


float propagate_loss_gradient_uf(
skepu::Index4D index,
skepu::Ten4<float> gradient,
skepu::Mat<float> weights)
{
	float sum = 0;
	for (size_t i = 0; i < gradient.size_l; ++i)
	{
		sum += gradient(index.i, 0, 0, i) * weights(index.l, i);
	}
	return sum;
}


/// binary cross entropy
/****************************** Binary Loss function *********************************************/

float binary_cross_entropy(skepu::Index1D index, skepu::Ten4<float> y, skepu::Ten4<float> y_pred) //s
{
	size_t item = index.i;

	float epsilon = 1e-9;
	float res = 0;
	float val = y_pred(item,0,0,0);

	if(val > 1.0 - epsilon){
		val = 1.0 - epsilon;
	}
	if(val < epsilon){
		val = epsilon;
	}
	res += y(item,0,0,0) * log(val) + (1 - y(item,0,0,0)) * log(1 - val);


	return res;
}

// backward
float BCE_grad_uf(skepu::Index4D index, skepu::Ten4<float> y_pred, skepu::Ten4<float> y ){
	size_t item = index.i;
	float epsilon = 1e-9;
	float val = y_pred(item,0,0,0);

	if(val > 1.0 - epsilon){
		val = 1.0 - epsilon; }
	if(val < epsilon){
		val = epsilon; }

	float numerator = val - y(item,0,0,0) ;
	float denominator = val * (1 - val);
	return numerator / denominator ;
}


// Forward

float flatmap_uf(float a) { return a; }


auto skel_flatmap = skepu::Map(flatmap_uf);

int copy_int_uf(int a) { return a; }
auto skel_copy_int = skepu::Map(copy_int_uf);

// The loss and the accuracy of a batch are a Map (per-sample values) followed by a Reduce. They could also be one
// MapReduce<0> each, e.g. skepu::MapReduce<0>(cross_entropy_batched_uf, sum_uf<float>), which avoids the temporary
// vector; the per-sample values are kept because evaluate() needs them to leave out the overlap of a remainder batch.
auto skel_cross_entropy_batched_loss_map = skepu::Map<0>(cross_entropy_batched_uf);
auto skel_accuracy_batched_map = skepu::Map<0>(accuracy_batched_uf);
auto skel_cross_entropy_batched_loss_reduce = skepu::Reduce(sum_uf<float>);
auto skel_accuracy_batched_reduce = skepu::Reduce(sum_uf<int>);

// Initialization
auto skel_init_zero = skepu::Map<0>(init_zero_uf);
auto skel_init_random_uniform = skepu::Map<0>(init_random_uniform_uf<float>);

// Backprop
auto skel_delta_cce = skepu::Map(cce_grad_uf);

auto skel_update_params = skepu::Map<2>(update_params_uf);

// Outputs: parameter, m, v. Elementwise inputs: parameter, gradient, m, v
auto skel_adam_update = skepu::Map<4>(adam_update_uf);


auto skel_propagate_loss_gradient         = skepu::Map(propagate_loss_gradient_uf);


// bce forward and back ward skel
// forward
auto skel_binary_cross_entropy_loss_map = skepu::Map<0>(binary_cross_entropy);
// back
auto skel_delta_bce = skepu::Map(BCE_grad_uf);

namespace skepu::dnn {
using Dimensions = std::tuple<size_t, size_t, size_t, size_t>;

enum class Level { Input, Hidden, Output };

enum class Init { Uniform, Zero, Xavier_uniform, Xavier_normal, He_uniform, He_normal };

enum class Act { None, ReLU, Sigmoid, TanH, SoftMax, LeakyReLU, ELU };

enum class Loss { CategoricalCrossEntropy, BinaryCrossEntropy };

enum class Opt { SGD, Adam};

using ModelType = float;

// Names used in model files
inline const char *to_string(Init i)
{
	switch (i)
	{
	case Init::Uniform: return "Uniform";
	case Init::Zero: return "Zero";
	case Init::Xavier_uniform: return "Xavier_uniform";
	case Init::Xavier_normal: return "Xavier_normal";
	case Init::He_uniform: return "He_uniform";
	case Init::He_normal: return "He_normal";
	}
	return "Uniform";
}

inline const char *to_string(Act a)
{
	switch (a)
	{
	case Act::None: return "None";
	case Act::ReLU: return "ReLU";
	case Act::Sigmoid: return "Sigmoid";
	case Act::TanH: return "TanH";
	case Act::LeakyReLU: return "LeakyReLU";
	case Act::ELU: return "ELU";
	case Act::SoftMax: return "SoftMax";
	}
	return "None";
}

inline bool init_from_string(const std::string &s, Init &out)
{
	for (Init i : {Init::Uniform, Init::Zero, Init::Xavier_uniform, Init::Xavier_normal, Init::He_uniform, Init::He_normal})
		if (s == to_string(i)) { out = i; return true; }
	return false;
}

inline bool act_from_string(const std::string &s, Act &out)
{
	for (Act a : {Act::None, Act::ReLU, Act::Sigmoid, Act::TanH, Act::SoftMax, Act::LeakyReLU, Act::ELU})
		if (s == to_string(a)) { out = a; return true; }
	return false;
}

struct DebugSink {
	template <typename T> DebugSink &operator<<(T const &) { return *this; }
};

static DebugSink debugSink;

#ifdef DNN_DEBUG
static auto &dnn_debug = std::cout;
#else
static auto &dnn_debug = debugSink;
#endif
}

static skepu::PrecompilerMarker endOf_DNN_Support_HPP;
#endif // SKEPU_PRECOMPILED
