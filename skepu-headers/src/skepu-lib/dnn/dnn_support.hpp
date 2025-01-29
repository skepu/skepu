#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED

static skepu::PrecompilerMarker startOf_DNN_Support_HPP;

template <typename T> std::string string_tensor_size(Tensor4<T> &t)
{
	std::ostringstream os;
	os << "(" << t.size_i() << " x " << t.size_j() << " x " << t.size_k() << " x " << t.size_l() << ")";
	return os.str();
}

namespace ml {
using Dimensions = std::tuple<size_t, size_t, size_t, size_t>;

enum class Level { Input, Hidden, Output };

enum class Init { Uniform, Zero, Xavier };

enum class Act { None, ReLU, Sigmoid, TanH, SoftMax };

using ModelType = float;

struct DebugSink {
	template <typename T> DebugSink &operator<<(T const &) { return *this; }
};

static DebugSink debugSink;

#ifdef DNN_DEBUG
static auto &dnn_debug = std::cout;
#else
static auto &dnn_debug = debugSink;
#endif




float flatten_uf(float x) { return x; }

float cross_entropy_uf(float y, float s) { return y * log(s); }

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
	float max_y = 0;
	float max_s = 0;
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

float mse_prime_uf(float a, float y) { return (y - a) * (y - a) * 0.5; }

float mse_uf(float a, float y) { return y - a; }

float delta_uf(float pred, float gold) { return pred - gold; }

// Works regardless of weights shape, i.e., on dense weight matrices,
// convolutional weight tensors, bias vectors
float update_params_uf(float old_weight, float gradient, float learning_rate)
{
	return old_weight - learning_rate * gradient;
}

template <typename T> T sum_uf(T a, T b) { return a + b; }

float product_uf(float a, float b) { return a * b; }

float flatmap_uf(float a) { return a; }

size_t argmax_uf(skepu::Index1D index, skepu::MatRow<float> DNN_CONST y)
{
	float max_v = -999999.f;
	size_t max_i = 0;
	for (size_t i = 0; i < y.cols; ++i)
	{
		if (y(i) > max_v) {
			max_v = y(i);
			max_i = i;
		}
	}
	return max_i;
}



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






auto skel_flatmap = skepu::Map(flatmap_uf);

// Keep these!
//auto skel_cross_entropy_batched_loss = skepu::MapReduce<0>(cross_entropy_batched_uf, sum_uf<float>);
//auto skel_accuracy_batched = skepu::MapReduce<0>(accuracy_batched_uf, sum_uf<int>);

// Map+Reduce variants
auto skel_cross_entropy_batched_loss_map = skepu::Map<0>(cross_entropy_batched_uf);
auto skel_accuracy_batched_map = skepu::Map<0>(accuracy_batched_uf);
auto skel_cross_entropy_batched_loss_reduce = skepu::Reduce(sum_uf<float>);
auto skel_accuracy_batched_reduce = skepu::Reduce(sum_uf<int>);

// Initialization
auto skel_init_zero = skepu::Map<0>(init_zero_uf);
auto skel_init_random_uniform = skepu::Map<0>(init_random_uniform_uf<float>);

// Backprop
auto skel_delta = skepu::Map(delta_uf);
auto skel_cost_prime = skepu::Map(mse_prime_uf);

auto skel_update_params = skepu::Map<2>(update_params_uf);



static skepu::PrecompilerMarker endOf_DNN_Support_HPP;