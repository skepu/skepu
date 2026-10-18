#pragma once
#include <cmath>
#include <string>
#include <vector>

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

// ---------------- labels: classes to one-hot vectors, and predictions to classes ----------------
//
// The labels of a classification are the n x 1 x 1 x C tensors of the one-hot vectors that the models are trained on and the predictions
// have the shape of; the datasets and the programs have the classes as numbers.
//
//   skepu::Tensor4<float> y = skepu::dnn::one_hot(classes, 10);   // classes: a vector (or tensor) of n class numbers, from 0 to 9
//   skepu::dnn::one_hot(y, classes);                              // the same, into a tensor that is made (the first y.size_i() classes)
//   skepu::Vector<int> predicted = skepu::dnn::argmax(scores);    // the class of each sample: the first largest element of its scores
//
// Both are skeleton calls: they work on all backends and on distributed containers, and the host does not touch the elements. The class
// numbers can be integers or floating-point numbers; one that is not an integer from 0 to C - 1 is an error (the check is a MapReduce, over
// all the classes that are given, also if more than n of them are).
namespace detail {

	template<typename T>
	float one_hot_element(skepu::Index4D index, skepu::Vec<T> classes)
	{
		return ((int)classes[index.i] == (int)index.l) ? 1.f : 0.f;
	}

	template<typename T>
	int class_out_of_range(T cls, int num_classes)
	{
		return ((int)cls < 0 || (int)cls >= num_classes || (T)(int)cls != cls) ? 1 : 0;
	}

	inline int sum_of_ints(int a, int b) { return a + b; }

	// The first of the largest scores of the sample
	template<typename R>
	R argmax_of_scores(skepu::Index1D index, skepu::Ten4<float> scores)
	{
		R best = 0;
		float largest = scores(index.i, 0, 0, 0);
		for (size_t c = 1; c < scores.size_l; ++c)
			if (scores(index.i, 0, 0, c) > largest)
			{
				largest = scores(index.i, 0, 0, c);
				best = (R)c;
			}
		return best;
	}
}

// Fills out (n x 1 x 1 x C) with the one-hot vectors of the first n of the classes
template<typename T>
void one_hot(skepu::Tensor4<float> &out, skepu::Vector<T> &classes SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("dnn.one_hot");
	if (out.size_j() != 1 || out.size_k() != 1)
		SKEPU_ERROR("skepu::dnn::one_hot: the labels must be n x 1 x 1 x classes, not " << out.size_i() << " x " << out.size_j() << " x " << out.size_k() << " x " << out.size_l());
	if (classes.size() < out.size_i())
		SKEPU_ERROR("skepu::dnn::one_hot: " << classes.size() << " classes for " << out.size_i() << " labels");

	auto check = skepu::MapReduce<1>(detail::class_out_of_range<T>, detail::sum_of_ints);
	check.setLabel("dnn::one_hot check");
	const int invalid = static_cast<int>(check(classes, (int)out.size_l()));
	if (invalid != 0)
		SKEPU_ERROR("skepu::dnn::one_hot: " << invalid << " of the classes are not integers from 0 to " << out.size_l() - 1);

	auto skel = skepu::Map<0>(detail::one_hot_element<T>);
	skel.setLabel("dnn::one_hot");
	skel(out, classes);
}

// The one-hot labels (n x 1 x 1 x num_classes) of the n classes
template<typename T>
skepu::Tensor4<float> one_hot(skepu::Vector<T> &classes, size_t num_classes, std::string label = "one-hot labels" SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("dnn.one_hot");
	skepu::Tensor4<float> out(classes.size(), 1, 1, num_classes, label, 0.f);
	one_hot(out, classes);
	return out;
}

// The class of each of the n samples of scores (n x 1 x 1 x C): the index of its first largest score (the integer type R is int by default)
template<typename R = int>
skepu::Vector<R> argmax(skepu::Tensor4<float> &scores, std::string label = "predicted classes" SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("dnn.argmax");
	if (scores.size_j() != 1 || scores.size_k() != 1)
		SKEPU_ERROR("skepu::dnn::argmax: the scores must be n x 1 x 1 x classes, not " << scores.size_i() << " x " << scores.size_j() << " x " << scores.size_k() << " x " << scores.size_l());
	skepu::Vector<R> classes(scores.size_i(), label);
	auto skel = skepu::Map<0>(detail::argmax_of_scores<R>);
	skel.setLabel("dnn::argmax");
	skel(classes, scores);
	return classes;
}

// ---- Preparing data sets. All of them are skeleton calls on the containers (no host loops), so they run on every backend and
// the data stay where the skeletons put them.

namespace detail
{
	// The sample i of the source (the same extents, fewer samples), and the same with (n, h, w, 1) images as (n, 1, h, w) sequences:
	// the rows are the time steps, the columns the channels
	inline float take_element(skepu::Index4D index, skepu::Ten4<float> src)
	{
		return src(index.i, index.j, index.k, index.l);
	}

	inline float sequence_element(skepu::Index4D index, skepu::Ten4<float> src)
	{
		return src(index.i, index.k, index.l, 0);
	}

	// (n, c, h, w) to (n, h, w, c): the element index is that of the result
	inline float chw_to_hwc_element(skepu::Index4D index, skepu::Ten4<float> src)
	{
		return src(index.i, index.l, index.j, index.k);
	}

	// Sums of one channel (double: the sums of a data set are large)
	inline double channel_sum(skepu::Index4D index, float v, int channel)
	{
		return (int)index.l == channel ? (double)v : 0.0;
	}

	inline double channel_square_sum(skepu::Index4D index, float v, int channel)
	{
		return (int)index.l == channel ? (double)v * (double)v : 0.0;
	}

	inline double add_double(double a, double b) { return a + b; }

	inline float standardize_element(skepu::Index4D index, float v, skepu::Vec<float> mean, skepu::Vec<float> stddev)
	{
		return (v - mean[index.l]) / stddev[index.l];
	}
}

static auto skel_take = skepu::Map<0>(detail::take_element);
static auto skel_sequence = skepu::Map<0>(detail::sequence_element);
static auto skel_chw_to_hwc = skepu::Map<0>(detail::chw_to_hwc_element);
static auto skel_channel_sum = skepu::MapReduce<1>(detail::channel_sum, detail::add_double);
static auto skel_channel_square_sum = skepu::MapReduce<1>(detail::channel_square_sum, detail::add_double);
static auto skel_standardize = skepu::Map<1>(detail::standardize_element);

// The first n samples of a data set (a copy)
inline skepu::Tensor4<float> take(skepu::Tensor4<float> &src, size_t n, std::string label = "subset" SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("dnn.take");
	if (n > src.size_i())
		SKEPU_ERROR("skepu::dnn::take: " << n << " samples requested of " << src.size_i());
	skepu::Tensor4<float> dst(n, src.size_j(), src.size_k(), src.size_l(), label);
	skel_take.setLabel("dnn::take");
	skel_take(dst, src);
	return dst;
}

// The first n samples (all by default) of a data set of (h, w, 1) images as sequences (1, h, w): the rows are the time steps and the
// columns the channels, as the 1D layers (Conv1D, MaxPooling1D) take them
inline skepu::Tensor4<float> as_sequence(skepu::Tensor4<float> &src, size_t n = 0, std::string label = "sequences" SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("dnn.as_sequence");
	if (src.size_l() != 1)
		SKEPU_ERROR("skepu::dnn::as_sequence: the images must have one channel, not " << src.size_l());
	if (n == 0) n = src.size_i();
	if (n > src.size_i())
		SKEPU_ERROR("skepu::dnn::as_sequence: " << n << " samples requested of " << src.size_i());
	skepu::Tensor4<float> dst(n, 1, src.size_j(), src.size_k(), label);
	skel_sequence.setLabel("dnn::as_sequence");
	skel_sequence(dst, src);
	return dst;
}

// The layout of a data set (n, channels, height, width), which is that of the raw CIFAR-10 files and of many other sources, as
// (n, height, width, channels), the layout of the layers
inline skepu::Tensor4<float> chw_to_hwc(skepu::Tensor4<float> &src, std::string label = "images (HWC)" SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("dnn.chw_to_hwc");
	skepu::Tensor4<float> dst(src.size_i(), src.size_k(), src.size_l(), src.size_j(), label);
	skel_chw_to_hwc.setLabel("dnn::chw_to_hwc");
	skel_chw_to_hwc(dst, src);
	return dst;
}

// The mean and standard deviation of each channel (the last extent) of a data set
struct ChannelStats
{
	skepu::Vector<float> mean, stddev;
};

// The statistics are made by skeletons (MapReduce), so every process of an MPI program has them: the code of a skepu::external
// call runs on one process only, and a value computed in it would not reach the others
inline ChannelStats channel_stats(skepu::Tensor4<float> &images, std::string label = "channel" SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("dnn.channel_stats");
	const size_t channels = images.size_l();
	ChannelStats stats{skepu::Vector<float>(channels, label + " means"), skepu::Vector<float>(channels, label + " deviations")};
	const double count = (double)images.size_i() * images.size_j() * images.size_k();
	skel_channel_sum.setLabel("dnn::channel_sum");
	skel_channel_square_sum.setLabel("dnn::channel_square_sum");
	std::vector<float> mean(channels), stddev(channels);
	for (size_t c = 0; c < channels; ++c)
	{
		const double sum = static_cast<double>(skel_channel_sum(images, (int)c));
		const double square_sum = static_cast<double>(skel_channel_square_sum(images, (int)c));
		mean[c] = sum / count;
		stddev[c] = std::sqrt(square_sum / count - (double)mean[c] * mean[c]);
	}
	skepu::external([&]
	{
		for (size_t c = 0; c < channels; ++c)
		{
			stats.mean(c) = mean[c];
			stats.stddev(c) = stddev[c];
		}
	}, skepu::write(stats.mean, stats.stddev));
	return stats;
}

// Each channel to zero mean and unit variance with the given statistics (of the training set, for both the training and the test set)
inline void standardize(skepu::Tensor4<float> &images, ChannelStats &stats SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("dnn.standardize");
	if (stats.mean.size() != images.size_l())
		SKEPU_ERROR("skepu::dnn::standardize: " << stats.mean.size() << " channel statistics for " << images.size_l() << " channels");
	skel_standardize.setLabel("dnn::standardize");
	skel_standardize(images, images, stats.mean, stats.stddev);
}

namespace detail
{
	// The process that does the file I/O (always, but under the MPI backend, where it is the first process; the other
	// processes do not write files, and std::cout of them is silent already)
	inline bool is_first_process()
	{
#ifdef SKEPU_MPI
		return skepu::mpi::rank() == 0;
#else
		return true;
#endif
	}
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
