#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

//#define SKEPU_ENABLE_EXCEPTIONS
#define SKEPU_DNN_NO_PRETRAINED_PARAMS "NO PRETRAINED PARAMS"
#define SKEPU_DNN_USE_MASKED_POOL 1
#define DNN_CONST /*const&*/
// #define DNN_DEBUG 1

#include <skepu>
#include <skepu-lib/io.hpp>

#include "csv_io.hpp"







float fully_connected_batched_uf(
	skepu::Index4D index,
	skepu::Vec<float> DNN_CONST bias,
	skepu::Mat<float> DNN_CONST weights,
	skepu::Ten4<float> DNN_CONST in
)
{
	size_t batch_index = index.i;
	size_t class_index = index.l;
	float res = bias(class_index);
	for (size_t i = 0; i < weights.rows; ++i)
		res += weights(i, class_index) * in(0, 0, batch_index, i);
	return res;
}

float sigmoid_uf(float x)
{
	return 1 / (1 + exp(-x)); // or tanh
}

float sigmoid_prime_uf(float x)
{
	float sigma_x = sigmoid_uf(x);
	return sigma_x * (1 - sigma_x);
}

// forward oass
float relu_uf(float x) { return (x > 0) ? x : 0; }

float leaky_relu_uf(float x) { return (x > 0) ? x :(0.01 * x); }

float P_relu_uf(float x, float alpha ){ return (x > 0) ? x :(alpha * x);}

float elu_uf(float x, float alpha){
	return (x > 0) ? x : alpha * (std::exp(x) - 1);
}

// bacCKWRaRD PASS
float relu_prime_uf(float x, float g)
{
	return (x > 0) ? g : 0 ;
}

float leaky_relu_prime_uf(float x)
{
	return (x > 0) ? 1.0 : 0.01 ;
}

float P_relu_prime_uf(float x, float alpha)
{
	return (x > 0) ? 1.0 : alpha ;
}

float elu_prime_uf(float x, float alpha)
{
	return (x > 0) ? 1.0 : alpha * std::exp(x);
}

// SoftMax, non-batched
float softmax_uf_1_map(float x) { return exp(x); }

float softmax_uf_1_reduce(float lhs, float rhs) { return lhs + rhs; }

float softmax_uf_2(float x, float sum) { return exp(x) / sum; }

// SoftMax, batched
float softmax_batched_uf_1(skepu::Pool4D<float> DNN_CONST pool)
{
	float sum = 0;
	for (size_t l = 0; l < pool.sl; l++)
		sum += exp(pool(0, 0, 0, l));
	return sum;
}

float softmax_batched_uf_2(skepu::Index4D index, float x, skepu::Ten4<float> sums)
{
	return exp(x) / sums(index.i, index.j, index.k, 0);
}


/** new softmax without MapPool */
float new_softmax_uf_1(skepu::Index4D index, skepu::Ten4<float> input)
{
  size_t item = index.i;
  float max_value = input(item,0,0,0);

  for(size_t i = 0; i < input.size_l; i++){
    if( input(item,0,0,i) > max_value ){
      max_value = input(item,0,0,i);
    }
  }

  return max_value;

}

float new_softmax_uf_2(skepu::Index4D index, skepu::Ten4<float> input, skepu::Ten4<float>  max_vals)
{
	float sum = 0;
	for (size_t l = 0; l < input.size_l; l++){
      sum += exp(input(index.i, 0, 0, l) - max_vals(index.i,0,0,0));
  }
	return sum;
}

float new_softmax_uf_3(skepu::Index4D index, float x, skepu::Ten4<float> sums, skepu::Ten4<float>  max_vals)
{
	return exp(x - max_vals(index.i,0,0,0)) / sums(index.i, index.j, index.k, 0);
}

/****************************** Softmax derivative  *********************************************/

float softmax_prime_uf(skepu::Index4D index , skepu::Ten4<float> gradient, skepu::Ten3<float> jacobian, int num_classes){
    float res = 0.0;
    for(size_t j = 0; j < num_classes; ++j){
      res += jacobian(index.i , index.l, j ) * gradient(index.i, 0,0,j);
    }
  return res;
}


float softmax_jacobian_uf(skepu::Index3D index , skepu::Ten4<float> s ){

  if ( index.j == index.k ){
    return s(index.i ,0 ,0 , index.j) * ((float) 1.0 - s(index.i ,0 ,0 , index.j) );
  }
  else {
    return -s(index.i ,0 ,0 , index.j) * s(index.i ,0 ,0 , index.k);
  }

}


float tanh_uf(float x) { return tanh(x); }

float tanh_prime_uf(float x)
{
	float t = tanh(x);
	return 1 - t * t;
}

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



float max_pooling_uf(skepu::Pool4D<float> DNN_CONST pool)
{
	float maxval = pool(0,0,0,0);
	for (size_t j = 0; j < pool.sj; ++j)
		for (size_t k = 0; k < pool.sk; ++k)
		{
			float val = pool(0, j, k, 0);
			maxval = (maxval > val) ? maxval : val;
		}
	return maxval;
}

skepu::multiple<float, char>
max_pooling_masked_uf(skepu::Pool4D<float> DNN_CONST pool)
{
	float maxval = pool(0,0,0,0);
	char mask = 0;
	for (size_t j = 0; j < pool.sj; ++j){
		for (size_t k = 0; k < pool.sk; ++k)
		{
			 float val = pool(0, j, k, 0);
			if (val > maxval)
			{
				maxval = val;
				mask = j * pool.sk + k;
			}
		}
  }
	return skepu::ret(maxval, mask);
}

float avg_pooling_uf(skepu::Pool4D<float> DNN_CONST pool)
{
	float sum = 0;
	for (size_t j = 0; j < pool.sj; ++j)
		for (size_t k = 0; k < pool.sk; ++k)
			sum += pool(0, j, k, 0);
	return sum / (pool.sj * pool.sk);
}



float convolutional_1d_uf(
	skepu::Index4D index,
	skepu::Region4D<float> DNN_CONST in,
	skepu::Mat<float> DNN_CONST weights,
	skepu::Vec<float> DNN_CONST bias
)
{
	float res = bias(index.l);
	for (size_t i = -in.oi; i <= in.oi; ++i)
		for (size_t j = 0; j < weights.cols; ++j)
			res += weights(i + in.oi, j) * in(0, 0, i, j - index.l);
	return res;
}

float convolutional_1d_connections_uf(
		skepu::Index2D index, skepu::Region2D<float> DNN_CONST in,
		skepu::Mat<float> DNN_CONST weights, skepu::Vec<float> DNN_CONST bias,
		skepu::Vec<int> DNN_CONST connection_map) // bias
{
	float res = bias(index.col);
	for (size_t i = -in.oi; i <= in.oi; ++i)
		if (connection_map(index.row) == 1)
		{
			for (size_t j = 0; j < weights.cols; ++j)
				res += weights(i + in.oi, j) * in(i, j - index.col);
		}
	return res;
}

float convolutional_2d_uf(
	skepu::Index4D index,
	skepu::Region4D<float> DNN_CONST in,
	skepu::Ten4<float> DNN_CONST weights,
	skepu::Vec<float> DNN_CONST bias
)
{
	float res = 0.0; //bias(index.l);
	for (int i = -in.oj; i <= in.oj; ++i)
		for (int j = -in.ok; j <= in.ok; ++j)
		{
			for (size_t k = 0; k < weights.size_k; ++k) // loop over input features
			{

				res += weights(i + in.oj, j + in.ok, k, index.l) * in(0, i, j, k - index.l);
			}
		}
	return res;
}

float convolutional_2d_connections_uf(
	skepu::Index4D index,
	skepu::Region4D<float> DNN_CONST in,
	skepu::Ten4<float> DNN_CONST weights,
	skepu::Vec<float> DNN_CONST bias,
	skepu::Mat<int> DNN_CONST connection_map
)
{
	float res = 0;
	for (size_t i = -in.oi; i <= in.oi; ++i)
		for (size_t j = -in.oj; j <= in.oj; ++j)
			if (connection_map(index.i, index.j) == 1)
			{
				for (size_t k = 0; k < weights.size_k; ++k)
					res += weights(0, i + in.oi, j + in.oi, k) * in(0, i, j, k - index.k);
			}
	return res; // bias
}

float mse_prime_uf(float a, float y) { return (y - a) * (y - a) * 0.5; }

float mse_uf(float a, float y) { return y - a; }

float delta_uf(float pred, float gold) { return pred - gold; }

// Works regardless of weights shape, i.e., on dense weight matrices,
// convolutional weight tensors, bias vectors
float update_params_uf(float old_weight, float gradient, float learning_rate)
{
	// return old_weight - learning_rate * gradient;
	const float cap = 0.1; // hyperparameter, hardcoded, affects learning performance and numerical stability
   float val = old_weight - learning_rate * gradient;
   if (val > cap) val = cap;
   if (val < -cap) val = -cap;
   return val;
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

size_t eval_uf(size_t a, size_t b) { return (a == b) ? 1 : 0; }

template <typename T> T dropout_uf(skepu::Random<1> &rand, T el, float rate)
{
	float p = rand.getNormalized();  
	return (p > rate) ? (el / (1.0 - rate)) : 0;
}

template <typename T> skepu::multiple<T, char> dropout_masked_uf(skepu::Random<1> &rand, T el, float rate)
{
	float p = rand.getNormalized();
	// char enabled = (p > rate);
	// T res = enabled ? (el / (1.0 - rate)) : 0;
	// return skepu::ret(res, enabled);
	if (p < rate){
    char mask = 0;
    return skepu::ret(0, mask);
  }
  else{
    char mask = 1;
    return skepu::ret( (el/(1 - rate)), mask);
  }

}

template <typename T> T dropout_masked_backprop_uf(T gradient, char enabled, float rate)
{
	return enabled ? gradient/* / (1.0 - rate)*/ : 0; // todo check
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

float propagate_dense_weight_gradient_uf(
	skepu::Index2D index,
	skepu::Ten4<float> inputs,
	skepu::Ten4<float> gradient)
{
	size_t num_batches = inputs.size_i;
	float sum = 0;
	for (size_t b = 0; b < num_batches; ++b)
		sum += inputs(b, 0, 0, index.row) * gradient(b, 0, 0, index.col); // indexing?
	return sum / num_batches;
}

float propagate_dense_bias_gradient_uf(
	skepu::Index1D index,
	skepu::Ten4<float> gradient)
{
	size_t num_batches = gradient.size_i;
	float sum = 0;
	for (size_t b = 0; b < num_batches; ++b)
		sum += gradient(b, 0, 0, index.i);
	return sum / num_batches;
}


float pool_max_backward_uf(
	skepu::Index4D index,
	skepu::Ten4<float> inputs,
	skepu::Ten4<float> gradient,
	size_t pool_size_j,
	size_t pool_size_k
)
{
	float maxval = 0;
	size_t max_idx_j = 0;
	size_t max_idx_k = 0;
	size_t base_j = (size_t)(index.j / pool_size_j) * pool_size_j;
	size_t base_k = (size_t)(index.k / pool_size_k) * pool_size_k;

	// Find the index of the max value in the pool of forward-inputs
	for (size_t j = 0; j < pool_size_j; ++j)
		for (size_t k = 0; k < pool_size_k; ++k)
		{
			float val = inputs(index.i, base_j + j, base_k + k, index.l);
			if (val > maxval)
			{
				maxval = val;
				max_idx_j = base_j + j;
				max_idx_k = base_k + k;
			}
		}

	// If this index is the index of the max value, the gradient is 1
	// and implicitly multiplied with the incoming gradient
	if (index.j == max_idx_j && index.k == max_idx_k)
	{
		return gradient(
			index.i,
			(size_t)(index.j / pool_size_j),
			(size_t)(index.k / pool_size_k),
			index.l
		);
	}

	// Otherwise, the element did not contribute to the forward-output and its gradient is 0
	return 0;
}

float pool_max_masked_backward_uf(
	skepu::Index4D index,
	skepu::Ten4<float> gradient,
	skepu::Ten4<char> mask,
	size_t pool_size_j,
	size_t pool_size_k
)
{
	// covering boundries
  if(index.j >= mask.size_j * pool_size_j  || index.k >= mask.size_k * pool_size_k ){
    return 0;
  }

	size_t small_j = (size_t)(index.j / pool_size_j);
	size_t small_k = (size_t)(index.k / pool_size_k);

	char maskval = mask(index.i, small_j, small_k, index.l);

	size_t max_idx_j = small_j * pool_size_j + (char)(maskval / 2); //pool_size_j
	size_t max_idx_k = small_k * pool_size_k + (maskval % 2); // j

	// If this index is the index of the max value, the gradient is 1
	// and implicitly multiplied with the incoming gradient
	if (index.j == max_idx_j && index.k == max_idx_k)
		return gradient(index.i, small_j, small_k, index.l);

	// Otherwise, the element did not contribute to the forward-output and its gradient is 0
	else return 0;
}


/// binary cross entropy
/****************************** Binary Loss function *********************************************/

float binary_cross_entropy(skepu::Index1D index, skepu::Ten4<float> y, skepu::Ten4<float> y_pred) //s
{
    size_t item = index.i;
    // std::cout<< item << "\n";
    // std::cout<< y << "\n";
    
    float epsilon = 1e-9;
    float res = 0;
    float val = y_pred(item,0,0,0);
    
      if(val > 1.0 - epsilon){
          val = 1.0 - epsilon;
      }
      if(val < epsilon){
          val = epsilon;
      }
      // res += y(item) * log(y_pred(item,0,0,0)) - (1 - y(item)) * log(1 - y_pred(item,0,0,0));
      res += y(item,0,0,0) * log(val) - (1 - y(item,0,0,0)) * log(1 - val);
    

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

  // return y_pred(item, 0,0,0) - y(item) ;
  float numerator = val - y(item,0,0,0) ;
  float denominator = val * (1 - val);
  return numerator / denominator ;
}


namespace skepu
{
template <typename T> std::string string_tensor_size(Tensor4<T> &t)
{
	std::ostringstream os;
	os << "(" << t.size_i() << " x " << t.size_j() << " x " << t.size_k() << " x " << t.size_l() << ")";
	return os.str();
}

namespace ml {
using Dimensions = std::tuple<size_t, size_t, size_t, size_t>;

enum class Level { Input, Hidden, Output };

enum class Init { Uniform, Zero, Xavier_uniform, Xavier_normal, He_uniform, He_normal };

enum class Act { None, ReLU, Sigmoid, TanH, SoftMax };

enum class Loss { CategoricalCrossEntropy, BinaryCrossEntropy };

enum class Opt { SGD, Adam};

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

// Forward
auto skel_flatmap = skepu::Map(flatmap_uf);
auto skel_dense = skepu::Map(fully_connected_batched_uf);

auto skel_act_sigmoid = skepu::Map(sigmoid_uf);
auto skel_act_relu = skepu::Map(relu_uf);
//	auto skel_act_softmax_1 = skepu::MapReduce(softmax_uf_1_map, softmax_uf_1_reduce);
auto skel_act_softmax_2 = skepu::Map<1>(softmax_uf_2);

auto skel_softmax_batched_1 = skepu::MapPool(softmax_batched_uf_1);
auto skel_softmax_batched_2 = skepu::Map(softmax_batched_uf_2);


auto new_softmax_skel_1 = skepu::Map<0>(new_softmax_uf_1);
auto new_softmax_skel_2 = skepu::Map<0>(new_softmax_uf_2);
auto new_softmax_skel_3 = skepu::Map<1>(new_softmax_uf_3);



auto skel_act_tanh = skepu::Map(tanh_uf);
auto skel_dropout = skepu::Map<1>(dropout_uf<float>);
auto skel_dropout_masked = skepu::Map<1>(dropout_masked_uf<float>);
auto skel_dropout_backprop_masked = skepu::Map<2>(dropout_masked_backprop_uf<float>);

auto skel_conv_1d = skepu::MapOverlap(convolutional_1d_uf);
auto skel_conv_2d = skepu::MapOverlap(convolutional_2d_uf);

auto skel_pool_max = skepu::MapPool(max_pooling_uf);
auto skel_pool_max_masked = skepu::MapPool(max_pooling_masked_uf);

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

auto skel_act_sigmoid_prime = skepu::Map(sigmoid_prime_uf);
auto skel_act_relu_prime = skepu::Map(relu_prime_uf);
auto skel_act_tanh_prime = skepu::Map(tanh_prime_uf);

auto skel_hadamard = skepu::Map(product_uf);
auto skel_argmax = skepu::Map(argmax_uf);

auto skel_propagate_loss_gradient         = skepu::Map(propagate_loss_gradient_uf);
auto skel_propagate_dense_weight_gradient = skepu::Map(propagate_dense_weight_gradient_uf);
auto skel_propagate_dense_bias_gradient   = skepu::Map(propagate_dense_bias_gradient_uf);

auto skel_pool_max_backward = skepu::Map<0>(pool_max_backward_uf);
auto skel_pool_max_masked_backward = skepu::Map<0>(pool_max_masked_backward_uf);


auto softmax_jacobian = skepu::Map<0>(softmax_jacobian_uf);
auto softmax_prime  = skepu::Map<0>(softmax_prime_uf);

// bce forward and back ward skel
// forward
auto skel_binary_cross_entropy_loss_map = skepu::Map<0>(binary_cross_entropy);
// auto skel_cross_entropy_batched_loss_reduce = skepu::Reduce(sum_uf<float>);
// back 
auto skel_delta_bce = skepu::Map(BCE_grad_uf);

namespace impl
{
template <typename T>
class SequentialModel;

template <typename T = float>
class Layer
{
public:
	Layer(Dimensions arg_output_size) : m_output_size{arg_output_size} {}
	Layer() : m_output_size{0, 0, 0, 0} {}

	Layer(Layer &) = default;
	Layer(Layer &&) = default;

	virtual ~Layer() {}

public:
	virtual Layer &&activation(Act new_act)
	{
		this->m_act = new_act;
		return std::move(*this);
	}

	// Override for layer classes with trainable parameters.
	virtual Layer &&pretrained(std::string fname)
	{
		this->m_pretrained_params = fname;
		return std::move(*this);
	}

	virtual Layer &&weights(skepu::ml::Init arg_init)
	{
		this->m_init_weights = arg_init;
		return std::move(*this);
	}

	virtual Layer &&biases(skepu::ml::Init arg_init)
	{
		this->m_init_biases = arg_init;
		return std::move(*this);
	}

	// Override for layer classes with trainable parameters.
	virtual bool is_trainable() const { return false; }

	virtual Tensor4<T>& get_weights() {
		// Tensor4<T> dummy;
		// return dummy;
		throw std::logic_error("get_weights() must be implemented in a derived class.");
	}
	virtual Tensor4<T>& get_gradients() {
		throw std::logic_error("get_gradients() must be implemented in a derived class.");
	}

	virtual Vector<T>& get_biases() {
		throw std::logic_error("get_biases() must be implemented in a derived class.");
	}
	virtual Vector<T>& get_biases_gradients() {
		throw std::logic_error("get_biases_gradients() must be implemented in a derived class.");
	}	
	
	virtual Matrix<T>& get_d_weights() {
		throw std::logic_error("get_d_weights() must be implemented in a derived class.");
	}
	virtual Matrix<T>& get_d_gradients() {
		throw std::logic_error("get_d_gradients() must be implemented in a derived class.");
	}



	std::string label()
	{
		std::ostringstream os;
		os << "L" << this->m_layer_id;
		return os.str();
	}

	virtual std::string summary()
	{
		std::ostringstream os;
		os << "[";
		if (this->m_is_setup)
			os << "Layer " << this->m_layer_id;
		else
			os << "Not initialized";

		os << "; out-shape{" << std::get<0>(this->m_output_size) << " x "
			 << std::get<1>(this->m_output_size) << " x "
			 << std::get<2>(this->m_output_size) << " x "
			 << std::get<3>(this->m_output_size);
		os << "}";
		os << "; batch-size{";
		os << this->m_batch_size;
		os << "}";
		os << "] ";
		return os.str();
	}

	// Subclasses must set their output size before calling super here!
	virtual void setup(Dimensions input_size)
	{
		this->m_is_setup = true;

		this->m_vals.init(
				std::get<0>(this->m_output_size), std::get<1>(this->m_output_size),
				std::get<2>(this->m_output_size), std::get<3>(this->m_output_size));
	}

	virtual void serialize(std::string base_file_name)
	{
		// No default action
	}

	[[noreturn]] virtual Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false)
	{
		SKEPU_ERROR("Layer: calling pure virtual 'forward' member function");
	}

	[[noreturn]] virtual Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y)
	{
		SKEPU_ERROR("Layer: calling pure virtual 'backward' member function");
	}

	//		private:

	// Override for layer classes with trainable parameters.
	// For internal use. Requires the layer to be setup first.
	virtual void load_pretrained() { /* No default action */ }

	// Override for layer classes with trainable parameters.
	virtual void init_parameters() { /* No default action */ }

public: // for now, protected later?
	int m_layer_id = -1;
	Level m_layer_level = Level::Hidden;

	bool m_is_setup = false;
	bool m_in_place = false; // when possible

	skepu::PRNG *m_prng;

	size_t m_batch_size;
	Dimensions m_output_size;

	Act m_act = Act::None; // default value?
	Init m_init_weights = Init::Uniform;
	Init m_init_biases = Init::Zero;

	std::string m_pretrained_params = SKEPU_DNN_NO_PRETRAINED_PARAMS;

	std::shared_ptr<Layer> *predecessor = nullptr;
	std::shared_ptr<Layer> *successor = nullptr;

	float m_learning_rate = 0;
	Tensor4<T> m_vals;

	friend class SequentialLayer;
};

template <typename T = float>
class Dense : public Layer<T>
{
public:
	Dense(size_t arg_output_size) : Layer<T>(), m_out(arg_output_size) {}

	Dense(Dense &&source) : Layer<T>(std::move(source))
	{
		this->m_weights = std::move(source.m_weights);
		this->m_biases = std::move(source.m_biases);
		this->m_out = source.m_out;
	}

	~Dense() override = default;

	Dense &&weights(skepu::ml::Init arg_init) override
	{
		Layer<T>::weights(arg_init);
		return std::move(*this);
	}

	Dense &&biases(skepu::ml::Init arg_init) override
	{
		Layer<T>::biases(arg_init);
		return std::move(*this);
	}

	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " Dense values");
		this->m_weights.setLabel(this->label() + " Dense weights");
		this->m_biases.setLabel(this->label() + " Dense biases");
		this->m_dW.setLabel(this->label() + " Dense weight-deltas");
		this->m_db.setLabel(this->label() + " Dense bias-deltas");
		this->m_temp.setLabel(this->label() + " Dense gradient");

		size_t out = this->m_out;
		size_t in = std::get<3>(input_size);

		this->m_weights.init(in, out);
		this->m_biases.init(out);

		this->m_output_size = {std::get<0>(input_size), 1, 1, out};

		this->load_pretrained();
		Layer<T>::setup(input_size);

		// Backprop temporaries
		this->m_db.init(this->m_biases.size());
		this->m_dW.init(this->m_weights.size_i(), this->m_weights.size_j());
		this->m_temp.init(
			std::get<0>(input_size), std::get<1>(input_size),
			std::get<2>(input_size), std::get<3>(input_size)
		);
	}

	virtual void serialize(std::string base_file_name) override
	{
		skepu::external(this->label() + " serialize", skepu::read(this->m_weights, this->m_biases), [&]
		{
			// Todo: serialize to CSV
			std::ofstream ofs(base_file_name + "_" + this->label() + ".csv");

			for(size_t i = 0; i < this->m_weights.size_i(); i++){
				for(size_t j = 0; j < this->m_weights.size_j(); j++){
					ofs << this->m_weights(i,j) ;
					ofs << "\n";
				}
			}

			for(size_t i = 0; i < this->m_biases.size_i(); i++){
				ofs << this->m_biases(i) << "\n";
			}
		});
	}

	bool is_trainable() const override { return true; }

	Dense &&pretrained(std::string fname) override
	{
		Layer<T>::pretrained(fname);
		return std::move(*this);
	}

	void load_pretrained() override
	{
		if (this->m_pretrained_params != SKEPU_DNN_NO_PRETRAINED_PARAMS)
		{
			// Load data from file
			std::vector<float> temp = deserialize_csv_helper<float>(this->m_pretrained_params);
			auto it = load_from_vector(this->m_weights, temp, temp.begin());
			it = load_from_vector(this->m_biases, temp, it);
			dnn_debug << "Dense loaded biases: " << this->m_biases << "\n";
			dnn_debug << "Dense loaded weights: " << this->m_weights << "\n";
		}
	}

	void init_parameters() override
	{
		// Weights
		if (this->m_init_weights == Init::Xavier_uniform)
		{
			dnn_debug << "Dense initialize weights to random uniform\n";

			double limit = sqrt(6.0 / (this->m_weights.size_i() + this->m_out));

			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if(this->m_init_weights == Init::Xavier_normal){
			double limit = sqrt(2.0 / (this->m_weights.size_i() + this->m_out));

			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, 0.0 , limit); // this need to be normal random
		}
		else if (this->m_init_weights == Init::He_uniform){
			double limit = sqrt(6.0 / (this->m_weights.size_i()));

			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if (this->m_init_weights == Init::He_normal){
			double limit = sqrt(2.0 / (this->m_weights.size_i()));

			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, 0.0 , limit); // this need to be normal random
		}
		else if (this->m_init_weights == Init::Uniform)
		{
			dnn_debug << "Dense initialize weights to random uniform\n";

			double limit = 1.0;
			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if (this->m_init_weights == Init::Zero)
		{
			dnn_debug << "Dense initialize biases to zero\n";
			skel_init_zero.setLabel(this->label() + " Zero-init dense weights");
			skel_init_zero(this->m_weights);
		}

		// Biases
		if (this->m_init_biases == Init::Uniform)
		{
			dnn_debug << "Dense initialize weights to random uniform\n";
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init dense biases");
			skel_init_random_uniform(this->m_biases, -1.0, 1.0);
		}
		else if (this->m_init_biases == Init::Zero) {
			dnn_debug << "Dense initialize biases to zero\n";
			skel_init_zero.setLabel(this->label() + " Zero-init dense biases");
			skel_init_zero(this->m_biases);
		}

		dnn_debug << "Dense initialized biases: " << this->m_biases << "\n";
		dnn_debug << "Dense initialized weights: " << this->m_weights << "\n";
	}

	Dense &&activation(Act new_act) override
	{
		Layer<T>::activation(new_act);
		return std::move(*this);
	}

	std::string summary() override
	{
		std::ostringstream os;
		os << Layer<T>::summary() << "Dense layer params{("
			 << this->m_weights.size_i() << " x " << this->m_weights.size_j()
			 << ") + " << this->m_biases.size() << " = "
			 << (this->m_weights.size() + this->m_biases.size()) << "}";
		return os.str();
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false) override
	{
		//	dnn_debug << "Dense input: " << string_tensor_size(*inputs) << "\n";
		//	dnn_debug << "Dense vals: " << string_tensor_size(this->m_vals) << "\n";
		skel_dense.setLabel(this->label() + " Dense forward");
		skel_dense(this->m_vals, this->m_biases, this->m_weights, *inputs);
		//	dnn_debug << "Dense inputs: " << *inputs << "\n";
		//	dnn_debug << "Dense vals: " << this->m_vals << "\n";

		return &this->m_vals;
	}

	// inputs: tensor in shape (batch, 1, 1, N)
	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
		dnn_debug << "Dense backprop\n";
		dnn_debug << "Dense vals: " << string_tensor_size(this->m_vals) << "\n";
		dnn_debug << "Dense input: " << string_tensor_size(*inputs) << "\n";
		dnn_debug << "Dense gradient: " << string_tensor_size(*gradient) << "\n";
		dnn_debug << "Dense temp: " << string_tensor_size(this->m_temp) << "\n";
		dnn_debug << "Dense weights: " << this->m_weights.total_rows() << " x " << this->m_weights.total_cols() << "\n";

		skel_propagate_loss_gradient.setLabel(this->label() + " Dense backprop-grad");
		skel_propagate_dense_weight_gradient.setLabel(this->label() + " Dense backprop weights");
		skel_propagate_dense_bias_gradient.setLabel(this->label() + " Dense backprop biases");

		if (this->m_layer_level != Level::Input)
			skel_propagate_loss_gradient(this->m_temp, *gradient, this->m_weights);

		skel_propagate_dense_weight_gradient(this->m_dW, *inputs, *gradient);
		skel_propagate_dense_bias_gradient(this->m_db, *gradient);

		// skel_update_params.setLabel(this->label() + " Dense update weights");
		// skel_update_params(this->m_weights, this->m_weights, this->m_dW, this->m_learning_rate);
		// skel_update_params.setLabel(this->label() + " Dense update biases");
		// skel_update_params(this->m_biases, this->m_biases, this->m_db, this->m_learning_rate);

		dnn_debug << "Dense new biases: " << this->m_biases << "\n";
		dnn_debug << "Dense new weights: " << this->m_weights << "\n";

		return (this->m_layer_level != Level::Input) ? &this->m_temp : nullptr;
	}

	Matrix<T>&  get_d_weights() override { return m_weights; }
	Matrix<T>&  get_d_gradients() override { return m_dW; }

	Vector<T>&  get_biases() override { return m_biases; }
	Vector<T>&  get_biases_gradients() override { return m_db; }



protected:
	size_t m_out = 0;

	Matrix<T> m_weights;
	Vector<T> m_biases;

	Vector<T> m_db;
	Matrix<T> m_dW;
	Tensor4<T> m_temp;
};



template <typename T = float>
class Activation : public Layer<T>
{
public:
	Activation(Act arg_act) : Layer<T>() { this->m_act = arg_act; }

	// Move constructor
	Activation(Activation<T> &&source) = default;

	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " Activation values");
		this->m_temp_vals.setLabel(this->label() + " Activation temporaries");
		this->m_temp_backprop.setLabel(this->label() + " Activation gradients");

		this->m_output_size = input_size;

		//	if (!this->m_in_place) ...

		this->m_temp_vals.init(
			std::get<0>(this->m_output_size),
			std::get<1>(this->m_output_size),
			std::get<2>(this->m_output_size),
			1
		);

		this->m_temp_backprop.init(
			std::get<0>(this->m_output_size), std::get<1>(this->m_output_size),
			std::get<2>(this->m_output_size), std::get<3>(this->m_output_size)
		);

		Layer<T>::setup(input_size);
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false) override
	{
		//	dnn_debug << "Act input: " << string_tensor_size(*inputs) << "\n";
		//	dnn_debug << "Act vals: " << string_tensor_size(this->m_vals) << "\n";

		Tensor4<T> *target = (this->m_in_place) ? inputs : &this->m_vals;

		if (this->m_act == Act::ReLU)
		{
			skel_act_relu.setLabel(this->label() + " ReLU");
			skel_act_relu(*target, *inputs);
		}
		else if (this->m_act == Act::Sigmoid)
		{
			skel_act_sigmoid.setLabel(this->label() + " Sigmoid");
			skel_act_sigmoid(*target, *inputs);
		}
		else if (this->m_act == Act::SoftMax)
		{
			dnn_debug << "Softmax input: " << *inputs << "\n";
			// std::cout << "Softmax input: " << *inputs << "\n";
			
			Tensor4<T> m_temp_value_max_reduce (inputs->size_i(),inputs->size_j(),inputs->size_k(), 1);

			new_softmax_skel_1(m_temp_value_max_reduce, *inputs);
			new_softmax_skel_2(this->m_temp_vals, *inputs, m_temp_value_max_reduce);
			new_softmax_skel_3(*target, *inputs, this->m_temp_vals, m_temp_value_max_reduce);

			// skel_softmax_batched_1.setPoolSize(1, 1, 1, inputs->size_l());
			// skel_softmax_batched_1.setStride(1, 1, 1, inputs->size_l());
			// skel_softmax_batched_1.setLabel(this->label() + " Softmax A");
			// skel_softmax_batched_2.setLabel(this->label() + " Softmax B");
			// skel_softmax_batched_1(this->m_temp_vals, *inputs);
			// skel_softmax_batched_2(*target, *inputs, this->m_temp_vals);
		}
		else if (this->m_act == Act::TanH)
		{
			skel_act_tanh.setLabel(this->label() + " TanH");
			skel_act_tanh(*target, *inputs);
		}
		return target;
	}

	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
		dnn_debug << "Activation function backprop\n";
		//	dnn_debug << "Act input: " << string_tensor_size(*inputs) << "\n";
		//	dnn_debug << "Act vals: " << string_tensor_size(this->m_vals) << "\n";

		//	Tensor4<T> *target = (this->m_in_place) ? inputs : &this->m_vals;
		Tensor4<T> *target = gradient;

		if (this->m_act == Act::ReLU)
		{
			skel_act_relu_prime.setLabel(this->label() + " ReLU derivative");
			skel_act_relu_prime(*target, *inputs, *gradient);
		}
		else if (this->m_act == Act::Sigmoid)
		{
			skel_act_sigmoid_prime.setLabel(this->label() + " Sigmoid derivative");
			skel_act_sigmoid_prime(*target, *inputs);
		}
		else if (this->m_act == Act::SoftMax)
		{

		  int n_class = this->m_vals.size_l();

		  skepu::Tensor3<float> jacobian(inputs->size_i(),n_class, n_class);

		  softmax_jacobian(jacobian, this->m_vals);
		  softmax_prime(*target, *gradient, jacobian, n_class);

		}
		else if (this->m_act == Act::TanH)
		{
			skel_act_tanh_prime.setLabel(this->label() + " TanH derivative");
			skel_act_tanh_prime(*target, *inputs);
		}

		return target;
	}

	std::string summary() override
	{
		std::ostringstream os;
		os << Layer<T>::summary();
		os << "Activation layer: ";
		if (this->m_act == Act::ReLU)
			os << "ReLU";
		else if (this->m_act == Act::Sigmoid)
			os << "Sigmoid";
		else if (this->m_act == Act::SoftMax)
			os << "SoftMax";
		else if (this->m_act == Act::TanH)
			os << "TanH";
		os << (this->m_in_place ? " (in-place) " : " (not in-place) ");
		return os.str();
	}

private:
	Tensor4<T> m_temp_vals, m_temp_backprop;
};



template <typename T = float>
class Conv1D : public Layer<T> {
public:
	Conv1D(size_t no_features) : Layer<T>(), m_no_features(no_features) {}

	// Move constructor
	Conv1D(Conv1D &&source)
	: Layer<T>(std::move(source)), m_no_features(source.m_no_features)
	{
		this->m_kernel = source.m_kernel;
		this->m_stride = source.m_stride;
		this->m_padding = source.m_padding;
		this->m_no_features = source.m_no_features;
		this->m_weights = std::move(source.m_weights);
		this->m_biases = std::move(source.m_biases);
	}

	~Conv1D() override = default;

	Conv1D &&weights(skepu::ml::Init arg_init) override
	{
		Layer<T>::weights(arg_init);
		return std::move(*this);
	}

	Conv1D &&biases(skepu::ml::Init arg_init) override
	{
		Layer<T>::biases(arg_init);
		return std::move(*this);
	}

	void setup(Dimensions input_size) override
	{
		Layer<T>::setup(input_size);

		size_t out = std::get<0>(this->m_output_size);
		size_t in = std::get<0>(input_size);

		this->m_weights.init(this->m_kernel, this->m_no_features);
		this->m_biases(in);

		this->m_vals.init(this->m_batch_size, 1, in, this->m_no_features);
		this->m_output_size = input_size;

		this->load_pretrained();
	}

	virtual void serialize(std::string base_file_name) override
	{
		skepu::external(this->label() + " serialize", skepu::read(this->m_weights, this->m_biases), [&]
		{
			// Todo: serialize to CSV
			// std::ofstream ofs(base_file_name + this->label() + ".csv");
		});
	}

	bool is_trainable() const override { return true; }

	Conv1D &&pretrained(std::string fname) override
	{
		Layer<T>::pretrained(fname);
		return std::move(*this);
	}

	void load_pretrained() override
	{
		if (this->m_pretrained_params != SKEPU_DNN_NO_PRETRAINED_PARAMS)
		{
			// Load data from file
			std::vector<float> temp = deserialize_csv_helper<float>(this->m_pretrained_params);
			auto it = load_from_vector(this->m_weights, temp, temp.begin());
			it = load_from_vector(this->m_biases, temp, it);
		}
	}

	void init_parameters() override
	{
		// Weights
		if (this->m_init_weights == Init::Uniform)
		{
			dnn_debug << "Conv1D initialize weights to random uniform\n";
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init Conv1D weights");
			skel_init_random_uniform(this->m_weights, -1.0, 1.0);
		}
		else if (this->m_init_weights == Init::Zero)
		{
			dnn_debug << "Conv1D initialize biases to zero\n";
			skel_init_zero.setLabel(this->label() + " Zero-init Conv1D weights");
			skel_init_zero(this->m_weights);
		}

		// Biases
		if (this->m_init_biases == Init::Uniform)
		{
			dnn_debug << "Conv1D initialize weights to random uniform\n";
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init Conv1D biases");
			skel_init_random_uniform(this->m_biases, -1.0, 1.0);
		}
		else if (this->m_init_biases == Init::Zero)
		{
			dnn_debug << "Conv1D initialize biases to zero\n";
			skel_init_zero.setLabel(this->label() + " Zero-init Conv1D biases");
			skel_init_zero(this->m_biases);
		}
	}

	Conv1D &&activation(Act new_act) override
	{
		Layer<T>::activation(new_act);
		return std::move(*this);
	}

	Conv1D &&kernel(size_t new_kernel)
	{
		this->m_kernel = new_kernel;
		return std::move(*this);
	}

	Conv1D &&strides(size_t new_stride)
	{
		this->m_stride = new_stride;
		return std::move(*this);
	}

	Conv1D &&padding(bool new_padding)
	{
		this->m_padding = new_padding;
		return std::move(*this);
	}

	// Overridden members

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "Conv1D layer with " << this->m_no_features
			 << " features; kernel size ";
		os << "{" << this->m_kernel << "}";
		os << "; stride ";
		os << "{" << this->m_stride << "}";
		os << " train_params{";
		os << this->m_weights.size() << " + ";
		os << this->m_biases.size() << "}";
		return os.str();
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false) override
	{
		skel_conv_1d.setOverlap(0, 0, (this->m_kernel - 1) / 2, 0);
		skel_conv_1d.setStride(1, 1, this->m_stride, 1);

		if (this->m_padding)
		{
			skel_conv_1d.setEdgeMode(skepu::Edge::Pad);
			skel_conv_1d.setPad(0);
		}

		skel_conv_1d(this->m_vals, *inputs, this->m_weights, this->m_biases);

		return inputs;
	}

	[[noreturn]] Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
		dnn_debug << "Not implemented\n";
		exit(1);
	}

protected:
	int m_no_features;
	size_t m_kernel{3};    // default
	size_t m_stride{1};    // default
	bool m_padding = true; // true = "same" in Keras, false = "valid" in Keras

	Matrix<T> m_weights; // TODO: 4D tensor here?
	Vector<T> m_biases;
};






float conv_2d_backward_uf(
	skepu::Index4D index,
	skepu::Region4D<float> gradient,
	skepu::Ten4<float> DNN_CONST weights)
{

		float res = 0;
		for (int i = -gradient.oj; i <= gradient.oj; ++i)
			for (int j = -gradient.ok; j <= gradient.ok; ++j){
				for (size_t l = 0; l < weights.size_l; ++l) {
	        res += weights(weights.size_i - (i + gradient.oj) - 1, weights.size_j - (j + gradient.ok) - 1, index.l, l) * gradient(0, i, j, l - index.l);
				}
			}
		return res;
}

float propagate_conv_2d_weight_gradient_uf(
	skepu::Index4D index,
	skepu::Ten4<float> inputs,
	skepu::Ten4<float> gradient)
{
	size_t num_batches = inputs.size_i;
	float sum = 0;
	for (size_t b = 0; b < num_batches; ++b)
		for (size_t j = 0; j < gradient.size_j; ++j)
			for (size_t k = 0; k < gradient.size_k; ++k)
				// sum += inputs(b, j + index.i, k + index.j, index.k) * gradient(b, gradient.size_j - 1 - j, gradient.size_k - 1 - k, index.l); // indexing!!!
        sum += inputs(b, j + index.i, k + index.j, index.k) * gradient(b,  j, k , index.l);
	return sum;
}

float propagate_conv_2d_bias_gradient_uf(
	skepu::Index1D index,
	skepu::Ten4<float> gradient)
{
	float sum = 0;
	size_t num_batches = gradient.size_i;
  	for (size_t b = 0; b < num_batches; ++b){
    for (size_t j = 0; j < gradient.size_j; ++j){
      for (size_t k = 0; k < gradient.size_k; ++k){
        sum += gradient(b, j, k, index.i);
      }
    }
  }
	return sum;
}

auto skel_conv_2d_backward = skepu::MapOverlap(conv_2d_backward_uf);
auto skel_propagate_conv_2d_weight_gradient = skepu::Map<0>(propagate_conv_2d_weight_gradient_uf);
auto skel_propagate_conv_2d_bias_gradient = skepu::Map<0>(propagate_conv_2d_bias_gradient_uf);



template <typename T = float>
class Conv2D : public Layer<T>
{
public:
	Conv2D(int no_features) : Layer<T>(), m_no_features(no_features) {}

	// Move constructor
	Conv2D(Conv2D &&source)
	: Layer<T>(std::move(source)), m_no_features(source.m_no_features)
	{
		this->m_kernel = std::move(source.m_kernel);
		this->m_strides = std::move(source.m_strides);
		this->m_padding = std::move(source.m_padding);
		this->m_no_features = source.m_no_features;
		this->m_weights.swap(source.m_weights);
		this->m_biases.swap(source.m_biases);
	}

	~Conv2D() override = default;

	Conv2D &&weights(skepu::ml::Init arg_init) override
	{
		Layer<T>::weights(arg_init);
		return std::move(*this);
	}

	Conv2D &&biases(skepu::ml::Init arg_init) override
	{
		Layer<T>::biases(arg_init);
		return std::move(*this);
	}

	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " Conv2D values");
		this->m_weights.setLabel(this->label() + " Conv2D weights");
		this->m_biases.setLabel(this->label() + " Conv2D biases");
		this->m_dW.setLabel(this->label() + " Conv2D weight deltas");
		this->m_db.setLabel(this->label() + " Conv2D bias deltas");
		this->m_temp.setLabel(this->label() + " Conv2D gradient");

		size_t out = std::get<0>(this->m_output_size);
		size_t in = std::get<0>(input_size);

		this->m_weights.init(
			std::get<0>(this->m_kernel),
			std::get<1>(this->m_kernel),
			std::get<3>(input_size),
			this->m_no_features
		);
		this->m_biases.init(this->m_no_features);

		this->m_output_size = {
				std::get<0>(input_size),
				std::get<1>(input_size) - ((std::get<0>(this->m_kernel) - 1)),
				std::get<2>(input_size) - ((std::get<1>(this->m_kernel) - 1)),
				this->m_no_features
		};

		this->load_pretrained();

		Layer<T>::setup(input_size);

		// Backprop temporaries
		this->m_db.init(this->m_biases.size());
		this->m_dW.init(
			this->m_weights.size_i(), this->m_weights.size_j(),
			this->m_weights.size_k(), this->m_weights.size_l()
		);
		this->m_temp.init(
			std::get<0>(input_size), std::get<1>(input_size),
			std::get<2>(input_size), std::get<3>(input_size)
		);
	}

	virtual void serialize(std::string base_file_name) override
	{
		skepu::external(this->label() + " serialize", skepu::read(this->m_weights, this->m_biases), [&]
		{
			// Todo: serialize to CSV
			std::ofstream ofs(base_file_name + "_" + this->label() + ".csv");

			for(size_t i = 0; i < this->m_weights.size_i(); i++){
				for(size_t j = 0; j < this->m_weights.size_j(); j++){
					for(size_t k = 0; k < this->m_weights.size_k(); k++){
						for(size_t l = 0; l < this->m_weights.size_l(); l++){
							ofs << this->m_weights(i,j,k,l) << "\n";
						}
					}
				}
			}


			for(size_t i = 0; i < this->m_biases.size_i(); i++){
				ofs << this->m_biases(i) << "\n";
			}

		});
	}

	bool is_trainable() const override { return true; }

	Conv2D &&pretrained(std::string fname) override
	{
		Layer<T>::pretrained(fname);
		return std::move(*this);
	}

	void load_pretrained() override
	{
		if (this->m_pretrained_params != SKEPU_DNN_NO_PRETRAINED_PARAMS)
		{
			// Load data from file
			std::vector<float> temp = deserialize_csv_helper<float>(this->m_pretrained_params);
			auto it = load_from_vector(this->m_weights, temp, temp.begin());
			it = load_from_vector(this->m_biases, temp, it);
			dnn_debug << "Conv2D loaded biases: " << this->m_biases << "\n";
			dnn_debug << "Conv2D loaded weights: " << this->m_weights << "\n";
		}
	}

	void init_parameters() override
	{
		// Weights
		if (this->m_init_weights == Init::Xavier_uniform)
		{
			dnn_debug << "Conv2D initialize weights to random uniform\n";
			int fan_in = this->m_weights.size_k() * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  int fan_out = this->m_no_features * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  double limit = std::sqrt(6.0 / (fan_in + fan_out));

			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init Conv2D weights");
			skel_init_random_uniform(this->m_weights, -limit, limit);
			// std::cout << "CL weights: " <<this->m_weights << '\n';
		}
		else if(this->m_init_weights == Init::Xavier_normal){
			int fan_in = this->m_weights.size_k() * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  int fan_out = this->m_no_features * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  double limit = std::sqrt(2.0 / (fan_in + fan_out));

			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init Conv2D weights");
			skel_init_random_uniform(this->m_weights, 0.0, limit);
		}
		else if(this->m_init_weights == Init::He_uniform){
			int fan_in = this->m_weights.size_k() * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  // int fan_out = this->m_no_features * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  double limit = std::sqrt(6.0 / (fan_in));

			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init Conv2D weights");
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if(this->m_init_weights == Init::He_normal){
			int fan_in = this->m_weights.size_k() * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  // int fan_out = this->m_no_features * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  double limit = std::sqrt(2.0 / (fan_in));

			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init Conv2D weights");
			skel_init_random_uniform(this->m_weights, 0.0, limit);
		}
		else if (this->m_init_weights == Init::Uniform)
		{
			dnn_debug << "Conv2D initialize weights to random uniform\n";
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init Conv2D weights");
			skel_init_random_uniform(this->m_weights, -1.0, 1.0);

		}
		else if (this->m_init_weights == Init::Zero)
		{
			dnn_debug << "Conv2D initialize weights to zero\n";
			skel_init_zero.setLabel(this->label() + " Zero-init Conv2D weights");
			skel_init_zero(this->m_weights);
		}

		// Biases
		if (this->m_init_biases == Init::Uniform)
		{
			dnn_debug << "Conv2D initialize biases to random uniform\n";
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init Conv2D biases");
			skel_init_random_uniform(this->m_biases, -1.0, 1.0);
		}
		else if (this->m_init_biases == Init::Zero)
		{
			dnn_debug << "Conv2D initialize biases to zero\n";
			skel_init_zero.setLabel(this->label() + " Zero-init Conv2D biases");
			skel_init_zero(this->m_biases);
		}
	}

	Conv2D &&activation(Act new_act) override
	{
		Layer<T>::activation(new_act);
		return std::move(*this);
	}

	Conv2D &&kernel(size_t k1, size_t k2)
	{
		this->m_kernel = {k1, k2};
		return std::move(*this);
	}

	Conv2D &&stride(size_t s1, size_t s2)
	{
		this->m_strides = {s1, s2};
		return std::move(*this);
	}

	Conv2D &&padding(bool new_padding)
	{
		this->m_padding = new_padding;
		return std::move(*this);
	}

	// Overridden members

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "Conv2D layer with " << this->m_no_features
			 << " features and kernel size ";
		os << "{" << std::get<0>(this->m_kernel) << ", "
			 << std::get<1>(this->m_kernel) << "}";
		os << " train_params{ (" << this->m_weights.size_i() << " x "
			 << this->m_weights.size_j() << " x " << this->m_weights.size_k() << " x "
			 << this->m_weights.size_l() << ")"
			 << (this->m_weights.size_i() * this->m_weights.size_j() *
					 this->m_weights.size_k() * this->m_weights.size_l())
			 << " + " << this->m_biases.size() << "}";
		os << " padding{" << (this->m_padding ? "on" : "off") << "}";
		return os.str();
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false) override
	{
		dnn_debug << "Conv2D input: " << string_tensor_size(*inputs) << "\n";
		dnn_debug << "Conv2D vals: " << string_tensor_size(this->m_vals) << "\n";

		skel_conv_2d.setOverlap(0, (std::get<0>(this->m_kernel) - 1) / 2, (std::get<1>(this->m_kernel) - 1) / 2, /*this->m_no_features - 1*/ 0);
		skel_conv_2d.setStride(1, std::get<0>(this->m_strides), std::get<1>(this->m_strides), 1);

		if (this->m_padding)
		{
			skel_conv_2d.setEdgeMode(skepu::Edge::Pad);
			skel_conv_2d.setPad(0);
		}

		dnn_debug << "Conv2D weights: " << this->m_weights << "\n";
		dnn_debug << "Conv2D biases: " << this->m_biases << "\n";

		skel_conv_2d.setLabel(this->label() + " Conv2D forward");
        std::cout << this->m_vals.size_j() << " vs " << inputs->size_j() << ", overlap=" << std::get<1>(skel_conv_2d.getOverlap()) << "\n";
		skel_conv_2d(this->m_vals, *inputs, this->m_weights, this->m_biases);

		dnn_debug << "Conv2D inputs: " << *inputs << "\n";
		dnn_debug << "Conv2D vals: " << this->m_vals << "\n";
		// std::cout << "Conv2D inputs: " << *inputs << "\n";
		// std::cout << "Conv2D vals: " << this->m_vals << "\n";

		return &this->m_vals;
	}

	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
		dnn_debug << "Conv2D backprop\n";
		dnn_debug << "Conv2D vals size: " << string_tensor_size(this->m_vals) << "\n";
		dnn_debug << "Conv2D input size: " << string_tensor_size(*inputs) << "\n";
		dnn_debug << "Conv2D gradient size: " << string_tensor_size(*gradient) << "\n";
		dnn_debug << "Conv2D temp size: " << string_tensor_size(this->m_temp) << "\n";
		dnn_debug << "Conv2D inputs: " << *inputs << "\n";
		dnn_debug << "Conv2D gradient: " << *gradient << "\n";

		if (this->m_layer_level != Level::Input)
		{
			skel_conv_2d_backward.setOverlap(0, (std::get<0>(this->m_kernel) - 1) / 2, (std::get<1>(this->m_kernel) - 1) / 2, /*this->m_no_features - 1*/ 0);
			skel_conv_2d_backward.setStride(1, std::get<0>(this->m_strides), std::get<1>(this->m_strides), 1); // incorrect - future work...
			skel_conv_2d_backward.setEdgeMode(skepu::Edge::Pad);
			skel_conv_2d_backward.setPad(0);
			skel_conv_2d_backward.setLabel(this->label() + " Conv2D backprop grad");
			skel_conv_2d_backward(this->m_temp, *gradient, this->m_weights);
			dnn_debug << "Conv2D temp: " << this->m_temp << "\n";
		}

		skel_propagate_conv_2d_weight_gradient.setLabel(this->label() + " Conv2D backprop weights");
		skel_propagate_conv_2d_bias_gradient.setLabel(this->label() + " Conv2D backprop biases");
		skel_propagate_conv_2d_weight_gradient(this->m_dW, *inputs, *gradient);
		skel_propagate_conv_2d_bias_gradient(this->m_db, *gradient);

		// skel_update_params.setLabel(this->label() + " Conv2D update weights");
		// skel_update_params(this->m_weights, this->m_weights, this->m_dW, this->m_learning_rate);
		// skel_update_params.setLabel(this->label() + " Conv2D update biases");
		// skel_update_params(this->m_biases,  this->m_biases,  this->m_db, this->m_learning_rate);

		dnn_debug << "Conv2D dW: " << this->m_dW << "\n";
		dnn_debug << "Conv2D db: " << this->m_db << "\n";
		dnn_debug << "Conv2D new biases: " << this->m_biases << "\n";
		dnn_debug << "Conv2D new weights: " << this->m_weights << "\n";

		return (this->m_layer_level != Level::Input) ? &this->m_temp : nullptr;
	}

	Tensor4<T>& get_weights() override { return m_weights; }
	Tensor4<T>& get_gradients() override { return m_dW; }

	Vector<T>& get_biases() override {return m_biases; }
	Vector<T>& get_biases_gradients() override {return m_db; }


protected:
	int m_no_features;
	std::tuple<int, int> m_kernel{3, 3};  // default
	std::tuple<int, int> m_strides{1, 1}; // default
	bool m_padding = false; // true = "same" in Keras, false = "valid" in Keras

	Tensor4<T> m_weights;
	Vector<T> m_biases;

	Vector<T> m_db;
	Tensor4<T> m_dW;
	Tensor4<T> m_temp;
};



template <typename T = float>
class Subsampling2D : public Layer<T>
{
public:
	Subsampling2D() : Layer<T>() {}

	// Move constructor
	Subsampling2D(Subsampling2D &&source) : Layer<T>(std::move(source))
	{
		this->m_kernel = source.m_kernel;
		this->m_strides = source.m_strides;
		this->m_padding = source.m_padding;
		this->m_vals = std::move(source.m_vals);
	}

	~Subsampling2D() override = default;

	void setup(Dimensions input_size) override
	{
		size_t out = std::get<0>(this->m_output_size);
		size_t in = std::get<0>(input_size);

		this->m_output_size = {
				std::get<0>(input_size),
				std::get<1>(input_size) / (std::get<0>(this->m_kernel)),
				std::get<2>(input_size) / (std::get<1>(this->m_kernel)),
				std::get<3>(input_size)
		};

		Layer<T>::setup(input_size);
	}

	virtual Subsampling2D &&kernel(size_t k1, size_t k2)
	{
		this->m_kernel = {k1, k2};
		return std::move(*this);
	}

	virtual Subsampling2D &&stride(size_t s1, size_t s2)
	{
		this->m_strides = {s1, s2};
		return std::move(*this);
	}

	virtual Subsampling2D &&padding(bool new_padding)
	{
		this->m_padding = new_padding;
		return std::move(*this);
	}

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "Subsampling layer with kernel size ";
		os << "{" << std::get<0>(this->m_kernel) << ", "
			 << std::get<1>(this->m_kernel) << "}";
		return os.str();
	}

protected:
	int no_features;
	std::tuple<int, int> m_kernel{1, 1};  // default
	std::tuple<int, int> m_strides{1, 1}; // default to pool size
	bool m_padding = true; // true = "same" in Keras, false = "valid" in Keras
};



template <typename T = float>
class MaxPooling2D : public Subsampling2D<T>
{
public:
	MaxPooling2D() : Subsampling2D<T>() {}

	MaxPooling2D(MaxPooling2D &&source)
			: Subsampling2D<T>(std::move(source)) // todo kernel
	{
		this->m_kernel = source.m_kernel;
		this->m_strides = source.m_strides;
		this->m_padding = source.m_padding;
	}

	~MaxPooling2D() override = default;

	MaxPooling2D &&kernel(size_t k1, size_t k2) override
	{
		Subsampling2D<T>::kernel(k1, k2);
		return std::move(*this);
	}

	MaxPooling2D &&stride(size_t s1, size_t s2) override
	{
		Subsampling2D<T>::stride(s1, s2);
		return std::move(*this);
	}

	MaxPooling2D &&padding(bool new_padding) override
	{
		Subsampling2D<T>::padding(new_padding);
		return std::move(*this);
	}

	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " MaxPool2D values");
		this->m_temp.setLabel(this->label() + " MaxPool2D gradients");

		Subsampling2D<T>::setup(input_size);

		// Backprop temporaries
		this->m_temp.init(
			std::get<0>(input_size), std::get<1>(input_size),
			std::get<2>(input_size), std::get<3>(input_size)
		);

#ifdef SKEPU_DNN_USE_MASKED_POOL
		this->m_mask.setLabel(this->label() + " MaxPool2D mask");
		this->m_mask.init(
			std::get<0>(this->m_output_size), std::get<1>(this->m_output_size),
			std::get<2>(this->m_output_size), std::get<3>(this->m_output_size)
		);
#endif
	}

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "MaxPooling2D layer with kernel size ";
		os << "{" << std::get<0>(this->m_kernel) << ", " << std::get<1>(this->m_kernel) << "}";
		return os.str();
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false) override
	{
		dnn_debug << "MaxPool2D input: " << string_tensor_size(*inputs) << "\n";
		dnn_debug << "MaxPool2D vals: " << string_tensor_size(this->m_vals) << "\n";

#ifdef SKEPU_DNN_USE_MASKED_POOL
//		std::cout << "Masked pool\n";
		skel_pool_max_masked.setPoolSize(1, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel), 1);
		// todo: override with set strides
		skel_pool_max_masked.setStride(1, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel), 1);
		skel_pool_max_masked.setLabel(this->label() + " MaxPool2D forward");
		skel_pool_max_masked(this->m_vals, this->m_mask, *inputs);
#else
		skel_pool_max.setPoolSize(1, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel), 1);
		// todo: override with set strides
		skel_pool_max.setStride(1, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel), 1);
		skel_pool_max.setLabel(this->label() + " MaxPool2D forward");
		skel_pool_max(this->m_vals, *inputs);
#endif

		dnn_debug << "MaxPool2D inputs: " << *inputs << "\n";
		dnn_debug << "MaxPool2D vals: " << this->m_vals << "\n";

		return &this->m_vals;
	}

	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
		dnn_debug << "MaxPool2D backprop\n";
		dnn_debug << "MaxPool2D vals size: " << string_tensor_size(this->m_vals) << "\n";
		dnn_debug << "MaxPool2D input size: " << string_tensor_size(*inputs) << "\n";
		dnn_debug << "MaxPool2D gradient size: " << string_tensor_size(*gradient) << "\n";
		dnn_debug << "MaxPool2D temp size: " << string_tensor_size(this->m_temp) << "\n";

#ifdef SKEPU_DNN_USE_MASKED_POOL
		skel_pool_max_masked_backward.setLabel(this->label() + " MaxPool2D backward");
		skel_pool_max_masked_backward(this->m_temp, *gradient, this->m_mask, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel));
#else
		skel_pool_max_backward.setLabel(this->label() + " MaxPool2D backward");
		skel_pool_max_backward(this->m_temp, *inputs, *gradient, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel));
#endif

		dnn_debug << "MaxPool2D inputs: " << *inputs << "\n";
		dnn_debug << "MaxPool2D gradient: " << *gradient << "\n";
		dnn_debug << "MaxPool2D vals: " << this->m_temp << "\n";

		return &this->m_temp;
	}

private:

	Tensor4<T> m_temp;
#ifdef SKEPU_DNN_USE_MASKED_POOL
	Tensor4<char> m_mask;
#endif
};



template <typename T = float>
class Dropout : public Layer<T>
{
public:
	Dropout(float arg_rate) : Layer<T>(), m_dropout_rate(arg_rate) {}

	// Move constructor
	Dropout(Dropout &&source)
	: Layer<T>(std::move(source)), m_dropout_rate(source.m_dropout_rate)
	{
		this->m_prng = std::move(source.m_prng);
	}

	~Dropout() override = default;

	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " Dropout values");
		this->m_mask.setLabel(this->label() + " Dropout mask");
		this->m_temp.setLabel(this->label() + " Dropout gradient");

		size_t out = std::get<0>(this->m_output_size);
		size_t in = std::get<0>(input_size);

		this->m_output_size = input_size;
		//	if (!this->m_in_place) ...
		Layer<T>::setup(input_size);

		this->m_temp.init(
			std::get<0>(input_size), std::get<1>(input_size),
			std::get<2>(input_size), std::get<3>(input_size)
		);
		this->m_mask.init(
			std::get<0>(input_size), std::get<1>(input_size),
			std::get<2>(input_size), std::get<3>(input_size)
		);
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false) override
	{
		if (!is_training) return inputs;

		// During training, drop values
		skel_dropout_masked.setPRNG(*this->m_prng);
		skel_dropout_masked.setLabel(this->label() + " Dropout forward");
		skel_dropout_masked(this->m_vals, this->m_mask, *inputs, this->m_dropout_rate);
		return &this->m_vals;
	}

	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
		skel_dropout_backprop_masked.setLabel(this->label() + " Dropout backward");
		skel_dropout_backprop_masked(this->m_temp, *gradient, this->m_mask, this->m_dropout_rate);
		return &this->m_temp;
	}

	std::string summary() override {
		std::stringstream os;
		os << Layer<T>::summary();
		os << "Dropout layer with rate " << this->m_dropout_rate;
		return os.str();
	}

private:
	float m_dropout_rate;
	Tensor4<char> m_mask;
	Tensor4<T> m_temp;
};



template <typename T = float>
class Flatten : public Layer<T>
{
public:
	Flatten() : Layer<T>() {}
	Flatten(Flatten &&) = default;
	~Flatten() override = default;

	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " Flatten values");
		this->m_temp.setLabel(this->label() + " Flatten gradient");

		this->m_output_size = {
			std::get<0>(input_size), 1, 1,
			std::get<1>(input_size) * std::get<2>(input_size) * std::get<3>(input_size)
		};

		this->m_temp.init(
			std::get<0>(input_size), std::get<1>(input_size),
			std::get<2>(input_size), std::get<3>(input_size)
		);

		Layer<T>::setup(input_size);
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool is_training=false) override
	{
		skel_flatmap.setLabel(this->label() + " Flatten forward");
		skel_flatmap(this->m_vals, *inputs);
		return &this->m_vals;
	}

	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
		dnn_debug << "Flatten backprop\n";
		dnn_debug << "Flatten vals size: " << string_tensor_size(this->m_vals) << "\n";
		dnn_debug << "Flatten input size: " << string_tensor_size(*inputs) << "\n";
		dnn_debug << "Flatten gradient size: " << string_tensor_size(*gradient) << "\n";
		dnn_debug << "Flatten temp size: " << string_tensor_size(this->m_temp) << "\n";

		skel_flatmap.setLabel(this->label() + " Flatten backward");
		skel_flatmap(this->m_temp, *gradient);

		dnn_debug << "Flatten inputs: " << *inputs << "\n";
		dnn_debug << "Flatten gradient: " << *gradient << "\n";
		dnn_debug << "Flatten temp: " << this->m_temp << "\n";

		return &this->m_temp;
	}

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "Flatten layer ";
		return os.str();
	}

private:

	Tensor4<T> m_temp;
};

template <typename T = float>
class LossFunction{
	private:
	Loss loss_type; 
	
	
	public: 
	// constructor 
	LossFunction(Loss type): loss_type(type){}

	// forward pass 
	float compute_loss(Tensor4<T>& batched_train_y, Tensor4<T>& predicted, size_t batch_size){
		skepu::Vector<float> temp_loss(batch_size, "Loss temp");
		float computed_loss = 0;

		if (loss_type == Loss::CategoricalCrossEntropy){
			skel_cross_entropy_batched_loss_map(temp_loss, batched_train_y, predicted);	
	 		computed_loss += -skel_cross_entropy_batched_loss_reduce(temp_loss) / batch_size;
			// std::cout<<"computed_loss" << computed_loss;
		}
		else if (loss_type == Loss::BinaryCrossEntropy){
			skel_binary_cross_entropy_loss_map(temp_loss, batched_train_y, predicted);
    		computed_loss += -skel_cross_entropy_batched_loss_reduce(temp_loss) / batch_size;	
		}

		return computed_loss;
	}

	// backward pass
	Tensor4<T> compute_gradient(Tensor4<T>& predicted, Tensor4<T>& batched_train_y){
		Tensor4<T> initial_gradient( // move out of loop
						predicted.size_i(), predicted.size_j(),
						predicted.size_k(), predicted.size_l(),
						"Initial gradient"
				);

		if(loss_type == Loss::CategoricalCrossEntropy){
			skel_delta(initial_gradient, predicted, batched_train_y);			
		}
		else if (loss_type == Loss::BinaryCrossEntropy){
			skel_delta_bce(initial_gradient, predicted, batched_train_y);			
		}
		return initial_gradient;	
	}

};

template<typename T = float>
class Optimizer{
	private:
	Opt optimizer_type;

	public:
	Optimizer(Opt opt_type): optimizer_type(opt_type){}

	void step(Layer<T>& layer){
		if(optimizer_type == Opt::SGD){
			if(auto* conv_layer = dynamic_cast<Conv2D<T>*>(&layer)){
				Tensor4<T>& m_weights = layer.get_weights();
				Tensor4<T>& m_dW  = layer.get_gradients();

				Vector<T>& m_biases = layer.get_biases();
				Vector<T>& m_db = layer.get_biases_gradients();
				auto learning_rate = layer.m_learning_rate;

			// skel_update_params.setLabel(this->label() + " Dense update weights");
			skel_update_params(m_weights, m_weights, m_dW, learning_rate);
			// skel_update_params.setLabel(this->label() + " Dense update biases");
			skel_update_params(m_biases, m_biases, m_db, learning_rate);
			}// conv
			else if(auto* conv_layer = dynamic_cast<Dense<T>*>(&layer)){
				Matrix<T>& m_weights = layer.get_d_weights(); 
				Matrix<T>& m_dW  = layer.get_d_gradients();

				Vector<T>& m_biases = layer.get_biases();
				Vector<T>& m_db = layer.get_biases_gradients();
				auto learning_rate = layer.m_learning_rate;
				
				skel_update_params(m_weights, m_weights, m_dW, learning_rate);
				skel_update_params(m_biases, m_biases, m_db, learning_rate);

			}// dense 
			
		}// end if sgd
	}

};



template <typename T = float>
class SequentialModel
{
public:
	SequentialModel(Dimensions arg_input_size)
	: m_input_size{arg_input_size}, m_batch_size{std::get<0>(arg_input_size)} {}

	SequentialModel(size_t arg_input_size, size_t arg_batch_size)
	: m_input_size{arg_input_size, 1, 1, 1}, m_batch_size{arg_batch_size} {}

	// move constructor
	SequentialModel(std::vector<std::shared_ptr<Layer<T>>> &&arg_layers)
	: m_layers{arg_layers}
	{
		this->init();
	}

	void init()
	{
		if (this->m_initialized)
			SKEPU_ERROR("SkePU-DNN: Attempting to initialize an already initialized SequentialModel");

		// Connect layers together in internal doubly-linked list
		std::shared_ptr<Layer<T>> *prev = nullptr;
		for (auto &current : this->m_layers)
		{
			current->predecessor = prev;
			if (prev)
				prev->get()->successor = &current;
		}

		this->m_layers[0]->m_layer_level = Level::Input;
		this->m_layers[this->m_layers.size() - 1]->m_layer_level = Level::Output;

		// Allocate data
		tracing::tracer().beginRegion("Model data allocation", __LINE__);
		Dimensions input_size = this->m_input_size;
		for (auto &current : this->m_layers)
		{
			//	dnn_debug << "Input size: " << std::get<0>(input_size) << ", " << std::get<1>(input_size) << ", " << std::get<2>(input_size) << ", " << std::get<3>(input_size) << ", " << "\n";
			//dnn_debug << current->summary() << "\n";
			current->setup(input_size); // on each layer
			input_size = current->m_output_size;
		}
		tracing::tracer().endRegion();

		this->m_initialized = true;
	}

	template <typename ConcreteLayer>
	void add(ConcreteLayer &&l)
	{
		Act activationFunc = l.m_act;
		l.m_batch_size = this->m_batch_size;
		l.m_layer_id = ++ this->m_layers_count;//this->m_layers.size();
		l.m_prng = &this->m_prng;
		this->m_layers.push_back(std::make_shared<typename std::remove_reference<ConcreteLayer>::type>(std::forward<ConcreteLayer>(l)));

		if (activationFunc != Act::None)
		{
			auto act_layer = std::make_shared<Activation<T>>(activationFunc);
			act_layer->m_batch_size = this->m_batch_size;
			act_layer->m_layer_id = this->m_layers_count; //this->m_layers.size(); // or are activations not their own layer?
			this->m_layers.push_back(act_layer);
		}
	}

	void init_parameters()
	{
		tracing::tracer().region("Model param init", __LINE__, [&]
		{
			for (auto &current : this->m_layers)
				current->init_parameters();
		});
	}

	void serialize(std::string base_file_name)
	{
		tracing::tracer().region("Serialize model parameters", __LINE__, [&]
		{
			for (auto &current : this->m_layers)
				current->serialize(base_file_name);
		});
	}

	Tensor4<T> &forward(Tensor4<T> &inputs, bool is_training=false)
	{
		if (!this->m_initialized)
			SKEPU_ERROR("SkePU-DNN: Attempting to use (forward pass) a not-initialized SequentialModel")

		Tensor4<T> *temp = &inputs;
		int i = 1;
		for (auto &current : this->m_layers)
		{
			dnn_debug << "Layer number " << i++ << ": " << current->summary() << "\n";
	//  auto time_good = skepu::benchmark::measureExecTime([&]() {
			temp = current->forward(temp, is_training);
	//	});
	//	dnn_debug << "Time " << time_good.count() / 1E6  << "\n\n";
		}
		this->m_result = temp;
		dnn_debug << "Final result: " << *this->m_result << "\n";
		return *temp;
	}

	std::tuple<float, float> evaluate(Tensor4<T> &test_x, Tensor4<T> &test_y, skepu::Matrix<int> *confusion_matrix = nullptr)
	{
		if (!this->m_initialized)
			this->init();

		// this cause error very earlier, not a even run anything	
		// test_x.flush();	
		// test_y.flush();	
		
		size_t num_valid_batches = test_x.size_i()/this->m_batch_size;
 		size_t remainder_valid_batch = test_x.size_i()%this->m_batch_size;

		if(remainder_valid_batch > 0 ){
  			num_valid_batches = num_valid_batches + 1 ;
 		}
		
		// std::cout << "valid bacth count: " << num_valid_batches <<" remainder: " <<remainder_valid_batch <<"\n";

		
		Tensor4<T> batched_test_x(
				this->m_batch_size,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"Batch eval data X"
		);
		// std::cout << "error 0 \n";	
		Tensor4<T> batched_test_y(this->m_batch_size, 1, 1, test_y.size_l(), "Batch eval data Y");
		// std::cout << batched_test_y.size_i() << batched_test_y.size_j() << batched_test_y.size_k() << batched_test_y.size_l() <<"\n";

		// std::cout << "error 1 \n";
		size_t remainder_test_batch_allocate = remainder_valid_batch;
		if (remainder_test_batch_allocate == 0) remainder_test_batch_allocate = 1;

       	// skepu::Tensor4<float> r_batch_data(remainder_test_batch_allocate, test_x.size_j(),test_x.size_k(), test_x.size_l());
       	// skepu::Tensor4<float> r_batch_label(remainder_test_batch_allocate, test_y.size_j(),test_y.size_k(), test_y.size_l());

		// std::cout << "error 2 \n";	
		skepu::Vector<float> temp_loss(this->m_batch_size, "Loss temp");
		skepu::Vector<int>   temp_acc(this->m_batch_size, "Accuracy temp");

		float total_loss = 0, total_accuracy =0 ;
		// const size_t mini_batch_count = test_x.size_i() / this->m_batch_size;

		// Tensor4<float> *actual_batched_test_x = nullptr;
		// Tensor4<float> *actual_batched_test_y = nullptr;

		// std::cout << "error 3 \n";
		for (size_t mini_batch = 0; mini_batch < num_valid_batches; ++mini_batch) //mini_batch_count
		{
			if(mini_batch == (num_valid_batches - 1)  && remainder_valid_batch > 0 )
     			{
       				// skel_flatmap(r_batch_data, test_x.begin() + mini_batch * r_batch_data.size());
       				// skel_flatmap(r_batch_label, test_y.begin() + mini_batch * r_batch_label.size());
					// std::cout << "error 4 "<< mini_batch <<" \n";
					skel_flatmap(batched_test_x, test_x.begin() + mini_batch * batched_test_x.size() - (this->m_batch_size - remainder_valid_batch) * batched_test_x.size_j() * batched_test_x.size_k() * batched_test_x.size_l()); // remainder + (remainder - batch_size) 
       				skel_flatmap(batched_test_y, test_y.begin() + mini_batch * batched_test_y.size() - (this->m_batch_size - remainder_valid_batch) * batched_test_y.size_j() * batched_test_y.size_k() * batched_test_y.size_l());
					// actual_batched_test_x = &r_batch_data;
					// actual_batched_test_y = &r_batch_label;

       // std::cout << "remainder batch: " << r_batch_data.size_i()<< '\n';
       // display_image(r_batch_data);
     			}
				
     			else{

					// std::cout << "error 5 "<< mini_batch <<"\n";
					// skepu::io::cout << batched_test_y << "\n";

					// std::cout << "size of batched test y: "<<string_tensor_size(batched_test_y)<< "\n";
					// std::cout << "size of  test y: "<<string_tensor_size(test_y) << "\n";

      				skel_flatmap.setLabel("Copy batch eval data X");
					// std::cout << "error 5 a "<< mini_batch << "\n";
					skel_flatmap(batched_test_x, test_x.begin() + mini_batch * batched_test_x.size());
					skel_flatmap.setLabel("Copy batch eval data Y");
					
					// test_y.flush();
					// batched_test_y.flush();
						
					skel_flatmap(batched_test_y, test_y.begin() + mini_batch * batched_test_y.size());
					
					

					// actual_batched_test_x = &batched_test_x;
					// actual_batched_test_y = &batched_test_y;

		       // std::cout << "full batch " <<batch_index <<" size: " <<batch_data.size_i() << '\n';
     			}

			// skel_flatmap.setLabel("Copy batch eval data X");
			// skel_flatmap(batched_test_x, test_x.begin() + mini_batch * batched_test_x.size());
			// skel_flatmap.setLabel("Copy batch eval data Y");
			// skel_flatmap(batched_test_y, test_y.begin() + mini_batch * batched_test_y.size());

			
			auto &predicted = this->forward(batched_test_x);

			// Loss and accuracy
		/*  skel_cross_entropy_batched_loss.setDefaultSize(this->m_batch_size);
			loss += -skel_cross_entropy_batched_loss(batched_test_y, predicted) / this->m_batch_size;
			skel_accuracy_batched.setDefaultSize(this->m_batch_size);
			accuracy += (float)skel_accuracy_batched(batched_test_y, predicted) / this->m_batch_size;*/

			// Map+Reduce variant

			// Tensor4<T> grad = loss_function->compute_gradient(predicted, batched_train_y);
			// Tensor4<T> *gradient = &grad;

			skel_cross_entropy_batched_loss_map.setLabel("Crossentropy A");
			skel_cross_entropy_batched_loss_reduce.setLabel("Crossentropy B");
			skel_accuracy_batched_map.setLabel("Accuracy A");
			skel_accuracy_batched_reduce.setLabel("Accuracy B");
			// skel_cross_entropy_batched_loss_map(temp_loss, batched_test_y, predicted);
			skel_accuracy_batched_map(temp_acc, batched_test_y, predicted);
			// float loss = -skel_cross_entropy_batched_loss_reduce(temp_loss) / this->m_batch_size;

			float loss = loss_function->compute_loss(batched_test_y, predicted, this->m_batch_size);
			float accuracy = (float)skel_accuracy_batched_reduce(temp_acc) / this->m_batch_size;

			total_loss += loss;
			total_accuracy += accuracy;

			// std::cout << "Batch " << mini_batch + 1 << " - Loss: " << loss << " - Accuracy: " << accuracy << std::endl;
			
			// GPU correctness code 
			batched_test_y.flush();
			predicted.flush();

			if (confusion_matrix)
			{
				for (size_t i = 0; i < batched_test_y.size_i(); ++i) { // rows
					int actual_label = 0; //true
					int predicted_label = 0; // predicted
					float max_actual = 0, max_predicted = 0;
					for (size_t j = 0; j < batched_test_y.size_l(); ++j) { // cols
						if (batched_test_y(i,0,0, j) > max_actual)
						{
							max_actual = batched_test_y(i, 0, 0, j);
							actual_label = j;
						}
						if (predicted(i,0,0, j) > max_predicted)
						{
							max_predicted = predicted(i, 0, 0, j);
							predicted_label = j;
						}
					}
          (*confusion_matrix)(actual_label, predicted_label) += 1;
        } // end of j loop
			} // end of if


		}

		return {total_loss / num_valid_batches, total_accuracy /num_valid_batches }; // correct? yes :) mini_batch_count
	}

	void display_image(skepu::Tensor4<float> &train_images){

	  int batch = train_images.size_i();
	  int height = train_images.size_j();
	  int width = train_images.size_k();
	  int channel = train_images.size_l();

	  for(size_t b =0; b < 5; ++b){
	    for(size_t i =0; i < height; ++i){
	      for(size_t j =0; j < width; ++j){
	        for(size_t c =0; c < channel; ++c){

	          if (train_images(b,i,j,c) > 0.75) {
	               std::cout << "#"; // Dense pixel
	           } else if (train_images(b,i,j,c) > 0.5) {
	               std::cout << "+"; // Medium pixel
	           } else if(train_images(b,i,j,c) >0.25){
	             std::cout << "." ;
	           } else {
	               std::cout << " "; // Sparse pixel
	           }
	        }
	      }
	      std::cout << '\n';
	    }
	    std::cout << '\n';
	  }
	}

	// std::string name = "29_Jan_2025_opencl";
	void store_results_per_batch(std::string name, int epoch, int batch_index, double loss, double accuracy){
	  std::ofstream file;

	// Open the file in append mode (std::ios::app)
	file.open("./result_final_1_7_Mar/training_results_per_batch_"+ name +".csv", std::ios::app);

	// If the file is empty, add a header line (optional)
	if (file.tellp() == 0) {
	    file << "Epoch, Batch_Index, Loss,  Accuracy \n";
	}

	// Write the values to the file as a new line
	file << epoch << ","
	     << batch_index << ","
	     << loss << ","
	     << accuracy << "\n";


	file.close();
	}

	void store_results(std::string name, int batch_size, double learning_rate, int e,double average_epoch_loss_t,
   double average_b_accuracy, double average_epoch_loss_v,double average_v_accuracy )
   {
     std::ofstream file;

   // Open the file in append mode (std::ios::app)
   file.open("./result_final_1_7_Mar/training_results_"+ name +".csv", std::ios::app);

   // If the file is empty, add a header line (optional)
   if (file.tellp() == 0) {
       file << "Batch Size,Learning Rate,Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n";
   }

   // Write the values to the file as a new line
   file << batch_size << ","
        << learning_rate << ","
        << e << ","
        << average_epoch_loss_t << ","
        << average_b_accuracy << ","
        << average_epoch_loss_v << ","
        << average_v_accuracy << "\n";

   file.close();
}


	// compiling loss function and optimizer 
	void configure(Loss loss_type, Opt optimizer_type){
		loss_function = std::make_unique<LossFunction<T>>(loss_type);
		optimizer = std::make_unique<Optimizer<T>>(optimizer_type);
		compiled = true;
	}


	// fit -> training 
	void fit(Tensor4<T> &train_x, Tensor4<T> &train_y, int epochs, float learning_rate, float validation_split,  const std::string &name, bool print_progress=true)
	{
		if(!compiled){
			throw std::runtime_error("Model must be compiled before training ");
		}

		if (!this->m_initialized)
			this->init();

		// (Re-)initialize layer trainable parameters
		this->init_parameters();

		size_t valid_size = train_x.size_i() * validation_split;
		size_t train_size = train_x.size_i() - valid_size ;
		// std::cout << "total train_x: " << train_x.size_i() << '\n';
		// std::cout << "valid_size: " <<valid_size <<" train_size: " <<train_size << '\n';

		
		Tensor4<T> train_split_x(
				train_size,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"Split train data X"
		);
		Tensor4<T> train_split_y(train_size, 1, 1, train_y.size_l(), "Split train data Y");

		Tensor4<T> valid_split_x(
				valid_size,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"Split valid data X"
		);
		Tensor4<T> valid_split_y(valid_size, 1, 1, train_y.size_l(), "Split valid data Y");

		skel_flatmap.setLabel("Copy Split training data X");
		skel_flatmap(train_split_x, train_x.begin());
		skel_flatmap.setLabel("Copy Split training data Y");
		skel_flatmap(train_split_y, train_y.begin());

		// std::cout << "train_split_x: " <<train_split_x.size_i()<< " train_split_x " << '\n';
		// display_image(train_split_x);

		skel_flatmap.setLabel("Copy split valid data X");
		skel_flatmap(valid_split_x, train_x.begin() + train_size * train_x.size_j() * train_x.size_k() * train_x.size_l());
		skel_flatmap.setLabel("Copy split valid data Y");
		skel_flatmap(valid_split_y, train_y.begin() + train_size * train_y.size_j() * train_y.size_k() * train_y.size_l());


		size_t num_train_batches = train_split_x.size_i()/this->m_batch_size;
 		size_t remainder_train_batch = train_split_x.size_i()%this->m_batch_size;
		
 		// size_t num_valid_batches = valid_x.size_i()/this->m_batch_size;
 		// size_t remainder_valid_batch = valid_x.size_i()%this->m_batch_size;

		// std::cout << "train batch count: " << num_train_batches <<" remainder: " <<remainder_train_batch <<"\n";
		// std::cout << "valid bacth count: " << num_valid_batches <<" remainder: " <<remainder_valid_batch <<"\n";


		if(remainder_train_batch > 0 ){
  			num_train_batches = num_train_batches + 1 ;
 		}

		
        // GPU backend bug: following is needed to correct software coherency
		train_split_x.flush();
		train_split_y.flush();
		valid_split_x.flush();
		valid_split_y.flush();

		// train_split_y = cut_tensor4(train_split_y, train_split_y.size_i());
		// train_split_x = cut_tensor4(train_split_x, train_split_x.size_i());
		// valid_split_x = cut_tensor4(valid_split_x, valid_split_x.size_i());
		// valid_split_y = cut_tensor4(valid_split_y, valid_split_y.size_i());

		Tensor4<T> batched_train_x(
				this->m_batch_size,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"Batch train data X"
		);
		
		Tensor4<T> batched_train_y(this->m_batch_size, 1, 1, train_y.size_l(), "Batch train data Y");

		size_t remainder_train_batch_allocate = remainder_train_batch;
		if (remainder_train_batch_allocate == 0) remainder_train_batch_allocate = 1;
       	// skepu::Tensor4<float> r_batch_data(remainder_train_batch_allocate, train_split_x.size_j(),train_split_x.size_k(), train_split_x.size_l());
       	// skepu::Tensor4<float> r_batch_label(remainder_train_batch_allocate, train_split_y.size_j(),train_split_y.size_k(), train_split_y.size_l());

	/*	Tensor4<T> r_batch_data(
				remainder_train_batch_allocate,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"r_batch_data train data X"
		);
		Tensor4<T> r_batch_label(remainder_train_batch_allocate, 1, 1, train_y.size_l(), "r_batch_label train data Y");*/

		// skepu::Vector<float> temp_loss(this->m_batch_size, "Loss temp");
		skepu::Vector<int>   temp_acc(this->m_batch_size, "Accuracy temp");

		for (size_t epoch = 0; epoch < epochs; ++epoch) // epoch
		{
			float total_train_loss = 0, total_train_accuracy = 0;
			// std::stringstream ss; ss << "Epoch " << (epoch + 1);
			// tracing::tracer().region("Epoch", __LINE__, [&]()
			// {
			// dnn_debug << "Epoch " << (epoch + 1) << "\n";

			// const size_t mini_batch_count = train_split_x.size_i() / this->m_batch_size;

			// std::cout << "batches: " <<mini_batch_count << '\n';

		//	Tensor4<float> *actual_batched_train_x = nullptr;
		//	Tensor4<float> *actual_batched_train_y = nullptr;
			// std::cout << "Loop execute for " << num_train_batches << " batches \n";

			for (size_t mini_batch = 0; mini_batch < num_train_batches ; ++mini_batch) //  mini_batch_count , mini batches, mini_batch_count
			{

				if(mini_batch == (num_train_batches - 1)  && remainder_train_batch > 0 )
     			{
					// std::cout <<"problem here, offset: " << mini_batch * batched_train_x.size() - (this->m_batch_size - remainder_train_batch) * batched_train_x.size_j() * batched_train_x.size_k() * batched_train_x.size_l()  << "\n";
       				std::cout << "1\n";
                    skel_flatmap(batched_train_x, train_split_x.begin() + mini_batch * batched_train_x.size() - (this->m_batch_size - remainder_train_batch) * batched_train_x.size_j() * batched_train_x.size_k() * batched_train_x.size_l()); // remainder + (remainder - batch_size) 
       				skel_flatmap(batched_train_y, train_split_y.begin() + mini_batch * batched_train_y.size() - (this->m_batch_size - remainder_train_batch) * batched_train_y.size_j() * batched_train_y.size_k() * batched_train_y.size_l());
				//	actual_batched_train_x = &r_batch_data;
				//	actual_batched_train_y = &r_batch_label;
					
       // std::cout << "remainder batch: " << r_batch_data.size_i()<< '\n';
       			//	display_image(r_batch_data);
     			}
     			else{
      			 	skel_flatmap.setLabel("Copy batch training data X");
					skel_flatmap(batched_train_x, train_split_x.begin() + mini_batch * batched_train_x.size());
					skel_flatmap.setLabel("Copy batch training data Y");
					skel_flatmap(batched_train_y, train_split_y.begin() + mini_batch * batched_train_y.size());
				//	actual_batched_train_x = &batched_train_x;
				//	actual_batched_train_y = &batched_train_y;

		       // std::cout << "full batch " <<batch_index <<" size: " <<batch_data.size_i() << '\n';
     			}
     


		// 		/************************ prepare batch - copying **************************/
				// dnn_debug << "\nMini-batch " << (mini_batch + 1) << " / " << mini_batch_count << "\n";
				// skel_flatmap.setLabel("Copy batch training data X");
				// skel_flatmap(batched_train_x, train_split_x.begin() + mini_batch * batched_train_x.size());
				// skel_flatmap.setLabel("Copy batch training data Y");
				// skel_flatmap(batched_train_y, train_split_y.begin() + mini_batch * batched_train_y.size());

				// std::cout << "batched_train_x: " <<batched_train_x.size_i()<< " batched_train_x  " << '\n';
				 // display_image(batched_train_x);

		// 		// skepu::io::cout << batched_train_x << '\n';
		// 		/************************ forward **************************/
		// 		std::stringstream ss; ss << "Epoch " << (epoch + 1) << " batch " << (mini_batch + 1) << " forward";
		// 		tracing::tracer().beginRegion("Forward", __LINE__);
		 			// std::cout << "error here \n";
					auto &predicted = this->forward(batched_train_x, true);
					// std::cout << "error here 1\n";
					
		// 		tracing::tracer().endRegion();
		//

				// std::cout << "predicted : " << predicted << '\n';

				// Tensor4<T> initial_gradient( // move out of loop
				// 		predicted.size_i(), predicted.size_j(),
				// 		predicted.size_k(), predicted.size_l(),
				// 		"Initial gradient"
				// );

				// skel_delta.setLabel("Loss gradient");
				// skel_delta(initial_gradient, predicted, batched_train_y);
				// Tensor4<T> *gradient = &initial_gradient;
				Tensor4<T> grad = loss_function->compute_gradient(predicted, batched_train_y);
				Tensor4<T> *gradient = &grad;
				// dnn_debug << "Initial gradient: " << *gradient << "\n";
		//
		// 		// accuracy
				// float loss = 0, 
				float accuracy = 0;

				// skel_cross_entropy_batched_loss_map.setLabel("Crossentropy A");
				// skel_cross_entropy_batched_loss_reduce.setLabel("Crossentropy B");
				// skel_accuracy_batched_map.setLabel("Accuracy A");
				// skel_accuracy_batched_reduce.setLabel("Accuracy B");

		// 		/// check
				// skel_cross_entropy_batched_loss_map(temp_loss, batched_train_y, predicted);
				// loss += -skel_cross_entropy_batched_loss_reduce(temp_loss) / this->m_batch_size;
				
				skel_accuracy_batched_map(temp_acc, batched_train_y, predicted);
				accuracy += (float)skel_accuracy_batched_reduce(temp_acc) / this->m_batch_size;
				float loss = loss_function->compute_loss(batched_train_y, predicted, this->m_batch_size);

				total_train_loss +=loss;
				total_train_accuracy +=accuracy;

				std::cout <<"Epoch " << epoch <<" Train Batch " << mini_batch + 1 << " - Loss: " << loss << " - Accuracy: " << accuracy << std::endl;

		// 		std::cout << "-- Mini-batch " << std::setw(6) << (mini_batch + 1) << " / " << std::setw(6) << mini_batch_count;
		// 		std::cout << " [";
		// 		const size_t bar_width = 30;
		// 		for (size_t i = 0; i < bar_width; ++i)
		// 		{
		// 			if (mini_batch / (float)mini_batch_count < i / (float)bar_width) std::cout << " ";
		// 			else std::cout << "=";
		// 		}
		// 		std::cout << "]";
		// 		std::cout	<< " - Loss: " << std::setw(10) << loss << ", Accuracy: " << std::setw(10) << accuracy << "\r";
		//
		//
		// 		/************** back ward pass ***************/
		// 		std::stringstream ss2; ss2 << "Epoch " << (epoch + 1) << " batch " << (mini_batch + 1) << " backward";
		// 		tracing::tracer().beginRegion("Backward", __LINE__);
				
				for (auto it = this->m_layers.rbegin(); it != this->m_layers.rend(); ++it)
				{
					auto &current_layer = **it;
					Tensor4<T> *forward_inputs = ((it + 1) != this->m_layers.rend()) ? &(**(it + 1)).m_vals : &batched_train_x;

					if (current_layer.is_trainable()) 
					{
						current_layer.m_learning_rate = learning_rate;
					}

					gradient = current_layer.backward(forward_inputs, gradient, &batched_train_y);

					if(current_layer.is_trainable()){
						optimizer->step(current_layer);
					}

		// 			dnn_debug << "Outgoing gradient: " << *gradient << "\n";
				}
		// 		tracing::tracer().endRegion();
				
                // store_results_per_batch(name, epoch + 1, mini_batch + 1, loss, accuracy);
		 	} // end of mini batch
		//
		// 	if (print_progress)
		// 	{
		// 		std::stringstream ss; ss << "Epoch " << (epoch + 1) << " evaluate";
		// 		tracing::tracer().beginRegion("Epoch evaluate", __LINE__);
				auto score = this->evaluate(valid_split_x, valid_split_y);

		// 		tracing::tracer().endRegion(); // \033[K
		// 		std::cout << "- Epoch " << std::setw(4) << (epoch + 1) << " / " << std::setw(4) << epochs << ":";
		// 		std::cout << " Valid Loss: " << std::setw(10) << std::get<0>(score) << ", Valid Accuracy: " << std::setw(10) << std::get<1>(score) << "\n";
		// 	}
		// }); // tracer epoch end
            std::cout <<"Epoch " << epoch <<" Train Loss: " << total_train_loss/num_train_batches <<  " Train Acc: " << total_train_accuracy/num_train_batches << " Vaildation Loss: " << std::get<0>(score) << " Validation Acc: " << std::get<1>(score) << std::endl;
			
            // store_results(name , this->m_batch_size, learning_rate, epoch + 1, total_train_loss/num_train_batches, total_train_accuracy/num_train_batches, std::get<0>(score), std::get<1>(score) );
		 } // end of epoch loop


	} // end of fit function

	std::string summary()
	{
		std::stringstream os;
		for (auto &layer : this->m_layers)
		{
			os << layer->summary() << "\n";
		}
		return os.str();
	}

	Layer<T> &getLayer(size_t i) { return *this->m_layers[i]; }

protected:
	bool m_initialized = false;

	std::vector<std::shared_ptr<Layer<T>>> m_layers{};
	size_t m_layers_count = 0;
	skepu::PRNG m_prng{0}; // default seed

	Dimensions m_input_size;
	size_t m_batch_size;

	Tensor4<T> *m_result = nullptr;

	std::unique_ptr<LossFunction<T>> loss_function; 
	std::unique_ptr<Optimizer<T>> optimizer;
	bool compiled = false;
};

template <typename T, typename ConcreteLayer>
SequentialModel<T> &&operator<<(SequentialModel<T> &&model, ConcreteLayer &&layer)
{
	model.add(std::forward<ConcreteLayer>(layer));
	return std::move(model);
}

template <typename T, typename ConcreteLayer>
SequentialModel<T> &operator<<(SequentialModel<T> &model, ConcreteLayer &&layer)
{
	model.add(std::forward<ConcreteLayer>(layer));
	return model;
}
} // namespace impl

template <typename T = float, typename... Args>
impl::Dense<T> Dense(Args &&...args)
{
	return impl::Dense<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::Conv1D<T> Conv1D(Args &&...args)
{
	return impl::Conv1D<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::Conv2D<T> Conv2D(Args &&...args)
{
	return impl::Conv2D<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::MaxPooling2D<T> MaxPooling2D(Args &&...args)
{
	return impl::MaxPooling2D<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::Flatten<T> Flatten(Args &&...args)
{
	return impl::Flatten<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::Dropout<T> Dropout(Args &&...args)
{
	return impl::Dropout<T>(std::forward<Args>(args)...);
}

template <typename T = float, typename... Args>
impl::SequentialModel<T> SequentialModel(Args &&...args)
{
	return impl::SequentialModel<T>(std::forward<Args>(args)...);
}







} // namespace ml

} // namespace skepu

#include "skepu-mnist.hpp"
//#include "skepu-cifar10.hpp"

float onehot_mapper_uf(skepu::Index4D idx, skepu::Vec<int> labels)
{
	int my_item = idx.i;
	int my_class = idx.l;
	int my_label = labels(my_item);
	return (my_label == my_class) ? 1 : 0;
}

skepu::Tensor4<float>
mnist_labels_to_onehot(skepu::Vector<mnist::label_t> &labels, size_t num_classes)
{
	skepu::Tensor4<float> result(labels.size(), 1, 1, num_classes, "MNIST-labels-onehot");
	auto onehot_mapper = skepu::Map<0>(onehot_mapper_uf);
	onehot_mapper.setLabel("MNIST-to-onehot");
	onehot_mapper(result, labels);
	return result;
}





float create_data_x_uf(skepu::Random<1> &rand, int c)
{
	float val = (int)(rand.getNormalized() * c);
	return val;
}

float create_data_y_uf(skepu::Index4D index, skepu::Ten4<float> x)
{
	float val = 0;
	if (index.l == x(index.i, 0, 0, 0))
		val = 1;
	return val;
}

/*void artifical_dataset(size_t batch_size, float learning_rate, size_t epochs)
{
	const size_t train_items = 4000, test_items = 1000;
	const size_t num_classes = 5;
	auto skel_generate_x = skepu::Map<0>(create_data_x_uf);
	auto skel_generate_y = skepu::Map<0>(create_data_y_uf);

	skepu::PRNG prng(0);
	skel_generate_x.setPRNG(prng);
	skel_generate_y.setPRNG(prng);

	skepu::Tensor4<float> train_x(train_items, 8, 8, 1);
	skepu::Tensor4<float> train_y(train_items, 1, 1, num_classes);
	skel_generate_x(train_x, num_classes);
	skel_generate_y(train_y, train_x);

	skepu::Tensor4<float> test_x(test_items, 8, 8, 1);
	skepu::Tensor4<float> test_y(test_items, 1, 1, num_classes);
	skel_generate_x(test_x, num_classes);
	skel_generate_y(test_y, test_x);

//  std::cout << "Train X: " << train_x << "\n";
//  std::cout << "Train Y: " << train_y << "\n";

//	std::cout << "Test X: " << test_x << "\n";
//  std::cout << "Test Y: " << test_y << "\n";

	using namespace skepu::ml;
	// Multi-layer perceptron
	auto model = SequentialModel(Dimensions{batch_size, 8, 8, 1})
		<< Conv2D(2).kernel(3, 3)
		//	.pretrained("./data/pretrained/artificial_cnn_params_layer_0.csv")
			.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(4).kernel(3, 3)
		//	.pretrained("./data/pretrained/artificial_cnn_params_layer_0.csv")
			.activation(Act::ReLU)
		<< Flatten()
		<< Dropout(0.5)
		<< Dense(10).weights(Init::Uniform).biases(Init::Zero)
		//	.pretrained("./data/pretrained/artificial_cnn_params_layer_3.csv")
			.activation(Act::ReLU)
		<< Dense(num_classes).weights(Init::Uniform).biases(Init::Zero)
		//	.pretrained("./data/pretrained/artificial_cnn_params_layer_4.csv")
			.activation(Act::SoftMax);

	model.init();
	model.init_parameters();
	std::cout << model.summary();

	// Training
	model.fit(train_x, train_y, epochs, learning_rate, 0.1);

	// Testing
	// auto score = model.evaluate(test_x, test_y);
	// std::cout << "Test loss: " << std::get<0>(score) << "\nTest accuracy: " << std::get<1>(score) << "\n";
}
*/

float gray_scale_kernel(skepu::Index4D index, 
skepu::Ten4<float> in)
{
    float res = 0;
    float pr = in(index.i,index.j,index.k,0);
    float pg = in(index.i,index.j,index.k,1);
    float pb = in(index.i,index.j,index.k,2);

    return 0.2989f * pr + 0.5870f * pg + 0.1140f * pb;
    

}


float gray_scale_kernel_pool(skepu::Pool4D<float> pool)
{
    float pr = pool(0,0,0,0);
    float pg = pool(0,0,0,1);
    float pb = pool(0,0,0,2);
    return 0.2989f * pr + 0.5870f * pg + 0.1140f * pb;
}

void convertCHWtoHWC(skepu::Tensor4<float> &images, skepu::Tensor4<float> &conv_images ){
    skepu::external(skepu::read(images), [&]{
        for(size_t batch = 0; batch < images.size_i(); batch++){
            for(size_t channel = 0; channel < images.size_j(); channel++){
                for(size_t row = 0; row < images.size_k(); row++){
                    for(size_t col = 0; col < images.size_l(); col++){
                        conv_images(batch,row,col,channel)= images(batch, channel,row, col);
                    }
                }
            }
        }
    }, skepu::write(conv_images));
}


std::tuple<skepu::Tensor4<float>, skepu::Tensor4<float>> 
binary_class(skepu::Tensor4<float> &train_images,skepu::Vector<mnist::label_t> &train_labels_temp, int num_classes, int class_a, int class_b){
	
	// count sample each class 
	skepu::Vector<int> count_vector(num_classes);
	for(int i = 0; i < train_labels_temp.size_i(); i++){
        count_vector(train_labels_temp(i))++;
    }

	// allocate memory
	int num_images_label = count_vector(class_a) + count_vector(class_b);
	// std::cout << "count :" << num_images_label << "\n";
    
    skepu::Tensor4<float> filtered_images(num_images_label,train_images.size_j(),train_images.size_k(), train_images.size_l());
    // skepu::Vector<int> filtered_labels(num_images_label);
	skepu::Tensor4<float> filtered_labels(num_images_label, 1,1,1);

	// classify classes 
	int index = 0;
    for(int i = 0; i < train_labels_temp.size_i(); i ++){
        if(train_labels_temp(i) == class_a || train_labels_temp(i) == class_b){
            for(int h = 0; h < train_images.size_j(); h++){
                for(int w = 0; w < train_images.size_k(); w++){
                    for(int c  = 0; c < train_images.size_l(); c++){
                        filtered_images(index,h,w,c) = train_images(i,h,w,c);
                    }
                }
            }
            filtered_labels(index,0,0,0) = (train_labels_temp(i) == 0)?0:1;
            
            index++;
        }   
    }

	// std::cout << "done filtering \n";

	// convert filter vector to 4D
	// skepu::Tensor4<int> train_y(batches,1,1, num_classes);
	// for(size_t b = 0; b < train_y.size_i(); b++){

    //     train_y(b,0,0,0) = filtered_labels(b);
    // }

	return {filtered_images, filtered_labels};
}


// #include <skepu-lib/ml.hpp>
void csv_results (int batch_size, std::string backend, int run, double test_loss, double test_acc, double training_time, double testing_time,std::string name ){ //, double preprocessing
	
	std::ofstream file;

	file.open("./result_final_1_7_Mar/"+name+"results_all.csv", std::ios::app);

	// If the file is empty, add a header line (optional)
	if (file.tellp() == 0) {
	    file << "batch, backend, Run, Test_loss, Test_acc, Traning_time, testing_time \n";
	}

	// Write the values to the file as a new line
	file << batch_size << ","
	     << backend << ","
	     << run << ","
         << test_loss << ","
         << test_acc << ","
	     << training_time<< ","
		 << testing_time<< "\n";
        //  "," 
        //  << preprocessing <<"\n";


	file.close();
}


int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " batch_size learning_rate epochs backend run \n";
		exit(1);
	}

	const size_t batch_size = atoi(argv[1]);
	const float learning_rate = atof(argv[2]);
	const size_t epochs = atof(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	spec.setCPUThreads(8);
	skepu::setGlobalBackendSpec(spec);
	auto run = atoi(argv[5]);



	// Dataset configuration
	const std::string param_path = "./data/pretrained/";

	const size_t num_classes = 10;

	using namespace skepu::ml;
	
	

	// Keras MNIST convolutional model -> simple example 
	auto modela = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Conv2D(32).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_0.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(64).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_2.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2) 
		<< Flatten() 
		<< Dropout(0.5)
		<< Dense(num_classes).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_6.csv")
				.activation(Act::SoftMax);


	// Multi-layer perceptron
	auto modelc = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Flatten() << Dropout(0.5)
		<< Dense(100).weights(Init::Uniform).biases(Init::Zero)
		.pretrained(param_path + "fc_hidden_params_layer_2.csv")
		.activation(Act::ReLU)
		<< Dense(num_classes).weights(Init::Uniform).biases(Init::Zero)
		.pretrained(param_path + "fc_hidden_params_layer_3.csv")
		.activation(Act::SoftMax);

	// Simpler conv network
	auto modeld = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Conv2D(1).kernel(3, 3)
				.pretrained(param_path + "simple_params_layer_0.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Flatten() //<< Dropout(0.5)
		<< Dense(num_classes)
				.pretrained(param_path + "simple_params_layer_3.csv")
				.activation(Act::SoftMax);

	// Simpler conv network
	auto modele = SequentialModel(Dimensions{batch_size, 28, 28, 1})
	//	<< Conv2D(1).kernel(3, 3)
	//			.activation(Act::ReLU)
	//	<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(12).kernel(3, 3)
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Flatten() << Dropout(0.5)
		<< Dense(num_classes)
				.activation(Act::SoftMax);

	// TF simple CIFAR dataset model 
	auto modelf = SequentialModel(Dimensions{batch_size, 32, 32, 3})
		<< Conv2D(32).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_0.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(64).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_2.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2) 
		<< Conv2D(64).kernel(3,3).weights(Init::Xavier_uniform).biases(Init::Zero)
		<< Flatten() //<< Dropout(0.5)
		<< Dense(64).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::ReLU)
		<< Dense(num_classes).weights(Init::Xavier_uniform).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_6.csv")
				.activation(Act::SoftMax);

    // Keras MNIST dataset model for binary
	auto model = SequentialModel(Dimensions{batch_size, 28,28,1})
    << Conv2D(32).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
            // .pretrained(param_path + "params_layer_0.csv")
            .activation(Act::ReLU)
    << MaxPooling2D().kernel(2, 2)
    << Conv2D(64).kernel(3, 3).weights(Init::Xavier_uniform).biases(Init::Zero)
            // .pretrained(param_path + "params_layer_2.csv")
            .activation(Act::ReLU)
    << MaxPooling2D().kernel(2, 2) 
    << Conv2D(64).kernel(3,3).weights(Init::Xavier_uniform).biases(Init::Zero)
    << Flatten() //<< Dropout(0.5)
    << Dense(128).weights(Init::Xavier_uniform).biases(Init::Zero).activation(Act::ReLU)
    << Dense(1).weights(Init::Xavier_uniform).biases(Init::Zero)
            // .pretrained(param_path + "params_layer_6.csv")
            .activation(Act::Sigmoid);            
	
	
	// Initialize model
	model.init(); // memory and such
	std::cout << model.summary();

    /*********************** MINST loading ************************** */ 

    const std::string data_path = "./data/mnist/";

    

    skepu::Tensor4<float> train_images_ = mnist::parse_mnist_images(data_path + "train-images.idx3-ubyte", 0, 1.0);
    skepu::Tensor4<float> test_images_  = mnist::parse_mnist_images(data_path + "t10k-images.idx3-ubyte", 0, 1.0);
    skepu::Vector<mnist::label_t> train_labels_temp = mnist::parse_mnist_labels(data_path + "train-labels.idx1-ubyte");
    skepu::Vector<mnist::label_t> test_labels_temp  = mnist::parse_mnist_labels(data_path + "t10k-labels.idx1-ubyte");

    // multi classification
    // skepu::Tensor4<float> train_labels = mnist_labels_to_onehot(train_labels_temp, num_classes);
    // skepu::Tensor4<float> test_labels  = mnist_labels_to_onehot(test_labels_temp,  num_classes);
    
    
// binary classification 
	int class_a = 0, class_b = 1;
	std::tuple<skepu::Tensor4<float>, skepu::Tensor4<float>> train_data = binary_class(train_images_,train_labels_temp, num_classes, class_a, class_b);
	std::tuple<skepu::Tensor4<float>, skepu::Tensor4<float>> test_data = binary_class(test_images_,test_labels_temp, num_classes, class_a, class_b);

	skepu::Tensor4<float> train_images = std::get<0>(train_data);
	skepu::Tensor4<float> train_labels = std::get<1>(train_data);

    skepu::Tensor4<float> test_images = std::get<0>(test_data);
	skepu::Tensor4<float> test_labels = std::get<1>(test_data);

	std::cout << "train images dimensions: "<<train_images.size_i() << train_images.size_j() << train_images.size_k() << train_images.size_l() << "\n";

	// model.display_image(train_images);
	// for(int i = 0; i < 5; i++){ //train_labels.size_i()
    //     std::cout << train_labels(i,0,0,0) << "\n";
    // }


	/*********************** CIFAR loading ************************** */ 
	/*const std::string data_path = "/Users/sehqu50/Documents/PHD/Dataset/cifar-10-batches-bin";	
	std::tuple<
   skepu::Vector<int>*, skepu::Tensor4<float>*> train_data = cifar10::load_CIFAR10_Train_Dataset(data_path);

    std::tuple<
   skepu::Vector<int>*, skepu::Tensor4<float>*> test_data = cifar10::load_CIFAR10_Test_Dataset(data_path);
    // std::cout << "label of the first image: " << cifar_labels(0);
    // std::cout << "Pixel value at (0,0,red) of the first image: " <<cifar_images(0,0,0,0);
    
    skepu::Vector<int> *train_labels_temp = std::get<0>(train_data);
    skepu::Tensor4<float> *train_Images_temp = std::get<1>(train_data);

    skepu::Vector<int> *test_labels_temp = std::get<0>(test_data);
    skepu::Tensor4<float> *test_Images_temp = std::get<1>(test_data);

	skepu::Tensor4<float> train_labels = mnist_labels_to_onehot(*train_labels_temp, num_classes);
	skepu::Tensor4<float> test_labels  = mnist_labels_to_onehot(*test_labels_temp,  num_classes);
    
	// std::cout << "Train_Images_org diemnsions: " <<train_Images_temp->size_i() << train_Images_temp->size_j() << train_Images_temp->size_k() << train_Images_temp->size_l() << "\n";
	// std::cout << "Train_Images_org diemnsions: " <<test_Images_temp->size_i() << test_Images_temp->size_j() << test_Images_temp->size_k() << test_Images_temp->size_l() << "\n";
*/
	
    /*********************** Convert cifar data set to NHWC format  ************************** */ 
	/*
    skepu::Tensor4<float> c_train_images(train_Images_temp->size_i(),train_Images_temp->size_k(), train_Images_temp->size_l(),train_Images_temp->size_j());
	skepu::Tensor4<float> c_test_images(test_Images_temp->size_i(),test_Images_temp->size_k(), test_Images_temp->size_l(),test_Images_temp->size_j());
	// std::cout << "Train_Images_converted diemnsions: " <<train_images.size_i() << train_images.size_j() << train_images.size_k() << train_images.size_l() << "\n";
	// std::cout << "Test_Images_converted diemnsions: " <<test_images.size_i() << test_images.size_j() << test_images.size_k() << test_images.size_l() << "\n";
	// std::cout << "probllem here \n";

	convertCHWtoHWC(*train_Images_temp,c_train_images);
	convertCHWtoHWC(*test_Images_temp, c_test_images);
*/
    /*********************** Preprocessing cifar data set RGB to Gray scale  ************************** */ 
    /*
    skepu::Tensor4<float> train_images(c_train_images.size_i(),c_train_images.size_j(), c_train_images.size_k(),  1 );
    skepu::Tensor4<float> test_images(c_test_images.size_i(),c_test_images.size_j(), c_test_images.size_k(),  1 );

    // std::cout << "Train_Images_converted diemnsions: " <<c_train_images.size_i() << c_train_images.size_j() << c_train_images.size_k() << c_train_images.size_l() << "\n";
	// std::cout << "Test_Images_converted diemnsions: " <<c_test_images.size_i() << c_test_images.size_j() << c_test_images.size_k() << c_test_images.size_l() << "\n";
    auto convert_pool = skepu::MapPool(gray_scale_kernel_pool);
    convert_pool.setPoolSize(1,1,1,3);

    
		// auto conv = skepu::MapOverlap(gray_scale_kernel);
		// conv.setOverlap(0,radius, radius,0);
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			convert_pool(train_images, c_train_images);//, imageInfo.elementsPerPixel);
            convert_pool(test_images, c_test_images);
		});
		// WritePngFileMatrix(outputMatrix, outputFile + "-average.png", colorType, imageInfo);
		std::cout << "Time for pre processing : " << (timeTaken.count() / 10E6) << "\n";


*/
    /*********************** Correction for GPU   ************************** */     
    // GPU software coherency bug: the following lines are reqired as workaround
	train_images.flush();
	train_labels.flush();
	test_images.flush();
	test_labels.flush();


	train_labels = cut_tensor4(train_labels, train_labels.size_i());
	train_images = cut_tensor4(train_images, train_images.size_i());	
	test_labels = cut_tensor4(test_labels, test_labels.size_i());
	test_images = cut_tensor4(test_images, test_images.size_i());


	/*********************** name of the file   ************************** */     	
	std::string name = "mnist_bce_"+std::to_string(batch_size) + argv[4];
	 
	/*********************** Training   ************************** */     
	
	model.configure(Loss::BinaryCrossEntropy, Opt::SGD);

	auto trainingtimeTaken = skepu::benchmark::measureExecTime([&]
	{
	    model.fit(train_images, train_labels, epochs, learning_rate, 0.1 , name,  true); //true is logging ->progress bar
    });
	
	/*********************** Testing   ************************** */ 
    double test_loss = 0 , test_acc = 0 ;    
	skepu::Matrix<int> confusion_matrix(num_classes,num_classes, 0);
	auto testingtimeTaken = skepu::benchmark::measureExecTime([&]
	{   
		auto score = model.evaluate(test_images, test_labels, &confusion_matrix);
		std::cout << "Test loss: " << std::get<0>(score) << " Test accuracy: " << std::get<1>(score) << "\n";
        // test_loss = std::get<0>(score);
        // test_acc = std::get<1>(score);
    });   

    // std::cout << "TRAIN_TIME: : " << (trainingtimeTaken.count() / 10E6) << "\n"; 
	// std::cout << "TEST_TIME: " << (testingtimeTaken.count() / 10E6) << "\n";
	
	// std::cout << "TRAIN_TIME: " << train_time << " sec" << std::endl;
    // std::cout << "TEST_TIME: " << test_time << " sec" << std::endl;

	//confusion matrix
	// skepu::io::cout << confusion_matrix << "\n";

	/*********************** Writing to the csv file   ************************** */     
	// csv_results (batch_size, argv[4], run, test_loss, test_acc, (trainingtimeTaken.count() / 10E6), (testingtimeTaken.count() / 10E6), (timeTaken.count() / 10E6), "cifar_gray_" );
    // csv_results (batch_size, argv[4], run, test_loss, test_acc, (trainingtimeTaken.count() / 10E6), (testingtimeTaken.count() / 10E6), "mnist_loop_10" );
	
    /*********************** Serialize weights to file   ************************** */     
	// model.serialize("skepu_dnn_trained_model_weights"); // generate parameter csv files
	return 0;
}
