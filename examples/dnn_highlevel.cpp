#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#define SKEPU_ENABLE_EXCEPTIONS
#define SKEPU_DNN_NO_PRETRAINED_PARAMS "NO PRETRAINED PARAMS"
#define SKEPU_DNN_USE_MASKED_POOL 1
#define DNN_CONST /*const&*/
//#define DNN_DEBUG 1

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

// bacCKWRaRD PASS
float relu_prime_uf(float x, float g)
{
	return (x > 0) ? g : 0 ;
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

size_t eval_uf(size_t a, size_t b) { return (a == b) ? 1 : 0; }

template <typename T> T dropout_uf(skepu::Random<1> &rand, T el, float rate)
{
	float p = rand.getNormalized();
	return (p > rate) ? (el / (1.0 - rate)) : 0;
}

template <typename T> skepu::multiple<T, char> dropout_masked_uf(skepu::Random<1> &rand, T el, float rate)
{
	float p = rand.getNormalized();
	char enabled = (p > rate);
	T res = enabled ? (el / (1.0 - rate)) : 0;
	return skepu::ret(res, enabled);
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

// Forward
auto skel_flatmap = skepu::Map(flatmap_uf);
auto skel_dense = skepu::Map(fully_connected_batched_uf);

auto skel_act_sigmoid = skepu::Map(sigmoid_uf);
auto skel_act_relu = skepu::Map(relu_uf);
//	auto skel_act_softmax_1 = skepu::MapReduce(softmax_uf_1_map, softmax_uf_1_reduce);
auto skel_act_softmax_2 = skepu::Map<1>(softmax_uf_2);

auto skel_softmax_batched_1 = skepu::MapPool(softmax_batched_uf_1);
auto skel_softmax_batched_2 = skepu::Map(softmax_batched_uf_2);

auto skel_act_tanh = skepu::Map(tanh_uf);
auto skel_dropout = skepu::Map<1>(dropout_uf<float>);
auto skel_dropout_masked = skepu::Map<1>(dropout_masked_uf<float>);
auto skel_dropout_backprop_masked = skepu::Map<2>(dropout_masked_backprop_uf<float>);

auto skel_conv_1d = skepu::MapOverlap(convolutional_1d_uf);
auto skel_conv_2d = skepu::MapOverlap(convolutional_2d_uf);

auto skel_pool_max = skepu::MapPool(max_pooling_uf);
auto skel_pool_max_masked = skepu::MapPool(max_pooling_masked_uf);

auto skel_cross_entropy_batched_loss = skepu::MapReduce<0>(cross_entropy_batched_uf, sum_uf<float>);
auto skel_accuracy_batched = skepu::MapReduce<0>(accuracy_batched_uf, sum_uf<int>);

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

	std::shared_ptr<skepu::PRNG> m_prng;

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
	//	this->m_prng = std::move(source.m_prng);
		
		dnn_debug << "Dense move, Address of PRNG: " << this->m_prng.get() << "\n";
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
		//	dnn_debug << "Dense loaded biases: " << this->m_biases << "\n";
		//	dnn_debug << "Dense loaded weights: " << this->m_weights << "\n";
		}
	}

	void init_parameters() override
	{
		dnn_debug << "Address of PRNG: " << this->m_prng.get() << "\n";
	//	skepu::PRNG prng{0}; // default seed
	//	this->m_prng = &prng; // TODO: fix model prng
		
		// Weights
		if (this->m_init_weights == Init::Xavier)
		{
		//	dnn_debug << "Dense initialize weights to random uniform\n";

			double limit = sqrt(6.0 / (this->m_weights.size_i() + this->m_out));

			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if (this->m_init_weights == Init::Uniform)
		{
		//	dnn_debug << "Dense initialize weights to random uniform\n";

			double limit = 1.0;
			skel_init_random_uniform.setLabel(this->label() + " Random-init dense weights");
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if (this->m_init_weights == Init::Zero)
		{
		//	dnn_debug << "Dense initialize biases to zero\n";
			skel_init_zero.setLabel(this->label() + " Zero-init dense weights");
			skel_init_zero(this->m_weights);
		}

		// Biases
		if (this->m_init_biases == Init::Uniform)
		{
		//	dnn_debug << "Dense initialize weights to random uniform\n";
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init dense biases");
			skel_init_random_uniform(this->m_biases, -1.0, 1.0);
		}
		else if (this->m_init_biases == Init::Zero) {
		//	dnn_debug << "Dense initialize biases to zero\n";
			skel_init_zero.setLabel(this->label() + " Zero-init dense biases");
			skel_init_zero(this->m_biases);
		}

	//	dnn_debug << "Dense initialized biases: " << this->m_biases << "\n";
	//	dnn_debug << "Dense initialized weights: " << this->m_weights << "\n";
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
		//dnn_debug << "Dense input: " << string_tensor_size(*inputs) << "\n";
		//dnn_debug << "Dense vals: " << string_tensor_size(this->m_vals) << "\n";
		skel_dense.setLabel(this->label() + " Dense forward");
		skel_dense(this->m_vals, this->m_biases, this->m_weights, *inputs);
	//	dnn_debug << "Dense inputs: " << *inputs << "\n";
	//	dnn_debug << "Dense vals: " << this->m_vals << "\n";

		return &this->m_vals;
	}

	// inputs: tensor in shape (batch, 1, 1, N)
	Tensor4<T> *backward(Tensor4<T> *inputs, Tensor4<T> *gradient, Tensor4<T> *y) override
	{
	/*	dnn_debug << "Dense backprop\n";
		dnn_debug << "Dense vals: " << string_tensor_size(this->m_vals) << "\n";
		dnn_debug << "Dense input: " << string_tensor_size(*inputs) << "\n";
		dnn_debug << "Dense gradient: " << string_tensor_size(*gradient) << "\n";
		dnn_debug << "Dense temp: " << string_tensor_size(this->m_temp) << "\n";
		dnn_debug << "Dense weights: " << this->m_weights.total_rows() << " x " << this->m_weights.total_cols() << "\n";*/

		skel_propagate_loss_gradient.setLabel(this->label() + " Dense backprop-grad");
		skel_propagate_dense_weight_gradient.setLabel(this->label() + " Dense backprop weights");
		skel_propagate_dense_bias_gradient.setLabel(this->label() + " Dense backprop biases");

		if (this->m_layer_level != Level::Input)
			skel_propagate_loss_gradient(this->m_temp, *gradient, this->m_weights);

		skel_propagate_dense_weight_gradient(this->m_dW, *inputs, *gradient);
		skel_propagate_dense_bias_gradient(this->m_db, *gradient);

		skel_update_params.setLabel(this->label() + " Dense update weights");
		skel_update_params(this->m_weights, this->m_weights, this->m_dW, this->m_learning_rate);
		skel_update_params.setLabel(this->label() + " Dense update biases");
		skel_update_params(this->m_biases, this->m_biases, this->m_db, this->m_learning_rate);

	//	dnn_debug << "Dense new biases: " << this->m_biases << "\n";
	//	dnn_debug << "Dense new weights: " << this->m_weights << "\n";

		return (this->m_layer_level != Level::Input) ? &this->m_temp : nullptr;
	}

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
		//	dnn_debug << "Softmax input: " << *inputs << "\n";
			// std::cout << "Softmax input: " << *inputs << "\n";

			skel_softmax_batched_1.setPoolSize(1, 1, 1, inputs->size_l());
			skel_softmax_batched_1.setStride(1, 1, 1, inputs->size_l());
			skel_softmax_batched_1.setLabel(this->label() + " Softmax A");
			skel_softmax_batched_2.setLabel(this->label() + " Softmax B");
			skel_softmax_batched_1(this->m_temp_vals, *inputs);
			skel_softmax_batched_2(*target, *inputs, this->m_temp_vals);
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

			for(size_t i = 0; i < this->m_weights.size_i(); i++)
				for(size_t j = 0; j < this->m_weights.size_j(); j++)
					for(size_t k = 0; k < this->m_weights.size_k(); k++)
						for(size_t l = 0; l < this->m_weights.size_l(); l++)
							ofs << this->m_weights(i,j,k,l) << "\n";

			for(size_t i = 0; i < this->m_biases.size_i(); i++)
				ofs << this->m_biases(i) << "\n";

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
		if (this->m_init_weights == Init::Xavier)
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

		skel_update_params.setLabel(this->label() + " Conv2D update weights");
		skel_update_params(this->m_weights, this->m_weights, this->m_dW, this->m_learning_rate);
		skel_update_params.setLabel(this->label() + " Conv2D update biases");
		skel_update_params(this->m_biases,  this->m_biases,  this->m_db, this->m_learning_rate);

		dnn_debug << "Conv2D dW: " << this->m_dW << "\n";
		dnn_debug << "Conv2D db: " << this->m_db << "\n";
		dnn_debug << "Conv2D new biases: " << this->m_biases << "\n";
		dnn_debug << "Conv2D new weights: " << this->m_weights << "\n";

		return (this->m_layer_level != Level::Input) ? &this->m_temp : nullptr;
	}

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
	/*	dnn_debug << "Flatten backprop\n";
		dnn_debug << "Flatten vals size: " << string_tensor_size(this->m_vals) << "\n";
		dnn_debug << "Flatten input size: " << string_tensor_size(*inputs) << "\n";
		dnn_debug << "Flatten gradient size: " << string_tensor_size(*gradient) << "\n";
		dnn_debug << "Flatten temp size: " << string_tensor_size(this->m_temp) << "\n";*/

		skel_flatmap.setLabel(this->label() + " Flatten backward");
		skel_flatmap(this->m_temp, *gradient);

	/*	dnn_debug << "Flatten inputs: " << *inputs << "\n";
		dnn_debug << "Flatten gradient: " << *gradient << "\n";
		dnn_debug << "Flatten temp: " << this->m_temp << "\n";*/

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
class SequentialModel
{
public:
	SequentialModel(Dimensions arg_input_size)
	: m_input_size{arg_input_size}, m_batch_size{std::get<0>(arg_input_size)} {
		
		dnn_debug << "SequentialModel: Address of MAIN PRNG: " << this->m_prng.get() << "\n";
	}

	SequentialModel(size_t arg_input_size, size_t arg_batch_size)
	: m_input_size{arg_input_size, 1, 1, 1}, m_batch_size{arg_batch_size} {
		
		dnn_debug << "SequentialModel: Address of MAIN PRNG: " << this->m_prng.get() << "\n";
	}
/*
	// move constructor
	SequentialModel(std::vector<std::shared_ptr<Layer<T>>> &&arg_layers)
	: m_layers{arg_layers}
	{
		this->init();
		dnn_debug << "SequentialModel: Address of MAIN PRNG: " << this->m_prng.get() << "\n";
	}*/

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
		l.m_prng = this->m_prng;
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
	//	dnn_debug << "Final result: " << *this->m_result << "\n";
		return *temp;
	}

	std::tuple<float, float> evaluate(Tensor4<T> &test_x, Tensor4<T> &test_y, skepu::Matrix<int> *confusion_matrix = nullptr)
	{
		if (!this->m_initialized)
			this->init();
		
		tracing::tracer().beginRegion("Evaluate", __LINE__);

		Tensor4<T> batched_test_x(
				this->m_batch_size,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"Batch eval data X"
		);
		Tensor4<T> batched_test_y(this->m_batch_size, 1, 1, test_y.size_l(), "Batch eval data Y");

		float total_loss = 0, total_accuracy =0 ;
		const size_t mini_batch_count = test_x.size_i() / this->m_batch_size;
		for (size_t mini_batch = 0; mini_batch < mini_batch_count; ++mini_batch) //mini_batch_count
		{
			skel_flatmap.setLabel("Copy batch eval data X");
			skel_flatmap(batched_test_x, test_x.begin() + mini_batch * batched_test_x.size());
			skel_flatmap.setLabel("Copy batch eval data Y");
			skel_flatmap(batched_test_y, test_y.begin() + mini_batch * batched_test_y.size());

			auto &predicted = this->forward(batched_test_x);

			// Loss and accuracy
			skel_cross_entropy_batched_loss.setLabel("Crossentropy");
			skel_accuracy_batched.setLabel("Accuracy");
			skel_cross_entropy_batched_loss.setDefaultSize(this->m_batch_size);
			skel_accuracy_batched.setDefaultSize(this->m_batch_size);
			float loss = -skel_cross_entropy_batched_loss(batched_test_y, predicted) / this->m_batch_size;
			float accuracy = (float)skel_accuracy_batched(batched_test_y, predicted) / this->m_batch_size;

			total_loss += loss;
			total_accuracy += accuracy;

			std::cout << "Valid Batch " << mini_batch + 1 << " - Loss: " << loss << " - Accuracy: " << accuracy << std::endl;
			
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
		
		tracing::tracer().endRegion();
		return {total_loss / mini_batch_count, total_accuracy / mini_batch_count}; // correct?
	}

	void display_image(skepu::Tensor4<float> &train_images)
	{
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



	void fit(Tensor4<T> &train_x, Tensor4<T> &train_y, int epochs, float learning_rate, float validation_split,  bool print_progress=true)
	{
		if (!this->m_initialized)
			this->init();

		// (Re-)initialize layer trainable parameters
		this->init_parameters();

		size_t valid_size = train_x.size_i() * validation_split;
		size_t train_size = train_x.size_i() - valid_size ;
		dnn_debug << "total train_x size_i: " << train_x.size_i() << '\n';
		dnn_debug << "valid_size: " << valid_size << " train_size: " << train_size << '\n';

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

		skel_flatmap.setLabel("Copy split valid data X");
		skel_flatmap(valid_split_x, train_x.begin() + train_size * train_x.size_j() * train_x.size_k() * train_x.size_l());
		skel_flatmap.setLabel("Copy split valid data Y");
		skel_flatmap(valid_split_y, train_y.begin() + train_size * train_y.size_j() * train_y.size_k() * train_y.size_l());

		// GPU backend bug: following is needed to correct software coherency
		train_split_x.flush();
		train_split_y.flush();
		valid_split_x.flush();
		valid_split_y.flush();

		Tensor4<T> batched_train_x(
				this->m_batch_size,
				std::get<1>(this->m_input_size),
				std::get<2>(this->m_input_size),
				std::get<3>(this->m_input_size),
				"Batch train data X"
		);
		Tensor4<T> batched_train_y(this->m_batch_size, 1, 1, train_y.size_l(), "Batch train data Y");

		for (size_t epoch = 0; epoch < epochs; ++epoch) // epoch
		{
			float total_train_loss = 0, total_train_accuracy = 0;
			std::stringstream epoch_ss; epoch_ss << "Epoch " << (epoch + 1);
			tracing::tracer().region(epoch_ss.str(), __LINE__, [&]()
			{
				dnn_debug << "Epoch " << (epoch + 1) << "\n";

			const size_t mini_batch_count = train_split_x.size_i() / this->m_batch_size;
			for (size_t mini_batch = 0; mini_batch < mini_batch_count; ++mini_batch) // mini batches, mini_batch_count
			{
				std::stringstream batch_ss; batch_ss << "Batch " << (mini_batch + 1);
				tracing::tracer().beginRegion(batch_ss.str(), __LINE__);
			//	dnn_debug << "\nMini-batch " << (mini_batch + 1) << " / " << mini_batch_count << "\n";
						
		// 		/************************ prepare batch - copying **************************/
				skel_flatmap.setLabel("Copy batch training data X");
				skel_flatmap(batched_train_x, train_split_x.begin() + mini_batch * batched_train_x.size());
				skel_flatmap.setLabel("Copy batch training data Y");
				skel_flatmap(batched_train_y, train_split_y.begin() + mini_batch * batched_train_y.size());
				
		// 		/************************ forward **************************/
		 		std::stringstream forward_ss; forward_ss << "Epoch " << (epoch + 1) << " batch " << (mini_batch + 1) << " forward";
		 		tracing::tracer().beginRegion(forward_ss.str(), __LINE__);
		 		auto &predicted = this->forward(batched_train_x, true);
		 		tracing::tracer().endRegion();

				Tensor4<T> initial_gradient( // move out of loop
						predicted.size_i(), predicted.size_j(),
						predicted.size_k(), predicted.size_l(),
						"Initial gradient"
				);

				skel_delta.setLabel("Loss gradient");
				skel_delta(initial_gradient, predicted, batched_train_y);
				Tensor4<T> *gradient = &initial_gradient;
				
				skel_cross_entropy_batched_loss.setLabel("Crossentropy");
				skel_accuracy_batched.setLabel("Accuracy");
				skel_cross_entropy_batched_loss.setDefaultSize(this->m_batch_size);
				skel_accuracy_batched.setDefaultSize(this->m_batch_size);
				float loss = -skel_cross_entropy_batched_loss(batched_train_y, predicted) / this->m_batch_size;
				float accuracy = (float)skel_accuracy_batched(batched_train_y, predicted) / this->m_batch_size;

				total_train_loss += loss;
				total_train_accuracy += accuracy;

				std::cout << "Train Batch " << mini_batch + 1 << " - Loss: " << loss << " - Accuracy: " << accuracy << std::endl;

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
		 		std::stringstream backward_ss; backward_ss << "Epoch " << (epoch + 1) << " batch " << (mini_batch + 1) << " backward";
		 		tracing::tracer().beginRegion(backward_ss.str(), __LINE__);
				for (auto it = this->m_layers.rbegin(); it != this->m_layers.rend(); ++it)
				{
					auto &current_layer = **it;
					Tensor4<T> *forward_inputs = ((it + 1) != this->m_layers.rend()) ? &(**(it + 1)).m_vals : &batched_train_x;

					if (current_layer.is_trainable()) current_layer.m_learning_rate = learning_rate;

					gradient = current_layer.backward(forward_inputs, gradient, &batched_train_y);
				}
		 		tracing::tracer().endRegion(); // Backward
		 		tracing::tracer().endRegion(); // Batch

		 	} // end of mini batch
			
			auto score = this->evaluate(valid_split_x, valid_split_y);
			
		// 	if (print_progress)
		// 	{
		// \033[K
		// 		std::cout << "- Epoch " << std::setw(4) << (epoch + 1) << " / " << std::setw(4) << epochs << ":";
		// 		std::cout << " Valid Loss: " << std::setw(10) << std::get<0>(score) << ", Valid Accuracy: " << std::setw(10) << std::get<1>(score) << "\n";
		// 	}
		 }); // tracer epoch end

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

public: // change to protected
	bool m_initialized = false;

	std::vector<std::shared_ptr<Layer<T>>> m_layers{};
	size_t m_layers_count = 0;
	std::shared_ptr<skepu::PRNG> m_prng = std::make_shared<skepu::PRNG>(0); // default seed

	Dimensions m_input_size;
	size_t m_batch_size;

	Tensor4<T> *m_result = nullptr;
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
	
//	skepu::ml::dnn_debug << "Labels: " << labels << "\n";
//	skepu::ml::dnn_debug << "OneHots: " << result << "\n";
	
	
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

void artifical_dataset(size_t batch_size, float learning_rate, size_t epochs)
{
	const size_t train_items = 10, test_items = 1;
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
	
	train_x.flush();
	train_y.flush();
	test_x.flush();
	test_y.flush();

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
	auto score = model.evaluate(test_x, test_y);
	std::cout << "Test loss: " << std::get<0>(score) << "\nTest accuracy: " << std::get<1>(score) << "\n";
}



// #include <skepu-lib/ml.hpp>

int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " batch_size learning_rate epochs backend\n";
		exit(1);
	}

	const size_t batch_size = atoi(argv[1]);
	const float learning_rate = atof(argv[2]);
	const size_t epochs = atof(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	spec.setCPUThreads(8);
	skepu::setGlobalBackendSpec(spec);

//  artifical_dataset(batch_size, learning_rate, epochs);  exit(0);

	// Dataset configuration
	const std::string data_path = "./data/mnist/";
	const std::string param_path = "./data/pretrained/";
	const size_t num_classes = 10;

	using namespace skepu::ml;

	// Keras MNIST convolutional model
	auto model = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Conv2D(32).kernel(3, 3).weights(Init::Xavier).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_0.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(64).kernel(3, 3).weights(Init::Xavier).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_2.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2) << Flatten() //<< Dropout(0.5)
		<< Dense(num_classes).weights(Init::Xavier).biases(Init::Zero)
				// .pretrained(param_path + "params_layer_6.csv")
				.activation(Act::SoftMax);
/*
	// Multi-layer perceptron
	auto modelb = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Flatten() << Dropout(0.5)
		<< Dense(100).weights(Init::Uniform).biases(Init::Zero)
		.pretrained(param_path + "fc_hidden_params_layer_2.csv")
		.activation(Act::ReLU)
		<< Dense(num_classes).weights(Init::Uniform).biases(Init::Zero)
		.pretrained(param_path + "fc_hidden_params_layer_3.csv")
		.activation(Act::SoftMax);

	// Simpler conv network
	auto modelc = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Conv2D(1).kernel(3, 3)
				.pretrained(param_path + "simple_params_layer_0.csv")
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Flatten() //<< Dropout(0.5)
		<< Dense(num_classes)
				.pretrained(param_path + "simple_params_layer_3.csv")
				.activation(Act::SoftMax);

	// Simpler conv network
	auto model = SequentialModel(Dimensions{batch_size, 28, 28, 1})
	//	<< Conv2D(1).kernel(3, 3)
	//			.activation(Act::ReLU)
	//	<< MaxPooling2D().kernel(2, 2)
		<< Conv2D(2).kernel(3, 3)
				.activation(Act::ReLU)
		<< MaxPooling2D().kernel(2, 2)
		<< Flatten() //<< Dropout(0.5)
		<< Dense(num_classes)
				.activation(Act::SoftMax);


	auto modele = SequentialModel(Dimensions{batch_size, 28, 28, 1})
		<< Flatten()
		<< Dense(100).weights(Init::Xavier).biases(Init::Zero)
		<< Dense(num_classes).weights(Init::Xavier).biases(Init::Zero)
			.activation(Act::SoftMax);*/
			
	{
	
	// Initialize model
	model.init(); // memory and such
	std::cout << model.summary();

	// MINST loading
	skepu::Tensor4<float> train_images = mnist::parse_mnist_images(data_path + "train-images.idx3-ubyte", 0, 1.0);
	skepu::Tensor4<float> test_images  = mnist::parse_mnist_images(data_path + "t10k-images.idx3-ubyte", 0, 1.0);
	skepu::Vector<mnist::label_t> train_labels_temp = mnist::parse_mnist_labels(data_path + "train-labels.idx1-ubyte");
	skepu::Vector<mnist::label_t> test_labels_temp  = mnist::parse_mnist_labels(data_path + "t10k-labels.idx1-ubyte");
	skepu::Tensor4<float> train_labels = mnist_labels_to_onehot(train_labels_temp, num_classes);
	skepu::Tensor4<float> test_labels  = mnist_labels_to_onehot(test_labels_temp,  num_classes);
	
	std::cout << "Train data size: " << train_images.size_i() << "\n";
	std::cout << "Test data size: " << test_images.size_i() << "\n";
	
	// GPU software coherency bug: the following lines are reqired as workaround
	train_images.flush();
	train_labels.flush();
	test_images.flush();
	test_labels.flush();

	
/*	train_labels = cut_tensor4(train_labels, 100);
	train_images = cut_tensor4(train_images, 100);*/
	test_labels = cut_tensor4(test_labels, 1000);
	test_images = cut_tensor4(test_images, 1000);
	
//	skepu::ml::dnn_debug << "train_labels: " << train_labels << "\n";
 //	skepu::ml::dnn_debug << "train_images: " << train_images << "\n";

	// Testing
	//auto score_pt = model.evaluate(test_images, test_labels);
	//std::cout << "Loss: " << std::get<0>(score_pt) << "\nAccuracy: " << std::get<1>(score_pt) << "\n";

	// Training
//	model.fit(train_images, train_labels, epochs, learning_rate, 0.1 ,  true); //true is logging ->progress bar

	// Testing
	skepu::Matrix<int> confusion_matrix(num_classes,num_classes, 0);

	auto score = model.evaluate(test_images, test_labels, &confusion_matrix);
	std::cout << "Test loss: " << std::get<0>(score) << "\nTest accuracy: " << std::get<1>(score) << "\n";

	//confusion matrix
	skepu::io::cout << confusion_matrix << "\n";

	// Serialize weights to file
	model.serialize("skepu_dnn_trained_model_weights"); // generate parameter csv files
	
	std::cout << "End of block\n";
	}
	std::cout << "End of main\n";

	return 0;
}
