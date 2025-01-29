

#define DNN_CONST /*const&*/

float fully_connected_uf(
	float bias,
	skepu::MatRow<float> DNN_CONST weights,
	skepu::Vec<float> DNN_CONST in
)
{
	float res = bias;
	for (size_t i = 0; i < in.size; ++i)
		res += weights(i) * in(i);
	return res;
}

float fully_connected_batched_uf(
	skepu::Index4D index,
	skepu::Vec<float> DNN_CONST bias,
	skepu::Mat<float> DNN_CONST weights,
	skepu::Ten4<float> DNN_CONST in
)
{
	size_t batch_index = index.k;
	float res = bias(index.l);
	for (size_t i = 0; i < in.size_k; ++i)
		res += weights(index.l, i)
		* in(0, 0, batch_index, i);
	return res;
}

float fully_connected_backprop_uf(
	//float bias,,
	float sigmoid_p,
	skepu::MatCol<float> DNN_CONST weights,
	skepu::Vec<float> DNN_CONST delta
)
{
	float res = 0;//bias;
	for (size_t i = 0; i < delta.size; ++i)
		res += weights(i) * delta(i);
	return res * sigmoid_p;
}

float fully_connected_batched_backprop_uf(
	//float bias,,
	skepu::Index2D index,
	float sigmoid_p,
	skepu::Mat<float> DNN_CONST weights,
	skepu::Mat<float> DNN_CONST delta
)
{
	size_t batch_index = index.row;
	float res = 0;//bias;
	for (size_t i = 0; i < delta.cols; ++i)
		res += weights(i, index.col) * delta(batch_index, i);
	return res * sigmoid_p;
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

float relu_uf(float x)
{
	return (x < 0) ? 0 : x;
}

float relu_prime_uf(float x)
{
	return (x > 0) ? 1 : 0;
}

// SoftMax for non-batched
float softmax_uf_1_map(float x)
{
	return exp(x);
}

float softmax_uf_1_reduce(float lhs, float rhs)
{
	return lhs + rhs;
}

float softmax_uf_2(float x, float sum)
{
	return exp(x) / sum;
}

// BATCHED SOFTMAX
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

float tanh_uf(float x)
{
	return tanh(x);
}


float flatten_uf(float x)
{
	return x;
}


float max_pooling_uf(
	skepu::Pool4D<float> DNN_CONST pool
)
{
	float maxval = 10e-10;
	for (size_t j = 0; j < pool.sj; ++j)
		for (size_t k = 0; k < pool.sk; ++k)
		{
			float val = pool(0, j, k, 0);
			maxval = (maxval > val) ? maxval : val;
		}
	return maxval;
}

float avg_pooling_uf(skepu::Pool4D<float> DNN_CONST pool /*coefficient, bias*/)
{
/*	float sum = 0;
	for (size_t i = 0; i < pool.si; ++i)
		for (size_t j = 0; j < pool.sj; ++j)
			sum += pool(i, j);
	return sum / (pool.si + pool.sj);*/
	return 0;
}

float generalized_pooling_uf(
	skepu::Index3D index,
	skepu::Pool3D<float> DNN_CONST pool
	/*coefficient, bias*/
)
{
	size_t i = index.i; // find my feature map
	float sum = 0;
	for (size_t j = 0; j < pool.sj; ++j)
		for (size_t k = 0; k < pool.sk; ++k)
			sum += pool(i, j, k);
	return sum /* * coefficient + bias*/;
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
	skepu::Index2D index,
	skepu::Region2D<float> DNN_CONST in,
	skepu::Mat<float> DNN_CONST weights,
	skepu::Vec<float> DNN_CONST bias,
	skepu::Vec<int> DNN_CONST connection_map
) // bias
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
	float res = 0;
	for (size_t i = -in.oi; i <= in.oi; ++i)
		for (size_t j = -in.oj; j <= in.oj; ++j)
			for (size_t k = 0; k < weights.size_k; ++k)
				res += weights(0, i + in.oi, j + in.oi, k)
				       * in(0, i, j, k - index.k);
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
	return res;
}


float mse_prime_uf(float a, float y)
{
	return (y - a) * (y - a) * 0.5;
}

float mse_uf(float a, float y)
{
	return y - a;
}

// Column-wise reduction over batch items
float generate_nabla_b_uf(
	skepu::Index1D index,
	skepu::Mat<float> delta
)
{
	size_t bias_index = index.i;
	size_t batch_size = delta.rows;
	float res = 0;
	for (size_t i = 0; i < batch_size; ++i)
		res += delta(i, bias_index);
	return res;
}

float generate_nabla_w_uf(
	skepu::Index2D index,
	skepu::Mat<float> delta,
	skepu::Mat<float> activations
)
{
	size_t batch_size = delta.rows;
	float res = 0;
	for (size_t i = 0; i < batch_size; ++i)
		res += delta(i, index.row) * activations(i, index.col);
	return res;
}

float update_uf(
	float old,
	float updated,
	float eta // learning rate
)
{
	return old - eta * updated;
}

template<typename T>
T sum_uf(T a, T b)
{
	return a + b;
}

float product_uf(float a, float b)
{
	return a * b;
}

float flatmap_uf(float a)
{
	return a;
}

float reshape_y_uf(skepu::Index2D index, skepu::Vec<size_t> DNN_CONST y)
{
	if (index.col == y(index.row))
		return 1.f;
	return 0.f;
}

size_t argmax_uf(skepu::Index1D index, skepu::MatRow<float> DNN_CONST y)
{
	float max_v = -999999.f;
	size_t max_i = 0;
	for (size_t i = 0; i < y.cols; ++i)
	{
		if (y(i) > max_v)
		{
			max_v = y(i);
			max_i = i;
		}
	}
	return max_i;
}

size_t eval_uf(size_t a, size_t b) { return (a == b) ? 1 : 0; }


template<typename T>
T dropout_uf(skepu::Random<1> &rand, T el, float rate)
{
	float p = rand.getNormalized();
	return (p > rate) ? (el * 1.0/(1.0 - rate)) : 0;
}





namespace skepu
{
	template<typename T>
	std::string string_tensor_size(skepu::Tensor4<T> &t)
	{
		std::ostringstream os;
		os << "("
		<< t.size_i() << " x "
		<< t.size_j() << " x "
		<< t.size_k() << " x "
		<< t.size_l() << ")";
		return os.str();
	}



	namespace ml
	{
		using Dimensions = std::tuple<size_t, size_t, size_t, size_t>;

		enum class Init
		{
			Random,
			Zero
		};
		
		enum class Activate
		{
			None,
			ReLU,
			Sigmoid,
			TanH,
			SoftMax
		};
		
		using ModelType = float;
		
		// Forward
		auto skel_flatmap = skepu::Map(flatmap_uf);
		auto skel_dense = skepu::Map(fully_connected_batched_uf);

		auto skel_act_sigmoid = skepu::Map(sigmoid_uf);
		auto skel_act_relu = skepu::Map(relu_uf);
		auto skel_act_softmax_1 = skepu::MapReduce(softmax_uf_1_map, softmax_uf_1_reduce);
		auto skel_act_softmax_2 = skepu::Map<1>(softmax_uf_2);

		auto skel_softmax_batched_1 = skepu::MapPool(softmax_batched_uf_1);
		auto skel_softmax_batched_2 = skepu::Map(softmax_batched_uf_2);

		auto skel_act_tanh = skepu::Map(tanh_uf);
		auto skel_dropout = skepu::Map<1>(dropout_uf<float>);

		auto skel_conv_1d = skepu::MapOverlap(convolutional_1d_uf);
		auto skel_conv_2d = skepu::MapOverlap(convolutional_2d_uf);

		auto skel_pool_max = skepu::MapPool(max_pooling_uf);
		
		// Backprop
		auto cost_prime = skepu::Map(mse_prime_uf);
		auto reshape_y = skepu::Map(reshape_y_uf);
		auto hidden_dense_backprop = skepu::Map(fully_connected_batched_backprop_uf);
		
		auto skel_act_sigmoid_prime = skepu::Map(sigmoid_prime_uf);
		auto skel_act_relu_prime = skepu::Map(relu_prime_uf);
		
		auto hadamard = skepu::Map(product_uf);
		auto generate_nabla_b = skepu::Map(generate_nabla_b_uf);
		auto generate_nabla_w = skepu::Map(generate_nabla_w_uf);
		auto update = skepu::Map<2>(update_uf);
		auto argmax = skepu::Map(argmax_uf);
		auto evaluate = skepu::MapReduce<2>(eval_uf, sum_uf<size_t>);
		auto cost_function = skepu::MapReduce(mse_uf, sum_uf<float>);
		
		namespace impl
		{
			template<typename T>
			class SequentialModel;

			template<typename T = float>
			class Layer
			{
			public:
				Layer(Dimensions arg_output_size)
				: m_output_size{arg_output_size}
				{
					// construct a layer base class
					std::cout << "LOG: " << "Layer with output size " 
					<< std::get<0>(this->m_output_size) << "\n";
				}

				Layer()
				: m_output_size{0,0,0,0}
				{
					// construct a layer base class
				}
				
				Layer(Layer &) = default;
				Layer(Layer &&) = default;
				
				virtual ~Layer()
				{
				
				}
				
			public:
			
				virtual Layer&& init(skepu::ml::Init arg_init) 
				{
					this->m_init = arg_init;
					return std::move(*this);
				}
				
				virtual Layer&& activation(Activate new_act)
				{
					this->m_act = new_act;
					return std::move(*this);
				}
				
				virtual std::string summary()
				{
					std::ostringstream os;
					os << "[";
					if (this->m_is_setup)
						os << "Initialized";
					else
						os << "Not initialized";
					
					os << "; size{"
					<< std::get<0>(this->m_output_size) << " x "
					<< std::get<1>(this->m_output_size) << " x "
					<< std::get<2>(this->m_output_size) << " x "
					<< std::get<3>(this->m_output_size);
					os << "}";
					os << "; batch_size{";
					os << this->m_batch_size;
					os << "}";
					os << "] ";
					return os.str();
				}
				
				
				virtual void setup(Dimensions input_size)
				{
					this->m_is_setup = true;
				}
				
				virtual skepu::Tensor4<T> *forward(skepu::Tensor4<T> *inputs)
				{
					std::cout << "WRONG\n";
					return inputs;
				}
				
				virtual void backward()
				{
				
				}
				
			public: // for now, protected later?
				
				bool m_is_setup = false;
				size_t m_batch_size;
				Dimensions m_output_size;

				Activate m_act = Activate::None; // default value?
				Init m_init = Init::Zero;
				
				std::shared_ptr<Layer> *predecessor = nullptr;
				std::shared_ptr<Layer> *successor = nullptr;
				
				friend class SequentialLayer;
			};
			
			
			
			
			
			template<typename T = float>
			class Dense: public Layer<T>
			{
			public:
				Dense(size_t arg_output_size)
				: Layer<T>(), m_out(arg_output_size)
				{
				}
				
				Dense(Dense &&source)
				: Layer<T>(source)
				{
					this->m_weights = std::move(source.m_weights);
					this->m_biases = std::move(source.m_biases);
					this->m_vals.swap(source.m_vals); // TODO: fix moves
					this->m_biases = std::move(source.m_biases);
					this->m_out = source.m_out;
				}
				
				~Dense() override
				{
					
				}
				
				// Overridden members
				
				Dense&& init(skepu::ml::Init arg_init) override
				{
					Layer<T>::init(arg_init);
					return std::move(*this);
				}
				
				void setup(Dimensions input_size) override
				{
					Layer<T>::setup(input_size);
					
					size_t out = this->m_out;
					size_t in = std::get<3>(input_size);
					
					this->m_weights.init(out, in);
					this->m_biases.init(out);

					this->m_weights.randomizeReal(-1, 1);
				//	this->m_biases.randomize(-1, 1);

					this->m_output_size = {
						std::get<0>(input_size),
						1,
						1,
						out
					};

					this->m_vals.init(
						std::get<0>(this->m_output_size),
						std::get<1>(this->m_output_size),
						std::get<2>(this->m_output_size),
						std::get<3>(this->m_output_size)
					);
					
				}
				
				Dense&& activation(Activate new_act) override
				{
					Layer<T>::activation(new_act);
					return std::move(*this);
				}
				
				std::string summary() override
				{
					std::ostringstream os;
					os << Layer<T>::summary();
					os << "Dense layer ";
					os << " train_params{ ("
					<< this->m_weights.size_i() << " x "
					<< this->m_weights.size_j() << ")";
					os << this->m_weights.size() << " + ";
					os << this->m_biases.size() << "}";
					return os.str();
				}
				
				skepu::Tensor4<T> *forward(skepu::Tensor4<T> *inputs) override
				{
					std::cout << "Dense input: " << string_tensor_size(*inputs) << "\n";
					std::cout << "Dense vals: " << string_tensor_size(this->m_vals) << "\n";
					
					skel_dense(this->m_vals, this->m_biases, this->m_weights, *inputs);
					return &this->m_vals;
				}
				
				void backward() override
				{
				
				}
				
			protected:
				size_t m_out = 0;
				
				skepu::Matrix<T> m_weights;
				skepu::Vector<T> m_biases;
				
				skepu::Tensor4<T> m_vals;
			};
			
			



			template<typename T = float>
			class Activation: public Layer<T>
			{
			public:
				
				Activation(Activate arg_act)
				: Layer<T>()
				{
					this->m_act = arg_act;
				}

				// Move constructor
				Activation(Activation<T> &&source)
				: Layer<T>(source)
				{
					this->m_vals = std::move(source.m_vals);
				}

				void setup(Dimensions input_size) override
				{
					Layer<T>::setup(input_size);
					
				//	size_t out = std::get<0>(this->m_output_size);
				//	size_t in = std::get<0>(input_size);

					this->m_output_size = input_size;

					this->m_vals.init(
						std::get<0>(this->m_output_size),
						std::get<1>(this->m_output_size),
						std::get<2>(this->m_output_size),
						std::get<3>(this->m_output_size)
					);

					this->m_temp_vals.init(
						std::get<0>(this->m_output_size),
						std::get<1>(this->m_output_size),
						std::get<2>(this->m_output_size),
						1
					);
				}

				skepu::Tensor4<T> *forward(skepu::Tensor4<T> *inputs) override
				{
					std::cout << "Act input: " << string_tensor_size(*inputs) << "\n";
					std::cout << "Act vals: " << string_tensor_size(this->m_vals) << "\n";

					if (this->m_act == Activate::ReLU)
					{
						skel_act_relu(this->m_vals, *inputs);
					}
					else if (this->m_act == Activate::Sigmoid)
					{
						skel_act_sigmoid(this->m_vals, *inputs);
					}
					else if (this->m_act == Activate::SoftMax)
					{
						/* // This is non-batched softmax
						T sum = skel_act_softmax_1(*inputs);
						skel_act_softmax_2(this->m_vals, *inputs, sum);*/

						// Batched
						skel_softmax_batched_1.setPoolSize(1, 1, 1, inputs->size_l());
						skel_softmax_batched_1.setStride(1, 1, 1, inputs->size_l());
						skel_softmax_batched_1(this->m_temp_vals, *inputs);
						skel_softmax_batched_2(this->m_vals, *inputs, this->m_temp_vals);
					}
					else if (this->m_act == Activate::TanH)
					{
						skel_act_tanh(this->m_vals, *inputs);
					}
					return &this->m_vals;
				}

				std::string summary() override
				{
					std::ostringstream os;
					os << Layer<T>::summary();
					os << "Activation layer: ";
					if (this->m_act == Activate::ReLU)
						os << "ReLU";
					else if (this->m_act == Activate::Sigmoid)
						os << "Sigmoid";
					else if (this->m_act == Activate::SoftMax)
						os << "SoftMax";
					else if (this->m_act == Activate::TanH)
						os << "TanH";
					return os.str();
				}

			private:

				skepu::Tensor4<T> m_temp_vals;
				skepu::Tensor4<T> m_vals;

			};



			template<typename T = float>
			class Conv1D: public Layer<T>
			{
			public:
				Conv1D(size_t no_features)
				: Layer<T>(), m_no_features(no_features)
				{
				}

				// Move constructor
				Conv1D(Conv1D &&source)
				: Layer<T>(source),
				m_no_features(source.m_no_features)
				{
					this->m_kernel = source.m_kernel;
					this->m_stride = source.m_stride;
					this->m_padding = source.m_padding;
					this->m_no_features = source.m_no_features;

					this->m_weights = std::move(source.m_weights);
					this->m_biases = std::move(source.m_biases);
				}
				
				~Conv1D() override
				{
					
				}
				
				Conv1D&& init(skepu::ml::Init arg_init) override
				{
					Layer<T>::init(arg_init);
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
				}

				Conv1D&& activation(Activate new_act) override
				{
					Layer<T>::activation(new_act);
					return std::move(*this);
				}
				
				Conv1D&& kernel(size_t new_kernel)
				{
					this->m_kernel = new_kernel;
					return std::move(*this);
				}
				
				Conv1D&& strides(size_t new_stride)
				{
					this->m_stride = new_stride;
					return std::move(*this);
				}
				
				Conv1D&& padding(bool new_padding)
				{
					this->m_padding = new_padding;
					return std::move(*this);
				}
				
				// Overridden members
				
				std::string summary() override
				{
					std::stringstream os;
					os << Layer<T>::summary();
					os << "Conv1D layer with " << this->m_no_features << " features; kernel size ";
					os << "{" << this->m_kernel << "}";
					os << "; stride ";
					os << "{" << this->m_stride << "}";
					os << " train_params{";
					os << this->m_weights.size() << " + ";
					os << this->m_biases.size() << "}";
					return os.str();
				}
				
				skepu::Tensor4<T> *forward(skepu::Tensor4<T> *inputs) override
				{
					std::cout << "CONV1D forward\n";
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
				
				void backward() override
				{
				
				}
				
			protected:
				
				int m_no_features;
				size_t m_kernel {1}; // default
				size_t m_stride {1}; // default
				bool m_padding = true; // true = "same" in Keras, false = "valid" in Keras
			
				skepu::Matrix<T> m_weights;
				skepu::Vector<T> m_biases;

				skepu::Tensor4<T> m_vals;
			};
			
			
			
			template<typename T = float>
			class Conv2D: public Layer<T>
			{
			public:
				Conv2D(int no_features)
				: Layer<T>(), m_no_features(no_features)
				{
				}

				// Move constructor
				Conv2D(Conv2D &&source)
				: Layer<T>(source),
				m_no_features(source.m_no_features)
				{
					this->m_kernel = std::move(source.m_kernel);
					this->m_strides = std::move(source.m_strides);
					this->m_padding = std::move(source.m_padding);
					this->m_no_features = source.m_no_features;
					this->m_weights.swap(source.m_weights);
					this->m_biases.swap(source.m_biases);
				}
				
				~Conv2D() override
				{
					
				}

				void setup(Dimensions input_size) override
				{
					Layer<T>::setup(input_size);
					
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

					this->m_vals.init(
						std::get<0>(this->m_output_size),
						std::get<1>(this->m_output_size),
						std::get<2>(this->m_output_size),
						std::get<3>(this->m_output_size)
					);
				}


				
				Conv2D&& init(skepu::ml::Init arg_init) override
				{
					Layer<T>::init(arg_init);
					return std::move(*this);
				}
				
				Conv2D&& activation(Activate new_act) override
				{
					Layer<T>::activation(new_act);
					return std::move(*this);
				}
				
				Conv2D&& kernel(size_t k1, size_t k2)
				{
					this->m_kernel = {k1, k2};
					return std::move(*this);
				}
				
				Conv2D&& stride(size_t s1, size_t s2)
				{
					this->m_strides = {s1, s2};
					return std::move(*this);
				}
				
				Conv2D&& padding(bool new_padding)
				{
					this->m_padding = new_padding;
					return std::move(*this);
				}
				
				// Overridden members
				
				std::string summary() override
				{
					std::stringstream os;
					os << Layer<T>::summary();
					os << "Conv2D layer with " << this->m_no_features << " features and kernel size ";
					os << "{" << std::get<0>(this->m_kernel) << ", " << std::get<1>(this->m_kernel) << "}";
					os << " train_params{ ("
					<< this->m_weights.size_i() << " x "
					<< this->m_weights.size_j() << " x "
					<< this->m_weights.size_k() << " x "
					<< this->m_weights.size_l() << ")"
					<< (this->m_weights.size_i() *
					this->m_weights.size_j() *
					this->m_weights.size_k() *
					this->m_weights.size_l()) << " + "
					<< this->m_biases.size() << "}";
					return os.str();
				}

				skepu::Tensor4<T> *forward(skepu::Tensor4<T> *inputs) override
				{
					std::cout << "Conv2D input: " << string_tensor_size(*inputs) << "\n";
					std::cout << "Conv2D vals: " << string_tensor_size(this->m_vals) << "\n";
					skel_conv_1d.setOverlap(0,
						(std::get<0>(this->m_kernel) - 1) / 2,
						(std::get<1>(this->m_kernel) - 1) / 2,
					0);
					skel_conv_1d.setStride(1,
						std::get<0>(this->m_strides),
						std::get<1>(this->m_strides),
					1);

					if (this->m_padding)
					{
						skel_conv_1d.setEdgeMode(skepu::Edge::Pad);
						skel_conv_1d.setPad(0);
					}

					skel_conv_2d(this->m_vals, *inputs, this->m_weights, this->m_biases);
					
					return &this->m_vals;
				}
				
				void backward() override
				{
				
				}
				
			protected:
				
				int m_no_features;
				std::tuple<int, int> m_kernel {1, 1}; // default
				std::tuple<int, int> m_strides {1, 1}; // default
				bool m_padding = false; // true = "same" in Keras, false = "valid" in Keras
				
				skepu::Tensor4<T> m_weights;
				skepu::Vector<T> m_biases;

				skepu::Tensor4<T> m_vals;
			
			};
			
			
			
			
			
			
			
			template<typename T = float>
			class Subsampling2D: public Layer<T>
			{
			public:
				Subsampling2D()
				: Layer<T>()
				{
				}

				// Move constructor
				Subsampling2D(Subsampling2D &&source)
				: Layer<T>(source)
				{
					this->m_kernel = source.m_kernel;
					this->m_strides = source.m_strides;
					this->m_padding = source.m_padding;
					this->m_vals = std::move(source.m_vals);
				}
				
				~Subsampling2D() override
				{
					
				}

				void setup(Dimensions input_size) override
				{
					Layer<T>::setup(input_size);
					
					std::cout << "Setup size Subsampling: "
					<< std::get<0>(input_size) << " x "
					<< std::get<1>(input_size) << " x "
					<< std::get<2>(input_size) << " x "
					<< std::get<3>(input_size) << "\n";
					size_t out = std::get<0>(this->m_output_size);
					size_t in = std::get<0>(input_size);

					this->m_output_size = {
						std::get<0>(input_size),
						std::get<1>(input_size) / (std::get<0>(this->m_kernel)),
						std::get<2>(input_size) / (std::get<1>(this->m_kernel)),
						std::get<3>(input_size)
					};

					this->m_vals.init(
						std::get<0>(this->m_output_size),
						std::get<1>(this->m_output_size),
						std::get<2>(this->m_output_size),
						std::get<3>(this->m_output_size)
					);
				}
				
				virtual Subsampling2D&& kernel(size_t k1, size_t k2)
				{
					this->m_kernel = {k1, k2};
					return std::move(*this);
				}
				
				virtual Subsampling2D&& stride(size_t s1, size_t s2)
				{
					this->m_strides = {s1, s2};
					return std::move(*this);
				}
				
				virtual Subsampling2D&& padding(bool new_padding)
				{
					this->m_padding = new_padding;
					return std::move(*this);
				}
				
				std::string summary() override
				{
					std::stringstream os;
					os << Layer<T>::summary();
					os << "Subsampling layer with kernel size ";
					os << "{" << std::get<0>(this->m_kernel) << ", " << std::get<1>(this->m_kernel) << "}";
					return os.str();
				}
				
			protected:
				
				int no_features;
				std::tuple<int, int> m_kernel {1, 1}; // default
				std::tuple<int, int> m_strides {1, 1}; // default to pool size
				bool m_padding = true; // true = "same" in Keras, false = "valid" in Keras
			
				skepu::Tensor4<T> m_vals;
			};
			
			
			
			template<typename T = float>
			class MaxPooling2D: public Subsampling2D<T>
			{
			public:
				MaxPooling2D()
				: Subsampling2D<T>()
				{
				}
				
				MaxPooling2D(MaxPooling2D &&source)
				: Subsampling2D<T>() // todo kernel
				{
					this->m_kernel = source.m_kernel;
					this->m_strides = source.m_strides;
					this->m_padding = source.m_padding;
				}
				
				~MaxPooling2D() override
				{
					
				}

				MaxPooling2D&& kernel(size_t k1, size_t k2) override
				{
					Subsampling2D<T>::kernel(k1, k2);
					return std::move(*this);
				}
				
				MaxPooling2D&& stride(size_t s1, size_t s2) override
				{
					Subsampling2D<T>::stride(s1, s2);
					return std::move(*this);
				}
				
				MaxPooling2D&& padding(bool new_padding) override
				{
					Subsampling2D<T>::padding(new_padding);
					return std::move(*this);
				}
				
				std::string summary() override
				{
					std::stringstream os;
					os << Layer<T>::summary();
					os << "MaxPooling2D layer with kernel size ";
					os << "{" << std::get<0>(this->m_kernel) << ", " << std::get<1>(this->m_kernel) << "}";
					return os.str();
				}
				
				skepu::Tensor4<T> *forward(skepu::Tensor4<T> *inputs) override
				{
					std::cout << "MaxPool2D input: " << string_tensor_size(*inputs) << "\n";
					std::cout << "MaxPool2D vals: " << string_tensor_size(this->m_vals) << "\n";
					

					skel_pool_max.setPoolSize(1,
						std::get<0>(this->m_kernel),
						std::get<1>(this->m_kernel),
					1);
					// todo: override with set strides
					skel_pool_max.setStride(1,
						std::get<0>(this->m_kernel),
						std::get<1>(this->m_kernel),
					1);
					skel_pool_max(this->m_vals, *inputs);
					return &this->m_vals;
				}
				
				void backward() override
				{
				
				}
			};
			
			
			
			
			
			template<typename T = float>
			class Dropout: public Layer<T>
			{
			public:
				Dropout(float arg_rate)
				: Layer<T>(), m_dropout_rate(arg_rate)
				{
				
				}
				
				// Move constructor
				Dropout(Dropout &&source)
				: Layer<T>(source), m_dropout_rate(source.m_dropout_rate)
				{
					this->m_vals = std::move(source.m_vals);
					this->m_prng = std::move(source.m_prng);
				}
				
				~Dropout() override
				{
					
				}

				void setup(Dimensions input_size) override
				{
					Layer<T>::setup(input_size);
					
					size_t out = std::get<0>(this->m_output_size);
					size_t in = std::get<0>(input_size);

					this->m_vals.init(1, 1, this->m_batch_size, in);
					this->m_output_size = input_size;
				}

				skepu::Tensor4<T> *forward(skepu::Tensor4<T> *inputs) override
				{
					skel_dropout.setPRNG(this->m_prng);
					skel_dropout(this->m_vals, *inputs, this->m_dropout_rate);
					return &this->m_vals;
				}
				
				std::string summary() override
				{
					std::stringstream os;
					os << Layer<T>::summary();
					os << "Dropout layer with rate " << this->m_dropout_rate;
					return os.str();
				}
			
			private:
				float m_dropout_rate;
				skepu::Tensor4<T> m_vals;
				skepu::PRNG m_prng;
			};
			
			
			
			
			template<typename T = float>
			class Flatten: public Layer<T>
			{
			public:
				
				Flatten()
				: Layer<T>()
				{
				}
				
				Flatten(Flatten&&) = default;
				
				~Flatten() override
				{
					
				}

				void setup(Dimensions input_size) override
				{
					Layer<T>::setup(input_size);
					
					this->m_output_size = {
						std::get<0>(input_size),
						1, 1,
						std::get<1>(input_size) * 
						std::get<2>(input_size) * 
						std::get<3>(input_size)
					};

					this->m_vals.init(
						std::get<0>(this->m_output_size),
						std::get<1>(this->m_output_size),
						std::get<2>(this->m_output_size),
						std::get<3>(this->m_output_size)
					);
				}

				skepu::Tensor4<T> *forward(skepu::Tensor4<T> *inputs) override
				{
					std::cout << "Flatten input: " << string_tensor_size(*inputs) << "\n";
					std::cout << "Flatten vals: " << string_tensor_size(this->m_vals) << "\n";
					
					skel_flatmap(this->m_vals, *inputs);

					return &this->m_vals;
				}

				std::string summary() override
				{
					std::stringstream os;
					os << Layer<T>::summary();
					os << "Flatten layer ";
					return os.str();
				}

			private:
				skepu::Tensor4<T> m_vals;
			};
			
			
			
			
			
			
			template<typename T = float>
			class SequentialModel
			{
			public:
			
				void init()
				{
					// Connect layers together in internal doubly-linked list
					std::shared_ptr<Layer<T>> *prev = nullptr;
					for (auto &current : this->layers)
					{
						current->predecessor = prev;
						if (prev) prev->get()->successor = &current;
					}
					
					// Allocate data
					Dimensions input_size = this->m_input_size;
					for (auto &current : this->layers)
					{
						current->setup(input_size);
						input_size = current->m_output_size;
					}
				
				}
				
				SequentialModel(Dimensions arg_input_size)
				: m_input_size{arg_input_size}, m_batch_size{std::get<0>(arg_input_size)}
				{

				}

				SequentialModel(size_t arg_input_size, size_t arg_batch_size)
				: m_input_size{arg_input_size, 1,1,1}, m_batch_size{arg_batch_size}
				{

				}
				
				// move constructor
				SequentialModel(std::vector<std::shared_ptr<Layer<T>>> &&arg_layers)
				: layers{arg_layers}
				{
					this->init();
				}
				
				template <typename ConcreteLayer>
				void add(ConcreteLayer &&l)
				{
					Activate activationFunc = l.m_act;
					l.m_batch_size = this->m_batch_size;
					this->layers.push_back(std::make_shared<typename std::remove_reference<ConcreteLayer>::type>(std::forward<ConcreteLayer>(l)));

					if (activationFunc != Activate::None)
					{
						auto act_layer = std::make_shared<Activation<T>>(activationFunc);
						act_layer->m_batch_size = this->m_batch_size;
						this->layers.push_back(act_layer);
					}

				}
				
				void forward(skepu::Tensor4<float> &inputs)
				{
					skepu::Tensor4<float> *temp = &inputs; 
					for (auto &current : this->layers)
					{
						temp = current->forward(temp);
					}
				}

				void backward(skepu::Tensor4<float> &train_data)
				{

					std::cout << "HERE\n";
					skepu::Tensor4<float> batched_train_input(
						this->m_batch_size,
						std::get<1>(this->m_input_size),
						std::get<2>(this->m_input_size),
						std::get<3>(this->m_input_size)
					);
					
					std::cout << "Train data size: " << string_tensor_size(train_data) << "\n";
					std::cout << "Batch data size: " << string_tensor_size(batched_train_input) << "\n";

					size_t mini_batch_count = train_data.size_j() / this->m_batch_size;

					for (size_t mini_batch = 0; mini_batch < mini_batch_count; ++mini_batch)
					{
						std::cout << "\nMini-batch " << (mini_batch+1) << " / " << mini_batch_count << "\n";
						skel_flatmap(batched_train_input, train_data.begin() + mini_batch * this->m_batch_size * train_data.size_k() * train_data.size_l());

						this->forward(batched_train_input);

					}
				}
				
				std::string summary()
				{
					std::stringstream os;
					for (auto &layer : this->layers)
					{
						os << layer->summary() << "\n";
					}
					return os.str();
					
				}
				
				
			protected:
				
				std::vector<std::shared_ptr<Layer<T>>> layers {};

				Dimensions m_input_size;
				size_t m_batch_size;
				
			};
			
			template <typename T, typename ConcreteLayer>
			SequentialModel<T> &operator<<(SequentialModel<T> &n, ConcreteLayer &&l)
			{
			n.add(std::forward<ConcreteLayer>(l));
			return n;
			}
		}

		template<typename T = float, typename... Args>
		impl::Dense<T>
		Dense(Args&&... args)
		{
			return impl::Dense<T>(std::forward<Args>(args)...);
		}

		template<typename T = float, typename... Args>
		impl::Conv1D<T>
		Conv1D(Args&&... args)
		{
			return impl::Conv1D<T>(std::forward<Args>(args)...);
		}

		template<typename T = float, typename... Args>
		impl::Conv2D<T>
		Conv2D(Args&&... args)
		{
			return impl::Conv2D<T>(std::forward<Args>(args)...);
		}

		template<typename T = float, typename... Args>
		impl::MaxPooling2D<T>
		MaxPooling2D(Args&&... args)
		{
			return impl::MaxPooling2D<T>(std::forward<Args>(args)...);
		}

		template<typename T = float, typename... Args>
		impl::Flatten<T>
		Flatten(Args&&... args)
		{
			return impl::Flatten<T>(std::forward<Args>(args)...);
		}

		template<typename T = float, typename... Args>
		impl::Dropout<T>
		Dropout(Args&&... args)
		{
			return impl::Dropout<T>(std::forward<Args>(args)...);
		}

		template<typename T = float, typename... Args>
		impl::SequentialModel<T>
		SequentialModel(Args&&... args)
		{
			return impl::SequentialModel<T>(std::forward<Args>(args)...);
		}
	
	}

}



