#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_Conv2D_HPP;



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


auto skel_conv_2d = skepu::MapOverlap(convolutional_2d_uf);
auto skel_conv_2d_backward = skepu::MapOverlap(conv_2d_backward_uf);
auto skel_propagate_conv_2d_weight_gradient = skepu::Map<0>(propagate_conv_2d_weight_gradient_uf);
auto skel_propagate_conv_2d_bias_gradient = skepu::Map<0>(propagate_conv_2d_bias_gradient_uf);



namespace skepu::dnn::impl
{
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
    
	Conv2D &&weights(Init arg_init) override
	{
		Layer<T>::weights(arg_init);
		return std::move(*this);
	}
    
	Conv2D &&biases(Init arg_init) override
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
		if (this->m_layer_level != Level::Input)
		{
    		this->m_temp.init(
    			std::get<0>(input_size), std::get<1>(input_size),
    			std::get<2>(input_size), std::get<3>(input_size)
    		);
		}
	}
    
	virtual void serialize(std::string base_file_name) override
	{
		skepu::external(this->label() + " serialize", skepu::read(this->m_weights, this->m_biases), [&]
		{
			std::ofstream ofs(base_file_name + "_" + this->label() + ".csv");
    
			for (size_t i = 0; i < this->m_weights.size_i(); i++)
				for (size_t j = 0; j < this->m_weights.size_j(); j++)
					for (size_t k = 0; k < this->m_weights.size_k(); k++)
						for (size_t l = 0; l < this->m_weights.size_l(); l++)
							ofs << this->m_weights(i,j,k,l) << "\n";
    
			for (size_t i = 0; i < this->m_biases.size_i(); i++)
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
		else if (this->m_init_weights == Init::Xavier_normal){
			int fan_in = this->m_weights.size_k() * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  int fan_out = this->m_no_features * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  double limit = std::sqrt(2.0 / (fan_in + fan_out));
    
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init Conv2D weights");
			skel_init_random_uniform(this->m_weights, 0.0, limit);
		}
		else if (this->m_init_weights == Init::He_uniform){
			int fan_in = this->m_weights.size_k() * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  // int fan_out = this->m_no_features * std::get<0>(this->m_kernel) * std::get<1>(this->m_kernel);
		  double limit = std::sqrt(6.0 / (fan_in));
    
			skel_init_random_uniform.setPRNG(*this->m_prng);
			skel_init_random_uniform.setLabel(this->label() + " Random-init Conv2D weights");
			skel_init_random_uniform(this->m_weights, -limit, limit);
		}
		else if (this->m_init_weights == Init::He_normal){
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
	//	dnn_debug << "Conv2D inputs: " << *inputs << "\n";
	//	dnn_debug << "Conv2D gradient: " << *gradient << "\n";
    
		if (this->m_layer_level != Level::Input)
		{
			skel_conv_2d_backward.setOverlap(0, (std::get<0>(this->m_kernel) - 1) / 2, (std::get<1>(this->m_kernel) - 1) / 2, /*this->m_no_features - 1*/ 0);
			skel_conv_2d_backward.setStride(1, std::get<0>(this->m_strides), std::get<1>(this->m_strides), 1); // incorrect - future work...
			skel_conv_2d_backward.setEdgeMode(skepu::Edge::Pad);
			skel_conv_2d_backward.setPad(0);
			skel_conv_2d_backward.setLabel(this->label() + " Conv2D backprop grad");
			skel_conv_2d_backward(this->m_temp, *gradient, this->m_weights);
		//	dnn_debug << "Conv2D temp: " << this->m_temp << "\n";
		}
    
		skel_propagate_conv_2d_weight_gradient.setLabel(this->label() + " Conv2D backprop weights");
		skel_propagate_conv_2d_bias_gradient.setLabel(this->label() + " Conv2D backprop biases");
		skel_propagate_conv_2d_weight_gradient(this->m_dW, *inputs, *gradient);
		skel_propagate_conv_2d_bias_gradient(this->m_db, *gradient);
	//	dnn_debug << "Conv2D dW: " << this->m_dW << "\n";
	//	dnn_debug << "Conv2D db: " << this->m_db << "\n";
    
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
}

namespace skepu::dnn
{
    template <typename T = float, typename... Args>
    impl::Conv2D<T> Conv2D(Args &&...args)
    {
	return impl::Conv2D<T>(std::forward<Args>(args)...);
    }
}

static skepu::PrecompilerMarker endOf_Conv2D_HPP;
#endif // SKEPU_PRECOMPILED
