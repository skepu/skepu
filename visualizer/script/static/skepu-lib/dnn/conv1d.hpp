#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_Conv1D_HPP;





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


auto skel_conv_1d = skepu::MapOverlap(convolutional_1d_uf);



namespace skepu::dnn::impl
{
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
    
	Conv1D &&weights(Init arg_init) override
	{
		Layer<T>::weights(arg_init);
		return std::move(*this);
	}
    
	Conv1D &&biases(Init arg_init) override
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
}

namespace skepu::dnn
{
    template <typename T = float, typename... Args>
    impl::Conv1D<T> Conv1D(Args &&...args)
    {
	return impl::Conv1D<T>(std::forward<Args>(args)...);
    }
}

static skepu::PrecompilerMarker endOf_Conv1D_HPP;
#endif // SKEPU_PRECOMPILED
