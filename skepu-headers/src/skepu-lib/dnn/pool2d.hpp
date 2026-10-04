#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_Pool2D_HPP;




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



auto skel_pool_max = skepu::MapPool(max_pooling_uf);
auto skel_pool_max_masked = skepu::MapPool(max_pooling_masked_uf);

auto skel_pool_max_backward = skepu::Map<0>(pool_max_backward_uf);
auto skel_pool_max_masked_backward = skepu::Map<0>(pool_max_masked_backward_uf);


namespace skepu::dnn::impl
{
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
		if (this->m_layer_level != Level::Input)
		{
    		this->m_temp.init(
    			std::get<0>(input_size), std::get<1>(input_size),
    			std::get<2>(input_size), std::get<3>(input_size)
    		);
		}
    
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
}

namespace skepu::dnn
{
    template <typename T = float, typename... Args>
    impl::MaxPooling2D<T> MaxPooling2D(Args &&...args)
    {
	return impl::MaxPooling2D<T>(std::forward<Args>(args)...);
    }
}

static skepu::PrecompilerMarker endOf_Pool2D_HPP;
#endif // SKEPU_PRECOMPILED
