#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_Pool2D_HPP;


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

	// The mask stores j * pool_size_k + k of the maximum within the pool window
	size_t max_idx_j = small_j * pool_size_j + (size_t)(maskval / pool_size_k);
	size_t max_idx_k = small_k * pool_size_k + (size_t)(maskval % pool_size_k);

	// If this index is the index of the max value, the gradient is 1
	// and implicitly multiplied with the incoming gradient
	if (index.j == max_idx_j && index.k == max_idx_k)
	return gradient(index.i, small_j, small_k, index.l);

	// Otherwise, the element did not contribute to the forward-output and its gradient is 0
	else return 0;
}

skepu::multiple<float, char>
max_pooling_masked_uf(skepu::Pool4D<float> DNN_CONST pool)
{
	float maxval = pool(0,0,0,0);
	char mask = 0;
	for (int j = 0; j < pool.sj; ++j){
		for (int k = 0; k < pool.sk; ++k)
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
	for (int j = 0; j < pool.sj; ++j)
		for (int k = 0; k < pool.sk; ++k)
			sum += pool(0, j, k, 0);
	return sum / (pool.sj * pool.sk);
}


// Gradient of average pooling: every input element gets the gradient of its pool divided by the pool size;
// elements outside the pooled area (input size not divisible by the pool size) get none.
float pool_avg_backward_uf(
	skepu::Index4D index,
	skepu::Ten4<float> gradient,
	size_t pool_size_j,
	size_t pool_size_k
)
{
	if (index.j >= gradient.size_j * pool_size_j || index.k >= gradient.size_k * pool_size_k)
		return 0;

	return gradient(index.i, index.j / pool_size_j, index.k / pool_size_k, index.l) / (float)(pool_size_j * pool_size_k);
}


auto skel_pool_max_masked = skepu::MapPool(max_pooling_masked_uf);
auto skel_pool_avg = skepu::MapPool(avg_pooling_uf);

auto skel_pool_max_masked_backward = skepu::Map<0>(pool_max_masked_backward_uf);
auto skel_pool_avg_backward = skepu::Map<0>(pool_avg_backward_uf);


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
		this->m_stride_requested = source.m_stride_requested;
		this->m_has_stride_request = source.m_has_stride_request;
		this->m_vals = std::move(source.m_vals);
	}

	~Subsampling2D() override = default;

	void setup(Dimensions input_size) override
	{
		if (this->m_has_stride_request && this->m_stride_requested != this->m_kernel)
			SKEPU_ERROR("SkePU-DNN: pooling supports only a stride equal to the pool size, got ("
				<< std::get<0>(this->m_stride_requested) << ", " << std::get<1>(this->m_stride_requested) << ") for pool size ("
				<< std::get<0>(this->m_kernel) << ", " << std::get<1>(this->m_kernel) << ")");

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

	// The stride always equals the pool size and the pooling is "valid" (no padding). Other values are
	// rejected (the stride is checked in setup(), when the kernel size is known) instead of being ignored.
	virtual Subsampling2D &&stride(size_t s1, size_t s2)
	{
		this->m_stride_requested = {(int)s1, (int)s2};
		this->m_has_stride_request = true;
		return std::move(*this);
	}

	virtual Subsampling2D &&padding(bool new_padding)
	{
		if (new_padding)
			SKEPU_ERROR("SkePU-DNN: pooling supports only 'valid' mode, padding is not implemented");
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
	std::tuple<int, int> m_stride_requested{1, 1};
	bool m_has_stride_request = false;
	};


	template <typename T = float>
	class MaxPooling2D : public Subsampling2D<T>
	{
	public:
	MaxPooling2D() : Subsampling2D<T>() {}

	MaxPooling2D(MaxPooling2D &&source)
			: Subsampling2D<T>(std::move(source))
	{
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
		this->m_mask.setLabel(this->label() + " MaxPool2D mask");
		this->m_mask.init(
			std::get<0>(this->m_output_size), std::get<1>(this->m_output_size),
			std::get<2>(this->m_output_size), std::get<3>(this->m_output_size)
		);
	}

	std::string describe() const override
	{
		return "MaxPooling2D kernel=" + std::to_string(std::get<0>(this->m_kernel)) + "x" + std::to_string(std::get<1>(this->m_kernel))
			+ this->common_attributes(false);
	}

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "MaxPooling2D layer with kernel size ";
		os << "{" << std::get<0>(this->m_kernel) << ", " << std::get<1>(this->m_kernel) << "}";
		return os.str();
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool /*is_training*/=false) override
	{
		skel_pool_max_masked.setPoolSize(1, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel), 1);
		// todo: override with set strides
		skel_pool_max_masked.setStride(1, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel), 1);
		skel_pool_max_masked.setLabel(this->label() + " MaxPool2D forward");
		skel_pool_max_masked(this->m_vals, this->m_mask, *inputs);

		return &this->m_vals;
	}

	Tensor4<T> *backward(Tensor4<T> */*inputs*/, Tensor4<T> *gradient, Tensor4<T> */*y*/) override
	{
		skel_pool_max_masked_backward.setLabel(this->label() + " MaxPool2D backward");
		skel_pool_max_masked_backward(this->m_temp, *gradient, this->m_mask, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel));

		return &this->m_temp;
	}

	private:

	Tensor4<T> m_temp;
	Tensor4<char> m_mask;
	};
}

namespace skepu::dnn::impl
{
	template <typename T = float>
	class AvgPooling2D : public Subsampling2D<T>
	{
	public:
	AvgPooling2D() : Subsampling2D<T>() {}

	AvgPooling2D(AvgPooling2D &&source)
			: Subsampling2D<T>(std::move(source))
	{
	}

	~AvgPooling2D() override = default;

	AvgPooling2D &&kernel(size_t k1, size_t k2) override
	{
		Subsampling2D<T>::kernel(k1, k2);
		return std::move(*this);
	}

	AvgPooling2D &&stride(size_t s1, size_t s2) override
	{
		Subsampling2D<T>::stride(s1, s2);
		return std::move(*this);
	}

	AvgPooling2D &&padding(bool new_padding) override
	{
		Subsampling2D<T>::padding(new_padding);
		return std::move(*this);
	}

	void setup(Dimensions input_size) override
	{
		this->m_vals.setLabel(this->label() + " AvgPool2D values");
		this->m_temp.setLabel(this->label() + " AvgPool2D gradients");

		Subsampling2D<T>::setup(input_size);

		// Backprop temporary
		if (this->m_layer_level != Level::Input)
		{
			this->m_temp.init(
				std::get<0>(input_size), std::get<1>(input_size),
				std::get<2>(input_size), std::get<3>(input_size)
			);
		}
	}

	std::string describe() const override
	{
		return "AvgPooling2D kernel=" + std::to_string(std::get<0>(this->m_kernel)) + "x" + std::to_string(std::get<1>(this->m_kernel))
			+ this->common_attributes(false);
	}

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "AvgPooling2D layer with kernel size ";
		os << "{" << std::get<0>(this->m_kernel) << ", " << std::get<1>(this->m_kernel) << "}";
		return os.str();
	}

	Tensor4<T> *forward(Tensor4<T> *inputs, bool /*is_training*/=false) override
	{
		skel_pool_avg.setPoolSize(1, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel), 1);
		skel_pool_avg.setStride(1, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel), 1);
		skel_pool_avg.setLabel(this->label() + " AvgPool2D forward");
		skel_pool_avg(this->m_vals, *inputs);

		return &this->m_vals;
	}

	Tensor4<T> *backward(Tensor4<T> */*inputs*/, Tensor4<T> *gradient, Tensor4<T> */*y*/) override
	{
		skel_pool_avg_backward.setLabel(this->label() + " AvgPool2D backward");
		skel_pool_avg_backward(this->m_temp, *gradient, std::get<0>(this->m_kernel), std::get<1>(this->m_kernel));

		return &this->m_temp;
	}

	private:

	Tensor4<T> m_temp;
	};
}

namespace skepu::dnn
{
	template <typename T = float, typename... Args>
	impl::AvgPooling2D<T> AvgPooling2D(Args &&...args)
	{
	return impl::AvgPooling2D<T>(std::forward<Args>(args)...);
	}
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
