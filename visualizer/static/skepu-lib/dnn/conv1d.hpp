#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOf_Conv1D_HPP;

// 1D convolution and pooling for sequence/signal data.
//
// A sequence is stored as a Tensor4 of shape (batch, 1, length, channels), i.e. a "one row" image.
// Conv1D and MaxPooling1D are thin layers on top of Conv2D and MaxPooling2D with kernels of shape
// (1 x K), so they reuse (and are verified by the gradient checks of) the 2D forward and backward
// skeletons. They therefore have no user functions or skeletons of their own.
//
// Requires conv2d.hpp and pool2d.hpp to be included first.
//
// Example:
//   SequentialModel(Dimensions{batch, 1, length, channels})
//      << Conv1D(16).kernel(5).activation(Act::ReLU)
//      << MaxPooling1D().kernel(2)
//      << Flatten()
//      << Dense(num_classes).activation(Act::SoftMax)

namespace skepu::dnn::impl
{
	// Valid (unpadded) 1D convolution with stride 1: output length = input length - kernel + 1
	template <typename T = float>
	class Conv1D : public Conv2D<T>
	{
	public:
	Conv1D(int no_features) : Conv2D<T>(no_features)
	{
		this->m_kernel = {1, 3}; // default
	}

	Conv1D(Conv1D &&source) : Conv2D<T>(std::move(source)) {}

	~Conv1D() override = default;

	Conv1D &&kernel(size_t k)
	{
		this->m_kernel = {1, (int)k};
		return std::move(*this);
	}

	// Strides and padding are not supported by the underlying 2D implementation
	Conv1D &&kernel(size_t, size_t) = delete;
	Conv1D &&stride(size_t, size_t) = delete;
	Conv1D &&padding(bool) = delete;

	Conv1D &&weights(Init arg_init) override
	{
		Conv2D<T>::weights(arg_init);
		return std::move(*this);
	}

	Conv1D &&biases(Init arg_init) override
	{
		Conv2D<T>::biases(arg_init);
		return std::move(*this);
	}

	Conv1D &&activation(Act new_act) override
	{
		Conv2D<T>::activation(new_act);
		return std::move(*this);
	}

	Conv1D &&pretrained(std::string fname) override
	{
		Conv2D<T>::pretrained(fname);
		return std::move(*this);
	}

	std::string describe() const override
	{
		return "Conv1D features=" + std::to_string(this->m_no_features) + " kernel=" + std::to_string(std::get<1>(this->m_kernel))
			+ this->common_attributes(true);
	}

	void setup(Dimensions input_size) override
	{
		if (std::get<1>(input_size) != 1)
			SKEPU_ERROR("SkePU-DNN: Conv1D expects input shape (batch, 1, length, channels), got height " << std::get<1>(input_size));
		if (std::get<2>(input_size) < (size_t)std::get<1>(this->m_kernel))
			SKEPU_ERROR("SkePU-DNN: Conv1D kernel size " << std::get<1>(this->m_kernel) << " is larger than the input length " << std::get<2>(input_size));
		Conv2D<T>::setup(input_size);
	}

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "Conv1D layer with " << this->m_no_features << " features and kernel size " << std::get<1>(this->m_kernel);
		os << " train_params{ (" << this->m_weights.size_j() << " x " << this->m_weights.size_k() << " x " << this->m_weights.size_l() << ")"
				<< this->m_weights.size() << " + " << this->m_biases.size() << "}";
		return os.str();
	}
	};


	// Max pooling along the sequence dimension, with stride equal to the pool size
	template <typename T = float>
	class MaxPooling1D : public MaxPooling2D<T>
	{
	public:
	MaxPooling1D() : MaxPooling2D<T>()
	{
		this->m_kernel = {1, 2}; // default
	}

	MaxPooling1D(MaxPooling1D &&source) : MaxPooling2D<T>(std::move(source)) {}

	~MaxPooling1D() override = default;

	using MaxPooling2D<T>::kernel; // the two-argument form is valid too: (1 x k)

	MaxPooling1D &&kernel(size_t k)
	{
		this->m_kernel = {1, (int)k};
		return std::move(*this);
	}

	std::string describe() const override
	{
		return "MaxPooling1D kernel=" + std::to_string(std::get<1>(this->m_kernel)) + this->common_attributes(false);
	}

	void setup(Dimensions input_size) override
	{
		if (std::get<1>(input_size) != 1)
			SKEPU_ERROR("SkePU-DNN: MaxPooling1D expects input shape (batch, 1, length, channels), got height " << std::get<1>(input_size));
		MaxPooling2D<T>::setup(input_size);
	}

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "MaxPooling1D layer with pool size " << std::get<1>(this->m_kernel);
		return os.str();
	}
	};


	// Average pooling along the sequence dimension, with stride equal to the pool size
	template <typename T = float>
	class AvgPooling1D : public AvgPooling2D<T>
	{
	public:
	AvgPooling1D() : AvgPooling2D<T>()
	{
		this->m_kernel = {1, 2}; // default
	}

	AvgPooling1D(AvgPooling1D &&source) : AvgPooling2D<T>(std::move(source)) {}

	~AvgPooling1D() override = default;

	using AvgPooling2D<T>::kernel; // the two-argument form is valid too: (1 x k)

	AvgPooling1D &&kernel(size_t k)
	{
		this->m_kernel = {1, (int)k};
		return std::move(*this);
	}

	std::string describe() const override
	{
		return "AvgPooling1D kernel=" + std::to_string(std::get<1>(this->m_kernel)) + this->common_attributes(false);
	}

	void setup(Dimensions input_size) override
	{
		if (std::get<1>(input_size) != 1)
			SKEPU_ERROR("SkePU-DNN: AvgPooling1D expects input shape (batch, 1, length, channels), got height " << std::get<1>(input_size));
		AvgPooling2D<T>::setup(input_size);
	}

	std::string summary() override
	{
		std::stringstream os;
		os << Layer<T>::summary();
		os << "AvgPooling1D layer with pool size " << std::get<1>(this->m_kernel);
		return os.str();
	}
	};
}

namespace skepu::dnn
{
	template <typename T = float, typename... Args>
	impl::Conv1D<T> Conv1D(Args &&...args)
	{
	return impl::Conv1D<T>(std::forward<Args>(args)...);
	}

	template <typename T = float, typename... Args>
	impl::MaxPooling1D<T> MaxPooling1D(Args &&...args)
	{
	return impl::MaxPooling1D<T>(std::forward<Args>(args)...);
	}

	template <typename T = float, typename... Args>
	impl::AvgPooling1D<T> AvgPooling1D(Args &&...args)
	{
	return impl::AvgPooling1D<T>(std::forward<Args>(args)...);
	}
}

static skepu::PrecompilerMarker endOf_Conv1D_HPP;
#endif // SKEPU_PRECOMPILED
