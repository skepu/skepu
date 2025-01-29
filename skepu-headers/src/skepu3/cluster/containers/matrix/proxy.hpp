#pragma once
#ifndef SKEPU_STARPU_MATRIX_PROXY_HPP
#define SKEPU_STARPU_MATRIX_PROXY_HPP 1

#include <skepu3/cluster/common.hpp>

namespace skepu {

template<typename T>
struct Mat
{
	T * data;
	size_t rows;
	size_t cols;

	Mat() : data{0}, rows{0}, cols{0} {}

	Mat(T * data, size_t rows, size_t cols)
	: data{data}, rows{rows}, cols{cols}
	{}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t pos) noexcept
	-> T &
	{
		return this->data[pos];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t pos) const noexcept
	-> T const &
	{
		return this->data[pos];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t i, size_t j) noexcept
	-> T &
	{
		return this->data[i * this->cols + j];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t i, size_t j) const noexcept
	-> T const &
	{
		return this->data[i * this->cols + j];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator[](size_t pos) noexcept
	-> T &
	{
		return this->data[pos];
	}

	auto
	operator[](size_t pos) const noexcept
	-> T const &
	{
		return this->data[pos];
	}
};

template<typename T>
struct MatRow
{
	T * data;
	size_t cols;

	MatRow() : data{0}, cols{0} {}

	#ifdef SKEPU_CUDA
	__host__ __device__
	#endif
	MatRow(T * data, size_t cols)
	: data{data}, cols{cols}
	{}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t pos) noexcept
	-> T &
	{
		return data[pos];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t pos) const noexcept
	-> T const &
	{
		return data[pos];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator[](size_t pos) noexcept
	-> T &
	{
		return data[pos];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator[](size_t pos) const noexcept
	-> T const &
	{
		return data[pos];
	}
};

template<typename T>
struct proxy_tag<MatRow<T>>
{
	typedef ProxyTag::MatRow type;
};

} // namespace skepu

#endif // SKEPU_STARPU_MATRIX_PROXY_HPP
