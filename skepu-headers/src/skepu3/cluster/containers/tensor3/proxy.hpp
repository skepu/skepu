#pragma once
#ifndef SKEPU_STARPU_TENSOR3_PROXY_HPP
#define SKEPU_STARPU_TENSOR3_PROXY_HPP 1

#include <cstddef>

namespace skepu {

template<typename T>
struct Ten3
{
	T * data;
	size_t size_i;
	size_t size_j;
	size_t size_k;
	size_t size_jk;
	size_t size;

	Ten3()
	: data{nullptr}, size{0}
	{}

	Ten3(T * dataptr, size_t i, size_t j, size_t k)
	: data{dataptr},
		size_i{i},
		size_j{j},
		size_k{k},
		size_jk{j*k},
		size{i * size_jk}
	{}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t const index)
	-> T &
	{
		return this->data[index];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t const index) const
	-> T const &
	{
		return this->data[index];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t const i, size_t const j, size_t const k) noexcept
	-> T &
	{
		return data[(i * size_jk) + (j * size_k) + k];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t const i, size_t const j, size_t const k) const noexcept
	-> T const &
	{
		return data[(i * size_jk) + (j * size_k) + k];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator[](size_t const index)
	-> T &
	{
		return this->data[index];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator[](size_t const index) const
	-> T const &
	{
		return this->data[index];
	}
};

} // namespace skepu

#endif // SKEPU_STARPU_TENSOR3_PROXY_HPP
