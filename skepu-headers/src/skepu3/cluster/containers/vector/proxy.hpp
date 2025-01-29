#pragma once
#ifndef SKEPU_STARPU_VECTOR_PROXY_HPP
#define SKEPU_STARPU_VECTOR_PROXY_HPP 1

#include <cstddef>

namespace skepu {

template<typename T>
struct Vec
{
	T * data;
	size_t size;

	Vec()
	: data{nullptr}, size{0}
	{}

	Vec(T *dataptr, size_t sizearg)
	: data{dataptr}, size{sizearg}
	{}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t index)
	-> T &
	{
		return this->data[index];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(size_t index) const
	-> T const &
	{
		return this->data[index];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator[](size_t index)
	-> T &
	{
		return this->data[index];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator[](size_t index) const
	-> T const &
	{
		return this->data[index];
	}
};

} // namespace skepu

#endif // SKEPU_STARPU_VECTOR_PROXY_HPP
