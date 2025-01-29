#pragma once
#ifndef SKEPU_STARPU_TENSOR4_PROXY_HPP
#define SKEPU_STARPU_TENSOR4_PROXY_HPP 1

#include <cstddef>

namespace skepu {

template<typename T>
struct Ten4
{
	T * data;
	size_t size_i;
	size_t size_j;
	size_t size_k;
	size_t size_l;

	size_t size_kl;
	size_t size_jkl;

	size_t size;

	Ten4()
	: data{nullptr},
		size_i{0},
		size_j{0},
		size_k{0},
		size_l{0},
		size_kl{0},
		size_jkl{0},
		size{0}
	{}

	Ten4(T * dataptr, size_t i, size_t j, size_t k, size_t l)
	: data{dataptr},
		size_i{i},
		size_j{j},
		size_k{k},
		size_l{l},
		size_kl{k*l},
		size_jkl{j*k*l},
		size{i*j*k*l}
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
	operator()(
		size_t const i, size_t const j, size_t const k, size_t const l) noexcept
	-> T &
	{
		return data[(i * size_jkl) + (j * size_kl) + (k * size_l) + l];
	}

	#ifdef SKEPU_CUDA
		__host__ __device__
	#endif
	auto
	operator()(
		size_t const i,
		size_t const j,
		size_t const k,
		size_t const l) const noexcept
	-> T const &
	{
		return data[(i * size_jkl) + (j * size_kl) + (k * size_l) + l];
	}
};

} // namespace skepu

#endif // SKEPU_STARPU_TENSOR4_PROXY_HPP
