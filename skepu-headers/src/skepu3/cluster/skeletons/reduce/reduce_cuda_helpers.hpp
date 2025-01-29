#pragma once
#ifndef SKEPU_REDUCE_CUDA_HELPERS_HPP
#define SKEPU_REDUCE_CUDA_HELPERS_HPP 1

#include <tuple>

#ifdef SKEPU_CUDA
#include <cuda.h>
#endif // SKEPU_CUDA

#include <starpu.h>

namespace skepu {
namespace util {

size_t constexpr static max_threads{1024};

#ifdef SKEPU_CUDA
auto inline
is_pow2(size_t x) noexcept
-> bool
{
   return ((x&(x-1))==0);
}

auto inline
next_pow2(size_t x) noexcept
-> size_t
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

template<typename T>
auto inline
generate_cuda_params(size_t count) noexcept
-> std::tuple<size_t, size_t, size_t>
{
	size_t threads =
		count < max_threads*2
			? next_pow2((count + 1)/ 2)
			: max_threads;
	size_t blocks = std::min(((count -1)/ (threads *2)) +1, 2*max_threads);
	size_t smem =
		threads < 32
			? 2*threads*sizeof(T)
			: threads*sizeof(T);

	return std::make_tuple(threads, blocks, smem);
}
#endif // SKEPU_CUDA

} // namespace util
} // namespace skepu


#endif // SKEPU_REDUCE_CUDA_HELPERS_HPP
