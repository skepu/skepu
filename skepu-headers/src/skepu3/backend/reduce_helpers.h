/*! \file reduce_kernels.h
 *  \brief Contains the OpenCL and CUDA kernels for the Reduce skeleton (used for both 1D and 2D reduce operation).
 */

#ifndef REDUCE_KERNELS_H
#define REDUCE_KERNELS_H


namespace skepu
{
namespace backend
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup ReduceKernels Reduce Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Reduce skeleton.
 * \{
 */



/*!
 * \brief A small helper to determine whether the number is a power of 2.
 *
 * \param x the actual number.
 * \return bool specifying whether number of power of 2 or not,
 */
inline bool isPow2(size_t x)
{
   return ((x&(x-1))==0);
}


/*!
 * \brief A helper to return a value that is nearest value that is power of 2.
 *
 * \param x The input number for which we need to find the nearest value that is power of 2.
 * \return The nearest value that is power of 2.
 */
inline size_t nextPow2(size_t x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}




/*!
 * Compute the number of threads and blocks to use for the reduction kernel.
 * We set threads / block to the minimum of maxThreads and n/2 where n is
 * problem size. We observe the maximum specified number of blocks, because
 * each kernel thread can process more than 1 elements.
 *
 * \param n Problem size.
 * \param maxBlocks Maximum number of blocks that can be used.
 * \param maxThreads Maximum number of threads that can be used.
 */
inline std::pair<size_t, size_t> getNumBlocksAndThreads(size_t n, size_t maxBlocks, size_t maxThreads)
{
	const size_t threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	const size_t blocks = std::min(maxBlocks, (n + (threads * 2 - 1)) / (threads * 2));
	return std::make_pair(threads, blocks);
}



#ifdef SKEPU_CUDA

/*!
 * Reduces the n partial results in d_data (one for each block of a first reduction kernel) to one, in d_data[0].
 * A kernel that reduces its input with several blocks must not write the output where it reads the input, as the
 * blocks are not synchronized: the output element of a block that finishes early may be one that another block
 * has not read yet. Passes with several blocks therefore go from one buffer to another; a pass with one block
 * can work in place.
 *
 * \param n The number of partial results.
 * \param launch Launches one pass: launch(size, numThreads, numBlocks, input, output).
 * \param stream The stream of the launches, for the copy of the result and the synchronization.
 */
template <typename T, typename Launch>
void ReducePartialResultsOnDevice(size_t n, size_t maxBlocks, size_t maxThreads, T *d_data, cudaStream_t stream, Launch launch)
{
	T *src = d_data, *tmp = nullptr;
	size_t threads, blocks;
	
	for (size_t s = n; s > 1; s = (s + (threads * 2 - 1)) / (threads * 2))
	{
		std::tie(threads, blocks) = getNumBlocksAndThreads(s, maxBlocks, maxThreads);
		
		if (blocks == 1)
			launch(s, threads, blocks, src, src);
		else
		{
			// (the other buffer is the one that this pass does not read: d_data at the first pass, then each in turn)
			T *dst = (src == d_data) ? tmp : d_data;
			if (src == d_data && !tmp)
			{
				CHECK_CUDA_ERROR(cudaMalloc(&tmp, sizeof(T) * n));
				dst = tmp;
			}
			launch(s, threads, blocks, src, dst);
			src = dst;
		}
	}
	
	if (src != d_data)
		CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, src, sizeof(T), cudaMemcpyDeviceToDevice, stream));
	if (tmp)
	{
		CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
		CHECK_CUDA_ERROR(cudaFree(tmp));
	}
}

/*!
 * Helper method used to call the actual CUDA kernel for reduction. Used when PINNED MEMORY is disabled
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param size size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template <typename ReduceFunc, typename T>
void CallReduceKernel(ReduceFunc kernel, size_t size, size_t numThreads, size_t numBlocks, T *d_idata, T *d_odata, bool enableIsPow2=true)
{
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	
	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	size_t smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
	
	// choose which of the optimized versions of reduction to launch
	if (isPow2(size) && enableIsPow2)
	{
		switch (numThreads)
		{
		case 512: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 512, true); break;
		case 256: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 256, true); break;
		case 128: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 128, true); break;
		case  64: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  64, true); break;
		case  32: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  32, true); break;
		case  16: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  16, true); break;
		case   8: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   8, true); break;
		case   4: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   4, true); break;
		case   2: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   2, true); break;
		case   1: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   1, true); break;
		}
	}
	else
	{
		switch (numThreads)
		{
		case 512: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 512, false); break;
		case 256: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 256, false); break;
		case 128: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 128, false); break;
		case  64: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  64, false); break;
		case  32: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  32, false); break;
		case  16: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  16, false); break;
		case   8: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   8, false); break;
		case   4: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   4, false); break;
		case   2: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   2, false); break;
		case   1: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   1, false); break;
		}
	}
}




#ifdef USE_PINNED_MEMORY
/*!
 * Helper method used to call the actual CUDA kernel for reduction. Used when PINNED MEMORY is enabled
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param size size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param stream CUDA stream to be used.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template<typename ReduceFunc, typename T>
void CallReduceKernel_WithStream(ReduceFunc kernel, size_t size, size_t numThreads, size_t numBlocks, T *d_idata, T *d_odata, cudaStream_t &stream, bool enableIsPow2=true)
{
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	
	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	size_t smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
	
	// choose which of the optimized versions of reduction to launch
	if (isPow2(size) && enableIsPow2)
	{
		switch (numThreads)
		{
		case 512: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 512, true); break;
		case 256: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 256, true); break;
		case 128: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 128, true); break;
		case  64: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  64, true); break;
		case  32: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  32, true); break;
		case  16: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  16, true); break;
		case   8: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   8, true); break;
		case   4: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   4, true); break;
		case   2: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   2, true); break;
		case   1: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   1, true); break;
		}
	}
	else
	{
		switch (numThreads)
		{
		case 512: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 512, false); break;
		case 256: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 256, false); break;
		case 128: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 128, false); break;
		case  64: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  64, false); break;
		case  32: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  32, false); break;
		case  16: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  16, false); break;
		case   8: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   8, false); break;
		case   4: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   4, false); break;
		case   2: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   2, false); break;
		case   1: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   1, false); break;
		}
	}
}
#endif



#ifdef USE_PINNED_MEMORY
/*!
 *  A helper function that is used to call the actual kernel for reduction. Used by other functions to call the actual kernel
 *  Internally, it just calls 2 kernels by setting their arguments. No synchronization is enforced.
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param n size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param maxThreads Maximum number of threads that can be used for kernel execution.
 *  \param maxBlocks Maximum number of blocks that can be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param stream CUDA stream to be used, only when using Pinned memory allocations.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template <typename ReduceFunc, typename T>
void ExecuteReduceOnADevice(ReduceFunc kernel, size_t n, size_t  numThreads, size_t  numBlocks, size_t  maxThreads, size_t  maxBlocks, T* d_idata, T* d_odata, unsigned int deviceID, cudaStream_t &stream, bool enableIsPow2 = true)
#else
/*!
 *  A helper function that is used to call the actual kernel for reduction. Used by other functions to call the actual kernel
 *  Internally, it just calls 2 kernels by setting their arguments. No synchronization is enforced.
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param n size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param maxThreads Maximum number of threads that can be used for kernel execution.
 *  \param maxBlocks Maximum number of blocks that can be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template <typename T, typename ReduceFunc>
void ExecuteReduceOnADevice(ReduceFunc kernel, size_t  n, size_t  numThreads, size_t  numBlocks, size_t  maxThreads, size_t  maxBlocks, T* d_idata, T* d_odata, unsigned int deviceID, bool enableIsPow2 = true)
#endif
{
	// execute the kernel
#ifdef USE_PINNED_MEMORY
	CallReduceKernel_WithStream(kernel, n, numThreads, numBlocks, d_idata, d_odata, stream, enableIsPow2);
#else
	CallReduceKernel(kernel, n, numThreads, numBlocks, d_idata, d_odata, enableIsPow2);
#endif
	
	// sum partial block sums on GPU
#ifdef USE_PINNED_MEMORY
	ReducePartialResultsOnDevice(numBlocks, maxBlocks, maxThreads, d_odata, stream,
		[&](size_t s, size_t t, size_t b, T *in, T *out) { CallReduceKernel_WithStream(kernel, s, t, b, in, out, stream, enableIsPow2); });
#else
	ReducePartialResultsOnDevice(numBlocks, maxBlocks, maxThreads, d_odata, 0,
		[&](size_t s, size_t t, size_t b, T *in, T *out) { CallReduceKernel(kernel, s, t, b, in, out, enableIsPow2); });
#endif
}

#endif

/*!
 *  \}
 */

}
}


#endif

