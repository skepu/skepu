#include <algorithm>

#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

static const char *ReduceKernelTemplate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}({{REDUCE_RESULT_TYPE}} *skepu_input, {{REDUCE_RESULT_TYPE}} *skepu_output, size_t skepu_n, size_t skepu_blockSize, bool skepu_nIsPow2)
{
	extern __shared__ {{REDUCE_RESULT_TYPE}} {{SHARED_BUFFER}}[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	size_t skepu_tid = threadIdx.x;
	size_t skepu_i = blockIdx.x * skepu_blockSize * 2 + threadIdx.x;
	size_t skepu_gridSize = skepu_blockSize * 2 * gridDim.x;

	{{REDUCE_RESULT_TYPE}} skepu_result;

	if (skepu_i < skepu_n)
	{
		skepu_result = skepu_input[skepu_i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		// This nIsPow2 opt is not valid when we use this kernel for sparse matrices as well where we
		// dont exactly now the elements when calculating thread- and block-size and nIsPow2 assum becomes invalid in some cases there which results in sever problems.
		// There we pass it always false
		if (skepu_nIsPow2 || skepu_i + skepu_blockSize < skepu_n)
			skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_input[skepu_i + skepu_blockSize]);
		skepu_i += skepu_gridSize;
	}

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (skepu_i < skepu_n)
	{
		skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_input[skepu_i]);
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (skepu_nIsPow2 || skepu_i + skepu_blockSize < skepu_n)
			skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_input[skepu_i + skepu_blockSize]);
		skepu_i += skepu_gridSize;
	}

	// each thread puts its local sum into shared memory
	{{SHARED_BUFFER}}[skepu_tid] = skepu_result;
	__syncthreads();

	// do reduction in shared mem
	if (skepu_blockSize >= 512) { if (skepu_tid < 256) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid + 256]); } __syncthreads(); }
	if (skepu_blockSize >= 256) { if (skepu_tid < 128) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid + 128]); } __syncthreads(); }
	if (skepu_blockSize >= 128) { if (skepu_tid <  64) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid +  64]); } __syncthreads(); }

	if (skepu_tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		// UPDATE: volatile causes issues with custom struct data types; use __syncwarp() instead
		/*volatile*/ {{REDUCE_RESULT_TYPE}}* skepu_smem = {{SHARED_BUFFER}};
		if (skepu_blockSize >=  64) { if (skepu_tid < 32) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid + 32]); } __syncwarp(); }
		if (skepu_blockSize >=  32) { if (skepu_tid < 16) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid + 16]); } __syncwarp(); }
		if (skepu_blockSize >=  16) { if (skepu_tid <  8) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  8]); } __syncwarp(); }
		if (skepu_blockSize >=   8) { if (skepu_tid <  4) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  4]); } __syncwarp(); }
		if (skepu_blockSize >=   4) { if (skepu_tid <  2) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  2]); } __syncwarp(); }
		if (skepu_blockSize >=   2) { if (skepu_tid <  1) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  1]); } __syncwarp(); }
	}

	// write result for this block to global mem
	if (skepu_tid == 0)
		skepu_output[blockIdx.x] = {{SHARED_BUFFER}}[0];
}
)~~~";


std::string createReduce1DKernelProgram_CU(SkeletonInstance &instance, UserFunction &reduceFunc, std::string dir)
{
	const std::string kernelName = ResultName + "_ReduceKernel_" + reduceFunc.uniqueName;
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(ReduceKernelTemplate_CU,
	{
		{"{{REDUCE_RESULT_TYPE}}",   reduceFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName},
		{"{{FUNCTION_NAME_REDUCE}}", reduceFunc.funcNameCUDA()},
		{"{{SHARED_BUFFER}}",        "sdata_" + instance}
	});
	return kernelName;
}


std::string createReduce2DKernelProgram_CU(SkeletonInstance &instance, UserFunction &rowWiseFunc, UserFunction &colWiseFunc, std::string dir)
{
	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_ReduceKernel_" + rowWiseFunc.uniqueName + "_" + colWiseFunc.uniqueName;
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(ReduceKernelTemplate_CU,
	{
		{"{{REDUCE_RESULT_TYPE}}",   rowWiseFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName + "_RowWise"},
		{"{{FUNCTION_NAME_REDUCE}}", rowWiseFunc.funcNameCUDA()},
		{"{{SHARED_BUFFER}}",        "sdata_" + instance}
	});
	FSOutFile << templateString(ReduceKernelTemplate_CU,
	{
		{"{{REDUCE_RESULT_TYPE}}",   colWiseFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName + "_ColWise"},
		{"{{FUNCTION_NAME_REDUCE}}", colWiseFunc.funcNameCUDA()},
		{"{{SHARED_BUFFER}}",        "sdata_" + instance}
	});
	return kernelName;
}
