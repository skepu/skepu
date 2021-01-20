#include "code_gen.h"
#include "code_gen_cu.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapReduceKernelTemplate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}({{KERNEL_PARAMS}} size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base)
{
	extern __shared__ {{REDUCE_RESULT_TYPE}} skepu_sdata[];

	size_t skepu_blockSize = blockDim.x;
	size_t skepu_tid = threadIdx.x;
	size_t skepu_i = blockIdx.x * skepu_blockSize + skepu_tid;
	size_t skepu_gridSize = skepu_blockSize * gridDim.x;
	{{REDUCE_RESULT_TYPE}} skepu_result;

	if (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		skepu_result = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
		//{{OUTPUT_BINDINGS}}
		skepu_i += skepu_gridSize;
	}

	while (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		auto skepu_tempMap = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
		skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_tempMap);
		skepu_i += skepu_gridSize;
	}

	skepu_sdata[skepu_tid] = skepu_result;
	__syncthreads();

	if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 512]); } __syncthreads(); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 256]); } __syncthreads(); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 128]); } __syncthreads(); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid +  64 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  64]); } __syncthreads(); }
	if (skepu_blockSize >=   64) { if (skepu_tid <  32 && skepu_tid +  32 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  32]); } __syncthreads(); }
	if (skepu_blockSize >=   32) { if (skepu_tid <  16 && skepu_tid +  16 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  16]); } __syncthreads(); }
	if (skepu_blockSize >=   16) { if (skepu_tid <   8 && skepu_tid +   8 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   8]); } __syncthreads(); }
	if (skepu_blockSize >=    8) { if (skepu_tid <   4 && skepu_tid +   4 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   4]); } __syncthreads(); }
	if (skepu_blockSize >=    4) { if (skepu_tid <   2 && skepu_tid +   2 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   2]); } __syncthreads(); }
	if (skepu_blockSize >=    2) { if (skepu_tid <   1 && skepu_tid +   1 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   1]); } __syncthreads(); }

	if (skepu_tid == 0)
		skepu_output[blockIdx.x] = skepu_sdata[skepu_tid];
}
)~~~";

const char *ReduceKernelTemplate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}({{REDUCE_RESULT_TYPE}} *skepu_input, {{REDUCE_RESULT_TYPE}} *skepu_output, size_t skepu_n, size_t skepu_blockSize, bool skepu_nIsPow2)
{
	extern __shared__ {{REDUCE_RESULT_TYPE}} skepu_sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	size_t skepu_tid = threadIdx.x;
	size_t skepu_i = blockIdx.x * skepu_blockSize*2 + threadIdx.x;
	size_t skepu_gridSize = skepu_blockSize * 2 * gridDim.x;
	{{REDUCE_RESULT_TYPE}} skepu_result;

	if(skepu_i < skepu_n)
	{
		skepu_result = skepu_input[skepu_i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		//This nIsPow2 opt is not valid when we use this kernel for sparse matrices as well where we
		// dont exactly now the elements when calculating thread- and block-size and nIsPow2 assum becomes invalid in some cases there which results in sever problems.
		// There we pass it always false
		if (skepu_nIsPow2 || skepu_i + skepu_blockSize < skepu_n)
			skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_input[skepu_i + skepu_blockSize]);
		skepu_i += skepu_gridSize;
	}

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a lParamer gridSize and therefore fewer elements per thread
	while(skepu_i < skepu_n)
	{
		skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_input[skepu_i]);
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (skepu_nIsPow2 || skepu_i + skepu_blockSize < skepu_n)
			skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_input[skepu_i + skepu_blockSize]);
		skepu_i += skepu_gridSize;
	}

	// each thread puts its local sum into shared memory
	skepu_sdata[skepu_tid] = skepu_result;
	__syncthreads();

	// do reduction in shared mem
	if (skepu_blockSize >= 1024) { if (skepu_tid < 512) { skepu_sdata[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_sdata[skepu_tid + 512]); } __syncthreads(); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256) { skepu_sdata[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_sdata[skepu_tid + 256]); } __syncthreads(); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128) { skepu_sdata[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_sdata[skepu_tid + 128]); } __syncthreads(); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64) { skepu_sdata[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_sdata[skepu_tid +  64]); } __syncthreads(); }

	if (skepu_tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile {{REDUCE_RESULT_TYPE}}* skepu_smem = skepu_sdata;
		if (skepu_blockSize >=  64) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid + 32]); }
		if (skepu_blockSize >=  32) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid + 16]); }
		if (skepu_blockSize >=  16) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  8]); }
		if (skepu_blockSize >=   8) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  4]); }
		if (skepu_blockSize >=   4) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  2]); }
		if (skepu_blockSize >=   2) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  1]); }
	}

	// write result for this block to global mem
	if (skepu_tid == 0)
		skepu_output[blockIdx.x] = skepu_sdata[0];
}
)~~~";


std::string createMapReduceKernelProgram_CU(UserFunction &mapFunc, UserFunction &reduceFunc, size_t arity, std::string dir)
{
	std::stringstream sourceStream, SSKernelParamList, SSMapFuncArgs;
	IndexCodeGen indexInfo = indexInitHelper_CU(mapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CU(mapFunc, SSKernelParamList);

	for (UserFunction::Param& param : mapFunc.elwiseParams)
	{
		if (!first) { SSMapFuncArgs << ", "; }
		SSKernelParamList << param.resolvedTypeName << " *" << param.name << ", ";
		SSMapFuncArgs << param.name << "[skepu_i]";
		first = false;
	}
	auto argsInfo = handleRandomAccessAndUniforms_CU(mapFunc, SSMapFuncArgs, SSKernelParamList, first);

	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_MapReduceKernel_" + mapFunc.uniqueName + "_" + reduceFunc.uniqueName;
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(MapReduceKernelTemplate_CU,
	{
		{"{{REDUCE_RESULT_TYPE}}",   reduceFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName},
		{"{{FUNCTION_NAME_MAP}}",    mapFunc.funcNameCUDA()},
		{"{{FUNCTION_NAME_REDUCE}}", reduceFunc.funcNameCUDA()},
		{"{{KERNEL_PARAMS}}",        SSKernelParamList.str()},
		{"{{MAP_ARGS}}",             SSMapFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}",    indexInfo.indexInit},
		{"{{OUTPUT_BINDINGS}}",      multiOutputAssign},
		{"{{PROXIES_UPDATE}}",       argsInfo.proxyInitializerInner},
		{"{{PROXIES_INIT}}",         argsInfo.proxyInitializer}
	});
	FSOutFile << templateString(ReduceKernelTemplate_CU,
	{
		{"{{REDUCE_RESULT_TYPE}}",   reduceFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName + "_ReduceOnly"},
		{"{{FUNCTION_NAME_REDUCE}}", reduceFunc.funcNameCUDA()}
	});
	return kernelName;
}
