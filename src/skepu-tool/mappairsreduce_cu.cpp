#include "code_gen.h"
#include "code_gen_cu.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapPairsReduceKernelTemplate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}({{KERNEL_PARAMS}} size_t skepu_Vsize, size_t skepu_Hsize, size_t skepu_base, bool skepu_transposed)
{
	extern __shared__ {{REDUCE_RESULT_TYPE}} {{SHARED_BUFFER}}[];
	
	size_t skepu_global_prng_id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_blockSize = blockDim.x;
	size_t skepu_tid = threadIdx.x;
	{{REDUCE_RESULT_TYPE}} skepu_result{};
	
	size_t skepu_thread_V = blockIdx.x;
	size_t skepu_thread_H = threadIdx.x;
	
	size_t skepu_lookup_V = (skepu_transposed == 0) ? skepu_thread_V : skepu_thread_H;
	size_t skepu_lookup_H = (skepu_transposed == 0) ? skepu_thread_H : skepu_thread_V;
	
	if (skepu_thread_H < skepu_Hsize)
	{
		{{INDEX_INITIALIZER}}
		skepu_result = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
		
		skepu_thread_H += skepu_blockSize;
		if (!skepu_transposed)
			skepu_lookup_H += skepu_blockSize;
		else
			skepu_lookup_V += skepu_blockSize;
	}
	
	while (skepu_thread_H < skepu_Hsize)
	{
		{{INDEX_INITIALIZER}}
		
		{{MAPPAIRS_RESULT_TYPE}} tempMap = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
		skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, tempMap);
		
		skepu_thread_H += skepu_blockSize;
		if (!skepu_transposed)
			skepu_lookup_H += skepu_blockSize;
		else
			skepu_lookup_V += skepu_blockSize;
	}
	
	
	{{SHARED_BUFFER}}[skepu_tid] = skepu_result;
	
	__syncthreads();
	
	// do reduction in shared mem
	if (skepu_blockSize >= 1024) { if (skepu_tid < 512) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid + 512]); } __syncthreads(); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid + 256]); } __syncthreads(); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid + 128]); } __syncthreads(); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid +  64]); } __syncthreads(); }

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

	if (skepu_tid == 0)
	{
		skepu_output[skepu_thread_V] = {{SHARED_BUFFER}}[skepu_tid];
	}
}
)~~~";


std::string createMapPairsReduceKernelProgram_CU(SkeletonInstance &instance, UserFunction &mapPairsFunc, UserFunction &reduceFunc, std::string dir)
{
	std::stringstream SSMapPairsFuncArgs, SSKernelParamList, SSHostKernelParamList, SSStrideInit, SSStrideCount;
	std::string indexInit = "";
	if (mapPairsFunc.indexed2D)
	{
		indexInit = "skepu::Index2D skepu_index { skepu_lookup_V, skepu_lookup_H };";
		SSMapPairsFuncArgs << "skepu_index";
	}
	IndexCodeGen indexInfo = indexInitHelper_CU(mapPairsFunc);
	bool first = !indexInfo.hasIndex;
	std::string multiOutputAssign = handleOutputs_CU(mapPairsFunc, SSKernelParamList);
	handleRandomParam_CU(mapPairsFunc, SSMapPairsFuncArgs, SSKernelParamList, first);
	
	size_t ctr = 0, stride_counter = 0;
	for (UserFunction::Param& param : mapPairsFunc.elwiseParams)
	{
		if (!first) { SSMapPairsFuncArgs << ", "; }
		SSStrideInit << "if (skepu_strides[" << stride_counter << "] < 0) { " << param.name << " += (-Vsize + 1) * skepu_strides[" << stride_counter << "]; }\n"; // TODO, group V and H
		SSKernelParamList << param.resolvedTypeName << " *" << param.name << ", ";
		if (ctr++ < mapPairsFunc.Varity) // vertical containers
			SSMapPairsFuncArgs << param.name << "[skepu_lookup_V]";
		else // horizontal containers
			SSMapPairsFuncArgs << param.name << "[skepu_lookup_H]";
		first = false;
	}
	
	auto argsInfo = handleRandomAccessAndUniforms_CU(mapPairsFunc, SSMapPairsFuncArgs, SSKernelParamList, first);
	
	std::stringstream SSKernelName;
	SSKernelName << instance << "_" << transformToCXXIdentifier(ResultName) << "_MapPairsReduceKernel_" << mapPairsFunc.uniqueName << "_Varity_" << mapPairsFunc.Varity << "_Harity_" << mapPairsFunc.Harity;
	const std::string kernelName = SSKernelName.str();
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(MapPairsReduceKernelTemplate_CU,
	{
		{"{{KERNEL_NAME}}",             kernelName},
		{"{{FUNCTION_NAME_MAPPAIRS}}",  mapPairsFunc.funcNameCUDA()},
		{"{{KERNEL_PARAMS}}",           SSKernelParamList.str()},
		{"{{HOST_KERNEL_PARAMS}}",      SSHostKernelParamList.str()},
		{"{{MAPPAIRS_ARGS}}",           SSMapPairsFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}",       indexInit},
		{"{{MULTI_TYPE}}",              mapPairsFunc.multiReturnTypeNameGPU()},
		{"{{USE_MULTIRETURN}}",         (mapPairsFunc.multipleReturnTypes.size() > 0) ? "1" : "0"},
		{"{{OUTPUT_ASSIGN}}",           multiOutputAssign},
		{"{{STRIDE_COUNT}}",            SSStrideCount.str()},
		{"{{STRIDE_INIT}}",             SSStrideInit.str()},
		{"{{MAPPAIRS_RESULT_TYPE}}",    mapPairsFunc.rawReturnTypeName},
		{"{{REDUCE_RESULT_TYPE}}",      reduceFunc.rawReturnTypeName},
		{"{{REDUCE_RESULT_CPU}}",       reduceFunc.resolvedReturnTypeName},
		{"{{FUNCTION_NAME_REDUCE}}",    reduceFunc.funcNameCUDA()},
		{"{{SHARED_BUFFER}}",           "sdata_" + instance}
	});
	
	return kernelName;
}
