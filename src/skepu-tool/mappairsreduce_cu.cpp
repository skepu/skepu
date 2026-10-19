#include "code_gen.h"
#include "code_gen_cu.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapPairsReduceKernelTemplate_CU = R"~~~(
// (the outputs of a user function that returns multiple values are reduced one by one with the reduction function)
#define SKEPU_REDUCE_STEP_{{KERNEL_NAME}}(OFFSET) \
	if (skepu_tid < (OFFSET) && skepu_tid + (OFFSET) < skepu_Hsize) \
		for (size_t skepu_k = 0; skepu_k < {{OUTPUT_COUNT}}; ++skepu_k) \
			{{SHARED_BUFFER}}[skepu_k * skepu_blockSize + skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_k * skepu_blockSize + skepu_tid], {{SHARED_BUFFER}}[skepu_k * skepu_blockSize + skepu_tid + (OFFSET)]); \
	__syncthreads();

__global__ void {{KERNEL_NAME}}({{KERNEL_PARAMS}} size_t skepu_Vsize, size_t skepu_Hsize, size_t skepu_base, bool skepu_transposed)
{
	extern __shared__ {{REDUCE_RESULT_TYPE}} {{SHARED_BUFFER}}[];
	
	size_t skepu_global_prng_id = 0; // (the number of a pair is its position in the order of the reduction: major * minorSize + minor)
	size_t skepu_blockSize = blockDim.x;
	size_t skepu_tid = threadIdx.x;
	{{REDUCE_RESULT_TYPE}} skepu_result[{{OUTPUT_COUNT}}];
	
	size_t skepu_thread_V = blockIdx.x;
	size_t skepu_thread_H = threadIdx.x;
	
	size_t skepu_lookup_V = (skepu_transposed == 0) ? skepu_thread_V : skepu_thread_H;
	size_t skepu_lookup_H = (skepu_transposed == 0) ? skepu_thread_H : skepu_thread_V;
	
	if (skepu_thread_H < skepu_Hsize)
	{
		skepu_global_prng_id = skepu_thread_V * skepu_Hsize + skepu_thread_H;
		{{INDEX_INITIALIZER}}
#if !{{USE_MULTIRETURN}}
		skepu_result[0] = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
#else
		auto skepu_out_temp = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
		{{OUTPUT_FIRST}}
#endif
		
		skepu_thread_H += skepu_blockSize;
		if (!skepu_transposed)
			skepu_lookup_H += skepu_blockSize;
		else
			skepu_lookup_V += skepu_blockSize;
	}
	
	while (skepu_thread_H < skepu_Hsize)
	{
		skepu_global_prng_id = skepu_thread_V * skepu_Hsize + skepu_thread_H;
		{{INDEX_INITIALIZER}}
#if !{{USE_MULTIRETURN}}
		{{MAPPAIRS_RESULT_TYPE}} tempMap = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
		skepu_result[0] = {{FUNCTION_NAME_REDUCE}}(skepu_result[0], tempMap);
#else
		auto skepu_out_temp = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
		{{OUTPUT_ACCUMULATE}}
#endif
		
		skepu_thread_H += skepu_blockSize;
		if (!skepu_transposed)
			skepu_lookup_H += skepu_blockSize;
		else
			skepu_lookup_V += skepu_blockSize;
	}
	
	// (a thread that has no pair holds nothing; the reduction steps only read the threads that have one)
	for (size_t skepu_k = 0; skepu_k < {{OUTPUT_COUNT}}; ++skepu_k)
		{{SHARED_BUFFER}}[skepu_k * skepu_blockSize + skepu_tid] = skepu_result[skepu_k];
	
	__syncthreads();
	
	// do reduction in shared mem
	if (skepu_blockSize >= 1024) { SKEPU_REDUCE_STEP_{{KERNEL_NAME}}(512) }
	if (skepu_blockSize >=  512) { SKEPU_REDUCE_STEP_{{KERNEL_NAME}}(256) }
	if (skepu_blockSize >=  256) { SKEPU_REDUCE_STEP_{{KERNEL_NAME}}(128) }
	if (skepu_blockSize >=  128) { SKEPU_REDUCE_STEP_{{KERNEL_NAME}}( 64) }
	if (skepu_blockSize >=   64) { SKEPU_REDUCE_STEP_{{KERNEL_NAME}}( 32) }
	if (skepu_blockSize >=   32) { SKEPU_REDUCE_STEP_{{KERNEL_NAME}}( 16) }
	if (skepu_blockSize >=   16) { SKEPU_REDUCE_STEP_{{KERNEL_NAME}}(  8) }
	if (skepu_blockSize >=    8) { SKEPU_REDUCE_STEP_{{KERNEL_NAME}}(  4) }
	if (skepu_blockSize >=    4) { SKEPU_REDUCE_STEP_{{KERNEL_NAME}}(  2) }
	if (skepu_blockSize >=    2) { SKEPU_REDUCE_STEP_{{KERNEL_NAME}}(  1) }

	if (skepu_tid == 0)
	{
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_thread_V] = {{SHARED_BUFFER}}[0];
#else
		skepu_thread_V = blockIdx.x;
		{{OUTPUT_STORE}}
#endif
	}
}
#undef SKEPU_REDUCE_STEP_{{KERNEL_NAME}}
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
	const size_t outputCount = std::max<size_t>(1, mapPairsFunc.multipleReturnTypes.size());
	std::stringstream SSOutputFirst, SSOutputAccumulate, SSOutputStore;
	for (size_t k = 0; k < mapPairsFunc.multipleReturnTypes.size(); ++k)
	{
		SSOutputFirst << "skepu_result[" << k << "] = skepu_out_temp.e" << k << ";\n";
		SSOutputAccumulate << "skepu_result[" << k << "] = " << reduceFunc.funcNameCUDA() << "(skepu_result[" << k << "], skepu_out_temp.e" << k << ");\n";
		SSOutputStore << "skepu_output_" << k << "[skepu_thread_V] = sdata_" << instance << "[" << k << " * skepu_blockSize];\n";
	}
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
		{"{{SHARED_BUFFER}}",           "sdata_" + instance},
		{"{{OUTPUT_COUNT}}",            std::to_string(outputCount)},
		{"{{OUTPUT_FIRST}}",            SSOutputFirst.str()},
		{"{{OUTPUT_ACCUMULATE}}",       SSOutputAccumulate.str()},
		{"{{OUTPUT_STORE}}",            SSOutputStore.str()}
	});
	
	return kernelName;
}
