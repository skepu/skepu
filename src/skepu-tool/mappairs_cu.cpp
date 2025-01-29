#include "code_gen.h"
#include "code_gen_cu.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapPairsKernelTemplate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}({{KERNEL_PARAMS}} size_t skepu_Vsize, size_t skepu_Hsize, size_t skepu_base)
{
	size_t skepu_n = skepu_Vsize * skepu_Hsize;
	size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_global_prng_id = skepu_i;
	size_t skepu_gridSize = blockDim.x * gridDim.x;
	size_t skepu_w2 = skepu_Hsize;

	while (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		auto skepu_res = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
		{{OUTPUT_BINDINGS}}
		skepu_i += skepu_gridSize;
	}
}
)~~~";


std::string createMapPairsKernelProgram_CU(SkeletonInstance &instance, UserFunction &mapPairsFunc, std::string dir)
{
	std::stringstream SSKernelParamList, SSMapPairsFuncArgs;
	IndexCodeGen indexInfo = indexInitHelper_CU(mapPairsFunc);
	bool first = !indexInfo.hasIndex;
	SSMapPairsFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CU(mapPairsFunc, SSKernelParamList);
	handleRandomParam_CU(mapPairsFunc, SSMapPairsFuncArgs, SSKernelParamList, first);
	
	size_t ctr = 0;
	for (UserFunction::Param& param : mapPairsFunc.elwiseParams)
	{
		if (!first) { SSMapPairsFuncArgs << ", "; }
		SSKernelParamList << param.resolvedTypeName << " *" << param.name << ", ";
		if (ctr++ < mapPairsFunc.Varity) // vertical containers
			SSMapPairsFuncArgs << param.name << "[skepu_i / skepu_Hsize]";
		else // horizontal containers
			SSMapPairsFuncArgs << param.name << "[skepu_i % skepu_Hsize]";
		first = false;
	}
	auto argsInfo = handleRandomAccessAndUniforms_CU(mapPairsFunc, SSMapPairsFuncArgs, SSKernelParamList, first);
	
	std::stringstream SSKernelName;
	SSKernelName << instance + "_" + transformToCXXIdentifier(ResultName) << "_MapPairsKernel_" << mapPairsFunc.uniqueName << "_Varity_" << mapPairsFunc.Varity << "_Harity_" << mapPairsFunc.Harity;
	const std::string kernelName = SSKernelName.str();
	
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(MapPairsKernelTemplate_CU,
	{
		{"{{KERNEL_NAME}}",            kernelName},
		{"{{FUNCTION_NAME_MAPPAIRS}}", mapPairsFunc.funcNameCUDA()},
		{"{{KERNEL_PARAMS}}",          SSKernelParamList.str()},
		{"{{MAPPAIRS_ARGS}}",          SSMapPairsFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}",      indexInfo.indexInit},
		{"{{OUTPUT_BINDINGS}}",        multiOutputAssign},
		{"{{PROXIES_UPDATE}}",         argsInfo.proxyInitializerInner},
		{"{{PROXIES_INIT}}",           argsInfo.proxyInitializer}
	});
	return kernelName;
}