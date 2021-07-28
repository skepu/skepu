#include "code_gen.h"
#include "code_gen_cu.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapKernelTemplate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}({{KERNEL_PARAMS}} size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base)
{
	size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_gridSize = blockDim.x * gridDim.x;
	{{PROXIES_INIT}}

	while (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		{{PROXIES_UPDATE}}
		auto skepu_res = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
		{{OUTPUT_BINDINGS}}
		skepu_i += skepu_gridSize;
	}
}
)~~~";


std::string createMapKernelProgram_CU(SkeletonInstance &instance, UserFunction &mapFunc, size_t arity, std::string dir)
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

	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_MapKernel_" + mapFunc.uniqueName;
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(MapKernelTemplate_CU,
	{
		{"{{KERNEL_NAME}}",       kernelName},
		{"{{FUNCTION_NAME_MAP}}", mapFunc.funcNameCUDA()},
		{"{{KERNEL_PARAMS}}",     SSKernelParamList.str()},
		{"{{MAP_ARGS}}",          SSMapFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}", indexInfo.indexInit},
		{"{{OUTPUT_BINDINGS}}",   multiOutputAssign},
		{"{{PROXIES_UPDATE}}",    argsInfo.proxyInitializerInner},
		{"{{PROXIES_INIT}}",      argsInfo.proxyInitializer}
	});
	
	return kernelName;
}
