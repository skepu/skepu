#include "code_gen.h"
#include "code_gen_cu.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapKernelTemplate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}({{KERNEL_PARAMS}} size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base, skepu::StrideList<{{STRIDE_COUNT}}> skepu_strides)
{
	size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_global_prng_id = skepu_i;
	size_t skepu_gridSize = blockDim.x * gridDim.x;
	{{PROXIES_INIT}}
	{{STRIDE_INIT}}

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
	std::stringstream SSKernelParamList, SSMapFuncArgs, SSStrideInit, SSStrideCount;
	IndexCodeGen indexInfo = indexInitHelper_CU(mapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CU(mapFunc, SSKernelParamList, true);
	handleRandomParam_CU(mapFunc, SSMapFuncArgs, SSKernelParamList, first);
	
	size_t stride_counter = 0;
	for (size_t er = 0; er < std::max<size_t>(1, mapFunc.multipleReturnTypes.size()); ++er)
	{
		std::stringstream namesuffix;
		if (mapFunc.multipleReturnTypes.size()) namesuffix << "_" << stride_counter;
		SSStrideInit << "if (skepu_strides[" << stride_counter << "] < 0) { skepu_output" << namesuffix.str() << " += (-skepu_n + 1) * skepu_strides[" << stride_counter << "]; }\n";
		stride_counter++;
	}
	
	for (UserFunction::Param& param : mapFunc.elwiseParams)
	{
		if (!first) { SSMapFuncArgs << ", "; }
		SSStrideInit << "if (skepu_strides[" << stride_counter << "] < 0) { " << param.name << " += (-skepu_n + 1) * skepu_strides[" << stride_counter << "]; }\n";
		SSKernelParamList << param.resolvedTypeName << " *" << param.name << ", ";
		SSMapFuncArgs << param.name << "[skepu_i * skepu_strides[" << stride_counter++ << "]]";
		first = false;
	}
	auto argsInfo = handleRandomAccessAndUniforms_CU(mapFunc, SSMapFuncArgs, SSKernelParamList, first);

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_MapKernel_" + mapFunc.uniqueName;
	SSStrideCount << (mapFunc.elwiseParams.size() + std::max<size_t>(1, mapFunc.multipleReturnTypes.size()));
	
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
		{"{{PROXIES_INIT}}",      argsInfo.proxyInitializer},
		{"{{STRIDE_COUNT}}",      SSStrideCount.str()},
		{"{{STRIDE_INIT}}",       SSStrideInit.str()}
	});
	
	return kernelName;
}
