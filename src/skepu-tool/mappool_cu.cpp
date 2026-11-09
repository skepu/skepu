#include "code_gen.h"
#include "code_gen_cu.h"

using namespace clang;

static const char *MapPool1DKernelTemplate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}({{KERNEL_PARAMS}} size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base, int skepu_pool_size, int skepu_pool_step)
{
	size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_global_prng_id = skepu_i;
	size_t skepu_gridSize = blockDim.x * gridDim.x;
	{{PROXIES_INIT}}

	while (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		{{PROXIES_UPDATE}}
		skepu::Pool1D<{{POOL_TYPE}}> skepu_region(skepu_pool_size, 1, {{INPUT_NAME}} + skepu_i * skepu_pool_step);
		auto skepu_res = {{FUNCTION_NAME}}({{MAP_ARGS}});
		{{OUTPUT_BINDINGS}}
		skepu_i += skepu_gridSize;
	}
}
)~~~";


std::string createMapPool1DKernelProgram_CU(SkeletonInstance &instance, UserFunction &poolFunc, std::string dir)
{
	std::stringstream SSKernelParamList, SSFuncArgs;
	IndexCodeGen indexInfo = indexInitHelper_CU(poolFunc);
	bool first = !indexInfo.hasIndex;
	SSFuncArgs << indexInfo.mapFuncParam;
	std::string outputBindings = handleOutputs_CU(poolFunc, SSKernelParamList, false);
	handleRandomParam_CU(poolFunc, SSFuncArgs, SSKernelParamList, first);

	UserFunction::RegionParam &poolParam = *poolFunc.regionParam;
	if (!first) { SSFuncArgs << ", "; }
	first = false;
	SSFuncArgs << "skepu_region";
	SSKernelParamList << "const " << poolParam.resolvedTypeName << " *" << poolParam.name << ", ";

	auto argsInfo = handleRandomAccessAndUniforms_CU(poolFunc, SSFuncArgs, SSKernelParamList, first, true);

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_MapPool1DKernel_" + poolFunc.uniqueName;

	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(MapPool1DKernelTemplate_CU,
	{
		{"{{KERNEL_NAME}}",       kernelName},
		{"{{FUNCTION_NAME}}",     poolFunc.funcNameCUDA()},
		{"{{KERNEL_PARAMS}}",     SSKernelParamList.str()},
		{"{{MAP_ARGS}}",          SSFuncArgs.str()},
		{"{{POOL_TYPE}}",         poolParam.resolvedTypeName},
		{"{{INPUT_NAME}}",        poolParam.name},
		{"{{INDEX_INITIALIZER}}", indexInfo.indexInit},
		{"{{OUTPUT_BINDINGS}}",   outputBindings},
		{"{{PROXIES_UPDATE}}",    argsInfo.proxyInitializerInner},
		{"{{PROXIES_INIT}}",      argsInfo.proxyInitializer}
	});

	return kernelName;
}
