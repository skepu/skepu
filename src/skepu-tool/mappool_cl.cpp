#include "code_gen.h"
#include "code_gen_cl.h"

using namespace clang;

// Windows are read from global memory, without staging
static const char *MapPool1DKernelTemplate_CL = R"~~~(
__kernel void {{KERNEL_NAME}}({{KERNEL_PARAMS}} {{SIZE_PARAMS}} size_t skepu_n, size_t skepu_base, int skepu_pool_size, int skepu_pool_step)
{
	size_t skepu_i = get_global_id(0);
	size_t skepu_global_prng_id = get_global_id(0);
	size_t skepu_gridSize = get_local_size(0) * get_num_groups(0);
	{{CONTAINER_PROXIES}}

	while (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
		skepu_pool1d_{{POOL_TYPE}} skepu_region = { .data = {{INPUT_NAME}} + skepu_i * skepu_pool_step, .si = skepu_pool_size, .stride = 1 };
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_i] = {{FUNCTION_NAME}}({{MAP_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME}}({{MAP_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
		skepu_i += skepu_gridSize;
	}
}
)~~~";

static const std::string Constructor = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	static cl_kernel skepu_kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
	}

	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;

		std::string source = skepu::backend::cl_helpers::replaceSizeT(R"###({{OPENCL_KERNEL}})###");

		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "{{KERNEL_NAME}}", &err);
			CL_CHECK_ERROR(err, "Error creating MapPool kernel '{{KERNEL_NAME}}'");

			skepu_kernels(counter++, &kernel);
		}

		initialized = true;
	}

	{{TEMPLATE_HEADER}}
	static void mapPool
	(
		cl_event *event, size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize,
		{{HOST_KERNEL_PARAMS}} {{SIZES_TUPLE_PARAM}}
		size_t skepu_n, size_t skepu_base, int skepu_pool_size, int skepu_pool_step
	)
	{
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernels(skepu_deviceID), {{KERNEL_ARGS}} {{SIZE_ARGS}} skepu_n, skepu_base, skepu_pool_size, skepu_pool_step);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernels(skepu_deviceID), 1, NULL, &skepu_globalSize, &skepu_localSize, 0, NULL, event);
		CL_CHECK_ERROR(skepu_err, "Error launching MapPool kernel");
	}
};
)~~~";


std::string createMapPool1DKernelProgram_CL(SkeletonInstance &instance, UserFunction &poolFunc, std::string dir)
{
	std::stringstream sourceStream, SSFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs;
	IndexCodeGen indexInfo = indexInitHelper_CL(poolFunc);
	bool first = !indexInfo.hasIndex;
	SSFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CL(poolFunc, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, false);
	handleRandomParam_CL(poolFunc, sourceStream, SSFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);

	UserFunction::RegionParam &poolParam = *poolFunc.regionParam;
	if (!first) { SSFuncArgs << ", "; }
	first = false;
	SSFuncArgs << "skepu_region";
	SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<" << poolParam.resolvedTypeName << "> *skepu_input, ";
	SSKernelArgs << "skepu_input->getDeviceDataPointer(), ";
	SSKernelParamList << "__global " << poolParam.typeNameOpenCL() << " *" << poolParam.name << ", ";

	handleUserTypesConstantsAndPrecision_CL({&poolFunc}, sourceStream);
	sourceStream << generateOpenCLRegion(1, poolParam);
	auto argsInfo = handleRandomAccessAndUniforms_CL(poolFunc, SSFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	proxyCodeGenHelper_CL(argsInfo.containerProxyTypes, sourceStream);
	sourceStream << generateUserFunctionCode_CL(poolFunc) << MapPool1DKernelTemplate_CL;

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_MapPool1DKernel_" + poolFunc.uniqueName;

	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor,
	{
		{"{{OPENCL_KERNEL}}",          sourceStream.str()},
		{"{{KERNEL_NAME}}",            kernelName},
		{"{{FUNCTION_NAME}}",          poolFunc.uniqueName},
		{"{{KERNEL_PARAMS}}",          SSKernelParamList.str()},
		{"{{HOST_KERNEL_PARAMS}}",     SSHostKernelParamList.str()},
		{"{{MAP_ARGS}}",               SSFuncArgs.str()},
		{"{{POOL_TYPE}}",              transformToCXXIdentifier(poolParam.resolvedTypeName)},
		{"{{INPUT_NAME}}",             poolParam.name},
		{"{{INDEX_INITIALIZER}}",      indexInfo.indexInit},
		{"{{KERNEL_CLASS}}",           "CLWrapperClass_" + kernelName},
		{"{{KERNEL_ARGS}}",            SSKernelArgs.str()},
		{"{{CONTAINER_PROXIES}}",      argsInfo.proxyInitializer},
		{"{{CONTAINER_PROXIE_INNER}}", argsInfo.proxyInitializerInner},
		{"{{SIZE_PARAMS}}",            indexInfo.sizeParams},
		{"{{SIZE_ARGS}}",              indexInfo.sizeArgs},
		{"{{SIZES_TUPLE_PARAM}}",      indexInfo.sizesTupleParam},
		{"{{TEMPLATE_HEADER}}",        indexInfo.templateHeader},
		{"{{MULTI_TYPE}}",             poolFunc.multiReturnTypeNameGPU()},
		{"{{USE_MULTIRETURN}}",        (poolFunc.multipleReturnTypes.size() > 0) ? "1" : "0"},
		{"{{OUTPUT_ASSIGN}}",          multiOutputAssign}
	});

	return kernelName;
}
