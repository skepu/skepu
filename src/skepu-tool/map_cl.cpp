#include "code_gen.h"
#include "code_gen_cl.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapKernelTemplate_CL = R"~~~(
__kernel void {{KERNEL_NAME}}({{KERNEL_PARAMS}} {{SIZE_PARAMS}} size_t skepu_n, size_t skepu_base)
{
	size_t skepu_i = get_global_id(0);
	size_t skepu_gridSize = get_local_size(0) * get_num_groups(0);
	{{CONTAINER_PROXIES}}

	while (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_i] = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
		skepu_i += skepu_gridSize;
	}
}
)~~~";


const std::string Constructor = R"~~~(
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

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "{{KERNEL_NAME}}", &err);
			CL_CHECK_ERROR(err, "Error creating map kernel '{{KERNEL_NAME}}'");

			skepu_kernels(counter++, &kernel);
		}

		initialized = true;
	}

	{{TEMPLATE_HEADER}}
	static void map
	(
		size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize,
		{{HOST_KERNEL_PARAMS}} {{SIZES_TUPLE_PARAM}}
		size_t skepu_n, size_t skepu_base
	)
	{
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernels(skepu_deviceID), {{KERNEL_ARGS}} {{SIZE_ARGS}} skepu_n, skepu_base);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernels(skepu_deviceID), 1, NULL, &skepu_globalSize, &skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching Map kernel");
	}
};
)~~~";


std::string createMapKernelProgram_CL(UserFunction &mapFunc, std::string dir)
{
	std::stringstream sourceStream, SSMapFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs;
	IndexCodeGen indexInfo = indexInitHelper_CL(mapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CL(mapFunc, SSHostKernelParamList, SSKernelParamList, SSKernelArgs);
	
	// Elementwise input data
	for (UserFunction::Param& param : mapFunc.elwiseParams)
	{
		if (!first) { SSMapFuncArgs << ", "; }
		SSKernelParamList << "__global " << param.rawTypeName << " *" << param.name << ", ";
		SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<" << param.resolvedTypeName << "> *" << param.name << ", ";
		SSKernelArgs << param.name << "->getDeviceDataPointer(), ";
		SSMapFuncArgs << param.name << "[skepu_i]";
		first = false;
	}

	auto argsInfo = handleRandomAccessAndUniforms_CL(mapFunc, SSMapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	handleUserTypesConstantsAndPrecision_CL({&mapFunc}, sourceStream);
	proxyCodeGenHelper_CL(argsInfo.containerProxyTypes, sourceStream);
	sourceStream << generateUserFunctionCode_CL(mapFunc) << MapKernelTemplate_CL;
	
	std::stringstream SSKernelName;
	SSKernelName << transformToCXXIdentifier(ResultName) << "_MapKernel_" << mapFunc.uniqueName << "_arity_" << mapFunc.Varity;
	const std::string kernelName = SSKernelName.str();
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor,
	{
		{"{{OPENCL_KERNEL}}",          sourceStream.str()},
		{"{{KERNEL_NAME}}",            kernelName},
		{"{{FUNCTION_NAME_MAP}}",      mapFunc.uniqueName},
		{"{{KERNEL_PARAMS}}",          SSKernelParamList.str()},
		{"{{HOST_KERNEL_PARAMS}}",     SSHostKernelParamList.str()},
		{"{{MAP_ARGS}}",               SSMapFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}",      indexInfo.indexInit},
		{"{{KERNEL_CLASS}}",           "CLWrapperClass_" + kernelName},
		{"{{KERNEL_ARGS}}",            SSKernelArgs.str()},
		{"{{CONTAINER_PROXIES}}",      argsInfo.proxyInitializer},
		{"{{CONTAINER_PROXIE_INNER}}", argsInfo.proxyInitializerInner},
		{"{{SIZE_PARAMS}}",            indexInfo.sizeParams},
		{"{{SIZE_ARGS}}",              indexInfo.sizeArgs},
		{"{{SIZES_TUPLE_PARAM}}",      indexInfo.sizesTupleParam},
		{"{{TEMPLATE_HEADER}}",        indexInfo.templateHeader},
		{"{{MULTI_TYPE}}",             mapFunc.multiReturnTypeNameGPU()},
		{"{{USE_MULTIRETURN}}",        (mapFunc.multipleReturnTypes.size() > 0) ? "1" : "0"},
		{"{{OUTPUT_ASSIGN}}",          multiOutputAssign}
	});

	return kernelName;
}
