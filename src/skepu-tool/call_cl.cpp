#include "code_gen.h"
#include "code_gen_cl.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *CallKernelTemplate_CL = R"~~~(
__kernel void {{KERNEL_NAME}}({{KERNEL_PARAMS}} int dummy)
{
	{{CONTAINER_PROXIES}}
	{{CONTAINER_PROXIE_INNER}}

	{{FUNCTION_NAME_CALL}}({{CALL_ARGS}});
}
)~~~";

const std::string Constructor = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	static cl_kernel kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
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
			CL_CHECK_ERROR(err, "Error creating Call kernel '{{KERNEL_NAME}}'");

			kernels(counter++, &kernel);
		}

		initialized = true;
	}

	static void call(size_t deviceID, size_t localSize, size_t globalSize, {{HOST_KERNEL_PARAMS}} bool dummy = 0)
	{
		skepu::backend::cl_helpers::setKernelArgs(kernels(deviceID), {{KERNEL_ARGS}} 0);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernels(deviceID), 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Call kernel");
	}
};
)~~~";


std::string createCallKernelProgram_CL(SkeletonInstance &instance, UserFunction &callFunc, std::string dir)
{
	std::stringstream sourceStream, SSCallFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs;
	std::stringstream SSStrideParams, SSStrideArgs, SSStrideInit;
	IndexCodeGen indexInfo = indexInitHelper_CL(callFunc);
	bool first = !indexInfo.hasIndex;
	SSCallFuncArgs << indexInfo.mapFuncParam;
	
	auto argsInfo = handleRandomAccessAndUniforms_CL(callFunc, SSCallFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	handleUserTypesConstantsAndPrecision_CL({&callFunc}, sourceStream);
	proxyCodeGenHelper_CL(argsInfo.containerProxyTypes, sourceStream);
	sourceStream << generateUserFunctionCode_CL(callFunc) << CallKernelTemplate_CL;

	const std::string kernelName = instance + "_" + ResultName + "_CallKernel_" + callFunc.uniqueName + "_";
	const std::string className = "CLWrapperClass_" + kernelName;

	std::string finalSource = Constructor;
	replaceTextInString(finalSource, "{{OPENCL_KERNEL}}", sourceStream.str());
	replaceTextInString(finalSource, "{{KERNEL_NAME}}", kernelName);
	replaceTextInString(finalSource, "{{FUNCTION_NAME_CALL}}", callFunc.uniqueName);
	replaceTextInString(finalSource, "{{KERNEL_PARAMS}}", SSKernelParamList.str());
	replaceTextInString(finalSource, "{{CALL_ARGS}}", SSCallFuncArgs.str());
	replaceTextInString(finalSource, "{{HOST_KERNEL_PARAMS}}", SSHostKernelParamList.str());
	replaceTextInString(finalSource, "{{KERNEL_CLASS}}", className);
	replaceTextInString(finalSource, "{{KERNEL_ARGS}}", SSKernelArgs.str());
	replaceTextInString(finalSource, "{{CONTAINER_PROXIES}}", argsInfo.proxyInitializer);
	replaceTextInString(finalSource, "{{CONTAINER_PROXIE_INNER}}", argsInfo.proxyInitializerInner);

	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;

	return kernelName;
}
