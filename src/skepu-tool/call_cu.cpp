#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *CallKernelTemplate_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME(SKEPU_KERNEL_PARAMS)
{
	SKEPU_FUNCTION_NAME_CALL(SKEPU_CALL_ARGS);
}
)~~~";


std::string createCallKernelProgram_CU(UserFunction &callFunc, std::string dir)
{
	const std::string kernelName = ResultName + "_CallKernel_" + callFunc.uniqueName + "_";;

	std::stringstream SSCallFuncParams, SSKernelParamList;
	bool first = true;

	for (UserFunction::RandomAccessParam& param : callFunc.anyContainerParams)
	{
		if (!first) { SSCallFuncParams << ", "; SSKernelParamList << ", "; }
		SSKernelParamList << param.fullTypeName << " " << param.name;
		SSCallFuncParams << param.name;
		first = false;
	}

	for (UserFunction::Param& param : callFunc.anyScalarParams)
	{
		if (!first) { SSCallFuncParams << ", "; SSKernelParamList << ", "; }
		SSKernelParamList << param.resolvedTypeName << " " << param.name;
		SSCallFuncParams << param.name;
		first = false;
	}

	std::string kernelSource = CallKernelTemplate_CU;
	replaceTextInString(kernelSource, "SKEPU_KERNEL_NAME", kernelName);
	replaceTextInString(kernelSource, "SKEPU_FUNCTION_NAME_CALL", callFunc.funcNameCUDA());
	replaceTextInString(kernelSource, "SKEPU_KERNEL_PARAMS", SSKernelParamList.str());
	replaceTextInString(kernelSource, "SKEPU_CALL_ARGS", SSCallFuncParams.str());

	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << kernelSource;
	return kernelName;
}
