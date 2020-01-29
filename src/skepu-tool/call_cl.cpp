#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *CallKernelTemplate_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME(SKEPU_KERNEL_PARAMS)
{
	SKEPU_CONTAINER_PROXIES
	SKEPU_CONTAINER_PROXIE_INNER

	SKEPU_FUNCTION_NAME_CALL(SKEPU_CALL_ARGS);
}
)~~~";

const std::string Constructor = R"~~~(
class SKEPU_KERNEL_CLASS
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

		std::string source = skepu::backend::cl_helpers::replaceSizeT(R"###(SKEPU_OPENCL_KERNEL)###");

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "SKEPU_KERNEL_NAME", &err);
			CL_CHECK_ERROR(err, "Error creating Call kernel 'SKEPU_KERNEL_NAME'");

			kernels(counter++, &kernel);
		}

		initialized = true;
	}

	static void call(size_t deviceID, size_t localSize, size_t globalSize SKEPU_HOST_KERNEL_PARAMS)
	{
		skepu::backend::cl_helpers::setKernelArgs(kernels(deviceID) SKEPU_KERNEL_ARGS);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernels(deviceID), 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Call kernel");
	}
};
)~~~";


std::string createCallKernelProgram_CL(UserFunction &callFunc, std::string dir)
{
	std::stringstream sourceStream, SSCallFuncParams, SSKernelParamList, SSHostKernelParamList, SSKernelArgs, SSProxyInitializer, SSProxyInitializerInner;
	std::map<ContainerType, std::set<std::string>> containerProxyTypes;
	bool first = true;

	for (UserFunction::RandomAccessParam& param : callFunc.anyContainerParams)
	{
		std::string name = "skepu_container_" + param.name;
		if (!first) { SSCallFuncParams << ", "; }
		SSHostKernelParamList << ", " << param.TypeNameHost() << " skepu_container_" << param.name;
		containerProxyTypes[param.containerType].insert(param.resolvedTypeName);
		switch (param.containerType)
		{
			case ContainerType::Vector:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name;
				SSKernelArgs << ", std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->size()";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name << ", .size = skepu_size_" << param.name << " };\n";
				break;
			
			case ContainerType::Matrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_rows_" << param.name << ", size_t skepu_cols_" << param.name;
				SSKernelArgs << ", std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->total_rows(), std::get<0>(" << name << ")->total_cols()";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name
					<< ", .rows = skepu_rows_" << param.name << ", .cols = skepu_cols_" << param.name << " };\n";
				break;
			
			case ContainerType::SparseMatrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name;
				SSKernelArgs << ", skepu_container_" << param.name << ".size()";
				break;
			
			case ContainerType::MatRow:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_cols_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->total_cols(), ";
				SSProxyInitializerInner << param.TypeNameOpenCL() << " " << param.name << " = { .data = (" << name << " + i * skepu_cols_" << param.name << "), .cols = skepu_cols_" << param.name << " };\n";
				break;
			
			case ContainerType::Tensor3:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", "
					<< "size_t skepu_size_i_" << param.name << ", size_t skepu_size_j_" << param.name << ", size_t skepu_size_k_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<0>(" << name << ")->size_i(), std::get<0>(" << name << ")->size_j(), std::get<0>(" << name << ")->size_k(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name
					<< ", .size_i = skepu_size_i_" << param.name << ", .size_j = skepu_size_j_" << param.name << ", .size_k = skepu_size_k_" << param.name << " };\n";
				break;
			
			case ContainerType::Tensor4:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", "
					<< "size_t skepu_size_i_" << param.name << ", size_t skepu_size_j_" << param.name << ", size_t skepu_size_k_" << param.name << ", size_t skepu_size_l_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<0>(" << name << ")->size_i(), std::get<0>(" << name << ")->size_j(), std::get<0>(" << name << ")->size_k(), std::get<0>(" << name << ")->size_l(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name
					<< ", .size_i = skepu_size_i_" << param.name << ", .size_j = skepu_size_j_" << param.name << ", .size_k = skepu_size_k_" << param.name << ", .size_l = skepu_size_l_" << param.name << " };\n";
				break;
		}
		SSCallFuncParams << param.name;
		first = false;
	}

	for (UserFunction::Param& param : callFunc.anyScalarParams)
	{
		if (!first) { SSCallFuncParams << ", "; SSKernelParamList << ", "; }
		SSKernelParamList << param.resolvedTypeName << " " << param.name;
		SSHostKernelParamList << ", " << param.resolvedTypeName << " " << param.name;
		SSKernelArgs << ", " << param.name;
		SSCallFuncParams << param.name;
		first = false;
	}

	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";

	// check for extra user-supplied opencl code for custome datatype
	// TODO: Also check the referenced UFs for referenced UTs skepu::userstruct
	for (UserType *RefType : callFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);

	if (callFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";

	for (const std::string &type : containerProxyTypes[ContainerType::Vector])
		sourceStream << generateOpenCLVectorProxy(type);

	for (const std::string &type : containerProxyTypes[ContainerType::Matrix])
		sourceStream << generateOpenCLMatrixProxy(type);

	for (const std::string &type : containerProxyTypes[ContainerType::SparseMatrix])
		sourceStream << generateOpenCLSparseMatrixProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::MatRow])
		sourceStream << generateOpenCLMatrixRowProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::Tensor3])
		sourceStream << generateOpenCLTensor3Proxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::Tensor4])
		sourceStream << generateOpenCLTensor4Proxy(type);

	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_CL(callFunc) << CallKernelTemplate_CL;

	const std::string kernelName = ResultName + "_CallKernel_" + callFunc.uniqueName + "_";
	const std::string className = "CLWrapperClass_" + kernelName;

	std::string finalSource = Constructor;
	replaceTextInString(finalSource, "SKEPU_OPENCL_KERNEL", sourceStream.str());
	replaceTextInString(finalSource, PH_KernelName, kernelName);
	replaceTextInString(finalSource, PH_CallFuncName, callFunc.uniqueName);
	replaceTextInString(finalSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(finalSource, PH_CallArgs, SSCallFuncParams.str());
	replaceTextInString(finalSource, "SKEPU_HOST_KERNEL_PARAMS", SSHostKernelParamList.str());
	replaceTextInString(finalSource, "SKEPU_KERNEL_CLASS", className);
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARGS", SSKernelArgs.str());
	replaceTextInString(finalSource, "SKEPU_CONTAINER_PROXIES", SSProxyInitializer.str());
	replaceTextInString(finalSource, "SKEPU_CONTAINER_PROXIE_INNER", SSProxyInitializerInner.str());

	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;

	return kernelName;
}
