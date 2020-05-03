#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapPairsKernelTemplate_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME(SKEPU_KERNEL_PARAMS size_t skepu_n, size_t skepu_Vsize, size_t skepu_Hsize, size_t skepu_base)
{
	size_t skepu_i = get_global_id(0);
	size_t skepu_gridSize = get_local_size(0) * get_num_groups(0);
	SKEPU_CONTAINER_PROXIES
	
	while (skepu_i < skepu_n)
	{
		SKEPU_INDEX_INITIALIZER
		SKEPU_CONTAINER_PROXIE_INNER
		skepu_output[skepu_i] = SKEPU_FUNCTION_NAME_MAPPAIRS(SKEPU_MAP_PARAMS);
		skepu_i += skepu_gridSize;
	}
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
			CL_CHECK_ERROR(err, "Error creating mappairs kernel 'SKEPU_KERNEL_NAME'");
			
			kernels(counter++, &kernel);
		}
		
		initialized = true;
	}
	
	static void map
	(
		size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize, SKEPU_HOST_KERNEL_PARAMS
		size_t skepu_n, size_t skepu_Vsize, size_t skepu_Hsize, size_t skepu_base
	)
	{
		skepu::backend::cl_helpers::setKernelArgs(kernels(skepu_deviceID), SKEPU_KERNEL_ARGS skepu_n, skepu_Vsize, skepu_Hsize, skepu_base);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), kernels(skepu_deviceID), 1, NULL, &skepu_globalSize, &skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching Map kernel");
	}
};
)~~~";


std::string createMapPairsKernelProgram_CL(UserFunction &mapPairsFunc, std::string dir)
{
	std::stringstream sourceStream, SSmapPairsFuncParams, SSKernelParamList, SSHostKernelParamList, SSKernelArgs, SSProxyInitializer, SSProxyInitializerInner;
	std::string indexInitializer;
	std::map<ContainerType, std::set<std::string>> containerProxyTypes;
	bool first = true;
	
	if (mapPairsFunc.indexed1D)
	{
		SSmapPairsFuncParams << "skepu_index";
		indexInitializer = "index1_t skepu_index = { .i = skepu_base + skepu_i };";
		first = false;
	}
	else if (mapPairsFunc.indexed2D)
	{
		SSmapPairsFuncParams << "skepu_index";
		indexInitializer = "index2_t skepu_index = { .row = (skepu_base + skepu_i) / skepu_Hsize, .col = (skepu_base + skepu_i) % skepu_Hsize };";
		first = false;
	}
	
	// Output data
	if (mapPairsFunc.multipleReturnTypes.size() == 0)
	{
		SSKernelParamList << " __global " << mapPairsFunc.resolvedReturnTypeName << "* skepu_output, ";
		SSHostKernelParamList << " skepu::backend::DeviceMemPointer_CL<" << mapPairsFunc.resolvedReturnTypeName << "> *skepu_output, ";
		SSKernelArgs << " skepu_output->getDeviceDataPointer(), ";
	}
	else
	{
		size_t outCtr = 0;
		for (std::string& outputType : mapPairsFunc.multipleReturnTypes)
		{
			SSKernelParamList << "__global " << outputType << "* skepu_output_" << outCtr << ", ";
			SSHostKernelParamList << " skepu::backend::DeviceMemPointer_CL<" << outputType << "> *skepu_output_" << outCtr << ", ";
			SSKernelArgs << " skepu_output_" << outCtr << "->getDeviceDataPointer(), ";
			outCtr++;
		}
	}
	
	size_t ctr = 0;
	for (UserFunction::Param& param : mapPairsFunc.elwiseParams)
	{
		if (!first) { SSmapPairsFuncParams << ", "; }
		SSKernelParamList << "__global " << param.resolvedTypeName << " *" << param.name << ", ";
		SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<" << param.resolvedTypeName << "> *" << param.name << ", ";
		SSKernelArgs << param.name << "->getDeviceDataPointer(), ";
		if (ctr++ < mapPairsFunc.Varity) // vertical containers
			SSmapPairsFuncParams << param.name << "[skepu_i / skepu_Hsize]";
		else // horizontal containers
			SSmapPairsFuncParams << param.name << "[skepu_i % skepu_Hsize]";
		first = false;
	}
	
	for (UserFunction::RandomAccessParam& param : mapPairsFunc.anyContainerParams)
	{
		std::string name = "skepu_container_" + param.name;
		if (!first) { SSmapPairsFuncParams << ", "; }
		SSHostKernelParamList << param.TypeNameHost() << " skepu_container_" << param.name << ", ";
		containerProxyTypes[param.containerType].insert(param.resolvedTypeName);
		switch (param.containerType)
		{
			case ContainerType::Vector:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->size(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name << ", .size = skepu_size_" << param.name << " };\n";
				break;
			
			case ContainerType::Matrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_rows_" << param.name << ", size_t skepu_cols_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->total_rows(), std::get<0>(" << name << ")->total_cols(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name
					<< ", .rows = skepu_rows_" << param.name << ", .cols = skepu_cols_" << param.name << " };\n";
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
			
			case ContainerType::SparseMatrix:
				SSKernelParamList
					<< "__global " << param.resolvedTypeName << " *" << name << ", "
					<< "__global size_t *" << param.name << "_row_pointers, "
					<< "__global size_t *" << param.name << "_col_indices, "
					<< "size_t skepu_size_" << param.name << ", ";
					
				SSKernelArgs
					<< "std::get<1>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<2>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<3>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<0>(" << name << ")->total_nnz(), ";
				
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { "
					<< ".data = " << name << ", "
					<< ".row_offsets = " << param.name << "_row_pointers, "
					<< ".col_indices = " << param.name << "_col_indices, "
					<< ".count = skepu_size_" << param.name << " };\n";
				break;
		}
		SSmapPairsFuncParams << param.name;
		first = false;
	}
	
	for (UserFunction::Param& param : mapPairsFunc.anyScalarParams)
	{
		if (!first) { SSmapPairsFuncParams << ", "; }
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSHostKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSKernelArgs << param.name << ", ";
		SSmapPairsFuncParams << param.name;
		first = false;
	}
	
	if (mapPairsFunc.requiresDoublePrecision)
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
	
	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";
	
	// check for extra user-supplied opencl code for custome datatype
	// TODO: Also check the referenced UFs for referenced UTs skepu::userstruct
	for (UserType *RefType : mapPairsFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);
	
	std::stringstream SSKernelName;
	SSKernelName << transformToCXXIdentifier(ResultName) << "_MapPairsKernel_" << mapPairsFunc.uniqueName << "_Varity_" << mapPairsFunc.Varity << "_Harity_" << mapPairsFunc.Harity;
	const std::string kernelName = SSKernelName.str();
	const std::string className = "CLWrapperClass_" + kernelName;
	
	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_CL(mapPairsFunc) << MapPairsKernelTemplate_CL;
	
	std::string finalSource = Constructor;
	replaceTextInString(finalSource, "SKEPU_OPENCL_KERNEL", sourceStream.str());
	replaceTextInString(finalSource, PH_MapResultType, mapPairsFunc.resolvedReturnTypeName);
	replaceTextInString(finalSource, PH_KernelName, kernelName);
	replaceTextInString(finalSource, PH_MapPairsFuncName, mapPairsFunc.uniqueName);
	replaceTextInString(finalSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(finalSource, "SKEPU_HOST_KERNEL_PARAMS", SSHostKernelParamList.str());
	replaceTextInString(finalSource, PH_MapParams, SSmapPairsFuncParams.str());
	replaceTextInString(finalSource, PH_IndexInitializer, indexInitializer);
	replaceTextInString(finalSource, "SKEPU_KERNEL_CLASS", className);
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARGS", SSKernelArgs.str());
	replaceTextInString(finalSource, "SKEPU_CONTAINER_PROXIES", SSProxyInitializer.str());
	replaceTextInString(finalSource, "SKEPU_CONTAINER_PROXIE_INNER", SSProxyInitializerInner.str());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;
	
	return kernelName;
}