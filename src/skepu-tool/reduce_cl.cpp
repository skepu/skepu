#include <algorithm>

#include "code_gen.h"
#include "code_gen_cl.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

static const char *ReduceKernelTemplate_CL = R"~~~(
__kernel void {{KERNEL_NAME}}(__global {{REDUCE_RESULT_TYPE}}* input, __global {{REDUCE_RESULT_TYPE}}* output, size_t n, __local {{REDUCE_RESULT_TYPE}}* sdata)
{
	size_t blockSize = get_local_size(0);
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0) * blockSize + get_local_id(0);
	size_t gridSize = blockSize * get_num_groups(0);
	{{REDUCE_RESULT_TYPE}} result;

	if (i < n)
	{
		result = input[i];
		i += gridSize;
	}

	while (i < n)
	{
		result = {{FUNCTION_NAME_REDUCE}}(result, input[i]);
		i += gridSize;
	}

	sdata[tid] = result;
	barrier(CLK_LOCAL_MEM_FENCE);

	if (blockSize >= 1024) { if (tid < 512 && tid + 512 < n) { sdata[tid] = {{FUNCTION_NAME_REDUCE}}(sdata[tid], sdata[tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=  512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = {{FUNCTION_NAME_REDUCE}}(sdata[tid], sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=  256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = {{FUNCTION_NAME_REDUCE}}(sdata[tid], sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=  128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = {{FUNCTION_NAME_REDUCE}}(sdata[tid], sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=   64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = {{FUNCTION_NAME_REDUCE}}(sdata[tid], sdata[tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=   32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = {{FUNCTION_NAME_REDUCE}}(sdata[tid], sdata[tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=   16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = {{FUNCTION_NAME_REDUCE}}(sdata[tid], sdata[tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=    8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = {{FUNCTION_NAME_REDUCE}}(sdata[tid], sdata[tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=    4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = {{FUNCTION_NAME_REDUCE}}(sdata[tid], sdata[tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=    2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = {{FUNCTION_NAME_REDUCE}}(sdata[tid], sdata[tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }

	if (tid == 0)
		output[get_group_id(0)] = sdata[tid];
}
)~~~";


const std::string Constructor1D = R"~~~(
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
			CL_CHECK_ERROR(err, "Error creating map kernel '{{KERNEL_NAME}}'");

			kernels(counter++, &kernel);
		}

		initialized = true;
	}

	static void reduce(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		skepu::backend::cl_helpers::setKernelArgs(kernels(deviceID), input, output, n);
		clSetKernelArg(kernels(deviceID), 3, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernels(deviceID), 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}
};
)~~~";


std::string createReduce1DKernelProgram_CL(SkeletonInstance &instance, UserFunction &reduceFunc, std::string dir)
{
	std::stringstream sourceStream;

	if (reduceFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";

	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";

	// check for extra user-supplied opencl code for custome datatype
	for (UserType *RefType : reduceFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_ReduceKernel_" + reduceFunc.uniqueName;
	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_CL(reduceFunc) << ReduceKernelTemplate_CL;

	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor1D,
	{
		{"{{OPENCL_KERNEL}}",        sourceStream.str()},
		{"{{KERNEL_CLASS}}",         "CLWrapperClass_" + kernelName},
		{"{{REDUCE_RESULT_TYPE}}",   reduceFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName},
		{"{{FUNCTION_NAME_REDUCE}}", reduceFunc.uniqueName}
	});
	return kernelName;
}



const std::string Constructor2D = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	enum
	{
		KERNEL_ROWWISE = 0,
		KERNEL_COLWISE,
		KERNEL_COUNT
	};

	static cl_kernel kernels(size_t deviceID, size_t kerneltype, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8][KERNEL_COUNT]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID][kerneltype] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID][kerneltype];
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

			cl_kernel rowwisekernel = clCreateKernel(program, "{{KERNEL_NAME}}_RowWise", &err);
			CL_CHECK_ERROR(err, "Error creating row-wise Reduce kernel '{{KERNEL_NAME}}'");

			cl_kernel colwisekernel = clCreateKernel(program, "{{KERNEL_NAME}}_ColWise", &err);
			CL_CHECK_ERROR(err, "Error creating col-wise Reduce kernel '{{KERNEL_NAME}}'");

			kernels(counter, KERNEL_ROWWISE, &rowwisekernel);
			kernels(counter, KERNEL_COLWISE, &colwisekernel);
			counter++;
		}

		initialized = true;
	}

	static void reduceRowWise(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_ROWWISE);
		skepu::backend::cl_helpers::setKernelArgs(kernel, input, output, n);
		clSetKernelArg(kernel, 3, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}

	static void reduceColWise(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_COLWISE);
		skepu::backend::cl_helpers::setKernelArgs(kernel, input, output, n);
		clSetKernelArg(kernel, 3, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}

	static void reduce(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		reduceRowWise(deviceID, localSize, globalSize, input, output, n, sharedMemSize);
	}
};
)~~~";

std::string createReduce2DKernelProgram_CL(SkeletonInstance &instance, UserFunction &rowWiseFunc, UserFunction &colWiseFunc, std::string dir)
{
	std::stringstream sourceStream;
	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_ReduceKernel_" + rowWiseFunc.uniqueName + "_" + colWiseFunc.uniqueName;

	if (rowWiseFunc.requiresDoublePrecision || colWiseFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";

	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";

	// check for extra user-supplied opencl code for custome datatype
	std::set<UserType*> referencedUTs;
	std::set_union(
		rowWiseFunc.ReferencedUTs.cbegin(), rowWiseFunc.ReferencedUTs.cend(),
		colWiseFunc.ReferencedUTs.cbegin(), colWiseFunc.ReferencedUTs.cend(), std::inserter(referencedUTs, referencedUTs.begin()));

	for (UserType *RefType : referencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);

	const std::string className = "CLWrapperClass_" + kernelName;

	sourceStream << KernelPredefinedTypes_CL;

	if (rowWiseFunc.refersTo(colWiseFunc))
		sourceStream << generateUserFunctionCode_CL(rowWiseFunc);
	else if (colWiseFunc.refersTo(rowWiseFunc))
		sourceStream << generateUserFunctionCode_CL(colWiseFunc);
	else
		sourceStream << generateUserFunctionCode_CL(rowWiseFunc) << generateUserFunctionCode_CL(colWiseFunc);

	sourceStream << templateString(ReduceKernelTemplate_CL,
	{
		{"{{REDUCE_RESULT_TYPE}}",   rowWiseFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName + "_RowWise"},
		{"{{FUNCTION_NAME_REDUCE}}", rowWiseFunc.uniqueName}
	});
	sourceStream << templateString(ReduceKernelTemplate_CL,
	{
		{"{{REDUCE_RESULT_TYPE}}",   colWiseFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName + "_ColWise"},
		{"{{FUNCTION_NAME_REDUCE}}", colWiseFunc.uniqueName}
	});

	std::string finalSource = Constructor2D;
	replaceTextInString(finalSource, "{{OPENCL_KERNEL}}", sourceStream.str());
	replaceTextInString(finalSource, "{{KERNEL_NAME}}", kernelName);
	replaceTextInString(finalSource, "{{KERNEL_CLASS}}", className);

	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;
	return kernelName;
}
