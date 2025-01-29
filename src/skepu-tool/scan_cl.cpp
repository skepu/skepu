#include "code_gen.h"
#include "code_gen_cl.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const std::string ScanKernel_CL = R"~~~(
__kernel void {{KERNEL_NAME}}_Scan(__global {{SCAN_TYPE}}* skepu_input, __global {{SCAN_TYPE}}* skepu_output, __global {{SCAN_TYPE}}* blockSums, size_t skepu_n, size_t skepu_numElements, __local {{SCAN_TYPE}}* skepu_sdata)
{
	const size_t threadIdx = get_local_id(0);
	const size_t blockDim = get_local_size(0);
	const size_t blockIdx = get_group_id(0);
	const size_t gridDim = get_num_groups(0);
	unsigned int pout = 0;
	unsigned int pin = 1;
	size_t mem = get_global_id(0);
	size_t gridSize = blockDim * gridDim;
	size_t numBlocks = skepu_numElements / blockDim + (skepu_numElements % blockDim == 0 ? 0:1);

	for (size_t blockNr = blockIdx; blockNr < numBlocks; blockNr += gridDim)
	{
		skepu_sdata[pout * skepu_n + threadIdx] = (mem < skepu_numElements) ? skepu_input[mem] : 0;
		barrier(CLK_LOCAL_MEM_FENCE);

		for (size_t skepu_offset = 1; skepu_offset < skepu_n; skepu_offset *=2)
		{
			pout = 1 - pout;
			pin = 1- pout;
			if (threadIdx >= skepu_offset)
				skepu_sdata[pout * skepu_n + threadIdx] = {{FUNCTION_NAME_SCAN}}(skepu_sdata[pin * skepu_n + threadIdx], skepu_sdata[pin * skepu_n + threadIdx - skepu_offset]);
			else
				skepu_sdata[pout * skepu_n + threadIdx] = skepu_sdata[pin * skepu_n + threadIdx];
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if (threadIdx == blockDim - 1)
			blockSums[blockNr] = skepu_sdata[pout * skepu_n + blockDim - 1];

		if (mem < skepu_numElements)
			skepu_output[mem] = skepu_sdata[pout * skepu_n + threadIdx];
		mem += gridSize;

		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
)~~~";

const std::string ScanUpdate_CL = R"~~~(
#define NULL 0
__kernel void {{KERNEL_NAME}}_ScanUpdate(__global {{SCAN_TYPE}}* data, __global {{SCAN_TYPE}}* sums, int isInclusive, {{SCAN_TYPE}} init, size_t skepu_n, __global {{SCAN_TYPE}}* ret, __local {{SCAN_TYPE}}* skepu_sdata)
{
	__local {{SCAN_TYPE}} skepu_offset;
	__local {{SCAN_TYPE}} inc_offset;
	const size_t threadIdx = get_local_id(0);
	const size_t blockDim = get_local_size(0);
	const size_t blockIdx = get_group_id(0);
	const size_t gridDim = get_num_groups(0);
	size_t gridSize = blockDim * gridDim;
	size_t mem = get_global_id(0);
	size_t numBlocks = skepu_n / blockDim + (skepu_n % blockDim == 0 ? 0:1);

	for (size_t blockNr = blockIdx; blockNr < numBlocks; blockNr += gridDim)
	{
		if (threadIdx == 0)
		{
			if (isInclusive == 0)
			{
				skepu_offset = init;
				if (blockNr > 0)
				{
					skepu_offset = {{FUNCTION_NAME_SCAN}}(skepu_offset, sums[blockNr-1]);
					inc_offset = sums[blockNr-1];
				}
			}
			else
			{
				if(blockNr > 0)
					skepu_offset = sums[blockNr-1];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (isInclusive == 1)
		{
			skepu_sdata[threadIdx] = (mem >= skepu_n)
				? 0 : (blockNr > 0)
					? {{FUNCTION_NAME_SCAN}}(skepu_offset, data[mem]) : data[mem];

			if (mem == skepu_n-1 && ret != NULL)
				*ret = skepu_sdata[threadIdx];
		}
		else
		{
			if (mem == skepu_n-1 && ret != NULL)
				*ret = {{FUNCTION_NAME_SCAN}}(inc_offset, data[mem]);

			skepu_sdata[threadIdx] = (threadIdx == 0)
				? skepu_offset : (mem-1 < skepu_n)
					? {{FUNCTION_NAME_SCAN}}(skepu_offset, data[mem-1]) : 0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (mem < skepu_n)
			data[mem] = skepu_sdata[threadIdx];
		mem += gridSize;

		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
)~~~";


const std::string ScanAdd_CL = R"~~~(
__kernel void {{KERNEL_NAME}}_ScanAdd(__global {{SCAN_TYPE}}* data, {{SCAN_TYPE}} skepu_sum, size_t skepu_n)
{
	size_t i = get_global_id(0);
	size_t gridSize = get_local_size(0) * get_num_groups(0);

	while (i < skepu_n)
	{
		data[i] = {{FUNCTION_NAME_SCAN}}(data[i], skepu_sum);
		i += gridSize;
	}
}
)~~~";


const std::string Constructor = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	enum
	{
		KERNEL_SCAN = 0,
		KERNEL_SCAN_UPDATE,
		KERNEL_SCAN_ADD,
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
			cl_kernel kernel_scan = clCreateKernel(program, "{{KERNEL_NAME}}_Scan", &err);
			CL_CHECK_ERROR(err, "Error creating Scan kernel '{{KERNEL_NAME}}'");

			cl_kernel kernel_scan_update = clCreateKernel(program, "{{KERNEL_NAME}}_ScanUpdate", &err);
			CL_CHECK_ERROR(err, "Error creating Scan update kernel '{{KERNEL_NAME}}'");

			cl_kernel kernel_scan_add = clCreateKernel(program, "{{KERNEL_NAME}}_ScanAdd", &err);
			CL_CHECK_ERROR(err, "Error creating Scan add kernel '{{KERNEL_NAME}}'");

			kernels(counter, KERNEL_SCAN,        &kernel_scan);
			kernels(counter, KERNEL_SCAN_UPDATE, &kernel_scan_update);
			kernels(counter, KERNEL_SCAN_ADD,    &kernel_scan_add);
			counter++;
		}

		initialized = true;
	}

	static void scan
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu::backend::DeviceMemPointer_CL<{{SCAN_TYPE}}> *skepu_input, skepu::backend::DeviceMemPointer_CL<{{SCAN_TYPE}}> *skepu_output, skepu::backend::DeviceMemPointer_CL<{{SCAN_TYPE}}> *blockSums,
		size_t skepu_n, size_t skepu_numElements, size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_SCAN);
		skepu::backend::cl_helpers::setKernelArgs(kernel, skepu_input->getDeviceDataPointer(), skepu_output->getDeviceDataPointer(), blockSums->getDeviceDataPointer(), skepu_n, skepu_numElements);
		clSetKernelArg(kernel, 5, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Scan kernel");
	}

	static void scanUpdate
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu::backend::DeviceMemPointer_CL<{{SCAN_TYPE}}> *data, skepu::backend::DeviceMemPointer_CL<{{SCAN_TYPE}}> *sums,
		int isInclusive, {{SCAN_TYPE}} init, size_t skepu_n,
		skepu::backend::DeviceMemPointer_CL<{{SCAN_TYPE}}> *ret,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_SCAN_UPDATE);
		cl_mem retCL = (ret != nullptr) ? ret->getDeviceDataPointer() : NULL;
		skepu::backend::cl_helpers::setKernelArgs(kernel, data->getDeviceDataPointer(), sums->getDeviceDataPointer(), isInclusive, init, skepu_n, retCL);
		clSetKernelArg(kernel, 6, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Scan update kernel");
	}

	static void scanAdd
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu::backend::DeviceMemPointer_CL<{{SCAN_TYPE}}> *data,
		{{SCAN_TYPE}} skepu_sum, size_t skepu_n
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_SCAN_ADD);
		skepu::backend::cl_helpers::setKernelArgs(kernel, data->getDeviceDataPointer(), skepu_sum, skepu_n);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Scan add kernel");
	}
};
)~~~";


std::string createScanKernelProgram_CL(SkeletonInstance &instance, UserFunction &scanFunc, std::string dir)
{
	std::stringstream sourceStream;

	if (scanFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";

	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";

	// check for extra user-supplied opencl code for custome datatype
	// TODO: Also check the referenced UFs for referenced UTs skepu::userstruct
	for (UserType *RefType : scanFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);

	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_CL(scanFunc) << ScanKernel_CL << ScanUpdate_CL << ScanAdd_CL;

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_ScanKernel_" + scanFunc.uniqueName;
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor,
	{
		{"{{OPENCL_KERNEL}}", sourceStream.str()},
		{"{{KERNEL_CLASS}}",  "CLWrapperClass_" + kernelName},
		{"{{SCAN_TYPE}}",           scanFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",         kernelName},
		{"{{FUNCTION_NAME_SCAN}}",  scanFunc.uniqueName}
	});
	return kernelName;
}
