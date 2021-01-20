#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const std::string ScanKernel_CU = R"~~~(
__global__ void {{KERNEL_NAME}}_ScanKernel({{SCAN_TYPE}}* skepu_input, {{SCAN_TYPE}}* skepu_output, {{SCAN_TYPE}}* blockSums, size_t skepu_n, size_t skepu_numElements)
{
	extern __shared__ {{SCAN_TYPE}} skepu_sdata[];

	size_t skepu_tid = threadIdx.x;
	unsigned int skepu_pout = 0;
	unsigned int skepu_pin = 1;
	size_t skepu_mem = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;
	size_t numBlocks = skepu_numElements / (blockDim.x) + (skepu_numElements % (blockDim.x) == 0 ? 0:1);
	size_t skepu_offset;

	for (size_t blockNr = blockIdx.x; blockNr < numBlocks; blockNr += gridDim.x)
	{
		skepu_sdata[skepu_pout * skepu_n + skepu_tid] = (skepu_mem < skepu_numElements) ? skepu_input[skepu_mem] : 0;

		__syncthreads();

		for (skepu_offset = 1; skepu_offset < skepu_n; skepu_offset *=2)
		{
			skepu_pout = 1 - skepu_pout;
			skepu_pin = 1 - skepu_pout;
			if (skepu_tid >= skepu_offset)
				skepu_sdata[skepu_pout * skepu_n + skepu_tid] = {{FUNCTION_NAME_SCAN}}(skepu_sdata[skepu_pin * skepu_n + skepu_tid], skepu_sdata[skepu_pin * skepu_n + skepu_tid - skepu_offset]);
			else
				skepu_sdata[skepu_pout * skepu_n + skepu_tid] = skepu_sdata[skepu_pin * skepu_n + skepu_tid];

			__syncthreads();
		}

		if (skepu_tid == blockDim.x - 1)
			blockSums[blockNr] = skepu_sdata[skepu_pout * skepu_n + blockDim.x-1];

		if (skepu_mem < skepu_numElements)
			skepu_output[skepu_mem] = skepu_sdata[skepu_pout * skepu_n + skepu_tid];

		skepu_mem += gridSize;

		__syncthreads();
	}
}
)~~~";

const std::string ScanUpdate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}_ScanUpdate({{SCAN_TYPE}} *data, {{SCAN_TYPE}} *sums, int isInclusive, {{SCAN_TYPE}} skepu_init, size_t skepu_n, {{SCAN_TYPE}} *skepu_ret)
{
	extern __shared__ {{SCAN_TYPE}} skepu_sdata[];

	__shared__ {{SCAN_TYPE}} skepu_offset;
	__shared__ {{SCAN_TYPE}} inc_offset;

	size_t skepu_tid = threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;
	size_t skepu_mem = blockIdx.x * blockDim.x + threadIdx.x;
	size_t numBlocks = skepu_n / (blockDim.x) + (skepu_n % (blockDim.x) == 0 ? 0:1);

	for (size_t blockNr = blockIdx.x; blockNr < numBlocks; blockNr += gridDim.x)
	{
		if (skepu_tid == 0)
		{
			if (isInclusive == 0)
			{
				skepu_offset = skepu_init;
				if (blockNr > 0)
				{
					skepu_offset = {{FUNCTION_NAME_SCAN}}(skepu_offset, sums[blockNr-1]);
					inc_offset = sums[blockNr - 1];
				}
			}
			else
			{
				if (blockNr > 0)
					skepu_offset = sums[blockNr - 1];
			}
		}

		__syncthreads();

		if (isInclusive == 1)
		{
			if (blockNr > 0)
				skepu_sdata[skepu_tid] = (skepu_mem < skepu_n) ? {{FUNCTION_NAME_SCAN}}(skepu_offset, data[skepu_mem]) : 0;
			else
				skepu_sdata[skepu_tid] = (skepu_mem < skepu_n) ? data[skepu_mem] : 0;

			if (skepu_mem == skepu_n - 1)
				*skepu_ret = skepu_sdata[skepu_tid];
		}
		else
		{
			if (skepu_mem == skepu_n - 1)
				*skepu_ret = {{FUNCTION_NAME_SCAN}}(inc_offset, data[skepu_mem]);

			if (skepu_tid == 0)
				skepu_sdata[skepu_tid] = skepu_offset;
			else
				skepu_sdata[skepu_tid] = (skepu_mem-1 < skepu_n) ? {{FUNCTION_NAME_SCAN}}(skepu_offset, data[skepu_mem - 1]) : 0;
		}

		__syncthreads();

		if (skepu_mem < skepu_n)
			data[skepu_mem] = skepu_sdata[skepu_tid];

		skepu_mem += gridSize;

		__syncthreads();
	}
}
)~~~";


const std::string ScanAdd_CU = R"~~~(
__global__ void {{KERNEL_NAME}}_ScanAdd({{SCAN_TYPE}} *data, {{SCAN_TYPE}} sum, size_t skepu_n)
{
	size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;

	while (skepu_i < skepu_n)
	{
		data[skepu_i] = {{FUNCTION_NAME_SCAN}}(data[skepu_i], sum);
		skepu_i += gridSize;
	}
}
)~~~";


std::string createScanKernelProgram_CU(UserFunction &scanFunc, std::string dir)
{
	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_Scan_" + scanFunc.uniqueName;
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(ScanKernel_CU + ScanUpdate_CU + ScanAdd_CU,
	{
		{"{{SCAN_TYPE}}",          scanFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",        kernelName},
		{"{{FUNCTION_NAME_SCAN}}", scanFunc.funcNameCUDA()}
	});
	return kernelName;
}
