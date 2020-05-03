#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapPairsKernelTemplate_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME(SKEPU_KERNEL_PARAMS size_t Vsize, size_t Hsize, size_t skepu_base)
{
	size_t skepu_n = Vsize * Hsize;
	size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_gridSize = blockDim.x * gridDim.x;

	while (skepu_i < skepu_n)
	{
		SKEPU_INDEX_INITIALIZER
		skepu_output[skepu_i] = SKEPU_FUNCTION_NAME_MAPPAIRS(SKEPU_MAPPAIRS_PARAMS);
		skepu_i += skepu_gridSize;
	}
}
)~~~";


std::string createMapPairsKernelProgram_CU(UserFunction &mapPairsFunc, std::string dir)
{
	std::stringstream sourceStream, SSKernelParamList, SSMapPairsFuncParams;
	std::string indexInitializer;
	bool first = true;
	
	if (mapPairsFunc.indexed1D)
	{
		SSMapPairsFuncParams << "skepu_index";
		indexInitializer = "skepu::Index1D skepu_index;\n\t\tskepu_index.i = base + skepu_i;";
		first = false;
	}
	else if (mapPairsFunc.indexed2D)
	{
		SSMapPairsFuncParams << "skepu_index";
		indexInitializer = "skepu::Index2D skepu_index;\n\t\tskepu_index.row = (skepu_base + skepu_i) / Hsize;\n\t\tskepu_index.col = (skepu_base + skepu_i) % Hsize;";
		first = false;
	}
	
	// Output data
	if (mapPairsFunc.multipleReturnTypes.size() == 0)
		SSKernelParamList << mapPairsFunc.resolvedReturnTypeName << "* skepu_output, ";
	else
	{
		size_t outCtr = 0;
		for (std::string& outputType : mapPairsFunc.multipleReturnTypes)
			SSKernelParamList << outputType << "* skepu_output_" << outCtr++ << ", ";
	}
	
	size_t ctr = 0;
	for (UserFunction::Param& param : mapPairsFunc.elwiseParams)
	{
		if (!first) { SSMapPairsFuncParams << ", "; }
		SSKernelParamList << param.resolvedTypeName << " *" << param.name << ", ";
		if (ctr++ < mapPairsFunc.Varity) // vertical containers
			SSMapPairsFuncParams << param.name << "[skepu_i / Hsize]";
		else // horizontal containers
			SSMapPairsFuncParams << param.name << "[skepu_i % Hsize]";
		first = false;
	}
	
	for (UserFunction::RandomAccessParam& param : mapPairsFunc.anyContainerParams)
	{
		if (!first) { SSMapPairsFuncParams << ", "; }
		SSKernelParamList << param.fullTypeName << " " << param.name << ", ";
		SSMapPairsFuncParams << param.name;
		first = false;
	}
	
	for (UserFunction::Param& param : mapPairsFunc.anyScalarParams)
	{
		if (!first) { SSMapPairsFuncParams << ", "; }
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSMapPairsFuncParams << param.name;
		first = false;
	}
	
	std::stringstream SSKernelName;
	SSKernelName << transformToCXXIdentifier(ResultName) << "_MapPairsKernel_" << mapPairsFunc.uniqueName << "_Varity_" << mapPairsFunc.Varity << "_Harity_" << mapPairsFunc.Harity;
	const std::string kernelName = SSKernelName.str();
	
	std::string kernelSource = MapPairsKernelTemplate_CU;
	replaceTextInString(kernelSource, PH_KernelName, kernelName);
	replaceTextInString(kernelSource, PH_MapPairsFuncName, mapPairsFunc.funcNameCUDA());
	replaceTextInString(kernelSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(kernelSource, PH_MapPairsParams, SSMapPairsFuncParams.str());
	replaceTextInString(kernelSource, PH_IndexInitializer, indexInitializer);
	
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << kernelSource;
	return kernelName;
}