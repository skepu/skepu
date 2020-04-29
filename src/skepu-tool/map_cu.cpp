#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapKernelTemplate_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME(SKEPU_KERNEL_PARAMS size_t w2, size_t w3, size_t w4, size_t n, size_t base)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;

	while (i < n)
	{
		SKEPU_INDEX_INITIALIZER
		auto res = SKEPU_FUNCTION_NAME_MAP(SKEPU_MAP_PARAMS);
		SKEPU_OUTPUT_BINDINGS
		i += gridSize;
	}
}
)~~~";


std::string createMapKernelProgram_CU(UserFunction &mapFunc, size_t arity, std::string dir)
{
	std::stringstream sourceStream, SSKernelParamList, SSMapFuncParams, SSOutputBindings;
	std::string indexInitializer;
	bool first = true;
	
	if (mapFunc.indexed1D || mapFunc.indexed2D || mapFunc.indexed3D || mapFunc.indexed4D)
	{
		SSMapFuncParams << "index";
		first = false;
	}
	
	if      (mapFunc.indexed1D) indexInitializer = "skepu::Index1D index;\nindex.i = base + i;";
	else if (mapFunc.indexed2D) indexInitializer = "skepu::Index2D index;\nindex.row = (base + i) / w2;\nindex.col = (base + i) % w2;";
	else if (mapFunc.indexed3D) indexInitializer = R"~~~(
		skepu::Index3D index;
		size_t cindex = base + i;
		index.i = cindex / (w2 * w3);
		cindex = cindex % (w2 * w3);
		index.j = cindex / (w3);
		index.k = cindex % (w3);
	)~~~";
	
	else if (mapFunc.indexed4D) indexInitializer = R"~~~(
		skepu::Index4D index;
		size_t cindex = base + i;
		
		index.i = cindex / (w2 * w3 * w4);
		cindex = cindex % (w2 * w3 * w4);
		
		index.j = cindex / (w3 * w4);
		cindex = cindex % (w3 * w4);
		
		index.k = cindex / (w4);
		index.l = cindex % (w4);
	)~~~";
	
	// Output data
	if (mapFunc.multipleReturnTypes.size() == 0)
	{
		SSKernelParamList << mapFunc.resolvedReturnTypeName << "* skepu_output, ";
		SSOutputBindings << "skepu_output[i] = res;";
	}
	else
	{
		size_t outCtr = 0;
		for (std::string& outputType : mapFunc.multipleReturnTypes)
		{
			SSKernelParamList << outputType << "* skepu_output_" << outCtr << ", ";
			SSOutputBindings << "skepu_output_" << outCtr << "[i] = std::get<" << outCtr << ">(res);\n";
			outCtr++;
		}
	}

	for (UserFunction::Param& param : mapFunc.elwiseParams)
	{
		if (!first) { SSMapFuncParams << ", "; }
		SSKernelParamList << param.resolvedTypeName << " *" << param.name << ", ";
		SSMapFuncParams << param.name << "[i]";
		first = false;
	}

	for (UserFunction::RandomAccessParam& param : mapFunc.anyContainerParams)
	{
		if (!first) { SSMapFuncParams << ", "; }
		SSKernelParamList << param.fullTypeName << " " << param.name << ", ";
		SSMapFuncParams << param.name;
		first = false;
	}

	for (UserFunction::Param& param : mapFunc.anyScalarParams)
	{
		if (!first) { SSMapFuncParams << ", "; }
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSMapFuncParams << param.name;
		first = false;
	}

	const std::string kernelName = ResultName + "_MapKernel_" + mapFunc.uniqueName;

	std::string kernelSource = MapKernelTemplate_CU;
	replaceTextInString(kernelSource, PH_MapResultType, mapFunc.resolvedReturnTypeName);
	replaceTextInString(kernelSource, PH_KernelName, kernelName);
	replaceTextInString(kernelSource, PH_MapFuncName, mapFunc.funcNameCUDA());
	replaceTextInString(kernelSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(kernelSource, PH_MapParams, SSMapFuncParams.str());
	replaceTextInString(kernelSource, PH_IndexInitializer, indexInitializer);
	replaceTextInString(kernelSource, "SKEPU_OUTPUT_BINDINGS", SSOutputBindings.str());

	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << kernelSource;
	return kernelName;
}
