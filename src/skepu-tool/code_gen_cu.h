#pragma once

#include <map>
#include <sstream>

#include "code_gen.h"
#include "data_structures.h"


struct IndexCodeGen
{
	std::string sizesTupleParam;
	std::string sizeParams;
	std::string sizeArgs;
	std::string indexInit;
	std::string mapFuncParam;
	std::string templateHeader;
	bool hasIndex = false;
	size_t dim;
};

IndexCodeGen indexInitHelper_CU(UserFunction &uf);


struct RandomAccessAndScalarsResult
{
	std::map<ContainerType, std::set<std::string>> containerProxyTypes;
	std::string proxyInitializer;
	std::string proxyInitializerInner;
};

RandomAccessAndScalarsResult handleRandomAccessAndUniforms_CU(
	UserFunction &func,
	std::stringstream& SSMapFuncArgs,
	std::stringstream& SSKernelParamList,
	bool &first,
	bool elementIndexIsLinear = false // (Map: skepu_i is the position in the output and the kernel has skepu_base and skepu_w2, as the OpenCL kernels)
);

void handleRandomParam_CU(
	UserFunction &func,
	std::stringstream& SSMapFuncArgs,
	std::stringstream& SSKernelParamList,
	bool &first
);

std::string handleOutputs_CU(UserFunction &func, std::stringstream &SSKernelParamList, bool strided = false, std::string index = "skepu_i");
std::string generateCUDAMultipleReturn(UserFunction &UF);