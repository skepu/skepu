#pragma once

#include <map>
#include <sstream>

#include "code_gen.h"
#include "data_structures.h"


std::string generateOpenCLVectorProxy(std::string typeName);
std::string generateOpenCLMatrixProxy(std::string typeName);
std::string generateOpenCLMatrixRowProxy(std::string typeName);
std::string generateOpenCLMatrixColProxy(std::string typeName);
std::string generateOpenCLSparseMatrixProxy(std::string typeName);
std::string generateOpenCLTensor3Proxy(std::string typeName);
std::string generateOpenCLTensor4Proxy(std::string typeName);

std::string generateOpenCLRegion(size_t dim, std::string typeName);

std::string generateOpenCLMultipleReturn(std::vector<std::string> &types);

extern const std::string KernelPredefinedTypes_CL;

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

IndexCodeGen indexInitHelper_CL(UserFunction &uf);


void proxyCodeGenHelper_CL(std::map<ContainerType, std::set<std::string>> containerProxyTypes, std::stringstream &sourceStream);


struct RandomAccessAndScalarsResult
{
	std::map<ContainerType, std::set<std::string>> containerProxyTypes;
	std::string proxyInitializer;
	std::string proxyInitializerInner;
};

RandomAccessAndScalarsResult handleRandomAccessAndUniforms_CL(
	UserFunction &func,
	std::stringstream& SSMapFuncArgs,
	std::stringstream& SSHostKernelParamList,
	std::stringstream& SSKernelParamList,
	std::stringstream& SSKernelArgs,
	bool &first
);

void handleUserTypesConstantsAndPrecision_CL(std::vector<UserFunction const*> funcs, std::stringstream &sourceStream);

std::string handleOutputs_CL(UserFunction &func, std::stringstream &SSHostKernelParamList, std::stringstream &SSKernelParamList, std::stringstream &SSKernelArgs, std::string index = "skepu_i");
