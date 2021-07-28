#pragma once

#include <map>
#include <sstream>

#include "code_gen.h"
#include "data_structures.h"


std::string generateOpenCLVectorProxy(UserFunction::RandomAccessParam const& param);
std::string generateOpenCLMatrixProxy(UserFunction::RandomAccessParam const& param);
std::string generateOpenCLMatrixRowProxy(UserFunction::RandomAccessParam const& param);
std::string generateOpenCLMatrixColProxy(UserFunction::RandomAccessParam const& param);
std::string generateOpenCLSparseMatrixProxy(UserFunction::RandomAccessParam const& param);
std::string generateOpenCLTensor3Proxy(UserFunction::RandomAccessParam const& param);
std::string generateOpenCLTensor4Proxy(UserFunction::RandomAccessParam const& param);

std::string generateOpenCLRegion(size_t dim, UserFunction::RegionParam const& param);
std::string generateOpenCLRandom();

std::string generateOpenCLMultipleReturn(UserFunction &UF);
std::string generateUserFunctionCode_CL(UserFunction &Func);
std::string generateUserTypeCode_CL(UserType &Type);

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


void proxyCodeGenHelper_CL(std::map<ContainerType, std::unordered_set<UserFunction::RandomAccessParam const*>> containerProxyTypes, std::stringstream &sourceStream);


struct RandomAccessAndScalarsResult
{
	std::map<ContainerType, std::unordered_set<UserFunction::RandomAccessParam const*>> containerProxyTypes;
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

void handleRandomParam_CL(
	UserFunction &func,
	std::stringstream& sourceStream,
	std::stringstream& SSMapFuncArgs,
	std::stringstream& SSHostKernelParamList,
	std::stringstream& SSKernelParamList,
	std::stringstream& SSKernelArgs,
	bool &first
);
void handleUserTypesConstantsAndPrecision_CL(std::vector<UserFunction const*> funcs, std::stringstream &sourceStream);

std::string handleOutputs_CL(UserFunction &func, std::stringstream &SSHostKernelParamList, std::stringstream &SSKernelParamList, std::stringstream &SSKernelArgs, bool strided = false, std::string index = "skepu_i");
