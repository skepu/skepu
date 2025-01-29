#pragma once

#include "globals.h"

extern llvm::cl::opt<std::string> ResultName;

using SkeletonInstance = std::string;

void generateUserFunctionStruct(UserFunction &UF, std::string InstanceName, clang::SourceLocation loc);

bool transformSkeletonInvocation(const Skeleton &skeleton, std::string InstanceName, std::vector<UserFunction*> FuncArgs, std::vector<size_t> arity, clang::VarDecl *d);

std::string replaceReferencesToOtherUFs(Backend backend, UserFunction &UF, std::function<std::string(UserFunction&)> nameFunc);

// CUDA generators
std::string createMapReduceKernelProgram_CU(SkeletonInstance&, UserFunction &mapFunc, UserFunction &reduceFunc, size_t arity, std::string dir);
std::string createMapKernelProgram_CU(SkeletonInstance&, UserFunction &mapFunc, size_t arity, std::string dir);
std::string createMapPairsKernelProgram_CU(SkeletonInstance&, UserFunction &mapPairsFunc, std::string dir);
std::string createMapPairsReduceKernelProgram_CU(SkeletonInstance&, UserFunction &mapFunc, UserFunction &reduceFunc, std::string dir);
std::string createScanKernelProgram_CU(SkeletonInstance&, UserFunction &scanFunc, std::string dir);
std::string createReduce1DKernelProgram_CU(SkeletonInstance&, UserFunction &reduceFunc, std::string dir);
std::string createReduce2DKernelProgram_CU(SkeletonInstance&, UserFunction &rowWiseFunc, UserFunction &colWiseFunc, std::string dir);
std::string createMapOverlap1DKernelProgram_CU(SkeletonInstance&, UserFunction &mapOverlapFunc, std::string dir);
std::string createMapOverlap2DKernelProgram_CU(SkeletonInstance&, UserFunction &mapOverlapFunc, std::string dir);
std::string createMapOverlap3DKernelProgram_CU(SkeletonInstance&, UserFunction &mapOverlapFunc, std::string dir);
std::string createMapOverlap4DKernelProgram_CU(SkeletonInstance&, UserFunction &mapOverlapFunc, std::string dir);
std::string createCallKernelProgram_CU(SkeletonInstance&, UserFunction &callFunc, std::string dir);


// OpenCL generators
std::string createMapReduceKernelProgram_CL(SkeletonInstance&, UserFunction &mapFunc, UserFunction &reduceFunc, std::string dir);
std::string createMapKernelProgram_CL(SkeletonInstance&, UserFunction &mapFunc, std::string dir);
std::string createMapPairsKernelProgram_CL(SkeletonInstance&, UserFunction &mapPairsFunc, std::string dir);
std::string createMapPairsReduceKernelProgram_CL(SkeletonInstance&, UserFunction &mapPairsFunc, UserFunction &reduceFunc, std::string dir);
std::string createScanKernelProgram_CL(SkeletonInstance&, UserFunction &scanFunc, std::string dir);
std::string createReduce1DKernelProgram_CL(SkeletonInstance&, UserFunction &reduceFunc, std::string dir);
std::string createReduce2DKernelProgram_CL(SkeletonInstance&, UserFunction &rowWiseFunc, UserFunction &colWiseFunc, std::string dir);
std::string createMapOverlap1DKernelProgram_CL(SkeletonInstance&, UserFunction &mapOverlapFunc, std::string dir);
std::string createMapOverlap2DKernelProgram_CL(SkeletonInstance&, UserFunction &mapOverlapFunc, std::string dir);
std::string createMapOverlap3DKernelProgram_CL(SkeletonInstance&, UserFunction &mapOverlapFunc, std::string dir);
std::string createMapOverlap4DKernelProgram_CL(SkeletonInstance&, UserFunction &mapOverlapFunc, std::string dir);
std::string createCallKernelProgram_CL(SkeletonInstance&, UserFunction &callFunc, std::string dir);
