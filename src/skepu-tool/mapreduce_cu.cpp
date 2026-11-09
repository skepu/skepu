#include <functional>
#include "code_gen.h"
#include "code_gen_cu.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapReduceKernelTemplate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}({{KERNEL_PARAMS}} size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base, skepu::StrideList<{{STRIDE_COUNT}}> skepu_strides)
{
	extern __shared__ {{REDUCE_RESULT_TYPE}} {{SHARED_BUFFER}}[];
	
	size_t skepu_global_prng_id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_blockSize = blockDim.x;
	size_t skepu_tid = threadIdx.x;
	size_t skepu_i = blockIdx.x * skepu_blockSize + skepu_tid;
	size_t skepu_gridSize = skepu_blockSize * gridDim.x;
	// (the threads that have a result are the first ones of the block; the others have nothing to combine)
	size_t skepu_blockStart = blockIdx.x * skepu_blockSize;
	size_t skepu_valid = (skepu_blockStart < skepu_n) ? ((skepu_n - skepu_blockStart < skepu_blockSize) ? skepu_n - skepu_blockStart : skepu_blockSize) : 0;
	{{REDUCE_RESULT_TYPE}} skepu_result;
	{{STRIDE_INIT}}

	if (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		skepu_result = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
		//{{OUTPUT_BINDINGS}}
		skepu_i += skepu_gridSize;
	}

	while (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		auto skepu_tempMap = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
		skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_tempMap);
		skepu_i += skepu_gridSize;
	}

	{{SHARED_BUFFER}}[skepu_tid] = skepu_result;
	__syncthreads();

	if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_tid], {{SHARED_BUFFER}}[skepu_tid + 512]); } __syncthreads(); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_tid], {{SHARED_BUFFER}}[skepu_tid + 256]); } __syncthreads(); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_tid], {{SHARED_BUFFER}}[skepu_tid + 128]); } __syncthreads(); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid +  64 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_tid], {{SHARED_BUFFER}}[skepu_tid +  64]); } __syncthreads(); }
	if (skepu_blockSize >=   64) { if (skepu_tid <  32 && skepu_tid +  32 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_tid], {{SHARED_BUFFER}}[skepu_tid +  32]); } __syncthreads(); }
	if (skepu_blockSize >=   32) { if (skepu_tid <  16 && skepu_tid +  16 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_tid], {{SHARED_BUFFER}}[skepu_tid +  16]); } __syncthreads(); }
	if (skepu_blockSize >=   16) { if (skepu_tid <   8 && skepu_tid +   8 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_tid], {{SHARED_BUFFER}}[skepu_tid +   8]); } __syncthreads(); }
	if (skepu_blockSize >=    8) { if (skepu_tid <   4 && skepu_tid +   4 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_tid], {{SHARED_BUFFER}}[skepu_tid +   4]); } __syncthreads(); }
	if (skepu_blockSize >=    4) { if (skepu_tid <   2 && skepu_tid +   2 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_tid], {{SHARED_BUFFER}}[skepu_tid +   2]); } __syncthreads(); }
	if (skepu_blockSize >=    2) { if (skepu_tid <   1 && skepu_tid +   1 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = {{FUNCTION_NAME_REDUCE}}({{SHARED_BUFFER}}[skepu_tid], {{SHARED_BUFFER}}[skepu_tid +   1]); } __syncthreads(); }

	if (skepu_tid == 0)
		skepu_output[blockIdx.x] = {{SHARED_BUFFER}}[skepu_tid];
}
)~~~";

const char *ReduceKernelTemplate_CU = R"~~~(
__global__ void {{KERNEL_NAME}}({{REDUCE_RESULT_TYPE}} *skepu_input, {{REDUCE_RESULT_TYPE}} *skepu_output, size_t skepu_n, size_t skepu_blockSize, bool skepu_nIsPow2)
{
	extern __shared__ {{REDUCE_RESULT_TYPE}} {{SHARED_BUFFER}}[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	size_t skepu_tid = threadIdx.x;
	size_t skepu_i = blockIdx.x * skepu_blockSize*2 + threadIdx.x;
	size_t skepu_gridSize = skepu_blockSize * 2 * gridDim.x;
	// (the threads that have a result are the first ones of the block; the others have nothing to combine)
	size_t skepu_blockStart = blockIdx.x * skepu_blockSize * 2;
	size_t skepu_valid = (skepu_blockStart < skepu_n) ? ((skepu_n - skepu_blockStart < skepu_blockSize) ? skepu_n - skepu_blockStart : skepu_blockSize) : 0;
	{{REDUCE_RESULT_TYPE}} skepu_result;

	if(skepu_i < skepu_n)
	{
		skepu_result = skepu_input[skepu_i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		//This nIsPow2 opt is not valid when we use this kernel for sparse matrices as well where we
		// dont exactly now the elements when calculating thread- and block-size and nIsPow2 assum becomes invalid in some cases there which results in sever problems.
		// There we pass it always false
		if (skepu_nIsPow2 || skepu_i + skepu_blockSize < skepu_n)
			skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_input[skepu_i + skepu_blockSize]);
		skepu_i += skepu_gridSize;
	}

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a lParamer gridSize and therefore fewer elements per thread
	while(skepu_i < skepu_n)
	{
		skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_input[skepu_i]);
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (skepu_nIsPow2 || skepu_i + skepu_blockSize < skepu_n)
			skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_input[skepu_i + skepu_blockSize]);
		skepu_i += skepu_gridSize;
	}

	// each thread puts its local sum into shared memory
	{{SHARED_BUFFER}}[skepu_tid] = skepu_result;
	__syncthreads();

	// do reduction in shared mem
	if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid + 512]); } __syncthreads(); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid + 256]); } __syncthreads(); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid + 128]); } __syncthreads(); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid + 64 < skepu_valid) { {{SHARED_BUFFER}}[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, {{SHARED_BUFFER}}[skepu_tid +  64]); } __syncthreads(); }

	if (skepu_tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		// UPDATE: volatile causes issues with custom struct data types; use __syncwarp() instead
		/*volatile*/ {{REDUCE_RESULT_TYPE}}* skepu_smem = {{SHARED_BUFFER}};
		if (skepu_blockSize >=  64) { if (skepu_tid < 32 && skepu_tid + 32 < skepu_valid) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid + 32]); } __syncwarp(); }
		if (skepu_blockSize >=  32) { if (skepu_tid < 16 && skepu_tid + 16 < skepu_valid) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid + 16]); } __syncwarp(); }
		if (skepu_blockSize >=  16) { if (skepu_tid <  8 && skepu_tid + 8 < skepu_valid) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  8]); } __syncwarp(); }
		if (skepu_blockSize >=   8) { if (skepu_tid <  4 && skepu_tid + 4 < skepu_valid) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  4]); } __syncwarp(); }
		if (skepu_blockSize >=   4) { if (skepu_tid <  2 && skepu_tid + 2 < skepu_valid) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  2]); } __syncwarp(); }
		if (skepu_blockSize >=   2) { if (skepu_tid <  1 && skepu_tid + 1 < skepu_valid) { skepu_smem[skepu_tid] = skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_smem[skepu_tid +  1]); } __syncwarp(); }
	}

	// write result for this block to global mem
	if (skepu_tid == 0)
		skepu_output[blockIdx.x] = {{SHARED_BUFFER}}[0];
}
)~~~";


// Each returned value is reduced separately. The kernels keep one array of shared memory for each value, in the raw buffer that the host sizes for all of them,
// and write one partial result for each value of each block. The second kernel reduces those partial results in place, with one block.
static std::string sharedArrays_CU(const std::vector<std::string> &types)
{
	std::string text = "\tsize_t skepu_offset = 0;\n";
	for (size_t k = 0; k < types.size(); ++k)
		text += "\tskepu_offset = (skepu_offset + alignof(" + types[k] + ") - 1) / alignof(" + types[k] + ") * alignof(" + types[k] + ");\n"
			"\t" + types[k] + " *skepu_sdata_" + std::to_string(k) + " = (" + types[k] + " *)({{SHARED_BUFFER}} + skepu_offset);\n"
			"\tskepu_offset += sizeof(" + types[k] + ") * skepu_blockSize;\n";
	return text;
}

static std::string multiValueKernels_CU(UserFunction &mapFunc, const std::string &reduceKernelName)
{
	const std::vector<std::string> &types = mapFunc.multipleReturnTypes;
	auto name = [](const char *base, size_t k) { return std::string(base) + std::to_string(k); };
	auto eachValue = [&](const std::function<std::string(size_t)> &text)
	{
		std::string all;
		for (size_t k = 0; k < types.size(); ++k)
			all += text(k);
		return all;
	};

	std::string tree;
	for (size_t half = 512; half >= 1; half /= 2)
		tree += "\tif (skepu_blockSize >= " + std::to_string(2 * half) + ") { if (skepu_tid < " + std::to_string(half) + " && skepu_tid + " + std::to_string(half) + " < skepu_valid) { "
			+ eachValue([&](size_t k) { return name("skepu_sdata_", k) + "[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(" + name("skepu_sdata_", k) + "[skepu_tid], "
				+ name("skepu_sdata_", k) + "[skepu_tid + " + std::to_string(half) + "]); "; })
			+ "} __syncthreads(); }\n";

	const std::string results = eachValue([&](size_t k) { return "\t" + types[k] + " " + name("skepu_result_", k) + ";\n"; });
	const std::string first = eachValue([&](size_t k) { return "\t\t" + name("skepu_result_", k) + " = skepu_out_temp.e" + std::to_string(k) + ";\n"; });
	const std::string next = eachValue([&](size_t k) { return "\t\t" + name("skepu_result_", k) + " = {{FUNCTION_NAME_REDUCE}}(" + name("skepu_result_", k) + ", skepu_out_temp.e" + std::to_string(k) + ");\n"; });
	const std::string store = eachValue([&](size_t k) { return "\t" + name("skepu_sdata_", k) + "[skepu_tid] = " + name("skepu_result_", k) + ";\n"; });
	const std::string write = eachValue([&](size_t k) { return "\t\t" + name("skepu_output_", k) + "[" + "{{WRITE_INDEX}}" + "] = " + name("skepu_sdata_", k) + "[0];\n"; });
	const std::string firstOnly = eachValue([&](size_t k) { return "\t\t" + name("skepu_result_", k) + " = " + name("skepu_output_", k) + "[skepu_i];\n"; });
	const std::string nextOnly = eachValue([&](size_t k) { return "\t\t" + name("skepu_result_", k) + " = {{FUNCTION_NAME_REDUCE}}(" + name("skepu_result_", k) + ", " + name("skepu_output_", k) + "[skepu_i]);\n"; });
	std::string reduceParams;
	for (size_t k = 0; k < types.size(); ++k)
		reduceParams += types[k] + " *" + name("skepu_output_", k) + ", ";

	auto replaceAll = [](std::string text, const std::string &from, const std::string &to)
	{
		for (size_t at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size()))
			text.replace(at, from.size(), to);
		return text;
	};
	const std::string arrays = sharedArrays_CU(types);

	const std::string mapReduce = std::string(R"~~~(
__global__ void {{KERNEL_NAME}}({{KERNEL_PARAMS}} size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base, skepu::StrideList<{{STRIDE_COUNT}}> skepu_strides)
{
	extern __shared__ char {{SHARED_BUFFER}}[];

	size_t skepu_global_prng_id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_blockSize = blockDim.x;
	size_t skepu_tid = threadIdx.x;
	size_t skepu_i = blockIdx.x * skepu_blockSize + skepu_tid;
	size_t skepu_gridSize = skepu_blockSize * gridDim.x;
	size_t skepu_blockStart = blockIdx.x * skepu_blockSize;
	size_t skepu_valid = (skepu_blockStart < skepu_n) ? ((skepu_n - skepu_blockStart < skepu_blockSize) ? skepu_n - skepu_blockStart : skepu_blockSize) : 0;
)~~~") + results + arrays + R"~~~(	{{STRIDE_INIT}}

	if (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		auto skepu_out_temp = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
)~~~" + first + R"~~~(		skepu_i += skepu_gridSize;
	}

	while (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		auto skepu_out_temp = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
)~~~" + next + R"~~~(		skepu_i += skepu_gridSize;
	}

)~~~" + store + "\t__syncthreads();\n\n" + tree + R"~~~(
	if (skepu_tid == 0)
	{
)~~~" + replaceAll(write, "{{WRITE_INDEX}}", "blockIdx.x") + R"~~~(	}
}
)~~~";

	const std::string reduceOnly = std::string("\n__global__ void ") + reduceKernelName + "(" + reduceParams + R"~~~(size_t skepu_n)
{
	extern __shared__ char {{SHARED_BUFFER}}[];

	size_t skepu_blockSize = blockDim.x;
	size_t skepu_tid = threadIdx.x;
	size_t skepu_i = threadIdx.x;
	size_t skepu_valid = (skepu_n < skepu_blockSize) ? skepu_n : skepu_blockSize;
)~~~" + results + arrays + R"~~~(
	if (skepu_i < skepu_n)
	{
)~~~" + firstOnly + R"~~~(		skepu_i += skepu_blockSize;
	}

	while (skepu_i < skepu_n)
	{
)~~~" + nextOnly + R"~~~(		skepu_i += skepu_blockSize;
	}

)~~~" + store + "\t__syncthreads();\n\n" + tree + R"~~~(
	if (skepu_tid == 0)
	{
)~~~" + replaceAll(write, "{{WRITE_INDEX}}", "0") + R"~~~(	}
}
)~~~";

	return mapReduce + reduceOnly;
}

std::string createMapReduceKernelProgram_CU(SkeletonInstance &instance, UserFunction &mapFunc, UserFunction &reduceFunc, size_t arity, std::string dir)
{
	std::stringstream SSKernelParamList, SSMapFuncArgs, SSStrideInit, SSStrideCount;
	IndexCodeGen indexInfo = indexInitHelper_CU(mapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CU(mapFunc, SSKernelParamList);
	handleRandomParam_CU(mapFunc, SSMapFuncArgs, SSKernelParamList, first);
	
	size_t stride_counter = 0;
	for (UserFunction::Param& param : mapFunc.elwiseParams)
	{
		if (!first) { SSMapFuncArgs << ", "; }
		SSStrideInit << "if (skepu_strides[" << stride_counter << "] < 0) { " << param.name << " += (-skepu_n + 1) * skepu_strides[" << stride_counter << "]; }\n";
		SSKernelParamList << param.resolvedTypeName << " *" << param.name << ", ";
		SSMapFuncArgs << param.name << "[skepu_i * skepu_strides[" << stride_counter++ << "]]";
		first = false;
	}
	auto argsInfo = handleRandomAccessAndUniforms_CU(mapFunc, SSMapFuncArgs, SSKernelParamList, first);

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_MapReduceKernel_" + mapFunc.uniqueName + "_" + reduceFunc.uniqueName;
	SSStrideCount << mapFunc.elwiseParams.size();
	
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	const bool multipleValues = mapFunc.multipleReturnTypes.size() > 0;
	const std::vector<std::pair<std::string, std::string>> kernelValues =
	{
		{"{{REDUCE_RESULT_TYPE}}",   reduceFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName},
		{"{{FUNCTION_NAME_MAP}}",    mapFunc.funcNameCUDA()},
		{"{{FUNCTION_NAME_REDUCE}}", reduceFunc.funcNameCUDA()},
		{"{{KERNEL_PARAMS}}",        SSKernelParamList.str()},
		{"{{MAP_ARGS}}",             SSMapFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}",    indexInfo.indexInit},
		{"{{OUTPUT_BINDINGS}}",      multiOutputAssign},
		{"{{PROXIES_UPDATE}}",       argsInfo.proxyInitializerInner},
		{"{{PROXIES_INIT}}",         argsInfo.proxyInitializer},
		{"{{STRIDE_COUNT}}",         SSStrideCount.str()},
		{"{{STRIDE_INIT}}",          SSStrideInit.str()},
		{"{{SHARED_BUFFER}}",        "sdata_" + instance}
	};
	if (multipleValues)
		FSOutFile << templateString(multiValueKernels_CU(mapFunc, kernelName + "_ReduceOnly"), kernelValues);
	else
	{
		FSOutFile << templateString(MapReduceKernelTemplate_CU, kernelValues);
		FSOutFile << templateString(ReduceKernelTemplate_CU,
		{
			{"{{REDUCE_RESULT_TYPE}}",   reduceFunc.resolvedReturnTypeName},
			{"{{KERNEL_NAME}}",          kernelName + "_ReduceOnly"},
			{"{{FUNCTION_NAME_REDUCE}}", reduceFunc.funcNameCUDA()},
			{"{{SHARED_BUFFER}}",        "sdata_" + instance}
		});
	}
	return kernelName;
}
