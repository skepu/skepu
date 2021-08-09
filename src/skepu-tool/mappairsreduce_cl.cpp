#include "code_gen.h"
#include "code_gen_cl.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapPairsReduceKernelTemplate_CL = R"~~~(
#define skepu_w2 skepu_Hsize
__kernel void {{KERNEL_NAME}}({{KERNEL_PARAMS}} size_t skepu_n, size_t skepu_Vsize, size_t skepu_Hsize, size_t skepu_base, int skepu_transposed, __local {{REDUCE_RESULT_TYPE}}* skepu_sdata)
{
	size_t skepu_global_prng_id = get_global_id(0);
	size_t skepu_blockSize = get_local_size(0);
	size_t skepu_tid = get_local_id(0);
	{{CONTAINER_PROXIES}}
	{{REDUCE_RESULT_TYPE}} skepu_result;
	
	size_t skepu_thread_V = get_group_id(0);
	size_t skepu_thread_H = get_local_id(0);
	
	size_t skepu_lookup_V = (skepu_transposed == 0) ? skepu_thread_V : skepu_thread_H;
	size_t skepu_lookup_H = (skepu_transposed == 0) ? skepu_thread_H : skepu_thread_V;
	
	if (skepu_thread_H < skepu_Hsize)
	{
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
#if !{{USE_MULTIRETURN}}
		skepu_result = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
		skepu_thread_H += skepu_blockSize;
		if (!skepu_transposed)
			skepu_lookup_H += skepu_blockSize;
		else
			skepu_lookup_V += skepu_blockSize;
	}
	
	while (skepu_thread_H < skepu_Hsize)
	{
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
#if !{{USE_MULTIRETURN}}
		{{MAPPAIRS_RESULT_TYPE}} tempMap = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
		skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, tempMap);
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAPPAIRS}}({{MAPPAIRS_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
		skepu_thread_H += skepu_blockSize;
		if (!skepu_transposed)
			skepu_lookup_H += skepu_blockSize;
		else
			skepu_lookup_V += skepu_blockSize;
	}
	
	skepu_sdata[skepu_tid] = skepu_result;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_Hsize) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_Hsize) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_Hsize) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid +  64 < skepu_Hsize) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   64) { if (skepu_tid <  32 && skepu_tid +  32 < skepu_Hsize) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   32) { if (skepu_tid <  16 && skepu_tid +  16 < skepu_Hsize) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   16) { if (skepu_tid <   8 && skepu_tid +   8 < skepu_Hsize) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    8) { if (skepu_tid <   4 && skepu_tid +   4 < skepu_Hsize) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    4) { if (skepu_tid <   2 && skepu_tid +   2 < skepu_Hsize) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    2) { if (skepu_tid <   1 && skepu_tid +   1 < skepu_Hsize) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }
	

	if (skepu_tid == 0)
	{
		skepu_output[skepu_thread_V] = skepu_sdata[skepu_tid];
	}
}
)~~~";


const std::string Constructor = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	static cl_kernel skepu_kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
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
			cl_kernel kernel_mappairsreduce = clCreateKernel(program, "{{KERNEL_NAME}}", &err);
			CL_CHECK_ERROR(err, "Error creating MapPairsReduce kernel '{{KERNEL_NAME}}'");

			skepu_kernels(counter++, &kernel_mappairsreduce);
		}

		initialized = true;
	}
	
	static void mapPairsReduce
	(
		size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize, {{HOST_KERNEL_PARAMS}}
		size_t skepu_n, size_t skepu_Vsize, size_t skepu_Hsize, size_t skepu_base, int skepu_transposed,
		size_t skepu_sharedMemSize
	)
	{
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernels(skepu_deviceID), {{KERNEL_ARGS}} skepu_n, skepu_Vsize, skepu_Hsize, skepu_base, skepu_transposed);
		clSetKernelArg(skepu_kernels(skepu_deviceID), {{KERNEL_ARG_COUNT}} + 5, skepu_sharedMemSize, NULL);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernels(skepu_deviceID), 1, NULL, &skepu_globalSize, &skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching MapPairsReduce kernel");
	}
	
	
};
)~~~";


std::string createMapPairsReduceKernelProgram_CL(SkeletonInstance &instance, UserFunction &mapPairsFunc, UserFunction &reduceFunc, std::string dir)
{
	std::stringstream sourceStream, SSMapPairsFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs;
	std::string indexInit = "";
	if (mapPairsFunc.indexed2D)
	{
		indexInit = "index2_t skepu_index = { .row = skepu_lookup_V, .col = skepu_lookup_H };";
		SSMapPairsFuncArgs << "skepu_index";
	}
	IndexCodeGen indexInfo = indexInitHelper_CL(mapPairsFunc);
	bool first = !indexInfo.hasIndex;
	std::string multiOutputAssign = handleOutputs_CL(mapPairsFunc, SSHostKernelParamList, SSKernelParamList, SSKernelArgs);
	handleRandomParam_CL(mapPairsFunc, sourceStream, SSMapPairsFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	
	size_t ctr = 0;
	for (UserFunction::Param& param : mapPairsFunc.elwiseParams)
	{
		if (!first) { SSMapPairsFuncArgs << ", "; }
		SSKernelParamList << "__global " << param.resolvedTypeName << " *" << param.name << ", ";
		SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<" << param.resolvedTypeName << "> *" << param.name << ", ";
		SSKernelArgs << param.name << "->getDeviceDataPointer(), ";
		if (ctr++ < mapPairsFunc.Varity) // vertical containers
			SSMapPairsFuncArgs << param.name << "[skepu_lookup_V]";
		else // horizontal containers
			SSMapPairsFuncArgs << param.name << "[skepu_lookup_H]";
		first = false;
	}
	
	auto argsInfo = handleRandomAccessAndUniforms_CL(mapPairsFunc, SSMapPairsFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	handleUserTypesConstantsAndPrecision_CL({&mapPairsFunc}, sourceStream);
	proxyCodeGenHelper_CL(argsInfo.containerProxyTypes, sourceStream);
	
	if (mapPairsFunc.refersTo(reduceFunc))
		sourceStream << generateUserFunctionCode_CL(mapPairsFunc);
	else if (reduceFunc.refersTo(mapPairsFunc))
		sourceStream << generateUserFunctionCode_CL(reduceFunc);
	else
		sourceStream << generateUserFunctionCode_CL(mapPairsFunc) << generateUserFunctionCode_CL(reduceFunc);
	sourceStream << MapPairsReduceKernelTemplate_CL;
	
	std::stringstream SSKernelName;
	SSKernelName << instance << "_" << transformToCXXIdentifier(ResultName) << "_MapPairsReduceKernel_" << mapPairsFunc.uniqueName << "_Varity_" << mapPairsFunc.Varity << "_Harity_" << mapPairsFunc.Harity;
	const std::string kernelName = SSKernelName.str();
	
	std::stringstream SSKernelArgCount;
	SSKernelArgCount << mapPairsFunc.numKernelArgsCL();
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor,
	{
		{"{{OPENCL_KERNEL}}",           sourceStream.str()},
		{"{{KERNEL_NAME}}",             kernelName},
		{"{{FUNCTION_NAME_MAPPAIRS}}",  mapPairsFunc.uniqueName},
		{"{{KERNEL_PARAMS}}",           SSKernelParamList.str()},
		{"{{HOST_KERNEL_PARAMS}}",      SSHostKernelParamList.str()},
		{"{{MAPPAIRS_ARGS}}",           SSMapPairsFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}",       indexInit},
		{"{{KERNEL_CLASS}}",            "CLWrapperClass_" + kernelName},
		{"{{KERNEL_ARGS}}",             SSKernelArgs.str()},
		{"{{KERNEL_ARG_COUNT}}",        SSKernelArgCount.str()},
		{"{{CONTAINER_PROXIES}}",       argsInfo.proxyInitializer},
		{"{{CONTAINER_PROXIE_INNER}}",  argsInfo.proxyInitializerInner},
		{"{{MULTI_TYPE}}",              mapPairsFunc.multiReturnTypeNameGPU()},
		{"{{USE_MULTIRETURN}}",         (mapPairsFunc.multipleReturnTypes.size() > 0) ? "1" : "0"},
		{"{{OUTPUT_ASSIGN}}",           multiOutputAssign},
		
		{"{{MAPPAIRS_RESULT_TYPE}}",    mapPairsFunc.rawReturnTypeName},
		{"{{REDUCE_RESULT_TYPE}}",      reduceFunc.rawReturnTypeName},
		{"{{REDUCE_RESULT_CPU}}",       reduceFunc.resolvedReturnTypeName},
		{"{{FUNCTION_NAME_REDUCE}}",    reduceFunc.uniqueName},
	});
	
	return kernelName;
}
