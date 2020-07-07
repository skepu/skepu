#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapReduceKernelTemplate_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME(SKEPU_KERNEL_PARAMS __global SKEPU_REDUCE_RESULT_TYPE* skepu_output, SKEPU_SIZE_PARAMS size_t skepu_n, size_t skepu_base, __local SKEPU_REDUCE_RESULT_TYPE* skepu_sdata)
{
	size_t skepu_blockSize = get_local_size(0);
	size_t skepu_tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * skepu_blockSize + skepu_tid;
	size_t skepu_gridSize = skepu_blockSize * get_num_groups(0);
	SKEPU_REDUCE_RESULT_TYPE skepu_result;
	SKEPU_CONTAINER_PROXIES

	if (skepu_i < skepu_n)
	{
		SKEPU_INDEX_INITIALIZER
		SKEPU_CONTAINER_PROXIE_INNER
		skepu_result = SKEPU_FUNCTION_NAME_MAP(SKEPU_MAP_PARAMS);
		skepu_i += skepu_gridSize;
	}

	while (skepu_i < skepu_n)
	{
		SKEPU_INDEX_INITIALIZER
		SKEPU_CONTAINER_PROXIE_INNER
		SKEPU_MAP_RESULT_TYPE tempMap = SKEPU_FUNCTION_NAME_MAP(SKEPU_MAP_PARAMS);
		skepu_result = SKEPU_FUNCTION_NAME_REDUCE(skepu_result, tempMap);
		skepu_i += skepu_gridSize;
	}

	skepu_sdata[skepu_tid] = skepu_result;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid +  64 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   64) { if (skepu_tid <  32 && skepu_tid +  32 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   32) { if (skepu_tid <  16 && skepu_tid +  16 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   16) { if (skepu_tid <   8 && skepu_tid +   8 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    8) { if (skepu_tid <   4 && skepu_tid +   4 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    4) { if (skepu_tid <   2 && skepu_tid +   2 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    2) { if (skepu_tid <   1 && skepu_tid +   1 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }

	if (skepu_tid == 0)
	{
		skepu_output[get_group_id(0)] = skepu_sdata[skepu_tid];
	}
}
)~~~";

static const char *ReduceKernelTemplate_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME_ReduceOnly(__global SKEPU_REDUCE_RESULT_TYPE* skepu_input, __global SKEPU_REDUCE_RESULT_TYPE* skepu_output, size_t skepu_n, __local SKEPU_REDUCE_RESULT_TYPE* skepu_sdata)
{
	size_t skepu_blockSize = get_local_size(0);
	size_t skepu_tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * skepu_blockSize + get_local_id(0);
	size_t skepu_gridSize = skepu_blockSize * get_num_groups(0);
	SKEPU_REDUCE_RESULT_TYPE skepu_result;

	if (skepu_i < skepu_n)
	{
		skepu_result = skepu_input[skepu_i];
		skepu_i += skepu_gridSize;
	}

	while (skepu_i < skepu_n)
	{
		skepu_result = SKEPU_FUNCTION_NAME_REDUCE(skepu_result, skepu_input[skepu_i]);
		skepu_i += skepu_gridSize;
	}

	skepu_sdata[skepu_tid] = skepu_result;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid +  64 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   64) { if (skepu_tid <  32 && skepu_tid +  32 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   32) { if (skepu_tid <  16 && skepu_tid +  16 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   16) { if (skepu_tid <   8 && skepu_tid +   8 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    8) { if (skepu_tid <   4 && skepu_tid +   4 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    4) { if (skepu_tid <   2 && skepu_tid +   2 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    2) { if (skepu_tid <   1 && skepu_tid +   1 < skepu_n) { skepu_sdata[skepu_tid] = SKEPU_FUNCTION_NAME_REDUCE(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }

	if (skepu_tid == 0)
	{
		skepu_output[get_group_id(0)] = skepu_sdata[skepu_tid];
	}
}
)~~~";


const std::string Constructor = R"~~~(
class SKEPU_KERNEL_CLASS
{
public:

	enum
	{
		KERNEL_MAPREDUCE = 0,
		KERNEL_REDUCE,
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

		std::string source = skepu::backend::cl_helpers::replaceSizeT(R"###(SKEPU_OPENCL_KERNEL)###");

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel_mapreduce = clCreateKernel(program, "SKEPU_KERNEL_NAME", &err);
			CL_CHECK_ERROR(err, "Error creating MapReduce kernel 'SKEPU_KERNEL_NAME'");

			cl_kernel kernel_reduce = clCreateKernel(program, "SKEPU_KERNEL_NAME_ReduceOnly", &err);
			CL_CHECK_ERROR(err, "Error creating MapReduce kernel 'SKEPU_KERNEL_NAME'");

			kernels(counter, KERNEL_MAPREDUCE, &kernel_mapreduce);
			kernels(counter, KERNEL_REDUCE,    &kernel_reduce);
			counter++;
		}

		initialized = true;
	}

	TEMPLATE_HEADER
	static void mapReduce
	(
		size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize,
		SKEPU_HOST_KERNEL_PARAMS
		skepu::backend::DeviceMemPointer_CL<SKEPU_REDUCE_RESULT_CPU> *skepu_output,
		SKEPU_SIZES_TUPLE_PARAM size_t skepu_n, size_t skepu_base,
		size_t skepu_sharedMemSize
	)
	{
		cl_kernel skepu_kernel = kernels(skepu_deviceID, KERNEL_MAPREDUCE);
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernel, SKEPU_KERNEL_ARGS skepu_output->getDeviceDataPointer(), SKEPU_SIZE_ARGS skepu_n, skepu_base);
		clSetKernelArg(skepu_kernel, SKEPU_KERNEL_ARG_COUNT, skepu_sharedMemSize, NULL);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernel, 1, NULL, &skepu_globalSize, &skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching MapReduce kernel");
	}

	static void reduceOnly
	(
		size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize,
		skepu::backend::DeviceMemPointer_CL<SKEPU_REDUCE_RESULT_CPU> *skepu_input, skepu::backend::DeviceMemPointer_CL<SKEPU_REDUCE_RESULT_CPU> *skepu_output,
		size_t skepu_n, size_t skepu_sharedMemSize
	)
	{
		cl_kernel skepu_kernel = kernels(skepu_deviceID, KERNEL_REDUCE);
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernel, skepu_input->getDeviceDataPointer(), skepu_output->getDeviceDataPointer(), skepu_n);
		clSetKernelArg(skepu_kernel, 3, skepu_sharedMemSize, NULL);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernel, 1, NULL, &skepu_globalSize, &skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching MapReduce reduce-only kernel");
	}
};
)~~~";

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

IndexCodeGen indexInitHelper_CL(UserFunction &uf)
{
	IndexCodeGen res;
	res.dim = 0;
	
	if (uf.indexed1D || uf.indexed2D || uf.indexed3D || uf.indexed4D)
	{
		res.mapFuncParam = "skepu_index";
		res.hasIndex = true;
	}
	else
	{
		res.templateHeader = "template<typename Ignore>";
		res.sizesTupleParam = "Ignore, ";
	}
	
	if (uf.indexed1D)
	{
		res.dim = 1;
		res.sizeParams = "";
		res.sizeArgs = "";
		res.sizesTupleParam = "std::tuple<size_t> skepu_sizes, ";
		res.indexInit = "index1_t skepu_index = { .i = skepu_base + skepu_i };";
	}
	else if (uf.indexed2D)
	{
		res.dim = 2;
		res.sizeParams = "size_t skepu_w2, ";
		res.sizeArgs = "std::get<1>(skepu_sizes), ";
		res.sizesTupleParam = "std::tuple<size_t, size_t> skepu_sizes, ";
		res.indexInit = "index2_t skepu_index = { .row = (skepu_base + skepu_i) / skepu_w2, .col = (skepu_base + skepu_i) % skepu_w2 };";
	}
	else if (uf.indexed3D)
	{
		res.dim = 3;
		res.sizeParams = "size_t skepu_w2, size_t skepu_w3, ";
		res.sizeArgs = "std::get<1>(skepu_sizes), std::get<2>(skepu_sizes), ";
		res.sizesTupleParam = "std::tuple<size_t, size_t, size_t> skepu_sizes, ";
		res.indexInit = R"~~~(
			size_t cindex = skepu_base + skepu_i;
			size_t ci = cindex / (skepu_w2 * skepu_w3);
			cindex = cindex % (skepu_w2 * skepu_w3);
			size_t cj = cindex / (skepu_w3);
			cindex = cindex % (skepu_w3);
			index3_t skepu_index = { .i = ci, .j = cj, .k = cindex };
		)~~~";
	}
	else if (uf.indexed4D)
	{
		res.dim = 4;
		res.sizeParams = "size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, ";
		res.sizeArgs = "std::get<1>(skepu_sizes), std::get<2>(skepu_sizes), std::get<3>(skepu_sizes), ";
		res.sizesTupleParam = "std::tuple<size_t, size_t, size_t, size_t> skepu_sizes, ";
		res.indexInit = R"~~~(
			size_t cindex = skepu_base + skepu_i;
			
			size_t ci = cindex / (skepu_w2 * skepu_w3 * skepu_w4);
			cindex = cindex % (skepu_w2 * skepu_w3 * skepu_w4);
			
			size_t cj = cindex / (skepu_w3 * skepu_w4);
			cindex = cindex % (skepu_w3 * skepu_w4);
			
			size_t ck = cindex / (skepu_w4);
			cindex = cindex % (skepu_w4);
			
			index4_t skepu_index = { .i = ci, .j = cj, .k = ck, .l = cindex };
		)~~~";
	}
	
	return res;
}


void proxyCodeGenHelper_CL(std::map<ContainerType, std::set<std::string>> containerProxyTypes, std::stringstream &sourceStream)
{
	for (const std::string &type : containerProxyTypes[ContainerType::Vector])
		sourceStream << generateOpenCLVectorProxy(type);

	for (const std::string &type : containerProxyTypes[ContainerType::Matrix])
		sourceStream << generateOpenCLMatrixProxy(type);

	for (const std::string &type : containerProxyTypes[ContainerType::SparseMatrix])
		sourceStream << generateOpenCLSparseMatrixProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::MatRow])
		sourceStream << generateOpenCLMatrixRowProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::Tensor3])
		sourceStream << generateOpenCLTensor3Proxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::Tensor4])
		sourceStream << generateOpenCLTensor4Proxy(type);
}


std::string createMapReduceKernelProgram_CL(UserFunction &mapFunc, UserFunction &reduceFunc, size_t arity, std::string dir)
{
	std::stringstream sourceStream, SSKernelParamList, SSMapFuncParams, SSHostKernelParamList, SSKernelArgs, SSProxyInitializer, SSProxyInitializerInner;
	std::map<ContainerType, std::set<std::string>> containerProxyTypes;
	
	IndexCodeGen indexInfo = indexInitHelper_CL(mapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapFuncParams << indexInfo.mapFuncParam;

	for (UserFunction::Param& param : mapFunc.elwiseParams)
	{
		if (!first) { SSMapFuncParams << ", "; }
		SSKernelParamList << "__global " << param.rawTypeName << " * user_" << param.name << ", ";
		SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<const " << param.resolvedTypeName << "> * user_" << param.name << ", ";
		SSKernelArgs << "user_" << param.name << "->getDeviceDataPointer(), ";
		SSMapFuncParams << "user_" << param.name << "[skepu_i]";
		first = false;
	}

	for (UserFunction::RandomAccessParam& param : mapFunc.anyContainerParams)
	{
		std::string name = "skepu_container_" + param.name;
		if (!first) { SSMapFuncParams << ", "; }
		SSHostKernelParamList << param.TypeNameHost() << " skepu_container_" << param.name << ", ";
		containerProxyTypes[param.containerType].insert(param.resolvedTypeName);
		switch (param.containerType)
		{
			case ContainerType::Vector:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->size(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name << ", .size = skepu_size_" << param.name << " };\n";
				break;
			
			case ContainerType::Matrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_rows_" << param.name << ", size_t skepu_cols_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->total_rows(), std::get<0>(" << name << ")->total_cols(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name
					<< ", .rows = skepu_rows_" << param.name << ", .cols = skepu_cols_" << param.name << " };\n";
				break;
			
			case ContainerType::SparseMatrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name << ", ";
				SSKernelArgs << "skepu_container_" << param.name << ".size(), ";
				break;
			
			case ContainerType::MatRow:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_cols_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->total_cols(), ";
				SSProxyInitializerInner << param.TypeNameOpenCL() << " " << param.name << " = { .data = (" << name << " + i * skepu_cols_" << param.name << "), .cols = skepu_cols_" << param.name << " };\n";
				break;
			
			case ContainerType::Tensor3:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", "
					<< "size_t skepu_size_i_" << param.name << ", size_t skepu_size_j_" << param.name << ", size_t skepu_size_k_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<0>(" << name << ")->size_i(), std::get<0>(" << name << ")->size_j(), std::get<0>(" << name << ")->size_k(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name
					<< ", .size_i = skepu_size_i_" << param.name << ", .size_j = skepu_size_j_" << param.name << ", .size_k = skepu_size_k_" << param.name << " };\n";
				break;
			
			case ContainerType::Tensor4:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", "
					<< "size_t skepu_size_i_" << param.name << ", size_t skepu_size_j_" << param.name << ", size_t skepu_size_k_" << param.name << ", size_t skepu_size_l_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<0>(" << name << ")->size_i(), std::get<0>(" << name << ")->size_j(), std::get<0>(" << name << ")->size_k(), std::get<0>(" << name << ")->size_l(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name
					<< ", .size_i = skepu_size_i_" << param.name << ", .size_j = skepu_size_j_" << param.name << ", .size_k = skepu_size_k_" << param.name << ", .size_l = skepu_size_l_" << param.name << " };\n";
				break;
		}
		SSMapFuncParams << param.name;
		first = false;
	}

	for (UserFunction::Param& param : mapFunc.anyScalarParams)
	{
		if (!first) { SSMapFuncParams << ", "; }
		SSKernelParamList << param.resolvedTypeName << " user_" << param.name << ", ";
		SSHostKernelParamList << param.resolvedTypeName << " user_" << param.name << ", ";
		SSKernelArgs << "user_" << param.name << ", ";
		SSMapFuncParams << "user_" << param.name;
		first = false;
	}

	assert(reduceFunc.elwiseParams.size() == 2);

	if (mapFunc.requiresDoublePrecision || reduceFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
	
	proxyCodeGenHelper_CL(containerProxyTypes, sourceStream);

	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";

		// check for extra user-supplied opencl code for custom datatype
		// TODO: Also check the referenced UFs for referenced UTs skepu::userstruct
		for (UserType *RefType : mapFunc.ReferencedUTs)
			sourceStream << generateUserTypeCode_CL(*RefType);

	std::stringstream SSKernelName;
	SSKernelName << transformToCXXIdentifier(ResultName) << "_MapReduceKernel_" << mapFunc.uniqueName << "_" << reduceFunc.uniqueName << "_arity_" << arity;
	const std::string kernelName = SSKernelName.str();
	const std::string className = "CLWrapperClass_" + kernelName;
	std::stringstream SSKernelArgCount;
	SSKernelArgCount << mapFunc.numKernelArgsCL() + 3 + std::max<int>(0, indexInfo.dim - 1);

	sourceStream << KernelPredefinedTypes_CL;

	if (mapFunc.refersTo(reduceFunc))
		sourceStream << generateUserFunctionCode_CL(mapFunc);
	else if (reduceFunc.refersTo(mapFunc))
		sourceStream << generateUserFunctionCode_CL(reduceFunc);
	else
		sourceStream << generateUserFunctionCode_CL(mapFunc) << generateUserFunctionCode_CL(reduceFunc);

	sourceStream << MapReduceKernelTemplate_CL << ReduceKernelTemplate_CL;

	std::string finalSource = Constructor;
	replaceTextInString(finalSource, "SKEPU_OPENCL_KERNEL", sourceStream.str());
	replaceTextInString(finalSource, "SKEPU_KERNEL_CLASS", className);
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARGS", SSKernelArgs.str());
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARG_COUNT", SSKernelArgCount.str());
	replaceTextInString(finalSource, "SKEPU_HOST_KERNEL_PARAMS", SSHostKernelParamList.str());
	replaceTextInString(finalSource, "SKEPU_CONTAINER_PROXIES", SSProxyInitializer.str());
	replaceTextInString(finalSource, "SKEPU_CONTAINER_PROXIE_INNER", SSProxyInitializerInner.str());
	replaceTextInString(finalSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(finalSource, PH_MapParams, SSMapFuncParams.str());
	replaceTextInString(finalSource, PH_ReduceResultType, reduceFunc.rawReturnTypeName);
	replaceTextInString(finalSource, "SKEPU_REDUCE_RESULT_CPU", reduceFunc.resolvedReturnTypeName);
	replaceTextInString(finalSource, PH_MapResultType, mapFunc.rawReturnTypeName);
	replaceTextInString(finalSource, PH_MapFuncName, mapFunc.uniqueName);
	replaceTextInString(finalSource, PH_ReduceFuncName, reduceFunc.uniqueName);
	replaceTextInString(finalSource, PH_KernelName, kernelName);
	replaceTextInString(finalSource, PH_IndexInitializer, indexInfo.indexInit);
	replaceTextInString(finalSource, "SKEPU_SIZE_PARAMS", indexInfo.sizeParams);
	replaceTextInString(finalSource, "SKEPU_SIZE_ARGS", indexInfo.sizeArgs);
	replaceTextInString(finalSource, "SKEPU_SIZES_TUPLE_PARAM", indexInfo.sizesTupleParam);
	replaceTextInString(finalSource, "TEMPLATE_HEADER", indexInfo.templateHeader);

	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;

	return kernelName;
}
