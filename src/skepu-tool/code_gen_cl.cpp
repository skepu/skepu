#include "code_gen_cl.h"

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




std::string generateOpenCLVectorProxy(std::string typeName)
{
  static const std::string OpenCLVectorTemplate = R"~~~(
typedef struct {
	__global {{CONTAINED_TYPE_CL}} *data;
	size_t size;
} skepu_vec_proxy_{{ESCAPED_TYPE_CL}};
{{CONTAINED_TYPE_CL}} skepu_vec_proxy_access_{{ESCAPED_TYPE_CL}}(skepu_vec_proxy_{{ESCAPED_TYPE_CL}} v, size_t i)
{ return v.data[i]; }
)~~~";
	std::string retval = OpenCLVectorTemplate;
	replaceTextInString(retval, "{{CONTAINED_TYPE_CL}}", typeName);
	replaceTextInString(retval, "{{ESCAPED_TYPE_CL}}", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLMatrixProxy(std::string typeName)
{
  static const std::string OpenCLMatrixTemplate = R"~~~(
typedef struct {
	__global {{CONTAINED_TYPE_CL}} *data;
	size_t rows;
	size_t cols;
} skepu_mat_proxy_{{ESCAPED_TYPE_CL}};
{{CONTAINED_TYPE_CL}} skepu_mat_proxy_access_{{ESCAPED_TYPE_CL}}(skepu_mat_proxy_{{ESCAPED_TYPE_CL}} m, size_t i, size_t j)
{ return m.data[i * m.cols + j]; }
)~~~";
	std::string retval = OpenCLMatrixTemplate;
	replaceTextInString(retval, "{{CONTAINED_TYPE_CL}}", typeName);
	replaceTextInString(retval, "{{ESCAPED_TYPE_CL}}", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLMatrixRowProxy(std::string typeName)
{
  static const std::string OpenCLMatrixRowTemplate = R"~~~(
typedef struct {
	__global {{CONTAINED_TYPE_CL}} *data;
	size_t cols;
} skepu_matrow_proxy_{{ESCAPED_TYPE_CL}};
static {{CONTAINED_TYPE_CL}} skepu_matrow_proxy_access_{{ESCAPED_TYPE_CL}}(skepu_matrow_proxy_{{ESCAPED_TYPE_CL}} mr, size_t i)
{ return mr.data[i]; }
)~~~";
  std::string retval = OpenCLMatrixRowTemplate;
	replaceTextInString(retval, "{{CONTAINED_TYPE_CL}}", typeName);
	replaceTextInString(retval, "{{ESCAPED_TYPE_CL}}", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLMatrixColProxy(std::string typeName)
{
  static const std::string OpenCLMatrixColTemplate = R"~~~(
typedef struct {
	__global {{CONTAINED_TYPE_CL}} *data;
	size_t rows;
	size_t cols;
} skepu_matcol_proxy_{{ESCAPED_TYPE_CL}};
static {{CONTAINED_TYPE_CL}} skepu_matcol_proxy_access_{{ESCAPED_TYPE_CL}}(skepu_matcol_proxy_{{ESCAPED_TYPE_CL}} mc, size_t i)
{ return mc.data[i * mc.cols]; }
)~~~";
	std::string retval = OpenCLMatrixColTemplate;
	replaceTextInString(retval, "{{CONTAINED_TYPE_CL}}", typeName);
	replaceTextInString(retval, "{{ESCAPED_TYPE_CL}}", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLSparseMatrixProxy(std::string typeName)
{
  static const std::string OpenCLSparseMatrixTemplate = R"~~~(
typedef struct {
	__global {{CONTAINED_TYPE_CL}} *data;
	__global size_t *row_offsets;
	__global size_t *col_indices;
	size_t count;
} skepu_sparse_mat_proxy_{{ESCAPED_TYPE_CL}};
)~~~";
	std::string retval = OpenCLSparseMatrixTemplate;
	replaceTextInString(retval, "{{CONTAINED_TYPE_CL}}", typeName);
	replaceTextInString(retval, "{{ESCAPED_TYPE_CL}}", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLTensor3Proxy(std::string typeName)
{
  static const std::string OpenCLTensor3Template = R"~~~(
typedef struct {
	__global {{CONTAINED_TYPE_CL}} *data;
	size_t size_i;
	size_t size_j;
	size_t size_k;
} skepu_ten3_proxy_{{ESCAPED_TYPE_CL}};
static {{CONTAINED_TYPE_CL}} skepu_ten3_proxy_access_{{ESCAPED_TYPE_CL}}(skepu_ten3_proxy_{{ESCAPED_TYPE_CL}} t, size_t i, size_t j, size_t k)
{ return t.data[i * t.size_j * t.size_k + j * t.size_k + k]; }
)~~~";
	std::string retval = OpenCLTensor3Template;
	replaceTextInString(retval, "{{CONTAINED_TYPE_CL}}", typeName);
	replaceTextInString(retval, "{{ESCAPED_TYPE_CL}}", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLTensor4Proxy(std::string typeName)
{
  static const std::string OpenCLTensor4Template = R"~~~(
typedef struct {
	__global {{CONTAINED_TYPE_CL}} *data;
	size_t size_i;
	size_t size_j;
	size_t size_k;
	size_t size_l;
} skepu_ten4_proxy_{{ESCAPED_TYPE_CL}};
static {{CONTAINED_TYPE_CL}} skepu_ten4_proxy_access_{{ESCAPED_TYPE_CL}}(skepu_ten4_proxy_{{ESCAPED_TYPE_CL}} t, size_t i, size_t j, size_t k, size_t l)
{ return t.data[i * t.size_j * t.size_k * t.size_l + j * t.size_k * t.size_l + k * t.size_l + l]; }
)~~~";
	std::string retval = OpenCLTensor4Template;
	replaceTextInString(retval, "{{CONTAINED_TYPE_CL}}", typeName);
	replaceTextInString(retval, "{{ESCAPED_TYPE_CL}}", transformToCXXIdentifier(typeName));
	return retval;
}




std::string generateOpenCLRegion(size_t dim, std::string typeName)
{
static const std::string OpenCLRegion1DTemplate = R"~~~(
typedef struct {
	__local {{CONTAINED_TYPE_CL}} *data;
	int oi;
	size_t stride;
} skepu_region1d_{{ESCAPED_TYPE_CL}};

static {{CONTAINED_TYPE_CL}} skepu_region_access_1d_{{ESCAPED_TYPE_CL}}(skepu_region1d_{{ESCAPED_TYPE_CL}} r, int i)
{ return r.data[i * r.stride]; }
)~~~";

static const std::string OpenCLRegion2DTemplate = R"~~~(
typedef struct {
	__local {{CONTAINED_TYPE_CL}} *data;
	int oi, oj;
	size_t stride;
} skepu_region2d_{{ESCAPED_TYPE_CL}};

static {{CONTAINED_TYPE_CL}} skepu_region_access_2d_{{ESCAPED_TYPE_CL}}(skepu_region2d_{{ESCAPED_TYPE_CL}} r, int i, int j)
{ return r.data[i * r.stride + j]; }
)~~~";

static const std::string OpenCLRegion3DTemplate = R"~~~(
typedef struct {
	__local {{CONTAINED_TYPE_CL}} *data;
	int oi, oj, ok;
	size_t stride1, stride2;
} skepu_region3d_{{ESCAPED_TYPE_CL}};

static {{CONTAINED_TYPE_CL}} skepu_region_access_3d_{{ESCAPED_TYPE_CL}}(skepu_region3d_{{ESCAPED_TYPE_CL}} r, int i, int j, int k)
{ return r.data[i * r.stride1 * r.stride2 + j * r.stride2 + k; }
)~~~";

static const std::string OpenCLRegion4DTemplate = R"~~~(
typedef struct {
	__local {{CONTAINED_TYPE_CL}} *data;
	int oi, oj, ok, ol;
	size_t stride1, stride2, stride3;
} skepu_region4d_{{ESCAPED_TYPE_CL}};

static {{CONTAINED_TYPE_CL}} skepu_region_access_4d_{{ESCAPED_TYPE_CL}}(skepu_region4d_{{ESCAPED_TYPE_CL}} r, int i, int j, int k, int l)
{ return r.data[i * r.stride1 * r.stride2 * r.stride3 + j * r.stride2 * r.stride3 + k * r.stride3 + l]; }
)~~~";
	
	std::string retval;
	switch (dim)
	{
	case 1: retval = OpenCLRegion1DTemplate; break;
	case 2: retval = OpenCLRegion2DTemplate; break;
	case 3: retval = OpenCLRegion3DTemplate; break;
	case 4: retval = OpenCLRegion4DTemplate; break;
	};
	replaceTextInString(retval, "{{CONTAINED_TYPE_CL}}", typeName);
	replaceTextInString(retval, "{{ESCAPED_TYPE_CL}}", transformToCXXIdentifier(typeName));
	return retval;
}


std::string generateOpenCLRandom()
{
static const std::string OpenCLRandomTemplate = R"~~~(
#define RND_STATE_T ulong
#define RND_NORMALIZED_T double
	
#define RND_MOD (1LL << 48)
#define RND_EXP 1
#define RND_BASE 0x5deece66d
#define RND_INC 5

typedef struct {
	RND_STATE_T m_state;
} skepu_random;

RND_STATE_T skepu_random_get(__global skepu_random *prng)
{
	for (size_t i = 0; i < RND_EXP; ++i)
		prng->m_state = (prng->m_state * RND_BASE + RND_INC) % RND_MOD;
	return prng->m_state;
}

RND_NORMALIZED_T skepu_random_get_normalized(__global skepu_random *prng)
{
	return (RND_NORMALIZED_T)skepu_random_get(prng) / RND_MOD;
}
)~~~";
	return OpenCLRandomTemplate;
};



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
	
	for (const std::string &type : containerProxyTypes[ContainerType::MatCol])
		sourceStream << generateOpenCLMatrixColProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::Tensor3])
		sourceStream << generateOpenCLTensor3Proxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::Tensor4])
		sourceStream << generateOpenCLTensor4Proxy(type);
}




RandomAccessAndScalarsResult handleRandomAccessAndUniforms_CL(
	UserFunction &func,
	std::stringstream& SSMapFuncArgs,
	std::stringstream& SSHostKernelParamList,
	std::stringstream& SSKernelParamList,
	std::stringstream& SSKernelArgs,
	bool &first
)
{
	RandomAccessAndScalarsResult res;
	std::stringstream SSProxyInitializer, SSProxyInitializerInner;
	
	// Random-access data
	for (UserFunction::RandomAccessParam& param : func.anyContainerParams)
	{
		std::string name = "skepu_container_" + param.name;
		if (!first) { SSMapFuncArgs << ", "; }
		SSHostKernelParamList << param.TypeNameHost() << " skepu_container_" << param.name << ", ";
		res.containerProxyTypes[param.containerType].insert(param.resolvedTypeName);
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
			
			case ContainerType::MatRow:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_cols_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->total_cols(), ";
				SSProxyInitializerInner << param.TypeNameOpenCL() << " " << param.name << " = { .data = (" << name << " + skepu_i * skepu_cols_" << param.name << "), .cols = skepu_cols_" << param.name << " };\n";
				break;
			
			case ContainerType::MatCol:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_rows_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->total_rows(), ";
				SSProxyInitializerInner << param.TypeNameOpenCL() << " " << param.name << " = { .data = (" << name << " + skepu_i), .rows = skepu_rows_" << param.name << " };\n";
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
			
			case ContainerType::SparseMatrix:
				SSKernelParamList
					<< "__global " << param.resolvedTypeName << " *" << name << ", "
					<< "__global size_t *" << param.name << "_row_pointers, "
					<< "__global size_t *" << param.name << "_col_indices, "
					<< "size_t skepu_size_" << param.name << ", ";

				SSKernelArgs
					<< "std::get<1>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<2>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<3>(" << name << ")->getDeviceDataPointer(), "
					<< "std::get<0>(" << name << ")->total_nnz(), ";

				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { "
					<< ".data = " << name << ", "
					<< ".row_offsets = " << param.name << "_row_pointers, "
					<< ".col_indices = " << param.name << "_col_indices, "
					<< ".count = skepu_size_" << param.name << " };\n";
				break;
		}
		SSMapFuncArgs << param.name;
		first = false;
	}

	// Scalar input data
	for (UserFunction::Param& param : func.anyScalarParams)
	{
		if (!first) { SSMapFuncArgs << ", "; }
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSHostKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSKernelArgs << param.name << ", ";
		SSMapFuncArgs << param.name;
		first = false;
	}
	
	res.proxyInitializer = SSProxyInitializer.str();
	res.proxyInitializerInner = SSProxyInitializerInner.str();
	return res;
}

const std::string KernelPredefinedTypes_CL = R"~~~(
#define SKEPU_USING_BACKEND_CL 1

typedef struct{
	size_t i;
} index1_t;

typedef struct {
	size_t row;
	size_t col;
} index2_t;

typedef struct {
	size_t i;
	size_t j;
	size_t k;
} index3_t;

typedef struct {
	size_t i;
	size_t j;
	size_t k;
	size_t l;
} index4_t;

static size_t get_device_id()
{
	return SKEPU_INTERNAL_DEVICE_ID;
}

#define VARIANT_OPENCL(block) block
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block)

// Size of basic integer types defined in OpenCL standard.
// Emulate stdint.h based on this.
typedef uchar    uint8_t;
typedef ushort   uint16_t;
typedef uint     uint32_t;
typedef ulong    uint64_t;

typedef char     int8_t;
typedef short    int16_t;
typedef int      int32_t;
typedef long     int64_t;

enum {
	SKEPU_EDGE_NONE = 0,
	SKEPU_EDGE_CYCLIC = 1,
	SKEPU_EDGE_DUPLICATE = 2,
	SKEPU_EDGE_PAD = 3,
};

)~~~";


void handleUserTypesConstantsAndPrecision_CL(std::vector<UserFunction const*> funcs, std::stringstream &sourceStream)
{
  // Double precision
  for (UserFunction const* func : funcs)
    if (func->requiresDoublePrecision)
    {
	    sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
      break;
    }
	
  // Predefined types
	sourceStream << KernelPredefinedTypes_CL;

	// User constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";
	
  // User types
  std::set<UserType*> referencedUTs{};
  for (UserFunction const* func : funcs)
    for (UserType *type : func->ReferencedUTs)
      referencedUTs.insert(type);
  
	for (UserType *type : referencedUTs)
		sourceStream << generateUserTypeCode_CL(*type);
}



std::string handleOutputs_CL(UserFunction &func, std::stringstream &SSHostKernelParamList, std::stringstream &SSKernelParamList, std::stringstream &SSKernelArgs, std::string index)
{
	if (func.multipleReturnTypes.size() == 0)
	{
		SSKernelParamList << " __global " << func.rawReturnTypeName << "* skepu_output, ";
		SSHostKernelParamList << " skepu::backend::DeviceMemPointer_CL<" << func.resolvedReturnTypeName << "> *skepu_output, ";
		SSKernelArgs << " skepu_output->getDeviceDataPointer(), ";
		return "";
	}
	else
	{
		std::stringstream SSMultiOutputAssign;
		size_t outCtr = 0;
		for (std::string& outputType : func.multipleReturnTypes)
		{
			SSKernelParamList << "__global " << outputType << "* skepu_output_" << outCtr << ", ";
			SSHostKernelParamList << " skepu::backend::DeviceMemPointer_CL<" << outputType << "> *skepu_output_" << outCtr << ", ";
			SSKernelArgs << " skepu_output_" << outCtr << "->getDeviceDataPointer(), ";
			SSMultiOutputAssign << " skepu_output_" << outCtr << "[" << index << "] = skepu_out_temp.e" << outCtr << ";\n";
			outCtr++;
		}
		return SSMultiOutputAssign.str();
	}
}