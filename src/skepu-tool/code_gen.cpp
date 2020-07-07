#include "code_gen.h"

using namespace clang;

std::string getSourceAsString(SourceRange range)
{
	int rangeSize = GlobalRewriter.getRangeSize(range);
	if (rangeSize == -1)
		return "";

	SourceLocation startLoc = range.getBegin();
	const char *strStart = GlobalRewriter.getSourceMgr().getCharacterData(startLoc);

	std::string exprString;
	exprString.assign(strStart, rangeSize);

	return exprString;
}

void printParamList(std::ostream &o, UserFunction &Func)
{
/*	bool first = true;
	for (const ParmVarDecl* Parm : f->parameters())
	{
		// default arg?
		if (!first) o << ", ";
		first = false;

		SourceRange SRParm = Parm->getSourceRange();
		std::string ParmText = Lexer::getSourceText(CharSourceRange::getTokenRange(SRParm), GlobalRewriter.getSourceMgr(), LangOptions(), 0);
		o << ParmText;
	}*/
	
	bool first = true;

	if (Func.indexed1D)
	{
		o << "skepu::Index1D " << Func.indexParam->name;
		first = false;
	}
	else if (Func.indexed2D)
	{
		o << "skepu::Index2D " << Func.indexParam->name;
		first = false;
	}
	else if (Func.indexed3D)
	{
		o << "skepu::Index3D " << Func.indexParam->name;
		first = false;
	}
	else if (Func.indexed4D)
	{
		o << "skepu::Index4D " << Func.indexParam->name;
		first = false;
	}

	for (UserFunction::Param& param : Func.elwiseParams)
	{
		if (!first) { o << ", "; }
		o << param.resolvedTypeName << " " << param.name; // HERE1
		first = false;
	}

	for (UserFunction::RandomAccessParam& param : Func.anyContainerParams)
	{
		if (!first) { o << ", "; }
		o << param.fullTypeName << " " << param.name;
		first = false;
	}

	for (UserFunction::Param& param : Func.anyScalarParams)
	{
		if (!first) { o << ", "; }
		if (param.astDeclNode->getOriginalType()->isPointerType())
			o << "__global ";
		o << param.resolvedTypeName << " " << param.name;
		first = false;
	}
	
}


void replaceTextInString(std::string& text, const std::string &find, const std::string &replace)
{
	std::string::size_type pos = 0;
	while ((pos = text.find(find, pos)) != std::string::npos)
	{
		text.replace(pos, find.length(), replace);
		pos += replace.length();
	}
}

std::string transformToCXXIdentifier(std::string &in)
{
	std::string out = in;
	replaceTextInString(out, ".", "__dot__");
	replaceTextInString(out, " ", "__space__");
	replaceTextInString(out, ":", "__colon__");
	return out;
}

static const std::string OpenCLVectorTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	size_t size;
} skepu_vec_proxy_SKEPU_ESCAPED_TYPE_CL;
)~~~";

static const std::string OpenCLMatrixTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	size_t rows;
	size_t cols;
} skepu_mat_proxy_SKEPU_ESCAPED_TYPE_CL;
)~~~";

static const std::string OpenCLMatrixRowTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	size_t cols;
} skepu_matrow_proxy_SKEPU_ESCAPED_TYPE_CL;
)~~~";

static const std::string OpenCLTensor3Template = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	size_t size_i;
	size_t size_j;
	size_t size_k;
} skepu_ten3_proxy_SKEPU_ESCAPED_TYPE_CL;
)~~~";

static const std::string OpenCLTensor4Template = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	size_t size_i;
	size_t size_j;
	size_t size_k;
	size_t size_l;
} skepu_ten4_proxy_SKEPU_ESCAPED_TYPE_CL;
)~~~";

static const std::string OpenCLSparseMatrixTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	__global size_t *row_offsets;
	__global size_t *col_indices;
	size_t count;
} skepu_sparse_mat_proxy_SKEPU_ESCAPED_TYPE_CL;
)~~~";

std::string generateOpenCLVectorProxy(std::string typeName)
{
	std::string retval = OpenCLVectorTemplate;
	replaceTextInString(retval, "SKEPU_CONTAINED_TYPE_CL", typeName);
	replaceTextInString(retval, "SKEPU_ESCAPED_TYPE_CL", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLMatrixProxy(std::string typeName)
{
	std::string retval = OpenCLMatrixTemplate;
	replaceTextInString(retval, "SKEPU_CONTAINED_TYPE_CL", typeName);
	replaceTextInString(retval, "SKEPU_ESCAPED_TYPE_CL", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLMatrixRowProxy(std::string typeName)
{
	std::string retval = OpenCLMatrixRowTemplate;
	replaceTextInString(retval, "SKEPU_CONTAINED_TYPE_CL", typeName);
	replaceTextInString(retval, "SKEPU_ESCAPED_TYPE_CL", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLSparseMatrixProxy(std::string typeName)
{
	std::string retval = OpenCLSparseMatrixTemplate;
	replaceTextInString(retval, "SKEPU_CONTAINED_TYPE_CL", typeName);
	replaceTextInString(retval, "SKEPU_ESCAPED_TYPE_CL", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLTensor3Proxy(std::string typeName)
{
	std::string retval = OpenCLTensor3Template;
	replaceTextInString(retval, "SKEPU_CONTAINED_TYPE_CL", typeName);
	replaceTextInString(retval, "SKEPU_ESCAPED_TYPE_CL", transformToCXXIdentifier(typeName));
	return retval;
}

std::string generateOpenCLTensor4Proxy(std::string typeName)
{
	std::string retval = OpenCLTensor4Template;
	replaceTextInString(retval, "SKEPU_CONTAINED_TYPE_CL", typeName);
	replaceTextInString(retval, "SKEPU_ESCAPED_TYPE_CL", transformToCXXIdentifier(typeName));
	return retval;
}




std::string generateOpenCLRegion(size_t dim, std::string typeName)
{
static const std::string OpenCLRegion1DTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	int oi;
	size_t stride;
} skepu_region1d_SKEPU_ESCAPED_TYPE_CL;

SKEPU_CONTAINED_TYPE_CL region_access_1d_SKEPU_ESCAPED_TYPE_CL(skepu_region1d_SKEPU_ESCAPED_TYPE_CL r, int i)
{ return r.data[i * r.stride]; }
)~~~";

static const std::string OpenCLRegion2DTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	int oi, oj;
	size_t stride;
} skepu_region2d_SKEPU_ESCAPED_TYPE_CL;

SKEPU_CONTAINED_TYPE_CL region_access_2d_SKEPU_ESCAPED_TYPE_CL(skepu_region2d_SKEPU_ESCAPED_TYPE_CL r, int i, int j)
{ return r.data[i * r.stride + j]; }
)~~~";

static const std::string OpenCLRegion3DTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	int oi, oj, ok;
	size_t stride1, stride2;
} skepu_region3d_SKEPU_ESCAPED_TYPE_CL;

SKEPU_CONTAINED_TYPE_CL region_access_3d_SKEPU_ESCAPED_TYPE_CL(skepu_region3d_SKEPU_ESCAPED_TYPE_CL r, int i, int j, int k)
{ return r.data[i * r.stride1 * r.stride2 + j * r.stride2 + k; }
)~~~";

static const std::string OpenCLRegion4DTemplate = R"~~~(
typedef struct {
	__global SKEPU_CONTAINED_TYPE_CL *data;
	int oi, oj, ok, ol;
	size_t stride1, stride2, stride3;
} skepu_region4d_SKEPU_ESCAPED_TYPE_CL;

SKEPU_CONTAINED_TYPE_CL region_access_4d_SKEPU_ESCAPED_TYPE_CL(skepu_region4d_SKEPU_ESCAPED_TYPE_CL r, int i, int j, int k, int l)
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
	replaceTextInString(retval, "SKEPU_CONTAINED_TYPE_CL", typeName);
	replaceTextInString(retval, "SKEPU_ESCAPED_TYPE_CL", transformToCXXIdentifier(typeName));
	return retval;
}


std::string generateOpenCLMultipleReturn(std::vector<std::string> &types)
{
	std::string retval = R"~~~(
		// MULTI-VALUED RETURN SUPPORTING CODE
		typedef struct {
			{{SKEPU_MULTIPLE_FIELDS}}
		} skepu_multiple_{{SKEPU_MULTIPLE_UID}};
		
		SKEPU_CONTAINED_TYPE_CL region_access_1d_SKEPU_ESCAPED_TYPE_CL(skepu_region1d_SKEPU_ESCAPED_TYPE_CL r, int i)
		{ return r.data[i * r.stride]; }
	)~~~";

	
	std::stringstream fieldString, uidString;
	
	size_t i = 0;
	for (auto &type : types)
	{
		fieldString << "\n" << type << " e" << i << ";";
		uidString << type << "_";
	}
	
	replaceTextInString(retval, "{{SKEPU_MULTIPLE_FIELDS}}", fieldString.str());
	replaceTextInString(retval, "{{SKEPU_MULTIPLE_UID}}", uidString.str());
	return retval;
}


std::string generateOpenCLMultipleReturn(UserFunction &UF)
{
	if (UF.multipleReturnTypes.size() > 0)
	{
		std::stringstream SSmultiReturnType, SSmultiReturnTypeDef, SSmultiReturnMakeStruct, SSmultiReturnMakeParams;
		SSmultiReturnType << "skepu_multiple";
		size_t ctr = 0;
		for (std::string &type : UF.multipleReturnTypes)
		{
			std::string divider = (ctr != UF.multipleReturnTypes.size() - 1) ? ", " : "";
			SSmultiReturnType << "_" << type;
			SSmultiReturnTypeDef << type << " e" << ctr << ";\n";
			SSmultiReturnMakeParams << type << " arg" << ctr << divider;
			SSmultiReturnMakeStruct << " .e" << ctr << " = arg" << ctr << divider;
			ctr++;
		}
		
		std::string codeTemplate = R"~~~(
			typedef struct {
				{{SKEPU_MULTIPLE_RETURN_TYPE_DEF}}
			} {{SKEPU_MULTIPLE_RETURN_TYPE}};
			
			static {{SKEPU_MULTIPLE_RETURN_TYPE}} make_{{SKEPU_MULTIPLE_RETURN_TYPE}} ({{SKEPU_MULTIPLE_RETURN_MAKE_PARAMS}})
			{
				{{SKEPU_MULTIPLE_RETURN_TYPE}} retval = { {{SKEPU_MULTIPLE_RETURN_MAKE_STRUCT}} };
				return retval;
			}
		)~~~";
		
		
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_TYPE}}", SSmultiReturnType.str());
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_TYPE_DEF}}", SSmultiReturnTypeDef.str());
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_MAKE_PARAMS}}", SSmultiReturnMakeParams.str());
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_MAKE_STRUCT}}", SSmultiReturnMakeStruct.str());
		
		return codeTemplate;
	}
	return "";
}

std::string replaceReferencesToOtherUFs(UserFunction &UF, std::function<std::string(UserFunction&)> nameFunc, bool isGPU = false)
{
	const FunctionDecl *f = UF.astDeclNode;
	if (f->getTemplatedKind() == FunctionDecl::TK_FunctionTemplateSpecialization)
		f = f->getTemplateInstantiationPattern();

// Find references to other userfunctions
	Rewriter R(GlobalRewriter.getSourceMgr(), LangOptions());

	for (auto &ref : UF.UFReferences)
		R.ReplaceText(ref.first->getCallee()->getSourceRange(), nameFunc(*ref.second));

	for (auto &ref : UF.UTReferences)
		R.ReplaceText(ref.first->getTypeLoc().getSourceRange(), "struct " + ref.second->name);

	for (auto subscript : UF.containerSubscripts)
		R.InsertText(subscript->getCallee()->getBeginLoc(), ".data");
		
	
	if (isGPU && UF.multipleReturnTypes.size() > 0)
	{
		std::stringstream SSmultiReturnType, SSmultiReturnTypeDef, SSmultiReturnMakeStruct, SSmultiReturnMakeParams;
		SSmultiReturnType << "skepu_multiple";
		for (std::string &type : UF.multipleReturnTypes)
			SSmultiReturnType << "_" << type;
		
		for (auto *ref : UF.ReferencedRets)
		{
			clang::SourceRange args(ref->getArg(0)->getBeginLoc(), ref->getArg(ref->getNumArgs()-1)->getEndLoc());
			std::string argString = getSourceAsString(args);
			
			
		//	llvm::errs() << "MAKE_FN(" << argString << ")\n";
			
		//	ref->dump();
		//	R.InsertText(ref->getBeginLoc(), "make_" + SSmultiReturnType.str() + "(" + argString + ")");
		}
	}
	
	
	const CompoundStmt *Body = dyn_cast<CompoundStmt>(f->getBody());
	SourceRange SRBody = SourceRange(Body->getBeginLoc().getLocWithOffset(1), Body->getEndLoc().getLocWithOffset(-1));
	return R.getRewrittenText(SRBody);
}


bool testAndSet(bool &arg, bool newVal = true)
{
	bool oldVal = arg;
	arg = newVal;
	return oldVal;
}

void generateUserFunctionStruct(UserFunction &UF, std::string InstanceName, clang::SourceLocation loc)
{
	static std::set<std::string> generatedStructs;
	static std::set<std::string> usingDecls;

	// Start by recursively generate functors for referenced user functions
	for (auto *referenced : UF.ReferencedUFs)
		generateUserFunctionStruct(*referenced, InstanceName, loc);

	// Continue generating functor for this user function
	const FunctionDecl *f = UF.astDeclNode;
	std::string FunctorName = SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName;

	if (std::find(generatedStructs.begin(), generatedStructs.end(), FunctorName) != generatedStructs.end())
		return; // Already generated (same names implies idential code generated)

	generatedStructs.insert(FunctorName);
	UF.instanceName = InstanceName;

	std::stringstream SSSkepuFunctorStruct;
	SSSkepuFunctorStruct << "\nstruct " << FunctorName;
	SSSkepuFunctorStruct << "\n{\n";

	// Code generation specific to templated user functions
	if (UF.fromTemplate)
		for (const UserFunction::TemplateArgument &arg : UF.templateArguments)
		{
			// If the template argument type is a nested namespace type, bring the namespace into scope
			if (arg.rawTypeName != arg.resolvedTypeName)
			{
				SSSkepuFunctorStruct << "using " << arg.rawTypeName << " = " << arg.resolvedTypeName << ";\n";
				usingDecls.insert(arg.rawTypeName);
			}
			
			// Resolve the template parameter type occurences by a type alias to the argument type
			SSSkepuFunctorStruct << "using " << arg.paramName << " = " << arg.rawTypeName << ";\n";
		}
	
	size_t outArity = std::max<size_t>(1, UF.multipleReturnTypes.size());

	SSSkepuFunctorStruct << "constexpr static size_t totalArity = " << f->param_size() << ";\n";
	SSSkepuFunctorStruct << "constexpr static size_t outArity = " << outArity << ";\n";
	SSSkepuFunctorStruct << "constexpr static bool indexed = " << (UF.indexed1D || UF.indexed2D || UF.indexed3D || UF.indexed4D) << ";\n";
	
	SSSkepuFunctorStruct << "using IndexType = ";
	if (UF.indexed1D) SSSkepuFunctorStruct << "skepu::Index1D;\n";
	else if (UF.indexed2D) SSSkepuFunctorStruct << "skepu::Index2D;\n";
	else if (UF.indexed3D) SSSkepuFunctorStruct << "skepu::Index3D;\n";
	else if (UF.indexed4D) SSSkepuFunctorStruct << "skepu::Index4D;\n";
	else SSSkepuFunctorStruct << "void;\n";

	SSSkepuFunctorStruct << "using ElwiseArgs = std::tuple<";
	bool first = true;
	for (UserFunction::Param& param : UF.elwiseParams)
	{
		if (!testAndSet(first, false)) SSSkepuFunctorStruct << ", ";
		SSSkepuFunctorStruct << param.resolvedTypeName;
	}
	SSSkepuFunctorStruct << ">;\n";

	SSSkepuFunctorStruct << "using ContainerArgs = std::tuple<";
	first = true;
	for (UserFunction::Param& param : UF.anyContainerParams)
	{
		if (!testAndSet(first, false)) SSSkepuFunctorStruct << ", ";
		SSSkepuFunctorStruct << param.fullTypeName;
	}
	SSSkepuFunctorStruct << ">;\n";

	SSSkepuFunctorStruct << "using UniformArgs = std::tuple<";
	first = true;
	for (UserFunction::Param& param : UF.anyScalarParams)
	{
		if (!testAndSet(first, false)) SSSkepuFunctorStruct << ", ";
		SSSkepuFunctorStruct << param.resolvedTypeName;
	}
	SSSkepuFunctorStruct << ">;\n";

	// Proxy tags
	SSSkepuFunctorStruct << "typedef std::tuple<";
	first = true;
	for (UserFunction::Param& param : UF.anyContainerParams)
	{
		if (!testAndSet(first, false)) SSSkepuFunctorStruct << ", ";
		if (param.fullTypeName.find("MatRow") != std::string::npos)
			SSSkepuFunctorStruct << "skepu::ProxyTag::MatRow";
		else
			SSSkepuFunctorStruct << "skepu::ProxyTag::Default";
	}
	SSSkepuFunctorStruct << "> ProxyTags;\n";

	// Access modes
	SSSkepuFunctorStruct << "constexpr static skepu::AccessMode anyAccessMode[] = {\n";
	for (auto param : UF.anyContainerParams)
	{
		SSSkepuFunctorStruct << "skepu::AccessMode::";
		if (param.accessMode == AccessMode::Read)
			SSSkepuFunctorStruct << "Read, ";
		else if (param.accessMode == AccessMode::Write)
			SSSkepuFunctorStruct << "Write, ";
		else if (param.accessMode == AccessMode::ReadWrite)
			SSSkepuFunctorStruct << "ReadWrite, ";
	}
	SSSkepuFunctorStruct << "};\n\n";

	SSSkepuFunctorStruct << "using Ret = " << UF.resolvedReturnTypeName << ";\n\n";
	if (UF.multipleReturnTypes.size() == 0 && UF.rawReturnTypeName != UF.resolvedReturnTypeName && (std::find(usingDecls.begin(), usingDecls.end(), UF.rawReturnTypeName) == usingDecls.end()))
		SSSkepuFunctorStruct << "using " << UF.rawReturnTypeName << " = " << UF.resolvedReturnTypeName << ";\n\n";
	SSSkepuFunctorStruct << "constexpr static bool prefersMatrix = " << (UF.indexed2D) << ";\n\n";

	// CUDA code
	if (GenCUDA)
	{
		SSSkepuFunctorStruct << "#define SKEPU_USING_BACKEND_CUDA 1\n";
		SSSkepuFunctorStruct << "#undef VARIANT_CPU\n";
		SSSkepuFunctorStruct << "#undef VARIANT_OPENMP\n";
		SSSkepuFunctorStruct << "#undef VARIANT_CUDA\n";
		SSSkepuFunctorStruct << "#define VARIANT_CPU(block)\n";
		SSSkepuFunctorStruct << "#define VARIANT_OPENMP(block)\n";
		SSSkepuFunctorStruct << "#define VARIANT_CUDA(block) block\n";
		SSSkepuFunctorStruct << "static inline SKEPU_ATTRIBUTE_FORCE_INLINE " << "__device__ " << UF.resolvedReturnTypeName << " CU(";
		printParamList(SSSkepuFunctorStruct, UF);
		SSSkepuFunctorStruct << ")\n{" << replaceReferencesToOtherUFs(UF, [InstanceName] (UserFunction &UF) { return SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName + "::CU"; }) << "\n}\n";
		SSSkepuFunctorStruct << "#undef SKEPU_USING_BACKEND_CUDA\n\n";
	}

	if (GenOMP)
	{
		SSSkepuFunctorStruct << "#define SKEPU_USING_BACKEND_OMP 1\n";
		SSSkepuFunctorStruct << "#undef VARIANT_CPU\n";
		SSSkepuFunctorStruct << "#undef VARIANT_OPENMP\n";
		SSSkepuFunctorStruct << "#undef VARIANT_CUDA\n";
		SSSkepuFunctorStruct << "#define VARIANT_CPU(block)\n";
		SSSkepuFunctorStruct << "#define VARIANT_OPENMP(block) block\n";
		SSSkepuFunctorStruct << "#define VARIANT_CUDA(block)\n";
		SSSkepuFunctorStruct << "static inline SKEPU_ATTRIBUTE_FORCE_INLINE " << UF.resolvedReturnTypeName << " OMP(";
		printParamList(SSSkepuFunctorStruct, UF);
		SSSkepuFunctorStruct << ")\n{" << replaceReferencesToOtherUFs(UF, [InstanceName] (UserFunction &UF) { return SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName + "::OMP"; }) << "\n}\n";
		SSSkepuFunctorStruct << "#undef SKEPU_USING_BACKEND_OMP\n\n";
	}

	// CPU code
	SSSkepuFunctorStruct << "#define SKEPU_USING_BACKEND_CPU 1\n";
	SSSkepuFunctorStruct << "#undef VARIANT_CPU\n";
	SSSkepuFunctorStruct << "#undef VARIANT_OPENMP\n";
	SSSkepuFunctorStruct << "#undef VARIANT_CUDA\n";
	SSSkepuFunctorStruct << "#define VARIANT_CPU(block) block\n";
	SSSkepuFunctorStruct << "#define VARIANT_OPENMP(block)\n";
	SSSkepuFunctorStruct << "#define VARIANT_CUDA(block) block\n";
	SSSkepuFunctorStruct << "static inline SKEPU_ATTRIBUTE_FORCE_INLINE " << UF.resolvedReturnTypeName << " CPU(";
	printParamList(SSSkepuFunctorStruct, UF);
	SSSkepuFunctorStruct << ")\n{" << replaceReferencesToOtherUFs(UF, [InstanceName] (UserFunction &UF) { return SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName + "::CPU"; }) << "\n}\n";
	SSSkepuFunctorStruct << "#undef SKEPU_USING_BACKEND_CPU\n};\n\n";
	
	if (GlobalRewriter.InsertTextAfter(loc, SSSkepuFunctorStruct.str()))
		SkePUAbort("Code gen target source loc not rewritable: UF " + UF.uniqueName + " for instance" + InstanceName);
}



std::string generateUserTypeCode_CL(UserType &Type)
{
	std::string def = getSourceAsString(Type.astDeclNode->getSourceRange());
	return "typedef " + def + " " + Type.name + ";\n";
}

std::string generateUserFunctionCode_CL(UserFunction &Func)
{
	std::stringstream SSFuncParamList, SSFuncParams, SSFuncSource;
	
	SSFuncSource << generateOpenCLMultipleReturn(Func);
	
	bool first = true;

	if (Func.indexed1D)
	{
		SSFuncParamList << "index1_t " << Func.indexParam->name;
		first = false;
	}
	else if (Func.indexed2D)
	{
		SSFuncParamList << "index2_t " << Func.indexParam->name;
		first = false;
	}
	else if (Func.indexed3D)
	{
		SSFuncParamList << "index3_t " << Func.indexParam->name;
		first = false;
	}
	else if (Func.indexed4D)
	{
		SSFuncParamList << "index4_t " << Func.indexParam->name;
		first = false;
	}

	for (UserFunction::Param& param : Func.elwiseParams)
	{
		if (!first) { SSFuncParamList << ", "; }
		SSFuncParamList << param.rawTypeName << " " << param.name; // HERE1
		first = false;
	}

	for (UserFunction::RandomAccessParam& param : Func.anyContainerParams)
	{
		if (!first) { SSFuncParamList << ", "; }
		SSFuncParamList << param.TypeNameOpenCL() << " " << param.name;
		first = false;
	}

	for (UserFunction::Param& param : Func.anyScalarParams)
	{
		if (!first) { SSFuncParamList << ", "; }
		if (param.astDeclNode->getOriginalType()->isPointerType())
			SSFuncParamList << "__global ";
		SSFuncParamList << param.resolvedTypeName << " " << param.name;
		first = false;
	}

	std::string transformedSource = replaceReferencesToOtherUFs(Func, [] (UserFunction &UF) { return UF.uniqueName; }, true);

	// Recursive call, potential for circular loop: TODO FIX!!
	for (UserFunction *RefFunc : Func.ReferencedUFs)
		SSFuncSource << generateUserFunctionCode_CL(*RefFunc);

	SSFuncSource << "static " << Func.rawReturnTypeName << " " << Func.uniqueName << "(" << SSFuncParamList.str() << ")\n{";
	for (UserFunction::TemplateArgument &arg : Func.templateArguments)
		SSFuncSource << "typedef " << arg.rawTypeName << " " << arg.paramName << ";\n";

	SSFuncSource << transformedSource << "\n}\n\n";
	return SSFuncSource.str();
}


bool transformSkeletonInvocation(const Skeleton &skeleton, std::string InstanceName, std::vector<UserFunction*> FuncArgs, std::vector<size_t> arity, VarDecl *d)
{
	SkePULog() << "Name of skeleton: " << skeleton.name << "\n";
	
	if (GlobalRewriter.RemoveText(d->getSourceRange()))
		SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);
	
	for (UserFunction* UF : FuncArgs)
	{
		const DeclContext *DeclCtx = d->getDeclContext();
		SourceLocation loc = d->getSourceRange().getBegin();
		if (isa<FunctionDecl>(DeclCtx))
		{
			loc = dyn_cast<FunctionDecl>(DeclCtx)->getSourceRange().getBegin();
		}
		
		generateUserFunctionStruct(*UF, InstanceName, loc);
	}


	std::stringstream SSTemplateArgs, SSCallArgs, SSNewDecl;

	bool first = true;
	if (skeleton.type == Skeleton::Type::Map || skeleton.type == Skeleton::Type::MapReduce)
	{
		SSTemplateArgs << arity[0];
		first = false;
	}
	else if (skeleton.type == Skeleton::Type::MapPairs || skeleton.type == Skeleton::Type::MapPairsReduce)
	{
		SSTemplateArgs << arity[0] << ", " << arity[1];
		first = false;
	}

	for (UserFunction *func : FuncArgs)
	{
		if (!first)
			SSTemplateArgs << ", ";
		SSTemplateArgs << SkePU_UF_Prefix << InstanceName << "_" << func->uniqueName;
		first = false;
	}

	if (GenCUDA)
	{
		std::string KernelName_CU;
		switch (skeleton.type)
		{
		case Skeleton::Type::MapReduce:
			KernelName_CU = createMapReduceKernelProgram_CU(*FuncArgs[0], *FuncArgs[1], arity[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "), decltype(&" << KernelName_CU << "_ReduceOnly)";
			SSCallArgs << KernelName_CU << ", " << KernelName_CU << "_ReduceOnly";
			break;
			
		case Skeleton::Type::Map:
			KernelName_CU = createMapKernelProgram_CU(*FuncArgs[0], arity[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;
			
		case Skeleton::Type::MapPairs:
			KernelName_CU = createMapPairsKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;
			
		case Skeleton::Type::MapPairsReduce:
			SkePUAbort("CUDA MapPairsReduce not implemented yet");
			// TODO
			break;
			
		case Skeleton::Type::Reduce1D:
			KernelName_CU = createReduce1DKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;
			
		case Skeleton::Type::Reduce2D:
			KernelName_CU = createReduce2DKernelProgram_CU(*FuncArgs[0], *FuncArgs[1], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_RowWise), decltype(&" << KernelName_CU << "_ColWise)";
			SSCallArgs << KernelName_CU << "_RowWise, " << KernelName_CU << "_ColWise";
			break;
			
		case Skeleton::Type::Scan:
			KernelName_CU = createScanKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_ScanKernel), decltype(&" << KernelName_CU << "_ScanUpdate), decltype(&" << KernelName_CU << "_ScanAdd)";
			SSCallArgs << KernelName_CU << "_ScanKernel, " << KernelName_CU << "_ScanUpdate, " << KernelName_CU << "_ScanAdd";
			break;
			
		case Skeleton::Type::MapOverlap1D:
			KernelName_CU = createMapOverlap1DKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_MapOverlapKernel_CU), decltype(&" << KernelName_CU << "_MapOverlapKernel_CU_Matrix_Row), decltype(&"
				<< KernelName_CU << "_MapOverlapKernel_CU_Matrix_Col), decltype(&" << KernelName_CU << "_MapOverlapKernel_CU_Matrix_ColMulti)";
			SSCallArgs << KernelName_CU << "_MapOverlapKernel_CU, " << KernelName_CU << "_MapOverlapKernel_CU_Matrix_Row, "
				<< KernelName_CU << "_MapOverlapKernel_CU_Matrix_Col, " << KernelName_CU << "_MapOverlapKernel_CU_Matrix_ColMulti";
			break;
			
		case Skeleton::Type::MapOverlap2D:
			KernelName_CU = createMapOverlap2DKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_conv_cuda_2D_kernel)";
			SSCallArgs << KernelName_CU << "_conv_cuda_2D_kernel";
			break;
			
		case Skeleton::Type::MapOverlap3D:
		case Skeleton::Type::MapOverlap4D:
			SkePUAbort("CUDA MapOverlap disabled in this release");
			
		case Skeleton::Type::Call:
			KernelName_CU = createCallKernelProgram_CU(*FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;
		}

		// Insert the code at the proper place
		const DeclContext *DeclCtx = d->getDeclContext();
		SourceLocation loc = d->getSourceRange().getBegin();
		if (isa<FunctionDecl>(DeclCtx))
		{
			loc = dyn_cast<FunctionDecl>(DeclCtx)->getSourceRange().getBegin();
		}
		if (GlobalRewriter.InsertText(loc, "#include \"" + KernelName_CU + ".cu\"\n"))
			SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);
	}
	else
	{
		SSTemplateArgs << ", bool";
		SSCallArgs << "false";

		// Insert dummy arguments
		for (size_t i = 1; i < skeleton.deviceKernelAmount; ++i)
		{
			SSTemplateArgs << ", bool";
			SSCallArgs << ", false";
		}
	}

	if (GenCL)
	{
		std::string KernelName_CL;
		switch (skeleton.type)
		{
		case Skeleton::Type::MapReduce:
			KernelName_CL = createMapReduceKernelProgram_CL(*FuncArgs[0], *FuncArgs[1], arity[0], ResultDir);
			break;
			
		case Skeleton::Type::Map:
			KernelName_CL = createMapKernelProgram_CL(*FuncArgs[0], arity[0], ResultDir);
			break;
			
		case Skeleton::Type::MapPairs:
			KernelName_CL = createMapPairsKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::MapPairsReduce:
			SkePUAbort("MapPairsReduce for OpenCL is not complete yet. Disable OpenCL code-gen for now.");
			break;
			
		case Skeleton::Type::Reduce1D:
			KernelName_CL = createReduce1DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::Reduce2D:
			KernelName_CL = createReduce2DKernelProgram_CL(*FuncArgs[0], *FuncArgs[1], ResultDir);
			break;
			
		case Skeleton::Type::Scan:
			KernelName_CL = createScanKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::MapOverlap1D:
			SkePUAbort("MapOverlap for OpenCL is disabled in this release. De-select OpenCL backend for this program.");
			KernelName_CL = createMapOverlap1DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::MapOverlap2D:
			SkePUAbort("MapOverlap for OpenCL is disabled in this release. De-select OpenCL backend for this program.");
			KernelName_CL = createMapOverlap2DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::MapOverlap3D:
			SkePUAbort("MapOverlap for OpenCL is disabled in this release. De-select OpenCL backend for this program.");
		//	KernelName_CL = createMapOverlap3DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::MapOverlap4D:
			SkePUAbort("MapOverlap for OpenCL is disabled in this release. De-select OpenCL backend for this program.");
		//	KernelName_CL = createMapOverlap4DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::Call:
			KernelName_CL = createCallKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
		}

		// Insert the code at the proper place
		SourceLocation loc = d->getSourceRange().getBegin();
		if (const FunctionDecl *DeclCtx = dyn_cast<FunctionDecl>(d->getDeclContext()))
			loc = DeclCtx->getSourceRange().getBegin();

		if (GlobalRewriter.InsertText(loc, "#include \"" + KernelName_CL + "_cl_source.inl\"\n"))
			SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);

		SSTemplateArgs << ", CLWrapperClass_" << KernelName_CL;
	}
	else
	{
		SSTemplateArgs << ", void";
	}

	if (d->isStaticLocal())
		SSNewDecl << "static ";
	SSNewDecl << "skepu::backend::" << skeleton.name << "<" << SSTemplateArgs.str() << "> " << InstanceName << "(" << SSCallArgs.str() << ")";

	if (GlobalRewriter.InsertText(d->getSourceRange().getBegin(), SSNewDecl.str()))
		SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);

	return true;
}
