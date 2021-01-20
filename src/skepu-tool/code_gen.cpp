#include "code_gen.h"

using namespace clang;

enum class Backend
{
	CPU, OpenMP, CUDA, OpenCL,
};

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
	
	if (UserFunction::RegionParam *param = Func.regionParam)
	{
		if (!first) { o << ", "; }
		o << param->fullTypeName << " " << param->name;
		first = false;
	}

	for (UserFunction::Param& param : Func.elwiseParams)
	{
		if (!first) { o << ", "; }
		o << param.resolvedTypeName << " " << param.name;
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

std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements)
{
	for(std::pair<std::string, std::string> &element : replacements)
		replaceTextInString(templ, element.first, element.second);
	return templ;
}

std::string transformToCXXIdentifier(std::string &in)
{
	std::string out = in;
	replaceTextInString(out, ".", "__dot__");
	replaceTextInString(out, " ", "__space__");
	replaceTextInString(out, ":", "__colon__");
	replaceTextInString(out, "-", "__hyphen__");
	return out;
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


std::string generateCUDAMultipleReturn(UserFunction &UF)
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
			SSmultiReturnMakeStruct << "arg" << ctr << divider;
			ctr++;
		}
		
		std::string codeTemplate = R"~~~(
			struct {{SKEPU_MULTIPLE_RETURN_TYPE}}
			{
				{{SKEPU_MULTIPLE_RETURN_TYPE_DEF}}
				
				static __device__ {{SKEPU_MULTIPLE_RETURN_TYPE}} make({{SKEPU_MULTIPLE_RETURN_MAKE_PARAMS}})
				{
					{{SKEPU_MULTIPLE_RETURN_TYPE}} retval = { {{SKEPU_MULTIPLE_RETURN_MAKE_STRUCT}} };
					return retval;
				}
			};
		)~~~";
		
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_TYPE}}", SSmultiReturnType.str());
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_TYPE_DEF}}", SSmultiReturnTypeDef.str());
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_MAKE_PARAMS}}", SSmultiReturnMakeParams.str());
		replaceTextInString(codeTemplate, "{{SKEPU_MULTIPLE_RETURN_MAKE_STRUCT}}", SSmultiReturnMakeStruct.str());
		
		return codeTemplate;
	}
	return "";
}



std::map<std::string, std::pair<int, std::string>> proxyInfo = {
	{"Vec",      {2, "skepu_vec_proxy_access_"}},
	{"Mat",      {3, "skepu_mat_proxy_access_"}},
	{"MatRow",   {2, "skepu_matrow_proxy_access_"}},
	{"MatCol",   {2, "skepu_matcol_proxy_access_"}},
	{"Ten3",     {4, "skepu_ten3_proxy_access_"}},
	{"Ten4",     {5, "skepu_ten4_proxy_access_"}},
	{"Region1D", {2, "skepu_region_access_1d_"}},
	{"Region2D", {3, "skepu_region_access_2d_"}},
	{"Region3D", {4, "skepu_region_access_3d_"}},
	{"Region4D", {5, "skepu_region_access_4d_"}},
};
std::string replaceReferencesToOtherUFs(Backend backend, UserFunction &UF, std::function<std::string(UserFunction&)> nameFunc)
{
	const FunctionDecl *f = UF.astDeclNode;
	if (f->getTemplatedKind() == FunctionDecl::TK_FunctionTemplateSpecialization)
		f = f->getTemplateInstantiationPattern();
	
	// Find references to other userfunctions
	Rewriter R(GlobalRewriter.getSourceMgr(), LangOptions());
	
	if (UF.multipleReturnTypes.size() > 0)
	{
		if (backend == Backend::OpenCL)
		{
			for (auto *ref : UF.ReferencedRets)
				R.ReplaceText(ref->getCallee()->getSourceRange(), "make_" + UF.multiReturnTypeNameGPU());
		}
		else if (backend == Backend::CUDA)
		{
			for (auto *ref : UF.ReferencedRets)
				R.ReplaceText(ref->getCallee()->getSourceRange(), UF.multiReturnTypeNameGPU() + "::make");
		}
	}
	
	for (auto &ref : UF.UFReferences)
		R.ReplaceText(ref.first->getCallee()->getSourceRange(), nameFunc(*ref.second));
	
	for (auto &ref : UF.UTReferences)
		R.ReplaceText(ref.first->getTypeLoc().getSourceRange(), "struct " + ref.second->name);
	
	for (auto subscript : UF.containerSubscripts)
		R.InsertText(subscript->getCallee()->getBeginLoc(), ".data");
	
	if (backend == Backend::OpenCL)
		for (auto subscript : UF.containerCalls)
		{
			DeclRefExpr* container = dyn_cast<clang::DeclRefExpr>(subscript->getArg(0));
			auto type = container->getDecl()->getType().getTypePtr();
			if (auto *innertype = dyn_cast<ElaboratedType>(type))
				type = innertype->getNamedType().getTypePtr();
			const auto *templateType = dyn_cast<TemplateSpecializationType>(type);
			
			std::string templateName = templateType->getTemplateName().getAsTemplateDecl()->getNameAsString();
			std::string varname = container->getNameInfo().getAsString();
			std::string typeName = templateType->getArg(0).getAsType().getAsString();
			
			int numArgs;
			std::string fname;
			std::tie(numArgs, fname) = proxyInfo[templateName];
			fname += transformToCXXIdentifier(typeName);
			std::string args = getSourceAsString(clang::SourceRange(subscript->getArg(1)->getBeginLoc(), subscript->getArg(numArgs-1)->getEndLoc()));
			R.ReplaceText(subscript->getSourceRange(), fname + "(" + varname + "," + args + ")");
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

static std::set<std::string> generatedStructs;
static std::set<std::string> usingDecls;

void generateUserFunctionStruct(UserFunction &UF, std::string InstanceName, clang::SourceLocation loc)
{
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
		else if (param.fullTypeName.find("MatCol") != std::string::npos)
			SSSkepuFunctorStruct << "skepu::ProxyTag::MatCol";
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
	
	
	SSSkepuFunctorStruct << generateCUDAMultipleReturn(UF);
	
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
		SSSkepuFunctorStruct << "static inline SKEPU_ATTRIBUTE_FORCE_INLINE " << "__device__ ";
		if (UF.multipleReturnTypes.size() > 0)
			SSSkepuFunctorStruct << UF.multiReturnTypeNameGPU();
		else
			SSSkepuFunctorStruct << UF.resolvedReturnTypeName;
		SSSkepuFunctorStruct << " CU(";
		printParamList(SSSkepuFunctorStruct, UF);
		SSSkepuFunctorStruct << ")\n{" << replaceReferencesToOtherUFs(Backend::CUDA, UF, [InstanceName] (UserFunction &UF) { return SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName + "::CU"; }) << "\n}\n";
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
		SSSkepuFunctorStruct << ")\n{" << replaceReferencesToOtherUFs(Backend::OpenMP, UF, [InstanceName] (UserFunction &UF) { return SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName + "::OMP"; }) << "\n}\n";
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
	SSSkepuFunctorStruct << ")\n{" << replaceReferencesToOtherUFs(Backend::CPU, UF, [InstanceName] (UserFunction &UF) { return SkePU_UF_Prefix + InstanceName + "_" + UF.uniqueName + "::CPU"; }) << "\n}\n";
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
	
	if (UserFunction::RegionParam *param = Func.regionParam)
	{
		if (!first) { SSFuncParamList << ", "; }
		if (param->containerType == ContainerType::Region1D)
			SSFuncParamList << "skepu_region1d_" << transformToCXXIdentifier(param->resolvedTypeName) << " " << param->name;
		else if (param->containerType == ContainerType::Region2D)
			SSFuncParamList << "skepu_region2d_" << transformToCXXIdentifier(param->resolvedTypeName) << " " << param->name;
		else if (param->containerType == ContainerType::Region3D)
			SSFuncParamList << "skepu_region3d_" << transformToCXXIdentifier(param->resolvedTypeName) << " " << param->name;
		else if (param->containerType == ContainerType::Region4D)
			SSFuncParamList << "skepu_region4d_" << transformToCXXIdentifier(param->resolvedTypeName) << " " << param->name;
		first = false;
	}

	for (UserFunction::Param& param : Func.elwiseParams)
	{
		if (!first) { SSFuncParamList << ", "; }
		SSFuncParamList << param.rawTypeName << " " << param.name;
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
//	if (param.astDeclNode->getOriginalType()->isPointerType())
//		SSFuncParamList << "__global ";
		SSFuncParamList << param.resolvedTypeName << " " << param.name;
		first = false;
	}

	std::string transformedSource = replaceReferencesToOtherUFs(Backend::OpenCL, Func, [] (UserFunction &UF) { return UF.uniqueName; });

	// Recursive call, potential for circular loop: TODO FIX!!
	for (UserFunction *RefFunc : Func.ReferencedUFs)
		SSFuncSource << generateUserFunctionCode_CL(*RefFunc);
	
		
	// The function itself
	SSFuncSource << "static ";
	if (Func.multipleReturnTypes.size() > 0)
		SSFuncSource << Func.multiReturnTypeNameGPU();
	else
		SSFuncSource << Func.rawReturnTypeName;
	SSFuncSource << " " << Func.uniqueName << "(" << SSFuncParamList.str() << ")\n{";
	for (UserFunction::TemplateArgument &arg : Func.templateArguments)
		SSFuncSource << "typedef " << arg.rawTypeName << " " << arg.paramName << ";\n";

	SSFuncSource << transformedSource << "\n}\n\n";
	return SSFuncSource.str();
}

static int skeletonCounter = 0;

bool transformSkeletonInvocation(const Skeleton &skeleton, std::string InstanceName, std::vector<UserFunction*> FuncArgs, std::vector<size_t> arity, VarDecl *d)
{
	generatedStructs = {};
	std::stringstream ss;
	ss << "skepu_skel_" << skeletonCounter++;
	std::string skeletonID = ss.str();
	
	SkePULog() << "Name of skeleton: " << skeleton.name << ", Unique ID: " << skeletonID << "\n";
	
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
		
		generateUserFunctionStruct(*UF, skeletonID + InstanceName, loc);
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
		SSTemplateArgs << SkePU_UF_Prefix << skeletonID << InstanceName << "_" << func->uniqueName;
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
			KernelName_CL = createMapReduceKernelProgram_CL(*FuncArgs[0], *FuncArgs[1], ResultDir);
			break;
			
		case Skeleton::Type::Map:
			KernelName_CL = createMapKernelProgram_CL(*FuncArgs[0], ResultDir);
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
			KernelName_CL = createMapOverlap1DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::MapOverlap2D:
			KernelName_CL = createMapOverlap2DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::MapOverlap3D:
			SkePUAbort("3D MapOverlap for OpenCL is disabled in this release. De-select OpenCL backend for this program.");
		//	KernelName_CL = createMapOverlap3DKernelProgram_CL(*FuncArgs[0], ResultDir);
			break;
			
		case Skeleton::Type::MapOverlap4D:
			SkePUAbort("4D MapOverlap for OpenCL is disabled in this release. De-select OpenCL backend for this program.");
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
