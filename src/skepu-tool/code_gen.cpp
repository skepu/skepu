#include "code_gen.h"
#include "code_gen_cu.h"

using namespace clang;

std::string lineDirectiveForSourceLoc(SourceLocation &loc)
{
	std::stringstream directive;
//	if (!DoNotGenLineDirectives)
//		directive << "#line " << GlobalRewriter.getSourceMgr().getSpellingLineNumber(loc) << " \"" + inputFileName << "\"\n";
	return directive.str();
}

std::string getSourceAsString(SourceRange range)
{
	int rangeSize = GlobalRewriter.getRangeSize(range);
	SkePULog() << "Range size: " << rangeSize << "\n";
	if (rangeSize == -1)
		return "";

	SourceLocation startLoc = range.getBegin();
	const char *strStart = GlobalRewriter.getSourceMgr().getCharacterData(startLoc);

	std::string exprString;
	exprString.assign(strStart, rangeSize);

	return exprString;
}

int getRangeSize(SourceRange range)
{
	return GlobalRewriter.getRangeSize(range);
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
	
	if (UserFunction::RandomParam *param = Func.randomParam)
	{
		if (!first) { o << ", "; }
		o << "skepu::Random<" << param->randomCount << "> ";
		if (param->isLValueReference) o << "& ";
		if (param->isRValueReference) o << "&& ";
		o << param->name;
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
	replaceTextInString(out, "<", "__lt__");
	replaceTextInString(out, ">", "__gt__");
	return out;
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
	SkePULog() << "Modifying UF code for " << nameFunc(UF) << "\n";
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
	
	if (backend == Backend::OpenCL)
	{
		for (auto *ref : UF.ReferencedGets)
		{
			auto* object = ref->getImplicitObjectArgument();
			if (auto *expr = dyn_cast<ImplicitCastExpr>(object))
				object = expr->getSubExpr();
			DeclRefExpr* object2 = dyn_cast<clang::DeclRefExpr>(object);
			std::string varname = object2->getNameInfo().getAsString();
			
			FunctionDecl *Func = ref->getDirectCallee();
			std::string name = Func->getName();
			std::string variant = ((name == "get") ? "get" : "get_normalized");
			
			R.ReplaceText(ref->getSourceRange(), "skepu_random_" + variant + "(" + varname + ")");
		}
	}
	
	for (auto &ref : UF.UFReferences)
	{
		SkePULog() << "--> Replacing UF reference with " << nameFunc(*ref.second) << "\n";
		R.ReplaceText(ref.first->getCallee()->getSourceRange(), nameFunc(*ref.second));
	}

	for (auto &ref : UF.UTReferences)
		R.ReplaceText(ref.first->getTypeLoc().getSourceRange(), "struct " + ref.second->name);
	
	for (auto subscript : UF.containerSubscripts)
		R.InsertText(subscript->getCallee()->getBeginLoc(), ".data");
	
	if (backend == Backend::OpenCL)
		for (auto subscript : UF.containerCalls)
		{
			Expr *arg0 = subscript->getArg(0);
			if (auto *expr = dyn_cast<ImplicitCastExpr>(arg0))
				arg0 = expr->getSubExpr();
			DeclRefExpr* container = dyn_cast<clang::DeclRefExpr>(arg0);
			
			auto type = container->getDecl()->getType().getTypePtr();
			if (auto *innertype = dyn_cast<ElaboratedType>(type))
				type = innertype->getNamedType().getTypePtr();
			const auto *templateType = dyn_cast<TemplateSpecializationType>(type);

			std::string templateName = templateType->getTemplateName().getAsTemplateDecl()->getNameAsString();
			std::string varname = container->getNameInfo().getAsString();
			std::string typeName = templateType->getArg(0).getAsType().getAsString();
			replaceTextInString(typeName, "struct ", "");
			
			int numArgs;
			std::string fname;
			std::tie(numArgs, fname) = proxyInfo[templateName];
			fname += transformToCXXIdentifier(typeName);
			std::string args = getSourceAsString(clang::SourceRange(subscript->getArg(1)->getBeginLoc(), subscript->getArg(numArgs-1)->getEndLoc()));
			R.ReplaceText(subscript->getSourceRange(), fname + "(" + varname + "," + args + ")");
		}
	
	if (backend == Backend::OpenCL)
		for (auto overload : UF.operatorOverloads)
		{
			auto ooc = overload->getOperator();
	//		overload->dump();
			std::string functionName;
			int operatorLength = 1;
			if (ooc == OO_Plus)
			{
				functionName = "complex_add"; // add inner type, differentiate between add, addr, addr2
				operatorLength = 1;
			}
			else if (ooc == OO_Minus)
			{
				functionName = "complex_sub"; // add inner type, differentiate between add, addr, addr2
				operatorLength = 1;
			}
			else if (ooc == OO_Star)
			{
				functionName = "complex_mul"; // add inner type, differentiate between add, addr, addr2
				operatorLength = 1;
			}
			else if (ooc == OO_Slash)
			{
				functionName = "complex_div"; // add inner type, differentiate between add, addr, addr2
				operatorLength = 1;
			}
			else if (ooc == OO_EqualEqual)
			{
				functionName = "complex_equal"; // add inner type, differentiate between add, addr, addr2
				operatorLength = 2;
			}
			else if (ooc == OO_ExclaimEqual)
			{
				functionName = "complex_notequal"; // add inner type, differentiate between add, addr, addr2
				operatorLength = 2;
			}
			else if (ooc == OO_PlusEqual)
			{
				functionName = "complex_assign_add"; // add inner type, differentiate between add, addr, addr2
				operatorLength = 2;
			}
			else if (ooc == OO_MinusEqual)
			{
				functionName = "complex_assign_sub"; // add inner type, differentiate between add, addr, addr2
				operatorLength = 2;
			}
			else if (ooc == OO_StarEqual)
			{
				functionName = "complex_assign_mul"; // add inner type, differentiate between add, addr, addr2
				operatorLength = 2;
			}
			
			R.InsertText(overload->getBeginLoc(), functionName + "(");
			R.InsertText(overload->getArg(1)->getBeginLoc().getLocWithOffset(getRangeSize(overload->getArg(1)->getSourceRange())), ")");
			R.ReplaceText(clang::SourceRange(overload->getOperatorLoc(), overload->getOperatorLoc().getLocWithOffset(operatorLength)), ",");
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

void generateUserFunctionStruct(UserFunction &UF, std::string InstanceName, clang::SourceLocation loc)
{
	std::set<std::string> usingDecls;
	
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
			if (arg.rawTypeName != arg.resolvedTypeName && (std::find(usingDecls.begin(), usingDecls.end(), arg.rawTypeName) == usingDecls.end()))
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
	SSSkepuFunctorStruct << "constexpr static bool usesPRNG = " << (UF.randomParam != nullptr) << ";\n";
	if (UF.randomParam != nullptr) SSSkepuFunctorStruct << "constexpr static size_t randomCount = " << UF.randomCount << ";\n";
	else SSSkepuFunctorStruct << "constexpr static size_t randomCount = SKEPU_NO_RANDOM;\n";
	
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

	// CUDA code
	if (GenCUDA)
	{
		SSSkepuFunctorStruct << generateCUDAMultipleReturn(UF);
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
	SSSkepuFunctorStruct << lineDirectiveForSourceLoc(loc);

	if (GlobalRewriter.InsertTextAfter(loc, SSSkepuFunctorStruct.str()))
		SkePUAbort("Code gen target source loc not rewritable: UF " + UF.uniqueName + " for instance" + InstanceName);
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

	// Find location to insert transformed user function code
	const DeclContext *DeclCtx = d->getDeclContext();
	SourceLocation loc = d->getSourceRange().getBegin();
	if (isa<FunctionDecl>(DeclCtx))
	{
		const FunctionDecl *f = dyn_cast<FunctionDecl>(DeclCtx);
		if (f->getTemplatedKind() == FunctionDecl::TK_FunctionTemplateSpecialization)
		{
			// If it is a template, we need to place the code before the template begins
			loc = f->getPrimaryTemplate()->getBeginLoc();
		}
		else
		{
			loc = f->getSourceRange().getBegin();
		}
	}

	for (UserFunction* UF : FuncArgs)
	{
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
			KernelName_CU = createMapReduceKernelProgram_CU(skeletonID, *FuncArgs[0], *FuncArgs[1], arity[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "), decltype(&" << KernelName_CU << "_ReduceOnly)";
			SSCallArgs << KernelName_CU << ", " << KernelName_CU << "_ReduceOnly";
			break;

		case Skeleton::Type::Map:
			KernelName_CU = createMapKernelProgram_CU(skeletonID, *FuncArgs[0], arity[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;

		case Skeleton::Type::MapPairs:
			KernelName_CU = createMapPairsKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;

		case Skeleton::Type::MapPairsReduce:
			KernelName_CU = createMapPairsReduceKernelProgram_CU(skeletonID, *FuncArgs[0], *FuncArgs[1], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;

		case Skeleton::Type::Reduce1D:
			KernelName_CU = createReduce1DKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;

		case Skeleton::Type::Reduce2D:
			KernelName_CU = createReduce2DKernelProgram_CU(skeletonID, *FuncArgs[0], *FuncArgs[1], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_RowWise), decltype(&" << KernelName_CU << "_ColWise)";
			SSCallArgs << KernelName_CU << "_RowWise, " << KernelName_CU << "_ColWise";
			break;

		case Skeleton::Type::Scan:
			KernelName_CU = createScanKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_ScanKernel), decltype(&" << KernelName_CU << "_ScanUpdate), decltype(&" << KernelName_CU << "_ScanAdd)";
			SSCallArgs << KernelName_CU << "_ScanKernel, " << KernelName_CU << "_ScanUpdate, " << KernelName_CU << "_ScanAdd";
			break;

		case Skeleton::Type::MapOverlap1D:
			KernelName_CU = createMapOverlap1DKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_MapOverlapKernel_CU), decltype(&" << KernelName_CU << "_MapOverlapKernel_CU_Matrix_Row), decltype(&"
				<< KernelName_CU << "_MapOverlapKernel_CU_Matrix_Col), decltype(&" << KernelName_CU << "_MapOverlapKernel_CU_Matrix_ColMulti)";
			SSCallArgs << KernelName_CU << "_MapOverlapKernel_CU, " << KernelName_CU << "_MapOverlapKernel_CU_Matrix_Row, "
				<< KernelName_CU << "_MapOverlapKernel_CU_Matrix_Col, " << KernelName_CU << "_MapOverlapKernel_CU_Matrix_ColMulti";
			break;

		case Skeleton::Type::MapOverlap2D:
			KernelName_CU = createMapOverlap2DKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_conv_cuda_2D_kernel)";
			SSCallArgs << KernelName_CU << "_conv_cuda_2D_kernel";
			break;

		case Skeleton::Type::MapOverlap3D:
			KernelName_CU = createMapOverlap3DKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << "_conv_cuda_3D_kernel)";
			SSCallArgs << KernelName_CU << "_conv_cuda_3D_kernel";
			break;
		
		case Skeleton::Type::MapOverlap4D:
			SkePUAbort("CUDA MapOverlap 4D disabled in this release");

		case Skeleton::Type::Call:
			KernelName_CU = createCallKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			SSTemplateArgs << ", decltype(&" << KernelName_CU << ")";
			SSCallArgs << KernelName_CU;
			break;
		}

		// Insert the code at the proper place
	/*	const DeclContext *DeclCtx = d->getDeclContext();
		SourceLocation loc = d->getSourceRange().getBegin();
		if (isa<FunctionDecl>(DeclCtx))
		{
			loc = dyn_cast<FunctionDecl>(DeclCtx)->getSourceRange().getBegin();
		}*/
		if (GlobalRewriter.InsertText(loc, "#include \"" + KernelName_CU + ".cu\"\n" + lineDirectiveForSourceLoc(loc)))
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
			KernelName_CL = createMapReduceKernelProgram_CL(skeletonID, *FuncArgs[0], *FuncArgs[1], ResultDir);
			break;

		case Skeleton::Type::Map:
			KernelName_CL = createMapKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::MapPairs:
			KernelName_CL = createMapPairsKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::MapPairsReduce:
			KernelName_CL = createMapPairsReduceKernelProgram_CL(skeletonID, *FuncArgs[0], *FuncArgs[1], ResultDir);
			break;

		case Skeleton::Type::Reduce1D:
			KernelName_CL = createReduce1DKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::Reduce2D:
			KernelName_CL = createReduce2DKernelProgram_CL(skeletonID, *FuncArgs[0], *FuncArgs[1], ResultDir);
			break;

		case Skeleton::Type::Scan:
			KernelName_CL = createScanKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::MapOverlap1D:
			KernelName_CL = createMapOverlap1DKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::MapOverlap2D:
			KernelName_CL = createMapOverlap2DKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::MapOverlap3D:
			KernelName_CL = createMapOverlap3DKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::MapOverlap4D:
			KernelName_CL = createMapOverlap4DKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::Call:
			KernelName_CL = createCallKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;
		}

		if (GlobalRewriter.InsertText(loc, "#include \"" + KernelName_CL + "_cl_source.inl\"\n" + lineDirectiveForSourceLoc(loc)))
			SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);

		SSTemplateArgs << ", CLWrapperClass_" << KernelName_CL;
	}
	else
	{
		SSTemplateArgs << ", void";
	}

	if(d->getStorageClass() == clang::StorageClass::SC_Static)
		SSNewDecl << "static ";
	SSNewDecl << "skepu::backend::" << skeleton.name << "<" << SSTemplateArgs.str() << "> " << InstanceName << "(" << SSCallArgs.str() << ")";

	if (GlobalRewriter.InsertText(d->getSourceRange().getBegin(), SSNewDecl.str()))
		SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);

	return true;
}
