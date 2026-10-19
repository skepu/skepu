#include <map>
#include <fstream>
#include <sha256.hpp>
#include "code_gen.h"
#include "code_gen_cu.h"

// The SHA-256 (hex) of a source file for header.files of the traces; empty if it cannot be read. Each file is read once.
std::string sourceFileHash(const std::string &path)
{
	static std::map<std::string, std::string> hashes;
	auto found = hashes.find(path);
	if (found != hashes.end())
		return found->second;
	std::string hash;
	std::ifstream in(path, std::ios::binary);
	std::ostringstream contents;
	if (in && (contents << in.rdbuf()))
	{
		skepu::tracing::SHA256 sha256;
		hash = sha256(contents.str());
	}
	return hashes[path] = hash;
}

using namespace clang;

std::string lineDirectiveForSourceLoc(SourceLocation &loc, int offset)
{
	std::stringstream directive;
	if (!DoNotGenLineDirectives)
		directive << "#line "
		    << GlobalRewriter.getSourceMgr().getSpellingLineNumber(loc) + offset
			<< " \"" + GlobalRewriter.getSourceMgr().getBufferName(loc).str() << "\"";
	return directive.str();
}


std::string removeNamespacesFromTypeName(std::string type)
{
	replaceTextInString(type, "struct ", "");
	auto pos = type.rfind("::");
	if (pos == std::string::npos)
		return type;      // (a type without a namespace, as unsigned int or double: nothing to remove)
	return type.substr(pos+2, type.size() - pos-2);
}

int getRangeSize(SourceRange range)
{
	return GlobalRewriter.getRangeSize(range);
}

void printParamList(std::ostream &o, UserFunction &Func, bool isCuda = false)
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
		std::string fullTypename = param->fullTypeName;
		if (isCuda)
			replaceTextInString(fullTypename, "Region4D", "Region4DCU");
		o << fullTypename << " " << param->name;
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
	{"Pool1D",   {2, "skepu_pool_access_1d_"}},
	{"Pool2D",   {3, "skepu_pool_access_2d_"}},
	{"Pool3D",   {4, "skepu_pool_access_3d_"}},
	{"Pool4D",   {5, "skepu_pool_access_4d_"}},
};
std::string replaceReferencesToOtherUFs(Backend backend, UserFunction &UF, std::function<std::string(UserFunction&)> nameFunc)
{
	SkePULog() << "Modifying UF code for " << nameFunc(UF) << "\n";
	const FunctionDecl *f = UF.astDeclNode;
	if (f->getTemplatedKind() == FunctionDecl::TK_FunctionTemplateSpecialization)
	{
		// An explicit specialization (template<> ...) has no instantiation
		// pattern, but it has a body of its own.
		if (const FunctionDecl *pattern = f->getTemplateInstantiationPattern())
			f = pattern;
	}

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
			std::string name = Func->getNameAsString();
			std::string variant = ((name == "get") ? "get" : "get_normalized");

			R.ReplaceText(ref->getSourceRange(), "skepu_random_" + variant + "(" + varname + ")");
		}
	}

	for (auto &ref : UF.UFReferences)
	{
		SkePULog() << "--> Replacing UF reference with " << nameFunc(*ref.second) << "\n";
		R.ReplaceText(ref.first->getCallee()->getSourceRange(), nameFunc(*ref.second));
	}

	// The names of user types in the body (not in the signature, which is not part of the code that is copied)
	if (const Stmt *body = f->getBody())
	{
		const SourceManager &SM = GlobalRewriter.getSourceMgr();
		for (auto &ref : UF.UTReferences)
		{
			const SourceRange range = ref.first;
			if (SM.isBeforeInTranslationUnit(range.getBegin(), body->getBeginLoc()) || SM.isBeforeInTranslationUnit(body->getEndLoc(), range.getEnd()))
				continue;

			std::string name = ref.second->typeNameOpenCL; // escaped qualified name, defined in the kernel code
			if (backend != Backend::OpenCL)
			{
				name = ref.second->type->getCanonicalTypeInternal().getAsString();
				replaceTextInString(name, "struct ", "");
			}
			R.ReplaceText(range, name);
		}
	}

	// The user constants that the body names with a qualifier: in OpenCL C they are macros with their plain names
	if (backend == Backend::OpenCL)
		if (const Stmt *body = f->getBody())
		{
			const SourceManager &SM = GlobalRewriter.getSourceMgr();
			for (auto &ref : UF.ConstantReferences)
			{
				if (SM.isBeforeInTranslationUnit(ref.first.getBegin(), body->getBeginLoc()) || SM.isBeforeInTranslationUnit(body->getEndLoc(), ref.first.getEnd()))
					continue;
				R.ReplaceText(ref.first, ref.second);
			}
		}

	// (after the container that is subscripted; the begin of the callee of the operator is not the [ in an instantiated template)
	for (auto subscript : UF.containerSubscripts)
		R.InsertText(Lexer::getLocForEndOfToken(subscript->getArg(0)->getEndLoc(), 0, GlobalRewriter.getSourceMgr(), GlobalRewriter.getLangOpts()), ".data");

	if (backend == Backend::OpenCL)
		for (auto subscript : UF.containerCalls)
		{
			Expr *arg0 = subscript->getArg(0);
			if (auto *expr = dyn_cast<ImplicitCastExpr>(arg0))
				arg0 = expr->getSubExpr();
			DeclRefExpr* container = dyn_cast<clang::DeclRefExpr>(arg0);

			auto type = container->getDecl()->getType().getTypePtr();
			const auto *templateType = dyn_cast<TemplateSpecializationType>(type);

			std::string templateName = templateType->getTemplateName().getAsTemplateDecl()->getNameAsString();
			std::string varname = container->getNameInfo().getAsString();
			// canonical, as the name of the proxy type that is defined for the container parameter
			std::string typeName = templateType->template_arguments()[0].getAsType().getCanonicalType().getAsString();
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

// Where an instance and its user functions are defined, as JSON for the tracer (see Tracer::call): the lines of the declaration
// or function (a lambda: its expression).
static std::string jsonQuoted(const std::string &s)
{
	std::string result = "\"";
	for (char c : s)
	{
		if (c == '"' || c == '\\')
			result += '\\';
		result += c;
	}
	return result + "\"";
}

// (the place that a #line directive gives, if there is one)
static std::string sourceRangeJSON(clang::SourceRange range)
{
	clang::SourceManager &SM = GlobalRewriter.getSourceMgr();
	clang::SourceLocation beginLoc = SM.getSpellingLoc(range.getBegin()), endLoc = SM.getSpellingLoc(range.getEnd());
	clang::PresumedLoc begin = SM.getPresumedLoc(beginLoc), end = SM.getPresumedLoc(endLoc);
	if (begin.isInvalid() || end.isInvalid())
		return "\"file\":" + jsonQuoted(SM.getFilename(beginLoc).str()) + ",\"line\":" + std::to_string(SM.getSpellingLineNumber(beginLoc))
			+ ",\"end_line\":" + std::to_string(SM.getSpellingLineNumber(endLoc));
	return "\"file\":" + jsonQuoted(begin.getFilename()) + ",\"line\":" + std::to_string(begin.getLine())
		+ ",\"end_line\":" + std::to_string(end.getLine());
}

static std::string definitionJSON(UserFunction &UF, VarDecl *d, const InlineInstance *inlineInstance)
{
	const bool lambda = UF.rawName.compare(0, 10, "lambda_uf_") == 0;
	std::string instance = inlineInstance ? sourceRangeJSON(inlineInstance->factoryCall->getSourceRange()) + ",\"anonymous\":true"
		: d ? sourceRangeJSON(d->getSourceRange()) : "";
	return "{\"instance\":{" + instance + "},\"function\":{\"name\":" + jsonQuoted(lambda ? "" : UF.rawName)
		+ ",\"kind\":" + (lambda ? "\"lambda\"" : "\"function\"") + "," + sourceRangeJSON(UF.astDeclNode->getSourceRange()) + "}}";
}

// (by the name of the struct of the user function of the instance: a user function that another one refers to has its struct made first)
static std::map<std::string, std::string> definitionsByStruct;

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
	if (definitionsByStruct.count(FunctorName))
		SSSkepuFunctorStruct << "static constexpr const char *definition = R\"SKEPUDEF(" << definitionsByStruct[FunctorName] << ")SKEPUDEF\";\n";

	// Code generation specific to templated user functions
	if (UF.fromTemplate)
		for (const UserFunction::TemplateArgument &arg : UF.templateArguments)
		{
			// If the template argument type is a nested namespace type, bring the namespace into scope
			if (arg.valueType.empty() && arg.rawTypeName != arg.resolvedTypeName && (std::find(usingDecls.begin(), usingDecls.end(), arg.rawTypeName) == usingDecls.end()))
			{
				SSSkepuFunctorStruct << "using " << arg.rawTypeName << " = " << arg.resolvedTypeName << ";\n";
				usingDecls.insert(arg.rawTypeName);
			}

			// Resolve the template parameter type occurences by a type alias to the argument type
			// (a constant for the value of a non-type parameter)
			SSSkepuFunctorStruct << arg.alias(true);
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

	if (UF.regionParam)
	{
		SSSkepuFunctorStruct << "constexpr static bool isPool = "
			<< ((UF.regionParam->isPool) ? "true" : "false") << ";\n";
	}

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
	{
		SSSkepuFunctorStruct << "using " << UF.rawReturnTypeName << " = " << UF.resolvedReturnTypeName << ";\n\n";
		usingDecls.insert(UF.rawReturnTypeName);
	}

	else if (UF.multipleReturnTypes.size() > 0)
		for (std::string type : UF.multipleReturnTypes)
		{
			std::string rawTypeName = removeNamespacesFromTypeName(type);
			std::string resolvedTypeName = type;

			if (rawTypeName != resolvedTypeName && (std::find(usingDecls.begin(), usingDecls.end(), rawTypeName) == usingDecls.end()))
			{
				SSSkepuFunctorStruct << "using " << rawTypeName << " = " << resolvedTypeName << ";\n\n";
				usingDecls.insert(rawTypeName);
			}
		}

	SSSkepuFunctorStruct << "constexpr static bool prefersMatrix = " << (UF.indexed2D) << ";\n\n";

	// Whether the function needs double precision (a skeleton that has it does not run on an OpenCL device that does not: see SkeletonBase::selectBackend)
	SSSkepuFunctorStruct << "constexpr static bool requiresDouble = " << (UF.requiresDoublePrecision ? 1 : 0) << ";\n\n";

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
		printParamList(SSSkepuFunctorStruct, UF, true);
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
	SSSkepuFunctorStruct << lineDirectiveForSourceLoc(loc) << "\n";

	if (GlobalRewriter.InsertTextAfter(loc, SSSkepuFunctorStruct.str()))
		SkePUAbort("Code gen target source loc not rewritable: UF " + UF.uniqueName + " for instance" + InstanceName);
}

static int skeletonCounter = 0;

// A skeleton instance in a function template is visited once per instantiation of the template, but its
// declaration exists once, in the template. Each instantiation can have other user functions (and so other
// kernels), so the declaration names the kernels through a selector class template that has the template
// parameters of the function template. For each instantiation, a specialization of the selector is
// emitted before the function template; it gives the names of that instantiation's user function structs,
// kernel classes and kernel functions as members, which the declaration refers to:
//
//   template<typename T> struct skepu_selector_0;                             // primary, declared once
//   template<> struct skepu_selector_0<float> { using uf0 = ...; using cl = ...; };
//   ...
//   skepu::backend::Map<1, typename skepu_selector_0<T>::uf0, bool, typename skepu_selector_0<T>::cl> skel("skel", false);
//
// Skeleton instances in other places than function templates at namespace scope are rewritten as they are.
namespace
{
	struct TemplateSite
	{
		std::string selector;            // name of the selector class template
		std::string header;              // its template parameter list
		std::string uses;                // the parameters as arguments: the selector's arguments in the declaration
		bool primaryEmitted = false;
		bool declarationRewritten = false;
		std::set<std::string> instantiations;                  // the template arguments handled so far
		std::map<std::string, std::string> membersBySignature; // user functions -> selector members, to share kernels
	};

	std::map<unsigned, TemplateSite> TemplateSites;

	// The function template instantiation that contains the declaration, if it is one that is handled this way
	const FunctionDecl *enclosingTemplateInstantiation(const VarDecl *d)
	{
		const FunctionDecl *f = dyn_cast<FunctionDecl>(d->getDeclContext());
		if (!f || f->getTemplatedKind() != FunctionDecl::TK_FunctionTemplateSpecialization)
			return nullptr;

		// An explicit specialization has a source text of its own
		if (f->getTemplateSpecializationKind() == TSK_ExplicitSpecialization)
			return nullptr;

		// Only templates at namespace scope: the generated code is placed before the template
		if (!f->getPrimaryTemplate()->getDeclContext()->getRedeclContext()->isFileContext())
			return nullptr;

		return f;
	}

	// The template parameter list as a declaration (without defaults) and as a list of arguments
	bool describeTemplateParameters(const FunctionTemplateDecl *T, const PrintingPolicy &policy, std::string &header, std::string &uses)
	{
		header.clear();
		uses.clear();
		for (const NamedDecl *param : *T->getTemplateParameters())
		{
			if (param->getNameAsString().empty())
				return false;

			std::string decl, name = param->getNameAsString(), pack;
			if (auto *type = dyn_cast<TemplateTypeParmDecl>(param))
			{
				pack = type->isParameterPack() ? "..." : "";
				decl = "typename" + pack + " " + name;
			}
			else if (auto *value = dyn_cast<NonTypeTemplateParmDecl>(param))
			{
				pack = value->isParameterPack() ? "..." : "";
				decl = value->getType().getAsString(policy) + pack + " " + name;
			}
			else
				return false; // template template parameters

			header += (header.empty() ? "" : ", ") + decl;
			uses += (uses.empty() ? "" : ", ") + name + pack;
		}
		return true;
	}

	// The template arguments of an instantiation, as the arguments of a specialization
	std::string printTemplateArguments(const FunctionDecl *f, const PrintingPolicy &policy)
	{
		std::string result;
		std::function<void(const TemplateArgument &)> add = [&](const TemplateArgument &arg)
		{
			if (arg.getKind() == TemplateArgument::Pack)
			{
				for (const TemplateArgument &element : arg.pack_elements())
					add(element);
				return;
			}
			std::string text;
			llvm::raw_string_ostream os(text);
			arg.print(policy, os, true);
			result += (result.empty() ? "" : ", ") + os.str();
		};
		for (const TemplateArgument &arg : f->getTemplateSpecializationArgs()->asArray())
			add(arg);
		return result;
	}
}

bool transformSkeletonInvocation(const Skeleton &skeleton, std::string InstanceName, std::vector<UserFunction*> FuncArgs, std::vector<size_t> arity, VarDecl *d, const InlineInstance *inlineInstance)
{
	// d is the declaration of a skeleton instance, or null for an anonymous instance (inlineInstance, a skeleton that is
	// created in the expression that calls it): it is then declared (static) before the enclosing function, and the factory
	// call is replaced by the instance
	std::string signature;
	for (UserFunction *UF : FuncArgs)
		signature += UF->uniqueName + ";";

	// The declaration in the pattern of a function template itself (not dependent on the template parameters, so
	// the visitor sees it as a skeleton instance): it is visited after the instantiations, which have rewritten it.
	// A template that is never instantiated is rewritten as it is.
	if (d)
	if (const FunctionDecl *pattern = dyn_cast<FunctionDecl>(d->getDeclContext()))
		if (pattern->getDescribedFunctionTemplate() && TemplateSites.count(d->getSourceRange().getBegin().getRawEncoding()))
		{
			SkePULog() << "Skeleton instance " << InstanceName << " in a function template was rewritten for its instantiations\n";
			return true;
		}

	// Skeleton instance in a function template (see above): the instantiation
	TemplateSite *site = nullptr;
	std::string instantiationArgs, selectorUse;
	if (const FunctionDecl *instantiation = d ? enclosingTemplateInstantiation(d) : nullptr)
	{
		PrintingPolicy policy(instantiation->getASTContext().getLangOpts());
		policy.FullyQualifiedName = true;
		policy.SuppressScope = false;

		unsigned key = d->getSourceRange().getBegin().getRawEncoding();
		site = &TemplateSites[key];
		if (site->selector.empty())
		{
			if (!describeTemplateParameters(instantiation->getPrimaryTemplate(), policy, site->header, site->uses))
				SkePUAbort("Skeleton instance " + InstanceName + " is in a function template with unnamed or template template parameters; not supported");
			site->selector = "skepu_selector_" + std::to_string(TemplateSites.size() - 1);
		}

		instantiationArgs = printTemplateArguments(instantiation, policy);
		selectorUse = site->selector + "<" + site->uses + ">";
		if (!site->instantiations.insert(instantiationArgs).second)
			return true;
		SkePULog() << "Skeleton instance " << InstanceName << " in a function template, instantiation <" << instantiationArgs << ">\n";
	}
	else if (const FunctionDecl *f = d ? dyn_cast<FunctionDecl>(d->getDeclContext()) : nullptr)
	{
		// A member function template (or other template that is not at namespace scope): the declaration is
		// rewritten once, so its instantiations can only differ in things that do not change the kernels
		if (f->getDescribedFunctionTemplate() || (f->getTemplatedKind() == FunctionDecl::TK_FunctionTemplateSpecialization
			&& f->getTemplateSpecializationKind() != TSK_ExplicitSpecialization))
		{
			static std::map<unsigned, std::string> transformedInstances;
			unsigned key = d->getSourceRange().getBegin().getRawEncoding();
			auto seen = transformedInstances.find(key);
			if (seen != transformedInstances.end())
			{
				if (seen->second != signature)
					SkePUAbort("Skeleton instance " + InstanceName + " in a function template that is not at namespace scope is instantiated with different user functions ("
						+ seen->second + " and " + signature + "); only function templates at namespace scope can be instantiated with several user functions");
				SkePULog() << "Skeleton instance " << InstanceName << " in a template was already transformed\n";
				return true;
			}
			transformedInstances[key] = signature;
		}
	}

	// Find location to insert transformed user function code
	const DeclContext *DeclCtx = d ? d->getDeclContext() : inlineInstance->enclosing;
	SourceLocation loc = d ? d->getSourceRange().getBegin() : inlineInstance->enclosing->getSourceRange().getBegin();
	if (isa<FunctionDecl>(DeclCtx))
	{
		const FunctionDecl *f = dyn_cast<FunctionDecl>(DeclCtx);
		if (f->getTemplatedKind() == FunctionDecl::TK_FunctionTemplateSpecialization)
		{
			// If it is a template, we need to place the code before the template begins
			loc = f->getPrimaryTemplate()->getBeginLoc();
		}
		else if (const FunctionTemplateDecl *pattern = f->getDescribedFunctionTemplate())
		{
			loc = pattern->getBeginLoc();
		}
		else
		{
			loc = f->getSourceRange().getBegin();
		}
	}

	// CUDA: the kernels are __global__ functions, which cannot be members of a class: for a skeleton in a member function the code goes
	// before the outermost class (and its template header) instead of inside it
	if (GenCUDA)
		if (const FunctionDecl *f = dyn_cast_or_null<FunctionDecl>(DeclCtx))
			for (const CXXRecordDecl *rd = dyn_cast<CXXRecordDecl>(f->getLexicalDeclContext()); rd; rd = dyn_cast<CXXRecordDecl>(rd->getLexicalDeclContext()))
			{
				if (const CXXRecordDecl *pattern = rd->getTemplateInstantiationPattern())
					rd = pattern;
				loc = rd->getDescribedClassTemplate() ? rd->getDescribedClassTemplate()->getBeginLoc() : rd->getBeginLoc();
			}

	// A function that is defined by a macro (as the test cases of Catch are): the code goes before the macro invocation
	if (loc.isMacroID())
		loc = GlobalRewriter.getSourceMgr().getExpansionLoc(loc);

	// The selector specialization of an instantiation, placed after the code it refers to
	auto emitSelectorSpecialization = [&](const std::string &members)
	{
		std::stringstream SSSelector;
		if (!site->primaryEmitted)
		{
			SSSelector << "template<" << site->header << "> struct " << site->selector << ";\n";
			site->primaryEmitted = true;
		}
		SSSelector << "template<> struct " << site->selector << "<" << instantiationArgs << ">\n{\n" << members << "};\n"
			<< lineDirectiveForSourceLoc(loc) << "\n";
		if (GlobalRewriter.InsertText(loc, SSSelector.str()))
			SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);
	};

	// Same user functions as an earlier instantiation: its kernels are shared
	if (site)
	{
		auto shared = site->membersBySignature.find(signature);
		if (shared != site->membersBySignature.end())
		{
			emitSelectorSpecialization(shared->second);
			return true;
		}
	}

	generatedStructs = {};
	std::stringstream ss;
	ss << "skepu_skel_" << skeletonCounter++;
	std::string skeletonID = ss.str();

	SkePULog() << "Name of skeleton: " << skeleton.name << ", Unique ID: " << skeletonID << "\n";

	bool rewriteDeclaration = !site || !site->declarationRewritten;
	if (rewriteDeclaration && d)
		if (GlobalRewriter.RemoveText(d->getSourceRange()))
			SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);

	for (UserFunction* UF : FuncArgs)
		definitionsByStruct[SkePU_UF_Prefix + skeletonID + InstanceName + "_" + UF->uniqueName] = definitionJSON(*UF, d, inlineInstance);
	for (UserFunction* UF : FuncArgs)
	{
		generateUserFunctionStruct(*UF, skeletonID + InstanceName, loc);
	}


	std::stringstream SSTemplateArgs, SSCallArgs, SSNewDecl;

	// The members of this instantiation's selector specialization; the declaration refers to them
	std::stringstream SSSelectorMembers;

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

	size_t ufIndex = 0;
	for (UserFunction *func : FuncArgs)
	{
		if (!first)
			SSTemplateArgs << ", ";
		std::string ufStruct = SkePU_UF_Prefix + skeletonID + InstanceName + "_" + func->uniqueName;
		if (site)
		{
			SSSelectorMembers << "using uf" << ufIndex << " = " << ufStruct << ";\n";
			SSTemplateArgs << "typename " << selectorUse << "::uf" << ufIndex;
		}
		else
			SSTemplateArgs << ufStruct;
		++ufIndex;
		first = false;
	}

	// An anonymous instance is named after its place in the source
	std::string instanceLabel = InstanceName;
	if (inlineInstance)
	{
		std::string file = inlineInstance->callFile;
		size_t slash = file.find_last_of("/\\");
		instanceLabel = skeleton.name + "@" + (slash == std::string::npos ? file : file.substr(slash + 1)) + ":" + std::to_string(inlineInstance->callLine);
	}
	SSCallArgs << "\"" << instanceLabel << "\", ";

	if (GenCUDA)
	{
		std::string KernelName_CU;
		std::vector<std::string> cuKernels; // the kernel functions that the skeleton is given, in order
		switch (skeleton.type)
		{
		case Skeleton::Type::MapReduce:
			KernelName_CU = createMapReduceKernelProgram_CU(skeletonID, *FuncArgs[0], *FuncArgs[1], arity[0], ResultDir);
			cuKernels = {KernelName_CU, KernelName_CU + "_ReduceOnly"};
			break;

		case Skeleton::Type::Map:
			KernelName_CU = createMapKernelProgram_CU(skeletonID, *FuncArgs[0], arity[0], ResultDir);
			cuKernels = {KernelName_CU};
			break;

		case Skeleton::Type::MapPairs:
			KernelName_CU = createMapPairsKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			cuKernels = {KernelName_CU};
			break;

		case Skeleton::Type::MapPairsReduce:
			KernelName_CU = createMapPairsReduceKernelProgram_CU(skeletonID, *FuncArgs[0], *FuncArgs[1], ResultDir);
			cuKernels = {KernelName_CU};
			break;

		case Skeleton::Type::Reduce1D:
			KernelName_CU = createReduce1DKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			cuKernels = {KernelName_CU};
			break;

		case Skeleton::Type::Reduce2D:
			KernelName_CU = createReduce2DKernelProgram_CU(skeletonID, *FuncArgs[0], *FuncArgs[1], ResultDir);
			cuKernels = {KernelName_CU + "_RowWise", KernelName_CU + "_ColWise"};
			break;

		case Skeleton::Type::Scan:
			KernelName_CU = createScanKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			cuKernels = {KernelName_CU + "_ScanKernel", KernelName_CU + "_ScanUpdate", KernelName_CU + "_ScanAdd"};
			break;

		case Skeleton::Type::MapOverlap1D:
		case Skeleton::Type::MapPool1D:
			KernelName_CU = createMapOverlap1DKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			cuKernels = {KernelName_CU + "_MapOverlapKernel_CU", KernelName_CU + "_MapOverlapKernel_CU_Matrix_Row",
				KernelName_CU + "_MapOverlapKernel_CU_Matrix_Col", KernelName_CU + "_MapOverlapKernel_CU_Matrix_ColMulti"};
			break;

		case Skeleton::Type::MapOverlap2D:
		case Skeleton::Type::MapPool2D:
			KernelName_CU = createMapOverlap2DKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			cuKernels = {KernelName_CU + "_conv_cuda_2D_kernel"};
			break;

		case Skeleton::Type::MapOverlap3D:
		case Skeleton::Type::MapPool3D:
			KernelName_CU = createMapOverlap3DKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			cuKernels = {KernelName_CU + "_conv_cuda_3D_kernel"};
			break;

		case Skeleton::Type::MapOverlap4D:
		case Skeleton::Type::MapPool4D:
			KernelName_CU = createMapOverlap4DKernelProgram_CU(skeletonID, *FuncArgs[0], ResultDir);
			cuKernels = {KernelName_CU + "_conv_cuda_4D_kernel"};
			break;

		}

		for (size_t k = 0; k < cuKernels.size(); ++k)
		{
			std::string kernelType = "decltype(&" + cuKernels[k] + ")", kernelValue = cuKernels[k];
			if (site)
			{
				SSSelectorMembers << "static constexpr auto cu" << k << " = &" << cuKernels[k] << ";\n";
				kernelValue = selectorUse + "::cu" + std::to_string(k);
				kernelType = "decltype(" + kernelValue + ")";
			}
			SSTemplateArgs << ", " << kernelType;
			SSCallArgs << (k > 0 ? ", " : "") << kernelValue;
		}

		// Insert the code at the proper place
	/*	const DeclContext *DeclCtx = d->getDeclContext();
		SourceLocation loc = d->getSourceRange().getBegin();
		if (isa<FunctionDecl>(DeclCtx))
		{
			loc = dyn_cast<FunctionDecl>(DeclCtx)->getSourceRange().getBegin();
		}*/
		if (GlobalRewriter.InsertText(loc, "#include \"" + KernelName_CU + ".cu\"\n" + lineDirectiveForSourceLoc(loc) + "\n"))
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
		case Skeleton::Type::MapPool1D:
			KernelName_CL = createMapOverlap1DKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::MapOverlap2D:
		case Skeleton::Type::MapPool2D:
			KernelName_CL = createMapOverlap2DKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::MapOverlap3D:
		case Skeleton::Type::MapPool3D:
			KernelName_CL = createMapOverlap3DKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		case Skeleton::Type::MapOverlap4D:
		case Skeleton::Type::MapPool4D:
			KernelName_CL = createMapOverlap4DKernelProgram_CL(skeletonID, *FuncArgs[0], ResultDir);
			break;

		}

		if (GlobalRewriter.InsertText(loc, "#include \"" + KernelName_CL + "_cl_source.inl\"\n" + lineDirectiveForSourceLoc(loc) + "\n"))
			SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);

		if (site)
		{
			SSSelectorMembers << "using cl = CLWrapperClass_" << KernelName_CL << ";\n";
			SSTemplateArgs << ", typename " << selectorUse << "::cl";
		}
		else
			SSTemplateArgs << ", CLWrapperClass_" << KernelName_CL;
	}
	else
	{
		SSTemplateArgs << ", void";
	}

	if (site)
	{
		site->membersBySignature[signature] = SSSelectorMembers.str();
		emitSelectorSpecialization(SSSelectorMembers.str());
	}

	if (!rewriteDeclaration)
		return true;
	if (site)
		site->declarationRewritten = true;

	if (inlineInstance)
	{
		// The instance is declared before the function; the factory call is replaced by a reference to it
		SSNewDecl << "static skepu::backend::" << skeleton.name << "<" << SSTemplateArgs.str() << "> " << InstanceName
			<< "(" << SSCallArgs.str() << ");\n" << lineDirectiveForSourceLoc(loc) << "\n";
		if (GlobalRewriter.InsertText(loc, SSNewDecl.str()))
			SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);

		// The source location of the call, as for the call of a skeleton variable
		std::stringstream SSUse;
		SSUse << InstanceName << ".setSourceLoc(\"" << inlineInstance->callFile << "\", \"" << sourceFileHash(inlineInstance->callFile) << "\", " << inlineInstance->callLine << ")";
		// (the call can span lines, which the replacement does not: the line numbers go on from where it ended)
		{
			SourceLocation callEnd = inlineInstance->factoryCall->getSourceRange().getEnd();
			SourceLocation callBegin = inlineInstance->factoryCall->getSourceRange().getBegin();
			SourceManager &SM = GlobalRewriter.getSourceMgr();
			if (!DoNotGenLineDirectives && SM.getSpellingLineNumber(callEnd) > SM.getSpellingLineNumber(callBegin))
				SSUse << "\n" << lineDirectiveForSourceLoc(callEnd) << "\n";
		}
		if (GlobalRewriter.ReplaceText(inlineInstance->factoryCall->getSourceRange(), SSUse.str()))
			SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);
		return true;
	}

	SourceLocation declStart = d->getSourceRange().getBegin();

	// The declaration replaces the original one on its line, up to the semicolon (which is not part of the
	// declaration's range): the rest of the line is kept, and so are the line numbers
	if(d->getStorageClass() == clang::StorageClass::SC_Static)
		SSNewDecl << "static ";
	SSNewDecl
	    << "skepu::backend::" << skeleton.name
		<< "<" << SSTemplateArgs.str()
		<< "> " << InstanceName << "("
		<< SSCallArgs.str() << ")";

	// (a declaration that spans lines is replaced by one line: the line numbers go on from where it ended, with the rest of that line, the semicolon)
	{
		SourceLocation declEnd = d->getSourceRange().getEnd();
		SourceManager &SM = GlobalRewriter.getSourceMgr();
		if (!DoNotGenLineDirectives && SM.getSpellingLineNumber(declEnd) > SM.getSpellingLineNumber(declStart))
			SSNewDecl << "\n" << lineDirectiveForSourceLoc(declEnd) << "\n";
	}

	if (GlobalRewriter.InsertText(declStart, SSNewDecl.str()))
		SkePUAbort("Code gen target source loc not rewritable: instance" + InstanceName);

	return true;
}
