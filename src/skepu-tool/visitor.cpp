#include "globals.h"
#include "visitor.h"

#ifndef SKEPU_LSP
#include "code_gen.h"
#endif


std::unordered_set<std::string> SkeletonInstancesString;
std::vector<InlineInstance> InlineInstances;

[[noreturn]] void SkePUAbort(std::string msg)
{
	llvm::errs() << "[SKEPU] INTERNAL FATAL ERROR: " << msg << "\n";
	exit(1);
}

// ------------------------------
// AST visitor
// ------------------------------

UserFunction *HandleUserFunction(clang::FunctionDecl *f)
{
	// Check so that this userfunction has not been treated already
	if (UserFunctions.find(f) != UserFunctions.end())
	{
		SkePULog() << "UF already handled!\n";
		return UserFunctions[f];
	}

	UserFunction *UF = new UserFunction(f);
	UserFunctions[f] = UF;

	return UF;
}


UserFunction *HandleFunctionPointerArg(clang::Expr *ArgExpr)
{
	// Check that the argument is a declaration reference
	if (!(clang::DeclRefExpr::classof(ArgExpr)))
		SkePUAbort("Userfunction argument not a DeclRefExpr");

	// Get the referee delcaration of the argument declaration reference
	clang::ValueDecl *ValDecl = clang::dyn_cast<clang::DeclRefExpr>(ArgExpr)->getDecl();

	// Check that the argument refers to a function declaration
	if (!clang::FunctionDecl::classof(ValDecl))
		SkePUAbort("Userfunction argument not referencing a function declaration");

	// Get the function declaration and function name
	clang::FunctionDecl *UserFunc = clang::dyn_cast<clang::FunctionDecl>(ValDecl);
	std::string FuncName = UserFunc->getNameInfo().getName().getAsString();

	return HandleUserFunction(UserFunc);
}

// The place for the code of a lambda user function in a skeleton instance: before the function that has the instance
static clang::SourceLocation lambdaCodeLocation(clang::VarDecl *d)
{
	if (const clang::FunctionDecl *DeclCtx = clang::dyn_cast<clang::FunctionDecl>(d->getDeclContext()))
		return DeclCtx->getSourceRange().getBegin();
	return d->getSourceRange().getBegin();
}

UserFunction *HandleLambdaArg(clang::Expr *ArgExpr, clang::SourceLocation codeLocation)
{
	clang::Expr *Arg = ArgExpr;

	if (clang::MaterializeTemporaryExpr *MatTempExpr = clang::dyn_cast<clang::MaterializeTemporaryExpr>(Arg))
		Arg = MatTempExpr->getSubExpr();

	clang::LambdaExpr *Lambda = clang::dyn_cast<clang::LambdaExpr>(Arg);
	if (!Lambda)
		SkePUAbort("User function (assumed lambda) argument not a lambda");

	if (Lambda->capture_size() > 0)
		SkePUAbort("User function lambda argument has non-empty capture list");

	clang::CXXMethodDecl *CallOperator = Lambda->getCallOperator();

	UserFunction *UF = new UserFunction(CallOperator, codeLocation);
	UserFunctions[CallOperator] = UF;

	return UF;
}


const Skeleton::Type* DeclIsValidSkeleton(clang::VarDecl *d)
{
	if (clang::isa<clang::ParmVarDecl>(d))
		return nullptr;

	if (d->isThisDeclarationADefinition() != clang::VarDecl::DefinitionKind::Definition)
		return nullptr;

	clang::Expr *InitExpr = d->getInit();
	if (!InitExpr)
		return nullptr;

    InitExpr = InitExpr->IgnoreImplicit();


	if (auto *CleanUpExpr = clang::dyn_cast<clang::ExprWithCleanups>(InitExpr))
		InitExpr = CleanUpExpr->getSubExpr();

	auto *ConstructExpr = clang::dyn_cast<clang::CXXConstructExpr>(InitExpr);

	if (ConstructExpr && ConstructExpr->getConstructionKind() != clang::CXXConstructionKind::Complete)
		return nullptr;

	auto *TempExpr = InitExpr;

	if (auto *MatTempExpr = clang::dyn_cast<clang::MaterializeTemporaryExpr>(TempExpr))
		TempExpr = MatTempExpr->getSubExpr();

	if (auto *BindTempExpr = clang::dyn_cast<clang::CXXBindTemporaryExpr>(TempExpr))
		TempExpr = BindTempExpr->getSubExpr();

	return FactoryCallSkeleton(clang::dyn_cast<clang::CallExpr>(TempExpr));
}

// The skeleton type of a call of a skeleton factory function (skepu::Map<1>(f)), nullptr for other calls
const Skeleton::Type* FactoryCallSkeleton(clang::CallExpr *CExpr)
{
	if (!CExpr)
		return nullptr;

	if (CExpr->getNumArgs() == 0)
		return nullptr;

	const clang::FunctionDecl *Callee = CExpr->getDirectCallee();

    if (!Callee)
        return nullptr;

    const clang::Type *RetType = Callee->getReturnType().getTypePtr();

	if (clang::isa<clang::DecltypeType>(RetType))
		RetType = clang::dyn_cast<clang::DecltypeType>(RetType)->getUnderlyingType().getTypePtr();

	if (!clang::isa<clang::TemplateSpecializationType>(RetType))
		return nullptr;

	const clang::TemplateDecl *Template = RetType->getAs<clang::TemplateSpecializationType>()->getTemplateName().getAsTemplateDecl();
	std::string TypeName = Template->getNameAsString();

	SkePULog() << "Candidate: " << TypeName << "\n";

	if (SkeletonsLookup.find(TypeName) == SkeletonsLookup.end())
		return nullptr;

	SkePULog() << "Accepted " << SkeletonsLookup.at(TypeName).name << "!\n";

	return &SkeletonsLookup.at(TypeName).type;
}

// The arity and the user functions of a call of a skeleton factory function (skepu::Map<1>(f))
static void HandleFactoryCall(clang::CallExpr *CExpr, clang::ASTContext &Context, clang::SourceLocation lambdaLocation,
	std::string &TypeName, std::vector<size_t> &arity, std::vector<UserFunction*> &FuncArgs)
{
	const clang::FunctionDecl *Callee = CExpr->getDirectCallee();
	const clang::Type *RetType = Callee->getReturnType().getTypePtr();

	if (clang::isa<clang::DecltypeType>(RetType))
		RetType = clang::dyn_cast<clang::DecltypeType>(RetType)->getUnderlyingType().getTypePtr();

	const clang::TemplateSpecializationType *Template = RetType->getAs<clang::TemplateSpecializationType>();
	TypeName = Template->getTemplateName().getAsTemplateDecl()->getNameAsString();
	Skeleton::Type skeletonType = SkeletonsLookup.at(TypeName).type;

	arity = { 0, 2 };
	switch (skeletonType)
	{
	case Skeleton::Type::Map:
	case Skeleton::Type::MapReduce:
		assert(Template->getNumArgs() > 0);
		arity[0] = Template->template_arguments()[0].getAsExpr()->EvaluateKnownConstInt(Context).getExtValue();
		break;
	case Skeleton::Type::MapPairs:
	case Skeleton::Type::MapPairsReduce:
		assert(Template->getNumArgs() > 1);
		arity[0] = Template->template_arguments()[0].getAsExpr()->EvaluateKnownConstInt(Context).getExtValue();
		arity[1] = Template->template_arguments()[1].getAsExpr()->EvaluateKnownConstInt(Context).getExtValue();
		break;
	case Skeleton::Type::MapOverlap1D:
	case Skeleton::Type::MapPool1D:
		arity[0] = 1; break;
	case Skeleton::Type::MapOverlap2D:
	case Skeleton::Type::MapPool2D:
		arity[0] = 1; break;
	case Skeleton::Type::MapOverlap3D:
	case Skeleton::Type::MapPool3D:
		arity[0] = 1; break;
	case Skeleton::Type::MapOverlap4D:
	case Skeleton::Type::MapPool4D:
		arity[0] = 1; break;
	default:
		break;
	}

	size_t i = 0;
	for (clang::Expr *expr : CExpr->arguments())
	{
		UserFunction *UF;

		// The argument may be an implcit cast, get the underlying expression
		if (clang::ImplicitCastExpr *ImplExpr = clang::dyn_cast<clang::ImplicitCastExpr>(expr))
			UF = HandleFunctionPointerArg(ImplExpr->IgnoreImpCasts());

		// It can also be an explicit cast, get the underlying expression
		else if (clang::UnaryOperator *UnaryCastExpr = clang::dyn_cast<clang::UnaryOperator>(expr))
			UF = HandleFunctionPointerArg(UnaryCastExpr->getSubExpr());

		// The user function is probably defined as a lambda
		else
			UF = HandleLambdaArg(expr, lambdaLocation);

		FuncArgs.push_back(UF);

		if (skeletonType == Skeleton::Type::MapPairs || skeletonType == Skeleton::Type::MapPairsReduce)
			UF->updateArgLists(arity[0], arity[1]);
		else
			UF->updateArgLists(arity[i++]);
	}
}

#ifdef SKEPU_LSP
Skeleton* 
#else
bool
#endif
HandleSkeletonInstance(clang::VarDecl *d)
{
	if (d->isThisDeclarationADefinition() != clang::VarDecl::DefinitionKind::Definition)
		SkePUAbort("Not a definition");

	clang::Expr *InitExpr = d->getInit();

    if (!InitExpr)
		SkePUAbort("Not an expression");

    InitExpr = InitExpr->IgnoreImplicit();

	if (clang::isa<clang::ExprWithCleanups>(InitExpr))
		InitExpr = clang::dyn_cast<clang::ExprWithCleanups>(InitExpr)->getSubExpr();

	clang::CXXConstructExpr *ConstructExpr = clang::dyn_cast<clang::CXXConstructExpr>(InitExpr);
	if (ConstructExpr && ConstructExpr->getConstructionKind() != clang::CXXConstructionKind::Complete)
		SkePUAbort("Not a complete constructor");

    clang::Expr *TempExpr = InitExpr;

    if (clang::isa<clang::MaterializeTemporaryExpr>(TempExpr))
        TempExpr = clang::dyn_cast<clang::MaterializeTemporaryExpr>(TempExpr)->getSubExpr();

	if (clang::isa<clang::CXXBindTemporaryExpr>(TempExpr))
		TempExpr = clang::dyn_cast<clang::CXXBindTemporaryExpr>(TempExpr)->getSubExpr();

	clang::CallExpr *CExpr = clang::dyn_cast<clang::CallExpr>(TempExpr);
	if (!CExpr)
		SkePUAbort("Not a call expression");

	const clang::FunctionDecl *Callee = CExpr->getDirectCallee();

	std::string InstanceName = d->getNameAsString();
	SkeletonInstancesString.insert(InstanceName);

	std::string TypeName;
	std::vector<size_t> arity;
	std::vector<UserFunction*> FuncArgs;
	HandleFactoryCall(CExpr, d->getASTContext(), lambdaCodeLocation(d), TypeName, arity, FuncArgs);

	#ifdef SKEPU_LSP
		Skeleton *skeleton = new Skeleton{};
		skeleton->type = SkeletonsLookup.at(TypeName).type;
		skeleton->name = SkeletonsLookup.at(TypeName).name;
		skeleton->userfunctionArgAmount = SkeletonsLookup.at(TypeName).userfunctionArgAmount;
		skeleton->deviceKernelAmount = SkeletonsLookup.at(TypeName).deviceKernelAmount;

		skeleton->variableName = d->getNameAsString();
		skeleton->parameters = FuncArgs;
		skeleton->retType = Callee->getReturnType();

		clang::SourceManager &SM = Callee->getASTContext().getSourceManager();
		clang::SourceRange range = d->getSourceRange();

		clang::clangd::Range clangdRange;
		clangdRange.start.line = SM.getSpellingLineNumber(range.getBegin()) - 1;
		clangdRange.start.character = SM.getSpellingColumnNumber(range.getBegin());
		clangdRange.end.line = SM.getSpellingLineNumber(range.getEnd()) - 1;
		clangdRange.end.character = SM.getSpellingColumnNumber(range.getEnd());

		clang::SourceLocation nameLoc = d->getLocation();
		skeleton->namePosition.line = SM.getSpellingLineNumber(nameLoc) - 1;
		skeleton->namePosition.character = SM.getSpellingColumnNumber(nameLoc) - 1;

		skeleton->selectionRange = clangdRange;
		clangdRange.start.character = 0;
		clangdRange.end.character = SM.getSpellingColumnNumber(range.getEnd()) + 1;
		skeleton->range = clangdRange;

		skeleton->location = d->getLocation();
		skeleton->locationString = d->getLocation().printToString(d->getASTContext().getSourceManager());


		skeleton->constructLayout();
		
		return skeleton;
	#else
		return transformSkeletonInvocation(SkeletonsLookup.at(TypeName), InstanceName, FuncArgs, arity, d);
	#endif
}

#ifndef SKEPU_LSP
bool HandleInlineInstance(const InlineInstance &instance)
{
	std::string TypeName;
	std::vector<size_t> arity;
	std::vector<UserFunction*> FuncArgs;
	SkeletonInstancesString.insert(instance.name);

	HandleFactoryCall(instance.factoryCall, instance.enclosing->getASTContext(), instance.enclosing->getSourceRange().getBegin(),
		TypeName, arity, FuncArgs);

	return transformSkeletonInvocation(SkeletonsLookup.at(TypeName), instance.name, FuncArgs, arity, nullptr, &instance);
}
#endif

// Returns nullptr if the user type can be ignored
UserType *HandleUserType(const clang::CXXRecordDecl *t)
{
	if (!t)
		return nullptr;

	std::string name = t->getNameAsString();

	// These types are handled separately
	if (name == "Index1D" || name == "Index2D" || name == "Index3D" || name == "Index4D"
		|| name == "Region1D" || name == "Region2D" || name == "Region3D" || name == "Region4D"
		|| name == "Pool1D" || name == "Pool2D" || name == "Pool3D" || name == "Pool4D"
		|| name == "Vec" || name == "Mat" || name == "Ten3" || name == "Ten4" || name == "MatRow" || name == "MatCol"
		|| name == "complex")
		return nullptr;

	SkePULog() << "Found user type: " << name << "\n";

	// Check if already handled, otherwise construct and add
	if (UserTypes.find(t) == UserTypes.end())
		UserTypes[t] = new UserType(t);

	return UserTypes[t];
}


SkePUASTVisitor::SkePUASTVisitor(clang::ASTContext *ctx, std::unordered_set<clang::VarDecl *> &instanceSet)
: SkeletonInstances(instanceSet), Context(ctx)
{}

bool SkePUASTVisitor::VisitVarDecl(clang::VarDecl *d)
{	
	#ifndef SKEPU_LSP
		std::string typeName = d->getType().getAsString();
		if (typeName.find("skepu::PrecompilerMarker") != std::string::npos)
		{
			std::string varName = d->getNameAsString();
			// Library files (blas.hpp, the DNN library) are transformed with the program that includes them,
			// and their transformed text is injected at the include (see libraryFileHandlerHelper).
			struct Marker { const char *begin, *end; clang::SourceLocation *beginLoc, *endLoc; bool *found; };
			static const Marker markers[] = {
				{"startOfBlasHPP",                 "endOfBlasHPP",                 &library_markers::blasBegin,            &library_markers::blasEnd,            &library_markers::foundBlas},
				{"startOfRandomizeHPP",            "endOfRandomizeHPP",            &library_markers::randomizeBegin,       &library_markers::randomizeEnd,       &library_markers::foundRandomize},
				{"startOf_ML_HPP",                 "endOf_ML_HPP",                 &library_markers::mlBegin,              &library_markers::mlEnd,              &library_markers::foundML},
				{"startOf_SequentialModel_HPP",    "endOf_SequentialModel_HPP",    &library_markers::sequentialModelBegin, &library_markers::sequentialModelEnd, nullptr},
				{"startOf_DNN_Support_HPP",        "endOf_DNN_Support_HPP",        &library_markers::dnnSupportBegin,      &library_markers::dnnSupportEnd,      nullptr},
				{"startOf_Activation_HPP",         "endOf_Activation_HPP",         &library_markers::activationBegin,      &library_markers::activationEnd,      nullptr},
				{"startOf_Layer_HPP",              "endOf_Layer_HPP",              &library_markers::layerBegin,           &library_markers::layerEnd,           nullptr},
				{"startOf_Dense_HPP",              "endOf_Dense_HPP",              &library_markers::denseBegin,           &library_markers::denseEnd,           nullptr},
				{"startOf_Dropout_HPP",            "endOf_Dropout_HPP",            &library_markers::dropoutBegin,         &library_markers::dropoutEnd,         nullptr},
				{"startOf_Flatten_HPP",            "endOf_Flatten_HPP",            &library_markers::flattenBegin,         &library_markers::flattenEnd,         nullptr},
				{"startOf_Conv1D_HPP",             "endOf_Conv1D_HPP",             &library_markers::conv1dBegin,          &library_markers::conv1dEnd,          nullptr},
				{"startOf_Conv2D_HPP",             "endOf_Conv2D_HPP",             &library_markers::conv2dBegin,          &library_markers::conv2dEnd,          nullptr},
				{"startOf_Pool2D_HPP",             "endOf_Pool2D_HPP",             &library_markers::pool2dBegin,          &library_markers::pool2dEnd,          nullptr},
			};
			for (const Marker &m : markers)
			{
				if (varName == m.begin)
				{
					*m.beginLoc = d->getSourceRange().getEnd();
					if (m.found) *m.found = true;
				}
				else if (varName == m.end)
					*m.endLoc = d->getSourceRange().getEnd();
			}
		}
	#endif

	if (DeclIsValidSkeleton(d))
	{
		SkePULog() << "Found instance: " << d->getNameAsString() << "\n";
		SkeletonInstances.insert(d);
	}
	else if (d->hasAttr<clang::SkepuUserConstantAttr>())
	{
		SkePULog() << "Found user constant: " << d->getNameAsString() << "\n";
		// The constant is put into the kernels from its initializer, so it must be one that the compiler knows: constexpr, with an initializer.
		// (A declaration without one, or one that is not constexpr, was accepted: the tool crashed or the kernels got a value that the
		// program may change.) The language server, which shares this visitor, must not end: it does not register the constant.
		if (!(d->isConstexpr() && d->hasDefinition() == clang::VarDecl::Definition))
		{
			#ifndef SKEPU_LSP
				SkePUAbort("The user constant " + d->getNameAsString() + " must be declared constexpr and have an initializer: [[skepu::userconstant]] constexpr T NAME = value;");
			#else
				SkePULog() << "Invalid user constant (it must be constexpr, with an initializer): " << d->getNameAsString() << "\n";
				return RecursiveASTVisitor<SkePUASTVisitor>::VisitVarDecl(d);
			#endif
		}
		UserConstants[d] = new UserConstant(d);
	}

	return RecursiveASTVisitor<SkePUASTVisitor>::VisitVarDecl(d);
}

// Is the type a SkePU container (a vector, matrix or tensor), as written for any element type?
static bool IsContainerType(clang::QualType type)
{
	type = type.getCanonicalType();
	const clang::TemplateDecl *templateDecl = nullptr;
	if (auto *specialization = type->getAs<clang::TemplateSpecializationType>())
		templateDecl = specialization->getTemplateName().getAsTemplateDecl();
	else if (auto *record = type->getAsCXXRecordDecl())
		if (auto *instance = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(record))
			templateDecl = instance->getSpecializedTemplate();
	if (!templateDecl)
		return false;

	const std::string name = templateDecl->getQualifiedNameAsString();
	return name == "skepu::Vector" || name == "skepu::Matrix" || name == "skepu::Tensor3" || name == "skepu::Tensor4";
}

// A container variable is named in the traces by its variable name: a call is added after its declaration, on the
// same line so that the line numbers stay. Only the declarations that are statements of a block are named (not
// the ones in a condition or a loop header, or in a macro or a header file).
bool SkePUASTVisitor::VisitCompoundStmt(clang::CompoundStmt *block)
{
	#ifndef SKEPU_LSP
		clang::SourceManager &SM = this->Context->getSourceManager();
		static std::set<unsigned> labelledStatements;

		for (clang::Stmt *child : block->body())
		{
			auto *s = clang::dyn_cast<clang::DeclStmt>(child);
			if (!s)
				continue;

			const clang::SourceLocation end = s->getEndLoc();
			if (end.isInvalid() || end.isMacroID() || s->getBeginLoc().isMacroID() || !SM.isWrittenInMainFile(end))
				continue;

			std::string calls;
			for (clang::Decl *decl : s->decls())
			{
				auto *d = clang::dyn_cast<clang::VarDecl>(decl);
				if (d && !d->isInvalidDecl() && !d->getType().isConstQualified() && IsContainerType(d->getType()))
					calls += " skepu::trace_label(" + d->getNameAsString() + ", \"" + d->getNameAsString() + "\");";
			}

			// (a block in a function template is visited once, but the same statement can be reached again)
			if (!calls.empty() && labelledStatements.insert(end.getRawEncoding()).second)
			{
				SkePULog() << "Naming containers after their variables:" << calls << "\n";
				GlobalRewriter.InsertTextAfterToken(end, calls);
			}
		}
	#endif
	return true;
}

// The factory call of an anonymous skeleton instance, if the object of a call is one: the factory call itself or
// a chain of settings (member calls that return the skeleton) on it. Otherwise nullptr.
static clang::CallExpr *findInlineFactory(clang::Expr *e)
{
	for (;;)
	{
		e = e->IgnoreImplicit()->IgnoreParens();
		if (auto *member = clang::dyn_cast<clang::CXXMemberCallExpr>(e))
		{
			e = member->getImplicitObjectArgument();
			if (!e)
				return nullptr;
			continue;
		}
		auto *call = clang::dyn_cast<clang::CallExpr>(e);
		return (call && FactoryCallSkeleton(call)) ? call : nullptr;
	}
}

bool SkePUASTVisitor::TraverseDecl(clang::Decl *d)
{
	auto *f = clang::dyn_cast_or_null<clang::FunctionDecl>(d);
	if (f)
		FunctionStack.push_back(f);
	bool result = RecursiveASTVisitor<SkePUASTVisitor>::TraverseDecl(d);
	if (f)
		FunctionStack.pop_back();
	return result;
}

bool SkePUASTVisitor::TraverseLambdaExpr(clang::LambdaExpr *l)
{
	++LambdaDepth;
	bool result = RecursiveASTVisitor<SkePUASTVisitor>::TraverseLambdaExpr(l);
	--LambdaDepth;
	return result;
}

bool SkePUASTVisitor::VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *c)
{
	#ifndef SKEPU_LSP
		if (c->getOperator() == clang::OO_Call)
		{
			// An anonymous skeleton instance: skepu::Map<1>(f)(out, in), or with settings before the call
			if (clang::CallExpr *factory = findInlineFactory(c->getArg(0)))
			{
				clang::SourceManager &SM = this->Context->getSourceManager();
				clang::SourceLocation callLoc = c->getBeginLoc();
				if (callLoc.isMacroID())
					SkePUAbort("A skeleton instance that is created in a macro expansion is not supported (anonymous skeleton instance)");

				const clang::FunctionDecl *f = FunctionStack.empty() ? nullptr : FunctionStack.back();
				if (!f || LambdaDepth > 0 || f->isTemplated() || f->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate
					|| clang::isa<clang::CXXMethodDecl>(f) || !f->getDeclContext()->getRedeclContext()->isFileContext())
					SkePUAbort("An anonymous skeleton instance is supported in a function (not a template, a member function or a lambda) "
						"at namespace scope; declare the skeleton as a variable otherwise: " + callLoc.printToString(SM));

				std::string file = SM.getFilename(SM.getSpellingLoc(callLoc)).str();
				unsigned line = SM.getSpellingLineNumber(callLoc);

				InlineInstance instance;
				instance.factoryCall = factory;
				instance.enclosing = f;
				instance.name = "skepu_inline_" + std::to_string(InlineInstances.size());
				instance.callFile = file;
				instance.callLine = line;
				InlineInstances.push_back(instance);
				SkePULog() << "Found anonymous instance " << instance.name << " at " << file << ":" << line << "\n";

				return RecursiveASTVisitor<SkePUASTVisitor>::VisitCallExpr(c);
			}

			clang::Expr *arg =	c->getArg(0);
			std::string type = arg->getType().getDesugaredType(*this->Context).getAsString();

			for (auto skeleton : SkeletonsLookup)
			{
				std::string skeletonName = std::get<0>(skeleton);
				if (type.find(skeletonName) != std::string::npos)
				{
					SkePULog() << "Found skeleton call " << type << "\n";

					if (clang::isa<clang::ImplicitCastExpr>(arg))
						arg = clang::dyn_cast<clang::ImplicitCastExpr>(arg)->getSubExpr();

					clang::ValueDecl *valuedecl = nullptr;
					if (clang::isa<clang::DeclRefExpr>(arg))
						valuedecl = clang::dyn_cast<clang::DeclRefExpr>(arg)->getDecl();

					clang::VarDecl *vardecl = nullptr;
					if (valuedecl && clang::isa<clang::VarDecl>(valuedecl))
						vardecl = clang::dyn_cast<clang::VarDecl>(valuedecl);

					// The skeleton is not called through a variable, e.g. through a
					// data member of a wrapper class (as in code generated by SkePY).
					// There is no instance to rewrite here, so leave the call alone.
					if (!vardecl)
					{
						SkePULog() << "Skipping skeleton call that is not made through a variable\n";
						continue;
					}

					std::string instance_name = vardecl->getNameAsString();
					SkePULog() << "Calling instance of name: " << instance_name << "\n";

					// (a call in a function template is visited once per instantiation)
					static std::set<unsigned> rewrittenCalls;
					if (!rewrittenCalls.insert(c->getSourceRange().getBegin().getRawEncoding()).second)
						continue;

					std::string file_name = this->Context->getSourceManager().getFilename(c->getSourceRange().getBegin()).str();
					unsigned line_nr = this->Context->getSourceManager().getSpellingLineNumber(c->getSourceRange().getBegin());

					// (the instance as the call names it, with its qualifier if it is in a namespace)
					std::string written_name = clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(arg->getSourceRange()), this->Context->getSourceManager(), this->Context->getLangOpts()).str();
					if (written_name.empty() || written_name.find_first_of("()[]{} ,") != std::string::npos)
						written_name = instance_name;

					std::stringstream ss;
					ss << "(" << written_name << ".setSourceLoc(\"" << file_name << "\", \"" << sourceFileHash(file_name) << "\", " << line_nr << ") , ";

					GlobalRewriter.InsertText(c->getSourceRange().getBegin(), ss.str());
					GlobalRewriter.InsertTextBefore(c->getSourceRange().getEnd(), ")");
				}
			}
		}

		auto callee = c->getCallee();

		if (clang::ImplicitCastExpr *ImplExpr = clang::dyn_cast<clang::ImplicitCastExpr>(callee))
			callee = ImplExpr->IgnoreImpCasts();
	#endif
	return RecursiveASTVisitor<SkePUASTVisitor>::VisitCallExpr(c);
}

#ifdef SKEPU_LSP
bool SkePUASTVisitor::VisitCallExpr(clang::CallExpr *ce){
	clang::SourceLocation startLoc = ce->getBeginLoc();
	clang::SourceManager &sourceManager = Context->getSourceManager();
	if (sourceManager.isMacroBodyExpansion(startLoc)) {
		// If it's a macro expansion, get the spelling location
		startLoc = sourceManager.getSpellingLoc(startLoc);
	}

	if (clang::FunctionDecl *FD = ce->getDirectCallee()) {
        // Get variable that was called
		if (ce->getNumArgs() <= 0) {
			return true;
		}
		clang::Expr *argExpr = ce->getArg(0);
		if (clang::DeclRefExpr *DRE = clang::dyn_cast<clang::DeclRefExpr>(argExpr)) {
			if (clang::VarDecl *VD = clang::dyn_cast<clang::VarDecl>(DRE->getDecl())) {
				if(DeclIsValidSkeleton(VD))
				{					
					SkeletonInvocation *skeletonInvocation = new SkeletonInvocation{};

					skeletonInvocation->retType = VD->getType();
					skeletonInvocation->variableName = VD->getNameAsString();

					skeletonInvocation->invloc = FD->getPointOfInstantiation();
					skeletonInvocation->defloc = VD->getLocation();

					skeletonInvocation->invLocationString = FD->getPointOfInstantiation().printToString(Context->getSourceManager()) ;
					skeletonInvocation->defLocationString = VD->getLocation().printToString(Context->getSourceManager());
					
					SkeletonInvocations.push_back(skeletonInvocation);
				}
			}
		}	
	}
	return true;
}
#endif

// Implementation of the ASTConsumer interface for reading an AST produced by the Clang parser.
SkePUASTConsumer::SkePUASTConsumer(clang::ASTContext *ctx, std::unordered_set<clang::VarDecl *> &instanceSet)
: Visitor(ctx, instanceSet)
{}

// Override the method that gets called for each parsed top-level
// declaration.
bool SkePUASTConsumer::HandleTopLevelDecl(clang::DeclGroupRef DR)
{
	for (clang::DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b)
		Visitor.TraverseDecl(*b);
	return true;
}
