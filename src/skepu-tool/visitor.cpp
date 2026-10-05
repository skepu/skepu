#include "globals.h"
#include "visitor.h"

#ifndef SKEPU_LSP
#include "code_gen.h"
#endif


std::unordered_set<std::string> SkeletonInstancesString;

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

UserFunction *HandleLambdaArg(clang::Expr *ArgExpr, clang::VarDecl *d)
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

	UserFunction *UF = new UserFunction(CallOperator, d);
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

	clang::CallExpr *CExpr = clang::dyn_cast<clang::CallExpr>(TempExpr);
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
	const clang::Type *RetType = Callee->getReturnType().getTypePtr();

	if (clang::isa<clang::DecltypeType>(RetType))
		RetType = clang::dyn_cast<clang::DecltypeType>(RetType)->getUnderlyingType().getTypePtr();

	const clang::TemplateSpecializationType *Template = RetType->getAs<clang::TemplateSpecializationType>();
	std::string TypeName = Template->getTemplateName().getAsTemplateDecl()->getNameAsString();
	Skeleton::Type skeletonType = SkeletonsLookup.at(TypeName).type;
	
	std::string InstanceName = d->getNameAsString();
	SkeletonInstancesString.insert(InstanceName);

	std::vector<size_t> arity = { 0, 2 };
	switch (skeletonType)
	{
	case Skeleton::Type::Map:
	case Skeleton::Type::MapReduce:
		assert(Template->getNumArgs() > 0);
		arity[0] = Template->template_arguments()[0].getAsExpr()->EvaluateKnownConstInt(d->getASTContext()).getExtValue();
		break;
	case Skeleton::Type::MapPairs:
	case Skeleton::Type::MapPairsReduce:
		assert(Template->getNumArgs() > 1);
		arity[0] = Template->template_arguments()[0].getAsExpr()->EvaluateKnownConstInt(d->getASTContext()).getExtValue();
		arity[1] = Template->template_arguments()[1].getAsExpr()->EvaluateKnownConstInt(d->getASTContext()).getExtValue();
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

	std::vector<UserFunction*> FuncArgs;
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
			UF = HandleLambdaArg(expr, d);

		FuncArgs.push_back(UF);

		if (skeletonType == Skeleton::Type::MapPairs || skeletonType == Skeleton::Type::MapPairsReduce)
			UF->updateArgLists(arity[0], arity[1]);
		else
			UF->updateArgLists(arity[i++]);
	}


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
		if (!(d->isConstexpr() && d->hasDefinition() == clang::VarDecl::Definition))
			SkePULog() << "Invalid!\n"; // TODO: diagnostic
		UserConstants[d] = new UserConstant(d);
	}

	return RecursiveASTVisitor<SkePUASTVisitor>::VisitVarDecl(d);
}

bool SkePUASTVisitor::VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *c)
{
	#ifndef SKEPU_LSP
		if (c->getOperator() == clang::OO_Call)
		{
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

					std::stringstream ss;
					ss << "(" << instance_name << ".setSourceLoc(\"" << file_name << "\", " << line_nr << ") , ";

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
