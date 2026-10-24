#include <cctype>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

#include "clang/Sema/Sema.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "data_structures.h"
#include "globals.h"
#include "visitor.h"

#ifdef SKEPU_LSP
#include <regex>
#else
#include "code_gen_cl.h"
#endif

using namespace clang;
using namespace clang::ast_matchers;

// Defined in skepu.cpp
extern std::unordered_set<std::string> AllowedFunctionNamesCalledInUFs;


// This visitor traverses a userfunction AST node and finds references to other userfunctions and usertypes.
class UserFunctionVisitor : public RecursiveASTVisitor<UserFunctionVisitor>
{
public:

	// Whether a name is qualified with std:: or ::, as the text says it (std::sin(float) is not always declared in std)
	bool isStdQualifier(const DeclRefExpr *ref)
	{
		const ASTContext &Context = ref->getDecl()->getASTContext();
		const SourceManager &SM = Context.getSourceManager();
		bool invalid = false;
		llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getCharRange(ref->getBeginLoc(), ref->getNameInfo().getBeginLoc()), SM, Context.getLangOpts(), &invalid);
		return !invalid && (text == "std::" || text == "::");
	}

	bool VisitCallExpr(CallExpr *c)
	{
		if (isa<CXXOperatorCallExpr>(c))
			return true;
		
		// The visitor may sometimes visit the same expression twice (unclear why)
		for (auto &ref : this->UFReferences)
		{
			// Ignore repeated visits
			if (ref.first == c)
			{
				return true;
			}
		}
		
		auto callee = c->getCallee();
		
		if (ImplicitCastExpr *ImplExpr = dyn_cast<ImplicitCastExpr>(callee))
			callee = ImplExpr->IgnoreImpCasts();
		

		if (auto *UnresolvedLookup = dyn_cast<UnresolvedLookupExpr>(callee))
		{
			std::string name = UnresolvedLookup->getName().getAsString();
			SkePULog() << "Found unresolved lookup expr " << UnresolvedLookup->getName() <<"\n";

			bool allowed = std::find(AllowedFunctionNamesCalledInUFs.begin(), AllowedFunctionNamesCalledInUFs.end(), name)
				!= AllowedFunctionNamesCalledInUFs.end();

			return allowed;
		}

		FunctionDecl *Func = c->getDirectCallee();
		std::string name = Func->getName().str();

		// A call that is qualified with std:: (or ::) is a call of the standard library function, which OpenCL C does not have:
		// the builtin of the same name is used there (see StdCallReferences)
		if (auto *ref = dyn_cast<DeclRefExpr>(callee))
			if (ref->hasQualifier() && !ref->getBeginLoc().isMacroID() && isStdQualifier(ref)
				&& (Func->getDeclContext()->getRedeclContext()->isStdNamespace() || AllowedFunctionNamesCalledInUFs.count(name)))
			{
				const bool allowed = AllowedFunctionNamesCalledInUFs.count(name) > 0;
				std::string plain = name;
				if (name == "abs" && c->getNumArgs() == 1 && c->getArg(0)->getType()->isFloatingType())
					plain = "fabs";
				StdCallReferences.emplace_back(ref->getSourceRange(), allowed ? plain : "", name);
				return true;
			}

		if (name == "")
		{
			// CXXConversionDecl?
			SkePULog() << "Ignored reference to function without known name.'\n";
		}
		else if (name == "ret")
		{
			SkePULog() << "Ignored reference to special function: '" << name << "'\n";
			ReferencedRets.insert(c);
		}
		else if (name == "get" || name == "getNormalized")
		{
			SkePULog() << "Ignored reference to special function: '" << name << "'\n";
			ReferencedGets.insert(dyn_cast<CXXMemberCallExpr>(c));
		}
		else if (std::find(AllowedFunctionNamesCalledInUFs.begin(), AllowedFunctionNamesCalledInUFs.end(), name) != AllowedFunctionNamesCalledInUFs.end())
		{
			// Called function is explicitly allowed
			SkePULog() << "Ignored reference to whitelisted function: '" << name << "'\n";
		}
		else
		{
			SkePULog() << "Found reference to other userfunction: '" << name << "'\n";
			UserFunction *UF = HandleUserFunction(Func);
			UF->updateArgLists(0);
			UFReferences.emplace_back(c, UF);
			ReferencedUFs.insert(UF);
		}

		return true;
	}

	bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *c)
	{
		auto ooc = c->getOperator();
		if (ooc == OO_Subscript)
			this->containerSubscripts.push_back(c);
		if (ooc== OO_Call /* further check the type? */)
			this->containerCalls.push_back(c);
		else if (ooc == OO_Plus || ooc == OO_Minus || ooc == OO_Star || ooc == OO_Slash
			|| ooc == OO_EqualEqual || ooc == OO_ExclaimEqual
			|| ooc == OO_PlusEqual || ooc == OO_MinusEqual || ooc == OO_StarEqual || ooc == OO_SlashEqual
		)
		{
			// We start handling overloaded complex type operators here
			this->operatorOverloads.push_back(c);
		}
		return true;
	}

	bool VisitTypeLoc(TypeLoc tl)
	{
		// An alias (a typedef or a using) that the body names but that is declared outside the function (not in a system header): the
		// backends copy the body to another scope, where the alias does not exist
		if (auto ttl = tl.getAs<TypedefTypeLoc>())
			if (function && function->hasBody())
			{
				const TypedefNameDecl *alias = ttl.getDecl();
				const SourceManager &SM = alias->getASTContext().getSourceManager();
				const SourceRange body = function->getBody()->getSourceRange();
				const bool inBody = !SM.isBeforeInTranslationUnit(tl.getBeginLoc(), body.getBegin()) && !SM.isBeforeInTranslationUnit(body.getEnd(), tl.getBeginLoc());
				const bool declaredInBody = !SM.isBeforeInTranslationUnit(alias->getLocation(), body.getBegin()) && !SM.isBeforeInTranslationUnit(body.getEnd(), alias->getLocation());
				if (inBody && !declaredInBody && !SM.isInSystemHeader(alias->getLocation()) && !tl.getBeginLoc().isMacroID())
				{
					const QualType underlying = alias->getUnderlyingType().getCanonicalType();
					std::string type = underlying.getAsString();
					if (type.find("skepu::complex") != std::string::npos)
						type = "skepu_complex_float"; // todo fix templated type (as in Param::typeNameOpenCL)
					else if (underlying->getAs<RecordType>())
					{
						replaceTextInString(type, "struct ", "");
						type = transformToCXXIdentifier(type);
					}
					TypeAliases[alias->getNameAsString()] = type;
				}
			}

		// The name of a class or struct as written, with its qualifier and keyword, if there are any
		auto rtl = tl.getAs<RecordTypeLoc>();
		if (!rtl || tl.getBeginLoc().isMacroID() || tl.getEndLoc().isMacroID())
			return true;

		const CXXRecordDecl *decl = dyn_cast<CXXRecordDecl>(rtl.getDecl());
		if (!decl || isa<ClassTemplateSpecializationDecl>(decl))
			return true;

		if (auto *userType = HandleUserType(decl))
		{
			// (the visitor may visit a node more than once)
			const SourceRange range = tl.getSourceRange();
			for (auto &ref : this->UTReferences)
				if (ref.first == range)
					return true;

			ReferencedUTs.insert(userType);
			UTReferences.emplace_back(range, userType);
		}
		return true;
	}

	// A user constant that the body names with a qualifier (a namespace): in OpenCL C the constant is a macro that has its
	// plain name, so the qualified spelling is replaced there (see ConstantReferences)
	bool VisitDeclRefExpr(DeclRefExpr *ref)
	{
		const VarDecl *var = dyn_cast<VarDecl>(ref->getDecl());
		if (var && var->hasAttr<SkepuUserConstantAttr>() && ref->hasQualifier() && !ref->getBeginLoc().isMacroID())
		{
			const SourceRange range = ref->getSourceRange();
			// (the visitor may visit a node more than once)
			for (auto &known : this->ConstantReferences)
				if (known.first == range)
					return true;
			ConstantReferences.emplace_back(range, var->getNameAsString());
		}
		return true;
	}

	bool VisitVarDecl(VarDecl *d)
	{
		if (auto *userType = HandleUserType(d->getType().getTypePtr()->getAsCXXRecordDecl()))
			ReferencedUTs.insert(userType);

		return true;
	}


	// The function that is visited, and the aliases of its body that are declared outside it, with the OpenCL type they stand for
	const FunctionDecl *function = nullptr;
	std::map<std::string, std::string> TypeAliases{};

	std::vector<std::pair<const CallExpr*, UserFunction*>> UFReferences{};
	std::set<UserFunction*> ReferencedUFs{};
	
	std::set<const CallExpr*> ReferencedRets{};
	std::set<CXXMemberCallExpr*> ReferencedGets{};

	// Where the function names a user type: the spelling is replaced in the code of the backends, because
	// the code is copied to another scope (a qualifier like detail:: or a name from the enclosing namespace
	// does not resolve there; in OpenCL C, a qualified name is not an identifier at all).
	std::vector<std::pair<SourceRange, UserType*>> UTReferences{};
	std::set<UserType*> ReferencedUTs{};

	// The qualified names of user constants in the function, with the plain name of the constant
	std::vector<std::pair<SourceRange, std::string>> ConstantReferences{};
	std::vector<std::tuple<SourceRange, std::string, std::string>> StdCallReferences{};

	std::vector<CXXOperatorCallExpr*> containerSubscripts{}, containerCalls{};
	std::vector<CXXOperatorCallExpr*> operatorOverloads{};
};



UserConstant::UserConstant(const VarDecl *v)
: astDeclNode(v)
{
	this->name = v->getNameAsString();
	this->typeName = v->getType().getAsString();
	
	#ifdef SKEPU_LSP
		SourceRange sr = v->getInit()->IgnoreImpCasts()->getSourceRange();
		SourceManager &SM = v->getASTContext().getSourceManager();
		clangd::Range clangdRange;
		clangdRange.start.line = SM.getSpellingLineNumber(sr.getBegin());
		clangdRange.start.character = SM.getSpellingColumnNumber(sr.getBegin());
		clangdRange.end.line = SM.getSpellingLineNumber(sr.getEnd());
		clangdRange.end.character = SM.getSpellingColumnNumber(sr.getEnd());

		if(clangdRange.start.line > 0)
		{
			clangdRange.start.line--;
			clangdRange.end.line--;
		}
		
		this->selectionRange = clangdRange;
	#endif

	SourceRange SRInit = v->getInit()->IgnoreImpCasts()->getSourceRange();
	this->definition = Lexer::getSourceText(CharSourceRange::getTokenRange(SRInit), GlobalRewriter.getSourceMgr(), LangOptions(), 0);
}


UserType::UserType(const CXXRecordDecl *t)
: astDeclNode(t), name(t->getNameAsString()), requiresDoublePrecision(false)
{
	this->type = t->getASTContext().getTypeDeclType((const TypeDecl*)t).getTypePtr();
	
	// Same as Param
	std::string rawTypeName = this->type->getCanonicalTypeInternal().getAsString();
	
	std::string resolvedTypeName = rawTypeName;
	
	// Remove 'struct'
	replaceTextInString(resolvedTypeName, "struct ", "");

	this->typeNameOpenCL = transformToCXXIdentifier(resolvedTypeName);
	// End same as Param

	#ifdef SKEPU_LSP
		SourceRange sr = t->getSourceRange();
		SourceManager &SM = t->getASTContext().getSourceManager();
		clangd::Range clangdRange;
		clangdRange.start.line = SM.getSpellingLineNumber(sr.getBegin());
		clangdRange.start.character = SM.getSpellingColumnNumber(sr.getBegin());
		clangdRange.end.line = SM.getSpellingLineNumber(sr.getEnd());
		clangdRange.end.character = SM.getSpellingColumnNumber(sr.getEnd());

		if(clangdRange.start.line > 0)
		{
			clangdRange.start.line--;
			clangdRange.end.line--;
		}
		
		this->selectionRange = clangdRange;
	#endif
	
	if (const RecordDecl *r = dyn_cast<RecordDecl>(t))
	{
		for (const FieldDecl *f : r->fields())
		{
			std::string fieldName = f->getNameAsString();
			std::string typeName = f->getType().getAsString();
			SkePULog() << "User type '" << this->name << "': found field '" << fieldName << "' of type " << typeName << "\n";

			if (typeName == "double")
				this->requiresDoublePrecision = true;
		}
	}
	#ifndef SKEPU_LSP
		static const std::string RunTimeTypeNameFunc = R"~~~(
		namespace {{NAMESPACE}} { template<> std::string getDataTypeCL<{{QUALIFIED_NAME}}>() { return "struct {{ESCAPED_NAME}}"; } }
		namespace {{NAMESPACE}} { template<> std::string getDataTypeDefCL<{{QUALIFIED_NAME}}>() { return {{TYPE_DEF}}; } }
		)~~~";

		if (GenCL)
		{
			std::string typeNameFunc = RunTimeTypeNameFunc;
			// (in an MPI program the node-level library is skepu_node: these are functions of it)
			replaceTextInString(typeNameFunc, "{{NAMESPACE}}", GenMPI ? "skepu_node" : "skepu");
			replaceTextInString(typeNameFunc, "{{QUALIFIED_NAME}}", resolvedTypeName);
			replaceTextInString(typeNameFunc, "{{ESCAPED_NAME}}", this->typeNameOpenCL);
			replaceTextInString(typeNameFunc, "{{TYPE_DEF}}", "R\"~~~(" + generateUserTypeCode_CL(*this) + ")~~~\"");

			// The specializations are in namespace skepu, so a type in a namespace gets them after the
			// outermost enclosing namespace (and they name the type with its qualifier)
			SourceLocation insertLoc = t->getEndLoc().getLocWithOffset(2);
			for (const DeclContext *dc = t->getDeclContext(); dc; dc = dc->getParent())
				if (const NamespaceDecl *ns = dyn_cast<NamespaceDecl>(dc))
					insertLoc = ns->getRBraceLoc().getLocWithOffset(1);

			// A type that is declared in a header (of the library, or of the program) is not in the file that the tool writes: the
			// specializations go into the main file, after the #include that brought the type in
			clang::SourceManager &typeSM = GlobalRewriter.getSourceMgr();
			const clang::FileID mainFile = typeSM.getMainFileID();
			if (typeSM.getFileID(insertLoc) != mainFile)
			{
				SourceLocation include = insertLoc;
				while (typeSM.getFileID(include) != mainFile)
				{
					SourceLocation outer = typeSM.getIncludeLoc(typeSM.getFileID(include));
					if (outer.isInvalid())
						break;
					include = outer;
				}
				if (typeSM.getFileID(include) != mainFile)
					return;
				const unsigned includeLine = typeSM.getSpellingLineNumber(include);
				insertLoc = typeSM.translateLineCol(mainFile, includeLine + 1, 1);
				typeNameFunc = "\n" + typeNameFunc + "\n";
				if (!DoNotGenLineDirectives)
					typeNameFunc += "#line " + std::to_string(includeLine + 1) + " \"" + typeSM.getBufferName(insertLoc).str() + "\"\n";
			}
			GlobalRewriter.InsertText(insertLoc, typeNameFunc);
		}
	#endif
}

std::string random_string(size_t length)
{
	auto randchar = []() -> char
	{
		const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
		const size_t max_index = sizeof(charset) - 1;
		return charset[ rand() % max_index ];
	};
	std::string str(length, 0);
	std::generate_n(str.begin(), length, randchar);
	return str;
}

UserFunction::Param::Param(const clang::ParmVarDecl *p)
: astDeclNode(p)
{
	this->name = p->getNameAsString();

	if (this->name == "")
		this->name = "unused_" + random_string(10);

	this->type = p->getOriginalType().getTypePtr();
	
	if (auto *referenceType = dyn_cast<ReferenceType>(type))
	{
		this->isReferenceType = true;
		if (isa<RValueReferenceType>(type)) this->isRValueReference = true;
		if (isa<LValueReferenceType>(type)) this->isLValueReference = true;
		type = referenceType->getPointeeType().getTypePtr();
	}
	
//	p->dump();
//	this->type->dump();
	
	this->rawTypeName = p->getOriginalType().getCanonicalType().getAsString();
	
	if (this->rawTypeName == "_Bool")
		this->rawTypeName = "bool";
	
	this->resolvedTypeName = this->rawTypeName;
	
	// Remove 'struct'
	replaceTextInString(this->resolvedTypeName, "struct ", "");

	if (	this->rawTypeName != "unsigned long"
		&&	this->rawTypeName != "unsigned char") // (the other built-in types are made identifiers: unsigned int is unsigned_int)
		this->escapedTypeName = transformToCXXIdentifier(this->resolvedTypeName);
	else
		this->escapedTypeName = this->rawTypeName;
	
	// A user type is named by its record name (UserTypes are looked up by that name)
	if (auto *userType = p->getOriginalType().getTypePtr()->getAs<clang::RecordType>())
		this->rawTypeName = userType->getDecl()->getNameAsString();
	
	SkePULog() << "#================#\n";
	SkePULog() << "Param [Elwise]: " << this->name << " of type " << this->rawTypeName << " resolving to " << this->resolvedTypeName << "\n";
	SkePULog() << "Escaped type name: " << this->escapedTypeName << "\n";
	SkePULog() << "Type name OpenCL: " << this->typeNameOpenCL() << "\n";
	SkePULog() << "#================#\n";
}

bool UserFunction::Param::constructibleFrom(const clang::ParmVarDecl *p)
{
	return !UserFunction::RandomParam::constructibleFrom(p) && !UserFunction::RegionParam::constructibleFrom(p) && !UserFunction::RandomAccessParam::constructibleFrom(p);
}

size_t UserFunction::Param::numKernelArgsCL() const
{
	return 1;
}

std::string UserFunction::Param::templateInstantiationType() const
{
	auto *type = this->astDeclNode->getOriginalType().getTypePtr();
    const auto *templateType = dyn_cast<TemplateSpecializationType>(type);
    return templateType->template_arguments()[0].getAsType().getCanonicalType().getAsString();
}


UserFunction::RandomAccessParam::RandomAccessParam(const ParmVarDecl *p)
: Param(p)
{
	this->fullTypeName = this->resolvedTypeName;
	this->unqualifiedFullTypeName = p->getOriginalType().getUnqualifiedType().getCanonicalType().getAsString();
	QualType underlying = p->getOriginalType();
	std::string qualifier;

	if (underlying.isConstQualified())
	{
		underlying = underlying.getUnqualifiedType();
		qualifier = "const";

		this->accessMode = AccessMode::Read;
		SkePULog() << "Read only access mode\n";
	}
	else
	{
		this->accessMode = AccessMode::ReadWrite;
		SkePULog() << "ReadWrite access mode\n";
	}

	auto *type = underlying.getTypePtr();

	const auto *templateType = dyn_cast<TemplateSpecializationType>(type);
	const clang::TemplateArgument containedTypeArg = templateType->template_arguments()[0];

	std::string templateName = templateType->getTemplateName().getAsTemplateDecl()->getNameAsString();
	this->containedType = containedTypeArg.getAsType().getTypePtr();
	
	
//	this->containedType->dump();
	
	if (auto *typedeft = dyn_cast<TypedefType>(this->containedType))
		this->containedType = typedeft->desugar().getTypePtr();

		
//	this->containedType->dump();
	this->resolvedTypeName = this->containedType->getCanonicalTypeInternal().getAsString();
	SkePULog() << "Candidate type name: '" << this->resolvedTypeName << "'\n";
	replaceTextInString(this->resolvedTypeName, "struct ", "");
	this->rawTypeName = this->resolvedTypeName;
	
	this->escapedTypeName = transformToCXXIdentifier(this->resolvedTypeName);
	
	if (templateName == "SparseMat")
	{
		this->containerType = ContainerType::SparseMatrix;
		this->accessMode = AccessMode::Read; // Override for sparse matrices
		SkePULog() << "Sparse Matrix of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Mat")
	{
		this->containerType = ContainerType::Matrix;
		SkePULog() << "Matrix of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "MatRow")
	{
		this->containerType = ContainerType::MatRow;
		SkePULog() << "Matrix Row of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "MatCol")
	{
		this->containerType = ContainerType::MatCol;
		SkePULog() << "Matrix Column of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Vec")
	{
		this->containerType = ContainerType::Vector;
		SkePULog() << "Vector of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Ten3")
	{
		this->containerType = ContainerType::Tensor3;
		SkePULog() << "Tensor3 of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Ten4")
	{
		this->containerType = ContainerType::Tensor4;
		SkePULog() << "Tensor4 of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Region1D")
	{
		this->containerType = ContainerType::Region1D;
		SkePULog() << "Region1D of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Region2D")
	{
		this->containerType = ContainerType::Region2D;
		SkePULog() << "Region2D of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Region3D")
	{
		this->containerType = ContainerType::Region3D;
		SkePULog() << "Region3D of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Region4D")
	{
		this->containerType = ContainerType::Region4D;
		SkePULog() << "Region4D of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Pool1D")
	{
		this->containerType = ContainerType::Pool1D;
		SkePULog() << "Pool1D of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Pool2D")
	{
		this->containerType = ContainerType::Pool2D;
		SkePULog() << "Pool2D of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Pool3D")
	{
		this->containerType = ContainerType::Pool3D;
		SkePULog() << "Pool3D of " << this->resolvedTypeName << "\n";
	}
	else if (templateName == "Pool4D")
	{
		this->containerType = ContainerType::Pool4D;
		SkePULog() << "Pool4D of " << this->resolvedTypeName << "\n";
	}
	else
		SkePUAbort("Unhandled proxy type: " + templateName);

	SkePULog() << "#----------------#\n";
	SkePULog() << "Param [RandomAccess]: " << this->name << " of type " << this->rawTypeName << " resolving to " << this->resolvedTypeName << " (or fully: " << this->fullTypeName << ")\n";
	SkePULog() << "Escaped type name: " << this->escapedTypeName << "\n";
	SkePULog() << "Type name OpenCL: " << this->typeNameOpenCL() << "\n";
	SkePULog() << "#----------------#\n";
}

bool UserFunction::RandomAccessParam::constructibleFrom(const clang::ParmVarDecl *p)
{
	auto *type = p->getOriginalType().getTypePtr();
	
	if (auto *referenceType = dyn_cast<ReferenceType>(type))
		type = referenceType->getPointeeType().getTypePtr();

	const auto *templateType = dyn_cast<TemplateSpecializationType>(type);
	if (!templateType) return false;

	std::string templateName = templateType->getTemplateName().getAsTemplateDecl()->getNameAsString();
	return (templateName == "SparseMat") || (templateName == "Mat") || (templateName == "Vec")
		|| (templateName == "Ten3")  || (templateName == "Ten4") || (templateName == "MatRow") || (templateName == "MatCol");
}

UserFunction::RegionParam::RegionParam(const ParmVarDecl *p)
: RandomAccessParam(p)
{
	auto *type = p->getOriginalType().getTypePtr();
	
	if (auto *referenceType = dyn_cast<ReferenceType>(type))
		type = referenceType->getPointeeType().getTypePtr();

	const auto *templateType = dyn_cast<TemplateSpecializationType>(type);
	std::string templateName = templateType->getTemplateName().getAsTemplateDecl()->getNameAsString();
	
	this->isPool = (templateName == "Pool1D")   || (templateName == "Pool2D")
	    || (templateName == "Pool3D")   || (templateName == "Pool4D");
}

bool UserFunction::RegionParam::constructibleFrom(const clang::ParmVarDecl *p)
{
    auto *type = p->getOriginalType().getTypePtr();
	
	if (auto *referenceType = dyn_cast<ReferenceType>(type))
		type = referenceType->getPointeeType().getTypePtr();

	const auto *templateType = dyn_cast<TemplateSpecializationType>(type);
	if (!templateType) return false;

	std::string templateName = templateType->getTemplateName().getAsTemplateDecl()->getNameAsString();
	SkePULog() << templateName << "\n";
	return (templateName == "Region1D") || (templateName == "Region2D")
	    || (templateName == "Region3D") || (templateName == "Region4D")
		|| (templateName == "Pool1D")   || (templateName == "Pool2D")
	    || (templateName == "Pool3D")   || (templateName == "Pool4D");
	
	SkePULog() << "#++++++++++++++++#\n";
	SkePULog() << "Param [Region] \n";
	SkePULog() << "#++++++++++++++++#\n";
}

UserFunction::RandomParam::RandomParam(const ParmVarDecl *p)
: Param(p)
{
	if (!this->isReferenceType)
		SkePUAbort("Random parameter must be of reference type.");
	
	auto *type = p->getOriginalType().getTypePtr();
	
	if (auto *referenceType = dyn_cast<ReferenceType>(type))
		type = referenceType->getPointeeType().getTypePtr();
	

	const auto *templateType = dyn_cast<TemplateSpecializationType>(type);
	if (templateType->template_arguments().size() > 0)
		this->randomCount = templateType->template_arguments()[0].getAsExpr()->EvaluateKnownConstInt(p->getASTContext()).getExtValue();
	else
		this->randomCount = 0;
}

bool UserFunction::RandomParam::constructibleFrom(const clang::ParmVarDecl *p)
{
	auto *type = p->getOriginalType().getTypePtr();
	
	if (auto *referenceType = dyn_cast<ReferenceType>(type))
		type = referenceType->getPointeeType().getTypePtr();
		
	const auto *templateType = dyn_cast<TemplateSpecializationType>(type);
	if (!templateType) return false;

	std::string templateName = templateType->getTemplateName().getAsTemplateDecl()->getNameAsString();
	SkePULog() << templateName << "\n";
	return templateName == "Random";
}

std::string UserFunction::RandomAccessParam::TypeNameOpenCL() const
{
	switch (this->containerType)
	{
		case ContainerType::Vector:
			return "skepu_vec_proxy_" + this->innerTypeNameOpenCL();
		case ContainerType::Matrix:
			return "skepu_mat_proxy_" + this->innerTypeNameOpenCL();
		case ContainerType::MatRow:
			return "skepu_matrow_proxy_" + this->innerTypeNameOpenCL();
		case ContainerType::MatCol:
				return "skepu_matcol_proxy_" + this->innerTypeNameOpenCL();
		case ContainerType::Tensor3:
			return "skepu_ten3_proxy_" + this->innerTypeNameOpenCL();
		case ContainerType::Tensor4:
			return "skepu_ten4_proxy_" + this->innerTypeNameOpenCL();
		case ContainerType::SparseMatrix:
			return "skepu_sparse_mat_proxy_" + this->innerTypeNameOpenCL();
		default:
			SkePUAbort("ERROR: TypeNameOpenCL: Invalid switch value");
			return "";
	}
}




std::string UserFunction::Param::typeNameOpenCL() const
{
	if (this->resolvedTypeName.find("skepu::complex") != std::string::npos)
	{
		return "skepu_complex_float"; // todo fix templated type
	}
	
	// A user type is defined in the kernel under its escaped qualified name (see generateUserTypeCode_CL);
	// the qualified C++ name (skepu::filter::RGBPixel) is not an identifier in OpenCL C.
	// (the type of a container argument is a record too, MatRow<signed char>, so that the name tells: a name that has no namespace and is not a template
	// is a type that OpenCL C has, or a user type that is defined under its own name, and is not changed)
	if (this->type && this->type->getAs<clang::RecordType>() && this->resolvedTypeName.find_first_of(":<") != std::string::npos)
		return this->escapedTypeName;
	
	return openCLBuiltinTypeName(this->resolvedTypeName);
}

std::string UserFunction::RandomAccessParam::innerTypeNameOpenCL() const
{
	if (this->resolvedTypeName.find("skepu::complex") != std::string::npos)
	{
		return "skepu_complex_float"; // todo fix templated type
	}
	
	return this->escapedTypeName;
}




std::string UserFunction::RandomAccessParam::TypeNameHost() const
{
	// In an MPI program the name skepu is the MPI backend, which has distributed containers: the OpenCL host code works
	// on the node-level ones
	const std::string ns = GenMPI ? "skepu_node" : "skepu";
	switch (this->containerType)
	{
		case ContainerType::Vector:
			return "std::tuple<" + ns + "::Vector<" + this->resolvedTypeName + "> *, skepu::backend::DeviceMemPointer_CL<" + this->resolvedTypeName + "> *>";
		case ContainerType::Matrix:
		case ContainerType::MatRow:
		case ContainerType::MatCol:
			return "std::tuple<" + ns + "::Matrix<" + this->resolvedTypeName + "> *, skepu::backend::DeviceMemPointer_CL<" + this->resolvedTypeName + "> *>";
		case ContainerType::Tensor3:
			return "std::tuple<" + ns + "::Tensor3<" + this->resolvedTypeName + "> *, skepu::backend::DeviceMemPointer_CL<" + this->resolvedTypeName + "> *>";
		case ContainerType::Tensor4:
			return "std::tuple<" + ns + "::Tensor4<" + this->resolvedTypeName + "> *, skepu::backend::DeviceMemPointer_CL<" + this->resolvedTypeName + "> *>";
		case ContainerType::SparseMatrix:
			return "std::tuple<" + ns + "::SparseMatrix<" + this->resolvedTypeName + "> *, skepu::backend::DeviceMemPointer_CL<" + this->resolvedTypeName + "> *, "
				+ "skepu::backend::DeviceMemPointer_CL<size_t> *, skepu::backend::DeviceMemPointer_CL<size_t> *>";
		default:
			SkePUAbort("ERROR: TypeNameHost: Invalid switch value");
			return "";
	}
}

size_t UserFunction::RandomAccessParam::numKernelArgsCL() const
{
	switch (this->containerType)
	{
		case ContainerType::Vector:
			return 2;
		case ContainerType::MatRow: // (data, cols, the number of the first row)
		case ContainerType::MatCol: // (data, rows, cols)
		case ContainerType::Matrix:
			return 3;
		case ContainerType::Tensor3:
			return 4;
		case ContainerType::Tensor4:
			return 5;
		case ContainerType::SparseMatrix:
			return 4;
		default:
			SkePUAbort("ERROR: numKernelArgsCL: Invalid switch value");
			return 0;
	}
}




UserFunction::TemplateArgument::TemplateArgument(std::string name, std::string rawType, std::string resolvedType, std::string valueType)
: paramName(name), rawTypeName(rawType), resolvedTypeName(resolvedType), valueType(valueType)
{}

std::string UserFunction::TemplateArgument::alias(bool constexprValue) const
{
	if (this->valueType.empty())
		return "using " + this->paramName + " = " + this->rawTypeName + ";\n";
	return std::string(constexprValue ? "constexpr static " : "const ") + this->valueType + " " + this->paramName + " = " + this->rawTypeName + ";\n";
}


UserFunction::UserFunction(CXXMethodDecl *f, SourceLocation codeLocation)
: UserFunction(f)
{
	this->rawName = this->uniqueName = "lambda_uf_" + random_string(10);
	this->codeLocation = codeLocation;
}

UserFunction::UserFunction(clang::FunctionDecl *f)
: astDeclNode(f), requiresDoublePrecision(false)
{
	// Function name
	this->rawName = f->getNameInfo().getName().getAsString();

	// Code location
	this->codeLocation = f->getSourceRange().getEnd().getLocWithOffset(1);

	#ifdef SKEPU_LSP
		SourceManager &SM = f->getASTContext().getSourceManager();

		FullSourceLoc startLoc(f->getSourceRange().getBegin(), SM);
		FullSourceLoc endLoc(f->getSourceRange().getEnd(), SM);

		clangd::Range clangdRange;
		clangdRange.start.line = startLoc.getLineNumber();
		clangdRange.start.character =startLoc.getColumnNumber();
		clangdRange.end.line = endLoc.getLineNumber();
		clangdRange.end.character = endLoc.getColumnNumber();

		this->range = clangdRange;

		clangdRange.start.line = SM.getSpellingLineNumber(f->getNameInfo().getBeginLoc());
		clangdRange.start.character = SM.getSpellingColumnNumber(f->getNameInfo().getBeginLoc());
		clangdRange.end.line = SM.getSpellingLineNumber(f->getNameInfo().getEndLoc());
		clangdRange.end.character = SM.getSpellingColumnNumber(f->getNameInfo().getEndLoc());
		
		this->selectionRange = clangdRange;
	#endif

	std::stringstream SSUniqueName;
	SSUniqueName << this->rawName;

	// Handle userfunction templates
	if (f->getTemplatedKind() == FunctionDecl::TK_FunctionTemplateSpecialization)
	{
		this->fromTemplate = true;
		const FunctionTemplateDecl *t = f->getPrimaryTemplate();
		const TemplateParameterList *TPList = t->getTemplateParameters();
		const TemplateArgumentList *TAList = f->getTemplateSpecializationArgs();

		for (unsigned i = 0; i < TPList->size(); ++i)
		{
			if (TAList->get(i).getKind() == clang::TemplateArgument::ArgKind::Integral)
			{
				// A non-type template parameter of integral type: its value replaces the name in the code
				const clang::TemplateArgument &arg = TAList->get(i);
				const auto *valueParam = dyn_cast<clang::NonTypeTemplateParmDecl>(TPList->getParam(i));
				if (valueParam)
				{
					llvm::APSInt integral = arg.getAsIntegral();
					std::string valueType = arg.getIntegralType().getAsString();
					std::string value;
					if (arg.getIntegralType()->isBooleanType())
						value = integral.getBoolValue() ? "true" : "false";
					else
						value = std::to_string(integral.isSigned() ? integral.getExtValue() : (long long)integral.getZExtValue());
					if (value[0] == '-')
						value = "(" + value + ")";

					// (an unnamed parameter, e.g. a SFINAE condition, has nothing to replace in the code)
					if (!valueParam->getNameAsString().empty())
						this->templateArguments.emplace_back(valueParam->getNameAsString(), value, value, valueType);
					std::string nameValue = value;
					for (char &c : nameValue)
						if (!isalnum((unsigned char)c)) c = '_';
					SSUniqueName << "_" << nameValue;
					SkePULog() << "Template value parameter: " << valueParam->getNameAsString() << " = " << value << " (" << valueType << ")\n";
					continue;
				}
			}

			if (TAList->get(i).getKind() != clang::TemplateArgument::ArgKind::Type)
			{
				SkePULog() << "Note: Ignored non-type template parameter\n";
		//		TPList->getParam(i)->dump();
				continue;
			}
			
			std::string paramName = TPList->getParam(i)->getNameAsString();
			std::string rawArgName = TAList->get(i).getAsType().getAsString();
			std::string resolvedArgName = rawArgName;
			if (auto *userType = TAList->get(i).getAsType().getTypePtr()->getAs<clang::RecordType>())
			{
				if (isa<ClassTemplateSpecializationDecl>(userType->getDecl()))
					rawArgName = userType->getDecl()->getNameAsString();
				else
				{
					// a user type: named by its qualified name in C++ code and by its escaped qualified name in the kernels
					resolvedArgName = TAList->get(i).getAsType().getCanonicalType().getAsString();
					replaceTextInString(resolvedArgName, "struct ", "");
					rawArgName = transformToCXXIdentifier(resolvedArgName);
				}
			}
			
			// Remove 'struct'
			replaceTextInString(resolvedArgName, "struct ", "");
			
			this->templateArguments.emplace_back(paramName, rawArgName, resolvedArgName);
			SSUniqueName << "_" << transformToCXXIdentifier(rawArgName);
			SkePULog() << "Template param: " << paramName << " = " << rawArgName << " resolving to " << resolvedArgName << "\n";
		}
	}

	this->uniqueName = SSUniqueName.str();
	SkePULog() << "### [UF] Created UserFunction object with unique name '" << this->uniqueName << "'\n";

	if (!f->doesThisDeclarationHaveABody())
		SkePUAbort("Fatal error: Did not find a body for function '" + this->rawName + "' called inside user function. If this is a common library function, you can override SkePU transformation check by using the -fnames argument.");
	//	GlobalRewriter.getSourceMgr().getDiagnostics().Report(f->getSourceRange().getEnd(), diag::err_skepu_no_userfunction_body) << this->rawName; // Segfault here

	// Type name as string
	this->rawReturnTypeName = f->getReturnType().getCanonicalType().getAsString();
	
	if (this->rawReturnTypeName == "_Bool")
		this->rawReturnTypeName = "bool";
	
	
	
	// Look for multiple return values (skepu::multiple)
	if (this->rawReturnTypeName.find("skepu::multiple") != std::string::npos || this->rawReturnTypeName.find("::tuple<") != std::string::npos)
	{
		SkePULog() << "  [UF " << this->uniqueName << "] Identified multi-valued return!\n";
		
	//	if (GenCUDA || GenCL)
	//		SkePUAbort("Multi-valued return is not enabled for GPU backends.");
		
		const auto *templateType = f->getReturnType().getTypePtr()->getAs<clang::TemplateSpecializationType>();
		for (const clang::TemplateArgument &arg : templateType->template_arguments())
		{
			// canonical: the types are used outside the scope of the function (in its functor struct and in the kernels)
			std::string argType = arg.getAsType().getCanonicalType().getAsString();
			SkePULog() << "    [UF " << this->uniqueName << "] Multi-return type: " << argType << "\n";
			this->multipleReturnTypes.push_back(argType);
		}
		
		std::string resolvedCompoundTypeName = "std::tuple<";
		for (size_t i = 0; i < this->multipleReturnTypes.size(); ++i)
		{
			resolvedCompoundTypeName += this->multipleReturnTypes[i];
			if (i != multipleReturnTypes.size() - 1) resolvedCompoundTypeName += ",";
		}
		resolvedCompoundTypeName += ">";
		this->resolvedReturnTypeName = resolvedCompoundTypeName;
	}
	else
	{
		this->resolvedReturnTypeName = this->rawReturnTypeName;
		
		if (auto *userType = f->getReturnType().getTypePtr()->getAs<clang::RecordType>())
			this->rawReturnTypeName = userType->getDecl()->getNameAsString();
		
		// Returning a templated type, resolve it
		for (UserFunction::TemplateArgument &arg : this->templateArguments)
			if (this->rawReturnTypeName == arg.paramName)
				this->resolvedReturnTypeName = arg.resolvedTypeName;
	}
	
	// remove 'struct'
	replaceTextInString(this->resolvedReturnTypeName, "struct ", "");
	
	SkePULog() << "  [UF " << this->uniqueName << "] Return type: " << this->rawReturnTypeName << " resolving to " << this->resolvedReturnTypeName << "\n";

	// Argument lists
	auto it = f->param_begin();

	if (f->param_size() > 0)
	{
		std::string name = (*it)->getOriginalType().getAsString();
		this->indexed1D = (name == "skepu::Index1D" || name == "struct skepu::Index1D");
		this->indexed2D = (name == "skepu::Index2D" || name == "struct skepu::Index2D");
		this->indexed3D = (name == "skepu::Index3D" || name == "struct skepu::Index3D");
		this->indexed4D = (name == "skepu::Index4D" || name == "struct skepu::Index4D");
	}

	if (this->indexed1D || this->indexed2D || this->indexed3D || this->indexed4D)
		this->indexParam = new UserFunction::Param(*it++);

	// Referenced functions and types
	UserFunctionVisitor UFVisitor;
	UFVisitor.function = this->astDeclNode;
	UFVisitor.TraverseDecl(this->astDeclNode);
	this->TypeAliases = UFVisitor.TypeAliases;

	this->ReferencedUFs = UFVisitor.ReferencedUFs;
	this->UFReferences = UFVisitor.UFReferences;

	this->ReferencedUTs = UFVisitor.ReferencedUTs;
	this->UTReferences = UFVisitor.UTReferences;
	this->ConstantReferences = UFVisitor.ConstantReferences;
	this->StdCallReferences = UFVisitor.StdCallReferences;
	
	this->ReferencedRets = UFVisitor.ReferencedRets;
	this->ReferencedGets = UFVisitor.ReferencedGets;

	this->containerSubscripts = UFVisitor.containerSubscripts;
	this->containerCalls = UFVisitor.containerCalls;
	
	this->operatorOverloads = UFVisitor.operatorOverloads;
	
	SkePULog() << "| Traversal analysis summary for UF " << this->uniqueName << "\n";
	SkePULog() << "| " << this->ReferencedUFs.size() << " unique referenced UFs\n";
	SkePULog() << "| " << this->UFReferences.size() << " total UF references\n";
	SkePULog() << "| " << this->ReferencedUTs.size() << " unique referenced UTs\n";
	SkePULog() << "| " << this->UTReferences.size() << " total UT references\n";
	SkePULog() << "| " << this->ReferencedRets.size() << " ret() expressions\n";
	SkePULog() << "| " << this->ReferencedGets.size() << " get() expressions (PRNG)\n";
	SkePULog() << "| " << this->containerCalls.size() << " container indexing calls\n";

	// Set requires double precision: from the signature, that is the type of the result (or the results of a skepu::multiple) and of every
	// argument, including the element type of a container or a region (MatRow<double>), from the user types and from the functions that it calls.
	// (Not from the body: a local variable or a cast to double is not seen, and a kernel with one is not built for a device without double precision.)
	auto mentionsDouble = [](const clang::QualType &type)
	{
		const std::string name = type.getCanonicalType().getAsString();
		for (size_t at = name.find("double"); at != std::string::npos; at = name.find("double", at + 1))
		{
			const bool wordBefore = at > 0 && (std::isalnum((unsigned char)name[at - 1]) || name[at - 1] == '_');
			const bool wordAfter = at + 6 < name.size() && (std::isalnum((unsigned char)name[at + 6]) || name[at + 6] == '_');
			if (!wordBefore && !wordAfter)
				return true;
		}
		return false;
	};
	if (this->astDeclNode)
	{
		if (mentionsDouble(this->astDeclNode->getReturnType()))
			this->requiresDoublePrecision = true;
		for (const clang::ParmVarDecl *param : this->astDeclNode->parameters())
			if (mentionsDouble(param->getType()))
				this->requiresDoublePrecision = true;
	}

	for (UserType *UT : this->ReferencedUTs)
		if (UT->requiresDoublePrecision)
			this->requiresDoublePrecision = true;
	for (UserFunction *called : this->ReferencedUFs)
		if (called->requiresDoublePrecision)
			this->requiresDoublePrecision = true;
}

std::set<UserType*> UserFunction::allReferencedUTs() const
{
	std::set<UserType*> types = this->ReferencedUTs;
	std::set<const UserFunction*> visited{this};
	std::vector<const UserFunction*> pending{this};
	while (!pending.empty())
	{
		const UserFunction *func = pending.back();
		pending.pop_back();
		for (UserFunction *called : func->ReferencedUFs)
			if (visited.insert(called).second)
			{
				types.insert(called->ReferencedUTs.begin(), called->ReferencedUTs.end());
				pending.push_back(called);
			}
	}
	return types;
}

std::string UserFunction::funcNameCUDA()
{
	return SkePU_UF_Prefix + this->instanceName + "_" + this->uniqueName + "::CU";
}

std::string UserFunction::returnTypeNameOpenCL()
{
	if (this->resolvedReturnTypeName.find("skepu::complex") != std::string::npos)
	{
		return "skepu_complex_float"; // todo fix templated type
	}
	
	// A user type is defined in the kernel under its escaped qualified name (see generateUserTypeCode_CL)
	if (this->multipleReturnTypes.empty())
	{
		const QualType returnType = this->astDeclNode->getReturnType();
		if (const auto *record = returnType->getAs<clang::RecordType>())
			if (!isa<ClassTemplateSpecializationDecl>(record->getDecl()))
			{
				std::string name = returnType.getCanonicalType().getAsString();
				replaceTextInString(name, "struct ", "");
				return transformToCXXIdentifier(name);
			}
	}

	return openCLBuiltinTypeName(this->rawReturnTypeName);
}

std::string UserFunction::multiReturnTypeNameGPU()
{
	std::string multiReturnType = "skepu_multiple";
		for (std::string &type : this->multipleReturnTypes)
		{
			replaceTextInString(type, "struct ", "");
			multiReturnType += "_" + transformToCXXIdentifier(type);
		}
	return multiReturnType;
}


size_t UserFunction::numKernelArgsCL()
{
	size_t count = std::max<size_t>(1, this->multipleReturnTypes.size());
	
	if (this->randomParam)
		count += 1;
	if (this->regionParam)
		count += 1;

	for (Param &p : this->elwiseParams)
		count += p.numKernelArgsCL();
	for (Param &p : this->anyContainerParams)
		count += p.numKernelArgsCL();
	for (Param &p : this->anyScalarParams)
		count += p.numKernelArgsCL();

	return count;
}

bool UserFunction::refersTo(UserFunction &other)
{
	// True if this is the other user function
	if (this == &other)
		return true;

	// True any of this's directly refered userfunctions refers to other
	for (auto *uf : this->ReferencedUFs)
		if (uf->refersTo(other))
			return true;

	return false;
}

void UserFunction::updateArgLists(size_t arity, size_t Harity)
{
	SkePULog() << "Trying with arity: " << arity << "\n";
	
	this->Varity = arity;
	this->Harity = Harity;

	this->randomParam = nullptr;
	this->regionParam = nullptr;
	this->elwiseParams.clear();
	this->anyContainerParams.clear();
	this->anyScalarParams.clear();

	auto it = this->astDeclNode->param_begin();
	const auto end = this->astDeclNode->param_end();

	if (this->indexed1D || this->indexed2D || this->indexed3D || this->indexed4D)
		it++;
	
	if (it != end && UserFunction::RandomParam::constructibleFrom(*it))
	{
		this->randomParam = new UserFunction::RandomParam(*it++);
		this->randomCount = this->randomParam->randomCount;
	}
	
	auto elwise_end = it + arity + Harity;
	
	if (it != end && UserFunction::RegionParam::constructibleFrom(*it))
		this->regionParam = new UserFunction::RegionParam(*it++);
	
	while (it != end && it != elwise_end && UserFunction::Param::constructibleFrom(*it))
		this->elwiseParams.emplace_back(*it++);

	while (it != end && UserFunction::RandomAccessParam::constructibleFrom(*it))
		this->anyContainerParams.emplace_back(*it++);

	while (it != end)
		this->anyScalarParams.emplace_back(*it++);

	// Find references to user types
	auto scanForType = [&] (const Type *p)
	{
		if (auto *userType = HandleUserType(p->getAsCXXRecordDecl()))
			ReferencedUTs.insert(userType);
	};

	// (a user type that only appears as the return type, e.g. in a compound literal,
	// has no variable or parameter that would refer to it)
	if (this->multipleReturnTypes.empty())
		scanForType(this->astDeclNode->getReturnType().getTypePtr());
	else if (const auto *templateType = this->astDeclNode->getReturnType().getTypePtr()->getAs<clang::TemplateSpecializationType>())
		for (const clang::TemplateArgument &arg : templateType->template_arguments())
			scanForType(arg.getAsType().getTypePtr());

	for (auto &param : this->elwiseParams)
		scanForType(param.type);

	for (auto &param : this->anyContainerParams)
	{
		scanForType(param.containedType);
		if (param.containerType == ContainerType::MatRow || param.containerType == ContainerType::MatCol)
			this->hasMatRowOrCol = true; 
	}

	for (auto &param : this->anyScalarParams)
		scanForType(param.type);

	SkePULog() << "Deduced indexed: " << (this->indexParam ? "yes" : "no") << "\n"
		<< "Found random parameter: " << (this->randomParam ? "yes" : "no") << "\n" 
		<< "Found region parameter: " << ((this->regionParam && !this->regionParam->isPool) ? "yes" : "no") << "\n" 
		<< "Found pool parameter: "   << ((this->regionParam && this->regionParam->isPool)  ? "yes" : "no") << "\n" 
		<< "Deduced elementwise arity: " << this->elwiseParams.size() << "\n"
		<< "Deduced random access arity: " << this->anyContainerParams.size() << "\n"
		<< "Deduced scalar arity: " << this->anyScalarParams.size() << "\n";
}

#ifdef SKEPU_LSP
std::string Skeleton::getDetails() const
{
	std::ostringstream oss;
	oss << this->name << "(";
	
	for(UserFunction* param : this->parameters)
	{
		if (param != this->parameters.front())
            oss << ", ";
        oss << param->resolvedReturnTypeName;
	}

	oss << ")";
	return oss.str();
}

void Skeleton::constructLayout()
{
	std::string IndexND = "";
	if(this->parameters.back()->indexed1D)
	{
		IndexND = "Vector";
	}
	else if(this->parameters.back()->indexed2D)
	{
		IndexND = "Matrix";
	}
	else if(this->parameters.back()->indexed3D)
	{
		IndexND = "Tensor3";
	}else if(this->parameters.back()->indexed4D)
	{
		IndexND = "Tensor4";
	}

	std::string retType = "";
	if(!this->parameters.empty())
	{
		retType = this->parameters.back()->resolvedReturnTypeName;
	}

	std::string tmpParam = retType;
	std::vector<std::string> retTypes;
	if (tmpParam.find("tuple") != std::string::npos) {
		// Multiret
		std::regex pattern("<([^<>]+)>");
		// Iterator for matching
		std::sregex_iterator iter(tmpParam.begin(), tmpParam.end(), pattern);
		std::sregex_iterator end;

		if (iter != end) {
			std::smatch match = *iter;
			std::string keywords = match.str(1); // Extracting the captured group

			// Splitting the keywords based on comma (',') delimiter
			std::regex comma_pattern("\\s*,\\s*");
			std::sregex_token_iterator key_iter(keywords.begin(), keywords.end(), comma_pattern, -1);
			std::sregex_token_iterator key_end;

			// Storing each keyword in a vector and printing
			std::vector<std::string> keyword_vector;
			while (key_iter != key_end) {
				std::string keyword = *key_iter;
				retTypes.push_back(keyword);
				++key_iter;
			}
		}
	}
	else
	{
		retTypes.push_back(tmpParam);
	}

	std::string tmp{};

	if(this->name == "Map" || this->name == "MapPairs" 
	|| this->name == "MapPairsReduce" || this->name == "MapOverlap1D"
	|| this->name == "MapOverlap2D" || this->name == "MapOverlap3D"
	|| this->name == "MapOverlap4D")
	{			
		int counter{0};
		for(auto const& elem : retTypes)
		{
			std::string resvParam{};
			if(IndexND != "")
			{
				resvParam = "Return: skepu::" + IndexND + "<" +  elem + ">& Return" + std::to_string(counter);
			}
			else 
			{
				resvParam = "Return: " + elem + "& Return" + std::to_string(counter) + " | (Can be skepu container)";
			}
			this->resolvedParams.push_back(resvParam) ;
			++counter;
		}
		this->resolvedReturnType = "void";
	}
	else 
	{
		if(!retTypes.empty())
		{
			this->resolvedReturnType = retTypes[0];
		}
		else 
		{
			this->resolvedReturnType = "void";
		}
	}

	if(!this->parameters.empty())
	{
		for(auto const& elem : this->parameters[0]->elwiseParams)
		{
			std::string resvParam{};
			if(IndexND != "")
			{
				resvParam = "skepu::" + IndexND + "<" + elem.resolvedTypeName + "> " + elem.name;
			}
			else 
			{
				resvParam = elem.resolvedTypeName + " "+ elem.name + " | (Can be skepu container of " + elem.resolvedTypeName + " type)";
			}
			this->resolvedParams.push_back(resvParam) ;
			//std::string elemStr = "skepu::" + IndexND + "<" + elem.resolvedTypeName + "> " + elem.name;
			//this->resolvedParams.push_back(elemStr);
		}

		for(auto const& elem : this->parameters[0]->anyContainerParams)
		{
			std::string elemStr ="skepu::"+ toString(elem.containerType)+"<"+elem.resolvedTypeName + "> " + elem.name;
			this->resolvedParams.push_back(elemStr);
		}

		for(auto const& elem : this->parameters[0]->anyScalarParams)
		{
			std::string elemStr = elem.resolvedTypeName + " " + elem.name;
			this->resolvedParams.push_back(elemStr);
		}
	}
}
#endif