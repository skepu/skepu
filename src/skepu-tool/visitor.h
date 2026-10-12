#pragma once

#include "globals.h"

#ifdef SKEPU_LSP
#include "data_structures.h"
#endif

// ------------------------------
// AST visitor
// ------------------------------

UserFunction *HandleUserFunction(clang::FunctionDecl *f);
const Skeleton::Type* FactoryCallSkeleton(clang::CallExpr *CExpr);
bool HandleInlineInstance(const InlineInstance &instance);
UserType *HandleUserType(const clang::CXXRecordDecl *t);

#ifdef SKEPU_LSP
Skeleton* 
#else
bool
#endif
HandleSkeletonInstance(clang::VarDecl *d);


class SkePUASTVisitor : public clang::RecursiveASTVisitor<SkePUASTVisitor>
{
public:

	SkePUASTVisitor(clang::ASTContext *ctx, std::unordered_set<clang::VarDecl *> &instanceSet);

	// The function that is being traversed (for anonymous skeleton instances)
	bool TraverseDecl(clang::Decl *d);
	bool TraverseLambdaExpr(clang::LambdaExpr *l);

	bool VisitVarDecl(clang::VarDecl *d);
	bool VisitCompoundStmt(clang::CompoundStmt *block);
	bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *c);

#ifdef SKEPU_LSP
	bool VisitCallExpr(clang::CallExpr *ce);
#endif

	std::unordered_set<clang::VarDecl *> &SkeletonInstances;

private:
	clang::ASTContext *Context;
	std::vector<const clang::FunctionDecl *> FunctionStack;
	unsigned LambdaDepth = 0;
};

// Implementation of the ASTConsumer interface for reading an AST produced by the Clang parser.
class SkePUASTConsumer : public clang::ASTConsumer
{
public:

	SkePUASTConsumer(clang::ASTContext *ctx, std::unordered_set<clang::VarDecl *> &instanceSet);

	// Override the method that gets called for each parsed top-level declaration.
	bool HandleTopLevelDecl(clang::DeclGroupRef DR) override;

private:
	SkePUASTVisitor Visitor;
};
