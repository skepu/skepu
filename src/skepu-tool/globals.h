#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

#include "clang/Lex/Lexer.h"
#include "clang/Basic/Diagnostic.h"

#include "data_structures.h"


extern llvm::cl::opt<bool> GenCUDA;
extern llvm::cl::opt<bool> GenOMP;
extern llvm::cl::opt<bool> GenCL;
extern llvm::cl::opt<bool> DoNotGenLineDirectives;

extern llvm::cl::opt<std::string> ResultName;
extern llvm::cl::opt<std::string> ResultDir;

extern llvm::cl::opt<bool> Verbose;

extern std::string inputFileName;

// User functions, name maps to AST entry and indexed indicator
extern std::unordered_map<const clang::FunctionDecl*, UserFunction*> UserFunctions;

// User functions, name maps to AST entry and indexed indicator
extern std::unordered_map<const clang::TypeDecl*, UserType*> UserTypes;

// User functions, name maps to AST entry and indexed indicator
extern std::unordered_map<const clang::VarDecl*, UserConstant*> UserConstants;


extern const std::unordered_map<std::string, Skeleton> SkeletonsLookup;
extern std::unordered_map<Skeleton::Type, std::string> SkeletonDescriptions;

extern clang::Rewriter GlobalRewriter;

#ifdef SKEPU_LSP
extern std::vector<Skeleton*> Skeletons;
extern std::vector<SkeletonInvocation*> SkeletonInvocations;

extern std::vector<SkepuContainerInvocation*> ContainerInvocations;
extern std::vector<SkepuContainer*> Containers;
#endif


const std::string SkePU_UF_Prefix {"skepu_userfunction_"};
extern size_t GlobalSkeletonIndex;

[[noreturn]] void SkePUAbort(std::string msg);
llvm::raw_ostream& SkePULog();

inline void replaceTextInString(std::string& text, const std::string &find, const std::string &replace)
{
	std::string::size_type pos = 0;
	while ((pos = text.find(find, pos)) != std::string::npos)
	{
		text.replace(pos, find.length(), replace);
		pos += replace.length();
	}
}

inline std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements)
{
	for(std::pair<std::string, std::string> &element : replacements)
		replaceTextInString(templ, element.first, element.second);
	return templ;
}

inline std::string transformToCXXIdentifier(std::string &in)
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

inline std::string getSourceAsString(clang::SourceRange range)
{
	int rangeSize = GlobalRewriter.getRangeSize(range);
	SkePULog() << "Range size: " << rangeSize << "\n";
	if (rangeSize == -1)
		return "";

	clang::SourceLocation startLoc = range.getBegin();
	const char *strStart = GlobalRewriter.getSourceMgr().getCharacterData(startLoc);

	std::string exprString;
	exprString.assign(strStart, rangeSize);

	return exprString;
}


std::string lineDirectiveForSourceLoc(clang::SourceLocation &loc, int offset = 0);

// Library markers: the extent of library files whose transformed text is injected into the main file
namespace library_markers
{
	extern bool foundBlas;
	extern clang::SourceLocation blasBegin, blasEnd;

	extern bool foundML;
	extern clang::SourceLocation mlBegin, mlEnd;
	extern clang::SourceLocation sequentialModelBegin, sequentialModelEnd;
	extern clang::SourceLocation dnnSupportBegin, dnnSupportEnd;
	extern clang::SourceLocation activationBegin, activationEnd;
	extern clang::SourceLocation layerBegin, layerEnd;
	extern clang::SourceLocation denseBegin, denseEnd;
	extern clang::SourceLocation dropoutBegin, dropoutEnd;
	extern clang::SourceLocation flattenBegin, flattenEnd;
	extern clang::SourceLocation conv1dBegin, conv1dEnd;
	extern clang::SourceLocation conv2dBegin, conv2dEnd;
	extern clang::SourceLocation pool2dBegin, pool2dEnd;
}

#ifdef SKEPU_LSP
class SkepuMatcher
{
public:
    SkepuMatcher(llvm::StringRef File);    

    bool runMatcher();

    std::vector<Skeleton*> _Skeletons;
    std::vector<UserFunction*> _Userfunctions;
    std::vector<UserType*> _UserTypes;
    std::vector<UserConstant*> _UserConstants;
    std::vector<SkepuContainer*> _Containers;
    std::vector<std::string> _SkeletonTypes;
    std::unordered_map<std::string, Skeleton> _SkeletonsStandard;

    std::unordered_map<std::string, std::string> _SkeletonDescriptions;

private:
    llvm::StringRef File;
};
#endif
