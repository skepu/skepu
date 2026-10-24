#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/ParentMapContext.h"
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
extern llvm::cl::opt<bool> GenMPI;
extern llvm::cl::opt<bool> DoNotGenLineDirectives;

extern llvm::cl::opt<std::string> ResultName;
extern llvm::cl::opt<std::string> ResultDir;

extern llvm::cl::opt<bool> NoTraceRegions;
extern llvm::cl::opt<bool> Verbose;

extern std::string inputFileName;

// User functions, name maps to AST entry and indexed indicator
extern std::unordered_map<const clang::FunctionDecl*, UserFunction*> UserFunctions;

// An anonymous skeleton instance: a skeleton that is created by a factory call and invoked in the same expression,
// perhaps with settings in between:  skepu::MapOverlap(f).setOverlap(1)(out, in)
// It is handled as if the factory call were a static instance, declared before the enclosing function,
// that the call expression refers to (the settings are then made at every call, as written).
struct InlineInstance
{
	clang::CallExpr *factoryCall;
	const clang::FunctionDecl *enclosing;
	std::string name;       // the identifier of the generated instance
	std::string label;      // its name at run time and in traces, from the call site
	std::string callFile;
	unsigned callLine;
};
extern std::vector<InlineInstance> InlineInstances;

// A function of the program (defined in the file) that can get a trace region of its own, and what decides if it does: a function
// that makes events (it uses skeletons or containers, or calls a function of SkePU), or calls such a function that is in the file
// (see PlantAutoRegions). A function that is called from a user function is never one.
struct ScopeCandidate
{
	const clang::FunctionDecl *definition;
	bool direct = false;                                 // makes events itself
	std::set<const clang::FunctionDecl *> callees;       // the functions it refers to (normalized, see NormalizedFunction)
};
extern std::map<const clang::FunctionDecl *, ScopeCandidate> ScopeCandidates;

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

// The name of a built-in type in OpenCL C: it has no long long (its long is 64 bits) and no long double
inline std::string openCLBuiltinTypeName(std::string type)
{
	replaceTextInString(type, "long double", "double");       // (as CUDA treats it)
	replaceTextInString(type, "unsigned long long", "unsigned long");
	replaceTextInString(type, "long long", "long");
	return type;
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

	extern bool foundRandomize;
	extern clang::SourceLocation randomizeBegin, randomizeEnd;

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
    /// \p SystemFlags: the flags that say where the headers of the system are for the file (-isysroot, -isystem, -I, ...), as the editor has them
    /// for it. Without them the parse has only what the driver finds from the environment (SDKROOT, say), which an editor may not have.
    SkepuMatcher(llvm::StringRef File, std::vector<std::string> SystemFlags = {});

    bool runMatcher();

    /// Whether the parse of the file reported errors (a header that was not found, say): what it found can be less than what is there
    bool _ParseFailed = false;

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
    std::vector<std::string> SystemFlags;
};
#endif
