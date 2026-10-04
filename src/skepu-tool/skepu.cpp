#include "globals.h"
#include "visitor.h"

using namespace clang;

// ------------------------------
// Precompiler options
// ------------------------------


llvm::cl::OptionCategory SkePUCategory("SkePU precompiler options");

llvm::cl::opt<std::string> ResultDir("dir", llvm::cl::desc("Directory of output files"), llvm::cl::cat(SkePUCategory));
llvm::cl::opt<std::string> ResultName("name", llvm::cl::desc("File name of main output file (without extension, e.g., .cpp or .cu)"), llvm::cl::cat(SkePUCategory));

llvm::cl::opt<bool> GenCUDA("cuda",  llvm::cl::desc("Generate CUDA backend"),   llvm::cl::cat(SkePUCategory));
llvm::cl::opt<bool> GenOMP("openmp", llvm::cl::desc("Generate OpenMP backend"), llvm::cl::cat(SkePUCategory));
llvm::cl::opt<bool> GenCL("opencl",  llvm::cl::desc("Generate OpenCL backend"), llvm::cl::cat(SkePUCategory));
llvm::cl::opt<bool> GenStarPUMPI("starpu-mpi",  llvm::cl::desc("Generate StarPU-MPI backend"), llvm::cl::cat(SkePUCategory));

llvm::cl::opt<bool> Verbose("verbose",  llvm::cl::desc("Verbose logging printout"), llvm::cl::cat(SkePUCategory));
llvm::cl::opt<bool> Silent("silent",  llvm::cl::desc("Disable normal printouts"), llvm::cl::cat(SkePUCategory));
llvm::cl::opt<bool> NoAddExtension("override-extension",  llvm::cl::desc("Do not automatically add file extension to output file (good for headers)"), llvm::cl::cat(SkePUCategory));

llvm::cl::opt<bool> DoNotGenLineDirectives("no-preserve-lines", llvm::cl::desc("Do not try to preserve line numbers from source file"),   llvm::cl::cat(SkePUCategory));

llvm::cl::opt<std::string> AllowedFuncNames("fnames", llvm::cl::desc("Function names which are allowed to be called from user functions (separated by space, e.g. -fnames \"conj csqrt\")"), llvm::cl::cat(SkePUCategory));

// Derived
static std::string mainFileName;
std::string inputFileName;


// ------------------------------
// Globals
// ------------------------------

// User functions, name maps to AST entry and indexed indicator
std::unordered_map<const FunctionDecl*, UserFunction*> UserFunctions;

// User functions, name maps to AST entry and indexed indicator
std::unordered_map<const TypeDecl*, UserType*> UserTypes;

// User functions, name maps to AST entry and indexed indicator
std::unordered_map<const VarDecl*, UserConstant*> UserConstants;

//std::unordered_set<std::pair<CXXOperatorCallExpr *, std::string>> skeletonInvocations;

// Explicitly allowed functions to call from user functions
std::unordered_set<std::string> AllowedFunctionNamesCalledInUFs
{
	"exp", "exp2", "exp2f",
	"sqrt", "copysign", "copysignf",
	"abs", "fabs",
	"max", "fmax",
	"min", "fmin",
	"pow",
	"log", "log2", "log10",
	"sin", "sinh", "asin", "asinh",
	"cos", "cosh", "acos", "acosh",
	"tan", "tanh", "atan", "atanh",
	"round", "ceil", "floor",
	"fmod", "trunc",
	"erf",
	"nan", "isnan", "isinf",
	"printf",
};

// Skeleton types lookup from internal SkePU class template name
const std::unordered_map<std::string, Skeleton> Skeletons =
{
	{"MapImpl",              {"Map",                Skeleton::Type::Map,                1, 1}},
	{"Reduce1D",             {"Reduce1D",           Skeleton::Type::Reduce1D,           1, 1}},
	{"Reduce2D",             {"Reduce2D",           Skeleton::Type::Reduce2D,           2, 2}},
	{"MapReduceImpl",        {"MapReduce",          Skeleton::Type::MapReduce,          2, 2}},
	{"ScanImpl",             {"Scan",               Skeleton::Type::Scan,               1, 3}},
	{"MapOverlap1D",         {"MapOverlap1D",       Skeleton::Type::MapOverlap1D,       1, 4}},
	{"MapOverlap2D",         {"MapOverlap2D",       Skeleton::Type::MapOverlap2D,       1, 1}},
	{"MapOverlap3D",         {"MapOverlap3D",       Skeleton::Type::MapOverlap3D,       1, 1}},
	{"MapOverlap4D",         {"MapOverlap4D",       Skeleton::Type::MapOverlap4D,       1, 1}},
	{"MapPool1D",            {"MapOverlap1D",          Skeleton::Type::MapPool1D,          1, 4}},
	{"MapPool2D",            {"MapPool2D",          Skeleton::Type::MapPool2D,          1, 1}},
	{"MapPool3D",            {"MapOverlap3D",          Skeleton::Type::MapPool3D,          1, 1}},
	{"MapPool4D",            {"MapOverlap4D",          Skeleton::Type::MapPool4D,          1, 1}},
	{"MapPairsImpl",         {"MapPairs",           Skeleton::Type::MapPairs,           1, 1}},
	{"MapPairsReduceImpl",   {"MapPairsReduce",     Skeleton::Type::MapPairsReduce,     2, 1}},
	{"CallImpl",             {"Call",               Skeleton::Type::Call,               1, 1}},
};

Rewriter GlobalRewriter;
size_t GlobalSkeletonIndex = 0;

namespace library_markers
{
    bool foundBlas = false;
    clang::SourceLocation blasBegin, blasEnd;
    
    bool foundML = false;
    clang::SourceLocation mlBegin, mlEnd;
    clang::SourceLocation sequentialModelBegin, sequentialModelEnd;
    clang::SourceLocation dnnSupportBegin, dnnSupportEnd;
    clang::SourceLocation activationBegin, activationEnd;
    clang::SourceLocation layerBegin, layerEnd;
    clang::SourceLocation denseBegin, denseEnd;
    clang::SourceLocation dropoutBegin, dropoutEnd;
    clang::SourceLocation flattenBegin, flattenEnd;
    clang::SourceLocation conv1dBegin, conv1dEnd;
    clang::SourceLocation conv2dBegin, conv2dEnd;
    clang::SourceLocation pool2dBegin, pool2dEnd;

}


llvm::raw_ostream& SkePULog()
{
	if (Verbose)
		return llvm::outs();
	else
		return llvm::nulls();
}

void libraryFileHandlerHelper(bool found, SourceLocation start, SourceLocation end, SourceLocation injectionLoc, std::string name)
{
    if (!found) return;

    // Get transformed source code to inject
    SourceManager &SM = GlobalRewriter.getSourceMgr();
	std::string transformedCode = GlobalRewriter.getRewrittenText(SourceRange(start, end));

	// Find injection point
	FileID file = GlobalRewriter.getSourceMgr().getFileID(injectionLoc);
	SourceLocation includeLoc = GlobalRewriter.getSourceMgr().getIncludeLoc(file);
	unsigned includeLine = GlobalRewriter.getSourceMgr().getSpellingLineNumber(includeLoc);
	includeLoc = GlobalRewriter.getSourceMgr().translateLineCol(SM.getMainFileID(), includeLine, 0);

	// Perform injection
	std::string insertion;
	insertion += "\n/* BEGIN " + name + ".HPP INJECTION */\n";
	if (!DoNotGenLineDirectives)
		insertion += lineDirectiveForSourceLoc(start) + "\n//";
	insertion += transformedCode;
	insertion += ";\n/* END " + name + ".HPP INJECTION*/\n";
	if (!DoNotGenLineDirectives)
		insertion += lineDirectiveForSourceLoc(includeLoc) + "\n";

	GlobalRewriter.InsertText(includeLoc, insertion);
}

// For each source file provided to the tool, a new FrontendAction is created.
class SkePUFrontendAction : public ASTFrontendAction
{
public:

	bool BeginSourceFileAction(CompilerInstance &CI) override
	{
		inputFileName = this->getCurrentFile().str();
		if (Verbose) SkePULog() << "** BeginSourceFileAction for: " << inputFileName << "\n";
		return true;
	}

	void EndSourceFileAction() override
	{
		SourceManager &SM = GlobalRewriter.getSourceMgr();
		SourceLocation SLStart = SM.getLocForStartOfFile(SM.getMainFileID());

		GlobalRewriter.InsertText(SLStart, "#define SKEPU_PRECOMPILED 1\n");
		if (GenOMP)  GlobalRewriter.InsertText(SLStart, "#define SKEPU_OPENMP 1\n");
		if (GenCL)   GlobalRewriter.InsertText(SLStart, "#define SKEPU_OPENCL 1\n");
		if (GenCUDA) GlobalRewriter.InsertText(SLStart, "#define SKEPU_CUDA 1\n");
		if (GenStarPUMPI) GlobalRewriter.InsertText(SLStart, "#define SKEPU_STARPU_MPI 1\n");
		if (!DoNotGenLineDirectives)
		    GlobalRewriter.InsertText(SLStart, lineDirectiveForSourceLoc(SLStart) + "\n");

        SkePULog() << "Found " << this->SkeletonInstances.size() << " skeleton instances\n";

		for (VarDecl *d : this->SkeletonInstances)
			HandleSkeletonInstance(d);
		
    
		libraryFileHandlerHelper(library_markers::foundBlas, library_markers::blasBegin, library_markers::blasEnd, library_markers::blasBegin, "BLAS");

		libraryFileHandlerHelper(library_markers::foundML, library_markers::mlBegin,         library_markers::mlEnd, library_markers::mlBegin, "ML");
		libraryFileHandlerHelper(library_markers::foundML, library_markers::dnnSupportBegin, library_markers::dnnSupportEnd, library_markers::mlBegin, "DNN_SUPPORT");
		libraryFileHandlerHelper(library_markers::foundML, library_markers::layerBegin,      library_markers::layerEnd, library_markers::mlBegin, "LAYER");
		libraryFileHandlerHelper(library_markers::foundML, library_markers::activationBegin, library_markers::activationEnd, library_markers::mlBegin, "ACTIVATION");
		libraryFileHandlerHelper(library_markers::foundML, library_markers::denseBegin,      library_markers::denseEnd, library_markers::mlBegin, "DENSE");
		libraryFileHandlerHelper(library_markers::foundML, library_markers::dropoutBegin,    library_markers::dropoutEnd, library_markers::mlBegin, "DROPOUT");
	 	libraryFileHandlerHelper(library_markers::foundML, library_markers::flattenBegin,    library_markers::flattenEnd, library_markers::mlBegin, "FLATTEN");
		libraryFileHandlerHelper(library_markers::foundML, library_markers::conv1dBegin,     library_markers::conv1dEnd, library_markers::mlBegin, "CONV1D");
		libraryFileHandlerHelper(library_markers::foundML, library_markers::conv2dBegin,     library_markers::conv2dEnd, library_markers::mlBegin, "CONV2D");
		libraryFileHandlerHelper(library_markers::foundML, library_markers::pool2dBegin,     library_markers::pool2dEnd, library_markers::mlBegin, "POOL2D");
		libraryFileHandlerHelper(library_markers::foundML, library_markers::sequentialModelBegin, library_markers::sequentialModelEnd, library_markers::mlBegin, "SEQUENTIAL_MODEL");

		if (Verbose) SkePULog() << "** EndSourceFileAction for: " << inputFileName << "\n";

		// Now emit the rewritten buffer.
		std::error_code EC;
		llvm::raw_fd_ostream OutFile(
			mainFileName, EC, llvm::sys::fs::FileAccess::FA_Write);
		GlobalRewriter.getEditBuffer(SM.getMainFileID()).write(OutFile);
		OutFile.close();
	}

	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override
	{
		if (Verbose) SkePULog() << "** Creating AST consumer for: " << file << "\n";
		GlobalRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
		return std::make_unique<SkePUASTConsumer>(&CI.getASTContext(), this->SkeletonInstances);
	}

private:
	std::unordered_set<clang::VarDecl *> SkeletonInstances;
};


int main(int argc, const char **argv)
{
    auto expectedParser = tooling::CommonOptionsParser::create(argc, argv, SkePUCategory);

    if (!expectedParser)
    {
        // Fail gracefully for unsupported options
        SkePULog() << "The expected parser for SkePU tool has failed \n";
        llvm::errs() << expectedParser.takeError();
    }


	tooling::CommonOptionsParser& op = expectedParser.get();
	tooling::ClangTool Tool(op.getCompilations(), op.getSourcePathList());

	if (ResultName == "")
		ResultName = op.getSourcePathList()[0];
	mainFileName = ResultDir + "/" + ResultName + (NoAddExtension ? "" : (GenCUDA ? ".cu" : ".cpp"));

	if (!Silent)
	{
		SkePULog() << "# ======================================= #\n";
		SkePULog() << "~   SkePU source-to-source compiler v3    ~\n";
		SkePULog() << "# --------------------------------------- #\n";
		SkePULog() << "   OpenMP gen:       " << (GenOMP ? "ON" : "OFF") << "\n";
		SkePULog() << "   CUDA gen:         " << (GenCUDA ? "ON" : "OFF") << "\n";
		SkePULog() << "   OpenCL gen:       " << (GenCL ? "ON" : "OFF") << "\n";
		SkePULog() << "   StarPU-MPI gen:   " << (GenStarPUMPI ? "ON" : "OFF") << "\n";
		SkePULog() << "   Main output file: " << mainFileName << "\n";
		SkePULog() << "# ======================================= #\n";
        for (int i = 1; i < argc; ++i)
            SkePULog() << "Argument " << i << ": " << argv[i] << "\n";
	}

	std::istringstream SSNames(AllowedFuncNames);
	std::vector<std::string> Names{std::istream_iterator<std::string>{SSNames}, std::istream_iterator<std::string>{}};
	for (std::string &name : Names)
		AllowedFunctionNamesCalledInUFs.insert(name);

	return Tool.run(tooling::newFrontendActionFactory<SkePUFrontendAction>().get());
}
