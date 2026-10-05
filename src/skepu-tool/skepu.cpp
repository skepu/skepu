#include "globals.h"
#include "visitor.h"

#ifdef SKEPU_LSP
#include "support/Logger.h"
#include <filesystem>
#include "clang/AST/Type.h"
#endif


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
std::unordered_map<const clang::FunctionDecl*, UserFunction*> UserFunctions;

// User functions, name maps to AST entry and indexed indicator
std::unordered_map<const clang::TypeDecl*, UserType*> UserTypes;

// User functions, name maps to AST entry and indexed indicator
std::unordered_map<const clang::VarDecl*, UserConstant*> UserConstants;

#ifdef SKEPU_LSP
std::vector<Skeleton*> Skeletons;
std::vector<SkeletonInvocation*> SkeletonInvocations;

std::vector<SkepuContainerInvocation*> ContainerInvocations;
std::vector<SkepuContainer*> Containers;


void clearGlobals()
{
	for (auto& pair : UserFunctions)
		delete pair.second;
	UserFunctions.clear();

	for (auto& pair : UserTypes)
		delete pair.second;
	UserTypes.clear();

	for (auto& pair : UserConstants)
		delete pair.second;
	UserConstants.clear();

	for (auto& pair : SkeletonInvocations)
		delete pair;
	SkeletonInvocations.clear();

	for (auto& pair : Skeletons)
		delete pair;
	Skeletons.clear();

	for (auto& pair : ContainerInvocations)
		delete pair;
	ContainerInvocations.clear();
	
	for (auto& pair : Containers)
		delete pair;
	Containers.clear();
}
#endif

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
const std::unordered_map<std::string, Skeleton> SkeletonsLookup =
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
	{"MapPool1D",            {"MapPool1D",          Skeleton::Type::MapPool1D,          1, 4}},
	{"MapPool2D",            {"MapPool2D",          Skeleton::Type::MapPool2D,          1, 1}},
	{"MapPool3D",            {"MapPool3D",          Skeleton::Type::MapPool3D,          1, 1}},
	{"MapPool4D",            {"MapPool4D",          Skeleton::Type::MapPool4D,          1, 1}},
	{"MapPairsImpl",         {"MapPairs",           Skeleton::Type::MapPairs,           1, 1}},
	{"MapPairsReduceImpl",   {"MapPairsReduce",     Skeleton::Type::MapPairsReduce,     2, 1}},
	{"CallImpl",             {"Call",               Skeleton::Type::Call,               1, 1}},
};

std::unordered_map<Skeleton::Type, std::string> SkeletonDescriptions =
{
	{Skeleton::Type::Map, 
  "Data-parallel element-wise application of a function with arbitrary arity."},
	{Skeleton::Type::Reduce1D, 
  "Reduction with 1D and 2D variations."},
	{Skeleton::Type::Reduce2D, 
  "Reduction with 1D and 2D variations."},
	{Skeleton::Type::MapReduce, 
  "Efficient chaining of Map and Reduce."},
	{Skeleton::Type::Scan, 
  "Generalized prefix sum."},
	{Skeleton::Type::MapOverlap1D, 
  "Stencil operation in 1D."},
	{Skeleton::Type::MapOverlap2D, 
  "Stencil operation in 2D."},
	{Skeleton::Type::MapOverlap3D, 
  "Stencil operation in 3D."},
	{Skeleton::Type::MapOverlap4D, 
  "Stencil operation in 4D."},
  	{Skeleton::Type::MapPool1D, 
  "Pool operation in 1D."},
	{Skeleton::Type::MapPool2D, 
  "Pool operation in 2D."},
	{Skeleton::Type::MapPool3D, 
  "Pool operation in 3D."},
	{Skeleton::Type::MapPool4D, 
  "Pool operation in 4D."},
	{Skeleton::Type::MapPairs, 
  "Cartesian product-style computation."},
	{Skeleton::Type::MapPairsReduce, 
  "Efficient chaining of MapPairs and Reduce."},
	{Skeleton::Type::Call, 
  "Generic multi-variant component."},
};



clang::Rewriter GlobalRewriter;
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

// Injects the transformed text of a library file (from its start/end markers) into the main file, at the include
void libraryFileHandlerHelper(bool found, clang::SourceLocation start, clang::SourceLocation end, clang::SourceLocation injectionLoc, std::string name)
{
	if (!found) return;

	clang::SourceManager &SM = GlobalRewriter.getSourceMgr();
	std::string transformedCode = GlobalRewriter.getRewrittenText(clang::SourceRange(start, end));

	clang::FileID file = SM.getFileID(injectionLoc);
	clang::SourceLocation includeLoc = SM.getIncludeLoc(file);
	unsigned includeLine = SM.getSpellingLineNumber(includeLoc);
	includeLoc = SM.translateLineCol(SM.getMainFileID(), includeLine, 0);

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

llvm::raw_ostream& SkePULog()
{
	if (Verbose)
		return llvm::outs();
	else
		return llvm::nulls();
}

// For each source file provided to the tool, a new FrontendAction is created.
class SkePUFrontendAction : public clang::ASTFrontendAction
{
public:

	bool BeginSourceFileAction(clang::CompilerInstance &CI) override
	{
		inputFileName = this->getCurrentFile().str();
		SkePULog() << "** BeginSourceFileAction for: " << inputFileName << "\n";
		return true;
	}

	void EndSourceFileAction() override
	{
		#ifdef SKEPU_LSP          
			for (clang::VarDecl *d : this->SkeletonInstances)
			{
				Skeleton *skeleton = HandleSkeletonInstance(d);
				if(skeleton != nullptr)
					Skeletons.push_back(skeleton);
			}
			for(SkeletonInvocation *Invocation: SkeletonInvocations)
			{
				for(Skeleton *skeleton: Skeletons)
				{
					if(skeleton == nullptr || Invocation == nullptr)
						continue;

					// TODO: Check if some more data points need to be compared such as special names and such
					if(Invocation->retType.isNull())
					{
						//clangd::log("Invocation ret is null");
					}

					if(skeleton->retType.isNull())
					{
						//clangd::log("Skeleton ret is null");
					}
				
					if(skeleton->variableName == Invocation->variableName && skeleton->retType == Invocation->retType)
					{
						skeleton->invocations.push_back(Invocation);
					}
				}
			}
		#else
			clang::SourceManager &SM = GlobalRewriter.getSourceMgr();
			clang::SourceLocation SLStart = SM.getLocForStartOfFile(SM.getMainFileID());

			GlobalRewriter.InsertText(SLStart, "#define SKEPU_PRECOMPILED 1\n");
			if (GenOMP)  GlobalRewriter.InsertText(SLStart, "#define SKEPU_OPENMP 1\n");
			if (GenCL)   GlobalRewriter.InsertText(SLStart, "#define SKEPU_OPENCL 1\n");
			if (GenCUDA) GlobalRewriter.InsertText(SLStart, "#define SKEPU_CUDA 1\n");
			if (GenStarPUMPI) GlobalRewriter.InsertText(SLStart, "#define SKEPU_STARPU_MPI 1\n");
			if (!DoNotGenLineDirectives)
				GlobalRewriter.InsertText(SLStart, lineDirectiveForSourceLoc(SLStart) + "\n");

			SkePULog() << "Found " << this->SkeletonInstances.size() << " skeleton instances\n";

			for (clang::VarDecl *d : this->SkeletonInstances)
				HandleSkeletonInstance(d);

			namespace lm = library_markers;
			libraryFileHandlerHelper(lm::foundBlas, lm::blasBegin, lm::blasEnd, lm::blasBegin, "BLAS");

			libraryFileHandlerHelper(lm::foundML, lm::mlBegin,              lm::mlEnd,              lm::mlBegin, "ML");
			libraryFileHandlerHelper(lm::foundML, lm::dnnSupportBegin,      lm::dnnSupportEnd,      lm::mlBegin, "DNN_SUPPORT");
			libraryFileHandlerHelper(lm::foundML, lm::layerBegin,           lm::layerEnd,           lm::mlBegin, "LAYER");
			libraryFileHandlerHelper(lm::foundML, lm::activationBegin,      lm::activationEnd,      lm::mlBegin, "ACTIVATION");
			libraryFileHandlerHelper(lm::foundML, lm::denseBegin,           lm::denseEnd,           lm::mlBegin, "DENSE");
			libraryFileHandlerHelper(lm::foundML, lm::dropoutBegin,         lm::dropoutEnd,         lm::mlBegin, "DROPOUT");
			libraryFileHandlerHelper(lm::foundML, lm::flattenBegin,         lm::flattenEnd,         lm::mlBegin, "FLATTEN");
			libraryFileHandlerHelper(lm::foundML, lm::conv2dBegin,          lm::conv2dEnd,          lm::mlBegin, "CONV2D");
			libraryFileHandlerHelper(lm::foundML, lm::pool2dBegin,          lm::pool2dEnd,          lm::mlBegin, "POOL2D");
			// after CONV2D and POOL2D: Conv1D/MaxPooling1D derive from them
			libraryFileHandlerHelper(lm::foundML, lm::conv1dBegin,          lm::conv1dEnd,          lm::mlBegin, "CONV1D");
			libraryFileHandlerHelper(lm::foundML, lm::sequentialModelBegin, lm::sequentialModelEnd, lm::mlBegin, "SEQUENTIAL_MODEL");

			if (Verbose) SkePULog() << "** EndSourceFileAction for: " << inputFileName << "\n";

			// Now emit the rewritten buffer.
			std::error_code EC;
			llvm::raw_fd_ostream OutFile(
				mainFileName, EC, llvm::sys::fs::FileAccess::FA_Write);
			GlobalRewriter.getEditBuffer(SM.getMainFileID()).write(OutFile);
			OutFile.close();
		#endif
	}


	std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef file) override
	{
    	SkePULog() << "** Creating AST consumer for: " << file << "\n";
		GlobalRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
		return std::make_unique<SkePUASTConsumer>(&CI.getASTContext(), this->SkeletonInstances);
	}

#ifdef SKEPU_LSP
	const std::unordered_set<clang::VarDecl *> &getResults() const
	{
		return SkeletonInstances;
	}
#endif

private:
	std::unordered_set<clang::VarDecl *> SkeletonInstances;
};

#ifdef SKEPU_LSP
SkepuMatcher::SkepuMatcher(llvm::StringRef File)
: File(File)
{}

bool SkepuMatcher::runMatcher()
{
	std::filesystem::path p(File.str());

	// Not cross-platform, see https://stackoverflow.com/questions/1528298/get-path-of-executable
	std::filesystem::path skepuDir(std::filesystem::canonical("/proc/self/exe"));//.parent_path().parent_path().parent_path().parent_path());
	while (!skepuDir.empty())
	{
		std::filesystem::path skepu_headers = skepuDir / "skepu-headers";
		std::filesystem::path llvm_dir = skepuDir / "llvm";

		if (std::filesystem::is_directory(skepu_headers) &&
			std::filesystem::is_directory(llvm_dir))
				break;
		
		std::filesystem::path parent = skepuDir.parent_path();

		if (skepuDir == parent)
			return false;

		skepuDir = parent;
	}

	std::string headerPath = skepuDir.string() + "/llvm/clang/lib/Headers";// tmpPathSkepuEx + "/../llvm/clang/lib/Headers";
	std::string skepuSrcPath = skepuDir.string() + "/skepu-headers/src"; //tmpPathSkepuEx + "/../skepu-headers/src";
  

	std::string filePath = p.parent_path().string() +"/"+ p.filename().string();
	std::string seperator = "--";
	std::string cXX = "-std=c++20";
	std::string expansion = "-Wno-expansion-to-defined";
	std::string include = "-I ";
	std::string argIncludeSkepu = "-include" + skepuSrcPath + "/skepu";
	std::string argSkepu ="-I" + skepuSrcPath;
		
	clang::tooling::FixedCompilationDatabase Compilations(
		".",
		{
			"-std=c++20",
			"-Wno-expansion-to-defined",
			"-I ",
			headerPath,
			argIncludeSkepu,
			argSkepu
		}
	);

	std::vector<std::string> Sources = {
		filePath
	};

	clang::tooling::ClangTool Tool(Compilations, Sources);

	clearGlobals();

	// This might be inefficient, it could be better to integrate SkePUFrontendAction
	// or the SkePUASTConsumer into 
	Tool.run(clang::tooling::newFrontendActionFactory<SkePUFrontendAction>().get());

	this->_Skeletons = std::move(Skeletons);
	this->_Containers = std::move(Containers);

	for (const auto& pair : SkeletonsLookup)
		this->_SkeletonsStandard.insert(std::make_pair(pair.first, pair.second));

	for(auto&& p: SkeletonsLookup )
	{
		this->_SkeletonDescriptions[p.second.name] = SkeletonDescriptions[p.second.type];
		this->_SkeletonTypes.push_back(p.second.name);
	}

	for(auto& skeleton : this->_Skeletons)
		skeleton->description = "Skeleton of " + skeleton->name + ": " + SkeletonDescriptions.find(skeleton->type)->second;

	for(auto par : UserFunctions)
		_Userfunctions.push_back(std::move(par.second));

	for(auto par : UserConstants)
		_UserConstants.push_back(std::move(par.second));

	for(auto par : UserTypes)
		_UserTypes.push_back(std::move(par.second));

	return true;
}
#else
int main(int argc, const char **argv)
{
    auto expectedParser = clang::tooling::CommonOptionsParser::create(argc, argv, SkePUCategory);

    if (!expectedParser)
    {
        // Fail gracefully for unsupported options
        SkePULog() << "The expected parser for SkePU tool has failed \n";
        llvm::errs() << expectedParser.takeError();
    }


	clang::tooling::CommonOptionsParser& op = expectedParser.get();
	clang::tooling::ClangTool Tool(op.getCompilations(), op.getSourcePathList());

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

	return Tool.run(clang::tooling::newFrontendActionFactory<SkePUFrontendAction>().get());
}
#endif