#include <algorithm>
#include "globals.h"
#include "visitor.h"

#ifdef SKEPU_LSP
#include "support/Logger.h"
#include <filesystem>
#include "clang/AST/Type.h"
#include "llvm/Support/FileSystem.h"
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
llvm::cl::opt<bool> GenMPI("mpi", llvm::cl::desc("Generate code for the MPI backend (distributed containers over the other backends)"), llvm::cl::cat(SkePUCategory));

llvm::cl::opt<bool> NoTraceRegions("no-trace-regions", llvm::cl::desc("Do not mark the functions of the program as trace regions (SKEPU_TRACE_AUTO_REGION)"), llvm::cl::cat(SkePUCategory));
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
std::map<const clang::FunctionDecl *, ScopeCandidate> ScopeCandidates;

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
	ScopeCandidates.clear();

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
};



clang::Rewriter GlobalRewriter;
size_t GlobalSkeletonIndex = 0;


namespace library_markers
{
	bool foundBlas = false;
	clang::SourceLocation blasBegin, blasEnd;

	bool foundRandomize = false;
	clang::SourceLocation randomizeBegin, randomizeEnd;

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

					// (only the return types are compared here)
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
			if (GenMPI)  GlobalRewriter.InsertText(SLStart, "#define SKEPU_MPI 1\n");
			if (!DoNotGenLineDirectives)
				GlobalRewriter.InsertText(SLStart, lineDirectiveForSourceLoc(SLStart) + "\n");

			SkePULog() << "Found " << this->SkeletonInstances.size() << " skeleton instances\n";

			// In the order of the source, not of the addresses of the declarations (which varies from run to run), with the declarations in
			// the pattern of a function template last: they are rewritten by the instantiations, which have to come first (see HandleSkeletonInstance)
			std::vector<clang::VarDecl *> instances(this->SkeletonInstances.begin(), this->SkeletonInstances.end());
			auto inPattern = [](clang::VarDecl *d)
			{
				const auto *function = clang::dyn_cast<clang::FunctionDecl>(d->getDeclContext());
				return function && function->getDescribedFunctionTemplate();
			};
			std::stable_sort(instances.begin(), instances.end(), [&](clang::VarDecl *a, clang::VarDecl *b)
			{
				if (inPattern(a) != inPattern(b))
					return inPattern(b);
				return SM.isBeforeInTranslationUnit(a->getBeginLoc(), b->getBeginLoc());
			});
			for (clang::VarDecl *d : instances)
				HandleSkeletonInstance(d);

			for (const InlineInstance &instance : InlineInstances)
				HandleInlineInstance(instance);

			// (-no-trace-regions, or -DSKEPU_NO_TRACE_REGIONS in the options of the compiler, for a frontend that has regions of its own: SkePY)
			if (!NoTraceRegions && !getCompilerInstance().getPreprocessor().isMacroDefined("SKEPU_NO_TRACE_REGIONS"))
				PlantAutoRegions();

			namespace lm = library_markers;
			libraryFileHandlerHelper(lm::foundBlas, lm::blasBegin, lm::blasEnd, lm::blasBegin, "BLAS");
			libraryFileHandlerHelper(lm::foundRandomize, lm::randomizeBegin, lm::randomizeEnd, lm::randomizeBegin, "RANDOMIZE");

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
SkepuMatcher::SkepuMatcher(llvm::StringRef File, std::vector<std::string> SystemFlags)
: File(File), SystemFlags(std::move(SystemFlags))
{}

bool SkepuMatcher::runMatcher()
{
	std::filesystem::path p(File.str());

	// The headers are found from the location of this program (clangd), by looking upward for them. Two layouts:
	// a build directory of SkePU (skepu-headers/src/skepu in the repository, and the resource directory of Clang, lib/clang/<version>/include, in the LLVM build)
	// and a package made by packaging/make-package.sh (include/skepu and lib/SkePU/clang-headers). The Clang headers are the resource directory, not
	// llvm/clang/lib/Headers, which lacks the headers that Clang generates.
	static int anchor;
	const std::string executable = llvm::sys::fs::getMainExecutable(nullptr, &anchor);
	if (executable.empty())
		return false;

	std::string headerPath, skepuSrcPath;
	for (std::filesystem::path dir = std::filesystem::path(executable).parent_path(); !dir.empty(); )
	{
		std::error_code ec;
		if (skepuSrcPath.empty())
		{
			if (std::filesystem::is_regular_file(dir / "skepu-headers" / "src" / "skepu", ec))
				skepuSrcPath = (dir / "skepu-headers" / "src").string();
			else if (std::filesystem::is_regular_file(dir / "include" / "skepu", ec))
				skepuSrcPath = (dir / "include").string();
		}
		if (headerPath.empty())
		{
			if (std::filesystem::is_directory(dir / "lib" / "SkePU" / "clang-headers", ec))
				headerPath = (dir / "lib" / "SkePU" / "clang-headers").string();
			else if (std::filesystem::is_directory(dir / "lib" / "clang", ec))
				for (const auto &version : std::filesystem::directory_iterator(dir / "lib" / "clang", ec))
					if (std::filesystem::is_directory(version.path() / "include", ec))
					{
						headerPath = (version.path() / "include").string();
						break;
					}
		}

		std::filesystem::path parent = dir.parent_path();
		if (parent == dir)
			break;
		dir = parent;
	}
	if (headerPath.empty() || skepuSrcPath.empty())
		return false;

	std::string filePath = p.parent_path().string() +"/"+ p.filename().string();
	std::string seperator = "--";
	std::string cXX = "-std=c++20";
	std::string expansion = "-Wno-expansion-to-defined";
	std::string argIncludeSkepu = "-include" + skepuSrcPath + "/skepu";
	std::string argSkepu ="-I" + skepuSrcPath;
		
	std::vector<std::string> flags = {
		"-std=c++20",
		"-Wno-expansion-to-defined",
		"-I" + headerPath, // (Clang's own headers: stdarg.h and others of the system headers need them, and a program that is not next to a lib/clang does not find them itself)
		argIncludeSkepu,
		argSkepu
	};

	// Where the headers of the system are (the standard library: <memory> is in skepu). The editor knows, from the compile command of the file;
	// the matcher does not look for them itself, as a parse that cannot find <memory> stops and finds nothing (an editor started from a desktop
	// does not have the environment of a terminal, SDKROOT for one).
	flags.insert(flags.end(), SystemFlags.begin(), SystemFlags.end());

	clang::tooling::FixedCompilationDatabase Compilations(".", flags);

	std::vector<std::string> Sources = {
		filePath
	};

	clang::tooling::ClangTool Tool(Compilations, Sources);

	clearGlobals();

	// This might be inefficient, it could be better to integrate SkePUFrontendAction
	// or the SkePUASTConsumer into 
	this->_ParseFailed = Tool.run(clang::tooling::newFrontendActionFactory<SkePUFrontendAction>().get()) != 0;

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
		SkePULog() << "   MPI gen:          " << (GenMPI ? "ON" : "OFF") << "\n";
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