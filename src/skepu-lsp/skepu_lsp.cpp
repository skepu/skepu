//===--- skepu_lsp.cpp - The SkePU additions to clangd ----------------------===//
//
// See skepu_lsp.h. Compiled as a part of clangDaemon (with SKEPU_LSP defined, see clangd.patch).
//
//===------------------------------------------------------------------------===//
#include "skepu_lsp.h"

#include "Hover.h"
#include "ParsedAST.h"
#include "SourceCode.h"
#include "data_structures.h"
#include "globals.h"
#include "support/Logger.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace clang {
namespace clangd {

namespace {

// What is known about a file

/// A copy of the parts of what SkepuMatcher found in a file that the features use (the matcher deletes its objects on its next run).
struct SkepuInfo
{
	/// The parse of the matcher reported errors and it found nothing: not known to have nothing, so asked again (a few times)
	bool Failed = false;

	struct Skeleton
	{
		std::string variableName, details, description, resolvedReturnType;
		std::vector<std::string> resolvedParams;
		Range range, selectionRange;
		int nameLine = -1;   // the line of the name of the instance
	};

	struct Named
	{
		std::string name;
		Range selectionRange;
	};

	/// A skeleton of SkePU (not an instance of one): its name, the name of its kind, and the number of user functions
	struct Standard
	{
		std::string name, type;
		size_t userfunctionArgAmount;
	};

	std::vector<Skeleton> Skeletons;                          // the instances in the file
	std::vector<Named> UserFunctions, UserConstants, UserTypes;
	std::vector<std::string> SkeletonTypes;                   // the names of the skeletons
	std::unordered_map<std::string, std::string> SkeletonDescriptions;   // name -> description
	std::unordered_map<std::string, Standard> SkeletonsStandard;          // name -> the skeleton
};

struct FileStamp
{
	llvm::sys::TimePoint<> Time;
	uint64_t Size = 0;
	bool operator==(const FileStamp &O) const { return Time == O.Time && Size == O.Size; }
};

FileStamp stampOf(llvm::StringRef File)
{
	FileStamp Stamp;
	llvm::sys::fs::file_status Status;
	if (!llvm::sys::fs::status(File, Status))
	{
		Stamp.Time = Status.getLastModificationTime();
		Stamp.Size = Status.getSize();
	}
	return Stamp;
}

std::mutex InfoMutex;   // also serializes the matcher, that has global data
struct CacheEntry
{
	FileStamp Stamp;
	std::shared_ptr<const SkepuInfo> Info;
	int Attempts = 0;
};

/// How many times an analysis that failed is made again, before the result of it (nothing) is kept until the file changes
constexpr int MaxAnalysisAttempts = 3;
std::map<std::string, CacheEntry> InfoCache;

/// The flags that say where the headers of the system are for the file that the editor has parsed: the sysroot and the directories of
/// includes, which are made absolute (the matcher does not run in the directory of the compile command).
std::vector<std::string> systemFlagsOf(const Preprocessor &PP)
{
	std::vector<std::string> Flags;
	const HeaderSearchOptions &HSO = PP.getHeaderSearchInfo().getHeaderSearchOpts();
	std::string Cwd;
	if (auto Dir = PP.getFileManager().getVirtualFileSystem().getCurrentWorkingDirectory())
		Cwd = *Dir;

	if (!HSO.Sysroot.empty() && HSO.Sysroot != "/")
	{
		Flags.push_back("-isysroot");
		Flags.push_back(HSO.Sysroot);
	}
	for (const auto &Entry : HSO.UserEntries)
	{
		llvm::SmallString<256> Path(Entry.Path);
		if (!Cwd.empty() && !llvm::sys::path::is_absolute(Path))
			llvm::sys::path::make_absolute(Cwd, Path);
		llvm::sys::path::remove_dots(Path, /*remove_dot_dot=*/true);

		const char *Option = nullptr;
		switch (Entry.Group)
		{
			case frontend::Quoted:    Option = "-iquote"; break;
			case frontend::Angled:    Option = Entry.IsFramework ? "-F" : "-I"; break;
			case frontend::System:
			case frontend::ExternCSystem:
			case frontend::CSystem:
			case frontend::CXXSystem:
			case frontend::ObjCSystem:
			case frontend::ObjCXXSystem: Option = Entry.IsFramework ? "-iframework" : "-isystem"; break;
			case frontend::After:     Option = "-idirafter"; break;
			default: break;
		}
		if (Option)
			Flags.push_back(std::string(Option) + std::string(Path));
	}
	return Flags;
}

std::shared_ptr<const SkepuInfo> analyze(llvm::StringRef File, std::vector<std::string> SystemFlags)
{
	auto Info = std::make_shared<SkepuInfo>();
	SkepuMatcher Matcher{File, std::move(SystemFlags)};
	if (!Matcher.runMatcher())
	{
		Info->Failed = true;
		log("SkePU: the analysis of {0} could not run", File);
		return Info;
	}

	for (const auto *s : Matcher._Skeletons)
	{
		SkepuInfo::Skeleton k;
		k.variableName = s->variableName;
		k.details = s->getDetails();
		k.description = s->description;
		k.resolvedReturnType = s->resolvedReturnType;
		k.resolvedParams = s->resolvedParams;
		k.range = s->range;
		k.selectionRange = s->selectionRange;
		k.nameLine = s->namePosition.line;
		Info->Skeletons.push_back(std::move(k));
	}
	for (const auto *f : Matcher._Userfunctions)
		Info->UserFunctions.push_back({f->rawName, f->selectionRange});
	for (const auto *c : Matcher._UserConstants)
		Info->UserConstants.push_back({c->name, c->selectionRange});
	for (const auto *t : Matcher._UserTypes)
		Info->UserTypes.push_back({t->name, t->selectionRange});
	Info->SkeletonTypes = Matcher._SkeletonTypes;
	Info->SkeletonDescriptions = Matcher._SkeletonDescriptions;
	for (auto &pair : Matcher._SkeletonsStandard)
		Info->SkeletonsStandard[pair.first] = {pair.second.name, pair.second.typeToString(pair.second.type), pair.second.userfunctionArgAmount};

	// The skeletons are ours (they are moved out of the globals of the matcher), but their invocations are deleted by the next
	// clearGlobals() of the matcher, and can be in more than one skeleton: the skeletons must not delete them.
	for (auto *s : Matcher._Skeletons)
	{
		s->invocations.clear();
		delete s;
	}
	Matcher._Skeletons.clear();
	if (Matcher._ParseFailed && Info->Skeletons.empty())
	{
		Info->Failed = true;
		log("SkePU: the parse of {0} had errors and the analysis found no skeletons", File);
	}
	return Info;
}

/// What is known about File (found when it is first asked for, and again when the file has changed on the disk: the matcher reads the file, not the
/// buffer of the editor)
/// \p PP is the preprocessor of the file as the editor parsed it: it says where the headers of the system are. An analysis that failed is made
/// again at the next request, a few times: it is not the same as a file that has nothing.
std::shared_ptr<const SkepuInfo> infoFor(llvm::StringRef File, const Preprocessor &PP)
{
	if (File.empty())
		return nullptr;
	std::lock_guard<std::mutex> Lock(InfoMutex);
	const FileStamp Stamp = stampOf(File);
	CacheEntry &Entry = InfoCache[File.str()];
	const bool Changed = !Entry.Info || !(Entry.Stamp == Stamp);
	if (Changed || (Entry.Info->Failed && Entry.Attempts < MaxAnalysisAttempts))
	{
		Entry.Attempts = Changed ? 1 : Entry.Attempts + 1;
		Entry.Info = analyze(File, systemFlagsOf(PP));
		Entry.Stamp = Stamp;
	}
	return Entry.Info;
}

// Signature help


void remove(std::string& str, char c)
{
	str.erase(std::remove(str.begin(), str.end(), c), str.end());
}

std::vector<std::string> getTemplateParameters(std::string input)
{
	remove(input, ' ');
	std::vector<std::string> parameters;
	size_t curr, next = input.find_first_of('<', 0), end = input.find_last_of('>');
	std::string parameter;
	while (next <= end && end != std::string::npos)
	{
		curr = next + 1;
		next = input.find_first_of(',', curr);
		if (next >= end)
			parameter = input.substr(curr, end - curr);
		else
		{
			parameter = input.substr(curr, next-curr);
			if (parameter.find("<") != std::string::npos)
			{
				next = input.find_first_of('>', curr) + 1;
				parameter = input.substr(curr, next-curr);
			}
		}
		parameters.push_back(parameter);
	}
	return parameters;
}

struct SkepuParam
{
	enum ParamType
	{
		ElWise,
		RandomAccess,
		Scalar,
		VElWise,
		HElWise,
	};

	ParamType type;
	std::string param;

	std::string toString() const
	{
		std::string ret = "param: " + param + ", type: ";
		switch (type) {
			case ParamType::ElWise:
				ret += "ElWise";
				break;
			case ParamType::RandomAccess:
				ret += "RandomAccess";
				break;
			case ParamType::Scalar:
				ret += "Scalar";
				break;
			case ParamType::VElWise:
				ret += "VElWise";
				break;
			case ParamType::HElWise:
				ret += "HElWise";
				break;
		}
		return ret;
	}
};
std::string convertDimToContainerType(const std::string& index)
{
	std::unordered_map<std::string, std::string> indexToContainer{
		{"1D", "Vector"},
		{"2D", "Matrix"},
		{"3D", "Tensor3"},
		{"4D", "Tensor4"}
	};

	for (auto& dim : indexToContainer)
		if (index.find(dim.first) != std::string::npos)
			return "skepu::" + dim.second;
	return "";
}

std::string getContainerType(const std::string& candidateType)
{
	if (candidateType == "")
		return "skepu::container";
	std::array<std::string, 4> dimensions{"Vector", "Matrix", "Tensor3", "Tensor4"};
	for (auto& dim : dimensions)
		if (candidateType.find("skepu::" + dim) != std::string::npos)
			return "skepu::" + dim;
	return "skepu::container";
}

std::string getContainerIteratorType(const std::string& candidateType)
{
	if (candidateType == "")
		return "skepu::containerIterator";
	std::array<std::string, 4> dimensions{"Vector", "Matrix", "Tensor3", "Tensor4"};
	for (auto& dim : dimensions)
		if (candidateType.find("skepu::" + dim + "Iterator") != std::string::npos)
			return "skepu::" + dim + "Iterator";
	return "skepu::containerIterator";
}

void addInputParameters(const std::vector<SkepuParam>& parameters, std::vector<std::string>& output, const std::string& container)
{
	size_t elWiseCounter{0};
	size_t randomAccessCounter{0};
	size_t scalarCounter{0};
	size_t VElWiseCounter{0};
	size_t HElWiseCounter{0};
	for (const SkepuParam& parameter : parameters)
	{
		switch (parameter.type)
		{
			case SkepuParam::ParamType::ElWise:
					output.emplace_back(container + "<" + parameter.param + ">& Input" + std::to_string(elWiseCounter++));
				break;
			case SkepuParam::ParamType::RandomAccess:
				output.emplace_back(parameter.param + " Random Access" + std::to_string(randomAccessCounter++));
				break;
			case SkepuParam::ParamType::Scalar:
				output.emplace_back(parameter.param + " Scalar" + std::to_string(scalarCounter++));
				break;
			case SkepuParam::ParamType::VElWise: // VElWise and HElWise are only used for MapPairs which can only have vectors as input
				output.emplace_back("skepu::Vector<" + parameter.param + ">& Vinput" + std::to_string(VElWiseCounter++));
				break;
			case SkepuParam::ParamType::HElWise:
				output.emplace_back("skepu::Vector<" + parameter.param + ">& Hinput" + std::to_string(HElWiseCounter++));
				break;
		}	
	}
}

void handleMap(const std::string& QualName, std::vector<std::string>& output, const std::string& potentialContainerType)
{
	std::vector<std::string> templateParameters = getTemplateParameters(QualName);
	std::vector<SkepuParam> functionParameters;
	std::string container = "";
	size_t paramCount = std::stoi(templateParameters[0]);
	size_t foundParams{0};
	// First three elements are param count, user-defined param count and return type
	for (size_t index{3}; index < templateParameters.size(); ++index)
	{
		std::string param = templateParameters[index];
		if (param.find("Index") != std::string::npos)
		{
			container = convertDimToContainerType(param);
			continue;
		}
		if (param.find("Random") != std::string::npos)
			continue;
		
		SkepuParam skepuParam;
		skepuParam.param = param;
		if (param.find("skepu") != std::string::npos)
			skepuParam.type = SkepuParam::ParamType::RandomAccess;
		else if (foundParams == paramCount)
			skepuParam.type = SkepuParam::ParamType::Scalar;
		else
		{
			++foundParams;
			skepuParam.type = SkepuParam::ParamType::ElWise;
		}
		functionParameters.push_back(skepuParam);
	}
	if (container == "")
		container = getContainerType(potentialContainerType);

	std::vector<std::string> returnParameters;
	if (templateParameters[2].find("tuple") != std::string::npos)
		returnParameters = getTemplateParameters(templateParameters[2]);
	else
		returnParameters.push_back(templateParameters[2]);

	size_t retCounter{0};
	for (auto& ret : returnParameters)
		output.emplace_back(container + "<" + ret + ">& Return" + std::to_string(retCounter++));


	addInputParameters(functionParameters, output, container);
}

void handleMapReduce(const std::string& QualName, std::vector<std::string>& output, const std::string& potentialContainerType, std::string& retType)
{
	std::vector<std::string> templateParameters = getTemplateParameters(QualName);
	retType = templateParameters[3]; // (the return type of the reduction)
	std::vector<SkepuParam> functionParameters;
	std::string container = "";
	size_t paramCount = std::stoi(templateParameters[0]);
	size_t foundParams{0};
	// First 4 elements are param count, user defined param count, map ret, and reduce ret
	for (size_t index{4}; index < templateParameters.size(); ++index)
	{
		std::string param = templateParameters[index];
		if (param.find("Index") != std::string::npos)
		{
			container = convertDimToContainerType(param);
			continue;
		}
		if (param.find("Random") != std::string::npos)
			continue;
		
		SkepuParam skepuParam;
		skepuParam.param = param;
		if (param.find("skepu") != std::string::npos)
			skepuParam.type = SkepuParam::ParamType::RandomAccess;
		else if (foundParams == paramCount)
			skepuParam.type = SkepuParam::ParamType::Scalar;
		else
		{
			++foundParams;
			skepuParam.type = SkepuParam::ParamType::ElWise;
		}
		functionParameters.push_back(skepuParam);
	}
	if (container == "")
		container = getContainerType(potentialContainerType);

	addInputParameters(functionParameters, output, container);
}

void handleReduce(const std::string& QualName, std::vector<std::string>& output, std::string& retType)
{
	std::vector<std::string> templateParameters = getTemplateParameters(QualName);
	retType = templateParameters[0];
	output.emplace_back("skepu::container<" + retType + ">& Input");
}

void handleReduceMat(const std::string& QualName, std::vector<std::string>& output, std::string& retType)
{
	std::vector<std::string> templateParameters = getTemplateParameters(QualName);
	retType = templateParameters[0];
	output.emplace_back("skepu::Vector<" + retType + ">& Return");
	output.emplace_back("skepu::Matrix<" + retType + ">& Input");
	retType = "skepu::Vector<" + retType + ">";
}

void handleReduce2D(const std::string& QualName, std::vector<std::string>& output, std::string& retType)
{
	std::vector<std::string> templateParameters = getTemplateParameters(QualName);
	retType = templateParameters[0];
	output.emplace_back("skepu::container<" + retType + ">& Input");
}

void handleScan(const std::string& QualName, std::vector<std::string>& output, const std::string& potentialContainerType)
{
	std::vector<std::string> templateParameters = getTemplateParameters(QualName);
	std::string type = templateParameters[0];
	std::string container = getContainerType(potentialContainerType);

	output.emplace_back(container + "<" + type + ">& Return");
	output.emplace_back(container + "<" + type + ">& Input");
}

void handleScanIterator(const std::string& QualName, std::vector<std::string>& output, const std::string& potentialContainerType)
{
	std::vector<std::string> templateParameters = getTemplateParameters(QualName);
	std::string type = templateParameters[0];
	std::string container = getContainerIteratorType(potentialContainerType);

	output.emplace_back(container + "<" + type + ">& Return iterator begin");
	output.emplace_back(container + "<" + type + ">& Return iterator end");
	output.emplace_back(container + "<" + type + ">& Input iterator begin");
}

std::string handleMapOverlap(size_t dim, const std::string& QualName, std::vector<std::string>& output)
{
	std::array<std::string, 5> containers{"", "Vector", "Matrix", "Tensor3", "Tensor4"};
	std::vector<std::string> templateParameters = getTemplateParameters(QualName);
	std::vector<SkepuParam> functionParameters;

	std::string inType;
	bool foundIn{false};
	for (auto& parameter : templateParameters)
		if (parameter.find("skepu::Region" + std::to_string(dim) + "D") != std::string::npos ||
			parameter.find("skepu::Pool" + std::to_string(dim) + "D") != std::string::npos)
		{
			inType = getTemplateParameters(parameter)[0];
			foundIn = true;
		}
		else if (foundIn)
		{
			SkepuParam skepuParam;
			skepuParam.param = parameter;
			if (parameter.find("skepu") != std::string::npos)
				skepuParam.type = SkepuParam::ParamType::RandomAccess;
			else
				skepuParam.type = SkepuParam::ParamType::Scalar;
			functionParameters.push_back(skepuParam);
		}

	std::vector<std::string> returnParameters;
	if (templateParameters[0].find("tuple") != std::string::npos)
		returnParameters = getTemplateParameters(templateParameters[0]);
	else
		returnParameters.push_back(templateParameters[0]);

	size_t retCounter{0};
	for (auto& ret : returnParameters)
		output.emplace_back("skepu::" + containers[dim] + "<" + ret + ">& Return" + std::to_string(retCounter++));

	output.emplace_back("skepu::" + containers[dim] + "<" + inType + ">& Input");

	addInputParameters(functionParameters, output, "");

	return inType;
}

void handleMapPairs(const std::string& QualName, std::vector<std::string>& output, std::string& retType)
{
	std::vector<std::string> templateParameters = getTemplateParameters(QualName);
	std::vector<SkepuParam> functionParameters;

	size_t VParamCount = std::stoi(templateParameters[0]);
	size_t HParamCount = std::stoi(templateParameters[1]);
	size_t foundVParams = 0;
	size_t foundHParams = 0;

	// First two elements are vertical param count and horizontal param count
	for (size_t index{3}; index < templateParameters.size(); ++index)
	{
		std::string param = templateParameters[index];
		if (param.find("Index") != std::string::npos)
			continue;
		if (param.find("Random") != std::string::npos)
			continue;

		SkepuParam skepuParam;
		skepuParam.param = param;
		if (param.find("skepu") != std::string::npos)
			skepuParam.type = SkepuParam::ParamType::RandomAccess;
		else if (foundVParams < VParamCount)
		{
			++foundVParams;
			skepuParam.type = SkepuParam::ParamType::VElWise;
		}
		else if (foundHParams < HParamCount)
		{
			++foundHParams;
			skepuParam.type = SkepuParam::ParamType::HElWise;
		}
		else
			skepuParam.type = SkepuParam::ParamType::Scalar;
		functionParameters.push_back(skepuParam);
	}


	std::vector<std::string> returnParameters;
	if (templateParameters[2].find("tuple") != std::string::npos)
		returnParameters = getTemplateParameters(templateParameters[2]);
	else
		returnParameters.push_back(templateParameters[2]);

	size_t retCounter{0};
	for (auto& ret : returnParameters)
		output.emplace_back("skepu::Matrix<" + ret + ">& Return" + std::to_string(retCounter++));


	addInputParameters(functionParameters, output, "");
}

void handleMapPairsReduce(const std::string& QualName, std::vector<std::string>& output, const std::string& potentialContainerType)
{
	std::vector<std::string> templateParameters = getTemplateParameters(QualName);
	std::vector<SkepuParam> functionParameters;

	size_t VParamCount = std::stoi(templateParameters[0]);
	size_t HParamCount = std::stoi(templateParameters[1]);
	size_t foundVParams = 0;
	size_t foundHParams = 0;

	// First four elements are vertical param count, horizontal param count, map ret, and reduce ret
	for (size_t index{4}; index < templateParameters.size(); ++index)
	{
		std::string param = templateParameters[index];
		if (param.find("Index") != std::string::npos)
			continue;
		if (param.find("Random") != std::string::npos)
			continue;

		SkepuParam skepuParam;
		skepuParam.param = param;
		if (param.find("skepu") != std::string::npos)
			skepuParam.type = SkepuParam::ParamType::RandomAccess;
		else if (foundVParams < VParamCount)
		{
			++foundVParams;
			skepuParam.type = SkepuParam::ParamType::VElWise;
		}
		else if (foundHParams < HParamCount)
		{
			++foundHParams;
			skepuParam.type = SkepuParam::ParamType::HElWise;
		}
		else
			skepuParam.type = SkepuParam::ParamType::Scalar;
		functionParameters.push_back(skepuParam);
	}

	output.emplace_back("skepu::Vector<" + templateParameters[3] + ">& Return");


	addInputParameters(functionParameters, output, "");
}

std::string generateParameterString(SignatureInformation& Sig, const std::string& skeletonName, const std::vector<std::string>& parameters, const std::string& retType)
{
	std::string label{"skepu::" + skeletonName + "("};
	for (auto iter = parameters.begin(); iter != parameters.end(); ++iter) {
		const std::string& elem = *iter;
		label += elem;
		ParameterInformation pi{};
		pi.labelString = elem;
		
		Sig.parameters.push_back(pi);
		if (iter != parameters.end() - 1)
		label += ", ";
	}

	label += ")";
	if(retType != "")
		label += " -> " + retType;
	return label;
}


} // namespace

void skepuSignatureHelp(Sema &S, unsigned CurrentArg, CodeCompleteConsumer::OverloadCandidate *Candidates, unsigned NumCandidates,
                        SignatureHelp &SigHelp)
{
	using OverloadCandidate = CodeCompleteConsumer::OverloadCandidate;

	const SourceManager &SM = S.getSourceManager();
	auto MainFile = SM.getFileEntryRefForID(SM.getMainFileID());
	if (!MainFile)
		return;
	auto Info = infoFor(MainFile->getName(), S.getPreprocessor());
	if (!Info)
		return;
	const auto &SkeletonDescriptions = Info->SkeletonsStandard;

	// What clangd has made of the candidates; the documentation of it is kept
	const SignatureInformation Base = SigHelp.signatures.empty() ? SignatureInformation{} : SigHelp.signatures.front();

	for (unsigned i = 0; i < NumCandidates; ++i)
	{
		OverloadCandidate &Candidate = Candidates[i];
		const FunctionDecl *fd = Candidate.getFunction();
		if (!fd)
			continue;
		const std::string qualified = fd->getQualifiedNameAsString(), name = fd->getNameAsString();
		if (qualified.find("skepu::") == std::string::npos)
			continue;

		// A call of a skeleton factory (skepu::Map(f)): the user functions that it takes
		if (name != "operator()")
		{
			const SkepuInfo::Standard *skeleton = nullptr;
			for (const auto &pair : SkeletonDescriptions)
				if (pair.second.name == name)
					skeleton = &pair.second;
			if (!skeleton || qualified != "skepu::" + name)
				continue;

			SignatureInformation Sig;
			Sig.label = qualified + "(";
			for (size_t k{0}; k < skeleton->userfunctionArgAmount; ++k)
			{
				ParameterInformation parameter;
				parameter.labelString = "(*func" + std::to_string(k) + ")()";
				Sig.label += (k ? ", " : "") + parameter.labelString;
				Sig.parameters.push_back(std::move(parameter));
			}
			Sig.label += ") -> auto";
			auto description = Info->SkeletonDescriptions.find(name);
			if (description != Info->SkeletonDescriptions.end())
				Sig.documentation.value = description->second;

			SigHelp.signatures.clear();
			SigHelp.signatures.push_back(std::move(Sig));
			SigHelp.activeSignature = 0;
			SigHelp.activeParameter = std::min<unsigned>(CurrentArg, skeleton->userfunctionArgAmount ? skeleton->userfunctionArgAmount - 1 : 0);
			return;
		}

		auto ftd = Candidate.getFunctionTemplate();
		if (!ftd)
			continue;

		std::string QualName = ftd->getQualifiedNameAsString();
		clangd::log("QualName: {0}", QualName);

		// Get name of skeleton
		size_t doubleColonPos = QualName.find("::");
		size_t openAngleBracketPos = QualName.find('<');
		std::string implName = "";
		if (doubleColonPos != std::string::npos && openAngleBracketPos != std::string::npos)
			implName = QualName.substr(doubleColonPos + 2, openAngleBracketPos - doubleColonPos - 2);

		if (implName == "")
			return;

		// Remove impl from name
		size_t pos = implName.find("impl::");
		if (pos != std::string::npos)
			implName.erase(pos, 6); // 6 is the length of "impl::"

		// Check if name is in descriptions
		auto it = SkeletonDescriptions.find(implName);
		if (it == SkeletonDescriptions.end())
			return;
		const std::string type = it->second.name;

		std::vector<std::string> all;
		std::string retType{};

		SigHelp.signatures.clear();
		SignatureInformation Sig = Base;
		Sig.parameters.clear();
		SigHelp.activeParameter = CurrentArg;
		SigHelp.activeSignature = 0;

		bool hasOneOverload{true};

		clangd::log("Checking {0}", type);

		std::string candidateType = "";
		if (NumCandidates > 0 && Candidates[0].getNumParams() > 0)
			candidateType = Candidates[0].getParamType(0).getAsString();

		if (type == "MapOverlap1D")
		{
			hasOneOverload = false;
			std::string inType = handleMapOverlap(1, QualName, all);
			SignatureInformation Sig2{Sig};
			Sig.label = generateParameterString(Sig, type, all, "");
			SigHelp.signatures.push_back(std::move(Sig));
			all.back() = "skepu::Matrix<" + inType + ">& Input";
			Sig2.label = generateParameterString(Sig2, type, all, "");
			Sig2.documentation.value = "Performs a rowwise or colwise MapOverlap from Matrix to Vector";
			SigHelp.signatures.push_back(std::move(Sig2));
		}
		else if (type == "MapPool1D")
			handleMapOverlap(1, QualName, all);
		else if (type == "MapOverlap2D" || type == "MapPool2D")
			handleMapOverlap(2, QualName, all);
		else if (type == "MapOverlap3D" || type == "MapPool3D")
			handleMapOverlap(3, QualName, all);
		else if (type == "MapOverlap4D" || type == "MapPool4D")
			handleMapOverlap(4, QualName, all);
		else if (type == "Map")
			handleMap(QualName, all, candidateType);
		else if (type == "MapReduce")
			handleMapReduce(QualName, all, candidateType, retType);
		else if (type == "Reduce1D")
		{
			hasOneOverload = false;
			SignatureInformation Sig2{Sig};
			if (CurrentArg == 0)
			{
				handleReduce(QualName, all, retType);
				Sig.label = generateParameterString(Sig, type, all, retType);
				SigHelp.signatures.push_back(std::move(Sig));
			}
			all.clear();
			handleReduceMat(QualName, all, retType);
			Sig2.label = generateParameterString(Sig2, type, all, retType);
			SigHelp.signatures.push_back(std::move(Sig2));
		}
		else if (type == "Reduce2D")
			handleReduce2D(QualName, all, retType);
		else if (type == "MapPairs")
			handleMapPairs(QualName, all, retType);
		else if (type == "MapPairsReduce")
			handleMapPairsReduce(QualName, all, retType);
		else if (type == "Scan")
		{
			hasOneOverload = false;
			if (candidateType.find("skepu::") != std::string::npos && candidateType.find("Iterator") == std::string::npos)
			{
				handleScan(QualName, all, candidateType);
				Sig.label = generateParameterString(Sig, type, all, retType);
				SigHelp.signatures.push_back(std::move(Sig));
			}
			else if (candidateType.find("skepu::") != std::string::npos && candidateType.find("Iterator") != std::string::npos)
			{
				handleScanIterator(QualName, all, candidateType);
				Sig.label = generateParameterString(Sig, type, all, retType);
				SigHelp.signatures.push_back(std::move(Sig));
			}
			else
			{
				SignatureInformation Sig2{Sig};
				handleScan(QualName, all, candidateType);
				Sig.label = generateParameterString(Sig, type, all, retType);
				SigHelp.signatures.push_back(std::move(Sig));

				all.clear();
				handleScanIterator(QualName, all, candidateType);
				Sig2.label = generateParameterString(Sig2, type, all, retType);
				SigHelp.signatures.push_back(std::move(Sig2));
			}
		}
		else
			clangd::log("Unsupported skeleton: {0}", type);

		if (hasOneOverload)
		{
			Sig.label = generateParameterString(Sig, type, all, retType);
			SigHelp.signatures.push_back(std::move(Sig));
		}
		return;
	}
}

// Hover

void skepuHover(ParsedAST &AST, const NamedDecl &Decl, const FunctionDecl *Function, HoverInfo &HI)
{
	auto Info = infoFor(AST.tuPath(), AST.getPreprocessor());
	if (!Info)
		return;

	// A skeleton of SkePU (skepu::Map and so on)
	bool isSkeleton = false;
	if (Function)
	{
		for (const auto &e : Info->SkeletonTypes)
		{
			if (e.find(Decl.getNameAsString()) == std::string::npos)
				continue;
			if (const auto *NSD = dyn_cast_or_null<NamespaceDecl>(Function->getEnclosingNamespaceContext()))
			{
				isSkeleton = NSD->getQualifiedNameAsString() == "skepu";
				break;
			}
		}
	}

	if (isSkeleton)
	{
		clangd::log("It's a SkePU Skeleton");
		HI.ReturnType = HoverInfo::PrintedType("auto");

		std::string nameInfoS = Function->getNameInfo().getAsString();
		size_t pos = nameInfoS.find("impl::");
		if (pos != std::string::npos)
			nameInfoS.erase(pos, 6); // 6 is the length of "impl::"

		auto description = Info->SkeletonDescriptions.find(nameInfoS);
		if (description != Info->SkeletonDescriptions.end())
			HI.Documentation = "Skeleton of " + nameInfoS + ": " + description->second;

		auto preHI = HI.Parameters.value_or(std::vector<HoverInfo::Param>{});
		HI.Parameters.emplace();
		for (const auto &param : preHI)
		{
			HoverInfo::Param out = param;
			out.Type = HoverInfo::PrintedType(param.Type ? param.Type->Type.c_str() : "");
			HI.Parameters->emplace_back(out);
		}

		HI.Definition = "";
		return;
	}

	// An instance of a skeleton (a variable of the file): the declaration of the name is where the instance is declared
	const std::string DeclName = Decl.getNameAsString();
	const SourceManager &SM = AST.getASTContext().getSourceManager();
	const int DeclLine = sourceLocToPosition(SM, SM.getSpellingLoc(Decl.getLocation())).line;
	for (const auto &skeleton : Info->Skeletons)
	{
		if (skeleton.variableName != DeclName || skeleton.nameLine != DeclLine)
			continue;

		HI.ReturnType = HoverInfo::PrintedType(skeleton.resolvedReturnType.c_str());
		HI.Documentation = skeleton.description;
		HI.Parameters.emplace();
		HI.Definition = "";
		for (const auto &param : skeleton.resolvedParams)
		{
			HoverInfo::Param out;
			out.Type = HoverInfo::PrintedType(param.c_str());
			HI.Parameters->emplace_back(out);
		}
		return;
	}
}

// Document symbols

std::vector<DocumentSymbol> skepuDocumentSymbols(ParsedAST &AST, std::vector<DocumentSymbol> Syms)
{
	auto Info = infoFor(AST.tuPath(), AST.getPreprocessor());
	std::vector<DocumentSymbol> newSyms{};
	static const SkepuInfo None;
	const SkepuInfo &I = Info ? *Info : None;

	DocumentSymbol normal{"Various", "", SymbolKind::Field};

	DocumentSymbol skeletons{"Skeletons", "Skepu Skeletons", SymbolKind::Field};
	for (const auto &skeleton : I.Skeletons)
	{
		DocumentSymbol temp{};
		temp.name = skeleton.variableName;
		temp.kind = SymbolKind::Method;
		temp.selectionRange = skeleton.selectionRange;
		temp.range = skeleton.range;
		temp.detail = skeleton.details;

		skeletons.children.push_back(temp);
	}

	if (!I.Skeletons.empty())
		newSyms.push_back(skeletons);

	DocumentSymbol userfunctions{"Userfunctions", "Skepu Userfunctions", SymbolKind::Function};
	DocumentSymbol userConstants{"User Constants", "Skepu User Constants", SymbolKind::Field};
	DocumentSymbol userTypes{"User Types", "Skepu User Types", SymbolKind::Field};

	std::vector<std::string> usedUsedFuncs{};
	for (auto &sy : Syms)
	{
		if (std::find(usedUsedFuncs.begin(), usedUsedFuncs.end(), sy.name) != usedUsedFuncs.end())
			continue;

		for (const auto &userfunc : I.UserFunctions)
		{
			if (userfunc.name != sy.name || !sy.range.contains(userfunc.selectionRange))
				continue;

			usedUsedFuncs.push_back(userfunc.name);
			DocumentSymbol temp{};
			temp = sy;
			temp.kind = SymbolKind::Function;
			userfunctions.children.push_back(temp);
			break;
		}

		for (const auto &uc : I.UserConstants)
		{
			if (uc.name != sy.name || !sy.range.contains(uc.selectionRange))
				continue;

			DocumentSymbol temp{};
			temp = sy;
			userConstants.children.push_back(temp);
			usedUsedFuncs.push_back(uc.name);
			break;
		}

		for (const auto &ut : I.UserTypes)
		{
			if (ut.name != sy.name || !sy.range.contains(ut.selectionRange) ||
				std::find(usedUsedFuncs.begin(), usedUsedFuncs.end(), ut.name) != usedUsedFuncs.end())
				continue;

			DocumentSymbol temp{};
			temp = sy;
			userTypes.children.push_back(temp);
			usedUsedFuncs.push_back(ut.name);
			break;
		}
	}

	for (auto &sy : Syms)
	{
		if (std::find(usedUsedFuncs.begin(), usedUsedFuncs.end(), sy.name) != usedUsedFuncs.end())
			continue;

		normal.children.push_back(sy);
	}

	if (!userConstants.children.empty())
		newSyms.push_back(userConstants);

	if (!userfunctions.children.empty())
		newSyms.push_back(userfunctions);

	if (!userTypes.children.empty())
		newSyms.push_back(userTypes);

	newSyms.push_back(normal);
	return newSyms;
}

} // namespace clangd
} // namespace clang
