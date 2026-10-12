#pragma once

#include <string>
#include <unordered_set>
#include <utility>

#include "clang/AST/AST.h"

#ifdef SKEPU_LSP
#include <Protocol.h>
#include <FindSymbols.h>
#endif

// ------------------------------
// Data structures
// ------------------------------

enum class Backend
{
	CPU, OpenMP, CUDA, OpenCL,
};

enum class AccessMode
{
	Read,
	Write,
	ReadWrite
};

enum class ContainerType
{
	Vector,
	Matrix,
	MatRow,
	MatCol,
	Tensor3,
	Tensor4,
	SparseMatrix,
	Region1D,
	Region2D,
	Region3D,
	Region4D,
	Pool1D,
	Pool2D,
	Pool3D,
	Pool4D
};

inline std::string toString(ContainerType type) {
	switch(type) {
		case ContainerType::Vector:
			return "Vector";
		case ContainerType::Matrix:
			return "Matrix";
		case ContainerType::MatRow:
			return "MatRow";
		case ContainerType::MatCol:
			return "MatCol";
		case ContainerType::Tensor3:
			return "Tensor3";
		case ContainerType::Tensor4:
			return "Tensor4";
		case ContainerType::SparseMatrix:
			return "SparseMatrix";
		case ContainerType::Region1D:
			return "Region1D";
		case ContainerType::Region2D:
			return "Region2D";
		case ContainerType::Region3D:
			return "Region3D";
		case ContainerType::Region4D:
			return "Region4D";
		case ContainerType::Pool1D:
			return "Pool1D";
		case ContainerType::Pool2D:
			return "Pool2D";
		case ContainerType::Pool3D:
			return "Pool3D";
		case ContainerType::Pool4D:
			return "Pool4D";
		default:
			return "Unknown";
	}
}

class UserConstant
{
public:
	const clang::VarDecl *astDeclNode;

	std::string name;
	std::string typeName;
	std::string definition;

#ifdef SKEPU_LSP
	clang::clangd::Range selectionRange;
#endif
	UserConstant(const clang::VarDecl *v);
};

class UserType
{
public:
	const clang::CXXRecordDecl *astDeclNode;

	std::string name;
	const clang::Type *type;
	std::string typeNameOpenCL;
	bool requiresDoublePrecision;

#ifdef SKEPU_LSP
	clang::clangd::Range selectionRange;
#endif

	UserType(const clang::CXXRecordDecl *t);
};

class UserFunction
{
public:

	struct TemplateArgument
	{
		const std::string paramName;
		const std::string rawTypeName;
		const std::string resolvedTypeName;
		// For a non-type template argument: the type of the parameter; the (raw and resolved) name is its value
		const std::string valueType;

		TemplateArgument(std::string name, std::string rawType, std::string resolvedType, std::string valueType = "");

		// Declaration that gives the parameter name its meaning in the code of the user function
		std::string alias(bool constexprValue) const;
	};

	struct Param
	{
		const clang::ParmVarDecl *astDeclNode;
		const clang::Type *type;
		
		std::string name;
		std::string rawTypeName;
		std::string resolvedTypeName;
		std::string escapedTypeName;
		std::string fullTypeName;
		std::string unqualifiedFullTypeName;
		bool isReferenceType = false;
		bool isRValueReference = false;
		bool isLValueReference = false;
		
		static bool constructibleFrom(const clang::ParmVarDecl *p);
		
		Param(const clang::ParmVarDecl *p);
		virtual ~Param() = default;
		
		std::string templateInstantiationType() const;
		virtual size_t numKernelArgsCL() const;
		std::string typeNameOpenCL() const;
	};

	struct RandomAccessParam: Param
	{
		AccessMode accessMode;
		ContainerType containerType;
		const clang::Type *containedType;
		
		static bool constructibleFrom(const clang::ParmVarDecl *p);
		
		RandomAccessParam(const clang::ParmVarDecl *p);
		virtual ~RandomAccessParam() = default;
		
		virtual size_t numKernelArgsCL() const override;
		std::string TypeNameOpenCL() const;
		std::string innerTypeNameOpenCL() const;
		std::string TypeNameHost() const;
	};
	
	struct RegionParam: RandomAccessParam
	{
		static bool constructibleFrom(const clang::ParmVarDecl *p);
		
		RegionParam(const clang::ParmVarDecl *p);

		bool isPool;
	};
	
	struct RandomParam: Param
	{
		static bool constructibleFrom(const clang::ParmVarDecl *p);
		
		size_t randomCount;
		
		RandomParam(const clang::ParmVarDecl *p);
	};

	void updateArgLists(size_t arity, size_t Harity = 0);

	bool refersTo(UserFunction &other);

	std::string funcNameCUDA();
	size_t numKernelArgsCL();
	std::string multiReturnTypeNameGPU();

	clang::FunctionDecl *astDeclNode;

	std::string rawName;
	std::string uniqueName;
	std::string rawReturnTypeName;
	std::string resolvedReturnTypeName;
	
	std::string returnTypeNameOpenCL();
	
	std::vector<std::string> multipleReturnTypes {};

	std::string instanceName;

	clang::SourceLocation codeLocation;

#ifdef SKEPU_LSP
	clang::clangd::Range selectionRange;
	clang::clangd::Range range;
#endif
	
	size_t Varity = 0, Harity = 0;
	
	RandomParam* randomParam = nullptr;
	size_t randomCount;
	
	RegionParam* regionParam = nullptr;
	std::vector<Param> elwiseParams{};
	std::vector<RandomAccessParam> anyContainerParams {};
	std::vector<Param> anyScalarParams {};
	Param *indexParam; // can be NULL

	std::vector<TemplateArgument> templateArguments{};

	std::vector<std::pair<const clang::CallExpr*, UserFunction*>> UFReferences{};
	std::set<UserFunction*> ReferencedUFs{};
	
	
	std::set<const clang::CallExpr*> ReferencedRets{};
	std::set<clang::CXXMemberCallExpr*> ReferencedGets{};

	// the places in the body where a user type is named (by a name that needs a qualifier or an alias, see UserFunctionVisitor)
	std::vector<std::pair<clang::SourceRange, UserType*>> UTReferences{};
	std::set<UserType*> ReferencedUTs{};
	// the aliases (typedef, using) that the body names and that are declared outside the function, with the OpenCL type of each
	std::map<std::string, std::string> TypeAliases{};


	std::vector<clang::CXXOperatorCallExpr*> containerSubscripts{}, containerCalls{};
	std::vector<clang::CXXOperatorCallExpr*> operatorOverloads{};

	bool fromTemplate = false;
	bool indexed1D = false;
	bool indexed2D = false;
	bool indexed3D = false;
	bool indexed4D = false;
	bool requiresDoublePrecision;
	bool hasMatRowOrCol = false;


	UserFunction(clang::FunctionDecl *f);
	UserFunction(clang::CXXMethodDecl *f, clang::SourceLocation codeLocation);
};

#ifdef SKEPU_LSP
struct SkepuContainerInvocation
{
	std::string variableName;

	ContainerType type;
	clang::SourceLocation defloc;
	clang::SourceLocation invloc;

	std::string defLocationString;
	std::string invLocationString;
};

struct SkepuContainer
{
	std::string type;
	std::string typenameType;
	std::string variableName;

	std::vector<SkepuContainerInvocation*> invocations;
	clang::clangd::Range selectionRange;
	clang::clangd::Range range;

	std::string locationString;
	
	~SkepuContainer()
	{
		for(auto *invocation: this->invocations)
		{
			delete invocation;
		}
	}
};

struct SkeletonInvocation
{
	clang::QualType retType;
	std::string variableName;

	clang::SourceLocation defloc;
	clang::SourceLocation invloc;

	std::string defLocationString;
	std::string invLocationString;
};
#endif

struct Skeleton
{
	enum class Type
	{
		Map,
		Reduce1D,
		Reduce2D,
		MapReduce,
		MapPairs,
		MapPairsReduce,
		Scan,
		MapOverlap1D,
		MapOverlap2D,
		MapOverlap3D,
		MapOverlap4D,
		MapPool1D,
		MapPool2D,
		MapPool3D,
		MapPool4D
	};

	std::string name;
	Type type;
	size_t userfunctionArgAmount;
	size_t deviceKernelAmount;

#ifdef SKEPU_LSP
	std::vector<std::string> defaultParams;

	std::vector<UserFunction*> parameters;
	std::vector<SkeletonInvocation*> invocations;
	
	clang::SourceLocation location;
	clang::clangd::Position namePosition;   // where the name of the instance is
	clang::clangd::Range selectionRange;
	clang::clangd::Range range;

	std::string locationString;

	clang::QualType retType;
	std::string resolvedReturnType;
	std::vector<std::string> resolvedParams;
	
	//std::string invocationDescription;
	std::string description;
	std::string variableName;

	//const std::string getDescription() const;
	//void printVariables();
	//const std::string getReturn() const;
	void constructLayout();

	std::string getDetails() const;

	~Skeleton()
	{
		for(auto *invocation: this->invocations)
		{
			delete invocation;
		}
	}

	std::string typeToString(Skeleton::Type type)
	{
		switch (type)
		{
			case Skeleton::Type::Map:
				return "Map";
			case Skeleton::Type::Reduce1D:
				return "Reduce1D";
			case Skeleton::Type::Reduce2D:
				return "Reduce2D";
			case Skeleton::Type::MapReduce:
				return "MapReduce";
			case Skeleton::Type::MapPairs:
				return "MapPairs";
			case Skeleton::Type::MapPairsReduce:
				return "MapPairsReduce";
			case Skeleton::Type::Scan:
				return "Scan";
			case Skeleton::Type::MapOverlap1D:
				return "MapOverlap1D";
			case Skeleton::Type::MapOverlap2D:
				return "MapOverlap2D";
			case Skeleton::Type::MapOverlap3D:
				return "MapOverlap3D";
			case Skeleton::Type::MapOverlap4D:
				return "MapOverlap4D";
			case Skeleton::Type::MapPool1D:
				return "MapPool1D";
			case Skeleton::Type::MapPool2D:
				return "MapPool2D";
			case Skeleton::Type::MapPool3D:
				return "MapPool3D";
			case Skeleton::Type::MapPool4D:
				return "MapPool4D";
			default:
				return "Unknown"; // or throw an exception
		}
	}
#endif
};