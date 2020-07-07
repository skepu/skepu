#pragma once

#include <string>
#include <unordered_set>
#include <utility>

#include "clang/AST/AST.h"



// ------------------------------
// Data structures
// ------------------------------

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
	Tensor3,
	Tensor4,
	SparseMatrix
};


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
		Call
	};

	std::string name;
	Type type;
	size_t userfunctionArgAmount;
	size_t deviceKernelAmount;
};


class UserConstant
{
public:
	const clang::VarDecl *astDeclNode;

	std::string name;
	std::string typeName;
	std::string definition;

	UserConstant(const clang::VarDecl *v);
};


class UserType
{
public:
	const clang::CXXRecordDecl *astDeclNode;

	std::string name;
	bool requiresDoublePrecision;

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

		TemplateArgument(std::string name, std::string rawType, std::string resolvedType);
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
		
		static bool constructibleFrom(const clang::ParmVarDecl *p);
		
		Param(const clang::ParmVarDecl *p);
		virtual ~Param() = default;
		
		std::string templateInstantiationType() const;
		virtual size_t numKernelArgsCL() const;
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
		std::string TypeNameOpenCL();
		std::string TypeNameHost();
	};

	void updateArgLists(size_t arity, size_t Harity = 0);

	bool refersTo(UserFunction &other);

	std::string funcNameCUDA();

	clang::FunctionDecl *astDeclNode;

	std::string rawName;
	std::string uniqueName;
	std::string rawReturnTypeName;
	std::string resolvedReturnTypeName;
	
	std::vector<std::string> multipleReturnTypes {};

	std::string instanceName;

	clang::SourceLocation codeLocation;
	
	size_t Varity = 0, Harity = 0;

	std::vector<Param> elwiseParams{};
	std::vector<RandomAccessParam> anyContainerParams {};
	std::vector<Param> anyScalarParams {};
	Param *indexParam; // can be NULL

	std::vector<TemplateArgument> templateArguments{};

	std::vector<std::pair<const clang::CallExpr*, UserFunction*>> UFReferences{};
	std::set<UserFunction*> ReferencedUFs{};
	
	
	std::set<const clang::CallExpr*> ReferencedRets{};

	std::vector<std::pair<const clang::TypeSourceInfo*, UserType*>> UTReferences{};
	std::set<UserType*> ReferencedUTs{};


	std::vector<clang::CXXOperatorCallExpr*> containerSubscripts{};

	bool fromTemplate = false;
	bool indexed1D = false;
	bool indexed2D = false;
	bool indexed3D = false;
	bool indexed4D = false;
	bool requiresDoublePrecision;

	size_t numKernelArgsCL();

	UserFunction(clang::FunctionDecl *f);
	UserFunction(clang::CXXMethodDecl *f, clang::VarDecl *d);
};
