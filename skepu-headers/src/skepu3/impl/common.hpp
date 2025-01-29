#pragma once

#define REQUIRES(...) typename std::enable_if<(__VA_ARGS__), bool>::type = 0
#define REQUIRES_B(...) typename std::enable_if<(__VA_ARGS__), bool>::type
#define REQUIRES_VALUE(...) typename std::enable_if<__VA_ARGS__::value, bool>::type = 0
#define REQUIRES_DEF(...) typename std::enable_if<(__VA_ARGS__), bool>::type

#define MAX_SIZE ((size_t)-1)

#define VARIANT_OPENCL(block)
#define VARIANT_CUDA(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CPU(block) block

#include <vector>
#include <array>
#include <iostream>
#include <utility>
#include <cassert>
#include <algorithm>
#include <functional>
#include <cstdbool>
#include <iomanip>
#include <unordered_map>
#include <atomic>



namespace skepu
{
	constexpr bool no_dealloc = false;

	struct PrecompilerMarker {};

	inline size_t closestPow2(size_t x)
	{
		size_t guess = 1;
		while (guess < x) { guess *= 2; }
		return guess;
	}


	class UniqueIdentifier
	{
	public:

		using ID = int;

		static ID generate()
		{
			static std::atomic<ID> id_counter{1000};
			return (ID)id_counter++;
		}
		
		static ID null()
		{
			return (ID)0;
		}
	};


	#define DEBUG_REG(var) skepu::debug.registerObjectName(&var, #var);
	#define DEBUG_REG_C(var) skepu::debug.registerObjectName(var.getAddress(), #var);

	class DebugInfo
	{
		std::unordered_map<const void*, std::string> objectNames;

	public:

		void registerObjectName(const void *object, std::string name)
		{
			std::cout << "registered object " << object << " with name " << name << "\n";
			this->objectNames[object] = name;
		}

		std::string getObjectName(const void *object)
		{
			auto it = this->objectNames.find(object);
			if (it == this->objectNames.end())
			{
				std::stringstream ss;
				ss << object;
				return ss.str();
			}
			else
				return (*it).second;
		}
	};

	static DebugInfo debug;

	// ----------------------------------------------------------------
	// sizes and indices structures
	// ----------------------------------------------------------------

	struct Index1D
	{
		size_t i;
	};

	struct Index2D
	{
		size_t row, col;
	};

	struct Index3D
	{
		size_t i, j, k;
	};

	struct Index4D
	{
		size_t i, j, k, l;
	};

	struct ProxyTag
	{
		struct Default {};
		struct MatRow {};
		struct MatCol {};
	};

	inline Index1D make_index(std::integral_constant<int, 1>, size_t index, size_t, size_t, size_t)
	{
		return Index1D{index};
	}

	inline Index2D make_index(std::integral_constant<int, 2>, size_t index, size_t size_j, size_t, size_t)
	{
		return Index2D{ index / size_j, index % size_j };
	}

	inline Index3D make_index(std::integral_constant<int, 3>, size_t index, size_t size_j, size_t size_k, size_t)
	{
		size_t ci = index / (size_j * size_k);
		index = index % (size_j * size_k);
		size_t cj = index / (size_k);
		index = index % (size_k);
		return Index3D{ ci, cj, index };
	}

	inline Index4D make_index(std::integral_constant<int, 4>, size_t index, size_t size_j, size_t size_k, size_t size_l)
	{
		size_t ci = index / (size_j * size_k * size_l);
		index = index % (size_j * size_k * size_l);
		size_t cj = index / (size_k * size_l);
		index = index % (size_k * size_l);
		size_t ck = index / (size_l);
		index = index % (size_l);
		return Index4D{ ci, cj, ck, index };
	}

	inline std::ostream & operator<<(std::ostream &o, Index1D idx)
	{
		return o << "Index1D(" << idx.i << ")";
	}

	inline std::ostream & operator<<(std::ostream &o, Index2D idx)
	{
		return o << "Index2D(" << idx.row << ", " << idx.col << ")";
	}

	inline std::ostream & operator<<(std::ostream &o, Index3D idx)
	{
		return o << "Index3D(" << idx.i << ", "  << idx.j << ", " << idx.k << ")";
	}

	inline std::ostream & operator<<(std::ostream &o, Index4D idx)
	{
		return o << "Index4D(" << idx.i << ", "  << idx.j << ", " << idx.k << ", " << idx.l << ")";
	}


	/*!
	 *  Enumeration of the different edge policies (what happens when a read outside the vector is performed) that the map overlap skeletons support.
	 */
	enum class Edge
	{
		None, Cyclic, Duplicate, Pad
	};

	enum class Overlap
	{
		RowWise, ColWise
	};

	enum class UpdateMode
	{
		Normal, RedBlack, Red, Black
	};

	// ReduceMode

	enum class ReduceMode
	{
		RowWise, ColWise
	};

	inline std::ostream &operator<<(std::ostream &o, ReduceMode m)
	{
		switch (m)
		{
		case ReduceMode::RowWise:
			o << "Rowwise"; break;
		case ReduceMode::ColWise:
			o << "Colwise"; break;
		default:
			o << "<Invalid reduce mode>";
		}
		return o;
	}

	// ScanMode

	enum class ScanMode
	{
		Inclusive, Exclusive
	};

	enum class AccessMode
	{
		Read,
		Write,
		ReadWrite,
		None
	};

	static inline constexpr bool hasReadAccess(AccessMode m)
	{
		return m == AccessMode::Read || m == AccessMode::ReadWrite;
	}

	static inline constexpr bool hasWriteAccess(AccessMode m)
	{
		return m == AccessMode::Write || m == AccessMode::ReadWrite;
	}

	enum class SkeletonType
	{
		Map,
		MapReduce,
		MapPairs,
		MapPairsReduce,
		Reduce1D,
		Reduce2D,
		Scan,
		MapOverlap1D,
		MapOverlap2D,
		MapOverlap3D,
		MapOverlap4D,
		Call,
	};

	// For multiple return Map variants

	template<typename... args>
	using multiple = std::tuple<args...>;

	template <typename... Args>
	auto ret(Args&&... args) -> decltype(std::make_tuple(std::forward<Args>(args)...)) {
		return std::make_tuple(std::forward<Args>(args)...);
	}

	// Parity for MapOverlap

	enum class Parity: int
	{
		None = -1, Even = 0, Odd = 1
	};

	constexpr static bool index_is_odd(size_t i, size_t j = 0, size_t k = 0, size_t l = 0)
	{
		return ((i + j + k + l) % 2) != 0;
	}

	constexpr static bool index_parity(Parity p, size_t i, size_t j = 0, size_t k = 0, size_t l = 0)
	{
		return ((i + j + k + l) % 2) == static_cast<int>(p);
	}


#ifdef SKEPU_OPENCL

	/*!
	 * helper to return data type in a string format using template specialication technique.
	 * Compile-time error if no overload is found.
	 */
	template<typename T>
	inline std::string getDataTypeCL();

	template<> inline std::string getDataTypeCL<char>          () { return "char";           }
	template<> inline std::string getDataTypeCL<unsigned char> () { return "unsigned char";  }
	template<> inline std::string getDataTypeCL<short>         () { return "short";          }
	template<> inline std::string getDataTypeCL<unsigned short>() { return "unsigned short"; }
	template<> inline std::string getDataTypeCL<int>           () { return "int";            }
	template<> inline std::string getDataTypeCL<unsigned int>  () { return "unsigned int";   }
	template<> inline std::string getDataTypeCL<long>          () { return "long";           }
	template<> inline std::string getDataTypeCL<unsigned long> () { return "unsigned long";  }
	template<> inline std::string getDataTypeCL<float>         () { return "float";          }
	template<> inline std::string getDataTypeCL<double>        () { return "double";         }

	template<typename T>
	inline std::string getDataTypeDefCL()
	{
		return "";
	}

#endif
}


#include "tracing.h"
#include "meta_helpers.hpp"
#include "skepu3/scalar.hpp"
#include "skepu3/vector.hpp"
#include "skepu3/matrix.hpp"
#include "skepu3/tensor.hpp"
#include "skepu3/sparse_matrix.hpp"
#include "random.hpp"
#include "skepu3/backend/skeleton_base.h"
#include "tracing.hpp"
#include "stride_list.hpp"

namespace skepu
{

	// Dummy base class for sequential skeleton classes.
	// Includes empty member functions which has no meaning in a sequential context.
	class SeqSkeletonBase
	{
	public:

		void setLabel(std::string label)
		{
			this->m_label = label;
		}

		std::string getLabel()
		{
			if (this->m_label != "")
				return this->m_label;

			std::stringstream ss;
			ss << skepu::debug.getObjectName(this);
			return ss.str();
		}

		void setSourceLoc(std::string&& file_name, int line_nr)
		{
#ifdef SKEPU_TRACING
			this->m_file_name = file_name;
			this->m_line_nr = line_nr;
#endif
		}

#ifdef SKEPU_TRACING
		std::tuple<std::string&, int> getSourceLoc()
		{
			return { this->m_file_name, this->m_line_nr };
		}
#endif

		void setBackend(BackendSpec) {}
		void resetBackend() {}
		std::string selectBackend(size_t size = 0)
		{
			return "N/A";
		}

		void setExecPlan(ExecPlan *plan)
		{
			delete plan;
		}

		template<typename... Args>
		void tune(Args&&... args) { }

		void setPRNG(PRNG &prng, size_t iterations = 1)
		{
			this->m_prng = &prng;
			this->m_prng->registerInstance(this, iterations);
		}

	protected:

		template<size_t randomCount, REQUIRES(randomCount != SKEPU_NO_RANDOM)>
		Random<randomCount> prepareRandom(size_t size)
		{
			if (this->m_prng == nullptr)
				SKEPU_ERROR("No random stream set in skeleton instance");

			return this->m_prng->template asRandom<randomCount>(size);
		}

		template<size_t randomCount, REQUIRES(randomCount != SKEPU_NO_RANDOM)>
		skepu::Vector<Random<randomCount>> prepareRandom(size_t size, size_t copies)
		{
			if (this->m_prng == nullptr)
				SKEPU_ERROR("No random stream set in skeleton instance");

			return this->m_prng->template asRandom<randomCount>(size, copies);
		}

		template<size_t randomCount, REQUIRES(randomCount == SKEPU_NO_RANDOM)>
		PRNG::Placeholder prepareRandom(size_t size = 0, size_t copies = 0)
		{
			return {};
		}

		std::string m_label = "";
		std::string m_file_name;
		int m_line_nr = -1;
		PRNG *m_prng = nullptr;
	};



	inline size_t elwise_i(std::tuple<>) { return 0; }
	inline size_t elwise_j(std::tuple<>) { return 0; }
	inline size_t elwise_k(std::tuple<>) { return 0; }
	inline size_t elwise_l(std::tuple<>) { return 0; }

	template<typename... Args>
	inline size_t elwise_i(std::tuple<Args...> &t)
	{
		return std::get<0>(t).getParent().size_i();
	}

	template<typename... Args>
	inline size_t elwise_j(std::tuple<Args...> &t)
	{
		return std::get<0>(t).getParent().size_j();
	}

	template<typename... Args>
	inline size_t elwise_k(std::tuple<Args...> &t)
	{
		return std::get<0>(t).getParent().size_k();
	}

	template<typename... Args>
	inline size_t elwise_l(std::tuple<Args...> &t)
	{
		return std::get<0>(t).getParent().size_l();
	}

	// ----------------------------------------------------------------
	// is_skepu_{vector|matrix|container} trait classes
	// ----------------------------------------------------------------

	template<typename T>
	struct is_skepu_vector: std::false_type {};

	template<typename T>
	struct is_skepu_vector<skepu::Vector<T>>: std::true_type {};


	template<typename T>
	struct is_skepu_matrix: std::false_type {};

	template<typename T>
	struct is_skepu_matrix<skepu::Matrix<T>>: std::true_type {};

	template<typename T>
	struct is_skepu_matrix<skepu::SparseMatrix<T>>: std::true_type {};


	template<typename T>
	struct is_skepu_tensor3: std::false_type {};

	template<typename T>
	struct is_skepu_tensor3<skepu::Tensor3<T>>: std::true_type {};


	template<typename T>
	struct is_skepu_tensor4: std::false_type {};

	template<typename T>
	struct is_skepu_tensor4<skepu::Tensor4<T>>: std::true_type {};


	template<typename T>
	struct is_skepu_container:
		std::integral_constant<bool,
			/* This is an or expression rewritten as an and expression because the
			 * compiler Mercurium cannot parse an constexpr or expression with three
			 * or more operands.
			 */
			!(!is_skepu_vector<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value &&
				!is_skepu_matrix<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value &&
				!is_skepu_tensor3<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value &&
				!is_skepu_tensor4<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value)>
		{};

	/** Check that all parameters in a pack are SkePU containers. */
	template<typename ...> struct are_skepu_containers;

	/* The empty pack is true. */
	template<> struct are_skepu_containers<> : std::true_type {};

	/* Check that first in pack is container and recurse the rest. */
	template<typename CAR, typename ... CDR>
	struct are_skepu_containers<CAR, CDR...>
	: std::integral_constant<bool,
			is_skepu_container<CAR>::value
			&& are_skepu_containers<CDR...>::value>
	{};

	template<typename T>
	struct is_skepu_vector_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_vector_proxy<skepu::Vec<T>>: std::true_type {};

	template<typename T>
	struct is_skepu_matrix_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_matrix_proxy<skepu::Mat<T>>: std::true_type {};

	template<typename T>
	struct is_skepu_matrix_proxy<skepu::MatRow<T>>: std::true_type {};

	template<typename T>
	struct is_skepu_matrix_proxy<skepu::MatCol<T>>: std::true_type {};

	template<typename T>
	struct is_skepu_matrix_proxy<skepu::SparseMat<T>>: std::true_type {};

	template<typename T>
	struct is_skepu_tensor3_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_tensor3_proxy<skepu::Ten3<T>>: std::true_type {};

	template<typename T>
	struct is_skepu_tensor4_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_tensor4_proxy<skepu::Ten4<T>>: std::true_type {};

	template<typename T>
	struct is_skepu_container_proxy :
		std::integral_constant<bool,
			/* This is an or expression rewritten as an and expression because the
			 * compiler Mercurium cannot parse an constexpr or expression with three
			 * or more operands.
			 */
			!(!is_skepu_vector_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value &&
				!is_skepu_matrix_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value &&
				!is_skepu_tensor3_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value &&
				!is_skepu_tensor4_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value)>
  {};

	template<typename T>
	struct remove_scalar
	{
		using type = T;
	};

	template<typename T>
	struct remove_scalar<Scalar<T>>
	{
		using type = T;
	};



	// ----------------------------------------------------------------
	// is_skepu_iterator trait class
	// ----------------------------------------------------------------

	template<typename T, typename Ret>
	struct is_skepu_iterator: bool_constant<
		/* This is an or expression rewritten as an and expression because the
		 * compiler Mercurium cannot parse an constexpr or expression with three
		 * or more operands.
		 */
		!(!std::is_same<T, typename Vector<Ret>::iterator>::value &&
			!std::is_same<T, typename Vector<Ret>::const_iterator>::value &&
			!std::is_same<T, typename Matrix<Ret>::iterator>::value)>
	{};


	// ----------------------------------------------------------------
	// index trait classes for skepu::IndexND (N in [1,2,3,4])
	// ----------------------------------------------------------------

	// returns the dimensionality of an index type.
	// that is, if the type is skepu::IndexND, then returns N, else 0.
	template<typename T>
	struct index_dimension: std::integral_constant<int, 0>{};

	template<>
	struct index_dimension<Index1D>: std::integral_constant<int, 1>{};

	template<>
	struct index_dimension<Index2D>: std::integral_constant<int, 2>{};

	template<>
	struct index_dimension<Index3D>: std::integral_constant<int, 3>{};

	template<>
	struct index_dimension<Index4D>: std::integral_constant<int, 4>{};

	// true iff T is a SkePU index type
	template<typename T>
	struct is_skepu_index: bool_constant<index_dimension<T>::value != 0>{};

	// true iff first element of Args is SkePU index type
	template<typename... Args>
	struct is_indexed: bool_constant<is_skepu_index<typename first_element<Args...>::type>::value>{};

	template<>
	struct is_indexed<>: std::false_type{};

	// ----------------------------------------------------------------
	// matrix row proxy trait class
	// ----------------------------------------------------------------

	template<typename T>
	struct proxy_tag {
		using type = ProxyTag::Default;
	};

	template<typename T>
	struct proxy_tag<MatRow<T>> {
		using type = ProxyTag::MatRow;
	};

	template<typename T>
	struct proxy_tag<MatCol<T>> {
		using type = ProxyTag::MatCol;
	};

	// ----------------------------------------------------------------
	// smart container size extractor
	// ----------------------------------------------------------------

	inline std::tuple<size_t, size_t, size_t, size_t> size_info(index_dimension<skepu::Index1D>, size_t i, size_t, size_t, size_t)
	{
		return std::tuple<size_t, size_t, size_t, size_t>{i, 1, 1, 1};
	}

	inline std::tuple<size_t, size_t, size_t, size_t> size_info(index_dimension<skepu::Index2D>, size_t i, size_t j, size_t, size_t)
	{
		return std::tuple<size_t, size_t, size_t, size_t>{i, j, 1, 1};
	}

	inline std::tuple<size_t, size_t, size_t, size_t> size_info(index_dimension<skepu::Index3D>, size_t i, size_t j, size_t k, size_t)
	{
		return std::tuple<size_t, size_t, size_t, size_t>{i, j, k, 1};
	}

	inline std::tuple<size_t, size_t, size_t, size_t> size_info(index_dimension<skepu::Index4D>, size_t i, size_t j, size_t k, size_t l)
	{
		return std::tuple<size_t, size_t, size_t, size_t>{i, j, k, l};
	}

	template<typename Index, typename... Args>
	inline auto size_info(Index, size_t, size_t, size_t, size_t, Args&&... args) -> decltype(get<0, Args...>(std::forward<Args>(args)...).getParent().size_info())
	{
		return get<0, Args...>(std::forward<Args>(args)...).getParent().size_info();
	}



	// ----------------------------------------------------------------
	// Arity deducer for Map user functions
	// ----------------------------------------------------------------

	#define SKEPU_UNSET_ARITY -1

	template<int DA, typename... Args>
	struct resolve_map_arity: index_constant<
		DA != SKEPU_UNSET_ARITY
		? DA
		: trait_count_first_not<is_skepu_container_proxy, Args...>::value
			- (is_indexed<Args...>::value ? 1 : 0)
			- (has_random<Args...>::value ? 1 : 0)
	> {};


	// ----------------------------------------------------------------
	// Smart Container Coherency Helpers
	// ----------------------------------------------------------------

	/*
	 * Base case for recursive variadic flush.
	 */
	template<FlushMode mode>
	void flush() {}

	/*
	 *
	 */
	template<FlushMode mode = FlushMode::Default, typename First, typename... Args>
	void flush(First&& first, Args&&... args)
	{
		first.flush(mode);
		flush<mode>(std::forward<Args>(args)...);
	}


	// ----------------------------------------------------------------
	// ConditionalIndexForwarder utility structure
	// ----------------------------------------------------------------

	template<bool indexed, bool randomized, typename Func>
	struct ConditionalIndexForwarder
	{
		using Ret = typename return_type<Func>::type;

		// Forward index, do not forward random

		template<typename Index, typename Rnd, typename... CallArgs,
			REQUIRES(is_skepu_index<Index>::value && indexed && !randomized)>
		static Ret forward(Func func, Index i, Rnd &&r, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}

		// Do not forward index, do not forward random

		template<typename Index, typename Rnd, typename... CallArgs,
			REQUIRES(is_skepu_index<Index>::value && !indexed && !randomized)>
		static Ret forward(Func func, Index, Rnd &&r, CallArgs&&... args)
		{
			return func(std::forward<CallArgs>(args)...);
		}

		// Forward index, forward random

		template<typename Index, typename Rnd, typename... CallArgs,
			REQUIRES(is_skepu_index<Index>::value && indexed && randomized)>
		static Ret forward(Func func, Index i, Rnd &&r, CallArgs&&... args)
		{
			return func(i, std::forward<Rnd>(r), std::forward<CallArgs>(args)...);
		}

		// Do not forward index, forward random

		template<typename Index, typename Rnd, typename... CallArgs,
			REQUIRES(is_skepu_index<Index>::value && !indexed && randomized)>
		static Ret forward(Func func, Index, Rnd &&r, CallArgs&&... args)
		{
			return func(std::forward<Rnd>(r), std::forward<CallArgs>(args)...);
		}
	};

}



namespace skepu_variadic_return
{
	template <typename T>
	struct is_tuple_impl : std::false_type {};

	template <typename... U>
	struct is_tuple_impl<std::tuple <U...>> : std::true_type {};

	template <typename T>
	struct is_tuple : is_tuple_impl<typename std::decay<T>::type> {};

	template<typename ... Out, typename ... Res>
	inline void bind(std::tuple<Out...> &&out, std::tuple<Res...> res) noexcept
	{
		out = res;
	}

	template<typename Out, typename Res, REQUIRES(!is_tuple<Res>::value)>
	inline void bind(std::tuple<Out> &&out, Res && res) noexcept
	{
		std::get<0>(out) = res;
	}


	template <typename ... Out, typename ... Res, std::size_t ... Is>
	void sumT(std::tuple<Out...> &&out, std::tuple<Res...> const &res, pack_indices<Is...> const &)
	{
		pack_expand((std::get<Is>(out) += std::get<Is>(res), 0)...);
	}

	template<typename ... Out, typename ... Res>
	inline void bindadd(std::tuple<Out...> &&out, std::tuple<Res...> const &res) noexcept
	{
		static_assert(sizeof...(Out) == sizeof...(Res), "Error in tuple return system");
		sumT(std::forward<std::tuple<Out...>>(out), res, typename make_pack_indices<sizeof...(Out), 0>::type{});
	}

	template<typename Out, typename Res, REQUIRES(!is_tuple<Res>::value)>
	inline void bindadd(std::tuple<Out> &&out, Res && res) noexcept
	{
		std::get<0>(out) += res;
	}


	template<typename...Ts>
	auto my_make_tuple(std::tuple<Ts...> &arg) -> std::tuple<Ts...>&
	{
		return arg;
	}

	template<typename T>
	std::tuple<T&> my_make_tuple(T &&arg)
	{
		return std::tie(arg);
	}
}

#ifndef SKEPU_VARIADIC_RETURN_IMPL
#define SKEPU_VARIADIC_RETURN_IMPL 1
#endif

#if SKEPU_VARIADIC_RETURN_IMPL == 0
#define SKEPU_VARIADIC_RETURN(lhs, rhs) std::tie(lhs) = (rhs);
#elif SKEPU_VARIADIC_RETURN_IMPL == 1
#define SKEPU_VARIADIC_RETURN(lhs, rhs) skepu_variadic_return::bind(std::tie(lhs), (rhs));

#define SKEPU_VARIADIC_RETURN_INPLACE(inplace, lhs, rhs)\
if (!inplace) \
{ skepu_variadic_return::bind(std::tie(lhs), (rhs)); } \
else \
{ skepu_variadic_return::bindadd(std::tie(lhs), (rhs)); } \




#else
#define SKEPU_VARIADIC_RETURN(lhs, rhs) std::tie(lhs) = skepu_variadic_return::my_make_tuple(rhs);


#define SKEPU_VARIADIC_RETURN_INPLACE(inplace, lhs, rhs) \
	if (!inplace) \
	{ std::tie(lhs) = skepu_variadic_return::my_make_tuple(rhs); } \
	else \
	{ std::tie(lhs) += skepu_variadic_return::my_make_tuple(rhs); } \

#endif
