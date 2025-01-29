#pragma once
#ifndef SKEPU_CLUSTER_COMMON_HPP
#define SKEPU_CLUSTER_COMMON_HPP 1

#define REQUIRES(...) typename std::enable_if<(__VA_ARGS__), bool>::type = 0
#define REQUIRES_VALUE(...) \
	typename std::enable_if<__VA_ARGS__::value, bool>::type = 0
#define REQUIRES_DEF(...) typename std::enable_if<(__VA_ARGS__), bool>::type

#define MAX_SIZE ((size_t)-1)

#define VARIANT_OPENCL(block)
#define VARIANT_CUDA(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CPU(block) block

#include <vector>
#include <iostream>
#include <utility>
#include <cassert>
#include <algorithm>
#include <functional>

#include "../impl/backend.hpp"
#include "../impl/meta_helpers.hpp"
#include "flush_mode.hpp"

namespace skepu
{
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
	};

auto inline
make_index(
	std::integral_constant<int, 1>,
	size_t index,
	size_t,
	size_t,
	size_t)
-> Index1D
{
	return Index1D{index};
}

auto inline
make_index(
	std::integral_constant<int, 2>,
	size_t index,
	size_t size_j,
	size_t,
	size_t)
-> Index2D
{
	return Index2D{ index / size_j, index % size_j };
}

auto inline
make_index(
	std::integral_constant<int, 3>,
	size_t index,
	size_t size_j,
	size_t size_k,
	size_t)
-> Index3D
{
	size_t ci = index / (size_j * size_k);
	index = index % (size_j * size_k);
	size_t cj = index / (size_k);
	index = index % (size_k);
	return Index3D{ ci, cj, index };
}

auto inline
make_index(
	std::integral_constant<int, 4>,
	size_t index,
	size_t size_j,
	size_t size_k,
	size_t size_l)
-> Index4D
{
	size_t ci = index / (size_j * size_k * size_l);
	index = index % (size_j * size_k * size_l);
	size_t cj = index / (size_k * size_l);
	index = index % (size_k * size_l);
	size_t ck = index / (size_l);
	index = index % (size_l);
	return Index4D{ ci, cj, ck, index };
}

auto inline
operator<<(std::ostream &o, Index1D idx)
-> std::ostream &
{
	return o << "Index1D(" << idx.i << ")";
}

auto inline
operator<<(std::ostream &o, Index2D idx)
-> std::ostream &
{
	return o << "Index2D(" << idx.row << ", " << idx.col << ")";
}

auto inline
operator<<(std::ostream &o, Index3D idx)
-> std::ostream &
{
	return o << "Index3D(" << idx.i << ", "  << idx.j << ", " << idx.k << ")";
}

auto inline
operator<<(std::ostream &o, Index4D idx)
-> std::ostream &
{
	return o
		<< "Index4D("
		<< idx.i
		<< ", "
		<< idx.j
		<< ", "
		<< idx.k
		<< ", "
		<< idx.l
		<< ")";
}



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

/* To be able to use getParent on containers. Those are private in MPI. */
struct cont
{
	template<typename Container>
	static auto
	getParent(Container && c)
	-> decltype(c.getParent())
	{
		return c.getParent();
	}
};

	// For multiple return Map variants
	template<typename... args>
	using multiple = std::tuple<args...>;

	template <typename... Args>
	auto ret(Args&&... args)
	-> decltype(std::make_tuple(std::forward<Args>(args)...))
	{
		return std::make_tuple(std::forward<Args>(args)...);
	}

	inline size_t elwise_i(std::tuple<>) { return 0; }
	inline size_t elwise_j(std::tuple<>) { return 0; }
	inline size_t elwise_k(std::tuple<>) { return 0; }
	inline size_t elwise_l(std::tuple<>) { return 0; }

	template<typename... Args>
	inline size_t elwise_i(std::tuple<Args...> &t)
 	{
		return cont::getParent(std::get<0>(t)).size_i();
 	}

 	template<typename... Args>
	inline size_t elwise_j(std::tuple<Args...> &t)
 	{
		return cont::getParent(std::get<0>(t)).size_j();
 	}

	template<typename... Args>
	inline size_t elwise_k(std::tuple<Args...> &t)
	{
		return cont::getParent(std::get<0>(t)).size_k();
	}

	template<typename... Args>
	inline size_t elwise_l(std::tuple<Args...> &t)
	{
		return cont::getParent(std::get<0>(t)).size_l();
	}

	// ----------------------------------------------------------------
	// is_skepu_{vector|matrix|container} trait classes
	// ----------------------------------------------------------------

	template<typename T>
	struct is_skepu_vector: std::false_type {};

	template<typename T>
	struct is_skepu_matrix: std::false_type {};

	template<typename T>
	struct is_skepu_tensor3: std::false_type {};

	template<typename T>
	struct is_skepu_tensor4: std::false_type {};

	template<typename T>
	struct is_skepu_container:
		std::integral_constant<bool,
			is_skepu_vector<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_matrix<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_tensor3<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_tensor4<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value> {};

	/** Check that all types in a parameter pack are SkePU containers. */
	template<typename ...> struct are_skepu_containers;

	/* Empty pack is true. */
	template<> struct are_skepu_containers<> : public std::true_type {};

	/* Check that first is a SkePU container and recurse the rest. */
	template<typename CAR, typename ... CDR>
	struct are_skepu_containers<CAR, CDR...>
	: std::integral_constant<bool,
			is_skepu_container<CAR>::value
			&& are_skepu_containers<CDR...>::value>
	{};

	template<typename T>
	struct is_skepu_vector_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_matrix_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_tensor3_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_tensor4_proxy: std::false_type {};

	template<typename T>
	struct is_skepu_container_proxy:
		std::integral_constant<bool,
			is_skepu_vector_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_matrix_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_tensor3_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_tensor4_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value>  {};

	// ----------------------------------------------------------------
	// is_skepu_iterator trait class
	// ----------------------------------------------------------------

	template<typename T, typename Ret>
	struct is_skepu_iterator: std::false_type {};

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
	struct is_indexed
	: bool_constant<is_skepu_index<
			typename first_element<Args...>::type>::value>{};

 	template<>
	struct is_indexed<>: std::false_type{};

	// ----------------------------------------------------------------
	// matrix row proxy trait class
	// ----------------------------------------------------------------

	template<typename T>
	struct proxy_tag {
		using type = ProxyTag::Default;
	};

	// ----------------------------------------------------------------
	// smart container size extractor
	// ----------------------------------------------------------------

	inline std::tuple<size_t>
	size_info(index_dimension<skepu::Index1D>, size_t i, size_t, size_t, size_t)
	{
		return {i};
	}

	inline std::tuple<size_t, size_t>
	size_info(index_dimension<skepu::Index2D>, size_t i, size_t j, size_t, size_t)
	{
		return {i, j};
	}

	inline std::tuple<size_t, size_t, size_t>
	size_info(
		index_dimension<skepu::Index3D>, size_t i, size_t j, size_t k, size_t)
	{
		return {i, j, k};
	}

	inline std::tuple<size_t, size_t, size_t, size_t>
	size_info(
		index_dimension<skepu::Index4D>, size_t i, size_t j, size_t k, size_t l)
	{
		return {i, j, k, l};
	}

	template<typename Index, typename... Args>
	inline auto
	size_info(Index, size_t, size_t, size_t, size_t, Args&&... args)
	-> decltype(get<0, Args...>(args...).getParent().size_info())
	{
		return get<0, Args...>(args...).getParent().size_info();
	}

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

 	template<bool indexed, typename Func>
 	struct ConditionalIndexForwarder
 	{
 		using Ret = typename return_type<Func>::type;

		// Forward index

		template<typename Index, typename... CallArgs, REQUIRES(is_skepu_index<Index>::value && indexed)>
		static Ret forward(Func func, Index i, CallArgs&&... args)
 		{
 			return func(i, std::forward<CallArgs>(args)...);
 		}

		template<typename Index, typename... CallArgs, REQUIRES(is_skepu_index<Index>::value && indexed)>
		static Ret forward_device(Func func, Index i, CallArgs&&... args)
 		{
 			return func(i, std::forward<CallArgs>(args)...);
 		}

		// Do not forward index

		template<typename Index, typename... CallArgs, REQUIRES(is_skepu_index<Index>::value && !indexed)>
		static Ret forward(Func func, Index, CallArgs&&... args)
 		{
 			return func(std::forward<CallArgs>(args)...);
 		}

		template<typename Index, typename... CallArgs, REQUIRES(is_skepu_index<Index>::value && !indexed)>
		static Ret forward_device(Func func, Index, CallArgs&&... args)
 		{
 			return func(std::forward<CallArgs>(args)...);
 		}
 	};

	template<typename T>
	struct base_type
	{
		typedef T type;
	};

	template<typename T>
	struct base_type<T &&>
	{
		typedef T type;
	};

} // namespace skepu

#endif // SKEPU_CLUSTER_COMMON_HPP
