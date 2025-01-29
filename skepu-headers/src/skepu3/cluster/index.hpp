#ifndef INDEX_HPP
#define INDEX_HPP

#include <cstddef>
#include <skepu3/impl/meta_helpers.hpp>

namespace skepu
{
	struct Index2D
	{
		size_t row;
		size_t col;
		size_t i;
		inline bool transpose(); // bool, so it can be used in pack_expand

		Index2D() noexcept = default;

		Index2D(size_t _row, size_t _col, size_t _i = 0) noexcept
		: row{_row}, col{_col}, i{_i}
		{}

		Index2D(Index2D const & other) noexcept
		: row{other.row}, col{other.col}, i{other.i}
		{}
	};
	inline Index2D operator+(const Index2D & lhs, const Index2D & rhs);


	// Because both skepu::Vector<T> and skepu::Matrix<T> use the
	// same data type, skepu::cluster::starpu_matrix_container<T>, as
	// their storage backend, they also use the same indexing.

	// IMPORTANT: Often in the implementation, only the relevant fields
	// are set. For example, a function returning a Index2D might not
	// set the `i` member correctly, while user functions expecting an
	// Index1D might not get a valid `row` field. Why?  Confusion. But
	// mostly because I think I forgot to add it in a few places.

	using Index1D = Index2D;
	using Size1D = Index2D;
	using Size2D = Index2D;
	using Offset1D = Index2D;
	using Offset2D = Index2D;



	// ConditionalIndexForwarder, partial copy of contents in
	// skepu/impl/common.hpp. It needs to be either copied or be
	// subjected to far to many ifdefs, due to the above definition of
	// Index1D.

	// ----------------------------------------------------------------
	// is_skepu_index trait class
	// ----------------------------------------------------------------

	template<typename T>
	struct is_skepu_index: std::false_type{};

	template<>
	struct is_skepu_index<Index2D>: std::true_type{};


	template<typename... Args>
	struct is_indexed
		: bool_constant<is_skepu_index<
			                typename pack_element<0, Args...>::type>::value> {};

	template<>
	struct is_indexed<>
		: std::false_type{};


	// ----------------------------------------------------------------
	// ConditionalIndexForwarder utility structure
	// ----------------------------------------------------------------

	// Implementations are left here, moving them to the .inl-file
	// results in somewhat horrific code.

	template<bool indexed, typename Func>
	struct ConditionalIndexForwarder
	{
		using Ret = typename return_type<Func>::type;

		template<typename... CallArgs>
		static Ret forward(Func func, Index2D i, CallArgs&&... args)
			{
				return func(i, std::forward<CallArgs>(args)...);
			}
	};

	template<typename Func>
	struct ConditionalIndexForwarder<false, Func>
	{
		using Ret = typename return_type<Func>::type;

		template<typename... CallArgs>
		static Ret forward(Func func, Index2D, CallArgs&&... args)
			{
				return func(std::forward<CallArgs>(args)...);
			}

		template<typename... CallArgs>
		static Ret forward_device(Func func, Index2D, CallArgs&&... args)
			{
				return func(std::forward<CallArgs>(args)...);
			}
	};
} // skepu

#include <skepu3/cluster/impl/index.inl>

#endif /* INDEX_HPP */
