#ifndef MAT_HPP
#define MAT_HPP

#include <skepu3/cluster/starpu_matrix_container.hpp>
#include <skepu3/cluster/matrix.hpp>
#include <cstddef>

namespace skepu
{
	template<typename T>
	struct Mat
	{
		using ContainerType = Matrix<T>;
		T* data;
		size_t rows;
		size_t cols;
		size_t ld;
		size_t size;
		inline const T & operator[](size_t index) const;
		inline T & operator[](const size_t index);
		inline T & operator()(const size_t row, const size_t col);
		Mat(void* buffer);
		Mat() = default;
		int offset(const Offset2D && offset);
	};
	namespace helpers
	{
		// We need the "unwrapped" types later on, so here is some templates
		// to help with that.
		template <typename T>
		struct mat_tuple_to_raw_type_tuple
		{
			static constexpr typename
			make_pack_indices<std::tuple_size<T>::value, 0>::type
			is{};

		private:
			template <size_t... Is>
			static auto unwrap_impl(pack_indices<Is...>)
				-> decltype (std::make_tuple(*(std::get<Is>(T{}).data)...))
				{
					return std::make_tuple(*(std::get<Is>(T{}).data)...);
				}

			static auto unwrap() -> decltype(unwrap_impl(is))
				{
					return unwrap_impl(is);
				}
		public:
			using unwrapped = decltype(unwrap());
		};
	}
}

#include <skepu3/cluster/impl/mat.inl>

#endif /* MAT_HPP */
