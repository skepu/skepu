#ifndef HANDLE_MODES_HPP
#define HANDLE_MODES_HPP

#include <starpu.h>
#include <tuple>
#include <skepu3/impl/meta_helpers.hpp>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{

			constexpr typename
			make_pack_indices<STARPU_NMAXBUFS>::type starpu_maxbufs_indice_pack {};

			template<size_t I>
			starpu_data_access_mode random_access_mode_impl();

			template<size_t ...Is>
			auto random_access_mode_impl(pack_indices<Is...>)
				-> decltype(std::make_tuple(random_access_mode_impl<Is>()...));

			template<typename... Hs>
			auto random_access_mode(const std::tuple<Hs...> &&)
				-> decltype(
					random_access_mode_impl(
						make_pack_indices<std::tuple_size<Hs...>::value>::type));

			template<size_t I>
			starpu_data_access_mode
			modes_from_codelet_impl(const starpu_codelet & cl);

			template<size_t ...Is>
			auto modes_from_codelet_impl(const starpu_codelet & cl,
			                             pack_indices<Is...>)
				-> decltype(
					std::make_tuple(modes_from_codelet_impl<Is>(cl)...));

			/**
			 * @brief Return a tuple of the modes in the codelet.
			 *
			 *   ***The tuple may be longer than `cl.nbuffers`***
			 */
			inline auto modes_from_codelet(const starpu_codelet & cl)
				-> decltype(
					modes_from_codelet_impl(cl, starpu_maxbufs_indice_pack));
		}
	}
}
#include <skepu3/cluster/impl/handle_modes.inl>

#endif /* HANDLE_MODES_HPP */
