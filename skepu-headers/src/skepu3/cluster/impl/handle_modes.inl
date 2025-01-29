#ifndef HANDLE_MODES_INL
#define HANDLE_MODES_INL

#include <skepu3/cluster/handle_modes.hpp>

#include <starpu.h>
#include <tuple>
#include <skepu3/impl/meta_helpers.hpp>


namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			template<size_t I>
			starpu_data_access_mode random_access_mode_impl()
			{
				return STARPU_R;
			}

			template<size_t ...Is>
			auto random_access_mode_impl(pack_indices<Is...>)
				-> decltype(std::make_tuple(random_access_mode_impl<Is>()...))
			{
				return std::make_tuple(random_access_mode_impl<Is>()...);
			}

			template<typename... Hs>
			auto random_access_mode(const std::tuple<Hs...> &&)
				-> decltype(
					random_access_mode_impl(
						make_pack_indices<std::tuple_size<Hs...>::value>::type))
			{
				constexpr typename
					make_pack_indices<std::tuple_size<Hs...>::value>::type is{};
				return random_access_mode_impl(is);
			}

			template<size_t I>
			starpu_data_access_mode modes_from_codelet_impl(const starpu_codelet & cl)
			{
				return cl.modes[I];
			}

			template<size_t ...Is>
			auto modes_from_codelet_impl(const starpu_codelet & cl,
			                             pack_indices<Is...>)
				-> decltype(
					std::make_tuple(modes_from_codelet_impl<Is>(cl)...))
			{
				return std::make_tuple(modes_from_codelet_impl<Is>(cl)...);
			}

			inline auto modes_from_codelet(const starpu_codelet & cl)
				-> decltype(
					modes_from_codelet_impl(cl, starpu_maxbufs_indice_pack))
			{
				return modes_from_codelet_impl(cl, starpu_maxbufs_indice_pack);
			}
		}
	}
}

#endif /* HANDLE_MODES_INL */
