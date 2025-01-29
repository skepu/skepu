#ifndef HELPERS_INL
#define HELPERS_INL


#include <skepu3/cluster/helpers.hpp>

#include <skepu3/impl/meta_helpers.hpp>
#include <tuple>

// Various helpers for the cluster implementation

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			template<typename Ret, typename Tuple, size_t ...Is>
			Ret unpack_tuple(const Tuple & t, pack_indices<Is...>)
			{
				Ret res = {std::get<Is>(t)...};
				return res;
			}

			/**
			 * @brief Unpack a tuple into some other type.
			 */
			template<typename Ret, typename Tuple>
			Ret unpack_tuple(const Tuple & t)
			{
				constexpr typename
					make_pack_indices<std::tuple_size<Tuple>::value>::type is{};
				return unpack_tuple<Ret>(t, is);
			}


			/**
			 * @brief Set the codelet mode to STARPU_R for each handle
			 *
			 */
			template<size_t... Is>
			void set_codelet_read_only_modes(pack_indices<Is...>, starpu_codelet& cl)
			{
				pack_expand(cl.modes[Is] = STARPU_R ...);
			}
		}
	}
}


#endif /* HELPERS_INL */
