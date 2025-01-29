#ifndef CONTAINER_CUT_INL
#define CONTAINER_CUT_INL

#include <algorithm>

#include <skepu3/cluster/container_cut.hpp>
#include <skepu3/cluster/helpers.hpp>
#include <skepu3/impl/meta_helpers.hpp>
#include <skepu3/cluster/starpu_matrix_container.hpp>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{

			template<typename... Args, size_t... Is>
			void starpu_handle_offsets
			::starpu_handle_offsets_impl(const std::tuple<Args...> & args,
			                             pack_indices<Is...>)
			{
				pack_expand(o[Is] = std::get<Is>(args)...);
			}

			template<typename... Args>
			/**
			 * @brief Initialize the offset array
			 */
			starpu_handle_offsets
			::starpu_handle_offsets(const std::tuple<Args...> & args) {
				constexpr typename
					make_pack_indices<std::tuple_size<
						typename std::tuple<Args...>>::value>::type is{};
				starpu_handle_offsets_impl(args, is);
			}


			/**
			 * @brief Create a `task_cut`
			 */
			template<typename... Handles>
			template<typename... Offsets>
			task_cut<Handles...>
			::task_cut(Size2D size,
			           std::tuple<Handles...> handles,
			           std::tuple<Offsets...> offsets)
			{
				task_size = size;
				task_handles = handles;
				task_offsets = unpack_tuple<decltype(offsets)>(offsets);
			}


			/**
			 * @brief Create a `task_cut` from a set of `handle_cut`
			 *
			 * Given the handles and their internal offsets, the largest
			 * possible cut will be generated. At the same time, generate
			 * the internal offsets, to be used within codelet
			 * implementations.
			 *
			 */
			template<typename... HandleCuts>
			auto create_task_cut(const Size2D & diff, HandleCuts... args)
				-> task_cut<typename
				            std::remove_reference<decltype(args.handle)>::type ...>
			{
				// Calculate the maximum possible task size
				Size2D size;
				size.row = std::min<size_t>({args.local_size.row ... , diff.row});
				size.col = std::min<size_t>({args.local_size.col ... , diff.col});
				size.i = size.col;

				// Assemble a result
				return task_cut<typename
				                std::remove_reference<decltype(args.handle)>::type ...>(
					                size,
					                std::make_tuple(args.handle...),
					                std::make_tuple(args.local_offset...));
			}
		}
	}
}


#endif /* CONTAINER_CUT_INL */
