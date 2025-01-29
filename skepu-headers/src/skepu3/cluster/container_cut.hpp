#ifndef CONTAINER_CUT_HPP
#define CONTAINER_CUT_HPP

#include <skepu3/cluster/index.hpp>
#include <skepu3/cluster/starpu_matrix_container.hpp>
#include <skepu3/cluster/handle_cut.hpp>
#include <starpu.h>
#include <tuple>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			/**
			 * @brief Since arrays are a tad problematic when given as
			 * function arguments, wrap it in a struct.
			 */
			struct starpu_handle_offsets
			{
				Offset2D o[STARPU_NMAXBUFS] = {};
				template<typename... Args, size_t... Is>
				void starpu_handle_offsets_impl(const std::tuple<Args...> & args,
				                                pack_indices<Is...>);
				template<typename... Args>
				starpu_handle_offsets(const std::tuple<Args...> & os);
				starpu_handle_offsets() {};
			};

			/**
			 * @brief A task_cut represents aligned parts of multiple
			 * handles, this is data that should be scheduled together
			 */
			template<typename... Handles>
			struct task_cut
			{
				Size2D task_size;
				std::tuple<Handles...> task_handles;
				starpu_handle_offsets task_offsets;
				template<typename... Offsets>
				task_cut(Size2D size,
				         std::tuple<Handles...> handles,
				         std::tuple<Offsets...> offsets);
			};

			/**
			 * @brief Create a set of handles and offsets that may be used
			 * within the task creation in the skeletons.
			 *
			 * @return task_cut<std::tuple<starpu_data_handle_t...>,
			 *                             std::tuple<Offset2D...>>
			 */
			template<typename... HandleCuts>
			auto create_task_cut(HandleCuts ...args)
				-> task_cut<typename
				            std::remove_reference<decltype(args.handle)>::type ...>;
		} // helpers
	} // cluster
} // skepu


#include <skepu3/cluster/impl/container_cut.inl>

#endif /* CONTAINER_CUT_HPP */
