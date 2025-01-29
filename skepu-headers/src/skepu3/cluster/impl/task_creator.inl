#ifndef TASK_CREATOR_INL
#define TASK_CREATOR_INL

#include <skepu3/cluster/task_creator.hpp>
#include <skepu3/cluster/handle_modes.hpp>
#include <skepu3/cluster/task_schedule_wrapper.hpp>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			template <typename... ContainerCuts,
			          typename... AnyHandles,
			          typename... UniformArgs>
			void
			create_tasks_one_to_one(struct starpu_codelet & cl,
			                        const Size2D & total_size,
			                        std::tuple<AnyHandles...> & any_args,
			                        std::tuple<UniformArgs...> & uniform_args,
			                        ContainerCuts & ...container_cuts)
			{
				Index2D current {};
				while(current.row < total_size.row)
				{
					const size_t max_rows {
						std::max<size_t>(container_cuts.block_height_from(current)...) };

					while(current.col < total_size.col)
					{
						auto tc =
							create_task_cut(container_cuts.largest_cut_from(current)...);
						assert(tc.size.row == max_rows);

						auto task_meta =
							std::forward_as_tuple(tc.task_size, tc.task_offsets);
						schedule_task(cl,
						              modes_from_codelet(cl),
						              std::tuple_cat(tc.handles, any_args),
						              std::tuple_cat(task_meta, uniform_args));


						current.col += tc.size.col;
					}
					current.row += max_rows;
				}
			}

		} // helpers
	} // cluster
} // skepu

#endif /* TASK_CREATOR_INL */
