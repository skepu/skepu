#ifndef TASK_CREATOR_HPP
#define TASK_CREATOR_HPP

#include <skepu3/cluster/container_cut.hpp>
#include <skepu3/cluster/index.hpp>
#include <starpu.h>

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
			                        ContainerCuts & ...container_cuts);
		}
	}
}


#include <skepu3/cluster/impl/task_creator.inl>

#endif /* TASK_CREATOR_HPP */
