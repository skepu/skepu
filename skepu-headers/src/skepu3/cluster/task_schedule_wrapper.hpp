#ifndef TASK_SCHEDULE_WRAPPER_HPP
#define TASK_SCHEDULE_WRAPPER_HPP

#include <starpu.h>
#include <mpi.h>
#include <skepu3/impl/meta_helpers.hpp>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			template<typename... M, typename... H, typename... C>
			void schedule_task(struct starpu_codelet* cl,
			                   std::tuple<M...>& modes,
			                   std::tuple<H...>& handles,
			                   std::tuple<C...>& constants);

			template<typename... M,
			         typename... H,
			         typename... C,
			         size_t... HIs,
			         size_t... CIs>
			void schedule_task_impl(struct starpu_codelet* cl,
			                        std::tuple<M...>& modes,
			                        std::tuple<H...>& handles,
			                        std::tuple<C...>& constants,
			                        pack_indices<HIs...>,
			                        pack_indices<CIs...>);

			template<typename ArgsT, size_t... Is>
			void schedule_task_impl(struct starpu_codelet* cl,
			                        ArgsT&& args,
			                        pack_indices<Is...>);



		} // helpers
	} // cluster
} // skepu

#include <skepu3/cluster/impl/task_schedule_wrapper.inl>


#endif /* TASK_SCHEDULE_WRAPPER_HPP */
