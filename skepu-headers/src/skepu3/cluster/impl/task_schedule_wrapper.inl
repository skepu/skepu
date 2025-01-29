#ifndef TASK_SCHEDULE_WRAPPER_INL
#define TASK_SCHEDULE_WRAPPER_INL

#include <skepu3/cluster/task_schedule_wrapper.hpp>

#include <starpu.h>
#include <mpi.h>
#include <skepu3/impl/meta_helpers.hpp>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			/**
			 * @brief Schedule a codelet with the given parameters.
			 *
			 * @param  starpu_codelet*&& cl
			 * @param  modes std::tuple<starpu_data_access_mode...>
			 * @param  handles std::tuple<starpu_data_handle_t...>
			 * @param  constants std::tuple<T...>
			 */
			template<typename ...M, typename ...H, typename ...C>
			void schedule_task(struct starpu_codelet* cl,
			                   std::tuple<M...>& modes,
			                   std::tuple<H...>& handles,
			                   std::tuple<C...>& constants)
			{
				// This function just creates some pack_indices so it's
				// possible to extract data from the tuples
				constexpr typename
					make_pack_indices<
						std::tuple_size<
							typename std::tuple<H...>>::value>::type is{};

				constexpr typename
					make_pack_indices<
						std::tuple_size<
							typename std::tuple<C...>>::value>::type cs{};

				schedule_task_impl(cl,
				                   modes,
				                   handles,
				                   constants,
				                   is, cs);
			}

			/**
			 * Helper function that throws around the arguments to get the
			 * correct ordering. A tad of template magic.
			 */
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
			                        pack_indices<CIs...>)
			{
				using res_type = decltype(std::tuple_cat(
					                          std::forward_as_tuple(
						                          std::get<HIs>(modes),
						                          std::get<HIs>(handles))...,
					                          std::forward_as_tuple(
						                          STARPU_VALUE,
						                          std::get<CIs>(constants),
						                          sizeof(std::get<CIs>(constants)))...));

				constexpr typename
					make_pack_indices<std::tuple_size<res_type>::value>::type is{};

				// Unpack and reorder the arguments to interleave the handles
				// and modes, then interleave the constants with STARPU_VALUE
				// and their size. This leaves us with a single tuple, passed
				// on to the next function.
				schedule_task_impl(
					cl,
					std::tuple_cat(
						std::forward_as_tuple(std::get<HIs>(modes),
						                      std::get<HIs>(handles))...,
						std::forward_as_tuple(STARPU_VALUE,
						                      &std::get<CIs>(constants),
						                      sizeof(std::get<CIs>(constants)))...),
					is);
			}


			/**
			 * Helper function that does the actual scheduling after the
			 * intermediate template magic step
			 */
			template<typename ArgsT, size_t... Is>
			void schedule_task_impl(struct starpu_codelet* cl,
			                        ArgsT&& args,
			                        pack_indices<Is...>)
			{
				starpu_mpi_task_insert(MPI_COMM_WORLD, cl, std::get<Is>(args)..., 0);
			}
		} // helpers
	} // cluster
} // skepu

#endif /* TASK_SCHEDULE_WRAPPER_INL */
