#ifndef MULTIVARIANT_TASK_INL
#define MULTIVARIANT_TASK_INL

#include <algorithm>

#include <skepu3/cluster/multivariant_task.hpp>
#include <skepu3/cluster/container_cut.hpp>
#include <skepu3/cluster/handle_modes.hpp>
#include <skepu3/cluster/helpers.hpp>
#include <skepu3/cluster/task_schedule_wrapper.hpp>
#include <climits>

namespace skepu
{
	namespace cluster
	{
		template<typename ResultArgs,
		         typename ElwiseArgs,
		         typename ContainerArgs,
		         typename UniformArgs,
		         typename Self>
		multivariant_task<ResultArgs,
		                  ElwiseArgs,
		                  ContainerArgs,
		                  UniformArgs,
		                  Self>
		::multivariant_task()
		{
			// Initialize performance model
			memset(&perf_model, 0, sizeof(starpu_perfmodel));
			starpu_perfmodel_init(&perf_model);
			perf_model.type = STARPU_HISTORY_BASED;
			// Not really used for now.
			perf_model.symbol = __PRETTY_FUNCTION__;

			starpu_codelet_init(&cl);
			cl.nbuffers = n_handles;
			cl.max_parallelism = INT_MAX;
			cl.type = STARPU_FORKJOIN; // For OpenMP
			cl.where = STARPU_CPU;
			cl.cpu_funcs[0] = cpu_starpu_func;
			cl.cpu_funcs_name[0] = __PRETTY_FUNCTION__;
			cl.modes[0] = STARPU_RW;
			/* Performance model not neede?
			 * Creates a problem with starpu_shutdown if used with MPI.
			 */
			//cl.model = &perf_model;

			helpers::set_codelet_read_only_modes(handle_indices, cl);
			for(size_t i {}; i < n_result; ++i)
			{
				cl.modes[i] = STARPU_RW;
			}
		}


		/**
		 * @brief This function performs some type magic in order to go
		 * from void** (but really "starpu_data_handle_t*"-ish), to
		 * something we can pass on to cpu.
		 *
		 * The following type is *required* for cpu:
		 *	template<typename MatT,
		 *	         size_t... RI,
		 *	         size_t... EI,
		 *	         size_t... CI,
		 *	         typename... Uniform>
		 *	static void cpu(const void * self,
		 *	                Size2D size,
		 *	                Offset2D global_offset,
		 *	                MatT && bufs,
		 *	                pack_indices<RI...>,
		 *	                pack_indices<EI...>,
		 *	                pack_indices<CI...>,
		 *	                Uniform... args);
		 *
		 * MatT will be a tuple containing a Mat<T>, created from each
		 * handle. The raw data pointers within each Mat<T> will be offset
		 * by some number of rows and columns, specified in a
		 * helpers::starpu_handle_offsets, passed through `void*
		 * argv`. Thus, cpu-implementations need not consern
		 * themselves with handle alignment, they may consider all buffers
		 * aligned.
		 */
		template<typename ResultArgs,
		         typename ElwiseArgs,
		         typename ContainerArgs,
		         typename UniformArgs,
		         typename Self>
		template<size_t... HI,
		         size_t... RI,
		         size_t... EI,
		         size_t... CI,
		         size_t... UI>
		void multivariant_task<ResultArgs,
		                       ElwiseArgs,
		                       ContainerArgs,
		                       UniformArgs,
		                       Self>
		::cpu_starpu_redirect(void** buffers,
		                      void* args,
		                      pack_indices<HI...>,
		                      pack_indices<RI...>,
		                      pack_indices<EI...>,
		                      pack_indices<CI...>,
		                      pack_indices<UI...>)
		{
			// Unpack uniform arguments and meta information
			UniformArgs uniform_args;
			task_data td;
			helpers::extract_constants(args, td,
			                           std::get<UI>(uniform_args)...);

			// Define the type of a tuple containing Mat<T> for the "type"
			// of each handle. (Why here and not in the header? Didn't find
			// a nice way to do it, pls fix.)
			using HandleMatT =
				decltype(
					std::make_tuple(
						Mat<typename std::tuple_element<HI, HandleT>::type>(NULL)...));
			// And then populate it via the raw buffers
			HandleMatT mats {buffers[HI]...};

			// Using the offsets, offset the pointers within each Mat<T> to
			// the correct location, so everything is aligned properly
			// within Self::cpu.
			pack_expand(
				std::get<HI>(mats).offset(std::forward<Offset2D>(td.offsets.o[HI]))...
				);

			// Finally, call the function.
			Self::cpu(td.self, td.size, td.index,
			          std::forward<HandleMatT>(mats),
			          result_handle_indices,
			          elwise_handle_indices,
			          container_handle_indices,
			          std::get<UI>(uniform_args)...);
		}


		/**
		 * @brief Add some pack expands and forward everything on to
		 * cpu_starpu_redirect
		 */
		template<typename ResultArgs,
		         typename ElwiseArgs,
		         typename ContainerArgs,
		         typename UniformArgs,
		         typename Self>
		void multivariant_task<ResultArgs,
		                       ElwiseArgs,
		                       ContainerArgs,
		                       UniformArgs,
		                       Self>
		::cpu_starpu_func(void** buffers, void* args)
		{
			cpu_starpu_redirect(buffers,
			                    args,
			                    handle_indices,
			                    result_handle_indices,
			                    elwise_handle_indices,
			                    container_handle_indices,
			                    ui);
		}

		template<typename ResultArgs,
		         typename ElwiseArgs,
		         typename ContainerArgs,
		         typename UniformArgs,
		         typename Self>
		template<typename... Args,
		         size_t... RI,
		         size_t... EI,
		         size_t... CI,
		         size_t... UI>
		void multivariant_task<ResultArgs,
		                       ElwiseArgs,
		                       ContainerArgs,
		                       UniformArgs,
		                       Self>
		::element_aligned_impl(pack_indices<RI...>,
		                       pack_indices<EI...>,
		                       pack_indices<CI...>,
		                       pack_indices<UI...>,
		                       Size2D total_size,
		                       Args & ...args)
		{
			auto modes = helpers::modes_from_codelet(cl);

			Index2D i {};
			while (i.row < total_size.row)
			{
				// Reset column counter
				i.col = 0;
				// We need to know how much work we have left to do.
				Size2D diff = total_size;
				diff.row -= i.row;
				diff.col -= i.col;
				diff.i -= i.i;

				// Find the maximum number of rows we can schedule at a time
				const size_t max_rows {
					std::min<size_t>({
							get<RI>(args...).block_height_from(i)...,
							get<EI>(args...).block_height_from(i)...,
							diff.row})
						};

				// Invalidate writable starpu_matrix_containers
				pack_expand(get<RI>(args...)
				            .getParent()
				            .data()
				            .invalidate_unpartition()...,
				            0);

				auto container_handles =
					std::make_tuple(get<CI>(args...).data().allgather()...);

				while (i.col < total_size.col)
				{
					i.i = i.col;
					diff.col = total_size.col - i.col;
					auto tc = helpers::create_task_cut(diff,
					                                   get<RI>(args...).largest_cut_from(i)...,
					                                   get<EI>(args...).largest_cut_from(i)...);
					assert(tc.task_size.row == max_rows);

					task_data td;
					td.size = tc.task_size;
					td.index = i;
					td.offsets = tc.task_offsets;
					td.self = (void*)this;

					auto handles = std::tuple_cat(tc.task_handles, container_handles);

					auto uniform_args = std::make_tuple(td, get<UI>(args...)...);
					helpers::schedule_task(&cl,
					                       modes,
					                       handles,
					                       uniform_args);
					i.col += tc.task_size.col;
				}
				i.row += max_rows;
			}
		}

		template<typename ResultArgs,
		         typename ElwiseArgs,
		         typename ContainerArgs,
		         typename UniformArgs,
		         typename Self>
		template<typename... Args>
		void multivariant_task<ResultArgs,
		                       ElwiseArgs,
		                       ContainerArgs,
		                       UniformArgs,
		                       Self>
		::element_aligned(Size2D size, Args& ... args)
		{
			element_aligned_impl(result_handle_indices,
			                     elwise_handle_indices,
			                     container_handle_indices,
			                     uniform_indices,
			                     size,
			                     args...);
		}


		template<typename ResultArgs,
		         typename ElwiseArgs,
		         typename ContainerArgs,
		         typename UniformArgs,
		         typename Self>
		template<typename... Args,
		         size_t... RI,
		         size_t... EI,
		         size_t... CI,
		         size_t... UI>
		void
		multivariant_task<ResultArgs,
		                  ElwiseArgs,
		                  ContainerArgs,
		                  UniformArgs,
		                  Self>
		::element_aligned_res_per_block_impl(pack_indices<RI...>,
		                                     pack_indices<EI...>,
		                                     pack_indices<CI...>,
		                                     pack_indices<UI...>,
		                                     Size2D total_size,
		                                     Args & ...args)
		{
			// Note that RI points to empty std::vector<starpu_var<T>>.
			auto modes = helpers::modes_from_codelet(cl);
			auto & partials = get<0>(args...);

			Index2D i {};
			while (i.row < total_size.row)
			{
				i.col = 0;
				// We need to know how much work we have left to do.
				Size2D diff = total_size;
				diff.row -= i.row;
				diff.col -= i.col;
				diff.i -= i.i;

				// Find the maximum number of rows we can schedule at a time
				const size_t max_rows {
					std::min<size_t>({
							get<EI>(args...).block_height_from(i)...,
								diff.row})
						};

				auto container_handles =
					std::make_tuple(get<CI>(args...).data().allgather()...);

				while (i.col < total_size.col)
				{
					// Create result handles
					partials.emplace_back(
						starpu_mpi_data_get_rank(get<1>(args...)
						                         .largest_cut_from(i).handle));

					// Create handle_cut with "mock" sizes
					Offset2D zero = {0,0,0};
					Offset2D maxsize = {SIZE_MAX, SIZE_MAX, SIZE_MAX};
					helpers::handle_cut block_res_hc(
						partials.back().handle,
						zero, maxsize);

					i.i = i.col;
					diff.col = total_size.col - i.col;
					auto tc =
						helpers::create_task_cut(diff,
						                         block_res_hc,
						                         get<EI>(args...).largest_cut_from(i)...);
					assert(tc.task_size.row == max_rows);

					task_data td;
					td.size = tc.task_size;
					td.index = i;
					td.offsets = tc.task_offsets;
					td.self = (void*)this;

					auto handles = std::tuple_cat(tc.task_handles, container_handles);

					auto uniform_args = std::make_tuple(td, get<UI>(args...)...);
					helpers::schedule_task(&cl,
					                       modes,
					                       handles,
					                       uniform_args);
					i.col += tc.task_size.col;
				}
				i.row += max_rows;
			}
		}

		template<typename ResultArgs,
		         typename ElwiseArgs,
		         typename ContainerArgs,
		         typename UniformArgs,
		         typename Self>
		template<typename... Args>
		void
		multivariant_task<ResultArgs,
		                  ElwiseArgs,
		                  ContainerArgs,
		                  UniformArgs,
		                  Self>
		::element_aligned_res_per_block(Size2D size,
		                                Args& ... args)
		{
			element_aligned_res_per_block_impl(result_handle_indices,
			                                   elwise_handle_indices,
			                                   container_handle_indices,
			                                   uniform_indices,
			                                   size,
			                                   args...);
		}





		template<typename ResultArgs,
		         typename ElwiseArgs,
		         typename ContainerArgs,
		         typename UniformArgs,
		         typename Self>
		template<typename... Args,
		         size_t... RI,
		         size_t... EI,
		         size_t... CI,
		         size_t... UI>
		void multivariant_task<ResultArgs,
		                       ElwiseArgs,
		                       ContainerArgs,
		                       UniformArgs,
		                       Self>
		::element_aligned_sweep_impl(pack_indices<RI...>,
		                             pack_indices<EI...>,
		                             pack_indices<CI...>,
		                             pack_indices<UI...>,
		                             Size2D total_size,
		                             const skepu::SweepMode dir,
		                             Args & ...args)
		{
			auto modes = helpers::modes_from_codelet(cl);

			Index2D i {};
			while (i.row < total_size.row)
			{
				i.col = 0;
				// We need to know how much work we have left to do.
				Size2D diff = total_size;
				diff.row -= i.row;
				diff.col -= i.col;
				diff.i -= i.i;

				// Find the maximum number of rows we can schedule at a time
				size_t max_rows {
					std::min<size_t>({
							get<EI>(args...).block_height_from(i)...,
								diff.row})
						};
				if(dir == skepu::SweepMode::RowWise) {
					max_rows =
						std::min<size_t>({get<RI>(args...)
									.getParent()
									. block_width_from({0, i.row}) ... ,
									max_rows});
				}

				assert(max_rows > 0);

				// Invalidate writable starpu_matrix_containers
				pack_expand(get<RI>(args...)
				            .getParent()
				            .data()
				            .invalidate_unpartition()...,
				            0);

				auto container_handles =
					std::make_tuple(get<CI>(args...).data().allgather()...);

				while (i.col < total_size.col)
				{
					i.i = i.col;
					diff.col = total_size.col - i.col;

					// Manipulate task cuts for the result vector
					Index2D r_i {0, dir == skepu::SweepMode::RowWise ? i.row : i.col, 0};
					auto r_cuts =
						std::make_tuple(get<RI>(args...).largest_cut_from(r_i) ... );

					// "Fake" the number of rows/cols, so create_task_cut works properly
					if(dir == skepu::SweepMode::RowWise) {
						// Transpose size if in row mode
						pack_expand(std::get<RI>(r_cuts).local_size.transpose() ... );
						pack_expand(std::get<RI>(r_cuts).local_size.col = SIZE_MAX ... );
					} else {
						pack_expand(std::get<RI>(r_cuts).local_size.row = SIZE_MAX ... );
					}


					auto tc =
						helpers::create_task_cut(diff,
						                         std::get<RI>(r_cuts)...,
						                         get<EI>(args...).largest_cut_from(i) ... );

					if(dir == skepu::SweepMode::RowWise) {
						// "Flip" the offsets for the result vectors
						pack_expand(tc.task_offsets.o[RI].transpose() ...);
					}

					// And remove offset in row-direction, there should be none.
					pack_expand(tc.task_offsets.o[RI].row = 0 ...);

					assert(tc.task_size.row == max_rows);

					task_data td;
					td.size = tc.task_size;
					td.index = i;
					td.offsets = tc.task_offsets;
					td.self = (void*)this;

					auto handles = std::tuple_cat(tc.task_handles, container_handles);

					auto uniform_args = std::make_tuple(td, get<UI>(args...)...);
					helpers::schedule_task(&cl,
					                       modes,
					                       handles,
					                       uniform_args);
					i.col += tc.task_size.col;
				}
				i.col = 0;
				i.row += max_rows;
			}
		}

		template<typename ResultArgs,
		         typename ElwiseArgs,
		         typename ContainerArgs,
		         typename UniformArgs,
		         typename Self>
		template<typename... Args>
		void multivariant_task<ResultArgs,
		                       ElwiseArgs,
		                       ContainerArgs,
		                       UniformArgs,
		                       Self>
		::element_aligned_sweep(Size2D size,
		                        const skepu::SweepMode dir,
		                        Args& ... args)
		{
			element_aligned_sweep_impl(result_handle_indices,
			                           elwise_handle_indices,
			                           container_handle_indices,
			                           uniform_indices,
			                           size,
			                           dir,
			                           args...);
		}
	} // cluster
} // skepu

#endif /* MULTIVARIANT_TASK_INL */
