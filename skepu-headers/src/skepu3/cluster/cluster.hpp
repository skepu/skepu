#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <memory>

#include <starpu.h>
#include <starpu_mpi.h>

namespace skepu {
	namespace cluster {
		namespace state {
			struct internal_state {
				starpu_conf conf;
				int mpi_rank;
				int mpi_size;
				int mpi_tag {1};

				internal_state() {
					starpu_conf_init(&conf);
					conf.sched_policy_name = "peager";
					if(conf.reserve_ncpus == -1)
						conf.reserve_ncpus = 0;
					if(conf.ncpus - conf.reserve_ncpus > 1)
					{
						conf.single_combined_worker = 1;
					}
					auto err = starpu_init(&conf);
					assert(!err);
					starpu_mpi_init(NULL, NULL, 1);

					mpi_rank = starpu_mpi_world_rank();
					mpi_size = starpu_mpi_world_size();
				};

				~internal_state() {
					starpu_mpi_shutdown();
				};
			};
		}
	}
}

namespace skepu {
	namespace cluster {
		namespace state {

			inline internal_state * s() {
				bool static initialized{true};
				auto deleter = [&](internal_state * ptr) {
					initialized = false;
					delete ptr;
				};
				std::unique_ptr<internal_state, decltype(deleter)> static g_state{
					new internal_state,
					deleter};

				if(initialized)
					return g_state.get();
				else
					return 0;
			}
		}

		static size_t
		mpi_rank() {
			return state::s()->mpi_rank;
		};

		static size_t
		mpi_size() {
			return state::s()->mpi_size;
		};

		/** Return a new mpi_tag for use with a handle, as a unique identifier for
		 * use in registering data handles.
		 *
		 * This function *must* be called coherently across all ranks.
		 */
		static size_t
		mpi_tag() {
			return state::s()->mpi_tag++;
		};

		static auto
		starpu_ncpus()
		-> int
		{
			return state::s()->conf.ncpus;
		}

		static auto
		barrier()
		-> void
		{
			if(state::s())
			{
				starpu_mpi_wait_for_all(MPI_COMM_WORLD);
				starpu_mpi_barrier(MPI_COMM_WORLD);
			}
		}
	}
}

#endif /* CLUSTER_HPP */
