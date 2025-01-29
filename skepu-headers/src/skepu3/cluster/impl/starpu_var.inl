#ifndef STARPU_VAR_INL
#define STARPU_VAR_INL

#include "skepu3/cluster/starpu_var.hpp"
#include "skepu3/cluster/cluster.hpp"

namespace skepu
{
	namespace cluster
	{
		template<typename T>
		starpu_var<T>
		::starpu_var(size_t owner)
		{
			int home_node = -1;
			if(skepu::cluster::mpi_rank() == owner)
			{
				starpu_malloc((void**)&data, sizeof(T));
				home_node = STARPU_MAIN_RAM;
			}
			starpu_matrix_data_register(&handle,
																	home_node,
																	data,
																	1,1,1, // 1x1 Matrix
																	sizeof(T));
			starpu_mpi_data_register(handle,
															 skepu::cluster::mpi_tag(),
															 owner);
		}

		template<typename T>
		starpu_var<T>
		::~starpu_var()
		{
			if(!initialized)
				return;
			if(data)
			{
				auto rank = skepu::cluster::mpi_rank();
				auto data_loc = starpu_mpi_data_get_rank(handle);
				starpu_data_unregister(handle);
				if(data_loc == rank)
					starpu_free((void*)data);
				data = 0;
			}
			initialized = false;
		}

		template<typename T>
		void starpu_var<T>
		::broadcast()
		{
			starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle);
		}

		template<typename T>
		T starpu_var<T>
		::get()
		{
			starpu_data_acquire(handle, STARPU_R);
			T res = ((T*)starpu_matrix_get_local_ptr(handle))[0];
			starpu_data_release(handle);
			return res;
		}
	}
}

#endif /* STARPU_VAR_INL */
