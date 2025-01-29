#pragma once

#include <cstddef>
#include <vector>
// Mockup of the cluster interface, to allow the SkePU precompiler to
// compile the code.

namespace skepu
{
	namespace cluster
	{
		static size_t
		mpi_rank() {
			return 0;
		}

		static size_t
		mpi_size() {
			return 1;
		}

		static size_t
		mpi_tag() {
			return 0;
		}

		static auto
		starpu_ncpus()
		-> int
		{
			return 0;
		}

		static auto
		barrier()
		-> void
		{}
	}
}
