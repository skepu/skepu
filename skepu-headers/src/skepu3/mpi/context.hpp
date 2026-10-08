#pragma once

// The MPI layer of the SkePU MPI backend: the communicator, the distribution of data and the (MPI-3) communication
// that the containers and the skeletons need. Nothing else in the backend calls MPI.

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <streambuf>
#include <string>
#include <vector>

// An error in the node-level library (SKEPU_ERROR) stops all processes
[[noreturn]] inline void skepu_mpi_abort()
{
	std::cerr.flush();
	MPI_Abort(MPI_COMM_WORLD, 1);
	std::abort();
}

namespace skepu
{
	namespace mpi
	{
		[[noreturn]] inline void fatal(const std::string &message)
		{
			std::cerr << "[SkePU MPI] " << message << std::endl;
			skepu_mpi_abort();
		}

		namespace detail
		{
			struct NullBuffer : std::streambuf
			{
				int overflow(int c) override { return c; }
			};

			// MPI is initialized before main and finalized after the static objects of the program (those that are
			// constructed later) are gone. Only the first process writes to standard output: a program that is
			// written for one process prints what it should print once. (SKEPU_MPI_ALL_OUTPUT=1 for output from all.)
			struct Context
			{
				int rank = 0, size = 1;
				MPI_Comm comm = MPI_COMM_WORLD;

				Context()
				{
					int initialized = 0;
					MPI_Initialized(&initialized);
					if (!initialized)
					{
						int provided = 0;
						MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
						owned = true;
					}
					MPI_Comm_rank(comm, &rank);
					MPI_Comm_size(comm, &size);

					const char *all = std::getenv("SKEPU_MPI_ALL_OUTPUT");
					if (rank != 0 && !(all && std::string(all) == "1"))
					{
						static NullBuffer null;
						std::cout.rdbuf(&null);
						if (!std::freopen("/dev/null", "w", stdout))
							std::cout.setstate(std::ios::failbit);
					}
				}

				~Context()
				{
					int finalized = 0;
					MPI_Finalized(&finalized);
					if (owned && !finalized)
						MPI_Finalize();
				}

				bool owned = false;
			};
		}

		inline detail::Context &context()
		{
			static detail::Context instance;
			return instance;
		}

		inline int rank() { return context().rank; }
		inline int size() { return context().size; }
		inline MPI_Comm comm() { return context().comm; }

		// Make sure that MPI is initialized when the program starts
		inline const bool initialized_at_startup = (context(), true);

		// The datatype of a (trivially copyable) element: a block of bytes of its size
		template<typename T>
		MPI_Datatype datatype()
		{
			static MPI_Datatype type = []
			{
				MPI_Datatype t;
				MPI_Type_contiguous((int)sizeof(T), MPI_BYTE, &t);
				MPI_Type_commit(&t);
				return t;
			}();
			return type;
		}

		// MPI-3: the counts of a message are int
		inline int checked_count(size_t n, const char *what)
		{
			if (n > (size_t)INT_MAX)
				fatal(std::string(what) + ": " + std::to_string(n) + " elements; at most " + std::to_string(INT_MAX) + " per message is supported (MPI-3)");
			return (int)n;
		}

		// ------------------------------------------------------------------
		// Block distribution of a container over its first dimension (the elements of a vector, the rows of
		// a matrix, ...): n indices, each of unit elements; the first n % p processes have one index more
		// ------------------------------------------------------------------

		struct BlockDistribution
		{
			size_t n = 0;
			int p = 1;
			size_t unit = 1;

			// Indices of the first dimension
			size_t count(int r) const { return n / p + ((size_t)r < n % p ? 1 : 0); }
			size_t offset(int r) const { return (n / p) * r + std::min((size_t)r, n % p); }

			// Elements
			size_t elements(int r) const { return count(r) * unit; }
			size_t element_offset(int r) const { return offset(r) * unit; }
			size_t total() const { return n * unit; }

			std::vector<int> counts() const
			{
				std::vector<int> c(p);
				for (int r = 0; r < p; ++r) c[r] = checked_count(elements(r), "distribution");
				return c;
			}

			std::vector<int> displacements() const
			{
				std::vector<int> d(p);
				for (int r = 0; r < p; ++r) d[r] = checked_count(element_offset(r), "distribution");
				return d;
			}
		};

		// ------------------------------------------------------------------
		// Communication
		// ------------------------------------------------------------------

		// Every process gets the whole sequence
		template<typename T>
		void allgather(const T *mine, T *whole, const BlockDistribution &d)
		{
			auto counts = d.counts(), displs = d.displacements();
			MPI_Allgatherv(mine, counts[rank()], datatype<T>(), whole, counts.data(), displs.data(), datatype<T>(), comm());
		}

		// The first process has the whole sequence, and gives each process its part
		template<typename T>
		void scatter_from_root(const T *whole, T *mine, const BlockDistribution &d)
		{
			auto counts = d.counts(), displs = d.displacements();
			MPI_Scatterv(whole, counts.data(), displs.data(), datatype<T>(), mine, counts[rank()], datatype<T>(), 0, comm());
		}

		// One value from each process, in the order of the processes, to every process
		template<typename T>
		std::vector<T> allgather_one(const T &mine)
		{
			std::vector<T> all(size());
			MPI_Allgather(&mine, 1, datatype<T>(), all.data(), 1, datatype<T>(), comm());
			return all;
		}
	}
}
