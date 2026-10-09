#pragma once

// The SkePU MPI backend (see README.md). Included by <skepu> when SKEPU_MPI is defined, after the node-level SkePU
// (compiled as the namespace skepu_node) has been included.

#ifdef SKEPU_CUDA
# error "The MPI backend does not support CUDA yet (the kernels need the origin of a distributed container: see the OpenCL Map)"
#endif

#include "skepu3/mpi/context.hpp"

namespace skepu
{
	// The program sees the node-level library under the name skepu, except for what is replaced by the backend: the
	// distributed containers and the skeleton instances that work on them (skepu::backend).
	using namespace ::skepu_node;
}

#include "skepu3/mpi/container.hpp"
#include "skepu3/mpi/vector.hpp"
#include "skepu3/mpi/matrix.hpp"
#include "skepu3/mpi/tensor.hpp"
#include "skepu3/mpi/external.hpp"
#include "skepu3/mpi/helpers.hpp"
#include "skepu3/mpi/skeletons.hpp"
#include "skepu3/mpi/overlap.hpp"
#include "skepu3/mpi/snapshot.hpp"
