# The options of the example programs. All programs are listed in _skepu_examples in
# CMakeLists.txt; this defines, for every one of them (<NAME> is the upper case program name):
#   SKEPU_EXAMPLE_<NAME>            build the program
#   SKEPU_EXAMPLE_<NAME>_<BACKEND>  build it for the backend (CUDA, OPENCL, OPENMP, MPI)

# Programs that are not built for these backends by default
set(_skepu_examples_no_cuda
	mmmult_row_col
	heat_diffusion) # no red-black update mode in CUDA
set(_skepu_examples_mpi
	cellular_automaton cma conjugate_gradient coulombic dotproduct horner mandelbrot maxmin miller_rabin
	mmmult mmmult_row_col montecarlo mvmult mvmult_row nbody nbody_mappairs ode_fisher ode_lorenz ppmcc psnr riemann_sum taylor)

foreach(_example IN LISTS _skepu_examples)
	string(TOUPPER ${_example} _uc)

	set(_cuda ${SKEPU_CUDA})
	if(_example IN_LIST _skepu_examples_no_cuda)
		set(_cuda OFF)
	endif()
	set(_mpi OFF)
	if(_example IN_LIST _skepu_examples_mpi)
		set(_mpi ${SKEPU_MPI})
	endif()

	option(SKEPU_EXAMPLE_${_uc} "" ${SKEPU_BUILD_EXAMPLES})
	option(SKEPU_EXAMPLE_${_uc}_CUDA "" ${_cuda})
	option(SKEPU_EXAMPLE_${_uc}_OPENCL "" ${SKEPU_OPENCL})
	option(SKEPU_EXAMPLE_${_uc}_OPENMP "" ${SKEPU_OPENMP})
	option(SKEPU_EXAMPLE_${_uc}_MPI "" ${_mpi})
	mark_as_advanced(FORCE
		SKEPU_EXAMPLE_${_uc}
		SKEPU_EXAMPLE_${_uc}_CUDA
		SKEPU_EXAMPLE_${_uc}_MPI
		SKEPU_EXAMPLE_${_uc}_OPENCL
		SKEPU_EXAMPLE_${_uc}_OPENMP)
endforeach()
