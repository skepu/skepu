# The options of the example programs. All programs are listed in _skepu_examples in
# CMakeLists.txt; this defines, for every one of them (<NAME> is the upper case program name):
#   SKEPU_EXAMPLE_<NAME>            build the program
#   SKEPU_EXAMPLE_<NAME>_<BACKEND>  build it for the backend (CUDA, OPENCL, OPENMP, MPI)

# The lists _skepu_examples, _skepu_examples_cpu_only (the sparse matrices are not supported by the CUDA and OpenCL backends yet) and _skepu_examples_mpi
# come from programs.txt (see CMakeLists.txt).

foreach(_example IN LISTS _skepu_examples)
	string(TOUPPER ${_example} _uc)

	set(_cuda ${SKEPU_CUDA})
	if(_example IN_LIST _skepu_examples_cpu_only)
		set(_cuda OFF)
	endif()
	set(_opencl ${SKEPU_OPENCL})
	if(_example IN_LIST _skepu_examples_cpu_only)
		set(_opencl OFF)
	endif()
	set(_mpi OFF)
	if(_example IN_LIST _skepu_examples_mpi)
		set(_mpi ${SKEPU_MPI})
	endif()

	option(SKEPU_EXAMPLE_${_uc} "" ${SKEPU_BUILD_EXAMPLES})
	option(SKEPU_EXAMPLE_${_uc}_CUDA "" ${_cuda})
	option(SKEPU_EXAMPLE_${_uc}_OPENCL "" ${_opencl})
	option(SKEPU_EXAMPLE_${_uc}_OPENMP "" ${SKEPU_OPENMP})
	option(SKEPU_EXAMPLE_${_uc}_MPI "" ${_mpi})
	mark_as_advanced(FORCE
		SKEPU_EXAMPLE_${_uc}
		SKEPU_EXAMPLE_${_uc}_CUDA
		SKEPU_EXAMPLE_${_uc}_MPI
		SKEPU_EXAMPLE_${_uc}_OPENCL
		SKEPU_EXAMPLE_${_uc}_OPENMP)
endforeach()
