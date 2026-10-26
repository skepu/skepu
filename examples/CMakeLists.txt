project(skepu-examples LANGUAGES CXX)

# The example programs. The same set as PROGS in the Makefile; keep them in step.
set(_skepu_examples
	backend_switching
	blas_tour
	cellular_automaton
	cma
	conjugate_gradient
	conjugate_gradient_sparse
	coulombic
	dotproduct
	edgedetect
	fusion_patterns
	heat_diffusion
	horner
	image_manip
	mandelbrot
	maxmin
	median
	miller_rabin
	mmmult
	mmmult_row_col
	montecarlo
	mvmult
	mvmult_row
	nbody
	nbody_mappairs
	ode_fisher
	ode_lorenz
	ozaki_gemm
	perlin
	ppmcc
	pso
	psnr
	riemann_sum
	taylor
)

include(CMake/example-options.cmake)

option(SKEPU_EXAMPLES_TRACING
	"Build the parallel examples with tracing (SKEPU_TRACING); the traces, for SkeVU, are written to trace/ in the build directory of the examples."
	OFF)

option(SKEPU_EXAMPLES_TRACING_SNAPSHOTS
	"With SKEPU_EXAMPLES_TRACING: the examples also take snapshots of containers (SKEPU_TRACING_SNAPSHOTS), written to snapshots/ in the build directory of the examples. They can be large and slow the programs down."
	OFF)

if(SKEPU_CUDA)
	enable_language(CUDA)
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})
add_library(lodepng STATIC lodepng.cpp)

# What a program needs besides SkePU is, by default, a property of the program and not of the backend: the sequential, parallel and MPI versions are built
# from the same source, so they link the same libraries. The libraries are found from the source (a program that includes lodepng.h links lodepng), and
# this function is called for each version of each program, so that no backend is forgotten. A library that only some backends need can be added to those
# targets after the call.
function(skepu_example_dependencies target example)
	file(READ ${CMAKE_CURRENT_LIST_DIR}/${example}.cpp _source)
	if(_source MATCHES "#include[ \t]+[\"<]lodepng\\.h[\">]")
		target_link_libraries(${target} PRIVATE lodepng)
	endif()
endfunction()

set(_trace_dir ${CMAKE_CURRENT_BINARY_DIR}/trace)
set(_snapshot_dir ${CMAKE_CURRENT_BINARY_DIR}/snapshots)
if(SKEPU_EXAMPLES_TRACING)
	file(MAKE_DIRECTORY ${_trace_dir} ${_snapshot_dir})
endif()

if(SKEPU_EXAMPLES_SEQ)
	# Meta target for all sequential programs.
	add_custom_target(all-sequential)

	foreach(example IN LISTS _skepu_examples)
		string(TOUPPER ${example} ucname)
		if(SKEPU_EXAMPLE_${ucname})
			add_executable(${example}_seq ${example}.cpp)
			target_link_libraries(${example}_seq PRIVATE SkePU::SkePU)
			set_target_properties(${example}_seq
				PROPERTIES
					RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sequential
					RUNTIME_OUTPUT_NAME ${example})
			skepu_example_dependencies(${example}_seq ${example})
			add_dependencies(all-sequential ${example}_seq)
		endif()
	endforeach()
endif()

if(SKEPU_EXAMPLES_PAR)
	# Meta target for all parallel programs
	add_custom_target(all-parallel)

	foreach(example IN LISTS _skepu_examples)
		string(TOUPPER ${example} ucname)

		if(SKEPU_EXAMPLE_${ucname})
			if(SKEPU_EXAMPLE_${ucname}_CUDA)
				list(APPEND ${example}_par_backends CUDA)
			endif()
			if(SKEPU_EXAMPLE_${ucname}_OPENCL)
				list(APPEND ${example}_par_backends OpenCL)
			endif()
			if(SKEPU_EXAMPLE_${ucname}_OPENMP)
				list(APPEND ${example}_par_backends OpenMP)
			endif()

			skepu_add_executable(${example}_par
				${${example}_par_backends}
				SKEPUSRC ${example}.cpp)

			set_target_properties(${example}_par
				PROPERTIES
					RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/parallel
					RUNTIME_OUTPUT_NAME ${example})
			skepu_example_dependencies(${example}_par ${example})
			if(SKEPU_EXAMPLES_TRACING)
				target_compile_definitions(${example}_par PRIVATE
					SKEPU_TRACING
					SKEPU_TRACE_FILE=${_trace_dir}/trace_${example}.json
					SKEPU_TRACE_SNAPSHOT_DIR=${_snapshot_dir})
				if(SKEPU_EXAMPLES_TRACING_SNAPSHOTS)
					target_compile_definitions(${example}_par PRIVATE SKEPU_TRACING_SNAPSHOTS)
				endif()
			endif()
			add_dependencies(all-parallel ${example}_par)
		endif()
	endforeach()
endif()

if(SKEPU_EXAMPLES_MPI)
	# Meta target for all mpi programs
	add_custom_target(all-mpi)

	foreach(example IN LISTS _skepu_examples)
		string(TOUPPER ${example} ucname)

		if(SKEPU_EXAMPLE_${ucname} AND SKEPU_EXAMPLE_${ucname}_MPI)
			list(APPEND ${example}_mpi_backends MPI)

			skepu_add_executable(${example}_mpi
				${${example}_mpi_backends}
				SKEPUSRC ${example}.cpp)
			set_target_properties(${example}_mpi
				PROPERTIES
					RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mpi
					RUNTIME_OUTPUT_NAME ${example})
			skepu_example_dependencies(${example}_mpi ${example})
			add_dependencies(all-mpi ${example}_mpi)
		endif()
	endforeach()
endif()
