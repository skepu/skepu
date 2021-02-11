project(skepu-examples LANGUAGES CXX)
include(CMake/example-options.cmake)

if(SKEPU_CUDA)
	enable_language(CUDA)
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})
add_library(lodepng STATIC lodepng.cpp)

set(_skepu_examples
	backendspec
	call
	cma
	complex
	containers
	coulombic
	dotproduct
	lambda
	mandelbrot
	map
	mapoverlap
	mapoverlap-tuple
	mappairs
	mappairs_tuple
	mapreduce
	maptuple
	maxmin
	median
	mmmult
	mmmult_row_col
	mvmult
	mvmult_row
	nbody
	nbody_mappairs
	objectoriented
	ppmcc
	psnr
	reduce
	scan
	taylor
	writeback
)

if(SKEPU_EXAMPLES_SEQ)
	# Meta target for all sequential programs.
	add_custom_target(all-sequential)

	if(SKEPU_EXAMPLE_NBODY_NOSKEPU)
		add_executable(nbody_noskepu nbody_noskepu.cpp)
		set_target_properties(nbody_noskepu PROPERTIES
			RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sequential)
		add_dependencies(all-sequential nbody_noskepu)
	endif()

	foreach(example IN LISTS _skepu_examples)
		string(TOUPPER ${example} ucname)
		if(SKEPU_EXAMPLE_${ucname})
			add_executable(${example}_seq ${example}.cpp)
			target_link_libraries(${example}_seq PRIVATE SkePU::SkePU)
			set_target_properties(${example}_seq
				PROPERTIES
					RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sequential
					RUNTIME_OUTPUT_NAME ${example})
			add_dependencies(all-sequential ${example}_seq)
		endif()
	endforeach()

	if(TARGET median_seq)
		target_link_libraries(median_seq PRIVATE lodepng)
	endif()
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
			add_dependencies(all-parallel ${example}_par)
		endif()
	endforeach()

	if(TARGET median_par)
		target_link_libraries(median_par PRIVATE lodepng)
	endif()
endif()

if(SKEPU_EXAMPLES_MPI)
	# Meta target for all mpi programs
	add_custom_target(all-mpi)

	foreach(example IN LISTS _skepu_examples)
		string(TOUPPER ${example} ucname)

		if(SKEPU_EXAMPLE_${ucname} AND SKEPU_EXAMPLE_${ucname}_MPI)
			list(APPEND ${example}_mpi_backends MPI)
			if(SKEPU_EXAMPLE_${ucname}_MPI_CUDA)
				list(APPEND ${example}_mpi_backends CUDA)
			endif()

			skepu_add_executable(${example}_mpi
				${${example}_mpi_backends}
				SKEPUSRC ${example}.cpp)
			set_target_properties(${example}_mpi
				PROPERTIES
					RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mpi
					RUNTIME_OUTPUT_NAME ${example})
			add_dependencies(all-mpi ${example}_mpi)
		endif()
	endforeach()
endif()
