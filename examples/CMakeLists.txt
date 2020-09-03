project(skepu-examples LANGUAGES CXX)

if(SKEPU_CUDA)
	enable_language(CUDA)
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})
add_library(lodepng STATIC lodepng.cpp)

set(_skepu_examples
	backendspec
	call
	cma
	containers
	coulombic
	dotproduct
	lambda
	mandelbrot
	map
	mapoverlap
	mappairs
	mapreduce
	maptuple
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

set(_skepu_mpi_examples
	# call
	# cma
	# complex
	# containers
	coulombic
	dotproduct
	# lambda
	mandelbrot
	map
	# mapoverlap
	mappairs
	# mappairs_tuple
	mapreduce
	maptuple
	maxmin
	# median
	mmmult
	# mmmult_row_col
	mvmult
	mvmult_row
	nbody
	nbody_mappairs
	objectoriented
	ppmcc
	psnr
	reduce
	# scan
	# taylor
	# writeback
)

if(SKEPU_EXAMPLES_SEQ)
	# Meta target for all sequential programs.
	add_custom_target(all-sequential)

	foreach(example IN LISTS _skepu_examples)
		add_executable(${example}_seq ${example}.cpp)
		target_link_libraries(${example}_seq PRIVATE SkePU::SkePU)
		set_target_properties(${example}_seq
			PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sequential
				RUNTIME_OUTPUT_NAME ${example})
		add_dependencies(all-sequential ${example}_seq)
	endforeach()

	if(TARGET median_seq)
		target_link_libraries(median_seq PRIVATE lodepng)
	endif()
endif()

if(SKEPU_EXAMPLES_PAR)
	# Meta target for all parallel programs
	add_custom_target(all-parallel)

#TODO: Enalbe when able
#	if(SKEPU_CUDA)
#		list(APPEND SKEPU_BACKENDS CUDA)
#	endif()
#	if(SKEPU_OPENCL)
#		list(APPEND SKEPU_BACKENDS OpenCL)
#	endif()
	if(SKEPU_OPENMP)
		list(APPEND SKEPU_BACKENDS OpenMP)
	endif()

	if(NOT SKEPU_BACKENDS)
		message(FATAL_ERROR "[SKEPU] Could not enable any parallel backends")
	endif()

	foreach(example IN LISTS _skepu_examples)
		skepu_add_executable(${example}_par
			${SKEPU_BACKENDS}
			SKEPUSRC ${example}.cpp)
		set_target_properties(${example}_par
			PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/parallel
				RUNTIME_OUTPUT_NAME ${example})
		add_dependencies(all-parallel ${example}_par)
	endforeach()

	if(TARGET median_par)
		target_link_libraries(median_par PRIVATE lodepng)
	endif()
endif()

if(SKEPU_EXAMPLES_MPI)
	# Meta target for all mpi programs
	add_custom_target(all-mpi)

	foreach(example IN LISTS _skepu_mpi_examples)
		skepu_add_executable(${example}_mpi
			MPI
			SKEPUSRC ${example}.cpp)
		set_target_properties(${example}_mpi
			PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mpi
				RUNTIME_OUTPUT_NAME ${example})
		add_dependencies(all-mpi ${example}_mpi)
	endforeach()
endif()
