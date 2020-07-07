project(skepu-examples LANGUAGES CXX CUDA)

include_directories(${CMAKE_CURRENT_LIST_DIR})
add_library(lodepng STATIC lodepng.cpp)

set(_skepu_examples
	backendspec
	# call
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
	# maptuple
	median
	mmmult
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
	# containers
	coulombic
	# dotproduct
	# lambda
	mandelbrot
	map
	# mapoverlap
	# mappairs
	# mapreduce
	# maptuple
	# median
	mmmult
	mvmult
	mvmult_row
	nbody
	# nbody_mappairs
	objectoriented
	# ppmcc
	# psnr
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

	foreach(example IN LISTS _skepu_examples)
		# TODO: Add CUDA when enabled in skepu-tool
		# TODO: Add OpenCL when enabled in skepu-tool
		skepu_add_executable(${example}_par
			OpenMP
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
