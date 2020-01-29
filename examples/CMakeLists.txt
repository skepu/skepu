project(skepu-examples LANGUAGES CXX CUDA)

include_directories(${CMAKE_CURRENT_LIST_DIR})
add_library(lodepng STATIC lodepng.cpp)

set(_skepu_programs
	call
	cma
	containers
	coulombic
	dotproduct
	# lambda
	mandelbrot
	map
	# mapoverlap
	mappairs
	mapreduce
	# maptuple
	# median
	mmmult
	mvmult
	mvmult_row
	nbody
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

	foreach(program IN LISTS _skepu_programs)
		add_executable(${program}_seq ${program}.cpp)
		target_link_libraries(${program}_seq PRIVATE SkePU::SkePU)
		set_target_properties(${program}_seq
			PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sequential
				RUNTIME_OUTPUT_NAME ${program})
		add_dependencies(all-sequential ${program}_seq)
	endforeach()

	if(TARGET median_seq)
		target_link_libraries(median_seq PRIVATE lodepng)
	endif()
endif()

if(SKEPU_EXAMPLES_PAR)
	# Meta target for all parallel programs
	add_custom_target(all-parallel)

	foreach(program IN LISTS _skepu_programs)
		# TODO: Add cuda when enabled in skepu-tool
		skepu_add_executable(${program}_par
			OpenCL OpenMP
			SKEPUSRC ${program}.cpp)
		set_target_properties(${program}_par
			PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/parallel
				RUNTIME_OUTPUT_NAME ${program})
		add_dependencies(all-parallel ${program}_par)
	endforeach()

	if(TARGET median_par)
		target_link_libraries(median_par PRIVATE lodepng)
	endif()
endif()

if(SKEPU_EXAMPLES_MPI)
	# Meta target for all mpi programs
	add_custom_target(all-mpi)

	skepu_add_executable(nbody_mpi
		MPI
		SKEPUSRC nbody_mpi.cpp)
	set_target_properties(nbody_mpi
		PROPERTIES
			RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mpi
			RUNTIME_OUTPUT_NAME nbody)
	add_dependencies(all-mpi nbody_mpi)
endif()
