project(skepu-examples LANGUAGES CXX)

# Programs and flags: examples/programs.txt, also read by programs.mk. The flags give _skepu_examples_cpu_only (no OpenCL, CUDA or MPI) and
# _skepu_examples_mpi, used by CMake/example-options.cmake.
set(_programs_table ${CMAKE_CURRENT_LIST_DIR}/programs.txt)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_programs_table})
file(STRINGS ${_programs_table} _program_rows REGEX "^[ \t]*[A-Za-z0-9_]")
set(_skepu_examples)
set(_skepu_examples_cpu_only)
set(_skepu_examples_mpi)
foreach(_row IN LISTS _program_rows)
	if(NOT _row MATCHES "^[ \t]*([A-Za-z0-9_]+)([ \t]+([^ \t]+))?")
		message(FATAL_ERROR "[SkePU] A row of ${_programs_table} that is not a program: ${_row}")
	endif()
	set(_program ${CMAKE_MATCH_1})
	string(REPLACE "," ";" _flags "${CMAKE_MATCH_3}")
	foreach(_flag IN LISTS _flags)
		if(NOT _flag MATCHES "^(-|cpu-only|mpi)$")
			message(FATAL_ERROR "[SkePU] ${_program} has a flag that is not known in ${_programs_table}: ${_flag}")
		endif()
	endforeach()
	list(APPEND _skepu_examples ${_program})
	if("cpu-only" IN_LIST _flags)
		list(APPEND _skepu_examples_cpu_only ${_program})
	endif()
	if("mpi" IN_LIST _flags)
		list(APPEND _skepu_examples_mpi ${_program})
	endif()
	if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/${_program}.cpp)
		message(FATAL_ERROR "[SkePU] ${_programs_table} has ${_program}, which has no source: ${_program}.cpp")
	endif()
endforeach()

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

# Extra libraries belong to the program, not the backend: all versions link the same ones, found from the source (lodepng.h links lodepng).
# Called for each version of each program; a library only some backends need can be added to those targets afterwards.
function(skepu_example_dependencies target example)
	file(READ ${CMAKE_CURRENT_LIST_DIR}/${example}.cpp _source)
	if(_source MATCHES "#include[ \t]+[\"<]lodepng\\.h[\">]")
		target_link_libraries(${target} PRIVATE lodepng)
	endif()
endfunction()

# A program that names a backend in the source (skepu::Backend::Type::CUDA) is built in the parallel version only where the host has that backend
# (SkePUCapabilities.cmake). Programs that take the backend from the command line or use the default have no requirement. <backends>: those the parallel
# version is built for (CUDA, OpenCL, OpenMP); <missing> gets the reasons it cannot be built, empty if it can.
function(skepu_example_missing example backends missing)
	file(READ ${CMAKE_CURRENT_LIST_DIR}/${example}.cpp _source)
	# comments are stripped first
	string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" "" _source "${_source}")
	string(REGEX REPLACE "//[^\n]*" "" _source "${_source}")
	string(REGEX MATCHALL "Backend::Type::[A-Za-z]+" _uses "${_source}")
	set(_reasons)
	set(_seen)
	foreach(_use IN LISTS _uses)
		string(REGEX REPLACE "^.*::" "" _type "${_use}")
		if(_type IN_LIST _seen)
			continue()
		endif()
		list(APPEND _seen ${_type})
		if(_type STREQUAL CPU)
			continue()
		elseif(_type STREQUAL OpenMP)
			set(_needs openmp)
		elseif(_type STREQUAL OpenCL)
			set(_needs opencl opencl_device)
		elseif(_type STREQUAL CUDA)
			set(_needs cuda)
		else()
			continue()
		endif()
		set(_lacking)
		foreach(_c IN LISTS _needs)
			if(NOT SKEPU_CAP_${_c})
				list(APPEND _lacking "${_c}: ${SKEPU_CAP_${_c}_WHY}")
			endif()
		endforeach()
		if(_lacking)
			list(APPEND _reasons ${_lacking})
		elseif(NOT _type IN_LIST backends)
			string(TOUPPER ${example} _uc)
			string(TOUPPER ${_type} _type_uc)
			list(APPEND _reasons "the program is not built for ${_type} (SKEPU_EXAMPLE_${_uc}_${_type_uc} is off)")
		endif()
	endforeach()
	set(${missing} "${_reasons}" PARENT_SCOPE)
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

			skepu_example_missing(${example} "${${example}_par_backends}" _missing)
			if(_missing)
				skepu_note_skipped_example(${example} parallel ${_missing})
				continue()
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
