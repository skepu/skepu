# SkePUCapabilities.cmake — what the build host has, for the tests (skepu_gate, REQUIRES in SkePUTests.cmake) and the examples (examples/CMakeLists.txt).
# Not installed.
#
# A capability is SKEPU_CAP_<name> (ON or OFF) with the reason it is off in SKEPU_CAP_<name>_WHY.
#
#   skepu_capability(<name> <available> <reason>)
#   skepu_note_skipped_example(<name> <build> <reasons>...)
#   skepu_collect_capabilities()   summary text, for skepu_print_config
#
# openmp, threads, opencl (library), opencl_device (GPU or accelerator), opencl_fp64, cuda (compiler), metal (backend built), metal_device, unified_memory,
# metal_fp64 (never on), mpi (compiler wrappers), mpi_run (mpiexec). No CUDA device probe: such tests stop by themselves.

function(skepu_capability name available why)
	if(available)
		set(SKEPU_CAP_${name} ON PARENT_SCOPE)
	else()
		set(SKEPU_CAP_${name} OFF PARENT_SCOPE)
	endif()
	set(SKEPU_CAP_${name}_WHY "${why}" PARENT_SCOPE)
	set_property(GLOBAL APPEND PROPERTY SKEPU_CAPABILITIES ${name})
endfunction()

skepu_capability(openmp "${SKEPU_OPENMP}" "${SKEPU_OPENMP_WHY}")

find_package(Threads QUIET)
skepu_capability(threads "${Threads_FOUND}" "no thread library was found")

skepu_capability(opencl "${SKEPU_OPENCL}" "${SKEPU_OPENCL_WHY}")

# Probe run as SkePU asks for devices. Skipped when cross-compiling or when it does not build: devices are then assumed,
# and tests that need them stop by themselves.
if(NOT SKEPU_OPENCL)
	skepu_capability(opencl_device OFF "${SKEPU_OPENCL_WHY}")
	skepu_capability(opencl_fp64 OFF "${SKEPU_OPENCL_WHY}")
else()
	set(_skepu_cl_gpus 1)
	set(_skepu_cl_fp64 1)
	set(_skepu_cl_note "")
	if(NOT CMAKE_CROSSCOMPILING)
		unset(_skepu_cl_run CACHE)
		unset(_skepu_cl_compile CACHE)
		try_run(_skepu_cl_run _skepu_cl_compile
			${CMAKE_BINARY_DIR}/probes/opencl ${CMAKE_CURRENT_LIST_DIR}/probes/opencl_devices.cpp
			LINK_LIBRARIES OpenCL::OpenCL
			RUN_OUTPUT_VARIABLE _skepu_cl_output)
		if(_skepu_cl_compile AND "${_skepu_cl_run}" STREQUAL "0" AND _skepu_cl_output MATCHES "gpu=([0-9]+) fp64=([0-9]+)")
			set(_skepu_cl_gpus ${CMAKE_MATCH_1})
			set(_skepu_cl_fp64 ${CMAKE_MATCH_2})
		else()
			set(_skepu_cl_note " (not probed)")
		endif()
	endif()
	skepu_capability(opencl_device "${_skepu_cl_gpus}" "no OpenCL GPU or accelerator device was found")
	set(SKEPU_CAP_opencl_device_NOTE "${_skepu_cl_gpus} found${_skepu_cl_note}")
	skepu_capability(opencl_fp64 "${_skepu_cl_fp64}" "no OpenCL device has double precision")
	set(SKEPU_CAP_opencl_fp64_NOTE "${_skepu_cl_fp64} of them${_skepu_cl_note}")
endif()

skepu_capability(cuda "${SKEPU_CUDA}" "${SKEPU_CUDA_WHY}")

# Metal: SKEPU_METAL says whether the backend is built; the probe asks the device, as for OpenCL. Unified memory is required because the
# backend uses container memory without copying.
set(_skepu_mtl_device 0)
set(_skepu_mtl_unified 0)
set(_skepu_mtl_why "Metal is only on Apple hardware")
set(_skepu_mtl_note "")
if(APPLE)
	set(_skepu_mtl_why "")
	if(CMAKE_CROSSCOMPILING)
		set(_skepu_mtl_device 1)
		set(_skepu_mtl_unified 1)
		set(_skepu_mtl_note " (not probed)")
	else()
		unset(_skepu_mtl_run CACHE)
		unset(_skepu_mtl_compile CACHE)
		try_run(_skepu_mtl_run _skepu_mtl_compile
			${CMAKE_BINARY_DIR}/probes/metal ${CMAKE_CURRENT_LIST_DIR}/probes/metal_device.cpp
			LINK_LIBRARIES "-framework Metal" "-framework Foundation" "-lobjc"
			RUN_OUTPUT_VARIABLE _skepu_mtl_output)
		if(_skepu_mtl_compile AND "${_skepu_mtl_run}" STREQUAL "0" AND _skepu_mtl_output MATCHES "device=([0-9]+) unified=([0-9]+)")
			set(_skepu_mtl_device ${CMAKE_MATCH_1})
			set(_skepu_mtl_unified ${CMAKE_MATCH_2})
		else()
			# not probed: tests that need them stop by themselves
			set(_skepu_mtl_device 1)
			set(_skepu_mtl_unified 1)
			set(_skepu_mtl_note " (not probed)")
		endif()
	endif()
endif()
skepu_capability(metal "${SKEPU_METAL}" "the Metal backend is not built")
skepu_capability(metal_device "${_skepu_mtl_device}" "${_skepu_mtl_why}")
if(NOT _skepu_mtl_why)
	set(SKEPU_CAP_metal_device_WHY "no Metal device was found")
endif()
skepu_capability(unified_memory "${_skepu_mtl_unified}" "${_skepu_mtl_why}")
if(NOT _skepu_mtl_why)
	set(SKEPU_CAP_unified_memory_WHY "the GPU does not share its memory with the CPU")
endif()
# no device can have it; not probed
skepu_capability(metal_fp64 OFF "the Metal Shading Language has no double precision")
set(SKEPU_CAP_metal_device_NOTE "found${_skepu_mtl_note}")
set(SKEPU_CAP_unified_memory_NOTE "shared with the CPU${_skepu_mtl_note}")

skepu_capability(mpi "${SKEPU_MPI}" "${SKEPU_MPI_WHY}")
if(SKEPU_MPI AND MPIEXEC_EXECUTABLE)
	skepu_capability(mpi_run ON "")
elseif(SKEPU_MPI)
	skepu_capability(mpi_run OFF "no mpiexec was found")
else()
	skepu_capability(mpi_run OFF "${SKEPU_MPI_WHY}")
endif()

# <build>: sequential, parallel or mpi.
function(skepu_note_skipped_example name build)
	string(REPLACE ";" ", " _why "${ARGN}")
	set_property(GLOBAL APPEND PROPERTY SKEPU_EXAMPLES_SKIPPED "${name}|${build}|${_why}")
endfunction()

# Title with an underline, indented like the blocks of skepu_print_config.
function(skepu_summary_heading out title)
	string(LENGTH "${title}" _length)
	string(REPEAT "-" ${_length} _line)
	set(${out} "\n    ${title}\n    ${_line}" PARENT_SCOPE)
endfunction()

# Summary text for skepu_print_config: capabilities, and skipped tests per missing capability.
function(skepu_collect_capabilities)
	set(_width 16)
	skepu_summary_heading(_text "Capabilities of this host")
	get_property(_caps GLOBAL PROPERTY SKEPU_CAPABILITIES)
	foreach(_c IN LISTS _caps)
		string(LENGTH "${_c}" _length)
		math(EXPR _pad "${_width} - ${_length}")
		string(REPEAT " " ${_pad} _gap)
		if(SKEPU_CAP_${_c})
			set(_state "ON ")
			set(_note "${SKEPU_CAP_${_c}_NOTE}")
		else()
			set(_state "OFF")
			set(_note "${SKEPU_CAP_${_c}_WHY}")
		endif()
		string(STRIP "${_c}${_gap}${_state}  ${_note}" _row)
		string(APPEND _text "\n    ${_row}")
	endforeach()

	get_property(_skipped GLOBAL PROPERTY SKEPU_SKIPPED_TESTS)
	list(LENGTH _skipped _count)
	if(_count)
		set(_by)
		foreach(_s IN LISTS _skipped)
			string(REGEX REPLACE "^[^|]*\\|" "" _names "${_s}")
			string(REPLACE "," ";" _parts "${_names}")
			foreach(_c IN LISTS _parts)
				if(NOT DEFINED SKEPU_SKIPPED_BY_${_c})
					set(SKEPU_SKIPPED_BY_${_c} 0)
					list(APPEND _by ${_c})
				endif()
				math(EXPR SKEPU_SKIPPED_BY_${_c} "${SKEPU_SKIPPED_BY_${_c}} + 1")
			endforeach()
		endforeach()
		skepu_summary_heading(_heading "Tests registered as skipped (${_count})")
		string(APPEND _text "\n${_heading}")
		foreach(_c IN LISTS _by)
			string(LENGTH "${_c}" _length)
			math(EXPR _pad "${_width} - ${_length}")
			string(REPEAT " " ${_pad} _gap)
			string(APPEND _text "\n    ${_c}${_gap}${SKEPU_SKIPPED_BY_${_c}}")
		endforeach()
	endif()
	# printed after the examples summary
	get_property(_examples GLOBAL PROPERTY SKEPU_EXAMPLES_SKIPPED)
	list(LENGTH _examples _example_count)
	set(_example_text "")
	if(_example_count)
		skepu_summary_heading(_example_text "Examples not built (${_example_count})")
		foreach(_e IN LISTS _examples)
			string(REPLACE "|" ";" _parts "${_e}")
			list(GET _parts 0 _name)
			list(GET _parts 1 _build)
			list(GET _parts 2 _why)
			string(APPEND _example_text "\n    ${_name} (${_build}): ${_why}")
		endforeach()
	endif()
	set_property(GLOBAL PROPERTY SKEPU_EXAMPLES_SKIPPED_TEXT "${_example_text}")
	set_property(GLOBAL PROPERTY SKEPU_CAPABILITIES_TEXT "${_text}")
endfunction()
