# SkePUCapabilities.cmake — what the build host has, for the tests and the examples that need it.
#
# A capability is a name with a value (SKEPU_CAP_<name>, ON or OFF) and the reason that it is off (SKEPU_CAP_<name>_WHY). A test that needs capabilities
# that the host lacks is not an error and not an expected failure: it is registered as a placeholder that ctest reports as skipped, with the reason, so that
# what is not tested on this host is seen at every run.
#
#   skepu_gate(<ok> <test-name> <capability>...)   <ok> is TRUE if the host has all of them; otherwise the placeholder <test-name> is registered
#   skepu_add_skipped_test(<test-name> <capabilities>...)
#   skepu_collect_capabilities()                     the text of the capabilities for the configuration summary (skepu_print_config)
#   skepu_note_skipped_example(<name> <build> <reasons>...)   an example that is not built, for the summary
#
# What can only be seen when the test runs (no device that works, a device that is too small) is for the test itself: it prints the reason and exits with 77
# (SKEPU_TEST_SKIP in tests/helpers/skip.hpp); the tests of skepu_add_test have SKIP_RETURN_CODE 77.
#
# The capabilities: openmp, threads, opencl (the library), opencl_device (a GPU or accelerator that SkePU would use), opencl_fp64 (one with double
# precision), cuda (a compiler), metal (the backend is built), metal_device and unified_memory (a Metal device, and memory shared with the CPU), metal_fp64
# (never: Metal has no double), mpi (the compiler wrappers), mpi_run (and mpiexec). A CUDA device is not probed here: a test that needs one stops by itself.

set(SKEPU_TEST_SKIP_CODE 77)

function(skepu_capability name available why)
	if(available)
		set(SKEPU_CAP_${name} ON PARENT_SCOPE)
	else()
		set(SKEPU_CAP_${name} OFF PARENT_SCOPE)
	endif()
	set(SKEPU_CAP_${name}_WHY "${why}" PARENT_SCOPE)
	set_property(GLOBAL APPEND PROPERTY SKEPU_CAPABILITIES ${name})
endfunction()

skepu_capability(openmp "${SKEPU_OPENMP}" "OpenMP was not found")

find_package(Threads QUIET)
skepu_capability(threads "${Threads_FOUND}" "no thread library was found")

skepu_capability(opencl "${SKEPU_OPENCL}" "OpenCL was not found")

# The OpenCL devices: a program is run that asks for them as SkePU does. It is not run when cross-compiling, or when it does not build:
# the devices are then assumed to be there, and the tests that need them stop by themselves.
if(NOT SKEPU_OPENCL)
	skepu_capability(opencl_device OFF "OpenCL was not found")
	skepu_capability(opencl_fp64 OFF "OpenCL was not found")
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

skepu_capability(cuda "${SKEPU_CUDA}" "no CUDA compiler was found")

# Metal: whether the backend is built (SKEPU_METAL, set where the backend is), and what the host has: a device, and memory that the GPU shares with the CPU
# (the backend uses the memory of the containers without copying it). A program is run that asks the device, as for OpenCL.
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
			# not probed: the tests that need them stop by themselves
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
# the Metal Shading Language has no double: no device can have it, and nothing is probed
skepu_capability(metal_fp64 OFF "the Metal Shading Language has no double precision")
set(SKEPU_CAP_metal_device_NOTE "found${_skepu_mtl_note}")
set(SKEPU_CAP_unified_memory_NOTE "shared with the CPU${_skepu_mtl_note}")

skepu_capability(mpi "${SKEPU_MPI}" "MPI was not found")
if(SKEPU_MPI AND MPIEXEC_EXECUTABLE)
	skepu_capability(mpi_run ON "")
elseif(SKEPU_MPI)
	skepu_capability(mpi_run OFF "no mpiexec was found")
else()
	skepu_capability(mpi_run OFF "MPI was not found")
endif()

# A placeholder for a test that cannot run on this host: ctest reports it as skipped, with the reason.
function(skepu_add_skipped_test name)
	set(_reasons)
	foreach(_c IN LISTS ARGN)
		list(APPEND _reasons "${_c}: ${SKEPU_CAP_${_c}_WHY}")
	endforeach()
	string(REPLACE ";" ", " _reasons "${_reasons}")
	# the reason is $0 of the shell, so that nothing in it needs quoting
	add_test(NAME ${name} COMMAND sh -c "echo \"SKIPPED: $0\"; exit ${SKEPU_TEST_SKIP_CODE}" "${_reasons}")
	set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE ${SKEPU_TEST_SKIP_CODE} LABELS skipped)
	string(REPLACE ";" "," _names "${ARGN}")
	set_property(GLOBAL APPEND PROPERTY SKEPU_SKIPPED_TESTS "${name}|${_names}")
endfunction()

# <ok> is TRUE if the host has the capabilities, and FALSE after registering the placeholder if it lacks any.
function(skepu_gate ok test)
	set(_missing)
	foreach(_c IN LISTS ARGN)
		if(NOT DEFINED SKEPU_CAP_${_c})
			message(FATAL_ERROR "skepu_gate(${test}): unknown capability '${_c}'")
		endif()
		if(NOT SKEPU_CAP_${_c})
			list(APPEND _missing ${_c})
		endif()
	endforeach()
	if(_missing)
		skepu_add_skipped_test(${test} ${_missing})
		set(${ok} FALSE PARENT_SCOPE)
	else()
		set(${ok} TRUE PARENT_SCOPE)
	endif()
endfunction()

# The OpenCL backends use another backend for a skeleton with double precision on a device that has none, and say so on stderr: a test that passes that way
# does not test the OpenCL backend. With SKEPU_STRICT_BACKEND set it is an error instead, and the tests are run so unless `_t_LENIENT` is set (the test is about
# the fallback). A test that uses doubles is declared with DOUBLES (skepu_add_test), and is skipped on the OpenCL backends where no device has them.
function(skepu_test_strict test)
	if(NOT _t_LENIENT)
		set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT SKEPU_STRICT_BACKEND=1)
	endif()
endfunction()

# An example that is not built, with the reasons, for the configuration summary (<build>: sequential, parallel or mpi).
function(skepu_note_skipped_example name build)
	string(REPLACE ";" ", " _why "${ARGN}")
	set_property(GLOBAL APPEND PROPERTY SKEPU_EXAMPLES_SKIPPED "${name}|${build}|${_why}")
endfunction()

# A test that stops with the skip code (see above) is reported as skipped.
function(skepu_test_may_skip test)
	set_tests_properties(${test} PROPERTIES SKIP_RETURN_CODE ${SKEPU_TEST_SKIP_CODE})
endfunction()

# "<title>" with a line of the same length under it, indented as the blocks of skepu_print_config are.
function(skepu_summary_heading out title)
	string(LENGTH "${title}" _length)
	string(REPEAT "-" ${_length} _line)
	set(${out} "\n    ${title}\n    ${_line}" PARENT_SCOPE)
endfunction()

# The text for the configuration summary (skepu_print_config), after the backends: the capabilities with what lacks, and how many tests are skipped for it.
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
	# the examples that are not built: printed after the summary of the examples
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
