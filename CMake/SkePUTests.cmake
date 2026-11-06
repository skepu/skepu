# SkePUTests.cmake — SkePU's own test suite. Included by tests/CMakeLists.txt; not installed.
#
# A test whose capabilities (SkePUCapabilities.cmake) the host lacks is registered as a placeholder that ctest reports as skipped, with the reason.
#
#   skepu_add_test(<name> ...)                      test program <name>.cpp for the backends
#   skepu_gate(<ok> <test-name> <capability>...)    <ok> is TRUE if the host has all; otherwise the placeholder <test-name> is registered
#   skepu_add_skipped_test(<test-name> <capabilities>...)
#
# What only a running test can see (no working device, a device too small) is reported by the test: it prints the reason and exits with 77
# (SKEPU_TEST_SKIP, tests/helpers/skip.hpp). Tests of skepu_add_test have SKIP_RETURN_CODE 77.

set(SKEPU_TEST_SKIP_CODE 77)

# Placeholder reported as skipped, with the reasons.
function(skepu_add_skipped_test name)
	set(_reasons)
	foreach(_c IN LISTS ARGN)
		list(APPEND _reasons "${_c}: ${SKEPU_CAP_${_c}_WHY}")
	endforeach()
	string(REPLACE ";" ", " _reasons "${_reasons}")
	# reasons go in $0 to avoid quoting
	add_test(NAME ${name} COMMAND sh -c "echo \"SKIPPED: $0\"; exit ${SKEPU_TEST_SKIP_CODE}" "${_reasons}")
	set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE ${SKEPU_TEST_SKIP_CODE} LABELS skipped)
	string(REPLACE ";" "," _names "${ARGN}")
	set_property(GLOBAL APPEND PROPERTY SKEPU_SKIPPED_TESTS "${name}|${_names}")
endfunction()

# <ok> is FALSE, after registering the placeholder, if a capability is missing.
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

# Without a double-precision device the OpenCL backends fall back to another backend and say so on stderr; a test passing that way tests nothing of OpenCL.
# SKEPU_STRICT_BACKEND makes the fallback an error; set unless `_t_LENIENT` (the test is about the fallback). Tests with doubles use DOUBLES.
function(skepu_test_strict test)
	if(NOT _t_LENIENT)
		set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT SKEPU_STRICT_BACKEND=1)
	endif()
endfunction()

# exit 77 is reported as skipped
function(skepu_test_may_skip test)
	set_tests_properties(${test} PROPERTIES SKIP_RETURN_CODE ${SKEPU_TEST_SKIP_CODE})
endfunction()

##################################
# Macros for Catch2 test programs
##################################

macro(skepu_add_test_case_sequential name)
	add_executable(${name}_seq_test ${name}.cpp)
	target_link_libraries(${name}_seq_test PRIVATE SkePU::SkePU)
	target_link_libraries(${name}_seq_test PRIVATE catch2_main)
	target_include_directories(${name}_seq_test PRIVATE ${test_helpers_dir})
	add_test(NAME ${name}_seq COMMAND ${name}_seq_test)
endmacro()

macro(skepu_add_test_case_cpu name)
	skepu_add_executable(${name}_cpu_test SKEPUSRC ${name}.cpp)
	target_link_libraries(${name}_cpu_test PRIVATE catch2_main)
	target_include_directories(${name}_cpu_test PRIVATE ${test_helpers_dir})
	add_test(NAME ${name}_cpu COMMAND ${name}_cpu_test)
endmacro()

macro(skepu_add_test_case_openmp name)
	skepu_gate(_gate_ok ${name}_openmp openmp)
	if(_gate_ok)
		skepu_add_executable(${name}_openmp_test OpenMP SKEPUSRC ${name}.cpp)
		target_link_libraries(${name}_openmp_test PRIVATE catch2_main)
		target_include_directories(${name}_openmp_test PRIVATE ${test_helpers_dir})
		add_test(NAME ${name}_openmp COMMAND ${name}_openmp_test)
		skepu_test_may_skip(${name}_openmp)
	endif()
endmacro()

macro(skepu_add_test_case_opencl name)
	skepu_gate(_gate_ok ${name}_opencl opencl opencl_device)
	if(_gate_ok)
		skepu_add_executable(${name}_opencl_test OpenCL SKEPUSRC ${name}.cpp)
		target_link_libraries(${name}_opencl_test PRIVATE catch2_main)
		target_include_directories(${name}_opencl_test PRIVATE ${test_helpers_dir})
		add_test(NAME ${name}_opencl COMMAND ${name}_opencl_test ${_test_args})
		skepu_test_may_skip(${name}_opencl)
		skepu_test_strict(${name}_opencl)
	endif()
endmacro()

macro(skepu_add_test_case_opencl_multi name)
	skepu_gate(_gate_ok ${name}_opencl_multi opencl opencl_device)
	if(_gate_ok)
		skepu_add_executable(${name}_opencl_multi_test OpenCL SKEPUSRC ${name}.cpp)
		target_link_libraries(${name}_opencl_multi_test PRIVATE catch2_main)
		target_include_directories(${name}_opencl_multi_test PRIVATE ${test_helpers_dir})
		target_compile_definitions(${name}_opencl_multi_test PRIVATE SKEPU_SIMULATE_MULTI_GPU)
		add_test(NAME ${name}_opencl_multi COMMAND ${name}_opencl_multi_test ${_test_args})
		skepu_test_may_skip(${name}_opencl_multi)
		skepu_test_strict(${name}_opencl_multi)
	endif()
endmacro()

macro(skepu_add_test_case_cuda name)
	skepu_gate(_gate_ok ${name}_cuda cuda)
	if(_gate_ok)
		skepu_add_executable(${name}_cuda_test CUDA SKEPUSRC ${name}.cpp)
		target_link_libraries(${name}_cuda_test PRIVATE catch2_main)
		target_include_directories(${name}_cuda_test PRIVATE ${test_helpers_dir})
		# tests compare with host results exactly: no fused multiply-add
		target_compile_options(${name}_cuda_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
		add_test(NAME ${name}_cuda COMMAND ${name}_cuda_test)
		skepu_test_may_skip(${name}_cuda)
	endif()
endmacro()

macro(skepu_add_test_case_hybrid_opencl name)
	skepu_gate(_gate_ok ${name}_hybrid_opencl openmp opencl opencl_device)
	if(_gate_ok)
		skepu_add_executable(${name}_hybrid_opencl_test OpenMP OpenCL SKEPUSRC ${name}.cpp)
		target_link_libraries(${name}_hybrid_opencl_test PRIVATE catch2_main)
		target_include_directories(${name}_hybrid_opencl_test PRIVATE ${test_helpers_dir})
		target_compile_definitions(${name}_hybrid_opencl_test PRIVATE SKEPU_SET_DEFAULT_BACKEND_HYBRID)
		add_test(NAME ${name}_hybrid_opencl COMMAND ${name}_hybrid_opencl_test)
		skepu_test_may_skip(${name}_hybrid_opencl)
		skepu_test_strict(${name}_hybrid_opencl)
	endif()
endmacro()

macro(skepu_add_test_case_hybrid_cuda name)
	skepu_gate(_gate_ok ${name}_hybrid_cuda openmp cuda)
	if(_gate_ok)
		skepu_add_executable(${name}_hybrid_cuda_test OpenMP CUDA SKEPUSRC ${name}.cpp)
		target_link_libraries(${name}_hybrid_cuda_test PRIVATE catch2_main)
		target_include_directories(${name}_hybrid_cuda_test PRIVATE ${test_helpers_dir})
		# tests compare with host results exactly: no fused multiply-add
		target_compile_options(${name}_hybrid_cuda_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
		target_compile_definitions(${name}_hybrid_cuda_test PRIVATE SKEPU_SET_DEFAULT_BACKEND_HYBRID)
		add_test(NAME ${name}_hybrid_cuda COMMAND ${name}_hybrid_cuda_test)
		skepu_test_may_skip(${name}_hybrid_cuda)
	endif()
endmacro()


# skepu_add_test(<name> [BACKENDS b1 [b2 ...]] [EXCLUDE b1 [b2 ...]] [ALSO b1 [b2 ...]] [REQUIRES c1 [c2 ...]] [DOUBLES] [LENIENT])
#
# Registers the Catch2 test program <name>.cpp for a set of backends, with a skepu_add_test_case_<backend> macro each:
#     skepu_add_test(map)                                   all the default backends
#     skepu_add_test(mappool EXCLUDE hybrid_cuda)           all the default backends but those
#     skepu_add_test(mapoverlap_edges BACKENDS sequential cpu openmp opencl)    only those
#     skepu_add_test(map ALSO opencl_multi)                 the default backends and those
#     skepu_add_test(dbl REQUIRES opencl_fp64)              skipped, on every backend, where the host has no GPU with double precision
#     skepu_add_test(types DOUBLES)                         cases with doubles in kernels are tagged [fp64] (Catch2); on OpenCL backends without a device for
#                                                           them they are left out and <test>_<backend>_fp64 is a skipped test (OpenCL tests run with
#                                                           SKEPU_STRICT_BACKEND=1, so a fallback is no pass)
#     skepu_add_test(fallback BACKENDS opencl LENIENT)      without SKEPU_STRICT_BACKEND: the test is about the fallback
# Backends: sequential, cpu, openmp, opencl, cuda, hybrid_opencl, hybrid_cuda (default set, SKEPU_TEST_DEFAULT_BACKENDS), and opencl_multi (simulated
# multiple devices; only with ALSO or BACKENDS). A backend the host lacks becomes a skipped placeholder. Comment why a backend is excluded.
set(SKEPU_TEST_DEFAULT_BACKENDS sequential cpu openmp opencl cuda hybrid_opencl hybrid_cuda)
set(SKEPU_TEST_ALL_BACKENDS ${SKEPU_TEST_DEFAULT_BACKENDS} opencl_multi)

function(skepu_add_test name)
	cmake_parse_arguments(_t "DOUBLES;LENIENT" "" "BACKENDS;EXCLUDE;ALSO;REQUIRES" ${ARGN})
	if(_t_UNPARSED_ARGUMENTS)
		message(FATAL_ERROR "skepu_add_test(${name}): unknown arguments: ${_t_UNPARSED_ARGUMENTS}")
	endif()
	if(_t_BACKENDS AND (_t_EXCLUDE OR _t_ALSO))
		message(FATAL_ERROR "skepu_add_test(${name}): BACKENDS is the whole set, it cannot be combined with EXCLUDE or ALSO")
	endif()

	if(_t_BACKENDS)
		set(_backends ${_t_BACKENDS})
	else()
		set(_backends ${SKEPU_TEST_DEFAULT_BACKENDS} ${_t_ALSO})
		foreach(_b IN LISTS _t_EXCLUDE)
			list(REMOVE_ITEM _backends ${_b})
		endforeach()
	endif()

	foreach(_b IN LISTS _t_BACKENDS _t_EXCLUDE _t_ALSO)
		if(NOT _b IN_LIST SKEPU_TEST_ALL_BACKENDS)
			message(FATAL_ERROR "skepu_add_test(${name}): unknown backend '${_b}' (backends: ${SKEPU_TEST_ALL_BACKENDS})")
		endif()
	endforeach()

	# REQUIRES: skipped on every backend where a capability is missing
	set(_missing)
	foreach(_c IN LISTS _t_REQUIRES)
		if(NOT DEFINED SKEPU_CAP_${_c})
			message(FATAL_ERROR "skepu_add_test(${name}): unknown capability '${_c}'")
		endif()
		if(NOT SKEPU_CAP_${_c})
			list(APPEND _missing ${_c})
		endif()
	endforeach()

	foreach(_b IN LISTS _backends)
# [fp64] cases need a device with doubles (hybrid_opencl does not use it for them); left out where there is none, a skipped test stands for them
		set(_test_args)
		set(_double_cap)
		if(_b STREQUAL opencl OR _b STREQUAL opencl_multi)
			set(_double_cap opencl_fp64)
		elseif(_b MATCHES "^metal")
			set(_double_cap metal_fp64)
		endif()
		if(_t_DOUBLES AND _double_cap AND NOT SKEPU_CAP_${_double_cap})
			set(_test_args "~[fp64]")
			skepu_add_skipped_test(${name}_${_b}_fp64 ${_double_cap})
		endif()
		if(_missing)
			if(_b STREQUAL sequential)
				skepu_add_skipped_test(${name}_seq ${_missing})
			else()
				skepu_add_skipped_test(${name}_${_b} ${_missing})
			endif()
		else()
			cmake_language(CALL skepu_add_test_case_${_b} ${name})
		endif()
	endforeach()
endfunction()
