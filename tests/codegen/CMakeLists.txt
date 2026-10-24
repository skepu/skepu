
# ------------------------------------------------
#   Built-in math functions
# ------------------------------------------------

# fundamental_types: fundamental_types, builtins (one program)
skepu_add_test(fundamental_types EXCLUDE sequential)


# ------------------------------------------------
#   User types that only the functions called by an operator use
# ------------------------------------------------

skepu_add_test(helper_user_types EXCLUDE sequential)


# ------------------------------------------------
#   Code on the line of a skeleton instance declaration
# ------------------------------------------------

# proxies: proxies, same_line (one program)
skepu_add_test(proxies EXCLUDE sequential)


# ------------------------------------------------
#   Anonymous skeleton instances (created and called in one expression, with chained settings)
# ------------------------------------------------

# inline_skeletons: inline_skeletons, inline_skeletons_all (one program)
skepu_add_test(inline_skeletons EXCLUDE sequential)


# ------------------------------------------------
#   Linking multiple precompiled files
# ------------------------------------------------

# TEST-ERROR: This does not work with the current tracing implementation.

skepu_add_executable(linking_cpu_test SKEPUSRC linking-1.cpp linking-2.cpp)
target_link_libraries(linking_cpu_test PRIVATE catch2_main)
add_test(linking_cpu linking_cpu_test)

# ------------------------------------------------
#   Tracing in a program of several source files
# ------------------------------------------------

# Two precompiled files that write to one trace, and a plain file that includes SkePU: the
# plain one is compiled without tracing support (SKEPU_TRACING is only honoured in
# precompiled files), and the program must still run, exit and write the trace of both
# precompiled files. (It aborted at exit, in the destructor of the tracer.)
skepu_add_executable(tracing_multi_test SKEPUSRC tracing_multi_main.cpp tracing_multi_other.cpp
	SRC tracing_multi_plain.cpp)
target_compile_definitions(tracing_multi_test PRIVATE SKEPU_TRACING)
add_test(NAME tracing_multi COMMAND tracing_multi_test)
set_tests_properties(tracing_multi PROPERTIES
	ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_multi.json"
	FIXTURES_SETUP tracing_multi_run)
add_test(NAME tracing_multi_trace COMMAND ${CMAKE_COMMAND}
	-DTRACE=${CMAKE_CURRENT_BINARY_DIR}/tracing_multi.json -DCALLS=2
	-P ${CMAKE_CURRENT_LIST_DIR}/check_trace.cmake)
set_tests_properties(tracing_multi_trace PROPERTIES FIXTURES_REQUIRED tracing_multi_run)

# Events from several threads: the tracer must take all of them.
find_package(Threads REQUIRED)
skepu_add_executable(tracing_threads_test SKEPUSRC tracing_threads.cpp)
target_compile_definitions(tracing_threads_test PRIVATE SKEPU_TRACING)
target_link_libraries(tracing_threads_test PRIVATE Threads::Threads)
add_test(NAME tracing_threads COMMAND tracing_threads_test)
set_tests_properties(tracing_threads PROPERTIES
	ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_threads.json"
	FIXTURES_SETUP tracing_threads_run)
add_test(NAME tracing_threads_trace COMMAND ${CMAKE_COMMAND}
	-DTRACE=${CMAKE_CURRENT_BINARY_DIR}/tracing_threads.json -DTYPE=allocation -DCALLS=16000 -DEXACT=ON
	-P ${CMAKE_CURRENT_LIST_DIR}/check_trace.cmake)
set_tests_properties(tracing_threads_trace PROPERTIES FIXTURES_REQUIRED tracing_threads_run)

# ------------------------------------------------
#   User function in included file
# ------------------------------------------------

skepu_add_executable(included_uf_cpu_test SKEPUSRC included_uf.cpp)
target_link_libraries(included_uf_cpu_test PRIVATE catch2_main)
target_include_directories(included_uf_cpu_test PRIVATE ./)
add_test(included_uf included_uf_cpu_test)

# ------------------------------------------------
#   Lambda expressions in skeleton instances
# ------------------------------------------------

skepu_add_test(lambda EXCLUDE sequential)

# ------------------------------------------------
#   Skeleton instances passed by reference to function templates
# ------------------------------------------------

skepu_add_test(skeleton_reference)

# ------------------------------------------------
#   Namespace in skeleton instances
# ------------------------------------------------

# namespace: namespace, objectoriented (one program)
skepu_add_test(namespace EXCLUDE sequential)


# ------------------------------------------------
#   Object-oriented SkePU usage
# ------------------------------------------------


# ------------------------------------------------
#   User types
# ------------------------------------------------

# codegen_types: library_types, usertype, namespaced_types, namespaced_types_same_name (one program)
skepu_add_test(codegen_types EXCLUDE sequential)

# ------------------------------------------------
#   User type that is only a return type
# ------------------------------------------------

# template_instances_types: template_instances_types, multiple_return_types, template_instances, usertype_return (one program)
skepu_add_test(template_instances_types EXCLUDE hybrid_opencl hybrid_cuda)


# ------------------------------------------------
#   Skeleton in a function template, several instantiations
# ------------------------------------------------


# Instantiated with different types (user functions, kernels), non-type parameters, Scan and MapOverlap


# User constants in the user functions of every skeleton, with several types
# userconstants_element_types: userconstants, element_types (one program)
skepu_add_test(userconstants_element_types)


# Regions of several threads that overlap in time: events belong to the region of their own thread.
skepu_add_executable(tracing_regions_threads_test SKEPUSRC tracing_regions_threads.cpp)
target_compile_definitions(tracing_regions_threads_test PRIVATE SKEPU_TRACING)
target_link_libraries(tracing_regions_threads_test PRIVATE Threads::Threads)
add_test(NAME tracing_regions_threads COMMAND tracing_regions_threads_test)
set_tests_properties(tracing_regions_threads PROPERTIES
	ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_regions_threads.json"
	FIXTURES_SETUP tracing_regions_threads_run)
add_test(NAME tracing_regions_threads_trace COMMAND ${CMAKE_COMMAND}
	-DTRACE=${CMAKE_CURRENT_BINARY_DIR}/tracing_regions_threads.json -DREGIONS=4 -DEVENTS=200
	-P ${CMAKE_CURRENT_LIST_DIR}/check_trace_regions.cmake)
set_tests_properties(tracing_regions_threads_trace PROPERTIES FIXTURES_REQUIRED tracing_regions_threads_run)

# The trace file name of a module (SKEPU_TRACE_MODULE), %m in SKEPU_TRACE_FILE, and flushing the trace.
foreach(_variant default_module env_module env_main)
	skepu_add_executable(tracing_file_names_${_variant}_test SKEPUSRC tracing_file_names.cpp)
	target_compile_definitions(tracing_file_names_${_variant}_test PRIVATE SKEPU_TRACING)
	if(NOT _variant STREQUAL env_main)
		target_compile_definitions(tracing_file_names_${_variant}_test PRIVATE SKEPU_TRACE_MODULE=mymod)
	endif()
	add_test(NAME tracing_file_names_${_variant} COMMAND tracing_file_names_${_variant}_test
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
	if(_variant STREQUAL default_module)
		set(_expected skepu_trace_mymod.json)
		set_tests_properties(tracing_file_names_${_variant} PROPERTIES ENVIRONMENT_MODIFICATION "SKEPU_TRACE_FILE=unset:")
	elseif(_variant STREQUAL env_module)
		set(_expected tracing_names_mymod.json)
		set_tests_properties(tracing_file_names_${_variant} PROPERTIES
			ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_names_%m.json")
	else()
		set(_expected tracing_names_main.json)
		set_tests_properties(tracing_file_names_${_variant} PROPERTIES
			ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_names_%m.json")
	endif()
	set_tests_properties(tracing_file_names_${_variant} PROPERTIES FIXTURES_SETUP tracing_file_names_${_variant}_run)
	add_test(NAME tracing_file_names_${_variant}_trace COMMAND ${CMAKE_COMMAND}
		-DTRACE=${CMAKE_CURRENT_BINARY_DIR}/${_expected} -DTYPE=allocation -DCALLS=2 -DEXACT=ON
		-P ${CMAKE_CURRENT_LIST_DIR}/check_trace.cmake)
	set_tests_properties(tracing_file_names_${_variant}_trace PROPERTIES FIXTURES_REQUIRED tracing_file_names_${_variant}_run)
endforeach()

# What the trace says about each skeleton, on each backend (the elements, parameters and names are the same on all)
foreach(_backend cpu openmp opencl)
	if(_backend STREQUAL openmp AND NOT SKEPU_OPENMP)
		continue()
	endif()
	if(_backend STREQUAL opencl AND NOT SKEPU_OPENCL)
		continue()
	endif()
	set(_flags)
	if(_backend STREQUAL openmp)
		set(_flags OpenMP)
	elseif(_backend STREQUAL opencl)
		set(_flags OpenCL)
	endif()
	skepu_add_executable(tracing_skeletons_${_backend}_test ${_flags} SKEPUSRC tracing_skeletons.cpp)
	target_link_libraries(tracing_skeletons_${_backend}_test PRIVATE catch2_main)
	target_compile_definitions(tracing_skeletons_${_backend}_test PRIVATE SKEPU_TRACING)
	add_test(NAME tracing_skeletons_${_backend} COMMAND tracing_skeletons_${_backend}_test)
	set_tests_properties(tracing_skeletons_${_backend} PROPERTIES
		ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_skeletons_${_backend}.json")
endforeach()

# Where the skeleton instances and their user functions are defined (the table of the instances in the header of the trace)
skepu_add_executable(tracing_definitions_test SKEPUSRC tracing_definitions.cpp)
target_link_libraries(tracing_definitions_test PRIVATE catch2_main)
target_compile_definitions(tracing_definitions_test PRIVATE SKEPU_TRACING)
add_test(NAME tracing_definitions COMMAND tracing_definitions_test)
set_tests_properties(tracing_definitions PROPERTIES
	ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_definitions.json")

# ... at the lines that a #line directive says (the code that another program generates)
skepu_add_executable(tracing_definitions_line_test SKEPUSRC tracing_definitions_line.cpp)
target_link_libraries(tracing_definitions_line_test PRIVATE catch2_main)
target_compile_definitions(tracing_definitions_line_test PRIVATE SKEPU_TRACING)
add_test(NAME tracing_definitions_line COMMAND tracing_definitions_line_test)
set_tests_properties(tracing_definitions_line PROPERTIES
	ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_definitions_line.json")

# A skeleton instance that is named again keeps the names of its earlier calls
skepu_add_executable(tracing_instance_labels_test SKEPUSRC tracing_instance_labels.cpp)
target_link_libraries(tracing_instance_labels_test PRIVATE catch2_main)
target_compile_definitions(tracing_instance_labels_test PRIVATE SKEPU_TRACING)
add_test(NAME tracing_instance_labels COMMAND tracing_instance_labels_test)
set_tests_properties(tracing_instance_labels PROPERTIES
	ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_instance_labels.json")

# The calls of the library functions (blas, randomize, dnn) are regions at the places of the calls
skepu_add_executable(tracing_library_regions_test SKEPUSRC tracing_library_regions.cpp)
target_link_libraries(tracing_library_regions_test PRIVATE catch2_main)
target_compile_definitions(tracing_library_regions_test PRIVATE SKEPU_TRACING)
add_test(NAME tracing_library_regions COMMAND tracing_library_regions_test)
set_tests_properties(tracing_library_regions PROPERTIES
	ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_library_regions.json")

# The containers that are declared as variables are named after the variables in the trace
skepu_add_executable(tracing_container_labels_test SKEPUSRC tracing_container_labels.cpp)
target_link_libraries(tracing_container_labels_test PRIVATE catch2_main)
target_compile_definitions(tracing_container_labels_test PRIVATE SKEPU_TRACING)
add_test(NAME tracing_container_labels COMMAND tracing_container_labels_test)
set_tests_properties(tracing_container_labels PROPERTIES
	ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_container_labels.json")

# skepu::multiple of built-in types with names of several words (the precompiler removed the first letter of the names of the aliases it makes)

# Skeletons with double precision on OpenCL devices that have it, and on those that do not (they run on another backend, or it is an error in strict mode)
skepu_add_test(opencl_double_fallback BACKENDS opencl)

# Elements of the types signed char and long long (the OpenCL names of the types), with rows and columns as arguments

# The builds of the OpenCL kernels are events of their skeleton instances
if(SKEPU_OPENCL)
	skepu_add_executable(tracing_compile_test OpenCL SKEPUSRC tracing_compile.cpp)
	target_link_libraries(tracing_compile_test PRIVATE catch2_main)
	target_compile_definitions(tracing_compile_test PRIVATE SKEPU_TRACING)
	add_test(NAME tracing_compile COMMAND tracing_compile_test)
	set_tests_properties(tracing_compile PROPERTIES
		ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_compile.json")
endif()

# The values of the scalars (results of reductions, norms) are in the trace
skepu_add_executable(tracing_scalar_values_test SKEPUSRC tracing_scalar_values.cpp)
target_link_libraries(tracing_scalar_values_test PRIVATE catch2_main)
target_compile_definitions(tracing_scalar_values_test PRIVATE SKEPU_TRACING)
add_test(NAME tracing_scalar_values COMMAND tracing_scalar_values_test)
set_tests_properties(tracing_scalar_values PROPERTIES
	ENVIRONMENT "SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/tracing_scalar_values.json")


# ------------------------------------------------
#   What skepu-tool refuses
#   (run the tool on a source and check its status and its message; needs the tool to be built)
# ------------------------------------------------

list(JOIN SKEPU_TOOL_SYSTEM_INCLUDE_FLAGS "|" _tool_error_flags)
set(_tool_error_flags "${_tool_error_flags}|-I|${CMAKE_SOURCE_DIR}/skepu-headers/src")

function(skepu_add_tool_error_test name source expected)
	set(_options "${ARGN}")
	string(REPLACE ";" "|" _options "${_options}")
	add_test(NAME ${name}
		COMMAND ${CMAKE_COMMAND}
			-DTOOL=${SKEPU_EXECUTABLE}
			-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/tool_errors/${source}
			-DOUT=${CMAKE_CURRENT_BINARY_DIR}/${name}_output
			-DFLAGS=${_tool_error_flags}
			-DEXPECT=${expected}
			-DOPTIONS=${_options}
			-P ${CMAKE_CURRENT_SOURCE_DIR}/expect_tool_error.cmake)
endfunction()

# A user constant is put into the kernels from its initializer: it must be constexpr, and have one
skepu_add_tool_error_test(tool_userconstant_not_constexpr userconstant_not_constexpr.cpp "must be declared constexpr")
skepu_add_tool_error_test(tool_userconstant_no_initializer userconstant_no_initializer.cpp "must be declared constexpr")
# (and one that is as it must be is accepted)
skepu_add_tool_error_test(tool_userconstant_valid userconstant_valid.cpp "")

# The standard library has no counterpart in OpenCL C, only the math functions (see AllowedFunctionNamesCalledInUFs) have
skepu_add_tool_error_test(tool_std_call_opencl std_call.cpp "calls std::stable_sort" -opencl)
skepu_add_tool_error_test(tool_std_call_cpu std_call.cpp "")
