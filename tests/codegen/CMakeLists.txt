
# ------------------------------------------------
#   Built-in math functions
# ------------------------------------------------

skepu_add_test_case_cpu(builtins)
skepu_add_test_case_openmp(builtins)
skepu_add_test_case_opencl(builtins)
skepu_add_test_case_cuda(builtins)
skepu_add_test_case_hybrid_opencl(builtins)
skepu_add_test_case_hybrid_cuda(builtins)

# ------------------------------------------------
#   Fundamental types
# ------------------------------------------------

skepu_add_test_case_cpu(fundamental_types)
skepu_add_test_case_openmp(fundamental_types)
skepu_add_test_case_opencl(fundamental_types)
skepu_add_test_case_cuda(fundamental_types)
skepu_add_test_case_hybrid_opencl(fundamental_types)
skepu_add_test_case_hybrid_cuda(fundamental_types)

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

skepu_add_test_case_cpu(lambda)
skepu_add_test_case_openmp(lambda)
skepu_add_test_case_opencl(lambda)
skepu_add_test_case_cuda(lambda)
#skepu_add_test_case_hybrid_opencl(lambda)
#skepu_add_test_case_hybrid_cuda(lambda)

# ------------------------------------------------
#   Namespace in skeleton instances
# ------------------------------------------------

# TEST-ERROR: skel::instance3 is not generated in the precompiled code

# skepu_add_test_case_cpu(namespace)
# skepu_add_test_case_openmp(namespace)
# skepu_add_test_case_opencl(namespace)
# skepu_add_test_case_cuda(namespace)
# skepu_add_test_case_hybrid_opencl(namespace)
# skepu_add_test_case_hybrid_cuda(namespace)

# ------------------------------------------------
#   Object-oriented SkePU usage
# ------------------------------------------------


# CUDA does not work because TestClass.scale() calls a __global__ kernel. The kernel then relies on the struct::CU function.
# 2 problems arise from this.
# 1. The kernel gets included inside the class and a __global__ function cannot be a member function.
# 2. The kernel calls the function struct::CU but needs to actually call TestClass::struct::CU.
skepu_add_test_case_cpu(objectoriented)
skepu_add_test_case_openmp(objectoriented)
skepu_add_test_case_opencl(objectoriented)
#skepu_add_test_case_cuda(objectoriented)
skepu_add_test_case_hybrid_opencl(objectoriented)
#skepu_add_test_case_hybrid_cuda(objectoriented)

# ------------------------------------------------
#   Container proxy access
# ------------------------------------------------

skepu_add_test_case_cpu(proxies)
skepu_add_test_case_openmp(proxies)
skepu_add_test_case_opencl(proxies)
skepu_add_test_case_cuda(proxies)
skepu_add_test_case_hybrid_opencl(proxies)
skepu_add_test_case_hybrid_cuda(proxies)

# ------------------------------------------------
#   User types
# ------------------------------------------------

skepu_add_test_case_cpu(usertype)
skepu_add_test_case_openmp(usertype)
skepu_add_test_case_opencl(usertype)
skepu_add_test_case_cuda(usertype)
skepu_add_test_case_hybrid_opencl(usertype)
skepu_add_test_case_hybrid_cuda(usertype)
# ------------------------------------------------
#   User type that is only a return type
# ------------------------------------------------

skepu_add_test_case_cpu(usertype_return)
skepu_add_test_case_openmp(usertype_return)
skepu_add_test_case_opencl(usertype_return)

# ------------------------------------------------
#   Skeleton in a function template, several instantiations
# ------------------------------------------------

skepu_add_test_case_cpu(template_instances)
skepu_add_test_case_openmp(template_instances)
skepu_add_test_case_opencl(template_instances)

# Instantiated with different types (user functions, kernels), non-type parameters, Scan and MapOverlap
skepu_add_test_case_sequential(template_instances_types)
skepu_add_test_case_cpu(template_instances_types)
skepu_add_test_case_openmp(template_instances_types)
skepu_add_test_case_opencl(template_instances_types)

# The same with double precision (OpenCL devices do not all have it)
skepu_add_test_case_sequential(template_instances_double)
skepu_add_test_case_cpu(template_instances_double)
skepu_add_test_case_openmp(template_instances_double)

# ------------------------------------------------
#   User types in namespaces (OpenCL type names)
# ------------------------------------------------

skepu_add_test_case_cpu(namespaced_types)
skepu_add_test_case_openmp(namespaced_types)
skepu_add_test_case_opencl(namespaced_types)
skepu_add_test_case_cuda(namespaced_types)
skepu_add_test_case_hybrid_opencl(namespaced_types)
skepu_add_test_case_hybrid_cuda(namespaced_types)

skepu_add_test_case_cpu(namespaced_types_same_name)
skepu_add_test_case_openmp(namespaced_types_same_name)
skepu_add_test_case_opencl(namespaced_types_same_name)
skepu_add_test_case_cuda(namespaced_types_same_name)
skepu_add_test_case_hybrid_opencl(namespaced_types_same_name)
skepu_add_test_case_hybrid_cuda(namespaced_types_same_name)
