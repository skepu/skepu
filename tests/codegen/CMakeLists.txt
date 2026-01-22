
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
skepu_add_test_case_hybrid_opencl(lambda)
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