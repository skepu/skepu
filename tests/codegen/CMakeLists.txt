
# ------------------------------------------------
#   Built-in math functions
# ------------------------------------------------
skepu_add_executable(builtins_cpu_test SKEPUSRC builtins.cpp)
target_link_libraries(builtins_cpu_test PRIVATE catch2_main)
add_test(builtins_cpu builtins_cpu_test)

skepu_add_executable(builtins_openmp_test OpenMP SKEPUSRC builtins.cpp)
target_link_libraries(builtins_openmp_test PRIVATE catch2_main)
add_test(builtins_openmp builtins_openmp_test)

skepu_add_executable(builtins_cuda_test CUDA SKEPUSRC builtins.cpp)
target_link_libraries(builtins_cuda_test PRIVATE catch2_main)
add_test(builtins_cuda builtins_cuda_test)

skepu_add_executable(builtins_opencl_test OpenCL SKEPUSRC builtins.cpp)
target_link_libraries(builtins_opencl_test PRIVATE catch2_main)
add_test(builtins_opencl builtins_opencl_test)

# ------------------------------------------------
#   Linking multiple precompiled files
# ------------------------------------------------
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