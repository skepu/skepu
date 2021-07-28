# ------------------------------------------------
#   Reduce fundamentals
# ------------------------------------------------

skepu_add_executable(reduce_cpu_test SKEPUSRC reduce.cpp)
target_link_libraries(reduce_cpu_test PRIVATE catch2_main)
add_test(reduce_cpu reduce_cpu_test)

if(SKEPU_OPENMP)
	skepu_add_executable(reduce_openmp_test OpenMP SKEPUSRC reduce.cpp)
	target_link_libraries(reduce_openmp_test PRIVATE catch2_main)
	add_test(reduce_openmp reduce_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(reduce_cuda_test CUDA SKEPUSRC reduce.cpp)
	target_link_libraries(reduce_cuda_test PRIVATE catch2_main)
	add_test(reduce_cuda reduce_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(reduce_opencl_test OpenCL SKEPUSRC reduce.cpp)
	target_link_libraries(reduce_opencl_test PRIVATE catch2_main)
	add_test(reduce_opencl reduce_opencl_test)
endif()

# ------------------------------------------------
#   Reduce start value
# ------------------------------------------------

skepu_add_executable(startval_cpu_test SKEPUSRC startval.cpp)
target_link_libraries(startval_cpu_test PRIVATE catch2_main)
add_test(startval_cpu startval_cpu_test)

if(SKEPU_OPENMP)
	skepu_add_executable(startval_openmp_test OpenMP SKEPUSRC startval.cpp)
	target_link_libraries(startval_openmp_test PRIVATE catch2_main)
	add_test(startval_openmp startval_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(startval_cuda_test CUDA SKEPUSRC startval.cpp)
	target_link_libraries(startval_cuda_test PRIVATE catch2_main)
	add_test(startval_cuda startval_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(startval_opencl_test OpenCL SKEPUSRC startval.cpp)
	target_link_libraries(startval_opencl_test PRIVATE catch2_main)
	add_test(startval_opencl startval_opencl_test)
endif()

# ------------------------------------------------
#   Reduce with 'or' user function
# ------------------------------------------------

skepu_add_executable(orreduce_cpu_test SKEPUSRC orreduce.cpp)
target_link_libraries(orreduce_cpu_test PRIVATE catch2_main)
add_test(orreduce_cpu orreduce_cpu_test)

if(SKEPU_OPENMP)
	skepu_add_executable(orreduce_openmp_test OpenMP SKEPUSRC orreduce.cpp)
	target_link_libraries(orreduce_openmp_test PRIVATE catch2_main)
	add_test(orreduce_openmp orreduce_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(orreduce_cuda_test CUDA SKEPUSRC orreduce.cpp)
	target_link_libraries(orreduce_cuda_test PRIVATE catch2_main)
	add_test(orreduce_cuda orreduce_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(orreduce_opencl_test OpenCL SKEPUSRC orreduce.cpp)
	target_link_libraries(orreduce_opencl_test PRIVATE catch2_main)
	add_test(orreduce_opencl orreduce_opencl_test)
endif()