# ------------------------------------------------
#   MapPairsReduce fundamentals
# ------------------------------------------------

skepu_add_executable(mappairsreduce_cpu_test SKEPUSRC mappairsreduce.cpp)
target_link_libraries(mappairsreduce_cpu_test PRIVATE catch2_main)
add_test(mappairsreduce_cpu mappairsreduce_cpu_test)

if(SKEPU_OPENMP)
  skepu_add_executable(mappairsreduce_openmp_test OpenMP SKEPUSRC mappairsreduce.cpp)
  target_link_libraries(mappairsreduce_openmp_test PRIVATE catch2_main)
  add_test(mappairsreduce_openmp mappairsreduce_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mappairsreduce_cuda_test CUDA SKEPUSRC mappairsreduce.cpp)
	target_link_libraries(mappairsreduce_cuda_test PRIVATE catch2_main)
	add_test(mappairsreduce_cuda mappairsreduce_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(mappairsreduce_opencl_test OpenCL SKEPUSRC mappairsreduce.cpp)
	target_link_libraries(mappairsreduce_opencl_test PRIVATE catch2_main)
	add_test(mappairsreduce_opencl mappairsreduce_opencl_test)
endif()

# ------------------------------------------------
#   MapPairsReduce with variadic return
# ------------------------------------------------

skepu_add_executable(mpr_multi_return_cpu_test SKEPUSRC mpr_multi_return.cpp)
target_link_libraries(mpr_multi_return_cpu_test PRIVATE catch2_main)
add_test(mpr_multi_return_cpu mpr_multi_return_cpu_test)

if(SKEPU_OPENMP)
  skepu_add_executable(mpr_multi_return_openmp_test OpenMP SKEPUSRC mpr_multi_return.cpp)
  target_link_libraries(mpr_multi_return_openmp_test PRIVATE catch2_main)
  add_test(mpr_multi_return_openmp mpr_multi_return_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mpr_multi_return_cuda_test CUDA SKEPUSRC mpr_multi_return.cpp)
	target_link_libraries(mpr_multi_return_cuda_test PRIVATE catch2_main)
	add_test(mpr_multi_return_cuda mpr_multi_return_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(mpr_multi_return_opencl_test OpenCL SKEPUSRC mpr_multi_return.cpp)
	target_link_libraries(mpr_multi_return_opencl_test PRIVATE catch2_main)
	add_test(mpr_multi_return_opencl mpr_multi_return_opencl_test)
endif()