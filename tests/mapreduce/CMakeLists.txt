# ------------------------------------------------
#   MapReduce fundamentals
# ------------------------------------------------

skepu_add_executable(mapreduce_cpu_test SKEPUSRC mapreduce.cpp)
target_link_libraries(mapreduce_cpu_test PRIVATE catch2_main)
add_test(mapreduce_cpu mapreduce_cpu_test)

if(SKEPU_OPENMP)
  skepu_add_executable(mapreduce_openmp_test OpenMP SKEPUSRC mapreduce.cpp)
  target_link_libraries(mapreduce_openmp_test PRIVATE catch2_main)
  add_test(mapreduce_openmp mapreduce_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mapreduce_cuda_test CUDA SKEPUSRC mapreduce.cpp)
	target_link_libraries(mapreduce_cuda_test PRIVATE catch2_main)
	add_test(mapreduce_cuda mapreduce_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(mapreduce_opencl_test OpenCL SKEPUSRC mapreduce.cpp)
	target_link_libraries(mapreduce_opencl_test PRIVATE catch2_main)
	add_test(mapreduce_opencl mapreduce_opencl_test)
endif()