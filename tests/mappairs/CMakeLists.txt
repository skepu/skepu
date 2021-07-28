# ------------------------------------------------
#   MapPairs fundamentals
# ------------------------------------------------

skepu_add_executable(mappairs_cpu_test SKEPUSRC mappairs.cpp)
target_link_libraries(mappairs_cpu_test PRIVATE catch2_main)
add_test(mappairs_cpu mappairs_cpu_test)

if(SKEPU_OPENMP)
  skepu_add_executable(mappairs_openmp_test OpenMP SKEPUSRC mappairs.cpp)
  target_link_libraries(mappairs_openmp_test PRIVATE catch2_main)
  add_test(mappairs_openmp mappairs_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mappairs_cuda_test CUDA SKEPUSRC mappairs.cpp)
	target_link_libraries(mappairs_cuda_test PRIVATE catch2_main)
	add_test(mappairs_cuda mappairs_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(mappairs_opencl_test OpenCL SKEPUSRC mappairs.cpp)
	target_link_libraries(mappairs_opencl_test PRIVATE catch2_main)
	add_test(mappairs_opencl mappairs_opencl_test)
endif()

# ------------------------------------------------
#   MapPairs with variadic return
# ------------------------------------------------

skepu_add_executable(mappairs_tuple_cpu_test SKEPUSRC mappairs_tuple.cpp)
target_link_libraries(mappairs_tuple_cpu_test PRIVATE catch2_main)
add_test(mappairs_tuple_cpu mappairs_tuple_cpu_test)

if(SKEPU_OPENMP)
  skepu_add_executable(mappairs_tuple_openmp_test OpenMP SKEPUSRC mappairs_tuple.cpp)
  target_link_libraries(mappairs_tuple_openmp_test PRIVATE catch2_main)
  add_test(mappairs_tuple_openmp mappairs_tuple_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mappairs_tuple_cuda_test CUDA SKEPUSRC mappairs_tuple.cpp)
	target_link_libraries(mappairs_tuple_cuda_test PRIVATE catch2_main)
	add_test(mappairs_tuple_cuda mappairs_tuple_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(mappairs_tuple_opencl_test OpenCL SKEPUSRC mappairs_tuple.cpp)
	target_link_libraries(mappairs_tuple_opencl_test PRIVATE catch2_main)
	add_test(mappairs_tuple_opencl mappairs_tuple_opencl_test)
endif()