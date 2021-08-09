# ------------------------------------------------
#   Map fundamentals
# ------------------------------------------------

skepu_add_executable(map_cpu_test SKEPUSRC map.cpp)
target_link_libraries(map_cpu_test PRIVATE catch2_main)
add_test(map_cpu map_cpu_test)

if(SKEPU_OPENMP)
	skepu_add_executable(map_openmp_test OpenMP SKEPUSRC map.cpp)
	target_link_libraries(map_openmp_test PRIVATE catch2_main)
	add_test(map_openmp map_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(map_cuda_test CUDA SKEPUSRC map.cpp)
	target_link_libraries(map_cuda_test PRIVATE catch2_main)
	add_test(map_cuda map_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(map_opencl_test OpenCL SKEPUSRC map.cpp)
	target_link_libraries(map_opencl_test PRIVATE catch2_main)
	add_test(map_opencl map_opencl_test)
endif()

# ------------------------------------------------
#   Map with index parameter
# ------------------------------------------------

skepu_add_executable(map_index_cpu_test SKEPUSRC map_index.cpp)
target_link_libraries(map_index_cpu_test PRIVATE catch2_main)
add_test(map_index_cpu map_index_cpu_test)

if(SKEPU_OPENMP)
	skepu_add_executable(map_index_openmp_test OpenMP SKEPUSRC map_index.cpp)
	target_link_libraries(map_index_openmp_test PRIVATE catch2_main)
	add_test(map_index_openmp map_index_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(map_index_cuda_test CUDA SKEPUSRC map_index.cpp)
	target_link_libraries(map_index_cuda_test PRIVATE catch2_main)
	add_test(map_index_cuda map_index_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(map_index_opencl_test OpenCL SKEPUSRC map_index.cpp)
	target_link_libraries(map_index_opencl_test PRIVATE catch2_main)
	add_test(map_index_opencl map_index_opencl_test)
endif()

# ------------------------------------------------
#   Matrix-vector multiplication
# ------------------------------------------------

skepu_add_executable(mvmult_cpu_test SKEPUSRC mvmult.cpp)
target_link_libraries(mvmult_cpu_test PRIVATE catch2_main)
add_test(mvmult_cpu mvmult_cpu_test)

if(SKEPU_OPENMP)
	skepu_add_executable(mvmult_openmp_test OpenMP SKEPUSRC mvmult.cpp)
	target_link_libraries(mvmult_openmp_test PRIVATE catch2_main)
	add_test(mvmult_openmp mvmult_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mvmult_cuda_test CUDA SKEPUSRC mvmult.cpp)
	target_link_libraries(mvmult_cuda_test PRIVATE catch2_main)
	add_test(mvmult_cuda mvmult_cuda_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mvmult_opencl_test OpenCL SKEPUSRC mvmult.cpp)
	target_link_libraries(mvmult_opencl_test PRIVATE catch2_main)
	add_test(mvmult_opencl mvmult_opencl_test)
endif()

# ------------------------------------------------
#   Map in place
# ------------------------------------------------

skepu_add_executable(map_in_place_cpu_test SKEPUSRC map_in_place.cpp)
target_link_libraries(map_in_place_cpu_test PRIVATE catch2_main)
add_test(map_in_place_cpu map_in_place_cpu_test)

if(SKEPU_OPENMP)
	skepu_add_executable(map_in_place_openmp_test OpenMP SKEPUSRC map_in_place.cpp)
	target_link_libraries(map_in_place_openmp_test PRIVATE catch2_main)
	add_test(map_in_place_openmp map_in_place_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(map_in_place_cuda_test CUDA SKEPUSRC map_in_place.cpp)
	target_link_libraries(map_in_place_cuda_test PRIVATE catch2_main)
	add_test(map_in_place_cuda map_in_place_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(map_in_place_opencl_test OpenCL SKEPUSRC map_in_place.cpp)
	target_link_libraries(map_in_place_opencl_test PRIVATE catch2_main)
	add_test(map_in_place_opencl map_in_place_opencl_test)
endif()

# ------------------------------------------------
#   Map with variadic return
# ------------------------------------------------

skepu_add_executable(map_tuple_cpu_test SKEPUSRC map_tuple.cpp)
target_link_libraries(map_tuple_cpu_test PRIVATE catch2_main)
add_test(map_tuple_cpu map_tuple_cpu_test)

if(SKEPU_OPENMP)
	skepu_add_executable(map_tuple_openmp_test OpenMP SKEPUSRC map_tuple.cpp)
	target_link_libraries(map_tuple_openmp_test PRIVATE catch2_main)
	add_test(map_tuple_openmp map_tuple_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(map_tuple_cuda_test CUDA SKEPUSRC map_tuple.cpp)
	target_link_libraries(map_tuple_cuda_test PRIVATE catch2_main)
	add_test(map_tuple_cuda map_tuple_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(map_tuple_opencl_test OpenCL SKEPUSRC map_tuple.cpp)
	target_link_libraries(map_tuple_opencl_test PRIVATE catch2_main)
	add_test(map_tuple_opencl map_tuple_opencl_test)
endif()


# ------------------------------------------------
#   Map with stride
# ------------------------------------------------

skepu_add_executable(map_stride_cpu_test SKEPUSRC map_stride.cpp)
target_link_libraries(map_stride_cpu_test PRIVATE catch2_main)
add_test(map_stride_cpu map_stride_cpu_test)

if(SKEPU_OPENMP)
	skepu_add_executable(map_stride_openmp_test OpenMP SKEPUSRC map_stride.cpp)
	target_link_libraries(map_stride_openmp_test PRIVATE catch2_main)
	add_test(map_stride_openmp map_stride_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(map_stride_cuda_test CUDA SKEPUSRC map_stride.cpp)
	target_link_libraries(map_stride_cuda_test PRIVATE catch2_main)
	add_test(map_stride_cuda map_stride_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(map_stride_opencl_test OpenCL SKEPUSRC map_stride.cpp)
	target_link_libraries(map_stride_opencl_test PRIVATE catch2_main)
	add_test(map_stride_opencl map_stride_opencl_test)
endif()