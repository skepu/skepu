# ------------------------------------------------
#   MapOverlap fundamentals
# ------------------------------------------------

skepu_add_executable(mapoverlap_cpu_test SKEPUSRC mapoverlap.cpp)
target_link_libraries(mapoverlap_cpu_test PRIVATE catch2_main)
add_test(mapoverlap_cpu mapoverlap_cpu_test)

if(SKEPU_OPENMP)
skepu_add_executable(mapoverlap_openmp_test OpenMP SKEPUSRC mapoverlap.cpp)
target_link_libraries(mapoverlap_openmp_test PRIVATE catch2_main)
add_test(mapoverlap_openmp mapoverlap_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mapoverlap_cuda_test CUDA SKEPUSRC mapoverlap.cpp)
	target_link_libraries(mapoverlap_cuda_test PRIVATE catch2_main)
	add_test(mapoverlap_cuda mapoverlap_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(mapoverlap_opencl_test OpenCL SKEPUSRC mapoverlap.cpp)
	target_link_libraries(mapoverlap_opencl_test PRIVATE catch2_main)
	add_test(mapoverlap_opencl mapoverlap_opencl_test)
endif()

# ------------------------------------------------
#   MapOverlap in place
# ------------------------------------------------

skepu_add_executable(mapoverlap_in_place_cpu_test SKEPUSRC mapoverlap_in_place.cpp)
target_link_libraries(mapoverlap_in_place_cpu_test PRIVATE catch2_main)
add_test(mapoverlap_in_place_cpu mapoverlap_in_place_cpu_test)

if(SKEPU_OPENMP)
skepu_add_executable(mapoverlap_in_place_openmp_test OpenMP SKEPUSRC mapoverlap_in_place.cpp)
target_link_libraries(mapoverlap_in_place_openmp_test PRIVATE catch2_main)
add_test(mapoverlap_in_place_openmp mapoverlap_in_place_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mapoverlap_in_place_cuda_test CUDA SKEPUSRC mapoverlap_in_place.cpp)
	target_link_libraries(mapoverlap_in_place_cuda_test PRIVATE catch2_main)
	add_test(mapoverlap_in_place_cuda mapoverlap_in_place_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(mapoverlap_in_place_opencl_test OpenCL SKEPUSRC mapoverlap_in_place.cpp)
	target_link_libraries(mapoverlap_in_place_opencl_test PRIVATE catch2_main)
	add_test(mapoverlap_in_place_opencl mapoverlap_in_place_opencl_test)
endif()

# ------------------------------------------------
#   MapOverlap with variadic return
# ------------------------------------------------

skepu_add_executable(mapoverlap_tuple_cpu_test SKEPUSRC mapoverlap_tuple.cpp)
target_link_libraries(mapoverlap_tuple_cpu_test PRIVATE catch2_main)
add_test(mapoverlap_tuple_cpu mapoverlap_tuple_cpu_test)

if(SKEPU_OPENMP)
skepu_add_executable(mapoverlap_tuple_openmp_test OpenMP SKEPUSRC mapoverlap_tuple.cpp)
target_link_libraries(mapoverlap_tuple_openmp_test PRIVATE catch2_main)
add_test(mapoverlap_tuple_openmp mapoverlap_tuple_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mapoverlap_tuple_cuda_test CUDA SKEPUSRC mapoverlap_tuple.cpp)
	target_link_libraries(mapoverlap_tuple_cuda_test PRIVATE catch2_main)
	add_test(mapoverlap_tuple_cuda mapoverlap_tuple_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(mapoverlap_tuple_opencl_test OpenCL SKEPUSRC mapoverlap_tuple.cpp)
	target_link_libraries(mapoverlap_tuple_opencl_test PRIVATE catch2_main)
	add_test(mapoverlap_tuple_opencl mapoverlap_tuple_opencl_test)
endif()

# ------------------------------------------------
#   MapOverlap update modes
# ------------------------------------------------

skepu_add_executable(updatemode_cpu_test SKEPUSRC updatemode.cpp)
target_link_libraries(updatemode_cpu_test PRIVATE catch2_main)
add_test(updatemode_cpu updatemode_cpu_test)

if(SKEPU_OPENMP)
skepu_add_executable(updatemode_openmp_test OpenMP SKEPUSRC updatemode.cpp)
target_link_libraries(updatemode_openmp_test PRIVATE catch2_main)
add_test(updatemode_openmp updatemode_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(updatemode_cuda_test CUDA SKEPUSRC updatemode.cpp)
	target_link_libraries(updatemode_cuda_test PRIVATE catch2_main)
	add_test(updatemode_cuda updatemode_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(updatemode_opencl_test OpenCL SKEPUSRC updatemode.cpp)
	target_link_libraries(updatemode_opencl_test PRIVATE catch2_main)
	add_test(updatemode_opencl updatemode_opencl_test)
endif()