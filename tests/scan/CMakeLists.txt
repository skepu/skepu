# ------------------------------------------------
#   Scan fundamentals
# ------------------------------------------------

skepu_add_executable(scan_cpu_test SKEPUSRC scan.cpp)
target_link_libraries(scan_cpu_test PRIVATE catch2_main)
add_test(scan_cpu scan_cpu_test)

if(SKEPU_OPENMP)
  skepu_add_executable(scan_openmp_test OpenMP SKEPUSRC scan.cpp)
  target_link_libraries(scan_openmp_test PRIVATE catch2_main)
  add_test(scan_openmp scan_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(scan_cuda_test CUDA SKEPUSRC scan.cpp)
	target_link_libraries(scan_cuda_test PRIVATE catch2_main)
	add_test(scan_cuda scan_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(scan_opencl_test OpenCL SKEPUSRC scan.cpp)
	target_link_libraries(scan_opencl_test PRIVATE catch2_main)
	add_test(scan_opencl scan_opencl_test)
endif()