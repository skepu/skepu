# ------------------------------------------------
#   BLAS library
# ------------------------------------------------

skepu_add_executable(blas_cpu_test SKEPUSRC blas.cpp)
target_link_libraries(blas_cpu_test PRIVATE catch2_main)
add_test(blas_cpu blas_cpu_test)

if(SKEPU_OPENMP)
  skepu_add_executable(blas_openmp_test OpenMP SKEPUSRC blas.cpp)
  target_link_libraries(blas_openmp_test PRIVATE catch2_main)
  add_test(blas_openmp blas_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(blas_cuda_test CUDA SKEPUSRC blas.cpp)
	target_link_libraries(blas_cuda_test PRIVATE catch2_main)
	add_test(blas_cuda blas_cuda_test)
endif()

if(SKEPU_OPENCL)
#	skepu_add_executable(blas_opencl_test OpenCL SKEPUSRC blas.cpp)
#	target_link_libraries(blas_opencl_test PRIVATE catch2_main)
#	add_test(blas_opencl blas_opencl_test)
endif()

# ------------------------------------------------
#   Complex fundamentals
# ------------------------------------------------

skepu_add_executable(complex_cpu_test SKEPUSRC complex.cpp)
target_link_libraries(complex_cpu_test PRIVATE catch2_main)
add_test(complex_cpu complex_cpu_test)

if(SKEPU_OPENMP)
  skepu_add_executable(complex_openmp_test OpenMP SKEPUSRC complex.cpp)
  target_link_libraries(complex_openmp_test PRIVATE catch2_main)
  add_test(complex_openmp complex_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(complex_cuda_test CUDA SKEPUSRC complex.cpp)
	target_link_libraries(complex_cuda_test PRIVATE catch2_main)
	add_test(complex_cuda complex_cuda_test)
endif()

if(SKEPU_OPENCL)
#	skepu_add_executable(complex_opencl_test OpenCL SKEPUSRC complex.cpp)
#	target_link_libraries(complex_opencl_test PRIVATE catch2_main)
#	add_test(complex_opencl complex_opencl_test)
endif()

# ------------------------------------------------
#   Random fundamentals
# ------------------------------------------------

skepu_add_executable(random_cpu_test SKEPUSRC random.cpp)
target_link_libraries(random_cpu_test PRIVATE catch2_main)
add_test(random_cpu random_cpu_test)

if(SKEPU_OPENMP)
  skepu_add_executable(random_openmp_test OpenMP SKEPUSRC random.cpp)
  target_link_libraries(random_openmp_test PRIVATE catch2_main)
  add_test(random_openmp random_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(random_cuda_test CUDA SKEPUSRC random.cpp)
	target_link_libraries(random_cuda_test PRIVATE catch2_main)
	add_test(random_cuda random_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(random_opencl_test OpenCL SKEPUSRC random.cpp)
	target_link_libraries(random_opencl_test PRIVATE catch2_main)
	add_test(random_opencl random_opencl_test)
endif()

# ------------------------------------------------
#   Standard library namespaces
# ------------------------------------------------

skepu_add_executable(using_namespace_cpu_test SKEPUSRC using_namespace.cpp)
target_link_libraries(using_namespace_cpu_test PRIVATE catch2_main)
add_test(using_namespace_cpu using_namespace_cpu_test)

if(SKEPU_OPENMP)
  skepu_add_executable(using_namespace_openmp_test OpenMP SKEPUSRC using_namespace.cpp)
  target_link_libraries(using_namespace_openmp_test PRIVATE catch2_main)
  add_test(using_namespace_openmp using_namespace_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(using_namespace_cuda_test CUDA SKEPUSRC using_namespace.cpp)
	target_link_libraries(using_namespace_cuda_test PRIVATE catch2_main)
	add_test(using_namespace_cuda using_namespace_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(using_namespace_opencl_test OpenCL SKEPUSRC using_namespace.cpp)
	target_link_libraries(using_namespace_opencl_test PRIVATE catch2_main)
	add_test(using_namespace_opencl using_namespace_opencl_test)
endif()