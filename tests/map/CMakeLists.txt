skepu_add_executable(mvmult_cpu_test
	SKEPUSRC mvmult.cpp)
target_link_libraries(mvmult_cpu_test
	PRIVATE catch2_main)
add_test(mvmult_cpu mvmult_cpu_test)

if(SKEPU_OPENMP)
	skepu_add_executable(mvmult_openmp_test
		OpenMP
		SKEPUSRC mvmult.cpp)
	target_link_libraries(mvmult_openmp_test
		PRIVATE catch2_main)
	add_test(mvmult_openmp mvmult_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mvmult_cuda_test
		CUDA
		SKEPUSRC mvmult.cpp)
	target_link_libraries(mvmult_cuda_test
		PRIVATE catch2_main)
	add_test(mvmult_cuda mvmult_cuda_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(mvmult_opencl_test
		OpenCL
		SKEPUSRC mvmult.cpp)
	target_link_libraries(mvmult_opencl_test
		PRIVATE catch2_main)
	add_test(mvmult_opencl mvmult_opencl_test)
endif()
