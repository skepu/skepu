# ------------------------------------------------
#   Built-in math functions
# ------------------------------------------------
skepu_add_executable(builtins_cpu_test SKEPUSRC builtins.cpp)
target_link_libraries(builtins_cpu_test PRIVATE catch2_main)
add_test(builtins_cpu builtins_cpu_test)

if(SKEPU_OPENMP)
skepu_add_executable(builtins_openmp_test OpenMP SKEPUSRC builtins.cpp)
target_link_libraries(builtins_openmp_test PRIVATE catch2_main)
add_test(builtins_openmp builtins_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(builtins_cuda_test CUDA SKEPUSRC builtins.cpp)
	target_link_libraries(builtins_cuda_test PRIVATE catch2_main)
	add_test(builtins_cuda builtins_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(builtins_opencl_test OpenCL SKEPUSRC builtins.cpp)
	target_link_libraries(builtins_opencl_test PRIVATE catch2_main)
	add_test(builtins_opencl builtins_opencl_test)
endif()

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

# ------------------------------------------------
#   Lambda expressions in skeleton instances
# ------------------------------------------------

skepu_add_executable(lambda_cpu_test SKEPUSRC lambda.cpp)
target_link_libraries(lambda_cpu_test PRIVATE catch2_main)
add_test(lambda_cpu lambda_cpu_test)

if(SKEPU_OPENMP)
skepu_add_executable(lambda_openmp_test OpenMP SKEPUSRC lambda.cpp)
target_link_libraries(lambda_openmp_test PRIVATE catch2_main)
add_test(lambda_openmp lambda_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(lambda_cuda_test CUDA SKEPUSRC lambda.cpp)
	target_link_libraries(lambda_cuda_test PRIVATE catch2_main)
	add_test(lambda_cuda lambda_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(lambda_opencl_test OpenCL SKEPUSRC lambda.cpp)
	target_link_libraries(lambda_opencl_test PRIVATE catch2_main)
	add_test(lambda_opencl lambda_opencl_test)
endif()

# ------------------------------------------------
#   Namespace in skeleton instances
# ------------------------------------------------

skepu_add_executable(namespace_cpu_test SKEPUSRC namespace.cpp)
target_link_libraries(namespace_cpu_test PRIVATE catch2_main)
add_test(namespace_cpu namespace_cpu_test)

if(SKEPU_OPENMP)
skepu_add_executable(namespace_openmp_test OpenMP SKEPUSRC namespace.cpp)
target_link_libraries(namespace_openmp_test PRIVATE catch2_main)
add_test(namespace_openmp namespace_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(namespace_cuda_test CUDA SKEPUSRC namespace.cpp)
	target_link_libraries(namespace_cuda_test PRIVATE catch2_main)
	add_test(namespace_cuda namespace_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(namespace_opencl_test OpenCL SKEPUSRC namespace.cpp)
	target_link_libraries(namespace_opencl_test PRIVATE catch2_main)
	add_test(namespace_opencl namespace_opencl_test)
endif()

# ------------------------------------------------
#   Object-oriented SkePU usage
# ------------------------------------------------

skepu_add_executable(objectoriented_cpu_test SKEPUSRC objectoriented.cpp)
target_link_libraries(objectoriented_cpu_test PRIVATE catch2_main)
add_test(objectoriented_cpu objectoriented_cpu_test)

if(SKEPU_OPENMP)
skepu_add_executable(objectoriented_openmp_test OpenMP SKEPUSRC objectoriented.cpp)
target_link_libraries(objectoriented_openmp_test PRIVATE catch2_main)
add_test(objectoriented_openmp objectoriented_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(objectoriented_cuda_test CUDA SKEPUSRC objectoriented.cpp)
	target_link_libraries(objectoriented_cuda_test PRIVATE catch2_main)
	add_test(objectoriented_cuda objectoriented_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(objectoriented_opencl_test OpenCL SKEPUSRC objectoriented.cpp)
	target_link_libraries(objectoriented_opencl_test PRIVATE catch2_main)
	add_test(objectoriented_opencl objectoriented_opencl_test)
endif()

# ------------------------------------------------
#   Container proxy access
# ------------------------------------------------

skepu_add_executable(proxies_cpu_test SKEPUSRC proxies.cpp)
target_link_libraries(proxies_cpu_test PRIVATE catch2_main)
add_test(proxies_cpu proxies_cpu_test)

if(SKEPU_OPENMP)
skepu_add_executable(proxies_openmp_test OpenMP SKEPUSRC proxies.cpp)
target_link_libraries(proxies_openmp_test PRIVATE catch2_main)
add_test(proxies_openmp proxies_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(proxies_cuda_test CUDA SKEPUSRC proxies.cpp)
	target_link_libraries(proxies_cuda_test PRIVATE catch2_main)
	add_test(proxies_cuda proxies_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(proxies_opencl_test OpenCL SKEPUSRC proxies.cpp)
	target_link_libraries(proxies_opencl_test PRIVATE catch2_main)
	add_test(proxies_opencl proxies_opencl_test)
endif()

# ------------------------------------------------
#   User types
# ------------------------------------------------

skepu_add_executable(usertype_cpu_test SKEPUSRC usertype.cpp)
target_link_libraries(usertype_cpu_test PRIVATE catch2_main)
add_test(usertype_cpu usertype_cpu_test)

if(SKEPU_OPENMP)
skepu_add_executable(usertype_openmp_test OpenMP SKEPUSRC usertype.cpp)
target_link_libraries(usertype_openmp_test PRIVATE catch2_main)
add_test(usertype_openmp usertype_openmp_test)
endif()

if(SKEPU_CUDA)
	skepu_add_executable(usertype_cuda_test CUDA SKEPUSRC usertype.cpp)
	target_link_libraries(usertype_cuda_test PRIVATE catch2_main)
	add_test(usertype_cuda usertype_cuda_test)
endif()

if(SKEPU_OPENCL)
	skepu_add_executable(usertype_opencl_test OpenCL SKEPUSRC usertype.cpp)
	target_link_libraries(usertype_opencl_test PRIVATE catch2_main)
	add_test(usertype_opencl usertype_opencl_test)
endif()