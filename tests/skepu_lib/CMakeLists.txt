include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ------------------------------------------------
#   BLAS library
# ------------------------------------------------

# The precompiled build of this test does not compile (blas.hpp: "expected nested-name-specifier before 'complex'"). The tests below
# (blas_checked, blas_complex) are the precompiled tests of the library.
skepu_add_test(blas BACKENDS sequential)

# ------------------------------------------------
#   Complex fundamentals
# ------------------------------------------------

skepu_add_test(complex)

# ------------------------------------------------
#   Random fundamentals, using namespace and the random fills of skepu-lib/randomize.hpp
# ------------------------------------------------

skepu_add_test(random)


# ------------------------------------------------
#   The removed member functions of the containers that fill them with random numbers: a call must not compile, and the error says what to use
# ------------------------------------------------

foreach(_case vector matrix matrix_real)
	string(TOUPPER ${_case} _define)
	add_executable(randomize_removed_${_case} EXCLUDE_FROM_ALL randomize_removed.cpp)
	target_link_libraries(randomize_removed_${_case} PRIVATE SkePU::SkePU)
	target_compile_definitions(randomize_removed_${_case} PRIVATE CASE_${_define})
	add_test(NAME randomize_removed_${_case}
		COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target randomize_removed_${_case})
	# (the build fails with the message of the static assertion; a build that succeeds does not match)
	set_tests_properties(randomize_removed_${_case} PROPERTIES
		PASS_REGULAR_EXPRESSION "is removed: use skepu::randomize::uniform"
		RUN_SERIAL TRUE)
endforeach()


# ------------------------------------------------
#   Stream wrappers (skepu-lib/io.hpp): cout, and cin that reads values and containers
# ------------------------------------------------

skepu_add_test(io EXCLUDE cuda hybrid_cuda)


# ------------------------------------------------
#   Argument parser (skepu-lib/args.hpp)
# ------------------------------------------------

skepu_add_test(args EXCLUDE hybrid_opencl)


# ------------------------------------------------
#   The user functions of examples/perlin.cpp: a Map with a PRNG that fills a grid of structs, and a Map that gathers from the grid
# ------------------------------------------------

skepu_add_test(perlin)


# ------------------------------------------------
#   skepu_external
# ------------------------------------------------

skepu_add_test(skepu_external)

# ------------------------------------------------
#   DNN library (skepu-lib/dnn)
#   Gradient checks, Adam, predict/evaluate and a synthetic training run (no data files needed)
# ------------------------------------------------

# With the precompiler: the same tests on the other backends. The backend is selected with the
# environment variable SKEPU_TEST_BACKEND (the test program runs on the sequential backend otherwise).
skepu_add_test(dnn EXCLUDE hybrid_opencl hybrid_cuda)
if(TEST dnn_cpu)
	set_tests_properties(dnn_cpu PROPERTIES ENVIRONMENT "SKEPU_TEST_BACKEND=CPU")
endif()
if(TEST dnn_openmp)
	set_tests_properties(dnn_openmp PROPERTIES ENVIRONMENT "SKEPU_TEST_BACKEND=OpenMP")
endif()
if(TEST dnn_opencl)
	set_tests_properties(dnn_opencl PROPERTIES ENVIRONMENT "SKEPU_TEST_BACKEND=OpenCL")
endif()

# Regression test on the logs of the MNIST / CIFAR-10 / signal examples. Cases whose data sets
# are missing from examples/data are skipped; see dnn_regression/README.md.
add_test(NAME dnn_regression
	COMMAND ${CMAKE_CURRENT_LIST_DIR}/dnn_regression/run.sh --cxx ${CMAKE_CXX_COMPILER} --tol 1e-2)

# ------------------------------------------------
#   BLAS library, checked float routines (blas_float, blas_level1, blas_level23, blas_semantics, blas_phase1: one program)
# ------------------------------------------------

skepu_add_test(blas_checked EXCLUDE hybrid_opencl hybrid_cuda)


# ------------------------------------------------
#   BLAS library, checked complex and mixed precision routines (blas_complex, blas_mixed_precision, blas_complex_triangle: one program)
#   (not on OpenCL: the kernels of the complex numbers are not complete there)
# ------------------------------------------------

skepu_add_test(blas_complex BACKENDS sequential cpu openmp)


# ------------------------------------------------
#   Image filter library (skepu-lib/filter): user types in a namespace
# ------------------------------------------------

skepu_add_test(filter EXCLUDE sequential)
