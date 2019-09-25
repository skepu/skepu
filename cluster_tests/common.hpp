#pragma once
#include "catch.hpp"

#include <skepu>
#include <iostream>
#define N 1024

#include "soffa.hpp"

#define FOR_N for(size_t n : {32, 33, 9, 1024,1000})

#define TEST_STARPU_MATRIX_CONTAINER
#define TEST_VECTOR
#define TEST_MAP_VECTOR

//#define BENCHMARK_1D_DOT PRODUCT
#define BENCHMARK_NBODY
//#define BENCHMARK_REDUCE
#define TEST_REDUCE
