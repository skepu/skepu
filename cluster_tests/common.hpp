#pragma once
#include "catch.hpp"

#include <skepu>
#include <iostream>
#define N 1024

#include "soffa.hpp"

#define FOR_N for(size_t n : std::vector<size_t>{32, 33, 9, 1024,1000})

#define TEST_STARPU_MATRIX_CONTAINER
#define TEST_VECTOR
#define TEST_MAP_VECTOR
#define TEST_REDUCE

//#define BENCHMARK_1D_DOTPRODUCT // MapReduce not implemented as of yet.
#define BENCHMARK_NBODY
#define BENCHMARK_REDUCE
