// Each part is in its own namespace; its test cases are prefixed with its name.

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <test_helper_automatic_setup.hpp>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>
#include <iostream>


// ======== orreduce ========

namespace orreduce_part
{
bool or_reduce (bool a, bool b)
{
	return (a || b);
}

auto reduce = skepu::Reduce(or_reduce);

TEST_CASE("orreduce: Reduce with 'or' user function")
{
  const size_t size{100};

  skepu::Vector<bool> bool_cond(size, false);
  bool_cond[0] = true;
  reduce.setStartValue(false);
  auto res = reduce(bool_cond);
}
}


// ======== reduce ========

namespace reduce_part
{
auto sum = skepu::Reduce(skepu::util::add<float>);
auto max_sum = skepu::Reduce(skepu::util::add<float>, skepu::util::max<float>);

TEST_CASE("reduce: Reduce fundamentals")
{
  const size_t size{100};
	
	skepu::Matrix<float> m(size / 2, size / 2);
	skepu::Vector<float> v(size), rv(size / 2);
	m.randomize(0, 10);
	v.randomize(0, 10);

	// With containers
	float r = sum(v);
	skepu::io::cout << "Sum of v: " << r << "\n";

	r = sum(m);
	skepu::io::cout << "Sum of m: " << r << "\n";

	sum.setReduceMode(skepu::ReduceMode::RowWise);
	sum(rv, m);
	skepu::io::cout << "1D Row-wise reducetion of m: " << rv << "\n";

	sum.setReduceMode(skepu::ReduceMode::ColWise);
	sum(rv,m);
	skepu::io::cout << "1D Column-wise reducetion of m: " << rv << "\n";

	// 2D reduce
	max_sum.setReduceMode(skepu::ReduceMode::RowWise);
	r = max_sum(m);
	skepu::io::cout << "Reduce 2D max row-sum: r = " << r << "\n";

	max_sum.setReduceMode(skepu::ReduceMode::ColWise);
	r = max_sum(m);
	skepu::io::cout << "Reduce 2D max col-sum: r = " << r << "\n";

	r = max_sum(v);
	skepu::io::cout << "Sum of v (Reduce2D): " << r << "\n";
}
}


// ======== startval ========

namespace startval_part
{
auto test_sum = skepu::Reduce([](int lhs, int rhs) -> int { return lhs + rhs; });
auto test_max = skepu::Reduce([](int lhs, int rhs) -> int { return lhs < rhs ? rhs : lhs; });
auto test_min = skepu::Reduce([](int lhs, int rhs) -> int { return lhs < rhs ? lhs : rhs; });

// The CHECK function from catch2 doesn't perform implicit conversion
// since it assumes that whatever value we pass to it (including non-primitive ones)
// are what we want to check. Therefore it doesn't convert Scalar<bool> to bool and
// so we have to do it manually.
TEST_CASE("startval: Initial value in reductions")
{
	size_t constexpr N{1000};

	skepu::Matrix<int> m(N,N);
	skepu::Vector<int> v1(N), v2(N);
	
	for (size_t i = 0; i < N; ++i)
	{
		v1(i) = i;
		v2(i) = -i - 1;
	}
	
	CHECK(static_cast<bool>(test_sum(v1) == 499500));
	CHECK(static_cast<bool>(test_max(v1) == N-1));
	CHECK(static_cast<bool>(test_min(v1) == 0));
	
	
	CHECK(static_cast<bool>(test_sum(v2) == -500500));
	CHECK(static_cast<bool>(test_max(v2) == 0));
	CHECK(static_cast<bool>(test_min(v2) == -1000));
	
	test_sum.setStartValue(100);
	test_max.setStartValue(std::numeric_limits<int>::min());
	test_min.setStartValue(std::numeric_limits<int>::min());
	
	CHECK(static_cast<bool>(test_sum(v2) == -500500 + 100));
	CHECK(static_cast<bool>(test_max(v2) == -1));
	CHECK(static_cast<bool>(test_min(v2) == std::numeric_limits<int>::min()));
	
	// TODO: Add tests for Matrix
}
}
