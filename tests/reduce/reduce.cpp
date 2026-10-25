// Each part is in its own namespace; its test cases are prefixed with its name.

#include "../../external/catch2/catch.hpp"
#define SKEPU_ENABLE_EXCEPTIONS   // (a Reduce with containers of the wrong sizes is an error that the test catches)
#include <skepu>
#include <skepu-lib/randomize.hpp>
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
	skepu::randomize::uniform(m, 0, 10);
	skepu::randomize::uniform(v, 0, 10);

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
	
}
}


// ======== reduce_matrix ========

// The reductions of a matrix to a vector, row-wise and column-wise, with the results checked; and the containers of the wrong sizes
namespace reduce_matrix_part
{
auto sum_int = skepu::Reduce([](int lhs, int rhs) -> int { return lhs + rhs; });

// m(r, c) = 10 r + c: the sum of row r is 40 r + 6, and that of column c is 3 c + 30
skepu::Matrix<int> make_matrix()
{
  skepu::Matrix<int> m(3, 4);
  for (size_t r = 0; r < 3; ++r)
    for (size_t c = 0; c < 4; ++c)
      m(r, c) = int(10 * r + c);
  return m;
}

TEST_CASE("reduce_matrix: rows, columns and the whole matrix")
{
  skepu::Matrix<int> m = make_matrix();
  skepu::Vector<int> rows(3), columns(4);

  sum_int.setReduceMode(skepu::ReduceMode::RowWise);
  sum_int(rows, m);
  rows.updateHost();
  CHECK(rows(0) == 6);
  CHECK(rows(1) == 46);
  CHECK(rows(2) == 86);

  sum_int.setReduceMode(skepu::ReduceMode::ColWise);
  sum_int(columns, m);
  columns.updateHost();
  for (size_t c = 0; c < 4; ++c)
    CHECK(columns(c) == int(3 * c + 30));

  const int whole = sum_int(m);
  CHECK(whole == 138);   // (6 + 46 + 86)
}

TEST_CASE("reduce_matrix: a result vector of the wrong size is an error, and is not written")
{
  skepu::Matrix<int> m = make_matrix();

  skepu::Vector<int> few(2, -1), many(5, -1);   // (the matrix has 3 rows and 4 columns)
  sum_int.setReduceMode(skepu::ReduceMode::RowWise);
  CHECK_THROWS(sum_int(few, m));
  CHECK_THROWS(sum_int(many, m));
  CHECK(few(0) == -1);
  CHECK(few(1) == -1);
  CHECK(many(0) == -1);

  skepu::Vector<int> three(3, -1);   // (the number of the rows is not that of the columns)
  sum_int.setReduceMode(skepu::ReduceMode::ColWise);
  CHECK_THROWS(sum_int(three, m));
  CHECK(three(0) == -1);
  CHECK(three(2) == -1);
}
}
