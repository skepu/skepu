#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>


auto test_sum = skepu::Reduce([](int lhs, int rhs) -> int { return lhs + rhs; });
auto test_max = skepu::Reduce([](int lhs, int rhs) -> int { return lhs < rhs ? rhs : lhs; });
auto test_min = skepu::Reduce([](int lhs, int rhs) -> int { return lhs < rhs ? lhs : rhs; });

TEST_CASE("Initial value in reductions")
{
	size_t constexpr N{1000};

	skepu::Matrix<int> m(N,N);
	skepu::Vector<int> v1(N), v2(N);
	
	for (size_t i = 0; i < N; ++i)
	{
		v1(i) = i;
		v2(i) = -i - 1;
	}
	
	CHECK(test_sum(v1) == 499500);
	CHECK(test_max(v1) == N-1);
	CHECK(test_min(v1) == 0);
	
	
	CHECK(test_sum(v2) == -500500);
	CHECK(test_max(v2) == 0);
	CHECK(test_min(v2) == -1000);
	
	test_sum.setStartValue(100);
	test_max.setStartValue(std::numeric_limits<int>::min());
	test_min.setStartValue(std::numeric_limits<int>::min());
	
	CHECK(test_sum(v2) == -500500 + 100);
	CHECK(test_max(v2) == -1);
	CHECK(test_min(v2) == std::numeric_limits<int>::min());
	
	// TODO: Add tests for Matrix
}
