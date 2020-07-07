#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>


auto sum = skepu::Reduce([](int lhs, int rhs) -> int { return lhs + rhs; });
auto max = skepu::Reduce([](int lhs, int rhs) -> int { return lhs < rhs ? rhs : lhs; });
auto min = skepu::Reduce([](int lhs, int rhs) -> int { return lhs < rhs ? lhs : rhs; });

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
	
	CHECK(sum(v1) == 499500);
	CHECK(max(v1) == N-1);
	CHECK(min(v1) == 0);
	
	
	CHECK(sum(v2) == -500500);
	CHECK(max(v2) == 0);
	CHECK(min(v2) == -1000);
	
	sum.setStartValue(100);
	max.setStartValue(std::numeric_limits<int>::min());
	min.setStartValue(std::numeric_limits<int>::min());
	
	CHECK(sum(v2) == -500500 + 100);
	CHECK(max(v2) == -1);
	CHECK(min(v2) == std::numeric_limits<int>::min());
	
	// TODO: Add tests for Matrix
}
