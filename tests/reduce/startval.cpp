#include "../../external/catch2/catch.hpp"

#include <iostream>
#include <skepu>


auto test_sum = skepu::Reduce([](int lhs, int rhs) -> int { return lhs + rhs; });
auto test_max = skepu::Reduce([](int lhs, int rhs) -> int { return lhs < rhs ? rhs : lhs; });
auto test_min = skepu::Reduce([](int lhs, int rhs) -> int { return lhs < rhs ? lhs : rhs; });

// The CHECK function from catch2 doesn't perform implicit conversion
// since it assumes that whatever value we pass to it (including non-primitive ones)
// are what we want to check. Therefore it doesn't convert Scalar<bool> to bool and
// so we have to do it manually.
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
