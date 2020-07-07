#include <catch2/catch.hpp>
#include <skepu>

#include "included_uf_header.h"

auto inverser = skepu::Map<1>(header_inverse);

TEST_CASE("Test user function in included header")
{
	size_t N = 100;
	skepu::Vector<float> v(N), res(100);
	for (size_t i = 0; i < N; ++i)
		v(i) = i+1;
	
	REQUIRE_NOTHROW(inverser(res, v));
	
	for (size_t i = 0; i < N; ++i)
		CHECK(res(i) == 1 / v(i));
}