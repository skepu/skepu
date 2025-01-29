#include <catch2/catch_test_macros.hpp>

#define SKEPU_PRECOMPILED
#include <skepu>

TEST_CASE("Create empty vector")
{
	skepu::Vector<int> vi;
	skepu::Vector<float> vf;
	skepu::Vector<double> vd;

	REQUIRE(vi.size() == 0);
	REQUIRE(vf.size() == 0);
	REQUIRE(vd.size() == 0);

	SECTION("Create vector of size 100K")
	{
		constexpr size_t n = 100000;
		vi = skepu::Vector<int>(n);
		vf = skepu::Vector<float>(n);
		vd = skepu::Vector<double>(n);

		CHECK(vi.size() == n);
		CHECK(vf.size() == n);
		CHECK(vd.size() == n);
	}
}
