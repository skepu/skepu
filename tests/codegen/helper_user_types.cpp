// A struct that only a helper function of the operator uses (the operator itself does not name it): the OpenCL kernel of the
// skeleton must define it also.
#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>

struct Pair
{
	int low, high;
};

int pair_sum(int a, int b)
{
	Pair p;
	p.low = a;
	p.high = b;
	return p.low + p.high;
}

int add_through_pair(int a, int b)
{
	return pair_sum(a, b);
}

auto scan_through_pair = skepu::Scan(add_through_pair);
auto reduce_through_pair = skepu::Reduce(add_through_pair);

TEST_CASE("helper_user_types: Scan and Reduce with an operator that calls a function that uses a struct")
{
	const size_t size = 100;
	skepu::Vector<int> v(size, 1), r(size);

	scan_through_pair.setScanMode(skepu::ScanMode::Inclusive);
	REQUIRE_NOTHROW(scan_through_pair(r, v));
	int result = reduce_through_pair(v);
	skepu::external(skepu::read(r), [&]
	{
		for (size_t i = 0; i < size; ++i)
			REQUIRE(r(i) == (int)i + 1);
		REQUIRE(result == (int)size);
	});
}
