#include "../../external/catch2/catch.hpp"

#include <skepu>

// The struct only appears as a return type (no variable or parameter of the type
// in the user function), so the OpenCL source needs it from the return type.
struct Pt
{
	float x, y;
};

Pt make_pt(skepu::Index1D index, float scale)
{
	return (Pt){ index.i * scale, index.i * scale + 1 };
}

auto gen = skepu::Map<0>(make_pt);

TEST_CASE("User type only in the return type")
{
	const size_t size{10};

	skepu::Vector<Pt> v(size);
	gen(v, 2.f);
	v.updateHost();

	for (size_t i = 0; i < size; ++i)
	{
		REQUIRE(v(i).x == i * 2.f);
		REQUIRE(v(i).y == i * 2.f + 1);
	}
}
