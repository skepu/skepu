#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>

float mix(float x) { return x - 0.5f; }
float half(float x) { return x * 0.5f; }
float dot(float a, float b) { return a * b; }
float select(float x) { return x + 1.f; }

float combine(float x) { return mix(x) + half(x) + dot(x, x) + select(x); }

auto reserved_map = skepu::Map<1>(combine);

// a user function named like a callable builtin is called instead, here with another result
float max(float a, float b) { return (a > b ? a : b) + 100.f; }
float min(float a, float b) { return (a < b ? a : b) - 100.f; }

float shadowed(float a, float b) { return max(a, b) + min(a, b); }

auto shadowed_map = skepu::Map<2>(shadowed);

TEST_CASE("User functions named like OpenCL builtins")
{
	skepu::Vector<float> v(10, 3.f), r(10);
	reserved_map(r, v);
	r.flush();
	CHECK(r(0) == 2.5f + 1.5f + 9.f + 4.f);
}

TEST_CASE("User functions named like the builtins that user functions may call")
{
	skepu::Vector<float> a(10, 3.f), b(10, 5.f), r(10);
	shadowed_map(r, a, b);
	r.flush();
	CHECK(r(0) == (5.f + 100.f) + (3.f - 100.f));
}
