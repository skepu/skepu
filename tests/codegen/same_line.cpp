#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>

// The precompiler replaces the declaration of a skeleton instance and must leave the rest of its line alone:
// other declarations and statements on the same line (and the text after them) are part of the program.

namespace
{
	float twice(float x) { return 2 * x; }
	float inc(float x) { return x + 1; }
}

auto first = skepu::Map<1>(twice); int after_first = 7; auto second = skepu::Map<1>(inc); int after_second = 8;

TEST_CASE("Code on the line of a named skeleton instance is kept")
{
	skepu::Vector<float> in(4, 1.f), a(4), b(4);
	auto local = skepu::Map<1>(inc); local(a, in); first(b, a); int trailing = 5; /* a comment */ int more = trailing + 1;
	second(a, in);
	a.flush(); b.flush();

	CHECK(after_first == 7);
	CHECK(after_second == 8);
	CHECK(trailing == 5);
	CHECK(more == 6);
	CHECK(a(0) == 2.f);
	CHECK(b(0) == 4.f);
}
