#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>

// Two different types with the same name in different namespaces, used in one kernel.

namespace first
{
	struct Value
	{
		float v;
	};
}

namespace second
{
	struct Value
	{
		int a, b;
	};

	namespace nested
	{
		struct Value
		{
			float x, y, z;
		};
	}
}

second::Value combine(first::Value f, second::nested::Value n)
{
	second::Value r;
	r.a = (int)f.v;
	r.b = (int)(n.x + n.y + n.z);
	return r;
}

auto combine_map = skepu::Map<2>(combine);

TEST_CASE("Types with the same name in different namespaces")
{
	const size_t size{10};
	skepu::Vector<first::Value> f(size);
	skepu::Vector<second::nested::Value> n(size);
	skepu::Vector<second::Value> r(size);
	for (size_t i = 0; i < size; ++i)
	{
		f(i).v = float(i);
		n(i).x = 1.f; n(i).y = 2.f; n(i).z = float(i);
	}

	combine_map(r, f, n);

	r.flush();
	for (size_t i = 0; i < size; ++i)
	{
		CHECK(r(i).a == int(i));
		CHECK(r(i).b == int(3 + i));
	}
}
