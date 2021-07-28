#include <catch2/catch.hpp>

#include <skepu>

// Test 1
namespace ufs1
{
	int mult_1(int a, int b)
	{
		return a * b;
	}
}

auto instance1 = skepu::Map<2>(ufs1::mult_1); // works (either qualified or unqualified)



// Test 2
namespace ufs2
{
	int inner_mult_2(int a, int b)
	{
		return a * b;
	}
}


int mult_2(int a, int b)
{
	return ufs2::inner_mult_2(a, b); // works (either qualified or unqualified)
}

auto instance2 = skepu::Map<2>(mult_2); // works



// Test 3
namespace skel
{
	int mult_3(int a, int b)
	{
		return a * b;
	}

	auto instance3 = skepu::Map<2>(mult_3); // works
}



TEST_CASE("Namespaced skeleton instantiations")
{
	const size_t size{100};

	skepu::Vector<int> v1(size, 3), v2(size, 7), r(size);

	instance1(r, v1, v2);
  r.flush();
  CHECK(r(0) == 21);
	
	instance2(r, v1, v2);
  r.flush();
  CHECK(r(0) == 21);
	
	skel::instance3(r, v1, v2);
  r.flush();
  CHECK(r(0) == 21);
}
