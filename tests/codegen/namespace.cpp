// Each part is in its own namespace; its test cases are prefixed with its name.

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <test_helper_automatic_setup.hpp>


// ======== namespace ========

namespace namespace_part
{
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



TEST_CASE("namespace: Namespaced skeleton instantiations")
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
}


// ======== objectoriented ========

namespace objectoriented_part
{
template<typename T>
class TestClass
{
	T _val;
	
public:
	
	TestClass(T val): _val{val}
	{}
	
	// Skeletons can be "hidden" inside classes like this.
	// The skeleton instance can also be declared static.
	skepu::Vector<T> scale(skepu::Vector<T> &v)
	{
		auto skel = skepu::Map<1>([](T e, T scale){ return e * scale; });
		skepu::Vector<T> res(v.size());
		return skel(res, v, this->_val);
	}
	
};



TEST_CASE("objectoriented: Object-oriented SkePU usage")
{
	const size_t size{100};
	
	skepu::Vector<int> v1(size);
	
	v1.flush();
	for (size_t i = 0; i < size; ++i)
	{
		v1(i) = i;
	}
	
	TestClass<int> test(5);
	
	auto res = test.scale(v1);
	
	res.flush();
	std::cout << "v1: " << v1 << "\nres: " << res << "\n\n";
	
}
}
