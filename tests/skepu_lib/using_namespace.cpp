#include "../../external/catch2/catch.hpp"

#include <iostream>
#include <skepu>
#include <skepu-lib/complex.hpp>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>

float test_uf(float a, float b){ return a + b;}


auto test1 = skepu::Map(test_uf);
auto test2 = skepu::Map([](float a, float b){ return a + b;});
auto test3 = skepu::Map<1>(test_uf);
auto test4 = skepu::Map<1>([](float a, float b){ return a + b;});


TEST_CASE("Standard library namespaces")
{
	const size_t size{1000};
	
	skepu::Vector<float> a(size, size), b(size, size), c(size, size);
	
	for (size_t i = 0; i < size; ++i)
	{
		a(i) = i;
		b(i) = 1;
	}
	
	// c is checked after each call: a(i) is i and b(i) is 1
	auto check = [&](auto expected)
	{
		skepu::external(skepu::read(c), [&]
		{
			for (size_t i = 0; i < size; ++i)
				CHECK(c(i) == Approx(expected(i)));
		});
	};
	
	test1(c, a, b);
	check([](size_t i){ return i + 1.f; });
	
	test2(c, a, b);
	check([](size_t i){ return i + 1.f; });
	
	test3(c, a, 100);
	check([](size_t i){ return i + 100.f; });
	
	test4(c, a, 100);
	check([](size_t i){ return i + 100.f; });
	
	// vector sum
	skepu::Map([](float lhs, float rhs){ return lhs + rhs;})(c, a, b);
	check([](size_t i){ return i + 1.f; });
	
	// vector scale (library function)
	skepu::Map<1>(skepu::util::mul<float>)(c, a, 2);
	check([](size_t i){ return i * 2.f; });
	
	using namespace skepu;
	
	// sum (library function + using namespace)
	float sum = Reduce(util::add<float>)(a);
	CHECK(sum == Approx(size * (size - 1) / 2.f));
}
