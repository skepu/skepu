#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>
#include <skepu-lib/complex.hpp>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>


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
	
	std::cout << "A: " << a << "\n";
	std::cout << "B: " << b << "\n";
	
	test1(c, a, b);
	test2(c, a, b);
	
	test3(c, a, 100);
	test4(c, a, 100);
	
	// vector sum
	skepu::Map([](float lhs, float rhs){ return lhs + rhs;})(c, a, b);
	
	// vector scale (library function)
	skepu::Map<1>(skepu::util::mul<float>)(c, a, 2);
	
  using namespace skepu;
  
	// dot product (library function + using namespace)
	float sum = Reduce(util::add<float>)(a);
	
	skepu::io::cout << "C: " << c << "\n";
//	std::cout << "sum: " << sum << "\n";
	
}