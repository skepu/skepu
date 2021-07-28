#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>
#include <skepu-lib/complex.hpp>

using Complex = skepu::complex::complex<float>;
using InnerType = Complex::value_type;


Complex my_test_uf(skepu::Index1D i, Complex elwise, skepu::Vec<Complex> proxy, Complex uniform)
{
	Complex result;
	return result;
}

auto my_test = skepu::Map(my_test_uf);
auto adder = skepu::Map(skepu::complex::add<Complex>);
auto sqnorms = skepu::Map(skepu::complex::sq_norm<Complex>);
auto conjreduce = skepu::MapReduce(skepu::complex::conj<Complex>, skepu::complex::add<Complex>);
auto divider = skepu::Map(skepu::complex::real_div<Complex>);

TEST_CASE("Complex fundamentals")
{
	const size_t size{1000};

	skepu::Vector<Complex> v1(size), v2(size), r(size);
	skepu::Vector<InnerType> w(size);
	
	for (size_t i = 0; i < size; ++i)
	{
		v1(i) = Complex{ (InnerType)i, (InnerType)i };
		v2(i) = Complex{ (InnerType)(i / 100.0), (InnerType)(i / 100.0) };
	}
	
	my_test(r, v1, v2, Complex());
	
	std::cout << "Map: v1 = " << v1 << "\n";
	std::cout << "Map: v2 = " << v2 << "\n";
	
	adder(r, v1, v2);
	std::cout << "Map: r = " << r << "\n";
	
	sqnorms(w, v1);
	std::cout << "Map: w = " << w << "\n";
	
	conjreduce.setStartValue(skepu::complex::FOne);
	Complex res = conjreduce(r);
	std::cout << "Res: " << res << "\n";
	
	divider(r, v1, w);
	std::cout << "Res: " << r << "\n";
	
}
