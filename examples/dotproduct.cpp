#include <iostream>
#include <utility>
#include <cfloat>

#include <skepu>

template<typename T>
T mult(T a, T b)
{
#if SKEPU_USING_BACKEND_CPU
	std::cout << "Mult " << a << " and " << b << "\n";
#endif
	return a * b;
}

template<typename T>
T add(T a, T b)
{
	return a + b;
}

auto dotprod = skepu::MapReduce<2>(mult<float>, add<float>);
float dotproduct(skepu::Vector<float> &a, skepu::Matrix<float> &b)
{
	
	return dotprod(a, b);
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[2])};
	dotprod.setBackend(spec);
	
	
	skepu::Vector<float> a(size), b(size);
	skepu::Matrix<float> c(sqrt(size), sqrt(size));
	a.randomize(0, 3);
	b.randomize(0, 2);
	c.randomize(0, 5);
	
	std::cout << a << "\n";
	std::cout << c << "\n";
	
	float res = dotproduct(a, c);
	
	std::cout << res << "\n";
	
	return 0;
}

