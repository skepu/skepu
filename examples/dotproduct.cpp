#include <iostream>
#include <utility>
#include <cfloat>

#include <skepu>

template<typename T>
T mult(T a, T b)
{
	return a * b;
}

template<typename T>
T add(T a, T b)
{
	return a + b;
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
	
	auto dotprod = skepu::MapReduce<2>(mult<float>, add<float>);
	dotprod.setBackend(spec);
	
	skepu::Vector<float> a(size), b(size);
	a.randomize(0, 3);
	b.randomize(0, 2);
	
	std::cout << a << "\n";
	std::cout << b << "\n";
	
	float res = dotprod(a, b);
	
	std::cout << res << "\n";
	
	return 0;
}
