#include <iostream>
#include <skepu>
#include <skepu-lib/complex.hpp>

namespace cplx = skepu::experimental::complex;
using Complex = cplx::FComplex;
using InnerType = cplx::value_type<Complex>::type;

// Main
int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		if(!skepu::cluster::mpi_rank())
			std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}

	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);

	skepu::Vector<Complex> v1(size), v2(size), r(size);
	skepu::Vector<InnerType> w(size);
	
	for (size_t i = 0; i < size; ++i)
	{
		v1(i) = Complex{ (InnerType)i, (InnerType)i };
		v2(i) = Complex{ (InnerType)(i / 100.0), (InnerType)(i / 100.0) };
	}
	
	std::cout << "Map: v1 = " << v1 << "\n";
	std::cout << "Map: v2 = " << v2 << "\n";
	
	auto adder = skepu::Map<2>(cplx::add<Complex>);
	adder(r, v1, v2);
	std::cout << "Map: r = " << r << "\n";
	
	auto sqnorms = skepu::Map<1>(cplx::sq_norm<Complex>);
	sqnorms(w, v1);
	std::cout << "Map: w = " << w << "\n";
	
	auto conjreduce = skepu::MapReduce(cplx::conj<Complex>, cplx::add<Complex>);
	conjreduce.setStartValue(skepu::experimental::complex::FOne);
	Complex res = conjreduce(r);
	std::cout << "Res: " << res << "\n";

	return 0;
}
