#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/blas.hpp>

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	
	auto dotprod = skepu::MapReduce(mult<float>, add<float>);
	dotprod.setBackend(spec);

	
	skepu::Vector<float> a(size, "A"), b(size, "B");
	a.randomize(0, 3);
	b.randomize(0, 2);
	
	skepu::io::cout << a << "\n";
	
	skepu::blas::scal(size, 2, a, 1);
	
	skepu::io::cout << a << "\n";
	
	return 0;
}
