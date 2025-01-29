#include <skepu>
#include <skepu-lib/io.hpp>

float gen(skepu::Random<1> &rand)
{
	return rand.getNormalized();
}


int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	
	auto random = skepu::Map<0>(gen);
	random.setBackend(spec);
	
	skepu::Tensor4<float> a(size, size, size, size);
	
	random(a);
	
	skepu::io::cout << a << "\n";
	
	return 0;
}
