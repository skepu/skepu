// The smallest SkePU program: a Map skeleton that adds one to every element of a vector.
// Run it with a backend as the argument: bin/hello CPU, bin/hello OpenMP, bin/hello OpenCL or bin/hello CUDA
#include <iostream>
#include <skepu>
#include <skepu-lib/io.hpp>

float addOne(float a)
{
	return a + 1;
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <backend>\n";
		return 1;
	}
	auto spec = skepu::BackendSpec{argv[1]};
	skepu::setGlobalBackendSpec(spec);

	auto map = skepu::Map(addOne);
	skepu::Vector<float> in(8, 1.0f, "in"), out(8, "out");
	map(out, in);
	skepu::io::cout << "out = " << out << "\n";
	return 0;
}
