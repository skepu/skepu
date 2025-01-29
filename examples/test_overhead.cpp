#include <skepu>
#include <skepu-lib/io.hpp>

float mult(float a, float b)
{
	return a * b;
}

float add(float a, float b)
{
	return a + b;
}


int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size iterations backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	const size_t iterations = atoi(argv[2]);
	auto spec = skepu::BackendSpec{argv[3]};
	skepu::setGlobalBackendSpec(spec);
	 
	auto multer = skepu::Map(mult);
	auto adder = skepu::Reduce(add);
	
	skepu::Vector<float> a(size), b(size), temp(size);
	a.randomize(0, 3);
	b.randomize(0, 2);
	
	skepu::io::cout << a << "\n";
	skepu::io::cout << b << "\n";
	
	const size_t samples = 5;
	for (size_t s = 0; s < samples; ++s)
	{
		auto time = skepu::benchmark::measureExecTime([&]
		{
			for (size_t i = 0; i < iterations; ++i)
			{
				multer(temp, a, b);
				adder(temp);
			}
		});
		
		std::cout << "Time: " << time.count() / 1E6 << "\n";
	}
	
	
	
	return 0;
}
