#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/util.hpp>

float f(float x)
{
	// Some arbitrary continuous function
	return -5.56 * pow(x, 4.f) + 1.34 * pow(x, 3.f) + 3.45 * x * x + 5 * x + 40;
}

float riemann(skepu::Index1D index, float start, float end, size_t samples, float offset)
{
	float dx = (end - start) / samples;
	float x = start + (index.i + offset) * dx;
	return f(x) * dx;
}

int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		skepu::io::cout << "Usage: " << argv[0] << " start end samples backend\n";
		exit(1);
	}
	
	const float start = atof(argv[1]);
	const float end = atof(argv[2]);
	const size_t samples = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	skepu::io::cout << "Integral from " << start << " to " << end << " in " << samples << " samples...\n";
	
	auto integral = skepu::MapReduce<0>(riemann, skepu::util::add<float>);
	integral.setDefaultSize(samples);
	
	float sum = integral(start, end, samples, 0.5);
	
	skepu::io::cout << "Integral from " << start << " to " << end << " = " << sum << "\n";
	
	return 0;
}

