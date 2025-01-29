#include <iostream>
#include <skepu>
#include <skepu-lib/io.hpp>

int main(int argc, char *argv[])
{
	// Parse program arguments
	size_t argi = 1;
	const size_t degree = atoi(argv[argi++]);
	skepu::Vector<float> coeffs(degree + 1, "C");
	skepu::external("argv", [&]
	{
		for (int i = 0; i <= degree; ++i)
			coeffs(i) = atof(argv[argi++]);
	}, skepu::write(coeffs));
	const size_t samples = atof(argv[argi++]);
	auto spec = skepu::BackendSpec{argv[argi]};
	skepu::setGlobalBackendSpec(spec);

	// Create skeleton instance
	auto hornerFused = skepu::Map<0>(
	[](skepu::Index1D idx, skepu::Vec<float> c, float start, float delta, size_t degree) -> skepu::multiple<float, float>
	{
		float x = start + idx.i * delta, y = c(degree);
		for (int i = degree - 1; i >= 0; --i)
			y = y * x + c(i);
		return skepu::ret(x, y);
	});

	// Apply Horner's rule on given polynomial coefficients
	const float start = 0, stop = 100;
	float delta = (stop - start) / (samples - 1);
	skepu::Vector<float> x_vals(samples, "X"), y_vals(samples, "Y");
	hornerFused(x_vals, y_vals, coeffs, start, delta, degree);
	skepu::io::cout << "X: " << x_vals << "\n";
	skepu::io::cout << "Y: " << y_vals << "\n";
}
