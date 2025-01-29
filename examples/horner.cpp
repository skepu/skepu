#include <iostream>

#include <skepu>
#include <skepu-lib/io.hpp>

skepu::Vector<float> generate(float start, float stop, size_t samples)
{
	auto gen_even = skepu::Map<0>([](skepu::Index1D idx, float start, float delta)
	{
		return start + idx.i * delta;
	});
	float delta = (stop - start) / (samples - 1);
	skepu::Vector<float> values(samples, "X");
	gen_even(values, start, delta);
	return values;
}

skepu::Vector<float> horner_eval(skepu::Vector<float> &coeffs, skepu::Vector<float> &x_vals)
{
	size_t degree = coeffs.size() - 1;
	auto muladd = skepu::Map<2>([](float a, float b, float c)
	{
		return a * b + c;
	});

	skepu::Vector<float> res(x_vals.size(), "Res", coeffs[degree]);
	for (int i = degree-1; i >= 0; --i)
	{
		SKEPU_TRACE_SCOPE("Iteration");
		muladd(res, res, x_vals, coeffs[i]);
	}
	return res;
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " degree coeffs samples backend\n";
		exit(1);
	}

	size_t argi = 1;
	const size_t degree = atoi(argv[argi++]);

	if (argc < degree + 4)
	{
		std::cout << "Usage: " << argv[0] << " degree coeffs samples backend\n";
		exit(1);
	}

	skepu::Vector<float> coeffs(degree + 1);

	// Read coefficients
	for (int i = degree; i >= 0; --i)
	{
		coeffs[i] = atof(argv[argi++]);
	}

	// Read x value
	size_t samples = atof(argv[argi++]);

	// Read backend
	auto spec = skepu::BackendSpec{argv[argi]};
	skepu::setGlobalBackendSpec(spec);

	skepu::Vector<float> x_vals = generate(0, 1, samples);
	skepu::io::cout << " X: " << x_vals << "\n";
	skepu::Vector<float> res = horner_eval(coeffs, x_vals);
	skepu::io::cout << "Values: " << res  << "\n";

	return 0;
}
