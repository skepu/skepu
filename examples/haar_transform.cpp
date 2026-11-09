#include <algorithm>
#include <cmath>
#include <vector>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/util.hpp>

constexpr float inv_sqrt2 [[skepu::userconstant]] = 0.70710678118654752f;

// Any power of two: pooling 2^k elements gives the average and the difference of k levels of pairs at once
float haar_average(skepu::Pool1D<float> window)
{
	float sum = 0;
	for (int i = 0; i < window.si; ++i)
		sum += window(i);
	return sum / sqrt((float)window.si);
}

float haar_difference(skepu::Pool1D<float> window)
{
	float sum = 0;
	for (int i = 0; i < window.si / 2; ++i)
		sum += window(i) - window(i + window.si / 2);
	return sum / sqrt((float)window.si);
}

float haar_inverse(skepu::Index1D i, const skepu::Vec<float> averages, const skepu::Vec<float> differences)
{
	float average = averages[i.i / 2], difference = differences[i.i / 2];
	return (i.i % 2 == 0 ? average + difference : average - difference) * inv_sqrt2;
}

float make_signal(skepu::Index1D i, size_t n)
{
	float t = (float)i.i / n;
	return sin(18.84956f * t) + (t > 0.5f ? 1.f : 0.f) + (i.i % 4099 == 0 ? 2.f : 0.f);
}

float drop_small(float difference, float threshold)
{
	return fabs(difference) < threshold ? 0.f : difference;
}

float is_nonzero(float x) { return x != 0.f ? 1.f : 0.f; }
float square(float x) { return x * x; }
float squared_error(float a, float b) { return (a - b) * (a - b); }

int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size threshold pool backend\n";
		exit(1);
	}

	const size_t n = atoi(argv[1]);
	const float threshold = atof(argv[2]);
	const size_t pool = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);

	if (n < 2 || (n & (n - 1)) != 0)
	{
		skepu::io::cout << "The size must be a power of two\n";
		exit(1);
	}
	if (pool != 2 && pool != 4 && pool != 8 && pool != 16)
	{
		skepu::io::cout << "The pool size must be 2, 4, 8 or 16\n";
		exit(1);
	}

	auto average = skepu::MapPool(haar_average);
	auto difference = skepu::MapPool(haar_difference);
	auto inverse = skepu::Map<0>(haar_inverse);
	auto energy = skepu::MapReduce<1>(square, skepu::util::add<float>);
	auto error = skepu::MapReduce<2>(squared_error, skepu::util::add<float>);
	auto nonzero = skepu::MapReduce<1>(is_nonzero, skepu::util::add<float>);
	auto drop = skepu::Map<1>(drop_small);

	skepu::Vector<float> original(n);
	skepu::Map<0>(make_signal)(original, n);

	// Transform: the averages are the signal at half the resolution, the differences what that loses. A stage of pool size 2^k
	// does k levels from the same input: the difference of each level, and the average of the last.
	std::vector<skepu::Vector<float>> differences;
	skepu::Vector<float> averages = original;
	while (averages.size() > 1)
	{
		const size_t window = std::min(pool, averages.size());
		for (size_t span = 2; span <= window; span *= 2)
		{
			skepu::Vector<float> level(averages.size() / span);
			difference.setPoolSizeAndStride(span);
			difference(level, averages);
			differences.push_back(level);
		}
		skepu::Vector<float> next_averages(averages.size() / window);
		average.setPoolSizeAndStride(window);
		average(next_averages, averages);
		averages = next_averages;
	}

	auto transformed_energy = energy(averages);
	for (auto &level : differences)
		transformed_energy += energy(level);
	skepu::io::cout << "Energy of the signal: " << energy(original) << "\n";
	skepu::io::cout << "Energy of the transform: " << transformed_energy << "\n";

	// Inverse transform, from the coarsest level up. Differences below the threshold are dropped first
	size_t kept = 1;
	for (auto &level : differences)
	{
		drop(level, level, threshold);
		kept += (size_t)nonzero(level);
	}

	skepu::Vector<float> restored = averages;
	for (size_t level = differences.size(); level-- > 0; )
	{
		skepu::Vector<float> finer(2 * restored.size());
		inverse(finer, restored, differences[level]);
		restored = finer;
	}

	skepu::io::cout << "Coefficients kept: " << kept << " of " << n << "\n";
	skepu::io::cout << "Root mean square error: " << sqrt(error(original, restored) / n) << "\n";

	return 0;
}
