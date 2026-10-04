#include <skepu>

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
