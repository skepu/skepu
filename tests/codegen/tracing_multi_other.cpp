// The second precompiled file of the tracing_multi test.
#include <skepu>

static int thrice(int a)
{
	return 3 * a;
}

int tracing_multi_other(size_t size)
{
	skepu::Vector<int> in(size, 1), out(size);
	auto map = skepu::Map<1>(thrice);
	map(out, in);

	int sum = 0;
	for (size_t i = 0; i < size; ++i)
		sum += out[i];
	return sum;
}
