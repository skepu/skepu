// A standard library function in a user function: fine for the CPU backends, but OpenCL C does not have it, and skepu-tool says so.
#include <algorithm>
#include <skepu>

float call_std(float x)
{
	float values[2] = {x, 2.f * x};
	std::stable_sort(values, values + 2);
	return values[0];
}

int main()
{
	skepu::Vector<float> in(4, 1.f), out(4);
	skepu::Map<1>(call_std)(out, in);
	return 0;
}
