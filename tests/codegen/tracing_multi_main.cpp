// Tracing in a program of several source files (see tracing_multi in CMakeLists.txt): two precompiled
// files with a skeleton each, which both write to the one trace, and a plain one.
#include <iostream>
#include <skepu>

int tracing_multi_other(size_t size);

static int twice(int a)
{
	return 2 * a;
}

int main()
{
	skepu::BackendSpec spec{skepu::Backend::Type::CPU};
	skepu::setGlobalBackendSpec(spec);

	size_t size = 100;
	skepu::Vector<int> in(size, 1), out(size);
	auto map = skepu::Map<1>(twice);
	map(out, in);

	// the program fails when something is wrong: the result, and the exit
	if (out[0] != 2 || out[size - 1] != 2 || tracing_multi_other(size) != 3 * (int)size)
	{
		std::cerr << "wrong result\n";
		return 1;
	}
	return 0;
}
