// The test of a package (make-package.sh): a Map that adds one, and a check of the result.
#include <iostream>
#include <skepu>

float addOne(float a)
{
	return a + 1;
}

int main(int argc, char *argv[])
{
	auto spec = skepu::BackendSpec{argc > 1 ? argv[1] : "CPU"};
	skepu::setGlobalBackendSpec(spec);
	auto map = skepu::Map(addOne);
	skepu::Vector<float> in(100, 1.0f), out(100);
	map(out, in);
	for (size_t i = 0; i < out.size(); ++i)
		if (out[i] != 2.0f)
		{
			std::cout << "wrong result\n";
			return 1;
		}
	std::cout << "ok\n";
	return 0;
}
