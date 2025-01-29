#include <iostream>
#include <skepu>

template<typename T>
T identity_f(T a)
{
	return a;
}

template<typename T>
T add_f(T a, T b)
{
	return a + b;
}

template<typename T>
T sub_f(T a, T b)
{
	return a - b;
}

template<typename T>
T mult_f(T a, T b)
{
	return a * b;
}

template<typename T>
T square_f(T a)
{
	return a * a;
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " size\n";
		exit(1);
	}

	const size_t size = atoi(argv[1]);

	auto add = skepu::Map<2>(add_f<float>);
	auto sub = skepu::Map<2>(sub_f<float>);
	auto mult = skepu::Map<2>(mult_f<float>);
	auto square = skepu::Map<1>(square_f<float>);
	auto copy = skepu::Map<1>([](float a) { return a; });
	auto generate = skepu::Map<0>([](skepu::Index1D index, float start) { return index.i + start; });

	skepu::Vector<float>
		v1(size, "v1", 1),
		v2(size, "v2", 2),
		v3(size, "v3", 3),
		v4(size, "v4", 4),
		v5(size, "v5", 5),
		v6(size, "v6", 6),
		v7(size, "v7", 7),
		v8(size, "v8", 8),
		v9(size, "v9", 9),
		v10(size, "v10", 10);

	add(v1, v3, v4);

	copy(v9, v1);

	mult(v2, v1, v3);

	square.setBackend(skepu::BackendSpec{"OpenMP"});
	square(v1, v2);

	add(v5, v5, v1);

	add(v5, v5, v9);

	add(v6, v7, generate(v6, 5.f));

	for (int i = 0; i < 5; i++)
		add(v8, v8, v8);

	skepu::external("External", skepu::read(v1, v2, v3, v4, v5), [&]{});

	return 0;
}
