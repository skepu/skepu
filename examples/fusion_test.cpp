#include <skepu>
#include <skepu-lib/io.hpp>

float mult(float a, float b)
{
	return a * b;
}

float add(float a, float b)
{
	return a + b;
}

void mapreduce_test_1(size_t size)
{
	SKEPU_TRACE_SCOPE("Test MapReduce 1");
	auto dotprod1 = skepu::Map(mult);
	auto dotprod2 = skepu::Reduce(add);

	skepu::Vector<float> a(size, "a"), b(size, "b"), temp(size, "temp");

	dotprod1(temp, a, b);

	float res2 = dotprod2(temp);
}

void mapreduce_test_2(size_t size)
{
	SKEPU_TRACE_SCOPE("Test MapReduce 2");
	auto m1 = skepu::Map(mult);
	auto mr2 = skepu::MapReduce(mult, add);

	skepu::Vector<float> a(size, "a", 1), b(size, "b", 2), temp(size, "temp", 3);

	m1(a, a, b);

	float res1 = mr2(a, b);
}

void map_test_1(size_t size)
{
	SKEPU_TRACE_SCOPE("Test Map 1, serial fusion x2");
	auto m1 = skepu::Map(mult);
	auto m2 = skepu::Map(mult);

	skepu::Vector<float> a(size, "a"), b(size, "b"), temp(size, "temp");

	m1(temp, a, b);
	m2(temp, temp, a);
}

void map_test_2(size_t size)
{
	SKEPU_TRACE_SCOPE("Test Map 2, serial fusion x3");
	auto m1 = skepu::Map(mult);
	auto m2 = skepu::Map(mult);
	auto m3 = skepu::Map(mult);

	skepu::Vector<float> a(size, "a"), b(size, "b"), temp(size, "temp");

	m1(temp, a, b);
	m2(temp, temp, a);
	m3(temp, temp, a);
}

void map_test_3(size_t size)
{
	SKEPU_TRACE_SCOPE("Test Map 3, parallel fusion x2");
	auto m1 = skepu::Map(mult);
	auto m2 = skepu::Map(mult);

	skepu::Vector<float> a(size, "a"), b(size, "b"), temp(size, "temp");

	m1(a, a, a);
	m2(b, b, b);
}

void map_test_4(size_t size)
{
	SKEPU_TRACE_SCOPE("Test Map 4, parallel fusion x3");
	auto m1 = skepu::Map(mult);
	auto m2 = skepu::Map(mult);
	auto m3 = skepu::Map(mult);

	skepu::Vector<float> a(size, "a"), b(size, "b"), c(size, "temp");

	m1(a, a, a);
	m2(b, b, b);
	m3(c, c, c);
}




float proxy(float a, float b, skepu::Vec<float> c)
{
	return a * b * c(0);
}

void map_test_5(size_t size)
{
	SKEPU_TRACE_SCOPE("Test Map 1, proxy, fusion x1");
	auto m1 = skepu::Map(mult);
	auto m2 = skepu::Map(proxy);

	skepu::Vector<float> a(size, "a"), b(size, "b"), temp(size, "temp");

	m1(temp, a, b);
	m2(temp, temp, b, a);
}


void map_test_no_1(size_t size)
{
	SKEPU_TRACE_SCOPE("Test Map 1, proxy, no fusion");
	auto m1 = skepu::Map(mult);
	auto m2 = skepu::Map(proxy);

	skepu::Vector<float> a(size, "a"), b(size, "b"), temp(size, "temp");

	m1(temp, a, b);
	m2(temp, a, b, temp);
}


int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}

	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
			skepu::Vector<float> a(size, "a"), b(size, "b"), temp(size, "temp");

	map_test_1(size);
	map_test_2(size);
	map_test_3(size);
	map_test_4(size);
	map_test_5(size);

	map_test_no_1(size);

	mapreduce_test_1(size);
	mapreduce_test_2(size);

	return 0;
}
