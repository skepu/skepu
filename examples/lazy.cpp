#include <iostream>
#include <skepu>

#define LAZY_EVAL

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
		v1(size, 1),
		v2(size, 2),
		v3(size, 3),
		v4(size, 4),
		v5(size, 5),
		v6(size, 6),
		v7(size, 7),
		v8(size, 8),
		v9(size, 9),
		v10(size, 10);
	
	DEBUG_REG(add); DEBUG_REG(sub); DEBUG_REG(mult); DEBUG_REG(square); DEBUG_REG(copy); DEBUG_REG(generate);
	DEBUG_REG(v1); DEBUG_REG(v2); DEBUG_REG(v3); DEBUG_REG(v4);
	DEBUG_REG(v5); DEBUG_REG(v6); DEBUG_REG(v7); DEBUG_REG(v8); DEBUG_REG(v9);
	
	add(v1, v3, v4);
	
	copy(v9, v1);
/*
	square(v5, v5);
	copy(v2, v5);
*/
	
	mult(v2, v1, v3);
	
	square(v1, v2);
	
	add(v5, v5, v1);
	
	add(v5, v5, v9);
	
	// Separate lineage component
	add(v6, v7, generate(v6, 5.f));
	
//	generate(v6, 5.f);
//	add(v6, v7, v6);
	
	// Separate lineage component
	for (int i = 0; i < 5; i++)
		add(v8, v8, v8);
	
#ifdef LAZY_EVAL
{
	using namespace skepu::LazyEvaluation;
	
	singleton.eliminateAllTransitiveDependencies();
//	listAllStartingPoints();
//	printLineage();
	
	
	auto time1 = skepu::benchmark::measureExecTime([&]()
	{
		singleton.evaluate(v8);
	});
	
	auto time2 = skepu::benchmark::measureExecTime([&]()
	{
		singleton.evaluateCacheAware(v8, 100000);
	});
	
	std::cout << "Non-Cache-aware: " << (time1.count() / 1E6) << " s.\n";
	std::cout << "    Cache-aware: " << (time2.count() / 1E6) << " s.\n";
	
	singleton.renderGraph("graph", true);
	
	singleton.evaluate();
	singleton.clearState();
}
#endif
	
//	std::cout << "v5 = " << v5 << "\n";
//	std::cout << "v6 = " << v6 << "\n";
//	std::cout << "v8 = " << v8 << "\n";
//	std::cout << "v10 = " << v10 << "\n";
	
	
//	for (int i = 0; i < 4; i++)
//		mult(v1, add(v2, v3, v1), add(v5, v6, v1));
	
#ifdef LAZY_EVAL
{
	using namespace skepu::LazyEvaluation;
	
	singleton.eliminateTransitiveDependencies();
	
	singleton.renderGraph("graph2", true);
	
	singleton.evaluate();
	singleton.clearState();
}
#endif
	
	return 0;
}

