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
	
/*	DEBUG_REG(add); DEBUG_REG(sub); DEBUG_REG(mult); DEBUG_REG(square); DEBUG_REG(copy); DEBUG_REG(generate);
	DEBUG_REG_C(v1); DEBUG_REG_C(v2); DEBUG_REG_C(v3); DEBUG_REG_C(v4);
	DEBUG_REG_C(v5); DEBUG_REG_C(v6); DEBUG_REG_C(v7); DEBUG_REG_C(v8); DEBUG_REG_C(v9);
*/
	
	add.setLabel("add");
	sub.setLabel("sub");
	mult.setLabel("mult");
	square.setLabel("square");
	copy.setLabel("copy");
	generate.setLabel("generate");
/*	v1.setLabel("v1");
	v2.setLabel("v2");
	v3.setLabel("v3");
	v4.setLabel("v4");
	v5.setLabel("v5");
	v6.setLabel("v6");
	v7.setLabel("v7");
	v8.setLabel("v8");
	v9.setLabel("v9");*/
	
	add(v1, v3, v4);
	
	copy(v9, v1);
	
	mult(v2, v1, v3);
	
	square(v1, v2);
	
	add(v5, v5, v1);
	
	add(v5, v5, v9);
	
	// Separate lineage component
	add(v6, v7, generate(v6, 5.f));
	
	generate(v6, 5.f);
	add(v6, v7, v6);
	
	// Separate lineage component
	for (int i = 0; i < 5; i++)
		add(v8, v8, v8);
	
	return 0;
}

