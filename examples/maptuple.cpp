#include <iostream>
#include <skepu>

skepu::multiple<int, float>
test1_f(skepu::Index1D index, int a, int b, skepu::Vec<float> c, int d)
{
	return skepu::ret(a * b, (float)a / b);
}

skepu::multiple<int, float>
test2_f(skepu::Index1D index, skepu::Vec<float> c)
{
	return skepu::ret(c[1], c[1] + 0.5);
}

int test_single_f(int a, int b, skepu::Vec<float> c, int d)
{
	return a * b * 10;
}


template<typename T>
T test_row_1_f(const skepu::MatRow<T> mr, const skepu::Vec<T> v)
{
	T res = 0;
	for (size_t i = 0; i < v.size; ++i)
		res += mr.data[i] * v.data[i];
	return res;
}

template<typename T>
T test_row_2_f(skepu::Index1D row, T a, T b, const skepu::MatRow<T> mr, const skepu::Mat<T> m, const skepu::Vec<T> v, T c)
{
	T res = a + b + c;
	for (size_t i = 0; i < v.size; ++i)
		res += mr.data[i] * v.data[i];
	return res;
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[2])};
	
	
	skepu::Vector<int> v1(size), v2(size), r1(size);
	skepu::Vector<float> r2(size);
	skepu::Vector<float> e = {2, 7};
	
	for (size_t i = 0; i < size; ++i)
	{
		v1(i) = i;
		v2(i) = 5;
	}
	
	std::cout << "v1: " << v1 << "\nv2: " << v2 << "\n\n";
	
	auto test1 = skepu::Map<2>(test1_f);
	test1.setBackend(spec);
	test1(r1, r2, v1, v2, e, 10);
	std::cout << "Test 1: r1 = " << r1 << "\nr2 = " << r2 << "\n\n";
	
	auto test2 = skepu::Map<0>(test2_f);
	test2.setBackend(spec);
	test2(r1, r2, e);
	std::cout << "Test 2: r1 = " << r1 << "\nr2 = " << r2 << "\n\n";
	
	
	
	auto test_single = skepu::Map<2>(test_single_f);
	test_single.setBackend(spec);
	test_single(r1, v1, v2, e, 10);
	std::cout << "Test Single: r1 = " << r1 << "\nr2 = " << r2 << "\n\n";
	
	
	
	skepu::Matrix<int> m(size, size);
	
	auto test_row_1 = skepu::Map<0>(test_row_1_f<int>);
	test_row_1.setBackend(spec);
	test_row_1(r1, m, v1);
	std::cout << "Test Row 1: r1 = " << r1 << "\n\n";
	
	
	auto test_row_2 = skepu::Map<2>(test_row_2_f<int>);
	test_row_2.setBackend(spec);
	test_row_2(r1, v1, v2, m, m, v1, 1);
	std::cout << "Test Row 2: r1 = " << r1 << "\n\n";
	
	
	return 0;
}

