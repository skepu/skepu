#include <iostream>
#include <skepu>


template<typename T>
T arr(skepu::Index1D row, const skepu::Mat<T> m, const skepu::Vec<T> v)
{
	T res = 0;
	for (size_t i = 0; i < v.size; ++i)
		res += m.data[row.i * m.cols + i] * v.data[i];
	return res;
}

auto mvprod = skepu::Map<0>(arr<float>);

int main(int argc, char *argv[])
{
	size_t size = 2048;
	skepu::BackendSpec spec{skepu::Backend::Type::CPU};
	if (argc >= 2)
	{
		size = atoi(argv[1]);
		spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[2])};
	}
	skepu::setGlobalBackendSpec(spec);
	
	skepu::Matrix<float> m(size, size);
	skepu::Vector<float> v(size), r(size);
	m.randomize(3, 9);
	v.randomize(0, 9);
	
	mvprod(r, m, v);

	return 0;
}
