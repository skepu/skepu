#include <iostream>
#include <skepu>
#include <skepu-lib/io.hpp>


template<typename T>
T mvmult_f(const skepu::MatRow<T> mr, const skepu::Vec<T> v)
{
	T res = 0;
	for (size_t i = 0; i < v.size; ++i)
		res += mr(i) * v(i);
	return res;
}

float add(float a, float b) { return a + b; }

// A helper function to calculate dense matrix-vector product. Used to verify that the SkePU output is correct.
template<typename T>
void directMV(skepu::Vector<T> &v, skepu::Matrix<T> &m, skepu::Vector<T> &res)
{
	int rows = m.size_i();
	int cols = m.size_j();
	
	for (int r = 0; r < rows; ++r)
	{
		T sum = T();
		for (int i = 0; i < cols; ++i)
		{
			sum += m(r,i) * v(i);
		}
		res(r) = sum;
	}
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	skepu::Vector<float> v(size), r(size), r2(size);
	skepu::Matrix<float> m(size, size);
	m.randomize(3, 9);
	v.randomize(0, 9);
	
	skepu::io::cout << "v: " << v << "\n";
	skepu::io::cout << "m: " << m << "\n";
	
	directMV(v, m, r);
	auto mvprod = skepu::Map(mvmult_f<float>);
	mvprod(r2, m, v);
	
	auto mvprod_red = skepu::MapReduce<0>(mvmult_f<float>, add);
	mvprod_red.setDefaultSize(size);
	auto res = mvprod_red(m, v);
	skepu::io::cout << "res: " << res << "\n";
	
	skepu::external(skepu::read(r, r2), [&]
	{ 
		std::cout << "r: " << r << "\n";
		std::cout << "r2: " << r2 << "\n";
		
		for (size_t i = 0; i < size; i++)
			if (r(i) != r2(i))
				std::cout << "Output error at index " << i << ": " << r2(i) << " vs " << r(i) << "\n";
	});

	return 0;
}
