#include <iostream>
#include <skepu>


template<typename T>
T arr(const skepu::MatRow<T> mr, const skepu::Vec<T> v)
{
	T res = 0;
	for (size_t i = 0; i < v.size; ++i)
		res += mr.data[i] * v.data[i];
	return res;
}

// A helper function to calculate dense matrix-vector product. Used to verify that the SkePU output is correct.
template<typename T>
void directMV(skepu::Vector<T> &v, skepu::Matrix<T> &m, skepu::Vector<T> &res)
{
	int rows = m.total_rows();
	int cols = m.total_cols();
	
	for (int r = 0; r < rows; ++r)
	{
		T sum = T();
		for (int i = 0; i < cols; ++i)
		{
			sum += m[r*cols+i] * v[i];
		}
		res[r] = sum;
	}
}

auto mvprod = skepu::MapTuple<0>(arr<float>);

void mvmult(skepu::Vector<float> &v, skepu::Matrix<float> &m, skepu::Vector<float> &res, skepu::BackendSpec *spec = nullptr)
{
	if (spec)
		mvprod.setBackend(*spec);
	
	mvprod(res, m, v);
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[2])};
	
	skepu::Vector<float> v(size), r(size), r2(size);
	skepu::Matrix<float> m(size, size);
	m.randomize(3, 9);
	v.randomize(0, 9);
	
	std::cout << "v: " << v << "\n";
	std::cout << "m: " << m << "\n";
	
	directMV(v, m, r);
	mvmult(v, m, r2, &spec);
	
	std::cout << "r: " << r << "\n";
	std::cout << "r2: " << r2 << "\n";
	
	for (size_t i = 0; i < size; i++)
		if (r[i] != r2[i])
			std::cout << "Output error at index " << i << ": " << r2[i] << " vs " << r[i] << "\n";
	
	return 0;
}
