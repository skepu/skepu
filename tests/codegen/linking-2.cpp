#include <iostream>
#include <skepu>


template<typename T>
T arr2(skepu::Index1D row, const skepu::Mat<T> m, const skepu::Vec<T> v)
{
	T res = 0;
	for (size_t i = 0; i < v.size; ++i)
		res += m.data[row.i * m.cols + i] * v.data[i];
	return res;
}

// A helper function to calculate dense matrix-vector product. Used to verify that the SkePU output is correct.
template<typename T>
void directMV2(skepu::Vector<T> &v, skepu::Matrix<T> &m, skepu::Vector<T> &res)
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

auto mvprod2 = skepu::Map<0>(arr2<float>);

void mvmult2(skepu::Vector<float> &v, skepu::Matrix<float> &m, skepu::Vector<float> &res, skepu::BackendSpec *spec = nullptr)
{
	if (spec)
		mvprod2.setBackend(*spec);
	
	mvprod2(res, m, v);
}

