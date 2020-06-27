#include <catch2/catch.hpp>

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

// TODO: Move global to within the test case when code gen bug is fixed
auto mvprod = skepu::Map<0>(arr<float>);
TEST_CASE("Matrix vector multiplication")
{
	size_t constexpr N{1000};

	skepu::Matrix<float> m(N,N);
	skepu::Vector<float> v(N), res(N), expected(N);

	m.randomize();
	v.randomize();

	directMV(v, m, expected);
	REQUIRE_NOTHROW(mvprod(res, m, v));

	skepu::external(skepu::read(res, expected), [&]{
		for(size_t i = 0; i < N; ++i)
			REQUIRE(res(i) == Approx(expected(i)).epsilon(1E-3));
	});
}
