#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>

template<typename T>
T test1_f(
	const skepu::Vec<T> v,
	const skepu::Mat<T> m,
	const skepu::MatRow<T> mr,
	const skepu::MatCol<T> mc,
	const skepu::Ten3<T> t3,
	const skepu::Ten4<T> t4)
{
	T a  = v(0);
	T b  = m(0,0);
	T c  = mr(0);
	T d  = mc(0);
	T e  = t3(0,0,0);
	T f  = t4(0,0,0,0);
	
	return a+b+c+d+e+f;
}

auto skel = skepu::Map(test1_f<float>);



// TEST PROXIES

int test1_proxy(const skepu::Vec<int> vec)
{
	return vec.size + vec(1);
}

int test2_proxy(const skepu::Mat<int> mat)
{
	return mat.rows * 100 + mat.cols + mat(1, 1);
}

int test2_proxy_row(const skepu::MatRow<int> row)
{
	return row.cols + row(1);
}

int test2_proxy_col(const skepu::MatCol<int> col)
{
	return col.rows + col(1);
}

int test3_proxy(const skepu::Ten3<int> ten3)
{
	return ten3.size_i * 10000 + ten3.size_j * 100 + ten3.size_k + ten3(1, 1, 1);
}

int test4_proxy(const skepu::Ten4<int> ten4)
{
	return ten4.size_i * 1000000 + ten4.size_j * 10000 + ten4.size_k * 100 + ten4.size_l + ten4(1, 1, 1, 1);
}

auto skel1_p = skepu::Map(test1_proxy);
auto skel2_p = skepu::Map(test2_proxy);
auto skel2_pr = skepu::Map(test2_proxy_row);
auto skel2_pc = skepu::Map(test2_proxy_col);
auto skel3_p = skepu::Map(test3_proxy);
auto skel4_p = skepu::Map(test4_proxy);



TEST_CASE("Container proxy reads in user function")
{
	size_t size{100};
	
	skepu::Vector<float> v(size), r(size);
	skepu::Matrix<float> m(size, size);
	skepu::Tensor3<float> t3(size, size, 1);
	skepu::Tensor4<float> t4(size, size, 1, 1);
	
	v(0) = 1;
	m(0,0) = 2;
	t3(0,0,0) = 3;
	t4(0,0,0,0) = 4;
	
	skel(r, v, m, m, m, t3, t4);
	r.flush();
	
	CHECK(r(0) == 1+2+2+2+3+4);
}
	
TEST_CASE("Container proxy sizes in user function")
{	
	constexpr int A = 903;
	skepu::Vector<int> dummy(1);

	const size_t size{100};
	
	skepu::Vector<int> vec(size, A);
	skepu::Matrix<int> mat(size, size+1, A);
	skepu::Tensor3<int> ten3(size, size+1, size+2, "", A);
	skepu::Tensor4<int> ten4(size, size+1, size+2, size+3, "", A);
	
	skel1_p(dummy, vec);
	dummy.flush();
	CHECK(dummy(0) == vec.size() + A);
	
	skel2_p(dummy, mat);
	dummy.flush();
	CHECK(dummy(0) == mat.total_rows() * 100 + mat.total_cols() + A);
	
	skel2_pr(dummy, mat); // matrow
	dummy.flush();
	CHECK(dummy(0) == mat.total_cols() + A);
	
	skel2_pc(dummy, mat); // matcol
	dummy.flush();
	CHECK(dummy(0) == mat.total_rows() + A);
	
	skel3_p(dummy, ten3);
	dummy.flush();
	CHECK(dummy(0) == ten3.size_i() * 10000 + ten3.size_j() * 100 + ten3.size_k() + A);
	
	skel4_p(dummy, ten4);
	dummy.flush();
	CHECK(dummy(0) == ten4.size_i() * 1000000 + ten4.size_j() * 10000 + ten4.size_k() * 100 + ten4.size_l() + A);
	
}







template<typename T>
T mmmult_f(const skepu::MatRow<T> ar, const skepu::MatCol<T> bc)
{
	T res = 0;
	for (size_t k = 0; k < ar.cols; ++k)
		res += ar(k) * bc(k);
	return res;
}
auto mmprod = skepu::Map(mmmult_f<float>);

TEST_CASE("MatRow and MatCol index mapping on matrix outputs")
{
	size_t height = 3;
	size_t width = 2;
	size_t inner = 1;
	
	skepu::Matrix<float> lhs(height, inner), rhs(inner, width), res(height, width), res2(height, width);
	
	skepu::external(skepu::read(lhs, rhs), [&]
	{
		for (size_t i = 0; i < height; ++i)
			lhs(i, 0) = i;
		
		for (size_t j = 0; j < width; ++j)
			rhs(0, j) = j;
	});
	
	mmprod(res, lhs, rhs);
	
	skepu::external(skepu::read(lhs, rhs), [&]
	{
		for (size_t i = 0; i < height; ++i)
			for (size_t j = 0; j < width; ++j)
				res(i, j) = i * j;
	});
	
}


template<typename T>
T col_sum_f(const skepu::MatCol<T> mc)
{
	T res = 0;
	for (size_t i = 0; i < mc.rows; ++i)
		res += mc(i);
	return res;
}

template<typename T>
T row_sum_f(const skepu::MatRow<T> mr)
{
	T res = 0;
	for (size_t i = 0; i < mr.cols; ++i)
		res += mr(i);
	return res;
}

auto col_sum = skepu::Map(col_sum_f<float>);
auto row_sum = skepu::Map(row_sum_f<float>);

TEST_CASE("MatRow and MatCol index mapping on vector outputs")
{
	const size_t rows = 7;
	const size_t cols = 5;
	
	skepu::Matrix<float> m(rows, cols);
	skepu::Vector<float> sums_per_col(cols), sums_per_row(rows);
	
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				m(i, j) = (i * 3 + j) % 8 + 1;
	}, skepu::write(m));
	
	col_sum(sums_per_col, m);
	row_sum(sums_per_row, m);
	
	skepu::external(skepu::read(m, sums_per_col, sums_per_row), [&]
	{
		for (size_t j = 0; j < cols; ++j)
		{
			float expected = 0;
			for (size_t i = 0; i < rows; ++i)
				expected += m(i, j);
			CHECK(sums_per_col(j) == expected);
		}
		for (size_t i = 0; i < rows; ++i)
		{
			float expected = 0;
			for (size_t j = 0; j < cols; ++j)
				expected += m(i, j);
			CHECK(sums_per_row(i) == expected);
		}
	});
}
