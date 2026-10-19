// Each part is in its own namespace; its test cases are prefixed with its name.

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <test_helper_automatic_setup.hpp>


// ======== proxies ========

namespace proxies_part
{
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



// Tensors with different values in all elements: the element access of the proxy must use the strides
int reversed_planes(skepu::Index3D i, const skepu::Ten3<int> t) { return t(t.size_i - 1 - i.i, i.j, i.k); }
int reversed_blocks(skepu::Index4D i, const skepu::Ten4<int> t) { return t(t.size_i - 1 - i.i, i.j, i.k, i.l); }
auto reverse_planes = skepu::Map<0>(reversed_planes);
auto reverse_blocks = skepu::Map<0>(reversed_blocks);

TEST_CASE("proxies: Tensor3 and Tensor4 proxies read elements by all the indices")
{
	{
		const size_t si = 5, sj = 3, sk = 4;
		skepu::Tensor3<int> a(si, sj, sk), rev(si, sj, sk);
		skepu::external([&]
		{
			for (size_t i = 0; i < si; ++i)
				for (size_t j = 0; j < sj; ++j)
					for (size_t k = 0; k < sk; ++k)
						a(i, j, k) = (int)(i * 100 + j * 10 + k);
		}, skepu::write(a));
		reverse_planes(rev, a);
		rev.flush();
		for (size_t i = 0; i < si; ++i)
			for (size_t j = 0; j < sj; ++j)
				for (size_t k = 0; k < sk; ++k)
					CHECK(rev(i, j, k) == (int)((si - 1 - i) * 100 + j * 10 + k));
	}

	{
		const size_t si = 4, sj = 3, sk = 2, sl = 5;
		skepu::Tensor4<int> a(si, sj, sk, sl), rev(si, sj, sk, sl);
		skepu::external([&]
		{
			for (size_t i = 0; i < si; ++i)
				for (size_t j = 0; j < sj; ++j)
					for (size_t k = 0; k < sk; ++k)
						for (size_t l = 0; l < sl; ++l)
							a(i, j, k, l) = (int)(i * 1000 + j * 100 + k * 10 + l);
		}, skepu::write(a));
		reverse_blocks(rev, a);
		rev.flush();
		for (size_t i = 0; i < si; ++i)
			for (size_t j = 0; j < sj; ++j)
				for (size_t k = 0; k < sk; ++k)
					for (size_t l = 0; l < sl; ++l)
						CHECK(rev(i, j, k, l) == (int)((si - 1 - i) * 1000 + j * 100 + k * 10 + l));
	}
}


TEST_CASE("proxies: Container proxy reads in user function")
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
	
TEST_CASE("proxies: Container proxy sizes in user function")
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

TEST_CASE("proxies: MatRow and MatCol index mapping on matrix outputs")
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

TEST_CASE("proxies: MatRow and MatCol index mapping on vector outputs")
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
}


// ======== same_line ========

namespace same_line_part
{
// The precompiler replaces the declaration of a skeleton instance and must leave the rest of its line alone:
// other declarations and statements on the same line (and the text after them) are part of the program.

namespace
{
	float twice(float x) { return 2 * x; }
	float inc(float x) { return x + 1; }
}

auto first = skepu::Map<1>(twice); int after_first = 7; auto second = skepu::Map<1>(inc); int after_second = 8;

TEST_CASE("same_line: Code on the line of a named skeleton instance is kept")
{
	skepu::Vector<float> in(4, 1.f), a(4), b(4);
	auto local = skepu::Map<1>(inc); local(a, in); first(b, a); int trailing = 5; /* a comment */ int more = trailing + 1;
	second(a, in);
	a.flush(); b.flush();

	CHECK(after_first == 7);
	CHECK(after_second == 8);
	CHECK(trailing == 5);
	CHECK(more == 6);
	CHECK(a(0) == 2.f);
	CHECK(b(0) == 4.f);
}
}
