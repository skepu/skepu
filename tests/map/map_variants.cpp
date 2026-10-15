// Each part is in its own namespace; its test cases are prefixed with its name.

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <iostream>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>


// ======== map_in_place ========

namespace map_in_place_part
{
int update_map(int value, int increment)
{
	return value + increment;
}

auto instance = skepu::Map<1>(update_map);

TEST_CASE("map_in_place: Map in place")
{
	const size_t size{100};

	skepu::Vector<int> values(size);
	values.randomize(0, 9);
	
	
	instance(values, values, 1);

}
}


// ======== map_index ========

namespace map_index_part
{
int uf_1d_a(skepu::Index1D index)
{
	return index.i;
}

int uf_1d_b(skepu::Index1D index, int val)
{
	return index.i;
}

auto skel_1d_a = skepu::Map(uf_1d_a);
auto skel_1d_b = skepu::Map(uf_1d_b);
auto skel_1d_c = skepu::Map<0>(uf_1d_b);

TEST_CASE("map_index: Map, indexed 1D")
{
	constexpr size_t size{1000};
	
	skepu::Vector<int> v(size), ra(size), rb(size);
	
	skel_1d_a(ra);
  skepu::external(skepu::read(ra), [&]
	{
		for (size_t i = 0; i < size; ++i)
			CHECK(ra(i) == i);
	});
	
	skel_1d_b(rb, v);
	skepu::external(skepu::read(rb), [&]
	{
		for (size_t i = 0; i < size; ++i)
			CHECK(rb(i) == i);
	});
	
	skel_1d_c(rb, 590); // just check that it runs correctly
}



int uf_2d_a(skepu::Index2D index)
{
	return index.row + index.col;
}

int uf_2d_b(skepu::Index2D index, int val)
{
	return index.row + index.col;
}

auto skel_2d_a = skepu::Map(uf_2d_a);
auto skel_2d_b = skepu::Map(uf_2d_b);
auto skel_2d_c = skepu::Map<0>(uf_2d_b);

TEST_CASE("map_index: Map, indexed 2D")
{
	constexpr size_t size_i{99}, size_j{78};
	
	skepu::Matrix<int> m(size_i, size_j), ra(size_i, size_j), rb(size_i, size_j);
	
	skel_2d_a(ra);
  skepu::external(skepu::read(ra), [&]
	{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
				CHECK(ra(i, j) == (i + j));
	});
	
	skel_2d_b(rb, m);
	skepu::external(skepu::read(rb), [&]
	{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
				CHECK(rb(i, j) == (i + j));
	});
	
	skel_2d_c(rb, 590); // just check that it runs correctly
}



int uf_3d_a(skepu::Index3D index)
{
	return index.i + index.j + index.k;
}

int uf_3d_b(skepu::Index3D index, int val)
{
	return index.i + index.j + index.k;
}

auto skel_3d_a = skepu::Map(uf_3d_a);
auto skel_3d_b = skepu::Map(uf_3d_b);
auto skel_3d_c = skepu::Map<0>(uf_3d_b);

TEST_CASE("map_index: Map, indexed 3D")
{
	constexpr size_t size_i{13}, size_j{7}, size_k{23};
	
	skepu::Tensor3<int> t3(size_i, size_j, size_k), ra(size_i, size_j, size_k), rb(size_i, size_j, size_k);
	
	skel_3d_a(ra);
  skepu::external(skepu::read(ra), [&]
	{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
				for (size_t k = 0; k < size_k; ++k)
					CHECK(ra(i, j, k) == (i + j + k));
	});
	
	skel_3d_b(rb, t3);
	skepu::external(skepu::read(rb), [&]
	{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
				for (size_t k = 0; k < size_k; ++k)
					CHECK(rb(i, j, k) == (i + j + k));
	});
	
	skel_3d_c(rb, 590); // just check that it runs correctly
}



int uf_4d_a(skepu::Index4D index)
{
	return index.i + index.j + index.k + index.l;
}

int uf_4d_b(skepu::Index4D index, int val)
{
	return index.i + index.j + index.k + index.l;
}

auto skel_4d_a = skepu::Map(uf_4d_a);
auto skel_4d_b = skepu::Map(uf_4d_b);
auto skel_4d_c = skepu::Map<0>(uf_4d_b);

TEST_CASE("map_index: Map, indexed 4D")
{
	constexpr size_t size_i{3}, size_j{7}, size_k{5}, size_l{13};
	
	skepu::Tensor4<int> t4(size_i, size_j, size_k, size_l), ra(size_i, size_j, size_k, size_l), rb(size_i, size_j, size_k, size_l);
	
	skel_4d_a(ra);
  skepu::external(skepu::read(ra), [&]
	{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
				for (size_t k = 0; k < size_k; ++k)
					for (size_t l = 0; l < size_l; ++l)
						CHECK(ra(i, j, k, l) == (i + j + k + l));
	});
	
	skel_4d_b(rb, t4);
	skepu::external(skepu::read(rb), [&]
	{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
				for (size_t k = 0; k < size_k; ++k)
					for (size_t l = 0; l < size_l; ++l)
						CHECK(rb(i, j, k, l) == (i + j + k + l));
	});
	
	skel_4d_c(rb, 590); // just check that it runs correctly
}
}


// ======== map_tuple ========

namespace map_tuple_part
{
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



auto test1 = skepu::Map(test1_f);
auto test2 = skepu::Map(test2_f);
auto test_row_1 = skepu::Map(test_row_1_f<int>);
auto test_single = skepu::Map(test_single_f);
auto test_row_2 = skepu::Map(test_row_2_f<int>);

TEST_CASE("map_tuple: Map with variadic return")
{
	constexpr size_t size{100};
	
	skepu::Vector<int> v1(size), v2(size), r1(size);
	skepu::Vector<float> r2(size);
	skepu::Vector<float> e = {2, 7};

	v1.flush();
	v2.flush();
	for (size_t i = 0; i < size; ++i)
	{
		v1(i) = i;
		v2(i) = 5;
	}

	skepu::external(
		skepu::read(v1, v2),
		[&]{
			std::cout << "v1: " << v1 << "\nv2: " << v2 << "\n\n";
		});

	test1(r1, r2, v1, v2, e, 10);

	skepu::external(
		skepu::read(r1,r2),
		[&]{
			std::cout << "Test 1: r1 = " << r1 << "\nr2 = " << r2 << "\n\n";
		});

	test2(r1, r2, e);

	skepu::external(
		skepu::read(r1,r2),
		[&]{
			std::cout << "Test 2: r1 = " << r1 << "\nr2 = " << r2 << "\n\n";
		});

	test_single(r1, v1, v2, e, 10);
	skepu::external(
		skepu::read(r1,r2),
		[&]{
			std::cout << "Test Single: r1 = " << r1 << "\nr2 = " << r2 << "\n\n";
		});

	skepu::Matrix<int> m(size, size);

	test_row_1(r1, m, v1);
	skepu::external(
		skepu::read(r1),
		[&]{
			std::cout << "Test Row 1: r1 = " << r1 << "\n\n";
		});

	test_row_2(r1, v1, v2, m, m, v1, 1);

	skepu::external(
		skepu::read(r1),
		[&]{
			std::cout << "Test Row 2: r1 = " << r1 << "\n\n";
		});
}




skepu::multiple<int, int> uf_mr_multi(int a, int b)
{
	return skepu::ret(a + b, a * b);
}

int sum(int a, int b)
{
	return a + b;
}

/*
void test_mapreduce_multi(size_t size)
{
	skepu::Vector<int> v1(size), v2(size);

	skepu::external(
		[&]{
			for (size_t i = 0; i < size; ++i)
			{
				v1(i) = i;
				v2(i) = 5;
			}},
		skepu::write(v1,v2));

	auto skel = skepu::MapReduce(uf_mr_multi, sum);

	int sum1, sum2;
	std::tie(sum1, sum2) = skel(v1, v2);
//	auto [sum1, sum2] = skel(v1, v2);
	skepu::external(
		[&]{
			std::cout << "sum1: " << sum1 << ", sum2: " << sum2 << std::endl;
		});
}*/
}


// ======== mvmult ========

namespace mvmult_part
{
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
TEST_CASE("mvmult: Matrix vector multiplication")
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
}
