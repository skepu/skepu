#include <catch2/catch.hpp>

#include <skepu>


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

TEST_CASE("Map, indexed 1D")
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

TEST_CASE("Map, indexed 2D")
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

TEST_CASE("Map, indexed 3D")
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

TEST_CASE("Map, indexed 4D")
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
