#include "skepu3/cluster/containers/partition.hpp"
#include "../../../external/catch2/catch.hpp"

#include <skepu3/cluster/skeletons/map/map.hpp>
#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/tensor3/tensor3.hpp>
#include <skepu3/cluster/containers/tensor4/tensor4.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>

struct simple_map_fn
{
	constexpr static size_t totalArity = 0;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP()
	-> int
	{
		return 10;
	}

	auto static inline
	CPU()
	-> int
	{
		return 10;
	}
};

TEST_CASE("Can create a simple map")
{
	REQUIRE_NOTHROW(skepu::backend::Map<0, simple_map_fn, bool, void>(false));
}

TEST_CASE("Simple map to vector")
{
	skepu::backend::Map<0, simple_map_fn, bool, void> map(false);

	SECTION("Small input vector")
	{
		skepu::Vector<int> v(10);
		REQUIRE_NOTHROW(map(v));
		v.flush();
		for(auto & e : v)
			REQUIRE(e == 10);
	}

/*
	SECTION("And with filters")
	{
		auto mfbs = skepu::max_filter_block_size;
		skepu::max_filter_block_size = 10 * sizeof(int);
		skepu::Vector<int> v(20 * skepu::cluster::mpi_size());
		REQUIRE_NOTHROW(map(v));
		v.flush();
		for(size_t i = 0; i < v.size(); ++i)
			REQUIRE(v(i) == 10);
		skepu::max_filter_block_size = mfbs;
	}
	*/
}

struct indexed_map_fn
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(skepu::Index1D index)
	-> int
	{
		return index.i;
	}

	auto static inline
	CPU(skepu::Index1D index)
	-> int
	{
		return index.i;
	}
};

TEST_CASE("Can create an indexed map")
{
	REQUIRE_NOTHROW(skepu::backend::Map<0, indexed_map_fn, bool, void>(false));
}

TEST_CASE("Can apply an indexed map to a vector.")
{
	skepu::Vector<int> v(10);
	skepu::backend::Map<0, indexed_map_fn, bool, void> map(false);
	REQUIRE_NOTHROW(map(v));
	v.flush();
	for(int i = 0; i < 10; ++i)
		REQUIRE(v(i) == i);
}

struct elwise_mult_fn
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<int,int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(int a, int b)
	-> int
	{
		return a * b;
	}

	auto static inline
	CPU(int a, int b)
	-> int
	{
		return a * b;
	}
};

TEST_CASE("Can create a Map with elwise arguments")
{
	REQUIRE_NOTHROW(skepu::backend::Map<2, elwise_mult_fn, bool, void>(false));
}

TEST_CASE("Can apply an elwise Map to a Vector")
{
	skepu::Vector<int> v1(10);
	skepu::Vector<int> v2(10);
	skepu::Vector<int> res(10);

	v1.flush();
	v2.flush();
	for(size_t i(0); i < v1.size(); ++i)
	{
		v1(i) = i;
		v2(i) = i;
	}

	skepu::backend::Map<2, elwise_mult_fn, bool, void> map(false);

	REQUIRE_NOTHROW(map(res,v1,v2));
	res.flush();
	v1.flush();
	v2.flush();
	for(size_t i(0); i < res.size(); ++i)
		REQUIRE(res(i) == (v1(i) * v2(i)));
}

struct indexed_elwise_mult_fn
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using ElwiseArgs = std::tuple<int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(skepu::Index1D idx, int val)
	-> int
	{
		return idx.i * val;
	}

	auto static inline
	CPU(skepu::Index1D idx, int val)
	-> int
	{
		return idx.i * val;
	}
};

TEST_CASE("Index and elwise Map.")
{
	REQUIRE_NOTHROW(
		skepu::backend::Map<1,indexed_elwise_mult_fn, bool, void>(false));
	auto map(skepu::backend::Map<1,indexed_elwise_mult_fn, bool, void>(false));
	skepu::Vector<int> v{2,2,2,2,2,2,2,2,2,2,2};
	skepu::Vector<int> res(v.size());
	map(res, v);
	res.flush();
	v.flush();
	for(size_t i(0); i < res.size(); ++i)
		REQUIRE(res(i) == (v(i) * i));
}

struct vector_copy_fn
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<skepu::Vec<int>>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<skepu::ProxyTag::Default> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {
		skepu::AccessMode::ReadWrite};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	static inline auto
	OMP(skepu::Vec<int> v)
	-> int
	{
		int sum(0);
		for(size_t i(0); i < v.size; ++i)
			sum += v(i);
		return sum;
	}

	static inline auto
	CPU(skepu::Vec<int> v)
	-> int
	{
		int sum(0);
		for(size_t i(0); i < v.size; ++i)
			sum += v(i);
		return sum;
	}
};

TEST_CASE("Map with container argument")
{
	REQUIRE_NOTHROW(skepu::backend::Map<0, vector_copy_fn, bool, void>(false));
	skepu::backend::Map<0, vector_copy_fn, bool, void> copy(false);

	SECTION("size 1")
	{
		size_t constexpr N{1};
		int constexpr expected{((N -1)*N)/2};
		skepu::Vector<int> v(N);
		skepu::Vector<int> res(v.size());

		v.flush();
		for(size_t i(0); i < v.size(); ++i)
			v(i) = i;
		skepu::cont::getParent(v).partition();

		REQUIRE_NOTHROW(copy(res, v));

		res.flush();
		for(size_t i(0); i < v.size(); ++i)
			REQUIRE(res(i) == expected);
	}

	SECTION("size 4")
	{
		size_t constexpr N{4};
		int constexpr expected{((N -1)*N)/2};
		skepu::Vector<int> v(N);
		skepu::Vector<int> res(v.size());

		v.flush();
		for(size_t i(0); i < v.size(); ++i)
			v(i) = i;
		skepu::cont::getParent(v).partition();

		REQUIRE_NOTHROW(copy(res, v));

		res.flush();
		for(size_t i(0); i < v.size(); ++i)
			REQUIRE(res(i) == expected);
	}

	SECTION("size 10000")
	{
		size_t constexpr N{10000};
		int constexpr expected{((N -1)*N)/2};
		skepu::Vector<int> v(N);
		skepu::Vector<int> res(v.size());

		v.flush();
		for(size_t i(0); i < v.size(); ++i)
			v(i) = i;
		skepu::cont::getParent(v).partition();

		REQUIRE_NOTHROW(copy(res, v));

		res.flush();
		for(size_t i(0); i < v.size(); ++i)
			REQUIRE(res(i) == expected);
	}
}

struct vector_scale_fn
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<int>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(int a, int b)
	-> int
	{
		return a * b;
	}

	auto static inline
	CPU(int a, int b)
	-> int
	{
		return a * b;
	}
};

TEST_CASE("Scaling vector using uniform argument.")
{
	REQUIRE_NOTHROW(skepu::backend::Map<1, vector_scale_fn, bool, void>(false));
	auto scale = skepu::backend::Map<1, vector_scale_fn, bool, void>(false);
	skepu::Vector<int> v(100000);
	skepu::Vector<int> res(v.size());
	int factor = 10;
	v.flush();
	for(size_t i(0); i < v.size(); ++i)
		v(i) = i;

	scale(res,v,factor);
	res.flush();
	v.flush();
	for(size_t i(0); i < v.size(); ++i)
		REQUIRE(res(i) == (v(i) * factor));
}

TEST_CASE("Scaling small tensor3 using uniform argument")
{
	size_t N(skepu::cluster::mpi_size() * 10);
	constexpr int factor(13);
	skepu::Tensor3<int> tensor(N,N,N);
	skepu::Tensor3<int> res(N,N,N);
	auto scale = skepu::backend::Map<1, vector_scale_fn, bool, void>(false);

	tensor.flush();
	for(size_t i(0); i < N; ++i)
		for(size_t j(0); j < N; ++j)
			for(size_t k(0); k < N; ++k)
			{
				tensor(i,j,k) = (i*N*N) + (j*N) + k;
			}

	scale(res, tensor, factor);

	res.flush();
	for(size_t i(0); i < N*N*N; ++i)
		REQUIRE(res(i) == tensor(i) * factor);
}

struct matrix_indexed_fn
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 1;

	auto static inline
	OMP(skepu::Index2D idx)
	-> int
	{
		return idx.row * idx.col;
	}

	auto static inline
	CPU(skepu::Index2D idx)
	-> int
	{
		return idx.row * idx.col;
	}
};

TEST_CASE("Mapping a matrix with index parameter.")
{
	constexpr size_t N = 100;
	skepu::backend::Map<0, matrix_indexed_fn, bool, void> map(false);
	skepu::Matrix<int> m(N,N);
	REQUIRE_NOTHROW(map(m));

	SECTION("Square matrix gets correct values.")
	{
		m.flush();
		map(m);
		m.flush();
		for(size_t i(0); i < m.size_i(); ++i)
			for(size_t j(0); i < m.size_j(); ++i)
				REQUIRE(m(i,j) == (i*j));
	}
}

struct matrix_scale_fn
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<int>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 1;

	auto static inline
	OMP(int elem, int factor)
	-> int
	{
		return elem * factor;
	}

	auto static inline
	CPU(int elem, int factor)
	-> int
	{
		return elem * factor;
	}
};

TEST_CASE("Scale matrix")
{
	auto constexpr N{100};
	auto constexpr factor{11};
	skepu::Matrix<int> m(N,N);
	skepu::Matrix<int> res(N,N);
	auto map =
		skepu::backend::Map<1, matrix_scale_fn, bool, void>(false);

	m.flush();
	auto m_it = m.begin();
	for(size_t i(0); i < m.size(); ++i, ++m_it)
		*m_it = (int)i;

	REQUIRE_NOTHROW(map(res,m,factor));
	res.flush();
	auto res_it = res.begin();
	for(size_t i(0); i < m.size(); ++i, ++res_it)
		REQUIRE(*res_it == (int)(i * factor));
}

struct matrix_vector_mult_fn
{
	constexpr static size_t totalArity = 3;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<skepu::Mat<int>, skepu::Vec<int>>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<
			skepu::ProxyTag::Default,
			skepu::ProxyTag::Default>
		ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {
		skepu::AccessMode::Read,
		skepu::AccessMode::Read};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(skepu::Index1D row, skepu::Mat<int> m, skepu::Vec<int> v)
	-> int
	{
		auto row_it = m.data + (row.i * m.cols);
		auto row_end_it = row_it + m.cols;
		auto v_it = v.data;
		int res(0);
		for(; row_it != row_end_it; ++row_it, ++v_it)
			res += *row_it * *v_it;
		return res;
	}

	auto static inline
	CPU(skepu::Index1D row, skepu::Mat<int> m, skepu::Vec<int> v)
	-> int
	{
		auto row_it = m.data + (row.i * m.cols);
		auto row_end_it = row_it + m.cols;
		auto v_it = v.data;
		int res(0);
		for(; row_it != row_end_it; ++row_it, ++v_it)
			res += *row_it * *v_it;
		return res;
	}
};

TEST_CASE("Matrix vector multiplication")
{
	size_t constexpr N{100};
	skepu::Vector<int> v(N);
	skepu::Matrix<int> m(N,N);
	skepu::Vector<int> res(N,N);
	auto map =
		skepu::backend::Map<0, matrix_vector_mult_fn, bool, void>(false);

	v.flush();
	m.flush();
	for(size_t i(0); i < N; ++i)
	{
		v(i) = i;
		for(size_t j(0); j < N; ++j)
			m(i,j) = i*j;
	}
	skepu::cont::getParent(v).partition();
	skepu::cont::getParent(m).partition();

	map(res, m, v);
	m.flush();
	v.flush();
	res.flush();

	for(size_t i(0); i < N; ++i)
	{
		int truth(0);
		for(size_t j(0); j < N; ++j)
			truth += v(j) * m(i,j);
		REQUIRE(res(i) == truth);
	}
}

struct matrow_vector_mult_fn
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<skepu::MatRow<int>, skepu::Vec<int>>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<
			skepu::ProxyTag::MatRow,
			skepu::ProxyTag::Default>
		ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {
		skepu::AccessMode::Read,
		skepu::AccessMode::Read};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(skepu::MatRow<int> m, skepu::Vec<int> v)
	-> int
	{
		auto res(0);
		for(size_t i(0); i < v.size; ++i)
			res += m(i) * v(i);
		return res;
	}

	auto static inline
	CPU(skepu::MatRow<int> m, skepu::Vec<int> v)
	-> int
	{
		auto res(0);
		for(size_t i(0); i < v.size; ++i)
			res += m(i) * v(i);
		return res;
	}
};

TEST_CASE("Matrix Vector mult using MatRow")
{
	constexpr int N{90};
	skepu::Matrix<int> m(N,N);
	skepu::Vector<int> v(N);
	skepu::Vector<int> res(N);
	std::vector<int> expected(N);
	auto mvmult =
		skepu::backend::Map<0, matrow_vector_mult_fn, bool, void>(false);

	m.flush();
	v.flush();
	for(int i(0); i < N; ++i)
	{
		v(i) = i;
		for(int j(0); j < N; ++j)
		{
			m(i,j) = i * j;
			expected[i] += i * j * j;
		}
	}
	mvmult(res, m, v);

	res.flush();
	for(int i(0); i < N; ++i)
		REQUIRE(res(i) == expected[i]);
}

struct tensor_elwise_index
{
	constexpr static size_t totalArity = 3;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<size_t, size_t>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(skepu::Index3D idx, size_t size_jk, size_t k)
	-> int
	{
		return (idx.i * size_jk) + (idx.j * k) + idx.k;
	}

	auto static inline
	CPU(skepu::Index3D idx, size_t size_jk, size_t k)
	-> int
	{
		return (idx.i * size_jk) + (idx.j * k) + idx.k;
	}
};

TEST_CASE("Intialize tensor with idx.")
{
	size_t constexpr N{10};
	skepu::Tensor3<int> tensor(N,N,N);
	skepu::backend::Map<0,tensor_elwise_index, bool, void> init_ten(false);
	init_ten(tensor, N*N, N);

	tensor.flush();
	for(size_t i(0); i < tensor.size(); ++i)
		REQUIRE((size_t)tensor(i) == i);
}

struct tensor_reduce_z
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<skepu::Ten3<int>>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<skepu::ProxyTag::Default> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {
		skepu::AccessMode::Read};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(skepu::Index1D idx, skepu::Ten3<int> & t)
	-> int
	{
		int res(0);
		auto start_it = t.data + (idx.i * t.size_jk);
		auto end_it = start_it + t.size_jk;

		for(; start_it != end_it; ++start_it)
			res += *start_it;

		return res;
	}

	auto static inline
	CPU(skepu::Index1D idx, skepu::Ten3<int> & t)
	-> int
	{
		int res(0);
		auto start_it = t.data + (idx.i * t.size_jk);
		assert(*start_it == 0);
		auto end_it = start_it + t.size_jk;

		for(; start_it != end_it; ++start_it)
			res += *start_it;

		return res;
	}
};

TEST_CASE("Tensor3 reduction to vector in i axis")
{
	size_t constexpr N(100);
	skepu::Tensor3<int> tensor(N,N,N);
	skepu::Vector<int> res(N);
	std::vector<int> expected;
	auto reduce = skepu::backend::Map<0, tensor_reduce_z, bool, void>(false);

	tensor.flush();
	for(size_t i(0); i < N; ++i)
	{
		int result(0);
		for(size_t j(0); j < N; ++j)
		{
			for(size_t k(0); k < N; ++k)
			{
				tensor(i,j,k) = i*j*k;
				result += i*j*k;
			}
		}
		expected.push_back(result);
	}

	REQUIRE_NOTHROW(reduce(res, tensor));
	res.flush();
	for(size_t i(0); i < N; ++i)
		REQUIRE(res(i) == expected[i]);
}

struct multiple_return_fn
{
	constexpr static size_t totalArity = 3;
	constexpr static size_t outArity = 2;
	constexpr static bool indexed = 1;
	using ElwiseArgs = std::tuple<int, double>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<skepu::ProxyTag::Default> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {
		skepu::AccessMode::Read};
	using Ret = std::tuple<int, double>;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	CPU(skepu::Index1D idx, int i, double d)
	-> std::tuple<int, double>
	{
		return std::make_tuple(idx.i * i, (double)(idx.i)*d);
	}

	auto static inline
	OMP(skepu::Index1D idx, int i, double d)
	-> std::tuple<int, double>
	{
		return std::make_tuple(idx.i * i, (double)(idx.i)*d);
	}
};

TEST_CASE("Multiple return")
{
	size_t constexpr N{100};
	REQUIRE_NOTHROW(
		skepu::backend::Map<2, multiple_return_fn, bool, void>(false));
	auto map = skepu::backend::Map<2, multiple_return_fn, bool, void>(false);
	skepu::Vector<int> vi(N, 1);
	skepu::Vector<double> vd(N, 0.5);
	skepu::Vector<int> res_i(N);
	skepu::Vector<double> res_d(N);

	REQUIRE_NOTHROW(map(res_i, res_d, vi, vd));
	res_i.flush();
	res_d.flush();
	for(size_t i(0); i < N; ++i)
	{
		REQUIRE(res_i(i) == (int)i);
		REQUIRE(res_d(i) == (double)i * 0.5);
	}
}

struct tensor_reduce_l
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<skepu::Ten4<int>>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<skepu::ProxyTag::Default> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {
		skepu::AccessMode::Read};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(skepu::Index1D idx, skepu::Ten4<int> & t)
	-> int
	{
		int res(0);
		auto start_it = t.data + (idx.i * t.size_jkl);
		auto end_it = start_it + t.size_jkl;

		for(; start_it != end_it; ++start_it)
			res += *start_it;

		return res;
		return 0;
	}

	auto static inline
	CPU(skepu::Index1D idx, skepu::Ten4<int> & t)
	-> int
	{
		int res(0);
		auto start_it = t.data + (idx.i * t.size_jkl);
		auto end_it = start_it + t.size_jkl;

		for(; start_it != end_it; ++start_it)
			res += *start_it;

		return res;
	}
};

TEST_CASE("Tensor4 reduction to vector in i axis")
{
	size_t constexpr N{10};
	skepu::Tensor4<int> tensor(N,N,N,N);
	skepu::Vector<int> res(N);
	std::vector<int> expected(N);
	auto reduce = skepu::backend::Map<0, tensor_reduce_l, bool, void>(false);

	tensor.flush();
	for(size_t i(0); i < N; ++i)
	{
		int result(0);
		for(size_t j(0); j < N; ++j)
			for(size_t k(0); k < N; ++k)
				for(size_t l(0); l < N; ++l)
				{
					tensor(i,j,k,l)= i*j*k*l;
					result += i*j*k*l;
				}
		expected[i] = result;
	}

	REQUIRE_NOTHROW(reduce(res, tensor));
	res.flush();
	for(size_t i(0); i < N; ++i)
		REQUIRE(res(i) == expected[i]);
}
