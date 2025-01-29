#include <random>

#include <catch2/catch_test_macros.hpp>

#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/skeletons/map/mapoverlap/2d.hpp>

struct smoothing_filter
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static
	CPU(skepu::Region2D<int> r)
	-> int
	{
		auto res = r(0,0);
		res += r(-1,0);
		res += r(1,0);
		res += r(0,-1);
		res += r(0,1);
		return res / 5;
	}

	auto static
	OMP(skepu::Region2D<int> r)
	-> int
	{
		auto res = r(0,0);
		res += r(-1,0);
		res += r(1,0);
		res += r(0,-1);
		res += r(0,1);
		return res / 5;
	}
};

TEST_CASE("Simple smoothing filter")
{
	int static constexpr overlap{1};
	skepu::backend::MapOverlap2D<smoothing_filter, bool, void> filter(false);
	filter.setOverlap(overlap);

	int const I(3 * skepu::cluster::mpi_size());
	int const J(3);
	skepu::Matrix<int> m(I, J);
	skepu::Matrix<int> res(I, J, -1);

	m.randomize(0,9);

	SECTION("using cyclic edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Cyclic);
		filter(res, m);

		m.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
			{
				int expected = m(i,j);
				expected += m((i -1 +I) %I, j);
				expected += m((i +1) %I, j);
				expected += m(i, (j -1 +J) %J);
				expected += m(i, (j +1) %J);
				expected /= 5;
				REQUIRE(res(i,j) == expected);
			}
	}

	SECTION("using duplicate edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Duplicate);
		filter(res, m);

		m.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
			{
				int expected = m(i, j);
				if(i == 0)
					expected += m(0, j);
				else
					expected += m(i -1, j);
				if(i == I -1)
					expected += m(I -1, j);
				else
					expected += m(i +1, j);

				if(j == 0)
					expected += m(i, 0);
				else
					expected += m(i, j -1);
				if(j == J -1)
					expected += m(i, J -1);
				else
					expected += m(i, j +1);

				expected /= 5;
				REQUIRE(res(i,j) == expected);
			}
	}

	SECTION("using pad edge mode")
	{
		int const pad = 1;
		filter.setEdgeMode(skepu::Edge::Pad);
		filter.setPad(pad);
		filter(res, m);

		m.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
			{
				int expected = m(i,j);
				if(i == 0)
					expected += pad;
				else
					expected += m(i -1, j);
				if(i == I -1)
					expected += pad;
				else
					expected += m(i +1, j);

				if(j == 0)
					expected += pad;
				else
					expected += m(i, j -1);
				if(j == J -1)
					expected += pad;
				else
					expected += m(i, j +1);

				expected /= 5;
				REQUIRE(res(i,j) == expected);
			}
	}

	SECTION("using no edge mode")
	{
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				res(i,j) = 0;

		filter.setEdgeMode(skepu::Edge::None);
		filter(res, m);

		m.flush();
		res.flush();
		for(int i(0); i < overlap; ++i)
			for(int j(0); j < J; ++j)
					REQUIRE(res(i,j) == 0);

		for(int i(overlap); i < I - overlap; ++i)
		{
			for(int j(0); j < overlap; ++j)
				REQUIRE(res(i,j) == 0);

			for(int j(overlap); j < J - overlap; ++j)
			{
				int expected = m(i, j);
				expected += m(i -1, j);
				expected += m(i +1, j);
				expected += m(i, j -1);
				expected += m(i, j +1);
				expected /= 5;
				REQUIRE(res(i,j) == expected);
			}

			for(int j(J - overlap); j < J; ++j)
				REQUIRE(res(i,j) == 0);
		}

		for(int i(I - overlap); i < I; ++i)
			for(int j(0); j < J; ++j)
				REQUIRE(res(i,j) == 0);
	}
}

struct smoothing_filter_ol_3
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static
	CPU(skepu::Region2D<int> r)
	-> int
	{
		auto res = r(0,0);
		for(int oi(-r.oi); oi < 0; ++oi)
			res += r(oi, 0);
		for(int oi(1); oi <= r.oi; ++oi)
			res += r(oi, 0);
		for(int oj(-r.oj); oj < 0; ++oj)
			res += r(0, oj);
		for(int oj(1); oj <= r.oj; ++oj)
			res += r(0, oj);
		return res / (2*r.oi + 2*r.oj +1);
	}

	auto static
	OMP(skepu::Region2D<int> r)
	-> int
	{
		auto res = r(0,0);
		for(int oi(-r.oi); oi < 0; ++oi)
			res += r(oi, 0);
		for(int oi(1); oi <= r.oi; ++oi)
			res += r(oi, 0);
		for(int oj(-r.oj); oj < 0; ++oj)
			res += r(0, oj);
		for(int oj(1); oj <= r.oj; ++oj)
			res += r(0, oj);
		return res / (2*r.oi + 2*r.oj +1);
	}
};

TEST_CASE("Smoothing filter with overlap on multiple ranks")
{
	int static constexpr overlap{3};
	skepu::backend::MapOverlap2D<smoothing_filter_ol_3, bool, void> filter(false);
	filter.setOverlap(overlap);

	int const I(std::max<int>(7,2 * skepu::cluster::mpi_size() -1));
	int const J(8);
	skepu::Matrix<int> m(I, J);
	skepu::Matrix<int> res(I, J, -1);

	m.randomize(0,9);

	SECTION("using cyclic edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Cyclic);
		filter(res, m);

		m.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
			{
				int expected = m(i,j);
				expected += m((i -3 +I) %I, j);
				expected += m((i -2 +I) %I, j);
				expected += m((i -1 +I) %I, j);
				expected += m((i +1) %I, j);
				expected += m((i +2) %I, j);
				expected += m((i +3) %I, j);
				expected += m(i, (j -3 +J) %J);
				expected += m(i, (j -2 +J) %J);
				expected += m(i, (j -1 +J) %J);
				expected += m(i, (j +1) %J);
				expected += m(i, (j +2) %J);
				expected += m(i, (j +3) %J);
				expected /= 13;
				REQUIRE(res(i,j) == expected);
			}
	}

	SECTION("using duplicate edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Duplicate);
		filter(res, m);

		m.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
			{
				int expected = m(i, j);

				for(int oi(-overlap); oi < 0; ++oi)
				{
					if(i + oi < 0)
						expected += m(0, j);
					else
						expected += m(i + oi, j);
				}
				for(int oi(1); oi <= overlap; ++oi)
				{
					if(i + oi < I)
						expected += m(i + oi, j);
					else
						expected += m(I -1, j);
				}

				for(int oj(-overlap); oj < 0; ++oj)
				{
					if(j + oj < 0)
						expected += m(i, 0);
					else
						expected += m(i, j + oj);
				}
				for(int oj(1); oj <= overlap; ++oj)
				{
					if(j + oj < J)
						expected += m(i, j + oj);
					else
						expected += m(i, J -1);
				}

				expected /= 13;
				REQUIRE(res(i,j) == expected);
			}
	}

	SECTION("using pad edge mode")
	{
		int const pad = 1;
		filter.setEdgeMode(skepu::Edge::Pad);
		filter.setPad(pad);
		filter(res, m);

		m.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
			{
				int expected = m(i, j);

				for(int oi(-overlap); oi < 0; ++oi)
				{
					if(i + oi < 0)
						expected += pad;
					else
						expected += m(i + oi, j);
				}
				for(int oi(1); oi <= overlap; ++oi)
				{
					if(i + oi < I)
						expected += m(i + oi, j);
					else
						expected += pad;
				}

				for(int oj(-overlap); oj < 0; ++oj)
				{
					if(j + oj < 0)
						expected += pad;
					else
						expected += m(i, j + oj);
				}
				for(int oj(1); oj <= overlap; ++oj)
				{
					if(j + oj < J)
						expected += m(i, j + oj);
					else
						expected += pad;
				}

				expected /= 13;
				REQUIRE(res(i,j) == expected);
			}
	}

	SECTION("using no edge mode")
	{
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				res(i,j) = 0;

		filter.setEdgeMode(skepu::Edge::None);
		filter(res, m);

		m.flush();
		res.flush();
		for(int i(0); i < overlap; ++i)
			for(int j(0); j < J; ++j)
					REQUIRE(res(i,j) == 0);

		for(int i(overlap); i < I - overlap; ++i)
		{
			for(int j(0); j < overlap; ++j)
				REQUIRE(res(i,j) == 0);

			for(int j(overlap); j < J - overlap; ++j)
			{
				int expected = m(i,j);
				expected += m((i -3 +I) %I, j);
				expected += m((i -2 +I) %I, j);
				expected += m((i -1 +I) %I, j);
				expected += m((i +1) %I, j);
				expected += m((i +2) %I, j);
				expected += m((i +3) %I, j);
				expected += m(i, (j -3 +J) %J);
				expected += m(i, (j -2 +J) %J);
				expected += m(i, (j -1 +J) %J);
				expected += m(i, (j +1) %J);
				expected += m(i, (j +2) %J);
				expected += m(i, (j +3) %J);
				expected /= 13;
				REQUIRE(res(i,j) == expected);
			}

			for(int j(J - overlap); j < J; ++j)
				REQUIRE(res(i,j) == 0);
		}

		for(int i(I - overlap); i < I; ++i)
			for(int j(0); j < J; ++j)
				REQUIRE(res(i,j) == 0);
	}
}

struct indexed_uf
{
	constexpr static size_t totalArity = 3;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using IndexType = skepu::Index2D;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<size_t>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};

	using Ret = int;

	constexpr static bool prefersMatrix = 0;

	auto static
	CPU(skepu::Index2D idx, skepu::Region2D<int>, size_t const N)
	-> int
	{
		return idx.row * N + idx.col;
	}

	auto static
	OMP(skepu::Index2D idx, skepu::Region2D<int>, size_t N)
	-> int
	{
		return idx.row * N + idx.col;
	}
};

TEST_CASE("Indexed 1d userfunction.")
{
	skepu::backend::MapOverlap2D<indexed_uf, bool, void>
		moi(false);
	moi.setOverlap(0);

	size_t const N{10 * skepu::cluster::mpi_size()};
	skepu::Matrix<int> m(N, N);
	skepu::Matrix<int> res(N, N);

	moi(res, m, N);

	m.flush();
	res.flush();
	for(size_t i(0); i < m.size_i(); ++i)
	{
		int const offset_i = i * N;
		for(size_t j(0); j < m.size_j(); ++j)
			REQUIRE(res(i, j) == offset_i + (int)j);
	}
}
