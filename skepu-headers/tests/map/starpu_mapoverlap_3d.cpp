#include <random>

#include <catch2/catch_test_macros.hpp>

#include <skepu3/cluster/external.hpp>
#include <skepu3/cluster/containers/tensor3/tensor3.hpp>
#include <skepu3/cluster/skeletons/map/mapoverlap/3d.hpp>

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
	CPU(skepu::Region3D<int> r)
	-> int
	{
		auto res = r(0,0,0);
		res += r(-1,0,0);
		res += r(1,0,0);
		res += r(0,-1,0);
		res += r(0,1,0);
		res += r(0,0,-1);
		res += r(0,0,1);
		return res / 7;
	}

	auto static
	OMP(skepu::Region3D<int> r)
	-> int
	{
		auto res = r(0,0,0);
		res += r(-1,0,0);
		res += r(1,0,0);
		res += r(0,-1,0);
		res += r(0,1,0);
		res += r(0,0,-1);
		res += r(0,0,1);
		return res / 7;
	}
};

TEST_CASE("Simple smoothing filter")
{
	int static constexpr overlap{1};
	skepu::backend::MapOverlap3D<smoothing_filter, bool, void> filter(false);
	filter.setOverlap(1);

	int const I(4 * skepu::cluster::mpi_size());
	int const J(4);
	int const K(4);
	skepu::Tensor3<int> t(I, J, K);
	skepu::Tensor3<int> res(I, J, K, -1);

	t.randomize(0,9);

	SECTION("using cyclic edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Cyclic);
		filter(res, t);

		t.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
				{
					int expected = t(i,j,k);
					expected += t((i -1 +I) %I, j,k);
					expected += t((i +1) %I, j,k);
					expected += t(i, (j -1 +J) %J, k);
					expected += t(i, (j +1) %J, k);
					expected += t(i, j, (k -1 + K) %K);
					expected += t(i, j, (k +1) %K);
					expected /= 7;
					REQUIRE(res(i,j,k) == expected);
				}
	}

	SECTION("using duplicate edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Duplicate);
		filter(res, t);

		t.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
				{
					int expected = t(i, j, k);
					if(i == 0)
						expected += t(0, j, k);
					else
						expected += t(i -1, j, k);
					if(i == I -1)
						expected += t(I -1, j, k);
					else
						expected += t(i +1, j, k);

					if(j == 0)
						expected += t(i, 0, k);
					else
						expected += t(i, j -1, k);
					if(j == J -1)
						expected += t(i, J -1, k);
					else
						expected += t(i, j +1, k);

					if(k == 0)
						expected += t(i, j, 0);
					else
						expected += t(i, j, k -1);
					if(k == K -1)
						expected += t(i, j, K -1);
					else
						expected += t(i, j, k +1);

					expected /= 7;
					REQUIRE(res(i,j,k) == expected);
				}
	}

	SECTION("using pad edge mode")
	{
		int const pad = 1;
		filter.setEdgeMode(skepu::Edge::Pad);
		filter.setPad(pad);
		filter(res, t);

		t.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
				{
					int expected = t(i,j,k);
					if(i == 0)
						expected += pad;
					else
						expected += t(i -1, j,k);
					if(i == I -1)
						expected += pad;
					else
						expected += t(i +1, j,k);

					if(j == 0)
						expected += pad;
					else
						expected += t(i, j -1, k);
					if(j == J -1)
						expected += pad;
					else
						expected += t(i, j +1, k);

					if(k == 0)
						expected += pad;
					else
						expected += t(i, j, k -1);
					if(k == K -1)
						expected += pad;
					else
						expected += t(i, j, k +1);

					expected /= 7;
					REQUIRE(res(i,j,k) == expected);
				}
	}

	SECTION("using no edge mode")
	{
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					res(i,j,k) = 0;

		filter.setEdgeMode(skepu::Edge::None);
		filter(res, t);

		t.flush();
		res.flush();
		for(int i(0); i < overlap; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					REQUIRE(res(i,j,k) == 0);

		for(int i(overlap); i < I - overlap; ++i)
		{
			for(int j(0); j < overlap; ++j)
				for(int k(0); k < K; ++k)
						REQUIRE(res(i,j,k) == 0);

			for(int j(overlap); j < J - overlap; ++j)
			{
				for(int k(0); k < overlap; ++k)
						REQUIRE(res(i,j,k) == 0);

				for(int k(overlap); k < K - overlap; ++k)
				{
					int expected = t(i, j, k);
					expected += t(i -1, j, k);
					expected += t(i +1, j, k);
					expected += t(i, j -1, k);
					expected += t(i, j +1, k);
					expected += t(i, j, k -1);
					expected += t(i, j, k +1);
					expected /= 7;
					REQUIRE(res(i,j,k) == expected);
				}

				for(int k(K - overlap); k < K; ++k)
					REQUIRE(res(i,j,k) == 0);
			}

			for(int j(J - overlap); j < J; ++j)
				for(int k(0); k < K; ++k)
					REQUIRE(res(i,j,k) == 0);
		}

		for(int i(I - overlap); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					REQUIRE(res(i,j,k) == 0);
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
	CPU(skepu::Region3D<int> r)
	-> int
	{
		auto res = r(0,0,0);
		for(int oi(-r.oi); oi < 0; ++oi)
			res += r(oi, 0, 0);
		for(int oi(1); oi <= r.oi; ++oi)
			res += r(oi, 0, 0);
		for(int oj(-r.oj); oj < 0; ++oj)
			res += r(0, oj, 0);
		for(int oj(1); oj <= r.oj; ++oj)
			res += r(0, oj, 0);
		for(int ok(-r.ok); ok < 0; ++ok)
			res += r(0, 0, ok);
		for(int ok(1); ok <= r.ok; ++ok)
			res += r(0, 0, ok);
		return res / (2*r.oi + 2*r.oj +2*r.ok +1);
	}

	auto static
	OMP(skepu::Region3D<int> r)
	-> int
	{
		auto res = r(0,0,0);
		for(int oi(-r.oi); oi < 0; ++oi)
			res += r(oi, 0, 0);
		for(int oi(1); oi <= r.oi; ++oi)
			res += r(oi, 0, 0);
		for(int oj(-r.oj); oj < 0; ++oj)
			res += r(0, oj, 0);
		for(int oj(1); oj <= r.oj; ++oj)
			res += r(0, oj, 0);
		for(int ok(-r.ok); ok < 0; ++ok)
			res += r(0, 0, ok);
		for(int ok(1); ok <= r.ok; ++ok)
			res += r(0, 0, ok);
		return res / (2*r.oi + 2*r.oj +2*r.ok +1);
	}
};

TEST_CASE("Smoothing filter with overlap on multiple ranks")
{
	int static constexpr overlap{3};
	skepu::backend::MapOverlap3D<smoothing_filter_ol_3, bool, void> filter(false);
	filter.setOverlap(overlap);

	int const I(std::max<int>(7, 2 * skepu::cluster::mpi_size() -1));
	int const J(8);
	int const K(8);
	skepu::Tensor3<int> t3(I, J, K);
	skepu::Tensor3<int> res(I, J, K, -1);

	t3.randomize(0,9);

	SECTION("using cyclic edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Cyclic);
		filter(res, t3);

		t3.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
				{
					int expected = t3(i,j,k);
					for(int oi(-overlap); oi < 0; ++oi)
						expected += t3((i + oi + I) %I, j, k);
					for(int oi(1); oi <= overlap; ++oi)
						expected += t3((i + oi + I) %I, j, k);
					for(int oj(-overlap); oj < 0; ++oj)
						expected += t3(i, (j + oj + J) %J, k);
					for(int oj(1); oj <= overlap; ++oj)
						expected += t3(i, (j + oj + J) %J, k);
					for(int ok(-overlap); ok < 0; ++ok)
						expected += t3(i, j, (k + ok + K) %K);
					for(int ok(1); ok <= overlap; ++ok)
						expected += t3(i, j, (k + ok + K) %K);
					expected /= 19;
					REQUIRE(res(i,j,k) == expected);
				}
	}

	SECTION("using duplicate edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Duplicate);
		filter(res, t3);

		t3.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
				{
					int expected = t3(i, j, k);

					for(int oi(-overlap); oi < 0; ++oi)
					{
						if(i + oi < 0)
							expected += t3(0, j, k);
						else
							expected += t3(i + oi, j, k);
					}
					for(int oi(1); oi <= overlap; ++oi)
					{
						if(i + oi < I)
							expected += t3(i + oi, j, k);
						else
							expected += t3(I -1, j, k);
					}

					for(int oj(-overlap); oj < 0; ++oj)
					{
						if(j + oj < 0)
							expected += t3(i, 0, k);
						else
							expected += t3(i, j + oj, k);
					}
					for(int oj(1); oj <= overlap; ++oj)
					{
						if(j + oj < J)
							expected += t3(i, j + oj, k);
						else
							expected += t3(i, J -1, k);
					}

					for(int ok(-overlap); ok < 0; ++ok)
					{
						if(k + ok < 0)
							expected += t3(i, j, 0);
						else
							expected += t3(i, j, k + ok);
					}
					for(int ok(1); ok <= overlap; ++ok)
					{
						if(k + ok < K)
							expected += t3(i, j, k + ok);
						else
							expected += t3(i, j, K -1);
					}

					expected /= 19;
					REQUIRE(res(i,j,k) == expected);
				}
	}

	SECTION("using pad edge mode")
	{
		int const pad = 1;
		filter.setEdgeMode(skepu::Edge::Pad);
		filter.setPad(pad);
		filter(res, t3);

		t3.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
				{
					int expected = t3(i, j, k);

					for(int oi(-overlap); oi < 0; ++oi)
					{
						if(i + oi < 0)
							expected += pad;
						else
							expected += t3(i + oi, j, k);
					}
					for(int oi(1); oi <= overlap; ++oi)
					{
						if(i + oi < I)
							expected += t3(i + oi, j, k);
						else
							expected += pad;
					}

					for(int oj(-overlap); oj < 0; ++oj)
					{
						if(j + oj < 0)
							expected += pad;
						else
							expected += t3(i, j + oj, k);
					}
					for(int oj(1); oj <= overlap; ++oj)
					{
						if(j + oj < J)
							expected += t3(i, j + oj, k);
						else
							expected += pad;
					}

					for(int ok(-overlap); ok < 0; ++ok)
					{
						if(k + ok < 0)
							expected += pad;
						else
							expected += t3(i, j, k +ok);
					}
					for(int ok(1); ok <= overlap; ++ok)
					{
						if(k + ok < K)
							expected += t3(i, j, k + ok);
						else
							expected += pad;
					}

					expected /= 19;
					REQUIRE(res(i,j,k) == expected);
				}
	}

	SECTION("using no edge mode")
	{
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					res(i,j,k) = 0;

		filter.setEdgeMode(skepu::Edge::None);
		filter(res, t3);

		t3.flush();
		res.flush();
		for(int i(0); i < overlap; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					REQUIRE(res(i,j,k) == 0);

		for(int i(overlap); i < I - overlap; ++i)
		{
			for(int j(0); j < overlap; ++j)
				for(int k(0); k < overlap; ++k)
					REQUIRE(res(i,j,k) == 0);

			for(int j(overlap); j < J - overlap; ++j)
			{
				for(int k(0); k < overlap; ++k)
					REQUIRE(res(i,j,k) == 0);

				for(int k(overlap); k < K -overlap; ++k)
				{
					int expected = t3(i,j,k);

					for(int oi(-overlap); oi < 0; ++oi)
						expected += t3(i +oi, j, k);
					for(int oi(1); oi <= overlap; ++oi)
						expected += t3(i +oi, j, k);

					for(int oj(-overlap); oj < 0; ++oj)
						expected += t3(i, j +oj, k);
					for(int oj(1); oj <= overlap; ++oj)
						expected += t3(i, j +oj, k);

					for(int ok(-overlap); ok < 0; ++ok)
						expected += t3(i, j, k +ok);
					for(int ok(1); ok <= overlap; ++ok)
						expected += t3(i, j, k +ok);

					expected /= 19;
					REQUIRE(res(i,j,k) == expected);
				}

				for(int k(K - overlap); k < K; ++k)
					REQUIRE(res(i,j,k) == 0);
			}

			for(int j(J - overlap); j < J; ++j)
				for(int k(K - overlap); k < K; ++k)
					REQUIRE(res(i,j,k) == 0);
		}

		for(int i(I - overlap); i < I; ++i)
			for(int j(J - overlap); j < J; ++j)
				for(int k(K - overlap); k < K; ++k)
					REQUIRE(res(i,j,k) == 0);
	}
}

struct indexed_uf
{
	constexpr static size_t totalArity = 4;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using IndexType = skepu::Index3D;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<size_t, size_t>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};

	using Ret = int;

	constexpr static bool prefersMatrix = 0;

	auto static
	CPU(
		skepu::Index3D idx,
		skepu::Region3D<int>,
		size_t const J,
		size_t const K)
	-> int
	{
		return idx.i * J*K + idx.j * K + idx.k;
	}

	auto static
	OMP(
		skepu::Index3D idx,
		skepu::Region3D<int>,
		size_t const J,
		size_t const K)
	-> int
	{
		return idx.i * J*K + idx.j * K + idx.k;
	}
};

TEST_CASE("Indexed 1d userfunction.")
{
	skepu::backend::MapOverlap3D<indexed_uf, bool, void>
		moi(false);
	moi.setOverlap(0);

	size_t const I{10 * skepu::cluster::mpi_size()};
	size_t const J{10};
	size_t const K{10};
	skepu::Tensor3<int> t3(I, J, K);
	skepu::Tensor3<int> res(I, J, K);

	moi(res, t3, J, K);

	t3.flush();
	res.flush();
	for(size_t i(0); i < t3.size_i(); ++i)
	{
		for(size_t j(0); j < t3.size_j(); ++j)
		{
			int const offset_j = i * J*K + j * K;
			for(size_t k(0); k < t3.size_k(); ++k)
				REQUIRE(res(i, j, k) == offset_j + (int)k);
		}
	}
}
