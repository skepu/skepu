#include <random>

#include "../../../external/catch2/catch.hpp"

#include <skepu3/cluster/external.hpp>
#include <skepu3/cluster/containers/tensor4/tensor4.hpp>
#include <skepu3/cluster/skeletons/map/mapoverlap/4d.hpp>

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
	CPU(skepu::Region4D<int> r)
	-> int
	{
		auto res = r(0,0,0,0);
		res += r(-1,0,0,0);
		res += r(1,0,0,0);
		res += r(0,-1,0,0);
		res += r(0,1,0,0);
		res += r(0,0,-1,0);
		res += r(0,0,1,0);
		res += r(0,0,0,-1);
		res += r(0,0,0,1);
		return res / 9;
	}

	auto static
	OMP(skepu::Region4D<int> r)
	-> int
	{
		auto res = r(0,0,0,0);
		res += r(-1,0,0,0);
		res += r(1,0,0,0);
		res += r(0,-1,0,0);
		res += r(0,1,0,0);
		res += r(0,0,-1,0);
		res += r(0,0,1,0);
		res += r(0,0,0,-1);
		res += r(0,0,0,1);
		return res / 9;
	}
};

TEST_CASE("Simple smoothing filter")
{
	int static constexpr overlap{1};
	skepu::backend::MapOverlap4D<smoothing_filter, bool, void> filter(false);
	filter.setOverlap(overlap);

	int const I(4 * overlap * skepu::cluster::mpi_size());
	int const J(4 * overlap);
	int const K(4 * overlap);
	int const L(4 * overlap);
	skepu::Tensor4<int> t(I, J, K, L);
	skepu::Tensor4<int> res(I, J, K, L, -1);

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
					for(int l(0); l < L; ++l)
					{
						int expected = t(i,j,k,l);
						expected += t((i -1 +I) %I, j,k,l);
						expected += t((i +1) %I, j,k,l);
						expected += t(i, (j -1 +J) %J, k, l);
						expected += t(i, (j +1) %J, k, l);
						expected += t(i, j, (k -1 + K) %K, l);
						expected += t(i, j, (k +1) %K, l);
						expected += t(i, j, k, (l -1 +L) %L);
						expected += t(i, j, k, (l +1) %L);
						expected /= 9;
						REQUIRE(res(i,j,k,l) == expected);
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
					for(int l(0); l < L; ++l)
					{
						int expected = t(i, j, k, l);
						if(i == 0)
							expected += t(0, j, k, l);
						else
							expected += t(i -1, j, k, l);
						if(i == I -1)
							expected += t(I -1, j, k, l);
						else
							expected += t(i +1, j, k, l);

						if(j == 0)
							expected += t(i, 0, k, l);
						else
							expected += t(i, j -1, k, l);
						if(j == J -1)
							expected += t(i, J -1, k, l);
						else
							expected += t(i, j +1, k, l);

						if(k == 0)
							expected += t(i, j, 0, l);
						else
							expected += t(i, j, k -1, l);
						if(k == K -1)
							expected += t(i, j, K -1, l);
						else
							expected += t(i, j, k +1, l);

						if(l == 0)
							expected += t(i, j, k, 0);
						else
							expected += t(i, j, k, l -1);
						if(l == L -1)
							expected += t(i, j, k, L -1);
						else
							expected += t(i, j, k, l +1);

						expected /= 9;
						if(res(i,j,k,l) != expected)
							printf("Rank %zu: error at res(%i,%i,%i,%i): "
								"expected: %i actual: %i\n",
								skepu::cluster::mpi_rank(), i, j, k, l, expected, res(i,j,k,l));
						REQUIRE(res(i,j,k,l) == expected);
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
					for(int l(0); l < L; ++l)
					{
						int expected = t(i,j,k,l);
						if(i == 0)
							expected += pad;
						else
							expected += t(i -1, j,k,l);
						if(i == I -1)
							expected += pad;
						else
							expected += t(i +1, j,k,l);

						if(j == 0)
							expected += pad;
						else
							expected += t(i, j -1, k, l);
						if(j == J -1)
							expected += pad;
						else
							expected += t(i, j +1, k, l);

						if(k == 0)
							expected += pad;
						else
							expected += t(i, j, k -1, l);
						if(k == K -1)
							expected += pad;
						else
							expected += t(i, j, k +1, l);

						if(l == 0)
							expected += pad;
						else
							expected += t(i, j, k, l -1);
						if(l == L -1)
							expected += pad;
						else
							expected += t(i, j, k, l +1);

						expected /= 9;
						REQUIRE(res(i,j,k,l) == expected);
					}
	}

	SECTION("using no edge mode")
	{
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					for(int l(0); l < L; ++l)
						res(i,j,k,l) = 0;

		filter.setEdgeMode(skepu::Edge::None);
		filter(res, t);

		t.flush();
		res.flush();
		for(int i(0); i < overlap; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);

		for(int i(overlap); i < I - overlap; ++i)
		{
			for(int j(0); j < overlap; ++j)
				for(int k(0); k < K; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);

			for(int j(overlap); j < J - overlap; ++j)
			{
				for(int k(0); k < overlap; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);

				for(int k(overlap); k < K - overlap; ++k)
				{
					for(int l(0); l < overlap; ++l)
						REQUIRE(res(i,j,k,l) == 0);

					for(int l(overlap); l < L - overlap; ++l)
					{
						int expected = t(i,j,k,l);
						expected += t(i -1, j,k,l);
						expected += t(i +1, j,k,l);
						expected += t(i, j -1, k, l);
						expected += t(i, j +1, k, l);
						expected += t(i, j, k -1, l);
						expected += t(i, j, k +1, l);
						expected += t(i, j, k, l -1);
						expected += t(i, j, k, l +1);
						expected /= 9;
						REQUIRE(res(i,j,k,l) == expected);
					}

					for(int l(L - overlap); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);
				}

				for(int k(K - overlap); k < K; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);
			}

			for(int j(J - overlap); j < J; ++j)
				for(int k(0); k < K; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);
		}

		for(int i(I - overlap); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);
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
	CPU(skepu::Region4D<int> r)
	-> int
	{
		auto res = r(0,0,0,0);
		for(int oi(-r.oi); oi < 0; ++oi)
			res += r(oi, 0, 0,  0);
		for(int oi(1); oi <= r.oi; ++oi)
			res += r(oi, 0, 0, 0);
		for(int oj(-r.oj); oj < 0; ++oj)
			res += r(0, oj, 0, 0);
		for(int oj(1); oj <= r.oj; ++oj)
			res += r(0, oj, 0, 0);
		for(int ok(-r.ok); ok < 0; ++ok)
			res += r(0, 0, ok, 0);
		for(int ok(1); ok <= r.ok; ++ok)
			res += r(0, 0, ok, 0);
		for(int ol(-r.ol); ol < 0; ++ol)
			res += r(0, 0, 0, ol);
		for(int ol(1); ol <= r.ol; ++ol)
			res += r(0, 0, 0, ol);
		return res / (2*r.oi + 2*r.oj +2*r.ok + 2*r.ol +1);
	}

	auto static
	OMP(skepu::Region4D<int> r)
	-> int
	{
		auto res = r(0,0,0,0);
		for(int oi(-r.oi); oi < 0; ++oi)
			res += r(oi, 0, 0,  0);
		for(int oi(1); oi <= r.oi; ++oi)
			res += r(oi, 0, 0, 0);
		for(int oj(-r.oj); oj < 0; ++oj)
			res += r(0, oj, 0, 0);
		for(int oj(1); oj <= r.oj; ++oj)
			res += r(0, oj, 0, 0);
		for(int ok(-r.ok); ok < 0; ++ok)
			res += r(0, 0, ok, 0);
		for(int ok(1); ok <= r.ok; ++ok)
			res += r(0, 0, ok, 0);
		for(int ol(-r.ol); ol < 0; ++ol)
			res += r(0, 0, 0, ol);
		for(int ol(1); ol <= r.ol; ++ol)
			res += r(0, 0, 0, ol);
		return res / (2*r.oi + 2*r.oj +2*r.ok + 2*r.ol +1);
	}
};

TEST_CASE("Smoothing filter with overlap on multiple ranks")
{
	int static constexpr overlap{3};
	skepu::backend::MapOverlap4D<smoothing_filter_ol_3, bool, void> filter(false);
	filter.setOverlap(overlap);

	int const I(std::max<int>(7, 2 * skepu::cluster::mpi_size() -1));
	int const J(4);
	int const K(4);
	int const L(4);
	skepu::Tensor4<int> t4(I, J, K, L);
	skepu::Tensor4<int> res(I, J, K, L, -1);

	t4.randomize(0,9);

	SECTION("using cyclic edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Cyclic);
		filter(res, t4);

		t4.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					for(int l(0); l < L; ++l)
					{
						int expected = t4(i,j,k,l);
						for(int oi(-overlap); oi < 0; ++oi)
							expected += t4((i + oi + I) %I, j, k, l);
						for(int oi(1); oi <= overlap; ++oi)
							expected += t4((i + oi + I) %I, j, k, l);
						for(int oj(-overlap); oj < 0; ++oj)
							expected += t4(i, (j + oj + J) %J, k, l);
						for(int oj(1); oj <= overlap; ++oj)
							expected += t4(i, (j + oj + J) %J, k, l);
						for(int ok(-overlap); ok < 0; ++ok)
							expected += t4(i, j, (k + ok + K) %K, l);
						for(int ok(1); ok <= overlap; ++ok)
							expected += t4(i, j, (k + ok + K) %K, l);
						for(int ol(-overlap); ol < 0; ++ol)
							expected += t4(i, j, k, (l +ol +L) %L);
						for(int ol(1); ol <= overlap; ++ol)
							expected += t4(i, j, k, (l +ol +L) %L);
						expected /= 25;
						REQUIRE(res(i,j,k,l) == expected);
					}
	}

	SECTION("using duplicate edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Duplicate);
		filter(res, t4);

		t4.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					for(int l(0); l < L; ++l)
					{
						int expected = t4(i, j, k, l);

						for(int oi(-overlap); oi < 0; ++oi)
						{
							if(i + oi < 0)
								expected += t4(0, j, k, l);
							else
								expected += t4(i + oi, j, k, l);
						}
						for(int oi(1); oi <= overlap; ++oi)
						{
							if(i + oi < I)
								expected += t4(i + oi, j, k, l);
							else
								expected += t4(I -1, j, k, l);
						}

						for(int oj(-overlap); oj < 0; ++oj)
						{
							if(j + oj < 0)
								expected += t4(i, 0, k, l);
							else
								expected += t4(i, j + oj, k, l);
						}
						for(int oj(1); oj <= overlap; ++oj)
						{
							if(j + oj < J)
								expected += t4(i, j + oj, k, l);
							else
								expected += t4(i, J -1, k, l);
						}

						for(int ok(-overlap); ok < 0; ++ok)
						{
							if(k + ok < 0)
								expected += t4(i, j, 0, l);
							else
								expected += t4(i, j, k + ok, l);
						}
						for(int ok(1); ok <= overlap; ++ok)
						{
							if(k + ok < K)
								expected += t4(i, j, k + ok, l);
							else
								expected += t4(i, j, K -1, l);
						}

						for(int ol(-overlap); ol < 0; ++ol)
						{
							if(l + ol < 0)
								expected += t4(i, j, k, 0);
							else
								expected += t4(i, j, k, l + ol);
						}
						for(int ol(1); ol <= overlap; ++ol)
						{
							if(l + ol < K)
								expected += t4(i, j, k, l + ol);
							else
								expected += t4(i, j, k, L -1);
						}

						expected /= 25;
						REQUIRE(res(i,j,k,l) == expected);
					}
	}

	SECTION("using pad edge mode")
	{
		int const pad = 1;
		filter.setEdgeMode(skepu::Edge::Pad);
		filter.setPad(pad);
		filter(res, t4);

		t4.flush();
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					for(int l(0); l < L; ++l)
					{
						int expected = t4(i, j, k, l);

						for(int oi(-overlap); oi < 0; ++oi)
						{
							if(i + oi < 0)
								expected += pad;
							else
								expected += t4(i + oi, j, k, l);
						}
						for(int oi(1); oi <= overlap; ++oi)
						{
							if(i + oi < I)
								expected += t4(i + oi, j, k, l);
							else
								expected += pad;
						}

						for(int oj(-overlap); oj < 0; ++oj)
						{
							if(j + oj < 0)
								expected += pad;
							else
								expected += t4(i, j + oj, k, l);
						}
						for(int oj(1); oj <= overlap; ++oj)
						{
							if(j + oj < J)
								expected += t4(i, j + oj, k, l);
							else
								expected += pad;
						}

						for(int ok(-overlap); ok < 0; ++ok)
						{
							if(k + ok < 0)
								expected += pad;
							else
								expected += t4(i, j, k +ok, l);
						}
						for(int ok(1); ok <= overlap; ++ok)
						{
							if(k + ok < K)
								expected += t4(i, j, k + ok, l);
							else
								expected += pad;
						}

						for(int ol(-overlap); ol < 0; ++ol)
						{
							if(l + ol < 0)
								expected += pad;
							else
								expected += t4(i, j, k, l +ol);
						}
						for(int ol(1); ol <= overlap; ++ol)
						{
							if(l + ol < K)
								expected += t4(i, j, k, l + ol);
							else
								expected += pad;
						}

						expected /= 25;
						REQUIRE(res(i,j,k,l) == expected);
					}
	}

	SECTION("using no edge mode")
	{
		res.flush();
		for(int i(0); i < I; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					for(int l(0); l < L; ++l)
						res(i,j,k,l) = 0;

		filter.setEdgeMode(skepu::Edge::None);
		filter(res, t4);

		t4.flush();
		res.flush();
		for(int i(0); i < overlap; ++i)
			for(int j(0); j < J; ++j)
				for(int k(0); k < K; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);

		for(int i(overlap); i < I - overlap; ++i)
		{
			for(int j(0); j < overlap; ++j)
				for(int k(0); k < overlap; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);

			for(int j(overlap); j < J - overlap; ++j)
			{
				for(int k(0); k < overlap; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);

				for(int k(overlap); k < K -overlap; ++k)
				{
					for(int l(0); l < overlap; ++l)
						REQUIRE(res(i,j,k,l) == 0);

					for(int l(overlap); l < L -overlap; ++l)
					{
						int expected = t4(i,j,k,l);

						for(int oi(-overlap); oi < 0; ++oi)
							expected += t4(i +oi, j, k, l);
						for(int oi(1); oi <= overlap; ++oi)
							expected += t4(i +oi, j, k, l);

						for(int oj(-overlap); oj < 0; ++oj)
							expected += t4(i, j +oj, k, l);
						for(int oj(1); oj <= overlap; ++oj)
							expected += t4(i, j +oj, k, l);

						for(int ok(-overlap); ok < 0; ++ok)
							expected += t4(i, j, k +ok, l);
						for(int ok(1); ok <= overlap; ++ok)
							expected += t4(i, j, k +ok, l);

						for(int ol(-overlap); ol < 0; ++ol)
							expected += t4(i, j, k, l +ol);
						for(int ol(1); ol <= overlap; ++ol)
							expected += t4(i, j, k, l +ol);

						expected /= 25;
						REQUIRE(res(i,j,k,l) == expected);
					}

					for(int l(L -overlap); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);
				}

				for(int k(K - overlap); k < K; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);
			}

			for(int j(J - overlap); j < J; ++j)
				for(int k(K - overlap); k < K; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);
		}

		for(int i(I - overlap); i < I; ++i)
			for(int j(J - overlap); j < J; ++j)
				for(int k(K - overlap); k < K; ++k)
					for(int l(0); l < L; ++l)
						REQUIRE(res(i,j,k,l) == 0);
	}
}

struct indexed_uf
{
	constexpr static size_t totalArity = 5;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using IndexType = skepu::Index4D;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<size_t, size_t, size_t>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};

	using Ret = int;

	constexpr static bool prefersMatrix = 0;

	auto static
	CPU(
		skepu::Index4D idx,
		skepu::Region4D<int>,
		size_t const J,
		size_t const K,
		size_t const L)
	-> int
	{
		return idx.i * J*K*L + idx.j * K*L + idx.k * L + idx.l;
	}

	auto static
	OMP(
		skepu::Index4D idx,
		skepu::Region4D<int>,
		size_t const J,
		size_t const K,
		size_t const L)
	-> int
	{
		return idx.i * J*K*L + idx.j * K*L + idx.k * L + idx.l;
	}
};

TEST_CASE("Indexed 1d userfunction.")
{
	skepu::backend::MapOverlap4D<indexed_uf, bool, void>
		moi(false);
	moi.setOverlap(0);

	size_t const I{10 * skepu::cluster::mpi_size()};
	size_t const J{10};
	size_t const K{10};
	size_t const L{10};
	skepu::Tensor4<int> t4(I, J, K, L);
	skepu::Tensor4<int> res(I, J, K, L);

	moi(res, t4, J, K, L);

	t4.flush();
	res.flush();
	for(size_t i(0); i < t4.size_i(); ++i)
		for(size_t j(0); j < t4.size_j(); ++j)
			for(size_t k(0); k < t4.size_k(); ++k)
			{
				int const offset_k = i * J*K*L + j * K*L + k * L;
				for(size_t l(0); l < t4.size_l(); ++l)
					REQUIRE(res(i, j, k, l) == offset_k + (int)l);
			}
}
