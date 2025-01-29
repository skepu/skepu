#include <catch2/catch_test_macros.hpp>

#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/skeletons/map/mapoverlap/1d.hpp>

struct smoothing_filter
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};

	using Ret = int;

	constexpr static bool prefersMatrix = 0;

	auto static
	CPU(skepu::Region1D<int> r)
	-> int
	{
		int res = 0;
		for(int i(-r.oi); i < r.oi; ++i)
			res += r(i);
		return res / ((r.oi *2) +1);
	}

	auto static
	OMP(skepu::Region1D<int> r)
	-> int
	{
		int res = 0;
		for(int i(-r.oi); i <= r.oi; ++i)
			res += r(i);
		return res / ((r.oi *2) +1);
	}
};

TEST_CASE("Vector smoothing filter")
{
	skepu::backend::MapOverlap1D<
			smoothing_filter, bool, bool, bool, bool, void>
		filter(false, false, false, false);
	auto static constexpr OVERLAP = int(2);
	auto static constexpr DIVISOR = (OVERLAP *2) +1;
	filter.setOverlap(OVERLAP);

	SECTION("with cyclic edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Cyclic);
		int const N{10 * (int)skepu::cluster::mpi_size()};
		skepu::Vector<int> v(N);
		skepu::Vector<int> res(N);
		std::vector<int> expected(N);

		v.flush();
		for(int i(0); i < N; ++i)
			v(i) = i;
		for(int i(0); i < N; ++i)
		{
			for(int j = i -OVERLAP; j <= i +OVERLAP; ++j)
				if(j < 0)
					expected[i] += v(j + N);
				else if(j >= N)
					expected[i] += v(j - N);
				else
					expected[i] += v(j);
			expected[i] /= DIVISOR;
		}

		REQUIRE_NOTHROW(filter(res, v));
		res.flush();
		for(int i(0); i < N; ++i)
			REQUIRE(res(i) == expected[i]);
	}

	SECTION("with duplicating edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Duplicate);
		int const N{10 * (int)skepu::cluster::mpi_size()};
		skepu::Vector<int> v(N);
		skepu::Vector<int> res(N);

		v.flush();
		for(int i(0); i < N; ++i)
			v(i) = i +1;

		REQUIRE_NOTHROW(filter(res, v));
		res.flush();
		for(int i(0); i < N -OVERLAP; ++i)
			REQUIRE(res(i) == i +1);
		for(int i(N -OVERLAP); i < N; ++i)
			REQUIRE(res(i) == i);
	}

	SECTION("with padding edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Pad);
		filter.setPad(-1);
		int const N{10 * (int)skepu::cluster::mpi_size()};
		skepu::Vector<int> v(N);
		skepu::Vector<int> res(N);

		v.flush();
		for(int i(0); i < N; ++i)
			v(i) = i;

		REQUIRE_NOTHROW(filter(res, v));
		res.flush();
		for(int i(0); i < N -OVERLAP; ++i)
			REQUIRE(res(i) == i);
		REQUIRE(res(N -2) == ((4 * (N -2)) -3)/5);
		REQUIRE(res(N -1) == ((3 * (N -1)) -5)/5);
	}

	SECTION("using no edge mode")
	{
		int const N{10 * (int)skepu::cluster::mpi_size()};
		skepu::Vector<int> v(N);
		skepu::Vector<int> res(N, 0);

		v.randomize(0,9);
		filter.setEdgeMode(skepu::Edge::None);
		filter(res, v);

		v.flush();
		res.flush();
		for(int i(0); i < OVERLAP; ++i)
			REQUIRE(res(i) == 0);

		for(int i(OVERLAP); i < N - OVERLAP; ++i)
		{
			int expected{0};
			for(int oi(i -OVERLAP); oi <= i + OVERLAP; ++oi)
				expected += v(oi);
			expected /= OVERLAP *2 +1;
			REQUIRE(res(i) == expected);
		}

		for(int i(N - OVERLAP); i < N; ++i)
			REQUIRE(res(i) == 0);
	}
}

TEST_CASE("Matrix 1D row-wise smoothening filter")
{
	int constexpr overlap{2};
	int const I{3 * overlap * (int)skepu::cluster::mpi_size()};
	int const J{3 * overlap};

	skepu::backend::MapOverlap1D<smoothing_filter, bool, bool, bool, bool, void>
		filter(false, false, false, false);
	skepu::Matrix<int> m(I, J);
	skepu::Matrix<int> res(I, J, 0);

	m.randomize(0,9);
	filter.setOverlap(overlap);

	SECTION("with edge mode Cyclic")
	{
		filter.setEdgeMode(skepu::Edge::Cyclic);
		REQUIRE_NOTHROW(filter(res, m));

		m.flush();
		res.flush();
		for(int row(0); row < I; ++row)
		{
			for(int col(0); col < J; ++col)
			{
				int expected{0};
				for(int i(col - overlap); i <= col + overlap; ++i)
					expected += m(row, (i + J) % J);
				expected /= 2*overlap +1;
				REQUIRE(res(row, col) == expected);
			}
		}
	}

	SECTION("with edge mode Duplicate")
	{
		filter.setEdgeMode(skepu::Edge::Duplicate);
		REQUIRE_NOTHROW(filter(res, m));

		m.flush();
		res.flush();
		for(int row(0); row < I; ++row)
		{
			for(int col(0); col < J; ++col)
			{
				int expected{0};
				for(int i(col - overlap); i <= col + overlap; ++i)
					expected +=
						(i < 0
						? m(row, 0)
						: i < J
							? m(row, i)
							: m(row, J -1));
				expected /= 2*overlap +1;
				REQUIRE(res(row, col) == expected);
			}
		}
	}

	SECTION("with edge mode Pad")
	{
		filter.setEdgeMode(skepu::Edge::Pad);
		REQUIRE_NOTHROW(filter(res, m));

		m.flush();
		res.flush();
		for(int row(0); row < I; ++row)
		{
			for(int col(0); col < J; ++col)
			{
				int expected{0};
				for(int i(col - overlap); i <= col + overlap; ++i)
					expected +=
						(i < 0
						? 0
						: i < J
							? m(row, i)
							: 0);
				expected /= 2*overlap +1;
				REQUIRE(res(row, col) == expected);
			}
		}
	}

	SECTION("with edge mode None")
	{
		filter.setEdgeMode(skepu::Edge::None);
		REQUIRE_NOTHROW(filter(res, m));

		m.flush();
		res.flush();
		for(int row(0); row < I; ++row)
		{
			for(int col(0); col < overlap; ++col)
				REQUIRE(res(row, col) == 0);
			for(int col(overlap); col < J - overlap; ++col)
			{
				int expected{0};
				for(int i(col - overlap); i <= col + overlap; ++i)
					expected += m(row, i);
				expected /= 2*overlap +1;
				REQUIRE(res(row, col) == expected);
			}
			for(int col(J - overlap); col < J; ++col)
				REQUIRE(res(row, col) == 0);
		}
	}
}

TEST_CASE("Matrix 1D col-wise smoothening filter")
{
	int constexpr overlap{2};
	int const I{3 * overlap};
	int const J{3 * overlap * (int)skepu::cluster::mpi_size()};

	skepu::backend::MapOverlap1D<smoothing_filter, bool, bool, bool, bool, void>
		filter(false, false, false, false);
	skepu::Matrix<int> m(I, J);
	skepu::Matrix<int> res(I, J, 0);

	m.randomize(0,9);
	filter.setOverlap(overlap);
	filter.setOverlapMode(skepu::Overlap::ColWise);

	SECTION("with edge mode Cyclic")
	{
		filter.setEdgeMode(skepu::Edge::Cyclic);
		REQUIRE_NOTHROW(filter(res, m));

		m.flush();
		res.flush();
		for(int row(0); row < I; ++row)
		{
			for(int col(0); col < J; ++col)
			{
				int expected{0};
				for(int i(row - overlap); i <= row + overlap; ++i)
					expected += m((i + I) % I, col);
				expected /= 2*overlap +1;
				REQUIRE(res(row, col) == expected);
			}
		}
	}

	SECTION("with edge mode Duplicate")
	{
		filter.setEdgeMode(skepu::Edge::Duplicate);
		REQUIRE_NOTHROW(filter(res, m));

		m.flush();
		res.flush();
		for(int row(0); row < I; ++row)
		{
			for(int col(0); col < J; ++col)
			{
				int expected{0};
				for(int i(row - overlap); i <= row + overlap; ++i)
					expected +=
						(i < 0
						? m(0, col)
						: i < I
							? m(i, col)
							: m(I -1, col));
				expected /= 2*overlap +1;
				REQUIRE(res(row, col) == expected);
			}
		}
	}

	SECTION("with edge mode Pad")
	{
		filter.setEdgeMode(skepu::Edge::Pad);
		REQUIRE_NOTHROW(filter(res, m));

		m.flush();
		res.flush();
		for(int row(0); row < I; ++row)
		{
			for(int col(0); col < J; ++col)
			{
				int expected{0};
				for(int i(row - overlap); i <= row + overlap; ++i)
					expected +=
						(i < 0
						? 0
						: i < I
							? m(i, col)
							: 0);
				expected /= 2*overlap +1;
				REQUIRE(res(row, col) == expected);
			}
		}
	}

	SECTION("with edge mode None")
	{
		filter.setEdgeMode(skepu::Edge::None);
		REQUIRE_NOTHROW(filter(res, m));

		m.flush();
		res.flush();

		for(int row(0); row < overlap; ++row)
			for(int col(0); col < J; ++col)
				REQUIRE(res(row, col) == 0);
		for(int row(overlap); row < I - overlap; ++row)
			for(int col(0); col < J; ++col)
			{
				int expected{0};
				for(int i(row - overlap); i <= row + overlap; ++i)
					expected += m(i, col);
				expected /= 2*overlap +1;
				REQUIRE(res(row, col) == expected);
			}
		for(int row(I - overlap); row < I; ++row)
			for(int col(0); col < J; ++col)
			REQUIRE(res(row, col) == 0);
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
	CPU(skepu::Region1D<int> r)
	-> int
	{
		auto res = r(0);
		for(int oi(-r.oi); oi < 0; ++oi)
			res += r(oi);
		for(int oi(1); oi <= r.oi; ++oi)
			res += r(oi);
		return res / (2*r.oi +1);
	}

	auto static
	OMP(skepu::Region1D<int> r)
	-> int
	{
		auto res = r(0);
		for(int oi(-r.oi); oi < 0; ++oi)
			res += r(oi);
		for(int oi(1); oi <= r.oi; ++oi)
			res += r(oi);
		return res / (2*r.oi +1);
	}
};

TEST_CASE("Smoothing filter with overlap on multiple ranks")
{
	int static constexpr overlap{3};
	skepu::backend::MapOverlap1D<
			smoothing_filter_ol_3, bool, bool, bool, bool, void>
		filter(false, false, false, false);
	filter.setOverlap(overlap);

	int const I(std::max<int>(7, 2 * skepu::cluster::mpi_size() -1));
	skepu::Vector<int> v(I);
	skepu::Vector<int> res(I, -1);

	v.randomize(0,9);

	SECTION("using cyclic edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Cyclic);
		filter(res, v);

		v.flush();
		res.flush();
		for(int i(0); i < I; ++i)
		{
			int expected = v(i);
			for(int oi(-overlap); oi < 0; ++oi)
				expected += v((i + oi + I) %I);
			for(int oi(1); oi <= overlap; ++oi)
				expected += v((i + oi + I) %I);
			expected /= 7;
			REQUIRE(res(i) == expected);
		}
	}

	SECTION("using duplicate edge mode")
	{
		filter.setEdgeMode(skepu::Edge::Duplicate);
		filter(res, v);

		v.flush();
		res.flush();
		for(int i(0); i < I; ++i)
		{
			int expected = v(i);

			for(int oi(-overlap); oi < 0; ++oi)
			{
				if(i + oi < 0)
					expected += v(0);
				else
					expected += v(i + oi);
			}
			for(int oi(1); oi <= overlap; ++oi)
			{
				if(i + oi < I)
					expected += v(i + oi);
				else
					expected += v(I -1);
			}

			expected /= 7;
			REQUIRE(res(i) == expected);
		}
	}

	SECTION("using pad edge mode")
	{
		int const pad = 1;
		filter.setEdgeMode(skepu::Edge::Pad);
		filter.setPad(pad);
		filter(res, v);

		v.flush();
		res.flush();
		for(int i(0); i < I; ++i)
		{
			int expected = v(i);

			for(int oi(-overlap); oi < 0; ++oi)
			{
				if(i + oi < 0)
					expected += pad;
				else
					expected += v(i + oi);
			}
			for(int oi(1); oi <= overlap; ++oi)
			{
				if(i + oi < I)
					expected += v(i + oi);
				else
					expected += pad;
			}

			expected /= 7;
			REQUIRE(res(i) == expected);
		}
	}

	SECTION("using no edge mode")
	{
		res.flush();
		for(int i(0); i < I; ++i)
			res(i) = 0;

		filter.setEdgeMode(skepu::Edge::None);
		filter(res, v);

		v.flush();
		res.flush();
		for(int i(0); i < overlap; ++i)
			REQUIRE(res(i) == 0);

		for(int i(overlap); i < I - overlap; ++i)
		{
			int expected = v(i);

			for(int oi(-overlap); oi < 0; ++oi)
				expected += v(i +oi);
			for(int oi(1); oi <= overlap; ++oi)
				expected += v(i +oi);

			expected /= 7;
			REQUIRE(res(i) == expected);
		}

		for(int i(I - overlap); i < I; ++i)
			REQUIRE(res(i) == 0);
	}
}

struct indexed_uf
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using IndexType = skepu::Index1D;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};

	using Ret = int;

	constexpr static bool prefersMatrix = 0;

	auto static
	CPU(skepu::Index1D idx, skepu::Region1D<int>)
	-> int
	{
		return idx.i;
	}

	auto static
	OMP(skepu::Index1D idx, skepu::Region1D<int>)
	-> int
	{
		return idx.i;
	}
};

TEST_CASE("Indexed 1d userfunction.")
{
	skepu::backend::MapOverlap1D<indexed_uf, bool, bool, bool, bool, void>
		moi(false,false,false,false);
	moi.setOverlap(0);

	SECTION("With vector")
	{
		skepu::Vector<int> v(10*skepu::cluster::mpi_size());
		skepu::Vector<int> res(10*skepu::cluster::mpi_size());
		moi(res, v);

		v.flush();
		res.flush();
		for(size_t i(0); i < v.size(); ++i)
			REQUIRE(res(i) == i);
	}

	SECTION("With matrix row_wise")
	{
		size_t const N{10 * skepu::cluster::mpi_size()};
		skepu::Matrix<int> m(N, N);
		skepu::Matrix<int> res(N, N);

		moi.setOverlapMode(skepu::Overlap::RowWise);
		moi(res, m);

		m.flush();
		res.flush();
		for(size_t i(0); i < m.size_i(); ++i)
			for(size_t j(0); j < m.size_j(); ++j)
				REQUIRE(res(i, j) == j);
	}

	SECTION("With matrix col_wise")
	{
		size_t const N{10 * skepu::cluster::mpi_size()};
		skepu::Matrix<int> m(N, N);
		skepu::Matrix<int> res(N, N);

		moi.setOverlapMode(skepu::Overlap::ColWise);
		moi(res, m);

		m.flush();
		res.flush();
		for(size_t i(0); i < m.size_i(); ++i)
			for(size_t j(0); j < m.size_j(); ++j)
				REQUIRE(res(i, j) == i);
	}
}
