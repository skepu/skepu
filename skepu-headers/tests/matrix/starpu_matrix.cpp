#include <catch2/catch_test_macros.hpp>

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/containers/matrix/matrix.hpp>

TEST_CASE("Constructor does not throw")
{
	REQUIRE_NOTHROW(skepu::Matrix<int>());
	REQUIRE_NOTHROW(skepu::Matrix<int>{});

	SECTION("Size and capacity is zero with default initialization")
	{
		skepu::Matrix<int> m;
		skepu::Matrix<int> n{};

		CHECK(m.size() == 0);
		CHECK(n.size() == 0);
	}
}

TEST_CASE("Initialization with size (int)")
{
	using size_type = skepu::Matrix<int>::size_type;
	size_type constexpr N = 10;

	for(size_type i(0); i < N; ++i)
		for(size_type j(0); j < N; ++j)
			REQUIRE_NOTHROW(skepu::Matrix<int>(i, j));

	SECTION("Size, capacity, and values are correct")
	{
		skepu::Matrix<int> m1(1000,1,1);
		// skepu::Matrix<int> m2(1,1000,3);
		skepu::Matrix<int> m3(1000,1000,5);
		REQUIRE(m1.size() == 1000);
		// REQUIRE(m2.size() == 1000);
		REQUIRE(m3.size() == 1000000);

		m1.flush();
		// m2.flush();
		m3.flush();
		for(size_t i; i < 1000; ++i)
		{
			REQUIRE(m1(i,0) == 1);
			// REQUIRE(m2(0,i) == 3);
			for(size_t j; j < 1000; ++j)
				REQUIRE(m3(i,j) == 5);
		}
	}
}

TEST_CASE("Initialization with size (float)")
{
	using size_type = skepu::Matrix<float>::size_type;
	size_type constexpr N = 10;

	for(size_type i(0); i < N; ++i)
		for(size_type j(0); j < N; ++j)
			REQUIRE_NOTHROW(skepu::Matrix<float>(i, j));

	SECTION("Size, capacity, and values are correct")
	{
		skepu::Matrix<float> m1(1000,1,1.13f);
		skepu::Matrix<float> m2(1,1000,3.33f);
		skepu::Matrix<float> m3(1000,1000,5.7f);
		REQUIRE(m1.size() == 1000);
		REQUIRE(m2.size() == 1000);
		REQUIRE(m3.size() == 1000000);

		m1.flush();
		m2.flush();
		m3.flush();
		for(size_t i; i < 1000; ++i)
		{
			REQUIRE(m1(i,0) == 1.13f);
			REQUIRE(m2(0,i) == 3.33f);
			for(size_t j; j < 1000; ++j)
				REQUIRE(m3(i,j) == 5.7f);
		}
	}
}

TEST_CASE("Initialization with initializer_list")
{
	REQUIRE_NOTHROW(skepu::Matrix<int>{1,2,3,4});

	SECTION("of size 1")
	{
		skepu::Matrix<int> m{1,1,17};
		REQUIRE(m.size() == 1);
		REQUIRE(m.size_i() == 1);
		REQUIRE(m.size_j() == 1);
		m.flush();
		CHECK(m(0,0) == 17);
	}

	SECTION("of size 2")
	{
		skepu::Matrix<int> m{1,2,1,7};
		skepu::Matrix<int> n{2,1,3,5};
		REQUIRE(m.size() == 2);
		REQUIRE(m.size_i() == 1);
		REQUIRE(m.size_j() == 2);
		REQUIRE(n.size() == 2);
		REQUIRE(n.size_i() == 2);
		REQUIRE(n.size_j() == 1);
		m.flush();
		n.flush();
		CHECK(m(0,0) == 1);
		CHECK(m(0,1) == 7);
		CHECK(n(0,0) == 3);
		CHECK(n(1,0) == 5);
	}
}

TEST_CASE("Copy c-tor")
{
	skepu::Matrix<int> m{1,2,3,4};
	REQUIRE_NOTHROW(skepu::Matrix<int>{m});

	SECTION("Copy has correct properties and values")
	{
		skepu::Matrix<int> n(m);
		REQUIRE(n.size() == m.size());
		m.flush();
		CHECK(m(0,0) == 3);
		CHECK(m(0,1) == 4);
		n.flush();
		CHECK(n(0,0) == 3);
		CHECK(n(0,1) == 4);
	}
}

TEST_CASE("Copy assignment")
{
	skepu::Matrix<int> m{2,2,3,4,5,6};
	skepu::Matrix<int> n;
	REQUIRE_NOTHROW(n = m);
	REQUIRE(m.size() == 4);
	REQUIRE(n.size() == 4);

	m.flush();
	n.flush();

	CHECK(m(0,0) == 3);
	CHECK(n(0,0) == 3);
	CHECK(m(0,1) == 4);
	CHECK(n(0,1) == 4);
	CHECK(m(1,0) == 5);
	CHECK(n(1,0) == 5);
	CHECK(m(1,1) == 6);
	CHECK(n(1,1) == 6);
}

TEST_CASE("Move c-tor")
{
	skepu::Matrix<int> m{1,2,3,4};
	REQUIRE_NOTHROW(skepu::Matrix<int>(std::move(m)));

	m = skepu::Matrix<int>{1,3,5,7,11};
	skepu::Matrix<int> n(std::move(m));
	REQUIRE(n.size() == 3);
	n.flush();
	CHECK(n(0,0) == 5);
	CHECK(n(0,1) == 7);
	CHECK(n(0,2) == 11);
}

TEST_CASE("Move assignment")
{
	skepu::Matrix<int> m{1,3,3,4,5};
	skepu::Matrix<int> n;
	REQUIRE_NOTHROW(n = std::move(m));

	REQUIRE(n.size() == 3);
	n.flush();
	CHECK(n(0,0) == 3);
	CHECK(n(0,1) == 4);
	CHECK(n(0,2) == 5);
}

TEST_CASE("Swapping two Matrices")
{
	skepu::Matrix<int> m{1,2,3,4};
	skepu::Matrix<int> u{2,2,5,7,11,13};
	REQUIRE_NOTHROW(std::swap(m, u));
	REQUIRE(m.size() == 4);
	REQUIRE(u.size() == 2);
	m.flush();
	CHECK(m(0,0) == 5);
	CHECK(m(0,1) == 7);
	CHECK(m(1,0) == 11);
	CHECK(m(1,1) == 13);
	u.flush();
	CHECK(u(0,0) == 3);
	CHECK(u(0,1) == 4);
}

TEST_CASE("Can set the value of the element at a specific position")
{
	skepu::Matrix<int> m{2,2,3,4,5,6};
	REQUIRE_NOTHROW(m.set(1,0,42));
	m.flush();
	CHECK(m(0,0) == 3);
	CHECK(m(0,1) == 4);
	CHECK(m(1,0) == 42);
	CHECK(m(1,1) == 6);
}

TEST_CASE("Begin and end are equal on empty container.")
{
	skepu::Matrix<int> m;
	REQUIRE_NOTHROW(m.begin());
	REQUIRE_NOTHROW(m.end());
	m.flush();
	REQUIRE(m.begin() == m.end());

	SECTION("But not equal otherwise")
	{
		m.resize(1,1);
		m.flush();
		m(0,0) = 13;
		REQUIRE(m.begin() != m.end());
		CHECK(m.begin() +1 == m.end());
		CHECK(1 + m.begin() == m.end());
		CHECK(m.begin() == m.end() -1);
		CHECK(*(m.begin()) == 13);
	}
}

TEST_CASE("Matrix with 1000000 int elements.")
{
	size_t constexpr N{1000};
	int constexpr val{5};
	REQUIRE_NOTHROW(skepu::Matrix<float>(N,N));
	skepu::Matrix<float> m(N,N, val);
	m.flush();
	for(size_t row(0); row < N; ++row)
	{
		for(size_t col(0); col < N; ++col)
			REQUIRE(m(row,col) == val);
	}
}

TEST_CASE("Can use float in matrices.")
{
	size_t constexpr N{1000};
	float constexpr val{2.11f};
	REQUIRE_NOTHROW(skepu::Matrix<float>(N,N));
	skepu::Matrix<float> m(N,N, val);
	m.flush();
	for(size_t row(0); row < N; ++row)
	{
		for(size_t col(0); col < N; ++col)
			REQUIRE(m(row,col) == val);
	}
}
TEST_CASE("Matrix transpose.")
{
	size_t const N{10 * skepu::cluster::mpi_size()};
	skepu::Matrix<int> m(N,N);

	m.flush();
	for(size_t i(0); i < N; ++i)
	{
		int row_start = i * N;
		for(size_t j(0); j < N; ++j)
			m(i,j) = row_start + j;
	}
	skepu::Matrix<int> mt(m);

	REQUIRE_NOTHROW(mt.transpose());

	for(size_t i(0); i < N; ++i)
		for(size_t j(0); j < N; ++j)
			REQUIRE(mt(i,j) == m(j,i));
}

TEST_CASE("Flushing container created from pointer updates the original array")
{
	size_t const R = 10 * skepu::cluster::mpi_size();
	size_t const C = 10;
	auto v = new int[R*C];
	skepu::Matrix<int> sv(v, R, C);

	auto & part = skepu::cont::getParent(sv);
	part.partition();
	part.invalidate_local_storage();

	auto rank = skepu::cluster::mpi_rank();

	int i = 0;
	while((size_t)i < R*C)
	{
		auto task_size = part.block_count_from(i);
		auto handle = part.handle_for(i);
		auto owner = (size_t)starpu_mpi_data_get_rank(handle);

		if(owner == rank)
		{
			starpu_data_acquire(handle, STARPU_RW);
			auto ptr = (int *)starpu_data_get_local_ptr(handle);
			for(int ti = 0; (size_t)ti < task_size; ++ti)
				ptr[ti] = ti +i;
			starpu_data_release(handle);
		}

		i += task_size;
	}

	sv.flush();
	for(i = 0; (size_t)i < R*C; ++i)
		REQUIRE(v[i] == i);
}
