#include <catch2/catch_test_macros.hpp>

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>

auto const mpi_size = skepu::cluster::mpi_size();

auto inline
expected_capacity(size_t count)
-> size_t
{
	auto part_size = count / mpi_size;
	if(count % mpi_size)
		++part_size;
	return part_size * mpi_size;
}

TEST_CASE("Constructor does not throw")
{
	REQUIRE_NOTHROW(skepu::Vector<int>());
	REQUIRE_NOTHROW(skepu::Vector<int>{});

	SECTION("Size and capacity is zero with default initialization")
	{
		skepu::Vector<int> v;

		CHECK(v.size() == 0);
		CHECK(v.capacity() == 0);
	}
}

TEST_CASE("Initialization with size")
{
	size_t constexpr N = 98765;

	REQUIRE_NOTHROW(skepu::Vector<int>(N));

	SECTION("Size and capacity are correct")
	{
		skepu::Vector<int> v(N);
		REQUIRE(v.size() == N);
		REQUIRE(v.capacity() == expected_capacity(N));
		v.flush();
		for(size_t i(0); i < N; ++i)
			REQUIRE(v(i) == 0);
	}
}

TEST_CASE("Initialization with size and value")
{
	using size_type = skepu::Vector<int>::size_type;
	size_type N = 100000;

	REQUIRE_NOTHROW(skepu::Vector<int>(N,N));

	SECTION("Size, capacity, and values are correct")
	{
		skepu::Vector<int> v(N, N);
		REQUIRE(v.size() == N);
		REQUIRE(v.capacity() == expected_capacity(N));
		v.flush();
		for(size_type i(0); i < N; ++i)
			REQUIRE(v(i) == (int)N);
	}
}

TEST_CASE("Initialization with initializer_list")
{
	REQUIRE_NOTHROW(skepu::Vector<int>{1,2,3});

	SECTION("of size 1")
	{
		skepu::Vector<int> v{17};
		REQUIRE(v.size() == 1);
		REQUIRE(v.capacity() == expected_capacity(1));
		v.flush();
		CHECK(v(0) == 17);
	}

	SECTION("of size 2")
	{
		skepu::Vector<int> v{1,7};
		REQUIRE(v.size() == 2);
		REQUIRE(v.capacity() == expected_capacity(2));
		v.flush();
		CHECK(v(0) == 1);
		CHECK(v(1) == 7);
	}

	SECTION("of size 3")
	{
		skepu::Vector<int> v{17,3, 11};
		REQUIRE(v.size() == 3);
		REQUIRE(v.capacity() == expected_capacity(3));
		v.flush();
		CHECK(v(0) == 17);
		CHECK(v(1) == 3);
		CHECK(v(2) == 11);
	}

	SECTION("of size 11")
	{
		using size_type = skepu::Vector<int>::size_type;

		skepu::Vector<int> v{0,1,2,3,4,5,6,7,8,9,10};
		REQUIRE(v.size() == 11);
		REQUIRE(v.capacity() == expected_capacity(11));
		v.flush();
		for(size_type i(0); i < 11; ++i)
			REQUIRE(v(i) == (int)i);
	}
}

TEST_CASE("Copy c-tor")
{
	skepu::Vector<int> v{1,2,3};
	REQUIRE_NOTHROW(skepu::Vector<int>{v});

	SECTION("Copy has correct properties and values")
	{
		skepu::Vector<int> u(v);
		REQUIRE(u.size() == v.size());
		REQUIRE(u.capacity() == v.capacity());
		v.flush();
		CHECK(v(0) == 1);
		CHECK(v(1) == 2);
		CHECK(v(2) == 3);
		u.flush();
		CHECK(u(0) == 1);
		CHECK(u(1) == 2);
		CHECK(u(2) == 3);
	}
}

TEST_CASE("Copy assignment")
{
	skepu::Vector<int> v{1,2,3,4,5};
	skepu::Vector<int> u;
	REQUIRE_NOTHROW(u = v);
	REQUIRE(v.size() == 5);
	REQUIRE(v.capacity() == expected_capacity(5));
	REQUIRE(u.size() == 5);
	REQUIRE(u.capacity() == v.capacity());

	v.flush();
	u.flush();
	for(size_t i(0); i < v.size(); ++i)
	{
		REQUIRE(v(i) == (int)(i +1));
		REQUIRE(u(i) == (int)(i +1));
	}
}

TEST_CASE("Move c-tor")
{
	skepu::Vector<int> v{1,2,3};
	REQUIRE_NOTHROW(skepu::Vector<int>(std::move(v)));

	v = skepu::Vector<int>{1,3,5,7,11};
	skepu::Vector<int> u(std::move(v));
	REQUIRE(u.size() == 5);
	REQUIRE(u.capacity() == expected_capacity(5));
	u.flush();
	CHECK(u(0) == 1);
	CHECK(u(1) == 3);
	CHECK(u(2) == 5);
	CHECK(u(3) == 7);
	CHECK(u(4) == 11);
}

TEST_CASE("Move assignment")
{
	skepu::Vector<int> v{1,2,3,4,5};
	skepu::Vector<int> u;
	REQUIRE_NOTHROW(u = std::move(v));

	REQUIRE(u.size() == 5);
	CHECK(u.capacity() == expected_capacity(5));
	u.flush();
	for(size_t i(0); i < u.size(); ++i)
		REQUIRE(u(i) == (int)(i +1));
}

TEST_CASE("Swapping two vectors")
{
	skepu::Vector<int> v{1,2,3,4,5};
	skepu::Vector<int> u{6,7};
	REQUIRE_NOTHROW(std::swap(v, u));
	REQUIRE(v.size() == 2);
	REQUIRE(u.size() == 5);
	v.flush();
	CHECK(v(0) == 6);
	CHECK(v(1) == 7);
	u.flush();
	CHECK(u(0) == 1);
	CHECK(u(1) == 2);
	CHECK(u(2) == 3);
	CHECK(u(3) == 4);
	CHECK(u(4) == 5);
}

TEST_CASE("Can set the value of the element at a specific position")
{
	skepu::Vector<int> v{1,2,3,4,5,6};
	REQUIRE_NOTHROW(v.set(2,42));
	v.flush();
	CHECK(v(0) == 1);
	CHECK(v(1) == 2);
	CHECK(v(2) == 42);
	CHECK(v(3) == 4);
	CHECK(v(4) == 5);
	CHECK(v(5) == 6);
}

TEST_CASE("Begin and end are equal on empty container.")
{
	skepu::Vector<int> v;
	REQUIRE_NOTHROW(v.begin());
	REQUIRE_NOTHROW(v.end());
	v.flush();
	REQUIRE(v.begin() == v.end());

	SECTION("But not equal otherwise")
	{
		v.resize(1);
		v.flush();
		v(0) = 13;
		REQUIRE(v.begin() != v.end());
		CHECK(v.begin() +1 == v.end());
		CHECK(1 + v.begin() == v.end());
		CHECK(v.begin() == v.end() -1);
		CHECK(*(v.begin()) == 13);
	}
}

TEST_CASE("Size_i returns same value as size")
{
	skepu::Vector<float> v(1000, 2.7f);
	CHECK(v.size_i() == v.size());

	SECTION("But size_{j,k,l} returns 0")
	{
		CHECK(v.size_j() == 0);
		CHECK(v.size_k() == 0);
		CHECK(v.size_l() == 0);
	}
}

TEST_CASE("Flushing container created from pointer updates the original array")
{
	size_t const N = 10 * skepu::cluster::mpi_size();
	auto v = new int[N];
	skepu::Vector<int> sv(v, N);

	auto & part = skepu::cont::getParent(sv);
	part.partition();
	part.invalidate_local_storage();

	auto rank = skepu::cluster::mpi_rank();

	int i = 0;
	while((size_t)i < N)
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
	for(i = 0; (size_t)i < N; ++i)
		REQUIRE(v[i] == i);
}
