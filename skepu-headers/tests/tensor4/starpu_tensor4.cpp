#include <catch2/catch_test_macros.hpp>

#include <skepu3/cluster/containers/tensor4/tensor4.hpp>

TEST_CASE("Default constructor")
{
	REQUIRE_NOTHROW(skepu::Tensor4<int>());

	SECTION("has no size")
	{
			auto tensor = skepu::Tensor4<int>();
			CHECK(tensor.size() == 0);
	}
}

TEST_CASE("Initialization with size")
{
	size_t constexpr I{1};
	size_t constexpr J{2};
	size_t constexpr K{3};
	size_t constexpr L{5};
	REQUIRE_NOTHROW(skepu::Tensor4<int>(I,J,K,L));
	skepu::Tensor4<int> tensor(I,J,K,L);
	REQUIRE(tensor.size() == (I*J*K*L));
	REQUIRE(tensor.size_i() == I);
	REQUIRE(tensor.size_j() == J);
	REQUIRE(tensor.size_k() == K);
	REQUIRE(tensor.size_l() == L);

	REQUIRE_NOTHROW(tensor.flush());
	for(size_t i(0); i < I; ++i)
		for(size_t j(0); j < J; ++j)
			for(size_t k(0); k < K; ++k)
				for(size_t l(0); l < L; ++l)
					REQUIRE(tensor(i,j,k,l) == 0);
}

TEST_CASE("Initialisation with size and value")
{
	size_t constexpr I{10};
	size_t constexpr J{7};
	size_t constexpr K{3};
	size_t constexpr L{3};
	int const val(11);
	REQUIRE_NOTHROW(skepu::Tensor4<int>(I, J, K, L, val));
	skepu::Tensor4<int> tensor(I, J, K, L, val);
	REQUIRE(tensor.size() == (I*J*K*L));
	REQUIRE(tensor.size_i() == I);
	REQUIRE(tensor.size_j() == J);
	REQUIRE(tensor.size_k() == K);
	REQUIRE(tensor.size_l() == L);

	REQUIRE_NOTHROW(tensor.flush());
	for(size_t i(0); i < I; ++i)
		for(size_t j(0); j < J; ++j)
			for(size_t k(0); k < K; ++k)
				for(size_t l(0); l < L; ++l)
					REQUIRE(tensor(i,j,k,l) == val);
}

TEST_CASE("Iterator")
{
	constexpr size_t N(10);
	skepu::Tensor4<int> tensor(N,N,N,N);
	std::vector<int> expected;
	expected.reserve(N*N*N*N);

	tensor.flush();
	for(size_t i(0); i < N; ++i)
		for(size_t j(0); j < N; ++j)
			for(size_t k(0); k < N; ++k)
				for(size_t l(0); l < N; ++l)
				{
					tensor(i,j,k,l) = i*j*k*l;
					expected.emplace_back(i*j*k*l);
				}
	REQUIRE(tensor.size() == expected.size());

	auto tens_it(tensor.begin());
	auto tens_end_it(tensor.end());
	auto exp_it(expected.begin());

	for(; tens_it != tens_end_it; ++tens_it, ++exp_it)
		REQUIRE(*tens_it == *exp_it);
}

TEST_CASE("Flushing container created from pointer updates the original array")
{
	size_t const I = 10 * skepu::cluster::mpi_size();
	size_t const J = 10;
	size_t const K = 10;
	size_t const L = 10;
	auto data = new int[I*J*K*L];
	skepu::Tensor4<int> t4(data, I, J, K, L);

	auto & part = skepu::cont::getParent(t4);
	part.partition();
	part.invalidate_local_storage();

	auto rank = skepu::cluster::mpi_rank();

	int i = 0;
	while((size_t)i < I*J*K*L)
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

	t4.flush();
	for(i = 0; (size_t)i < I*J*K*L; ++i)
		REQUIRE(data[i] == i);
}
