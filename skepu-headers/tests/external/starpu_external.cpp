#include <random>

#include <catch2/catch_test_macros.hpp>

#include <skepu3/cluster/external.hpp>
#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/tensor3/tensor3.hpp>
#include <skepu3/cluster/containers/tensor4/tensor4.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> int_dist;
std::uniform_real_distribution<float> float_dist;

TEST_CASE("Only rank 0 calls operator")
{
	bool reched_operator(false);

	REQUIRE_NOTHROW(
		skepu::external([&]
		{
			reched_operator = true;
		}));

	if(!skepu::cluster::mpi_rank())
		CHECK(reched_operator);
	else
		CHECK(!reched_operator);
}

template<typename T>
struct container_stub
{
	bool flushed;
	bool writeable;
	bool partitioned;

	void gather_to_root() { flushed = true; }
	void make_ext_w() { writeable = true; }
	void scatter_from_root() { partitioned = true; }

	auto inline
	getParent()
	-> container_stub &
	{
		return *this;
	}
};

namespace skepu {

template<>
struct is_skepu_container<container_stub<int> &> : public std::true_type
{};

} // namespace skepu

TEST_CASE("Read only external call only gathers container")
{
	container_stub<int> c{false,false,false};

	REQUIRE_NOTHROW(
		skepu::external(
			skepu::read(c),
			[&]{}));

	CHECK(c.flushed);
	CHECK_FALSE(c.writeable);
	CHECK_FALSE(c.partitioned);
}

TEST_CASE("Write only external call only scatters container")
{
	container_stub<int> c{false,false,false};

	REQUIRE_NOTHROW(
		skepu::external(
			[&]{},
			skepu::write(c)));

	CHECK_FALSE(c.flushed);
	CHECK(c.writeable);
	CHECK(c.partitioned);
}

TEST_CASE("Read and Write gathers and scatters containers")
{
	container_stub<int> c{false,false,false};

	REQUIRE_NOTHROW(
		skepu::external(
			skepu::read(c),
			[&]{},
			skepu::write(c)));

	CHECK(c.flushed);
	CHECK(c.writeable);
	CHECK(c.partitioned);
}

TEST_CASE("Two containers in read write external call")
{
	container_stub<int> c1{false,false,false};
	container_stub<int> c2{false,false,false};

	REQUIRE_NOTHROW(
		skepu::external(
			skepu::read(c1, c2),
			[&]{},
			skepu::write(c1, c2)));

	CHECK(c1.flushed);
	CHECK(c1.writeable);
	CHECK(c1.partitioned);
	CHECK(c2.flushed);
	CHECK(c2.writeable);
	CHECK(c2.partitioned);
}

TEST_CASE("One read and one write container in external call")
{
	container_stub<int> c1{false,false,false};
	container_stub<int> c2{false,false,false};

	REQUIRE_NOTHROW(
		skepu::external(
			skepu::read(c1),
			[&]{},
			skepu::write(c2)));

	CHECK(c1.flushed);
	CHECK_FALSE(c1.writeable);
	CHECK_FALSE(c1.partitioned);
	CHECK_FALSE(c2.flushed);
	CHECK(c2.writeable);
	CHECK(c2.partitioned);
}

TEST_CASE("Vector is gathered before operator call")
{
	size_t constexpr N{10};
	auto expected = std::vector<int>(N * skepu::cluster::mpi_size());
	auto local = std::vector<int>(N);
	for(auto & e : local)
		e = int_dist(gen);
	MPI_Gather(
		local.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);
	skepu::Vector<int> v(N * skepu::cluster::mpi_size());
	auto & partition = skepu::cont::getParent(v);
	auto handle = partition.handle_for(N * skepu::cluster::mpi_rank());
	starpu_data_acquire(handle, STARPU_RW);
	auto part_it = (int *)starpu_vector_get_local_ptr(handle);
	std::copy(local.begin(), local.end(), part_it);
	starpu_data_release(handle);

	skepu::external(
		skepu::read(v),
		[&]
		{
			auto v_it = v.begin();
			auto v_end = v.end();
			auto e_it = expected.begin();
			for(; v_it != v_end; ++v_it, ++e_it)
				REQUIRE(*v_it == *e_it);
		}
	);
}

TEST_CASE("Vector is scattered after the operator call")
{
	size_t constexpr N{1000};
	auto init_v = std::vector<int>(N * skepu::cluster::mpi_size());
	auto expected = std::vector<int>(N);
	for(auto & e : init_v)
		e = int_dist(gen);
	MPI_Scatter(
		init_v.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);

	skepu::Vector<int> v(N * skepu::cluster::mpi_size());
	skepu::external(
		[&]
		{
			auto v_it = v.begin();
			auto v_end = v.end();
			auto i_it = init_v.begin();
			for(; v_it != v_end; ++v_it, ++i_it)
				*v_it = *i_it;
		},
		skepu::write(v));

	auto & partition = skepu::cont::getParent(v);
	auto handle = partition.handle_for(skepu::cluster::mpi_rank() * N);
	starpu_data_acquire(handle, STARPU_R);
	auto it = (int *)starpu_vector_get_local_ptr(handle);
	for(size_t i(0); i < N; ++i)
		REQUIRE(it[i] == expected[i]);
	starpu_data_release(handle);
}

TEST_CASE("Vectors can be initialized in external calls")
{
	SECTION("initialied with size 1")
	{
		size_t constexpr N{1};
		std::vector<float> expected_values(N);

		skepu::Vector<float> v;
		skepu::external(
			[&]
			{
				v.init(N);
				auto expected_it = expected_values.begin();
				for(auto v_it = v.begin(); v_it != v.end(); ++v_it, ++expected_it)
				{
					*expected_it = float_dist(gen);
					*v_it = *expected_it;
				}
			},
			skepu::write(v)
		);

		MPI_Bcast(expected_values.data(), expected_values.size(), MPI_FLOAT,
			0, MPI_COMM_WORLD);
		REQUIRE(v.size() == expected_values.size());
		v.flush();
		auto v_it = v.begin();
		for(auto & val : expected_values)
			REQUIRE(*(v_it++) == val);
	}

	SECTION("initialied with size 1000")
	{
		size_t constexpr N{1000};
		std::vector<float> expected_values(N);

		skepu::Vector<float> v;
		skepu::external(
			[&]
			{
				v.init(N);
				auto expected_it = expected_values.begin();
				for(auto v_it = v.begin(); v_it != v.end(); ++v_it, ++expected_it)
				{
					*expected_it = float_dist(gen);
					*v_it = *expected_it;
				}
			},
			skepu::write(v)
		);

		MPI_Bcast(expected_values.data(), expected_values.size(), MPI_FLOAT,
			0, MPI_COMM_WORLD);
		REQUIRE(v.size() == expected_values.size());
		v.flush();
		auto v_it = v.begin();
		for(auto & val : expected_values)
			REQUIRE(*(v_it++) == val);
	}
}

TEST_CASE("Matrix is gathered before operator call")
{
	size_t constexpr R{10};
	size_t constexpr C{10};
	size_t constexpr N{R*C};
	auto expected = std::vector<int>(N * skepu::cluster::mpi_size());
	auto local = std::vector<int>(N);
	for(auto & e : local)
		e = int_dist(gen);
	MPI_Gather(
		local.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);
	skepu::Matrix<int> m(R * skepu::cluster::mpi_size(), C);
	auto & partition = skepu::cont::getParent(m);
	auto handle = partition.handle_for(N * skepu::cluster::mpi_rank());
	starpu_data_acquire(handle, STARPU_RW);
	auto part_it = (int *)starpu_matrix_get_local_ptr(handle);
	std::copy(local.begin(), local.end(), part_it);
	starpu_data_release(handle);

	skepu::external(
		skepu::read(m),
		[&]
		{
			auto v_it = m.begin();
			auto v_end = m.end();
			auto e_it = expected.begin();
			for(; v_it != v_end; ++v_it, ++e_it)
				REQUIRE(*v_it == *e_it);
		}
	);
}

TEST_CASE("Matrix is scattered after the operator call")
{
	size_t constexpr R{10};
	size_t constexpr C{10};
	size_t constexpr N{R*C};
	auto init_v = std::vector<int>(N * skepu::cluster::mpi_size());
	auto expected = std::vector<int>(N);
	for(auto & e : init_v)
		e = int_dist(gen);
	MPI_Scatter(
		init_v.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);

	skepu::Matrix<int> m(R * skepu::cluster::mpi_size(), C);
	skepu::external(
		[&]
		{
			auto v_it = m.begin();
			auto v_end = m.end();
			auto i_it = init_v.begin();
			for(; v_it != v_end; ++v_it, ++i_it)
				*v_it = *i_it;
		},
		skepu::write(m)
	);

	auto & partition = skepu::cont::getParent(m);
	auto handle = partition.handle_for(skepu::cluster::mpi_rank() * N);
	starpu_data_acquire(handle, STARPU_R);
	auto it = (int *)starpu_matrix_get_local_ptr(handle);
	for(size_t i(0); i < N; ++i)
		REQUIRE(it[i] == expected[i]);
	starpu_data_release(handle);
}

TEST_CASE("Matrices can be initialized in external calls")
{
	SECTION("initialized with size (1,1)")
	{
		size_t constexpr R{1};
		size_t constexpr C{1};
		std::vector<float> expected_values(R*C);

		skepu::Matrix<float> m;
		skepu::external(
			[&]
			{
				m.init(R,C);
				auto expected_it = expected_values.begin();
				for(auto m_it = m.begin(); m_it != m.end(); ++m_it, ++expected_it)
				{
					*expected_it = float_dist(gen);
					*m_it = *expected_it;
				}
			},
			skepu::write(m)
		);

		MPI_Bcast(expected_values.data(), expected_values.size(), MPI_FLOAT,
			0, MPI_COMM_WORLD);
		REQUIRE(m.size() == expected_values.size());
		REQUIRE(m.size_i() == R);
		REQUIRE(m.size_j() == C);
		m.flush();
		auto m_it = m.begin();
		for(auto & val : expected_values)
			REQUIRE(*(m_it++) == val);
	}

	SECTION("initialized with size (100,100)")
	{
		size_t constexpr R{100};
		size_t constexpr C{100};
		std::vector<float> expected_values(R*C);

		skepu::Matrix<float> m;
		skepu::external(
			[&]
			{
				m.init(R,C);
				auto expected_it = expected_values.begin();
				for(auto m_it = m.begin(); m_it != m.end(); ++m_it, ++expected_it)
				{
					*expected_it = float_dist(gen);
					*m_it = *expected_it;
				}
			},
			skepu::write(m)
		);

		MPI_Bcast(expected_values.data(), expected_values.size(), MPI_FLOAT,
			0, MPI_COMM_WORLD);
		REQUIRE(m.size() == expected_values.size());
		REQUIRE(m.size_i() == R);
		REQUIRE(m.size_j() == C);
		m.flush();
		auto m_it = m.begin();
		for(auto & val : expected_values)
			REQUIRE(*(m_it++) == val);
	}
}

TEST_CASE("Tensor3 is gathered before operator call")
{
	size_t constexpr I{10};
	size_t constexpr J{10};
	size_t constexpr K{10};
	size_t constexpr N{I*J*K};
	auto expected = std::vector<int>(N * skepu::cluster::mpi_size());
	auto local = std::vector<int>(N);
	for(auto & e : local)
		e = int_dist(gen);
	MPI_Gather(
		local.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);
	skepu::Tensor3<int> t3(I * skepu::cluster::mpi_size(), J, K);
	auto & partition = skepu::cont::getParent(t3);

	auto handle = partition.handle_for(N * skepu::cluster::mpi_rank());
	starpu_data_acquire(handle, STARPU_RW);
	auto part_it = (int *)starpu_block_get_local_ptr(handle);
	std::copy(local.begin(), local.end(), part_it);
	starpu_data_release(handle);

	skepu::external(
		skepu::read(t3),
		[&]
		{
			auto v_it = t3.begin();
			auto v_end = t3.end();
			auto e_it = expected.begin();
			for(; v_it != v_end; ++v_it, ++e_it)
				REQUIRE(*v_it == *e_it);
		}
	);
}

TEST_CASE("Tensor3 is scattered after the operator call")
{
	size_t constexpr I{10};
	size_t constexpr J{10};
	size_t constexpr K{10};
	size_t constexpr N{I*J*K};
	auto init_v = std::vector<int>(N * skepu::cluster::mpi_size());
	auto expected = std::vector<int>(N);
	for(auto & e : init_v)
		e = int_dist(gen);
	MPI_Scatter(
		init_v.data(), N, MPI_INT,
		expected.data(), N, MPI_INT,
		0, MPI_COMM_WORLD);

	skepu::Tensor3<int> t3(I * skepu::cluster::mpi_size(), J, K);
	skepu::external(
		[&]
		{
			auto v_it = t3.begin();
			auto v_end = t3.end();
			auto i_it = init_v.begin();
			for(; v_it != v_end; ++v_it, ++i_it)
				*v_it = *i_it;
		},
		skepu::write(t3)
	);

	auto & partition = skepu::cont::getParent(t3);
	auto handle = partition.handle_for(skepu::cluster::mpi_rank() * N);
	starpu_data_acquire(handle, STARPU_R);
	auto it = (int *)starpu_block_get_local_ptr(handle);
	for(size_t i(0); i < N; ++i)
		REQUIRE(it[i] == expected[i]);
	starpu_data_release(handle);
}

TEST_CASE("Tensor3 can be initialized in external calls")
{
	SECTION("initialized with size (1,1,1)")
	{
		size_t constexpr I{1};
		size_t constexpr J{1};
		size_t constexpr K{1};
		std::vector<float> expected_values(I*J*K);

		skepu::Tensor3<float> t3;
		skepu::external(
			[&]
			{
				t3.init(I,J,K);
				auto expected_it = expected_values.begin();
				for(auto t3_it = t3.begin(); t3_it != t3.end(); ++t3_it, ++expected_it)
				{
					*expected_it = float_dist(gen);
					*t3_it = *expected_it;
				}
			},
			skepu::write(t3)
		);

		MPI_Bcast(expected_values.data(), expected_values.size(), MPI_FLOAT,
			0, MPI_COMM_WORLD);
		REQUIRE(t3.size() == expected_values.size());
		REQUIRE(t3.size_i() == I);
		REQUIRE(t3.size_j() == J);
		REQUIRE(t3.size_k() == K);
		t3.flush();
		auto t3_it = t3.begin();
		for(auto & val : expected_values)
			REQUIRE(*(t3_it++) == val);
	}

	SECTION("initialized with size (10,10,10)")
	{
		size_t constexpr I{10};
		size_t constexpr J{10};
		size_t constexpr K{10};
		std::vector<float> expected_values(I*J*K);

		skepu::Tensor3<float> t3;
		skepu::external(
			[&]
			{
				t3.init(I,J,K);
				auto expected_it = expected_values.begin();
				for(auto t3_it = t3.begin(); t3_it != t3.end(); ++t3_it, ++expected_it)
				{
					*expected_it = float_dist(gen);
					*t3_it = *expected_it;
				}
			},
			skepu::write(t3)
		);

		MPI_Bcast(expected_values.data(), expected_values.size(), MPI_FLOAT,
			0, MPI_COMM_WORLD);
		REQUIRE(t3.size() == expected_values.size());
		REQUIRE(t3.size_i() == I);
		REQUIRE(t3.size_j() == J);
		REQUIRE(t3.size_k() == K);
		t3.flush();
		auto t3_it = t3.begin();
		for(auto & val : expected_values)
			REQUIRE(*(t3_it++) == val);
	}
}

TEST_CASE("Tensor 4 initialization")
{
	SECTION("Initialize with operator()")
	{
		size_t constexpr I{10};
		size_t constexpr J{10};
		size_t constexpr K{10};
		size_t constexpr L{10};
		std::vector<float> expected_values(I*J*K*L);

		skepu::Tensor4<float> t4;
		skepu::external(
			[&]
			{
				t4.init(I,J,K,L);
				for(size_t i(0); i < I; ++i)
					for(size_t j(0); j < J; ++j)
						for(size_t k(0); k < K; ++k)
							for(size_t l(0); l < L; ++l)
							{
								size_t exp_pos =
									(i * J*K*L) + (j * K*L) + (k * L) + l;
								expected_values.at(exp_pos) = float_dist(gen);
								t4(i,j,k,l) = expected_values[exp_pos];
							}
			},
			skepu::write(t4)
		);

		MPI_Bcast(expected_values.data(), expected_values.size(), MPI_FLOAT,
			0, MPI_COMM_WORLD);
		REQUIRE(t4.size() == expected_values.size());
		REQUIRE(t4.size_i() == I);
		REQUIRE(t4.size_j() == J);
		REQUIRE(t4.size_k() == K);
		REQUIRE(t4.size_l() == L);
		t4.flush();
		auto t4_it = t4.begin();
		for(auto & val : expected_values)
			REQUIRE(*(t4_it++) == val);
	}

	SECTION("Initialize with iterators")
	{
		size_t constexpr I{10};
		size_t constexpr J{10};
		size_t constexpr K{10};
		size_t constexpr L{10};
		std::vector<float> expected_values(I*J*K*L);

		skepu::Tensor4<float> t4;
		skepu::external(
			[&]
			{
				t4.init(I,J,K,L);
				auto expected_it = expected_values.begin();
				for(auto t4_it = t4.begin(); t4_it != t4.end(); ++t4_it, ++expected_it)
				{
					*expected_it = float_dist(gen);
					*t4_it = *expected_it;
				}
			},
			skepu::write(t4)
		);

		MPI_Bcast(expected_values.data(), expected_values.size(), MPI_FLOAT,
			0, MPI_COMM_WORLD);
		REQUIRE(t4.size() == expected_values.size());
		REQUIRE(t4.size_i() == I);
		REQUIRE(t4.size_j() == J);
		REQUIRE(t4.size_k() == K);
		REQUIRE(t4.size_l() == L);
		t4.flush();
		auto t4_it = t4.begin();
		for(auto & val : expected_values)
			REQUIRE(*(t4_it++) == val);
	}
}
