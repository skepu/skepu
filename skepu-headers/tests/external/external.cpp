#include <catch2/catch_test_macros.hpp>

#include <skepu>


TEST_CASE("External calls the supplied operator")
{
	bool operator_called{false};

	skepu::external([&]
	{
		operator_called = true;
	});

	CHECK(operator_called);
}

struct container_stub
{
	bool flushed;
	bool invalidated;

	auto inline
	flush()
	-> void
	{
		flushed = true;
	}
	
	auto inline
	invalidateDeviceData() noexcept
	-> void
	{
		invalidated = true;
	}
};

namespace skepu {

template<>
struct is_skepu_container<container_stub &>
: std::true_type
{};

} // namespace skepu

TEST_CASE("One container is flushed")
{
	container_stub c{false};

	REQUIRE_FALSE(c.flushed);
	REQUIRE_NOTHROW(
		skepu::external(
			skepu::read(c),
			[&]{}
	));
	CHECK(c.flushed);
}

TEST_CASE("Two containers are flushed")
{
	container_stub c1{false};
	container_stub c2{false};

	REQUIRE_FALSE(c1.flushed);
	REQUIRE_FALSE(c2.flushed);
	REQUIRE_NOTHROW(
		skepu::external(
			skepu::read(c1, c2),
			[&]{}
	));
	CHECK(c1.flushed);
	CHECK(c2.flushed);
}

TEST_CASE("Write only containers are not flushed")
{
	container_stub c1{false};
	container_stub c2{false};

	REQUIRE_FALSE(c1.flushed);
	REQUIRE_FALSE(c2.flushed);
	REQUIRE_NOTHROW(
		skepu::external(
			[&]{},
			skepu::write(c1, c2)
	));

	CHECK_FALSE(c1.flushed);
	CHECK_FALSE(c2.flushed);
}
