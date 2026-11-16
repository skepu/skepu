// isHostValid(): a check of whether a device holds newer data than the host copy; it copies nothing.
#define SKEPU_ENABLE_EXCEPTIONS

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <test_helper_automatic_setup.hpp>

namespace host_valid_part
{
int inc(int a) { return a + 1; }

auto increment = skepu::Map<1>(inc);

static bool device_backend()
{
	const auto type = skepu::internalGlobalBackendSpecAccessor().type();
	return type == skepu::Backend::Type::OpenCL || type == skepu::Backend::Type::CUDA;
}

TEST_CASE("host_valid: a skeleton that wrote a container on a device leaves the host copy stale until it is fetched")
{
	skepu::Vector<int> v(100, 1), r(100, 0);
	skepu::Matrix<int> m(10, 10, 1), mr(10, 10, 0);

	REQUIRE(v.isHostValid());
	REQUIRE(m.isHostValid());

	increment(r, v);
	increment(mr, m);

	// the input was only read
	REQUIRE(v.isHostValid());
	REQUIRE(m.isHostValid());
	if (device_backend())
	{
		REQUIRE(!r.isHostValid());
		REQUIRE(!mr.isHostValid());
	}

	r.updateHost();
	mr.updateHost();
	REQUIRE(r.isHostValid());
	REQUIRE(mr.isHostValid());
	REQUIRE(r(0) == 2);
	REQUIRE(mr(9, 9) == 2);
}
}
