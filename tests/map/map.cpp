#include <catch2/catch.hpp>

#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>

auto mult = skepu::Map(skepu::util::mul<int>);
	
TEST_CASE("Map fundamentals")
{
	constexpr size_t size{1000};
	
	skepu::Vector<int> v1(size, 3), v2(size, 7), r(size);
	
	mult(r, v1, v2);
  r.flush();
  CHECK(r(0) == 21);
	
	mult(r.begin(), r.end(), v1.begin(), v2.begin());
  r.flush();
  CHECK(r(0) == 21);
	
	mult(r.begin(), r.end(), v1, v2);
  r.flush();
  CHECK(r(0) == 21);
	
	mult(r, v1.begin(), v2);
  r.flush();
  CHECK(r(0) == 21);
}

