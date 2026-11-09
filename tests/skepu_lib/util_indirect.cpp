#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <skepu-lib/util.hpp>
#include <test_helper_automatic_setup.hpp>

float sq_plus(float x, float y)
{
	return skepu::util::square(x) + y;
}

float bigger_square(float a, float b)
{
	return skepu::util::max(skepu::util::square(a), skepu::util::square(b));
}

int mixed(int a, float b)
{
	return skepu::util::add(a, b) > 5 ? skepu::util::mul(a, 2) : skepu::util::negate(a);
}

float explicit_args(float a, float b)
{
	return skepu::util::add<float>(skepu::util::mul<float>(a, a), skepu::util::sub<float>(b, 1.f));
}

float window(skepu::Region1D<float> r, float scale)
{
	return skepu::util::mul(skepu::util::add<float>(r(-1), r(1)), scale);
}

int scan_max(int a, int b)
{
	return skepu::util::max<int>(a, b);
}

int scan_add(int a, int b)
{
	return skepu::util::add(a, b);
}

float outer(float v, float h)
{
	return skepu::util::sub(skepu::util::mul(v, h), skepu::util::square(h));
}

int pool_max(skepu::Pool1D<int> p)
{
	int m = p(0);
	for (int i = 1; i < p.si; ++i)
		m = skepu::util::max(m, p(i));
	return m;
}

float square_index(skepu::Index1D i)
{
	return skepu::util::square((float)i.i);
}

auto explicit_map = skepu::Map<2>(explicit_args);
auto window_overlap = skepu::MapOverlap(window);
auto max_scan = skepu::Scan(scan_max);
auto add_scan = skepu::Scan(scan_add);
auto max_reduce = skepu::Reduce(scan_max);
auto outer_pairs = skepu::MapPairs<1, 1>(outer);
auto pool_maxima = skepu::MapPool(pool_max);
auto squares = skepu::Map<0>(square_index);
auto sq_plus_map = skepu::Map<2>(sq_plus);
auto bigger_map = skepu::Map<2>(bigger_square);
auto mixed_map = skepu::Map<2>(mixed);
auto sum_sq = skepu::MapReduce<1>([](float x) { return skepu::util::square(x); }, skepu::util::add<float>);

TEST_CASE("util functions called from a user function")
{
	constexpr size_t size{100};
	skepu::Vector<float> a(size, 3.f), b(size, -4.f), r(size);
	sq_plus_map(r, a, b);
	r.flush();
	CHECK(r(0) == 5.f);
	bigger_map(r, a, b);
	r.flush();
	CHECK(r(0) == 16.f);

	skepu::Vector<int> i(size, 4), ri(size);
	skepu::Vector<float> f(size, 2.5f);
	mixed_map(ri, i, f);
	ri.flush();
	CHECK(ri(0) == 8);

	const float total = sum_sq(a);
	CHECK(total == 900.f);

	// template arguments given explicitly inside a user function
	explicit_map(r, a, b);
	r.flush();
	CHECK(r(0) == 4.f);

	// MapOverlap
	skepu::Vector<float> w(size, 3.f), rw(size);
	window_overlap.setOverlap(1);
	window_overlap.setEdgeMode(skepu::Edge::Duplicate);
	window_overlap(rw, w, 0.5f);
	rw.flush();
	CHECK(rw(0) == 3.f);
	CHECK(rw(50) == 3.f);

	// Scan and Reduce with a user function that calls util
	skepu::Vector<int> ones(size, 1), pre(size);
	add_scan.setScanMode(skepu::ScanMode::Inclusive);
	add_scan(pre, ones);
	pre.flush();
	CHECK(pre(0) == 1);
	CHECK(pre(size - 1) == (int)size);

	skepu::Vector<int> ramp(size), running_max(size);
	for (size_t k = 0; k < size; ++k)
		ramp(k) = (int)((k * 7) % 13);
	max_scan.setScanMode(skepu::ScanMode::Inclusive);
	max_scan(running_max, ramp);
	running_max.flush();
	CHECK(running_max(size - 1) == 12);
	const int largest = max_reduce(ramp);
	CHECK(largest == 12);

	// MapPairs
	skepu::Vector<float> v(4, 2.f), h(3, 5.f);
	skepu::Matrix<float> pairs(4, 3);
	outer_pairs(pairs, v, h);
	pairs.flush();
	CHECK(pairs(1, 2) == -15.f);

	// MapPool
	const size_t n = 50;
	skepu::Vector<int> in(n), pooled(n - 2);
	for (size_t k = 0; k < n; ++k)
		in(k) = (int)((k * k) % 17);
	pool_maxima.setPoolSize(3);
	pool_maxima.setStride(1);
	pool_maxima(pooled, in);
	pooled.flush();
	for (size_t k = 0; k + 2 < n; ++k)
		CHECK(pooled(k) == std::max({in(k), in(k + 1), in(k + 2)}));

	// Map without element inputs (index only)
	skepu::Vector<float> gen(size);
	squares(gen);
	gen.flush();
	CHECK(gen(7) == 49.f);
}
