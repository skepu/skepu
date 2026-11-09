#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <skepu-lib/util.hpp>
#include <limits>
#include <test_helper_automatic_setup.hpp>

auto equal_map = skepu::Map(skepu::util::equal<int>);
auto not_equal_map = skepu::Map(skepu::util::not_equal<int>);
auto less_map = skepu::Map(skepu::util::less<int>);
auto less_equal_map = skepu::Map(skepu::util::less_equal<int>);
auto greater_map = skepu::Map(skepu::util::greater<int>);
auto greater_equal_map = skepu::Map(skepu::util::greater_equal<int>);

auto greater_than = skepu::Map<1>(skepu::util::greater<float, float>);
auto less_than_mixed = skepu::Map(skepu::util::less<float, int>);

auto zero_map = skepu::Map(skepu::util::is_zero<float>);
auto nonzero_map = skepu::Map(skepu::util::is_nonzero<float>);
auto positive_map = skepu::Map(skepu::util::is_positive<float>);
auto negative_map = skepu::Map(skepu::util::is_negative<float>);
auto nan_map = skepu::Map(skepu::util::is_nan<float>);
auto finite_map = skepu::Map(skepu::util::is_finite<float>);

auto range_map = skepu::Map<1>(skepu::util::in_range<float, float>);
auto close_map = skepu::Map<2>(skepu::util::is_close<float, float>);
auto close_same = skepu::Map<3>(skepu::util::is_close<float>);

auto not_map = skepu::Map(skepu::util::logical_not<int>);
auto any_of = skepu::Reduce(skepu::util::logical_or<int>);
auto all_of = skepu::Reduce(skepu::util::logical_and<int>);
auto count_positive = skepu::MapReduce<1>(skepu::util::is_positive<float>, skepu::util::add<int>);

TEST_CASE("util comparisons")
{
	constexpr size_t n = 12;
	skepu::Vector<int> a(n), b(n, 2), r(n);
	for (size_t i = 0; i < n; ++i)
		a(i) = (int)(i % 4);

	equal_map(r, a, b);
	r.flush();
	for (size_t i = 0; i < n; ++i) CHECK(r(i) == (a(i) == 2));
	not_equal_map(r, a, b);
	r.flush();
	for (size_t i = 0; i < n; ++i) CHECK(r(i) == (a(i) != 2));
	less_map(r, a, b);
	r.flush();
	for (size_t i = 0; i < n; ++i) CHECK(r(i) == (a(i) < 2));
	less_equal_map(r, a, b);
	r.flush();
	for (size_t i = 0; i < n; ++i) CHECK(r(i) == (a(i) <= 2));
	greater_map(r, a, b);
	r.flush();
	for (size_t i = 0; i < n; ++i) CHECK(r(i) == (a(i) > 2));
	greater_equal_map(r, a, b);
	r.flush();
	for (size_t i = 0; i < n; ++i) CHECK(r(i) == (a(i) >= 2));
}

TEST_CASE("util comparisons with a uniform and with operands of different types")
{
	constexpr size_t n = 8;
	skepu::Vector<float> a(n);
	skepu::Vector<int> bound(n, 3), r(n);
	for (size_t i = 0; i < n; ++i)
		a(i) = (float)i + 0.5f;

	greater_than(r, a, 3.f);
	r.flush();
	for (size_t i = 0; i < n; ++i) CHECK(r(i) == (a(i) > 3.f));

	less_than_mixed(r, a, bound);
	r.flush();
	for (size_t i = 0; i < n; ++i) CHECK(r(i) == (a(i) < 3));
}

TEST_CASE("util unary predicates")
{
	const float nan = std::numeric_limits<float>::quiet_NaN();
	const float inf = std::numeric_limits<float>::infinity();
	const std::vector<float> values{0.f, -1.f, 2.f, nan, inf, -inf, 1e-30f};
	skepu::Vector<float> v(values.size());
	for (size_t i = 0; i < values.size(); ++i)
		v(i) = values[i];
	skepu::Vector<int> r(values.size());

	zero_map(r, v);
	r.flush();
	for (size_t i = 0; i < values.size(); ++i) CHECK(r(i) == (values[i] == 0));
	nonzero_map(r, v);
	r.flush();
	for (size_t i = 0; i < values.size(); ++i) CHECK(r(i) == (values[i] != 0));
	positive_map(r, v);
	r.flush();
	for (size_t i = 0; i < values.size(); ++i) CHECK(r(i) == (values[i] > 0));
	negative_map(r, v);
	r.flush();
	for (size_t i = 0; i < values.size(); ++i) CHECK(r(i) == (values[i] < 0));
	nan_map(r, v);
	r.flush();
	for (size_t i = 0; i < values.size(); ++i) CHECK(r(i) == (values[i] != values[i]));
	finite_map(r, v);
	r.flush();
	for (size_t i = 0; i < values.size(); ++i) CHECK(r(i) == (i != 3 && i != 4 && i != 5));
}

TEST_CASE("util in_range and is_close")
{
	skepu::Vector<float> v(6), r_in(6);
	skepu::Vector<int> r(6);
	const float xs[] = {-1.f, 0.f, 0.5f, 1.f, 1.0001f, 7.f};
	for (size_t i = 0; i < 6; ++i)
		v(i) = xs[i];
	range_map(r, v, 0.f, 1.f);
	r.flush();
	const int in_expected[] = {0, 1, 1, 1, 0, 0};
	for (size_t i = 0; i < 6; ++i) CHECK(r(i) == in_expected[i]);

	skepu::Vector<float> a(5), b(5);
	const float av[] = {1.f, 1000.f, 1000.f, 0.f, -5.f};
	const float bv[] = {1.0005f, 1000.5f, 1002.f, 0.01f, -5.001f};
	for (size_t i = 0; i < 5; ++i)
	{
		a(i) = av[i];
		b(i) = bv[i];
	}
	skepu::Vector<int> c(5);
	const int close_expected[] = {1, 1, 0, 0, 1};
	close_map(c, a, b, 1e-3f);
	c.flush();
	for (size_t i = 0; i < 5; ++i) CHECK(c(i) == close_expected[i]);

	skepu::Vector<float> tol(5, 1e-3f);
	close_same(c, a, b, tol);
	c.flush();
	for (size_t i = 0; i < 5; ++i) CHECK(c(i) == close_expected[i]);
}

TEST_CASE("util logical functions and counting")
{
	constexpr size_t n = 20;
	skepu::Vector<int> none(n, 0), some(n, 0), every(n, 1), r(n);
	some(7) = 1;

	all_of.setStartValue(1);
	const int any_none = any_of(none), any_some = any_of(some), all_some = all_of(some), all_every = all_of(every);
	CHECK(any_none == 0);
	CHECK(any_some == 1);
	CHECK(all_some == 0);
	CHECK(all_every == 1);

	not_map(r, some);
	r.flush();
	for (size_t i = 0; i < n; ++i) CHECK(r(i) == (i != 7));

	skepu::Vector<float> v(n);
	for (size_t i = 0; i < n; ++i)
		v(i) = (float)i - 4.5f;
	const int positives = count_positive(v);
	CHECK(positives == 15);
}
