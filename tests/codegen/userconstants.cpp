#include "../../external/catch2/catch.hpp"

#include <algorithm>
#include <skepu>
#include <test_helper_automatic_setup.hpp>

// User constants ([[skepu::userconstant]]) in the user functions of every skeleton, with several types, written both ways
// (before the declaration and after the name), from a namespace, initialized from other constants, used by a function that
// a user function calls, and by a user function that is a template. The host reads the constants too, to check the results.
// (Double precision is in userconstants_double.cpp: OpenCL devices do not all have it.)

[[skepu::userconstant]] constexpr float
	SCALE_F = 2.5f,
	OFFSET_F = 0.25f;

[[skepu::userconstant]] constexpr int
	OFFSET_I = -7,
	LIMIT_I = 100;

constexpr unsigned MASK_U [[skepu::userconstant]] = 0xF0F0u;
constexpr size_t STRIDE_S [[skepu::userconstant]] = 3;
constexpr bool NEGATE [[skepu::userconstant]] = true;

// Initialized from other constants
[[skepu::userconstant]] constexpr float DERIVED_F = SCALE_F * 2 + OFFSET_F;

namespace constants
{
	[[skepu::userconstant]] constexpr int NAMESPACED_I = 5;
}

// ---- Map

float map_float(float x) { return x * SCALE_F + OFFSET_F; }
int map_int(int x) { return x + OFFSET_I; }
unsigned map_unsigned(unsigned x) { return x & MASK_U; }
size_t map_index(skepu::Index1D i) { return i.i * STRIDE_S; }
float map_derived(float x) { return x + DERIVED_F; }
int map_namespaced(int x) { return x * constants::NAMESPACED_I; }
int map_bool(int x) { return NEGATE ? -x : x; }

// A function that a user function calls uses a constant
float scaled(float x) { return x * SCALE_F; }
float map_indirect(float x) { return scaled(x) + OFFSET_F; }

// A user function that is a template
template<typename T>
T map_template(T x) { return x * (T)SCALE_F + (T)OFFSET_I; }

template<typename T>
void apply_template(skepu::Vector<T> &in, skepu::Vector<T> &out)
{
	auto skel = skepu::Map<1>(map_template<T>);
	skel(out, in);
}

auto skel_map_float = skepu::Map<1>(map_float);
auto skel_map_int = skepu::Map<1>(map_int);
auto skel_map_unsigned = skepu::Map<1>(map_unsigned);
auto skel_map_index = skepu::Map<0>(map_index);
auto skel_map_derived = skepu::Map<1>(map_derived);
auto skel_map_namespaced = skepu::Map<1>(map_namespaced);
auto skel_map_bool = skepu::Map<1>(map_bool);
auto skel_map_indirect = skepu::Map<1>(map_indirect);

TEST_CASE("User constants in Map: float, int, unsigned, size_t, bool, derived, in a namespace, indirect")
{
	const size_t n{50};
	skepu::Vector<float> f(n), fr(n);
	skepu::Vector<int> i(n), ir(n);
	skepu::Vector<unsigned> u(n), ur(n);
	skepu::Vector<size_t> sr(n);
	for (size_t k = 0; k < n; ++k) { f[k] = (float)k; i[k] = (int)k; u[k] = 0x1234u + 0x111u * (unsigned)k; }

	skel_map_float(fr, f);
	fr.updateHost();
	for (size_t k = 0; k < n; ++k) REQUIRE(fr(k) == (float)k * SCALE_F + OFFSET_F);

	skel_map_int(ir, i);
	ir.updateHost();
	for (size_t k = 0; k < n; ++k) REQUIRE(ir(k) == (int)k + OFFSET_I);

	skel_map_unsigned(ur, u);
	ur.updateHost();
	for (size_t k = 0; k < n; ++k) REQUIRE(ur(k) == (u(k) & MASK_U));

	skel_map_index(sr);
	sr.updateHost();
	for (size_t k = 0; k < n; ++k) REQUIRE(sr(k) == k * STRIDE_S);

	skel_map_derived(fr, f);
	fr.updateHost();
	for (size_t k = 0; k < n; ++k) REQUIRE(fr(k) == (float)k + DERIVED_F);

	skel_map_namespaced(ir, i);
	ir.updateHost();
	for (size_t k = 0; k < n; ++k) REQUIRE(ir(k) == (int)k * constants::NAMESPACED_I);

	skel_map_bool(ir, i);
	ir.updateHost();
	for (size_t k = 0; k < n; ++k) REQUIRE(ir(k) == (NEGATE ? -(int)k : (int)k));

	skel_map_indirect(fr, f);
	fr.updateHost();
	for (size_t k = 0; k < n; ++k) REQUIRE(fr(k) == (float)k * SCALE_F + OFFSET_F);
}

TEST_CASE("User constants in a Map that is a template, two element types")
{
	const size_t n{30};
	skepu::Vector<float> f(n), fr(n);
	skepu::Vector<int> i(n), ir(n);
	for (size_t k = 0; k < n; ++k) { f[k] = (float)k; i[k] = (int)k; }

	apply_template(f, fr);
	apply_template(i, ir);
	fr.updateHost(); ir.updateHost();
	for (size_t k = 0; k < n; ++k)
	{
		REQUIRE(fr(k) == (float)k * SCALE_F + (float)OFFSET_I);
		REQUIRE(ir(k) == (int)k * (int)SCALE_F + OFFSET_I);
	}
}

// ---- Reduce, MapReduce, Scan: the constant is in the reducing function (a sum that is clipped at the limit, which is associative for non-negative numbers)

int clipped_add(int a, int b) { int s = a + b; return s > LIMIT_I ? LIMIT_I : s; }
int offset_square(int x) { return x * x + OFFSET_I + 7; }

auto skel_reduce = skepu::Reduce(clipped_add);
auto skel_mapreduce = skepu::MapReduce<1>(offset_square, clipped_add);
auto skel_scan = skepu::Scan(clipped_add);

TEST_CASE("User constants in Reduce, MapReduce and Scan")
{
	const size_t n{40};
	skepu::Vector<int> v(n), prefix(n);
	for (size_t k = 0; k < n; ++k) v[k] = (int)(k % 4);

	int expected = 0;
	for (size_t k = 0; k < n; ++k) expected = std::min(LIMIT_I, expected + (int)(k % 4));
	REQUIRE((int)skel_reduce(v) == expected);

	int mapped = 0;
	for (size_t k = 0; k < n; ++k) mapped = std::min(LIMIT_I, mapped + (int)((k % 4) * (k % 4)));
	REQUIRE((int)skel_mapreduce(v) == mapped);

	skel_scan(prefix, v);
	prefix.updateHost();
	int running = 0;
	for (size_t k = 0; k < n; ++k)
	{
		running = std::min(LIMIT_I, running + (int)(k % 4));
		REQUIRE(prefix(k) == running);
	}
}

// ---- MapOverlap and MapPairs

int overlap_weighted(skepu::Region1D<int> r) { return r(-1) * STRIDE_S + r(0) + r(1) * OFFSET_I; }
int overlap_weighted_2d(skepu::Region2D<int> r) { return r(-1, 0) + r(1, 0) + r(0, -1) * OFFSET_I + r(0, 1) * (int)STRIDE_S; }
int pair_combined(int a, int b) { return a * OFFSET_I + b * (int)STRIDE_S; }

TEST_CASE("User constants in MapOverlap (1D and 2D) and MapPairs")
{
	const size_t n{40};
	skepu::Vector<int> v(n), r(n);
	for (size_t k = 0; k < n; ++k) v[k] = (int)(k * 3 % 11);

	auto overlap = skepu::MapOverlap(overlap_weighted);
	overlap.setOverlap(1);
	overlap.setEdgeMode(skepu::Edge::Pad);
	overlap.setPad(0);
	overlap(r, v);
	r.updateHost();
	for (size_t k = 0; k < n; ++k)
	{
		const int left = k > 0 ? v(k - 1) : 0, right = k + 1 < n ? v(k + 1) : 0;
		REQUIRE(r(k) == left * (int)STRIDE_S + v(k) + right * OFFSET_I);
	}

	const size_t rows{12}, cols{9};
	skepu::Matrix<int> m(rows, cols), mr(rows, cols);
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < cols; ++j)
			m(i, j) = (int)((i * 7 + j * 3) % 13);
	m.flush();
	auto overlap2d = skepu::MapOverlap(overlap_weighted_2d);
	overlap2d.setOverlap(1, 1);
	overlap2d.setEdgeMode(skepu::Edge::Pad);
	overlap2d.setPad(0);
	overlap2d(mr, m);
	mr.updateHost();
	auto at = [&](long i, long j) { return (i < 0 || j < 0 || i >= (long)rows || j >= (long)cols) ? 0 : m((size_t)i, (size_t)j); };
	for (long i = 0; i < (long)rows; ++i)
		for (long j = 0; j < (long)cols; ++j)
			REQUIRE(mr((size_t)i, (size_t)j) == at(i - 1, j) + at(i + 1, j) + at(i, j - 1) * OFFSET_I + at(i, j + 1) * (int)STRIDE_S);

	skepu::Vector<int> a(7), b(5);
	for (int k = 0; k < 7; ++k) a[k] = k + 1;
	for (int k = 0; k < 5; ++k) b[k] = 10 * (k + 1);
	skepu::Matrix<int> pr(7, 5);
	auto pairs = skepu::MapPairs(pair_combined);
	pairs(pr, a, b);
	pr.updateHost();
	for (size_t i = 0; i < 7; ++i)
		for (size_t j = 0; j < 5; ++j)
			REQUIRE(pr(i, j) == a(i) * OFFSET_I + b(j) * (int)STRIDE_S);
}
