// User constants in the user functions of the skeletons on distributed containers: the same output as on a single node, whatever the number of processes.

#include <algorithm>
#include <iostream>
#include <skepu>

[[skepu::userconstant]] constexpr float SCALE_F = 2.5f, OFFSET_F = 0.25f;
[[skepu::userconstant]] constexpr int OFFSET_I = -7, LIMIT_I = 100;
constexpr size_t STRIDE_S [[skepu::userconstant]] = 3;
[[skepu::userconstant]] constexpr float DERIVED_F = SCALE_F * 2 + OFFSET_F;
namespace constants { [[skepu::userconstant]] constexpr int NAMESPACED_I = 5; }

float map_float(float x) { return x * SCALE_F + OFFSET_F + DERIVED_F; }
int map_namespaced(int x) { return x * constants::NAMESPACED_I + OFFSET_I; }
size_t map_index(skepu::Index1D i) { return i.i * STRIDE_S; }
int clipped_add(int a, int b) { int s = a + b; return s > LIMIT_I ? LIMIT_I : s; }
int offset_square(int x) { return x * x + OFFSET_I + 7; }
int overlap_weighted(skepu::Region1D<int> r) { return r(-1) * (int)STRIDE_S + r(0) + r(1) * OFFSET_I; }
int pair_combined(int a, int b) { return a * OFFSET_I + b * (int)STRIDE_S; }

template<typename C>
static long long checksum(C &c)
{
	c.flush();
	long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (long long)(c.getAddress()[i] * 100);
	return h;
}

int main()
{
	const size_t n = 103;
	skepu::Vector<float> f(n), fr(n);
	skepu::Vector<int> v(n), vr(n), pre(n);
	skepu::Vector<size_t> sr(n);
	skepu::external([&]
	{
		for (size_t i = 0; i < n; ++i) { f(i) = (float)i; v(i) = (int)(i % 4); }
	}, skepu::write(f, v));

	skepu::Map<1>(map_float)(fr, f);
	std::cout << "map float: " << checksum(fr) << "\n";
	skepu::Map<1>(map_namespaced)(vr, v);
	std::cout << "map namespaced: " << checksum(vr) << "\n";
	skepu::Map<0>(map_index)(sr);
	std::cout << "map index: " << checksum(sr) << "\n";

	std::cout << "reduce: " << (int)skepu::Reduce(clipped_add)(v) << "\n";
	std::cout << "mapreduce: " << (int)skepu::MapReduce<1>(offset_square, clipped_add)(v) << "\n";
	skepu::Scan(clipped_add)(pre, v);
	std::cout << "scan: " << checksum(pre) << "\n";

	auto overlap = skepu::MapOverlap(overlap_weighted);
	overlap.setOverlap(1);
	overlap.setEdgeMode(skepu::Edge::Pad);
	overlap.setPad(0);
	overlap(vr, v);
	std::cout << "mapoverlap: " << checksum(vr) << "\n";

	skepu::Vector<int> a(17), b(11);
	skepu::external([&]
	{
		for (size_t i = 0; i < 17; ++i) a(i) = (int)i + 1;
		for (size_t i = 0; i < 11; ++i) b(i) = 10 * ((int)i + 1);
	}, skepu::write(a, b));
	skepu::Matrix<int> pr(17, 11);
	skepu::MapPairs(pair_combined)(pr, a, b);
	std::cout << "mappairs: " << checksum(pr) << "\n";
	return 0;
}
