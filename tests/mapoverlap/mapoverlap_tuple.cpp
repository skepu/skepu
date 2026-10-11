#include "../../external/catch2/catch.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include <skepu>
#include <test_helper_automatic_setup.hpp>


skepu::multiple<float, int>
over_1d_multi(skepu::Index1D idx, skepu::Region1D<float> r, int scale)
{
	return skepu::ret((r(-2)*4 + r(-1)*2 + r(0) + r(1)*2 + r(2)*4) / scale, idx.i);
}

skepu::multiple<float, int>
over_2d_multi(skepu::Index2D idx, skepu::Region2D<float> r, const skepu::Mat<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			res += r(i, j) * stencil(i + r.oi, j + r.oj);
	return skepu::ret(res, idx.row + idx.col);;
}

skepu::multiple<float, int>
over_3d_multi(skepu::Index3D idx, skepu::Region3D<float> r, skepu::Ten3<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				res += r(i, j, k) * stencil(i + r.oi, j + r.oj, k + r.ok);
	return skepu::ret(res, idx.i + idx.j + idx.k);
}

skepu::multiple<float, int>
over_4d_multi(skepu::Index4D idx, skepu::Region4D<float> r, skepu::Ten4<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				for (int l = -r.ol; l <= r.ol; ++l)
					res += r(i, j, k, l) * stencil(i + r.oi, j + r.oj, k + r.ok, l + r.ol);
	return skepu::ret(res, idx.i + idx.j + idx.k + idx.l);
}


auto conv4_m_l = skepu::MapOverlap([](skepu::Index4D idx, skepu::Region4D<float> r, skepu::Ten4<float> stencil) -> skepu::multiple<float, int> 
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				for (int l = -r.ol; l <= r.ol; ++l)
					res += r(i, j, k, l) * stencil(i + r.oi, j + r.oj, k + r.ok, l + r.ol);
	return skepu::ret(res, idx.i + idx.j + idx.k + idx.l);
});


auto conv = skepu::MapOverlap(over_1d_multi);
auto conv2_m = skepu::MapOverlap(over_2d_multi);
auto conv3_m = skepu::MapOverlap(over_3d_multi);
auto conv4_m = skepu::MapOverlap(over_4d_multi);

// The result of over_1d_multi (its first value) at i, for the elements d of a vector, an edge mode and the padding 0
static float expected_1d(const std::vector<float> &d, long i, skepu::Edge mode, float scale)
{
	const long n = d.size();
	auto at = [&](long k) -> float
	{
		if (k >= 0 && k < n)
			return d[k];
		switch (mode)
		{
		case skepu::Edge::Cyclic:    return d[((k % n) + n) % n];
		case skepu::Edge::Duplicate: return d[std::min<long>(std::max<long>(k, 0), n - 1)];
		default:                     return 0.f;
		}
	};
	return (at(i - 2)*4 + at(i - 1)*2 + at(i) + at(i + 1)*2 + at(i + 2)*4) / scale;
}

TEST_CASE("MapOverlap 1D with variadic return")
{
	const long size{20};
	const skepu::Edge modes[] = {skepu::Edge::Cyclic, skepu::Edge::Duplicate, skepu::Edge::Pad};
	
	conv.setOverlap(2);
	conv.setPad(0);
	
	SECTION("vector")
	{
		skepu::Vector<float> v(size, 10), rv1(size);
		skepu::Vector<int> rv2(size);
		
		skepu::external([&]{
			for (long i = 0;  i < size; ++i)
				v(i) = i;
		}, skepu::write(v));
		
		std::vector<float> d(size);
		for (long i = 0; i < size; ++i)
			d[i] = i;
		
		for (auto mode : modes)
		{
			conv.setEdgeMode(mode);
			conv(rv1, rv2, v, 13);
			
			skepu::external(skepu::read(rv1, rv2), [&]{
				for (long i = 0; i < size; ++i)
				{
					CHECK(rv1(i) == Approx(expected_1d(d, i, mode, 13)));
					CHECK(rv2(i) == i);
				}
			});
		}
	}
	
	SECTION("matrix, row-wise and column-wise")
	{
		skepu::Matrix<float> m(size, size, 10), rm1(size, size);
		skepu::Matrix<int> rm2(size, size);
		
		skepu::external([&]{
			long i = 0;
			for (long y = 0; y < size; ++y)
				for (long x = 0; x < size; ++x)
					m(y, x) = i++;
		}, skepu::write(m));
		
		// (the value of m at (y, x) is y * size + x)
		for (auto overlap : {skepu::Overlap::RowWise, skepu::Overlap::ColWise})
		{
			const bool rowwise = overlap == skepu::Overlap::RowWise;
			conv.setOverlapMode(overlap);
			
			for (auto mode : modes)
			{
				conv.setEdgeMode(mode);
				conv(rm1, rm2, m, 13);
				
				skepu::external(skepu::read(rm1, rm2), [&]{
					for (long y = 0; y < size; ++y)
						for (long x = 0; x < size; ++x)
						{
							// the elements along the direction of the overlap, and the position in it
							std::vector<float> d(size);
							for (long k = 0; k < size; ++k)
								d[k] = rowwise ? y * size + k : k * size + x;
							const long position = rowwise ? x : y;
							
							CHECK(rm1(y, x) == Approx(expected_1d(d, position, mode, 13)));
							CHECK(rm2(y, x) == position);
						}
				});
			}
		}
	}
}


TEST_CASE("MapOverlap 2d with variadic return")
{
	constexpr size_t size_i{33}, size_j{56};
	
	skepu::Matrix<float> m(size_i, size_j, 10), filter(2*1+1, 2*1+1, 1), rm2(size_i, size_j);
	skepu::Matrix<int> ret_m_int(size_i, size_j);
	
	conv2_m.setOverlap(1, 1);
	conv2_m.setEdgeMode(skepu::Edge::Duplicate);
	conv2_m(rm2, ret_m_int, m, filter);
	
	// Every element has nine neighbours of 10 (a duplicated edge element is 10 too); the second result is the sum of the indices
	skepu::external(skepu::read(rm2, ret_m_int), [&]{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
			{
				CHECK(rm2(i, j) == Approx(90.f));
				CHECK(ret_m_int(i, j) == int(i + j));
			}
	});
}


TEST_CASE("MapOverlap 3D with variadic return")
{
	constexpr size_t size_i{13}, size_j{7}, size_k{23};
	
	skepu::Tensor3<float> ten3(size_i, size_j, size_k, "", 1), stencil3(2*1+1, 2*1+1, 2*1+1, "", 1), ret_ten3(size_i, size_j, size_k);
	skepu::Tensor3<int> ret_ten3_int(size_i, size_j, size_k);
	
	conv3_m.setOverlap(1, 1, 1);
	conv3_m.setEdgeMode(skepu::Edge::Duplicate);
	conv3_m(ret_ten3, ret_ten3_int, ten3, stencil3);
	
	skepu::external(skepu::read(ret_ten3, ret_ten3_int), [&]{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
				for (size_t k = 0; k < size_k; ++k)
				{
					CHECK(ret_ten3(i, j, k) == Approx(27.f));
					CHECK(ret_ten3_int(i, j, k) == int(i + j + k));
				}
	});
}


TEST_CASE("MapOverlap 4D with variadic return")
{
	constexpr size_t size_i{3}, size_j{7}, size_k{5}, size_l{13};
	
	skepu::Tensor4<float> ten4(size_i, size_j, size_k, size_l, "", 1), stencil4(2*1+1, 2*1+1, 2*1+1, 2*1+1, "", 1), ret_ten4(size_i, size_j, size_k, size_l);
	skepu::Tensor4<int> ret_ten4_int(size_i, size_j, size_k, size_l);
	
	conv4_m.setOverlap(1, 1, 1, 1);
	conv4_m.setEdgeMode(skepu::Edge::Duplicate);
	conv4_m(ret_ten4, ret_ten4_int, ten4, stencil4);
	
	skepu::external(skepu::read(ret_ten4, ret_ten4_int), [&]{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
				for (size_t k = 0; k < size_k; ++k)
					for (size_t l = 0; l < size_l; ++l)
					{
						CHECK(ret_ten4(i, j, k, l) == Approx(81.f));
						CHECK(ret_ten4_int(i, j, k, l) == int(i + j + k + l));
					}
	});
}
