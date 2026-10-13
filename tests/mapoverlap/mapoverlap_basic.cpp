// Merged from: mapoverlap_2d.cpp, mapoverlap_2d_expanding.cpp, mapoverlap_4d.cpp, mapoverlap_4d_expanding.cpp, mapoverlap_indexed.cpp, mapoverlap_tuple.cpp
// (one program instead of one for each: the build of a test program has a cost of its own; each part is in a namespace, and its test cases have its name as a prefix)

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <algorithm>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>
#include <iostream>
#include <vector>


// ======== mapoverlap_2d ========

namespace mapoverlap_2d_part
{
float over_2d(skepu::Region2D<int> r, const skepu::Mat<float> stencil)
{
	return r(1,1);
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			res += r(i, j) * stencil(i + r.oi, j + r.oj);
	return res;
}

auto conv2 = skepu::MapOverlap(over_2d);


TEST_CASE("mapoverlap_2d: MapOverlap 2D fundamentals")
{
	const size_t size{25};
	
	conv2.setOverlap(2, 2);
	conv2.setPad(-5);

	skepu::Matrix<int> m(size, size);
	skepu::Matrix<float> rm_a(size-4, size-4), rm_b(size, size), rm_c(size, size), rm_d(size, size);
	skepu::external(
		skepu::read(m),
		[&]
		{
			int i = 0;
			for(size_t y = 0; y < size; ++y)
				for(size_t x = 0; x < size; ++x)
					m(y, x) = i++;
		},
		skepu::write(m)
	);

	skepu::Matrix<float> filter(2*1+1, 2*1+1, 1);
	
	// The user function gives the element at (+1, +1) from the center (the stencil in it is not reached).
	// The value of m at (y, x) is y * size + x.
	auto value = [&](long y, long x) { return float(y * long(size) + x); };
	auto clamp = [&](long v) { return std::min<long>(std::max<long>(v, 0), size - 1); };
	
	conv2.setEdgeMode(skepu::Edge::None);
	conv2(rm_a, m, filter);
	
	conv2.setEdgeMode(skepu::Edge::Cyclic);
	conv2(rm_b, m, filter);
	
	conv2.setEdgeMode(skepu::Edge::Duplicate);
	conv2(rm_c, m, filter);
	
	conv2.setEdgeMode(skepu::Edge::Pad);
	conv2(rm_d, m, filter);
	
	skepu::external(
		skepu::read(rm_a, rm_b, rm_c, rm_d),
		[&]
		{
			// None: the output has the elements that have a full neighbourhood, so its element (y, x) is at (y + 2, x + 2) of m
			for (long y = 0; y < long(size) - 4; ++y)
				for (long x = 0; x < long(size) - 4; ++x)
					CHECK(rm_a(y, x) == value(y + 3, x + 3));
			
			for (long y = 0; y < long(size); ++y)
				for (long x = 0; x < long(size); ++x)
				{
					CHECK(rm_b(y, x) == value((y + 1) % size, (x + 1) % size));
					CHECK(rm_c(y, x) == value(clamp(y + 1), clamp(x + 1)));
					CHECK(rm_d(y, x) == ((y + 1 >= long(size) || x + 1 >= long(size)) ? -5.f : value(y + 1, x + 1)));
				}
		});
}
}


// ======== mapoverlap_2d_expanding ========

namespace mapoverlap_2d_expanding_part
{
int overlap2D(skepu::Region2D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        for (int j = -r.oj; j <= r.oj; ++j)
            val += r(i, j);
    return val;
}

std::vector<std::vector<int>> generateExpected(skepu::Matrix<int>& input, int out_size_i, int out_size_j, std::tuple<int, int> overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    std::vector<std::vector<int>> res;
    res.resize(out_size_i);
    for (int i = 0; i < res.size(); ++i)
        res[i].resize(out_size_j);


    const size_t elements_in_region = (std::get<0>(overlap) * 2 + 1) * (std::get<1>(overlap) * 2 + 1);
    int data[elements_in_region];

    skepu::Region2D<int> region(std::get<0>(overlap), std::get<1>(overlap),
                                input.size_i(), input.size_j(),
                                (std::get<1>(overlap) * 2 + 1), &data[elements_in_region/2]);
    

    skepu::Index2D offset = {0, 0};
    if (edge == skepu::Edge::None)
    {
        offset.row = std::get<0>(overlap);
        offset.col = std::get<1>(overlap);
    }
    else
    {
        offset.row = (input.size_i() - out_size_i) / 2;
        offset.col = (input.size_j() - out_size_j) / 2;
    }

    for (int i = 0; i < res.size(); ++i)
        for (int j = 0; j < res[i].size(); ++j)
        {
            int pos = 0;
            for (int oi = -std::get<0>(overlap); oi <= std::get<0>(overlap); ++oi)
                for (int oj = -std::get<1>(overlap); oj <= std::get<1>(overlap); ++oj)             
                {
                    int ii = i + oi + offset.row;
                    int jj = j + oj + offset.col;
                    if (ii < 0 || ii > input.size_i() - 1 ||
                        jj < 0 || jj > input.size_j() - 1)
                    {
                        switch (edge)
                        {
                            case skepu::Edge::Duplicate:
                                ii = std::clamp(ii, 0, static_cast<int>(input.size_i() - 1));
                                jj = std::clamp(jj, 0, static_cast<int>(input.size_j() - 1));
                                break;
                            case skepu::Edge::Cyclic:
                                ii = (ii + input.size_i()) % input.size_i();
                                jj = (jj + input.size_j()) % input.size_j();
                                break;
                            case skepu::Edge::Pad:
                                ii = -1;
                                break;
                            case skepu::Edge::None:
                                throw;
                                break;
                        }
                    }
                    if (ii == -1)
                        data[pos++] = pad;
                    else
                        data[pos++] = input(ii, jj);
                }
            res[i][j] = overlap2D(region);
        }
    return res;
}

void compare(skepu::Matrix<int>& actual, std::vector<std::vector<int>>& expected)
{
    actual.flush();
    for (int i = 0; i < actual.size_i(); ++i)
        for (int j = 0; j < actual.size_j(); ++j)
        {
            INFO("Checking at (" + std::to_string(i) + ", " + std::to_string(j) + "), comparing " + std::to_string(actual(i, j)) + " and " + std::to_string(expected[i][j]));
            CHECK(actual(i, j) == expected[i][j]);
        }
}

auto skel = skepu::MapOverlap(overlap2D);

constexpr int size_upper_bound = 10;
constexpr int out_size_offset = 2;
constexpr int overlap = 1;
constexpr int pad = 10;

TEST_CASE("mapoverlap_2d_expanding: Expanding MapOverlap 2D")
{
    constexpr std::array<skepu::Edge, 3> regular_edges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    skel.setOverlap(overlap, overlap);
    skel.setPad(pad);

    for (int in_size_i = 1; in_size_i <= size_upper_bound; ++in_size_i)
        for (int in_size_j = 1; in_size_j <= size_upper_bound; ++in_size_j)
            for (int out_size_i = in_size_i; out_size_i <= size_upper_bound; out_size_i += 2)
                for (int out_size_j = in_size_j; out_size_j <= size_upper_bound; out_size_j += 2)
                {
                    skepu::Matrix<int> input(in_size_i, in_size_j),
                                       actual(out_size_i + out_size_offset, out_size_j + out_size_offset);
                    skepu::external(skepu::read(input), [&]{
                        int val = 0;
                        for (int i = 0; i < input.size_i(); ++i)
                            for (int j = 0; j < input.size_j(); ++j)
                                input(i, j) = ++val;
                    }, skepu::write(input));


                    for (skepu::Edge edge : regular_edges)
                    {
                        INFO("input = " + std::to_string(in_size_i) + "x" + std::to_string(in_size_j) +
                             ", output = " + std::to_string(actual.size_i()) + "x" + std::to_string(actual.size_j())  + ", edge = " + skepu::to_string(edge));
                        skel.setEdgeMode(edge);
                        skel(actual, input);
                        std::vector<std::vector<int>> expected = generateExpected(input, actual.size_i(), actual.size_j(), skel.getOverlap(), skel.getEdgeMode(), skel.getPad());
                        compare(actual, expected);
                    }

                }
}
}


// ======== mapoverlap_4d ========

namespace mapoverlap_4d_part
{
float over_4d(skepu::Region4D<int> r, skepu::Ten4<float> stencil)
{
	return r(1,0,0,0);
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				for (int l = -r.ol; l <= r.ol; ++l)
					res +=
						r(i, j, k, l) * stencil(i + r.oi, j + r.oj, k + r.ok, l + r.ol);
	return res;
}

auto conv4 = skepu::MapOverlap(over_4d);


TEST_CASE("mapoverlap_4d: MapOverlap 4D fundamentals")
{
	const size_t size{20};
	
	conv4.setOverlap(1, 1, 1, 1);
	conv4.setPad(-5);

	skepu::Tensor4<int> ten4(size, size, size, size);
	skepu::Tensor4<float> ret_ten4(size, size, size, size);
	skepu::Tensor4<float> stencil4(2*1+1, 2*1+1, 2*1+1, 2*1+1, "", 1);

	skepu::external(
		[&]
		{
			auto i(0);
			for(auto & e : ten4)
				e = i++;
		},
		skepu::write(ten4));

	// The user function gives the element at +1 in the first dimension. The value of ten4 at (a, b, c, d) is ((a * size + b) * size + c) * size + d.
	auto value = [&](long a, long b, long c, long d) { return float(((a * long(size) + b) * long(size) + c) * long(size) + d); };
	auto clamp = [&](long v) { return std::min<long>(std::max<long>(v, 0), size - 1); };
	
	auto checkAll = [&](auto expected)
	{
		skepu::external(skepu::read(ret_ten4), [&]
		{
			for (long a = 0; a < long(size); ++a)
				for (long b = 0; b < long(size); ++b)
					for (long c = 0; c < long(size); ++c)
						for (long d = 0; d < long(size); ++d)
							CHECK(ret_ten4(a, b, c, d) == expected(a, b, c, d));
		});
	};

	conv4.setEdgeMode(skepu::Edge::Cyclic);
	conv4(ret_ten4, ten4, stencil4);
	checkAll([&](long a, long b, long c, long d) { return value((a + 1) % size, b, c, d); });

	conv4.setEdgeMode(skepu::Edge::Duplicate);
	conv4(ret_ten4, ten4, stencil4);
	checkAll([&](long a, long b, long c, long d) { return value(clamp(a + 1), b, c, d); });

	conv4.setEdgeMode(skepu::Edge::Pad);
	conv4(ret_ten4, ten4, stencil4);
	checkAll([&](long a, long b, long c, long d) { return a + 1 >= long(size) ? -5.f : value(a + 1, b, c, d); });
}
}


// ======== mapoverlap_4d_expanding ========

namespace mapoverlap_4d_expanding_part
{
int overlap4D(skepu::Region4D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        for (int j = -r.oj; j <= r.oj; ++j)
            for (int k = -r.ok; k <= r.ok; ++k)
                for (int l = -r.ol; l <= r.ol; ++l)
                    val += r(i, j, k, l);
    return val;
}

std::vector<std::vector<std::vector<std::vector<int>>>> generateExpected(skepu::Tensor4<int>& input, int out_size_i, int out_size_j, int out_size_k, int out_size_l,
                                                                         std::tuple<int, int, int, int> overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    std::vector<std::vector<std::vector<std::vector<int>>>> res;
    res.resize(out_size_i);
    for (int i = 0; i < res.size(); ++i)
    {
        res[i].resize(out_size_j);
        for (int j = 0; j < res[i].size(); ++j)
        {
            res[i][j].resize(out_size_k);
            for (int k = 0; k < res[i][j].size(); ++k)
                res[i][j][k].resize(out_size_l);
        }
    }


    const size_t elements_in_region = (std::get<0>(overlap) * 2 + 1) * (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1);
    int data[elements_in_region];

    skepu::Region4D<int> region(std::get<0>(overlap), std::get<1>(overlap), std::get<2>(overlap), std::get<3>(overlap),
                                input.size_i(), input.size_j(), input.size_k(), input.size_l(),
                                (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1),
                                (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1),
                                std::get<3>(overlap) * 2 + 1, &data[elements_in_region/2]);
    

    skepu::Index4D offset = {0, 0, 0, 0};
    if (edge == skepu::Edge::None)
    {
        offset.i = std::get<0>(overlap);
        offset.j = std::get<1>(overlap);
        offset.k = std::get<2>(overlap);
        offset.l = std::get<3>(overlap);
    }
    else
    {
        offset.i = (input.size_i() - out_size_i) / 2;
        offset.j = (input.size_j() - out_size_j) / 2;
        offset.k = (input.size_k() - out_size_k) / 2;
        offset.l = (input.size_l() - out_size_l) / 2;
    }

    for (int i = 0; i < res.size(); ++i)
        for (int j = 0; j < res[i].size(); ++j)
            for (int k = 0; k < res[i][j].size(); ++k)
                for (int l = 0; l < res[i][j][k].size(); ++l)
                {
                    int pos = 0;
                    for (int oi = -std::get<0>(overlap); oi <= std::get<0>(overlap); ++oi)
                        for (int oj = -std::get<1>(overlap); oj <= std::get<1>(overlap); ++oj)
                            for (int ok = -std::get<2>(overlap); ok <= std::get<2>(overlap); ++ok)
                                for (int ol = -std::get<3>(overlap); ol <= std::get<3>(overlap); ++ol)
                                {
                                    int ii = i + oi + offset.i;
                                    int jj = j + oj + offset.j;
                                    int kk = k + ok + offset.k;
                                    int ll = l + ol + offset.l;
                                    if (ii < 0 || ii > input.size_i() - 1 ||
                                        jj < 0 || jj > input.size_j() - 1 ||
                                        kk < 0 || kk > input.size_k() - 1 ||
                                        ll < 0 || ll > input.size_l() - 1)
                                    {
                                        switch (edge)
                                        {
                                            case skepu::Edge::Duplicate:
                                                ii = std::clamp(ii, 0, static_cast<int>(input.size_i() - 1));
                                                jj = std::clamp(jj, 0, static_cast<int>(input.size_j() - 1));
                                                kk = std::clamp(kk, 0, static_cast<int>(input.size_k() - 1));
                                                ll = std::clamp(ll, 0, static_cast<int>(input.size_l() - 1));
                                                break;
                                            case skepu::Edge::Cyclic:
                                                ii = (ii + input.size_i()) % input.size_i();
                                                jj = (jj + input.size_j()) % input.size_j();
                                                kk = (kk + input.size_k()) % input.size_k();
                                                ll = (ll + input.size_l()) % input.size_l();
                                                break;
                                            case skepu::Edge::Pad:
                                                ii = -1;
                                                break;
                                            case skepu::Edge::None:
                                                throw;
                                                break;
                                        }
                                    }
                                    if (ii == -1)
                                        data[pos++] = pad;
                                    else
                                        data[pos++] = input(ii, jj, kk, ll);
                                }
                    res[i][j][k][l] = overlap4D(region);
                }
    return res;
}

void compare(skepu::Tensor4<int>& actual, std::vector<std::vector<std::vector<std::vector<int>>>>& expected)
{
    actual.flush();
    for (int i = 0; i < actual.size_i(); ++i)
        for (int j = 0; j < actual.size_j(); ++j)
            for (int k = 0; k < actual.size_k(); ++k)
                for (int l = 0; l < actual.size_l(); ++l)
                {
                    INFO("Checking at (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ", " + std::to_string(l) + "), comparing " +
                         std::to_string(actual(i, j, k, l)) + " and " + std::to_string(expected[i][j][k][l]));
                    CHECK(actual(i, j, k, l) == expected[i][j][k][l]);
                }
}

auto skel = skepu::MapOverlap(overlap4D);

constexpr int size_upper_bound = 4;
constexpr int out_size_offset = 2;
constexpr int overlap = 1;
constexpr int pad = 10;

TEST_CASE("mapoverlap_4d_expanding: Expanding MapOverlap 4D")
{
    constexpr std::array<skepu::Edge, 3> regular_edges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    skel.setOverlap(overlap, overlap, overlap, overlap);
    skel.setPad(pad);

    for (int in_size_i = 1; in_size_i <= size_upper_bound; ++in_size_i)
        for (int in_size_j = 1; in_size_j <= size_upper_bound; ++in_size_j)
            for (int in_size_k = 1; in_size_k <= size_upper_bound; ++in_size_k)
                for (int in_size_l = 1; in_size_l <= size_upper_bound; ++in_size_l)
                    for (int out_size_i = in_size_i; out_size_i <= size_upper_bound; out_size_i += 2)
                        for (int out_size_j = in_size_j; out_size_j <= size_upper_bound; out_size_j += 2)
                            for (int out_size_k = in_size_k; out_size_k <= size_upper_bound; out_size_k += 2)
                                for (int out_size_l = in_size_l; out_size_l <= size_upper_bound; out_size_l += 2)
                                {
                                    skepu::Tensor4<int> input(in_size_i, in_size_j, in_size_k, in_size_l),
                                                    actual(out_size_i + out_size_offset, out_size_j + out_size_offset, out_size_k + out_size_offset, out_size_l + out_size_offset);
                                    skepu::external(skepu::read(input), [&]{
                                        int val = 0;
                                        for (int i = 0; i < input.size_i(); ++i)
                                            for (int j = 0; j < input.size_j(); ++j)
                                                for (int k = 0; k < input.size_k(); ++k)
                                                    for (int l = 0; l < input.size_l(); ++l)
                                                        input(i, j, k, l) = ++val;
                                    }, skepu::write(input));


                                    for (skepu::Edge edge : regular_edges)
                                    {
                                        INFO("input = " + std::to_string(in_size_i) + "x" + std::to_string(in_size_j) + "x" + std::to_string(in_size_k) + "x" + std::to_string(in_size_l) +
                                            ", output = " + std::to_string(actual.size_i()) + "x" + std::to_string(actual.size_j()) + "x" + std::to_string(actual.size_k()) + "x" + std::to_string(actual.size_l()) +
                                            ", edge = " + skepu::to_string(edge));
                                        skel.setEdgeMode(edge);
                                        skel(actual, input);
                                        std::vector<std::vector<std::vector<std::vector<int>>>> expected = generateExpected(input, actual.size_i(), actual.size_j(), actual.size_k(), actual.size_l(), skel.getOverlap(), skel.getEdgeMode(), skel.getPad());
                                        compare(actual, expected);
                                    }

                                }
}
}


// ======== mapoverlap_indexed ========

namespace mapoverlap_indexed_part
{
const int size = 10;

int overlap_1d(skepu::Index1D index, skepu::Region1D<int> r)
{
    return index.i;
}

int overlap_2d(skepu::Index2D index, skepu::Region2D<int> r, int size)
{
    return index.row * size + index.col;
}

int overlap_3d(skepu::Index3D index, skepu::Region3D<int> r, int size)
{
    return index.i * size * size + index.j * size + index.k;
}

int overlap_4d(skepu::Index4D index, skepu::Region4D<int> r, int size)
{
    return index.i * size * size * size + index.j * size * size + index.k * size + index.l;
}

auto skel_1d = skepu::MapOverlap(overlap_1d);
auto skel_2d = skepu::MapOverlap(overlap_2d);
auto skel_3d = skepu::MapOverlap(overlap_3d);
auto skel_4d = skepu::MapOverlap(overlap_4d);

TEST_CASE("mapoverlap_indexed: MapOverlap 1D")
{
    skepu::Vector<int> input(size), output(size);
    skel_1d.setOverlap(1); // The wrap memory used for OpenCL will have a size of 0 otherwise
    skel_1d.setEdgeMode(skepu::Edge::Duplicate);
    skel_1d(output, input);
    skepu::external(skepu::read(output), [&]{
        int expected = 0;
        for (int i = 0; i < size; ++i)
            {
                INFO("MapOverlap1D, index: " + std::to_string(i));
                CHECK(output(i) == expected++);
            }
    });

    skel_1d.setEdgeMode(skepu::Edge::None);
    skepu::Vector<int> output_none(size - 2);
    skel_1d(output_none, input);
    skepu::external(skepu::read(output_none), [&]{
        int expected = 0;
        for (int i = 0; i < output_none.size(); ++i)
            {
                INFO("MapOverlap1D none, index: " + std::to_string(i));
                CHECK(output_none(i) == expected++);
            }
    });
}

TEST_CASE("mapoverlap_indexed: MapOverlap 1D RowWise")
{
    skepu::Matrix<int> input(size, size), output(size, size);
    skel_1d.setOverlap(1);
    skel_1d.setEdgeMode(skepu::Edge::Duplicate);
    skel_1d.setOverlapMode(skepu::Overlap::RowWise);
    skel_1d(output, input);
    skepu::external(skepu::read(output), [&]{
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
            {
                INFO("MapOverlap2D RoWwise, index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output(i, j) == j);
            }
    });

    skel_1d.setOverlap(1);
    skel_1d.setEdgeMode(skepu::Edge::None);
    skepu::Matrix<int> output_none(size, size - 2);
    skel_1d(output_none, input);
    skepu::external(skepu::read(output_none), [&]{
        for (int i = 0; i < output_none.size_i(); ++i)
            for (int j = 0; j < output_none.size_j(); ++j)
            {
                INFO("MapOverlap2D RoWwise none , index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output_none(i, j) == j);
            }
    });
}

TEST_CASE("mapoverlap_indexed: MapOverlap 1D ColWise")
{
    skepu::Matrix<int> input(size, size), output(size, size);
    skel_1d.setOverlap(1);
    skel_1d.setEdgeMode(skepu::Edge::Duplicate);
    skel_1d.setOverlapMode(skepu::Overlap::ColWise);
    skel_1d(output, input);
    skepu::external(skepu::read(output), [&]{
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
            {
                INFO("MapOverlap2D ColWise, index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output(i, j) == i);
            }
    });

    skel_1d.setOverlap(1);
    skel_1d.setEdgeMode(skepu::Edge::None);
    skepu::Matrix<int> output_none(size - 2, size);
    skel_1d(output_none, input);
    skepu::external(skepu::read(output_none), [&]{
        for (int i = 0; i < output_none.size_i(); ++i)
            for (int j = 0; j < output_none.size_j(); ++j)
            {
                INFO("MapOverlap1D ColWise none , index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output_none(i, j) == i);
            }
    });
}

TEST_CASE("mapoverlap_indexed: MapOverlap 2D")
{
    skepu::Matrix<int> input(size, size), output(size, size);
    skel_2d.setOverlap(0, 0);
    skel_2d.setEdgeMode(skepu::Edge::Duplicate);
    skel_2d(output, input, size);
    skepu::external(skepu::read(output), [&]{
        int expected = 0;
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
            {
                INFO("MapOverlap2D, index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output(i, j) == expected++);
            }
    });

    skel_2d.setOverlap(1, 1);
    skel_2d.setEdgeMode(skepu::Edge::None);
    skepu::Matrix<int> output_none(size - 2, size - 2);
    skel_2d(output_none, input, size - 2);
    skepu::external(skepu::read(output_none), [&]{
        int expected = 0;
        for (int i = 0; i < output_none.size_i(); ++i)
            for (int j = 0; j < output_none.size_j(); ++j)
            {
                INFO("MapOverlap2D none , index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output_none(i, j) == expected++);
            }
    });
}

TEST_CASE("mapoverlap_indexed: MapOverlap 3D")
{
    skepu::Tensor3<int> input(size, size, size), output(size, size, size);
    skel_3d.setOverlap(0, 0, 0);
    skel_3d.setEdgeMode(skepu::Edge::Duplicate);
    skel_3d(output, input, size);
    skepu::external(skepu::read(output), [&]{
        int expected = 0;
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                for (int k = 0; k < size; ++k)
                {
                    INFO("MapOverlap3D, index: (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ")");
                    CHECK(output(i, j, k) == expected++);
                }
    });

    skel_3d.setOverlap(1, 1, 1);
    skel_3d.setEdgeMode(skepu::Edge::None);
    skepu::Tensor3<int> output_none(size - 2, size - 2, size - 2);
    skel_3d(output_none, input, size - 2);
    skepu::external(skepu::read(output_none), [&]{
        int expected = 0;
        for (int i = 0; i < output_none.size_i(); ++i)
            for (int j = 0; j < output_none.size_j(); ++j)
                for (int k = 0; k < output_none.size_k(); ++k)
                {
                    INFO("MapOverlap3D none, index: (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ")");
                    CHECK(output_none(i, j, k) == expected++);
                }
    });
}

TEST_CASE("mapoverlap_indexed: MapOverlap 4D")
{
    skepu::Tensor4<int> input(size, size, size, size), output(size, size, size, size);
    skel_4d.setOverlap(0, 0, 0, 0);
    skel_4d.setEdgeMode(skepu::Edge::Duplicate);
    skel_4d(output, input, size);
    skepu::external(skepu::read(output), [&]{
        int expected = 0;
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                for (int k = 0; k < size; ++k)
                    for (int l = 0; l < size; ++l)
                    {
                        INFO("MapOverlap4D, index: (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ", " + std::to_string(l) + ")");
                        CHECK(output(i, j, k, l) == expected++);
                    }
    });

    skel_4d.setOverlap(1, 1, 1, 1);
    skel_4d.setEdgeMode(skepu::Edge::None);
    skepu::Tensor4<int> output_none(size - 2, size - 2, size - 2, size - 2);
    skel_4d(output_none, input, size - 2);
    skepu::external(skepu::read(output_none), [&]{
        int expected = 0;
        for (int i = 0; i < output_none.size_i(); ++i)
            for (int j = 0; j < output_none.size_j(); ++j)
                for (int k = 0; k < output_none.size_k(); ++k)
                    for (int l = 0; l < output_none.size_l(); ++l)
                    {
                        INFO("MapOverlap4D none, index: (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ", " + std::to_string(l) + ")");
                        CHECK(output_none(i, j, k, l) == expected++);
                    }
    });
}
}


// ======== mapoverlap_tuple ========

namespace mapoverlap_tuple_part
{
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

TEST_CASE("mapoverlap_tuple: MapOverlap 1D with variadic return")
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


TEST_CASE("mapoverlap_tuple: MapOverlap 2d with variadic return")
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


TEST_CASE("mapoverlap_tuple: MapOverlap 3D with variadic return")
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


TEST_CASE("mapoverlap_tuple: MapOverlap 4D with variadic return")
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
}
