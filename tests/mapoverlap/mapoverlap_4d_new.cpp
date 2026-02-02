#define SKEPU_ENABLE_EXCEPTIONS
#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/io.hpp>
#include <cmath>
#include <type_traits>
#define PRECISION 1E-3
template <typename T>
void is_similar(T a, T b)
{
	CHECK((
		(std::isnan(a) && std::isnan(b)) ||
		(std::isinf(a) && std::isinf(b)) ||
		(a == Approx(b).epsilon(PRECISION)))
	);
}

int clamp(int const& val, int const& min, int const& max)
{
    if (val < min)
        return min;
    if (val > max)
        return max;
    return val;
}

int overlap_sum(skepu::Region4D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        for (int j = -r.oj; j <= r.oj; ++j)
            for (int k = -r.ok; k <= r.ok; ++k)
                for (int l = -r.ol; l <= r.ol; ++l)
                    val += r(i, j, k, l);
    return val;
}

std::vector<std::vector<std::vector<std::vector<int>>>> generate_expected(skepu::Tensor4<int>& input, std::tuple<int, int, int, int> overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    std::vector<std::vector<std::vector<std::vector<int>>>> res;
    int outsize[4];
    if (edge == skepu::Edge::None)
    {
        outsize[0] = input.size_i() - std::get<0>(overlap) * 2;
        outsize[1] = input.size_j() - std::get<1>(overlap) * 2;
        outsize[2] = input.size_k() - std::get<2>(overlap) * 2;
        outsize[3] = input.size_l() - std::get<3>(overlap) * 2; 
    }
    else
    {
        outsize[0] = input.size_i();
        outsize[1] = input.size_j();
        outsize[2] = input.size_k();
        outsize[3] = input.size_l();
    }
    res.resize(outsize[0]);
    for (int i = 0; i < res.size(); ++i)
    {
        res[i].resize(outsize[1]);
        for (int j = 0; j < res[i].size(); ++j)
        {
            res[i][j].resize(outsize[2]);
            for (int k = 0; k < res[i][j].size(); ++k)
                res[i][j][k].resize(outsize[3]);
        }
    }

    int data[(std::get<0>(overlap) * 2 + 1) * (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1)];
    
    skepu::Region4D<int> region(std::get<0>(overlap), std::get<1>(overlap), std::get<2>(overlap), std::get<3>(overlap),
                                input.size_i(), input.size_j(), input.size_k(), input.size_l(),
                                (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1),
                                (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1),
                                std::get<3>(overlap) * 2 + 1, data);
    
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
                                    int ii = i + oi + ((edge == skepu::Edge::None) ? std::get<0>(overlap) : 0);
                                    int jj = j + oj + ((edge == skepu::Edge::None) ? std::get<1>(overlap) : 0);
                                    int kk = k + ok + ((edge == skepu::Edge::None) ? std::get<2>(overlap) : 0);
                                    int ll = l + ol + ((edge == skepu::Edge::None) ? std::get<3>(overlap) : 0);
                                    if (ii < 0 || ii > input.size_i() - 1 ||
                                        jj < 0 || jj > input.size_j() - 1 ||
                                        kk < 0 || kk > input.size_k() - 1 ||
                                        ll < 0 || ll > input.size_l() - 1)
                                    {
                                        switch (edge)
                                        {
                                            case skepu::Edge::Duplicate:
                                                ii = clamp(ii, 0, input.size_i() - 1);
                                                jj = clamp(jj, 0, input.size_j() - 1);
                                                kk = clamp(kk, 0, input.size_k() - 1);
                                                ll = clamp(ll, 0, input.size_l() - 1);
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
                    res[i][j][k][l] = overlap_sum(region);
                }
    return res;
}

template <typename T>
void compare(skepu::Tensor4<T>& actual, std::vector<std::vector<std::vector<std::vector<T>>>>& expected)
{
    actual.flush();
    for (int i = 0; i < actual.size_i(); ++i)
        for (int j = 0; j < actual.size_j(); ++j)
            for (int k = 0; k < actual.size_k(); ++k)
                for (int l = 0; l < actual.size_l(); ++l)
                {
                    INFO("Checking at (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ", " + std::to_string(l) + "), comparing " + std::to_string(actual(i, j, k, l)) + " and " + std::to_string(expected[i][j][k][l]));
                    if constexpr (std::is_floating_point<T>::value)
                        is_similar(actual(i, j, k, l), expected[i][j][k][l]);
                    else
                        CHECK(actual(i, j, k, l) == expected[i][j][k][l]);
                }
}



auto overlap = skepu::MapOverlap(overlap_sum);

struct SizeRange
{
    int start, end;
};


TEST_CASE("Fundamentals")
{
    // These sizeRanges can be quite chaotic. Doing 1-10 instead of 1-5 can
    // cause some of the backends to execute for several minutes instead of around a couple of seconds.
    // Singular large values don't seem to impact performance that much. For example, 50-50 adds around 10
    // seconds to each backend.
    std::vector<SizeRange> sizeRanges{SizeRange{1, 3}, SizeRange{20, 20}};
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};

    int overlap_radius[4] = {1, 1, 1, 1};

    for (SizeRange const& sizeRange : sizeRanges)
    {
        int start = sizeRange.start, end = sizeRange.end;
        for (int i = start; i <= end; ++i)
            for (int j = start; j <= end; ++j)
                for (int k = start; k <= end; ++k)
                    for (int l = start; l <= end; ++l)
                    {
                        skepu::Tensor4<int> input(i, j, k, l, "Input");
                        input.flush();
                        int val = 0;
                        for (int ii = 0; ii < i; ++ii)
                            for (int jj = 0; jj < j; ++jj)
                                for (int kk = 0; kk < k; ++kk)
                                    for (int ll = 0; ll < l; ++ll)
                                        input(ii, jj, kk, ll) = ++val;

                        skepu::Tensor4<int> actual(input.size_i(), input.size_j(), input.size_k(), input.size_l(), "Output", 0);
                        std::vector<std::vector<std::vector<std::vector<int>>>> expected;
                        overlap.setOverlap(overlap_radius[0], overlap_radius[1], overlap_radius[2], overlap_radius[3]);
                        overlap.setPad(1);
                        
                        for (skepu::Edge const& edge : regularEdges)
                        {
                            INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l) + ", edge = " + skepu::to_string(edge));
                            overlap.setEdgeMode(edge);
                            overlap(actual, input);
                            expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), overlap.getPad());
                            compare(actual, expected);
                        }
                        
                        if (i > overlap_radius[0] * 2 &&
                            j > overlap_radius[1] * 2 &&
                            k > overlap_radius[2] * 2 &&
                            l > overlap_radius[3] * 2)
                        {
                            INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l) + ", edge = None");
                            skepu::Tensor4<int> actual2(input.size_i() - overlap_radius[0] * 2, input.size_j() - overlap_radius[1] * 2,
                                                        input.size_k() - overlap_radius[2] * 2, input.size_l() - overlap_radius[3] * 2, "Output", 0);
                            overlap.setEdgeMode(skepu::Edge::None);
                            overlap(actual2, input);
                            expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
                            compare(actual2, expected);
                        }
                    }
    }
}

TEST_CASE("Overlap ranges")
{
    // Going above 3 on overlapUpperBound has severe impact on performance.
    // Seq goes from around 3s to 50s if overlapUpperBound is set to 4.
    int inputSize = 7;
    int overlapUpperBound = 2;


    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    skepu::Tensor4<int> input(inputSize, inputSize, inputSize, inputSize, "Input", 1);
    input.flush();
    int val = 0;
    for (int ii = 0; ii < inputSize; ++ii)
        for (int jj = 0; jj < inputSize; ++jj)
            for (int kk = 0; kk < inputSize; ++kk)
                for (int ll = 0; ll < inputSize; ++ll)
                    input(ii, jj, kk, ll) = ++val;

    skepu::Tensor4<int> actual(input.size_i(), input.size_j(), input.size_k(), input.size_l(), "Output", 0);
    std::vector<std::vector<std::vector<std::vector<int>>>> expected;
    overlap.setPad(1);

    for (int i = 0; i <= overlapUpperBound; ++i)
        for (int j = 0; j <= overlapUpperBound; ++j)
            for (int k = 0; k <= overlapUpperBound; ++k)
                for (int l = 0; l <= overlapUpperBound; ++l)
                {
                    overlap.setOverlap(i, j, k, l);
                    
                    for (skepu::Edge const& edge : regularEdges)
                    {
                        INFO("overlap = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l) + ", edge = " + skepu::to_string(edge));
                        overlap.setEdgeMode(edge);
                        overlap(actual, input);
                        expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), overlap.getPad());
                        compare(actual, expected);
                    }
                    
                    {
                        INFO("overlap = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l) + ", edge = None");
                        skepu::Tensor4<int> actual2(input.size_i() - i * 2, input.size_j() - j * 2, input.size_k() - k * 2, input.size_l() - l * 2, "Output", 0);
                        overlap.setEdgeMode(skepu::Edge::None);
                        overlap(actual2, input);
                        expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
                        compare(actual2, expected);
                    }
                }
}

TEST_CASE("Invalid sizes")
{
    overlap.setOverlap(1, 1, 1, 1);
    std::tuple<int, int, int, int> overlapOffsetTemp = overlap.getOverlap();
    int overlapOffset[4];
    overlapOffset[0] = std::get<0>(overlapOffsetTemp) * 2;
    overlapOffset[1] = std::get<1>(overlapOffsetTemp) * 2;
    overlapOffset[2] = std::get<2>(overlapOffsetTemp) * 2;
    overlapOffset[3] = std::get<3>(overlapOffsetTemp) * 2;
    skepu::Tensor4<int> input(10, 10, 10, 10), correctSize(input.size_i(), input.size_j(), input.size_k(), input.size_l()),
    small_i(input.size_i()-1, input.size_j(), input.size_k(), input.size_l()), small_j(input.size_i(), input.size_j()-1, input.size_k(), input.size_l()),
    small_k(input.size_i(), input.size_j(), input.size_k()-1, input.size_l()), small_l(input.size_i(), input.size_j(), input.size_k(), input.size_l()-1),
    large_i(input.size_i()+1, input.size_j(), input.size_k(), input.size_l()), large_j(input.size_i(), input.size_j()+1, input.size_k(), input.size_l()),
    large_k(input.size_i(), input.size_j(), input.size_k()+1, input.size_l()), large_l(input.size_i(), input.size_j(), input.size_k(), input.size_l()+1),
    correctSizeNone(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]),
    small_i_none(input.size_i()-overlapOffset[0]-1, input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]),
    small_j_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1]-1, input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]),
    small_k_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]-1, input.size_l()-overlapOffset[3]),
    small_l_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]-1),
    large_i_none(input.size_i()-overlapOffset[0]+1, input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]),
    large_j_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1]+1, input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]),
    large_k_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]+1, input.size_l()-overlapOffset[3]),
    large_l_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]+1);

    skepu::Edge regularEdges[] = {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};

    for (skepu::Edge edge : regularEdges)
    {
        INFO("edge = " + skepu::to_string(edge));
        overlap.setEdgeMode(edge);
        CHECK_THROWS_WITH(overlap(small_i, input), Catch::Matchers::Contains("input/output tensor4 size i mismatch"));
        CHECK_THROWS_WITH(overlap(small_j, input), Catch::Matchers::Contains("input/output tensor4 size j mismatch"));
        CHECK_THROWS_WITH(overlap(small_k, input), Catch::Matchers::Contains("input/output tensor4 size k mismatch"));
        CHECK_THROWS_WITH(overlap(small_l, input), Catch::Matchers::Contains("input/output tensor4 size l mismatch"));
        CHECK_THROWS_WITH(overlap(large_i, input), Catch::Matchers::Contains("input/output tensor4 size i mismatch"));
        CHECK_THROWS_WITH(overlap(large_j, input), Catch::Matchers::Contains("input/output tensor4 size j mismatch"));
        CHECK_THROWS_WITH(overlap(large_k, input), Catch::Matchers::Contains("input/output tensor4 size k mismatch"));
        CHECK_THROWS_WITH(overlap(large_l, input), Catch::Matchers::Contains("input/output tensor4 size l mismatch"));
    }

    {
        INFO("edge = None");
        overlap.setEdgeMode(skepu::Edge::None);
        CHECK_THROWS_WITH(overlap(small_i_none, input), Catch::Matchers::Contains("input/output tensor4 size i mismatch"));
        CHECK_THROWS_WITH(overlap(small_j_none, input), Catch::Matchers::Contains("input/output tensor4 size j mismatch"));
        CHECK_THROWS_WITH(overlap(small_k_none, input), Catch::Matchers::Contains("input/output tensor4 size k mismatch"));
        CHECK_THROWS_WITH(overlap(small_l_none, input), Catch::Matchers::Contains("input/output tensor4 size l mismatch"));
        CHECK_THROWS_WITH(overlap(large_i_none, input), Catch::Matchers::Contains("input/output tensor4 size i mismatch"));
        CHECK_THROWS_WITH(overlap(large_j_none, input), Catch::Matchers::Contains("input/output tensor4 size j mismatch"));
        CHECK_THROWS_WITH(overlap(large_k_none, input), Catch::Matchers::Contains("input/output tensor4 size k mismatch"));
        CHECK_THROWS_WITH(overlap(large_l_none, input), Catch::Matchers::Contains("input/output tensor4 size l mismatch"));
        overlap.setOverlap(6);
        CHECK_THROWS_WITH(overlap(correctSizeNone, input), Catch::Matchers::Contains("input/output tensor4 size i mismatch"));
    }
}

skepu::multiple<int, float> overlap_sum_2(skepu::Region4D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        for (int j = -r.oj; j <= r.oj; ++j)
            for (int k = -r.ok; k <= r.ok; ++k)
                for (int l = -r.ol; l <= r.ol; ++l)
                    val += r(i, j, k, l);
    return skepu::ret(val, val+0.5f);
}

std::pair<std::vector<std::vector<std::vector<std::vector<int>>>>, std::vector<std::vector<std::vector<std::vector<float>>>>> generate_expected_2(skepu::Tensor4<int>& input, std::tuple<int, int, int, int> overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    std::vector<std::vector<std::vector<std::vector<int>>>> res;
    std::vector<std::vector<std::vector<std::vector<float>>>> res2;
    int outsize[4];
    if (edge == skepu::Edge::None)
    {
        outsize[0] = input.size_i() - std::get<0>(overlap) * 2;
        outsize[1] = input.size_j() - std::get<1>(overlap) * 2;
        outsize[2] = input.size_k() - std::get<2>(overlap) * 2;
        outsize[3] = input.size_l() - std::get<3>(overlap) * 2; 
    }
    else
    {
        outsize[0] = input.size_i();
        outsize[1] = input.size_j();
        outsize[2] = input.size_k();
        outsize[3] = input.size_l();
    }
    res.resize(outsize[0]);
    res2.resize(outsize[0]);
    for (int i = 0; i < res.size(); ++i)
    {
        res[i].resize(outsize[1]);
        res2[i].resize(outsize[1]);
        for (int j = 0; j < res[i].size(); ++j)
        {
            res[i][j].resize(outsize[2]);
            res2[i][j].resize(outsize[2]);
            for (int k = 0; k < res[i][j].size(); ++k)
            {
                res[i][j][k].resize(outsize[3]);
                res2[i][j][k].resize(outsize[3]);
            }
        }
    }

    int data[(std::get<0>(overlap) * 2 + 1) * (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1)];
    
    skepu::Region4D<int> region(std::get<0>(overlap), std::get<1>(overlap), std::get<2>(overlap), std::get<3>(overlap),
                                input.size_i(), input.size_j(), input.size_k(), input.size_l(),
                                (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1),
                                (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1),
                                std::get<3>(overlap) * 2 + 1, data);
    
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
                                    int ii = i + oi + ((edge == skepu::Edge::None) ? std::get<0>(overlap) : 0);
                                    int jj = j + oj + ((edge == skepu::Edge::None) ? std::get<1>(overlap) : 0);
                                    int kk = k + ok + ((edge == skepu::Edge::None) ? std::get<2>(overlap) : 0);
                                    int ll = l + ol + ((edge == skepu::Edge::None) ? std::get<3>(overlap) : 0);
                                    if (ii < 0 || ii > input.size_i() - 1 ||
                                        jj < 0 || jj > input.size_j() - 1 ||
                                        kk < 0 || kk > input.size_k() - 1 ||
                                        ll < 0 || ll > input.size_l() - 1)
                                    {
                                        switch (edge)
                                        {
                                            case skepu::Edge::Duplicate:
                                                ii = clamp(ii, 0, input.size_i() - 1);
                                                jj = clamp(jj, 0, input.size_j() - 1);
                                                kk = clamp(kk, 0, input.size_k() - 1);
                                                ll = clamp(ll, 0, input.size_l() - 1);
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
                    std::tuple<int, float> output = overlap_sum_2(region);
                    res[i][j][k][l] = std::get<0>(output);
                    res2[i][j][k][l] = std::get<1>(output);
                }
    return {res, res2};
}

auto overlap2 = skepu::MapOverlap(overlap_sum_2);

TEST_CASE("Multi return")
{
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    int overlap_radius[4] = {1, 1, 1, 1};
    overlap2.setOverlap(overlap_radius[0], overlap_radius[1], overlap_radius[2], overlap_radius[3]);
    overlap2.setPad(1);

    for (int i = 1; i <= 3; ++i)
        for (int j = 1; j <= 3; ++j)
            for (int k = 1; k <= 3; ++k)
                for (int l = 1; l <= 3; ++l)
                {
                    skepu::Tensor4<int> input(i, j, k, l, "Input");
                    input.flush();
                    int val = 0;
                    for (int ii = 0; ii < i; ++ii)
                        for (int jj = 0; jj < j; ++jj)
                            for (int kk = 0; kk < k; ++kk)
                                for (int ll = 0; ll < l; ++ll)
                                    input(ii, jj, kk, ll) = ++val;

                    skepu::Tensor4<int> actual(input.size_i(), input.size_j(), input.size_k(), input.size_l(), "Output", 0);
                    skepu::Tensor4<float> actual2(input.size_i(), input.size_j(), input.size_k(), input.size_l(), "Output", 0);
                        
                    std::pair<std::vector<std::vector<std::vector<std::vector<int>>>>, std::vector<std::vector<std::vector<std::vector<float>>>>> expected;
                    for (skepu::Edge edge : regularEdges)
                    {
                        overlap2.setEdgeMode(edge);
                        overlap2(actual, actual2, input);
                        expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode(), overlap2.getPad());
                        {
                            INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l) + ", edge = " + skepu::to_string(edge) + ", actual");
                            compare(actual, expected.first);
                        }
                        {
                            INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l) + ", edge = " + skepu::to_string(edge) + ", actual2");
                            compare(actual2, expected.second);
                        }
                    }

                    if (i > overlap_radius[0] * 2 &&
                        j > overlap_radius[1] * 2 &&
                        k > overlap_radius[2] * 2 &&
                        l > overlap_radius[3] * 2)
                    {
                        skepu::Tensor4<int> actual3(input.size_i()-overlap_radius[0]*2, input.size_j()-overlap_radius[1]*2, input.size_k()-overlap_radius[2]*2, input.size_l()-overlap_radius[3]*2, "Output", 0);
                        skepu::Tensor4<float> actual4(input.size_i()-overlap_radius[0]*2, input.size_j()-overlap_radius[1]*2, input.size_k()-overlap_radius[2]*2, input.size_l()-overlap_radius[3]*2, "Output", 0);
                        overlap2.setEdgeMode(skepu::Edge::None);
                        overlap2(actual3, actual4, input);
                        expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode());
                        {
                            INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l) + ", edge = None" + ", actual3");
                            compare(actual3, expected.first);
                        }
                        {
                            INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l) + ", edge = None" + ", actual4");
                            compare(actual4, expected.second);
                        }
                    }
                }
}

TEST_CASE("Invalid sizes 2 output tensor4s")
{
    overlap2.setOverlap(1, 1, 1, 1);
    std::tuple<int, int, int, int> overlapOffsetTemp = overlap2.getOverlap();
    int overlapOffset[4];
    overlapOffset[0] = std::get<0>(overlapOffsetTemp) * 2;
    overlapOffset[1] = std::get<1>(overlapOffsetTemp) * 2;
    overlapOffset[2] = std::get<2>(overlapOffsetTemp) * 2;
    overlapOffset[3] = std::get<3>(overlapOffsetTemp) * 2;
    skepu::Tensor4<int> input(10, 10, 10, 10), correctSize(input.size_i(), input.size_j(), input.size_k(), input.size_l()),
    correctSizeNone(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]);
    skepu::Tensor4<float> small_i(input.size_i()-1, input.size_j(), input.size_k(), input.size_l()), small_j(input.size_i(), input.size_j()-1, input.size_k(), input.size_l()),
    small_k(input.size_i(), input.size_j(), input.size_k()-1, input.size_l()), small_l(input.size_i(), input.size_j(), input.size_k(), input.size_l()-1),
    large_i(input.size_i()+1, input.size_j(), input.size_k(), input.size_l()), large_j(input.size_i(), input.size_j()+1, input.size_k(), input.size_l()),
    large_k(input.size_i(), input.size_j(), input.size_k()+1, input.size_l()), large_l(input.size_i(), input.size_j(), input.size_k(), input.size_l()+1),
    small_i_none(input.size_i()-overlapOffset[0]-1, input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]),
    small_j_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1]-1, input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]),
    small_k_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]-1, input.size_l()-overlapOffset[3]),
    small_l_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]-1),
    large_i_none(input.size_i()-overlapOffset[0]+1, input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]),
    large_j_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1]+1, input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]),
    large_k_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]+1, input.size_l()-overlapOffset[3]),
    large_l_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2], input.size_l()-overlapOffset[3]+1);

    skepu::Edge regularEdges[] = {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};

    for (skepu::Edge edge : regularEdges)
    {
        INFO("edge = " + skepu::to_string(edge));
        overlap2.setEdgeMode(edge);
        CHECK_THROWS_WITH(overlap2(correctSize, small_i, input), Catch::Matchers::Contains("invalid output tensor4 size i"));
        CHECK_THROWS_WITH(overlap2(correctSize, small_j, input), Catch::Matchers::Contains("invalid output tensor4 size j"));
        CHECK_THROWS_WITH(overlap2(correctSize, small_k, input), Catch::Matchers::Contains("invalid output tensor4 size k"));
        CHECK_THROWS_WITH(overlap2(correctSize, small_l, input), Catch::Matchers::Contains("invalid output tensor4 size l"));
        CHECK_THROWS_WITH(overlap2(correctSize, large_i, input), Catch::Matchers::Contains("invalid output tensor4 size i"));
        CHECK_THROWS_WITH(overlap2(correctSize, large_j, input), Catch::Matchers::Contains("invalid output tensor4 size j"));
        CHECK_THROWS_WITH(overlap2(correctSize, large_k, input), Catch::Matchers::Contains("invalid output tensor4 size k"));
        CHECK_THROWS_WITH(overlap2(correctSize, large_l, input), Catch::Matchers::Contains("invalid output tensor4 size l"));
    }

    {
        INFO("edge = None");
        overlap2.setEdgeMode(skepu::Edge::None);
        CHECK_THROWS_WITH(overlap2(correctSizeNone, small_i_none, input), Catch::Matchers::Contains("invalid output tensor4 size i"));
        CHECK_THROWS_WITH(overlap2(correctSizeNone, small_j_none, input), Catch::Matchers::Contains("invalid output tensor4 size j"));
        CHECK_THROWS_WITH(overlap2(correctSizeNone, small_k_none, input), Catch::Matchers::Contains("invalid output tensor4 size k"));
        CHECK_THROWS_WITH(overlap2(correctSizeNone, small_l_none, input), Catch::Matchers::Contains("invalid output tensor4 size l"));
        CHECK_THROWS_WITH(overlap2(correctSizeNone, large_i_none, input), Catch::Matchers::Contains("invalid output tensor4 size i"));
        CHECK_THROWS_WITH(overlap2(correctSizeNone, large_j_none, input), Catch::Matchers::Contains("invalid output tensor4 size j"));
        CHECK_THROWS_WITH(overlap2(correctSizeNone, large_k_none, input), Catch::Matchers::Contains("invalid output tensor4 size k"));
        CHECK_THROWS_WITH(overlap2(correctSizeNone, large_l_none, input), Catch::Matchers::Contains("invalid output tensor4 size l"));
    }
}