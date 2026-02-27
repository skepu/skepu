#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/io.hpp>

int clamp(int const& val, int const& min, int const& max)
{
    if (val < min)
        return min;
    if (val > max)
        return max;
    return val;
}

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

TEST_CASE("Expanding MapOverlap 4D")
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