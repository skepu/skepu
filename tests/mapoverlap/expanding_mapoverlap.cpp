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
                                ii = clamp(ii, 0, input.size_i() - 1);
                                jj = clamp(jj, 0, input.size_j() - 1);
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

TEST_CASE("Expanding MapOverlap 2D")
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