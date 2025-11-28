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

std::vector<int> generate_expected(skepu::Vector<int>& input, int overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    std::vector<int> res;
    if (edge == skepu::Edge::None)
        res.resize(input.size() - overlap * 2);
    else
        res.resize(input.size());
    
    for (int i = 0; i < res.size(); ++i)
    {
        int sum = 0;
        for (int o = -overlap; o <= overlap; ++o)
        {
            int pos = i + o + ((edge == skepu::Edge::None) ? overlap : 0);
            if (pos < 0 or pos > input.size() - 1)
            {
                switch (edge)
                {
                    case skepu::Edge::Duplicate:
                        pos = clamp(pos, 0, input.size() - 1);
                        break;
                    case skepu::Edge::Cyclic:
                        pos = (pos + input.size()) % input.size();
                        break;
                    case skepu::Edge::Pad:
                        pos = -1;
                        break;
                    case skepu::Edge::None:
                        throw;
                        break;
                }
            }
            if (pos == -1)
                sum += pad;
            else
                sum += input[pos];
        }
        res[i] = sum;
    }
    return res;
}

void compare(skepu::Vector<int>& actual, std::vector<int>& expected)
{
    actual.flush();
    for (int i = 0; i < actual.size(); ++i)
    {
        CHECK(actual[i] == expected[i]);
    }
}

int overlap_sum(skepu::Region1D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        val += r(i);
    return val;
}

auto overlap = skepu::MapOverlap(overlap_sum);

TEST_CASE("Fundamentals")
{
    int size = 100;

    skepu::Vector<int> input{1, 2, 3, 4, 5};
    skepu::Vector<int> actual(input.size(), 0);
    skepu::Vector<int> actual2(input.size() - 2, 0);
    std::vector<int> expected;

    
    overlap.setOverlap(1);
    overlap.setEdgeMode(skepu::Edge::Duplicate);
    overlap(actual, input);
    expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
    compare(actual, expected);


    overlap.setEdgeMode(skepu::Edge::Cyclic);
    overlap(actual, input);
    expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
    compare(actual, expected);
    
    overlap.setEdgeMode(skepu::Edge::Pad);
    int pad = 10;
    overlap.setPad(pad);
    overlap(actual, input);
    expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), pad);
    compare(actual, expected);

    overlap.setEdgeMode(skepu::Edge::None);
    overlap(actual2, input);
    expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
    compare(actual2, expected);
}
