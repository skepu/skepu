#define SKEPU_ENABLE_EXCEPTIONS
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

int overlap_sum(skepu::Region1D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        val += r(i);
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
    
    int data[overlap*3];
    for (int i = 0; i < res.size(); ++i)
    {
        int sum = 0;
        int dataIndex = 0;
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
                data[dataIndex++] = pad;
            else
                data[dataIndex++] = input[pos];
        }
        skepu::Region1D<int> r{overlap, 1, &data[overlap]};
        res[i] = overlap_sum(r);
    }
    return res;
}

void compare(skepu::Vector<int>& actual, std::vector<int>& expected)
{
    actual.flush();
    for (int i = 0; i < actual.size(); ++i)
    {
        INFO("i = " + std::to_string(i));
        CHECK(actual(i) == expected[i]);
    }
}

auto overlap = skepu::MapOverlap(overlap_sum);

TEST_CASE("Fundamentals")
{
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    int pad = 10;
    int overlapRadius = 1;
    overlap.setOverlap(overlapRadius);
    overlap.setPad(pad);

    for (int size = 2; size <= 100; ++size)
    {
        skepu::Vector<int> input(size, 1);
        skepu::Vector<int> actual(input.size(), 0);
        std::vector<int> expected;
        skepu::external(skepu::read(input), [&]{
        for (int i = 0; i < input.size(); ++i)
        {
            input(i) = i;
        }
        }, skepu::write(input));

        // TODO: Add overlaps larger than input size.

        for (skepu::Edge edge : regularEdges)
        {
            INFO("size = " + std::to_string(size) + ", edge = " + skepu::to_string(edge));
            overlap.setEdgeMode(edge);
            overlap(actual, input);
            expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), pad);
            compare(actual, expected);
        }

        if (size > overlapRadius * 2)
        {
            INFO("size = " + std::to_string(size) + ", edge = None");
            skepu::Vector<int> actual2(input.size() - 2, 0);
            overlap.setEdgeMode(skepu::Edge::None);
            overlap(actual2, input);
            expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
            compare(actual2, expected);
        }
    }
}
/*
std::ostream& operator<<(std::ostream& os, std::vector<int> v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i < v.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}


// This does not work because the start and end get larger than the vector itself
// Discuss with August.

TEST_CASE("Overlap radius")
{
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate};
    int pad = 10;
    

    for (int overlapRadius = 6; overlapRadius <= 6; ++overlapRadius)
    {
        overlap.setOverlap(overlapRadius);
        skepu::Vector<int> input(10, 1);
        skepu::Vector<int> actual(input.size(), 0);
        std::vector<int> expected;
        skepu::external(skepu::read(input), [&]{
        for (int i = 0; i < input.size(); ++i)
        {
            input(i) = i;
        }
        }, skepu::write(input));

        for (skepu::Edge edge : regularEdges)
        {
            INFO("overlapRadius = " + std::to_string(overlapRadius) + ", edge = " + skepu::to_string(edge));
            overlap.setEdgeMode(skepu::Edge::Duplicate);
            overlap(actual, input);
            expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), pad);
            if (overlapRadius == 6)
            {
                std::cout << actual << "\n";
                std::cout << expected << "\n";
            }
            compare(actual, expected);
        }

        if (10 > overlapRadius * 2)
        {
            INFO("overlapRadius = " + std::to_string(overlapRadius) + ", edge = None");
            skepu::Vector<int> actual2(input.size() - overlapRadius * 2, 0);
            overlap.setEdgeMode(skepu::Edge::None);
            overlap(actual2, input);
            expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
            compare(actual2, expected);
        }
    }
}*/

TEST_CASE("Invalid sizes")
{
    overlap.setOverlap(1);
    int overlapOffset = overlap.getOverlap() * 2;
    skepu::Vector<int> input(10), correctSize(10), small(input.size()-1), large(input.size()+1),
    correctSizeNone(input.size()-overlapOffset), smallNone(input.size()-overlapOffset-1), largeNone(input.size()-overlapOffset+1);

    skepu::Edge regularEdges[] = {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};

    for (skepu::Edge edge : regularEdges)
    {
        INFO("edge = " + skepu::to_string(edge));
        overlap.setEdgeMode(edge);
        CHECK_THROWS_WITH(overlap(small, input), Catch::Matchers::Contains("input/output vector size mismatch"));
        CHECK_THROWS_WITH(overlap(large, input), Catch::Matchers::Contains("input/output vector size mismatch"));
    }

    INFO("edge = None");
    overlap.setEdgeMode(skepu::Edge::None);
    CHECK_THROWS_WITH(overlap(smallNone, input), Catch::Matchers::Contains("input/output vector size mismatch"));
    CHECK_THROWS_WITH(overlap(largeNone, input), Catch::Matchers::Contains("input/output vector size mismatch"));
    overlap.setOverlap(6);
    CHECK_THROWS_WITH(overlap(correctSizeNone, input), Catch::Matchers::Contains("input/output vector size mismatch"));
}

skepu::multiple<int, int> overlap_sum_2(skepu::Region1D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        val += r(i);
    return skepu::ret(val, val+1);
}

std::vector<std::vector<int>> generate_expected_2(skepu::Vector<int>& input, int overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    std::vector<int> res, res2;
    if (edge == skepu::Edge::None)
        res.resize(input.size() - overlap * 2);
    else
        res.resize(input.size());
    res2.resize(res.size());
    
    int data[overlap*3];
    for (int i = 0; i < res.size(); ++i)
    {
        int sum = 0;
        int dataIndex = 0;
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
                data[dataIndex++] = pad;
            else
                data[dataIndex++] = input[pos];
        }
        skepu::Region1D<int> r{overlap, 1, &data[overlap]};
        std::tuple<int, int> output = overlap_sum_2(r);
        res[i] = std::get<0>(output);
        res2[i] = std::get<1>(output);
    }
    return {res, res2};
}

auto overlap2 = skepu::MapOverlap(overlap_sum_2);

TEST_CASE("Multi return")
{
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    int pad = 10;
    int overlapRadius = 1;
    overlap2.setOverlap(overlapRadius);
    overlap2.setPad(pad);

    for (int size = 2; size <= 100; ++size)
    {
        skepu::Vector<int> input(size, 1);
        skepu::Vector<int> actual(input.size(), 0), actual2(input.size(), 0);
        std::vector<std::vector<int>> expected;
        skepu::external(skepu::read(input), [&]{
        for (int i = 0; i < input.size(); ++i)
        {
            input(i) = i;
        }
        }, skepu::write(input));

        // TODO: Add overlaps larger than input size.

        for (skepu::Edge edge : regularEdges)
        {
            overlap2.setEdgeMode(edge);
            overlap2(actual, actual2, input);
            expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode(), pad);
            {
                INFO("size = " + std::to_string(size) + ", edge = " + skepu::to_string(edge) + ", actual");
                compare(actual, expected[0]);
            }
            {
                INFO("size = " + std::to_string(size) + ", edge = " + skepu::to_string(edge) + ", actual2");
                compare(actual2, expected[1]);
            }
        }

        if (size > overlapRadius * 2)
        {
            skepu::Vector<int> actual3(input.size() - 2, 0), actual4(input.size() - 2, 0);
            overlap2.setEdgeMode(skepu::Edge::None);
            overlap2(actual3, actual4, input);
            expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode());
            {
                INFO("size = " + std::to_string(size) + ", edge = None" + ", actual3");
                compare(actual3, expected[0]);
            }
            {
                INFO("size = " + std::to_string(size) + ", edge = None" + ", actual4");
                compare(actual4, expected[1]);
            }
        }
    }
}

TEST_CASE("Invalid sizes 2 output vectors")
{
    overlap2.setOverlap(1);
    int overlapOffset = overlap2.getOverlap() * 2;
    skepu::Vector<int> input(10), correctSize(10), small(input.size()-1), large(input.size()+1),
    correctSizeNone(input.size()-overlapOffset), smallNone(input.size()-overlapOffset-1), largeNone(input.size()-overlapOffset+1);

    skepu::Edge similarEdgeModes[] = {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};

    for (skepu::Edge edge : similarEdgeModes)
    {
        overlap2.setEdgeMode(edge);
        CHECK_THROWS_WITH(overlap2(correctSize, small, input), Catch::Matchers::Contains("invalid output vector size"));
        CHECK_THROWS_WITH(overlap2(correctSize, large, input), Catch::Matchers::Contains("invalid output vector size"));
    }

    overlap2.setEdgeMode(skepu::Edge::None);
    CHECK_THROWS_WITH(overlap2(correctSizeNone, smallNone, input), Catch::Matchers::Contains("invalid output vector size"));
    CHECK_THROWS_WITH(overlap2(correctSizeNone, largeNone, input), Catch::Matchers::Contains("invalid output vector size"));
}