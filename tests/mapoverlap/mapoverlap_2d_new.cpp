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

int overlap_sum(skepu::Region2D<int> r)
{
    int val = 0;
    for (int row = -r.oi; row <= r.oi; ++row)
        for (int col = -r.oj; col <= r.oj; ++col)
        {
            val += r(row, col);
        }
    return val;
}

std::vector<std::vector<int>> generate_expected(skepu::Matrix<int>& input, std::tuple<int, int> overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    std::vector<std::vector<int>> res;
    int outsize[2];
    if (edge == skepu::Edge::None)
    {
        outsize[0] = input.total_rows() - std::get<0>(overlap) * 2;
        outsize[1] = input.total_cols() - std::get<1>(overlap) * 2;
    }
    else
    {
        outsize[0] = input.total_rows();
        outsize[1] = input.total_cols();
    }
    res.resize(outsize[0]);
    for (int row = 0; row < res.size(); ++row)
        res[row].resize(outsize[1]);


    const size_t elementsInRegion = (std::get<0>(overlap) * 2 + 1) * (std::get<1>(overlap) * 2 + 1);
    int data[elementsInRegion];

    skepu::Region2D<int> region(std::get<0>(overlap), std::get<1>(overlap),
                                input.total_rows(), input.total_cols(),
                                (std::get<1>(overlap) * 2 + 1), &data[elementsInRegion/2]);
    

    for (int row = 0; row < res.size(); ++row)
        for (int col = 0; col < res[row].size(); ++col)
        {
            int pos = 0;
            for (int oi = -std::get<0>(overlap); oi <= std::get<0>(overlap); ++oi)
                for (int oj = -std::get<1>(overlap); oj <= std::get<1>(overlap); ++oj)             
            {
                int ii = row + oi + ((edge == skepu::Edge::None) ? std::get<0>(overlap) : 0);
                int jj = col + oj + ((edge == skepu::Edge::None) ? std::get<1>(overlap) : 0);
                if (ii < 0 || ii > input.total_rows() - 1 ||
                    jj < 0 || jj > input.total_cols() - 1)
                {
                    switch (edge)
                    {
                        case skepu::Edge::Duplicate:
                            ii = clamp(ii, 0, input.total_rows() - 1);
                            jj = clamp(jj, 0, input.total_cols() - 1);
                            break;
                        case skepu::Edge::Cyclic:
                            ii = (ii + input.total_rows()) % input.total_rows();
                            jj = (jj + input.total_cols()) % input.total_cols();
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
            res[row][col] = overlap_sum(region);
        }
    return res;
}

void compare(skepu::Matrix<int>& actual, std::vector<std::vector<int>>& expected)
{
    actual.flush();
    for (int row = 0; row < actual.total_rows(); ++row)
        for (int col = 0; col < actual.total_cols(); ++col)
    {
        INFO("pos = " + std::to_string(row) + "x" + std::to_string(col));
        CHECK(actual(row, col) == expected[row][col]);
    }
}

auto overlap = skepu::MapOverlap(overlap_sum);

TEST_CASE("Fundamentals")
{
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    int pad = 10;
    int overlapRadius = 1;
    overlap.setOverlap(overlapRadius, overlapRadius);
    overlap.setPad(pad);

    for (int rows = 2; rows <= 11; ++rows)
        for (int cols = 2; cols <= 11; ++cols)
    {
        skepu::Matrix<int> input(rows, cols, 1);
        skepu::Matrix<int> actual(input.total_rows(), input.total_cols(), 0);
        std::vector<std::vector<int>> expected;
        skepu::external(skepu::read(input), [&]{
            for (int row = 0; row < input.total_rows(); ++row)
                for (int col = 0; col < input.total_cols(); ++col)
                    {
                        input(row, col) = row * input.total_cols() + col + 1;
                    }
        }, skepu::write(input));

        // TODO: Add overlaps larger than input size.

        for (skepu::Edge edge : regularEdges)
        {
            INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = " + skepu::to_string(edge));
            overlap.setEdgeMode(edge);
            overlap(actual, input);
            expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), pad);
            compare(actual, expected);
        }

        if (rows > overlapRadius * 2 &&
            cols > overlapRadius * 2)
        {
            INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = None");
            skepu::Matrix<int> actual2(input.total_rows() - overlapRadius * 2, input.total_cols() - overlapRadius * 2, 0);
            overlap.setEdgeMode(skepu::Edge::None);
            overlap(actual2, input);
            expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
            compare(actual2, expected);
        }
    }
}

TEST_CASE("Overlap ranges")
{
    // Going above 3 on overlapUpperBound has severe impact on performance.
    // Seq goes from around 3s to 50s if overlapUpperBound is set to 4.
    int inputSize = 10;
    int overlapUpperBound = 3;


    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    skepu::Matrix<int> input(inputSize, inputSize, "Input", 1);
    input.flush();
    int val = 0;
    for (int ii = 0; ii < inputSize; ++ii)
        for (int jj = 0; jj < inputSize; ++jj)
                    input(ii, jj) = ++val;

    skepu::Matrix<int> actual(input.total_rows(), input.total_cols(), "Output", 0);
    std::vector<std::vector<int>> expected;
    overlap.setPad(10);

    for (int i = 0; i <= overlapUpperBound; ++i)
        for (int j = 0; j <= overlapUpperBound; ++j)
        {
            overlap.setOverlap(i, j);
            
            for (skepu::Edge const& edge : regularEdges)
            {
                INFO("overlap = " + std::to_string(i) + "x" + std::to_string(j) + ", edge = " + skepu::to_string(edge));
                overlap.setEdgeMode(edge);
                overlap(actual, input);
                expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), overlap.getPad());
                compare(actual, expected);
            }
            
            {
                INFO("overlap = " + std::to_string(i) + "x" + std::to_string(j) + ", edge = None");
                skepu::Matrix<int> actual2(input.total_rows() - i * 2, input.total_cols() - j * 2, "Output", 0);
                overlap.setEdgeMode(skepu::Edge::None);
                overlap(actual2, input);
                expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
                compare(actual2, expected);
            }
        }
}
/*
TEST_CASE("Invalid sizes")
{
    overlap.setOverlap(1, 1);
    std::tuple<int, int> overlapOffsetTemp = overlap.getOverlap();
    int overlapOffset[2];
    overlapOffset[0] = std::get<0>(overlapOffsetTemp) * 2;
    overlapOffset[1] = std::get<1>(overlapOffsetTemp) * 2;
    skepu::Matrix<int> input(10, 10), correctSize(10, 10), smallRow(input.total_rows()-1, input.total_cols()), smallCol(input.total_rows(), input.total_cols()-1),
    largeRow(input.total_rows()+1, input.total_cols()), largeCol(input.total_rows(), input.total_cols()+1),
    correctSizeNone(input.total_rows()-overlapOffset[0], input.total_cols()-overlapOffset[1]), smallRowNone(input.total_rows()-overlapOffset[0]-1, input.total_cols()-overlapOffset[1]),
    smallColNone(input.total_rows()-overlapOffset[0], input.total_cols()-overlapOffset[1]-1), largeRowNone(input.total_rows()-overlapOffset[0]+1, input.total_cols()-overlapOffset[1]),
    largeColNone(input.total_rows()-overlapOffset[0], input.total_cols()-overlapOffset[1]+1);

    skepu::Edge regularEdges[] = {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};

    for (skepu::Edge edge : regularEdges)
    {
        INFO("edge = " + skepu::to_string(edge));
        overlap.setEdgeMode(edge);
        CHECK_THROWS_WITH(overlap(smallRow, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
        CHECK_THROWS_WITH(overlap(largeRow, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
        CHECK_THROWS_WITH(overlap(smallCol, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
        CHECK_THROWS_WITH(overlap(largeCol, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
    }

    INFO("edge = None");
    overlap.setEdgeMode(skepu::Edge::None);
    CHECK_THROWS_WITH(overlap(smallRowNone, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
    CHECK_THROWS_WITH(overlap(largeRowNone, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
    CHECK_THROWS_WITH(overlap(smallColNone, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
    CHECK_THROWS_WITH(overlap(largeColNone, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
    overlap.setOverlap(6);
    CHECK_THROWS_WITH(overlap(correctSizeNone, input), Catch::Matchers::Contains("input/output matrix count mismatch"));
}*/
/*
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
}*/