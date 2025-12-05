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

std::vector<std::vector<int>> generate_expected(skepu::Matrix<int>& input, int overlap, skepu::Edge edge, skepu::Overlap overlapMode, int pad = 0)
{
    input.flush();
    std::vector<std::vector<int>> res;
    if (edge == skepu::Edge::None && overlapMode == skepu::Overlap::ColWise)
        res.resize(input.total_rows() - overlap * 2);
    else
        res.resize(input.total_rows());
    
    for (int col = 0; col < res.size(); ++col)
    {
        if (edge == skepu::Edge::None && overlapMode == skepu::Overlap::RowWise)
            res[col].resize(input.total_cols() - overlap * 2);
        else
            res[col].resize(input.total_cols());
    }
    
    int data[overlap*3];
    if (overlapMode == skepu::Overlap::RowWise)
        for (int row = 0; row < res.size(); ++row)
        {
            for (int col = 0; col < res[row].size(); ++col)
            {
                int sum = 0;
                int dataIndex = 0;
                for (int o = -overlap; o <= overlap; ++o)
                {
                    int pos = col + o + ((edge == skepu::Edge::None) ? overlap : 0);
                    if (pos < 0 || pos > input.total_cols() - 1)
                    {
                        switch (edge)
                        {
                            case skepu::Edge::Duplicate:
                                pos = clamp(pos, 0, input.total_cols() - 1);
                                break;
                            case skepu::Edge::Cyclic:
                                pos = (pos + input.total_cols()) % input.total_cols();
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
                        data[dataIndex++] = input(row, pos);
                }
                skepu::Region1D<int> r{overlap, 1, &data[overlap]};
                res[row][col] = overlap_sum(r);
            }
        }
    else
        for (int col = 0; col < res[0].size(); ++col)
        {
            for (int row = 0; row < res.size(); ++row)
            {
                int sum = 0;
                int dataIndex = 0;
                for (int o = -overlap; o <= overlap; ++o)
                {
                    int pos = row + o + ((edge == skepu::Edge::None) ? overlap : 0);
                    if (pos < 0 || pos > input.total_rows() - 1)
                    {
                        switch (edge)
                        {
                            case skepu::Edge::Duplicate:
                                pos = clamp(pos, 0, input.total_rows() - 1);
                                break;
                            case skepu::Edge::Cyclic:
                                pos = (pos + input.total_rows()) % input.total_rows();
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
                        data[dataIndex++] = input(pos, col);
                }
                skepu::Region1D<int> r{overlap, 1, &data[overlap]};
                res[row][col] = overlap_sum(r);
            }
        }
    return res;
}

void compare(skepu::Matrix<int>& actual, std::vector<std::vector<int>>& expected)
{
    actual.flush();
    for (int row = 0; row < actual.total_rows(); ++row)
    {
        for (int col = 0; col < actual.total_cols(); ++col)
        {
            INFO("pos = " + std::to_string(row) + "x" + std::to_string(col));
            CHECK(actual(row, col) == expected[row][col]);
        }
    }
}

auto overlap = skepu::MapOverlap(overlap_sum);

TEST_CASE("Fundamentals")
{
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    std::vector<skepu::Overlap> overlapModes{skepu::Overlap::RowWise, skepu::Overlap::ColWise};
    int pad = 10;
    int overlap_radius = 1;
    overlap.setOverlap(overlap_radius);
    overlap.setPad(10);

    for (int rows = 2; rows <= 9; ++rows)
        for (int cols = 2; cols <= 9; ++cols)
        {
            skepu::Matrix<int> input(rows, cols, 1);
            skepu::Matrix<int> actual(input.total_rows(), input.total_cols(), 0);
            std::vector<std::vector<int>> expected;
            skepu::external(skepu::read(input), [&]{
                for (int i = 0; i < input.total_rows(); ++i)
                    for (int j = 0; j < input.total_cols(); ++j)
                        {
                            input(i, j) = i * input.total_cols() + j + 1;
                        }
            }, skepu::write(input));

            // TODO: Add overlaps larger than input size.

            for (skepu::Overlap overlapMode : overlapModes)
            {
                overlap.setOverlapMode(overlapMode);
                for (skepu::Edge edge : regularEdges)
                {
                    overlap.setEdgeMode(edge);
                    overlap(actual, input);
                    expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), overlap.getOverlapMode(), pad);
                    INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = " + skepu::to_string(edge) + ", actual");
                    compare(actual, expected);
                }
            }

            overlap.setEdgeMode(skepu::Edge::None);
            if (cols > overlap_radius * 2)
            {
                skepu::Matrix<int> actual2(input.total_rows(), input.total_cols() - 2, 0);
                overlap.setOverlapMode(skepu::Overlap::RowWise);
                overlap(actual2, input);
                INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = None" + ", actual2");
                expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), overlap.getOverlapMode());
                compare(actual2, expected);
            }

            if (rows > overlap_radius * 2)
            {
                skepu::Matrix<int> actual3(input.total_rows() - 2, input.total_cols(), 0);
                overlap.setOverlapMode(skepu::Overlap::ColWise);
                overlap(actual3, input);
                expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), overlap.getOverlapMode());
                INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = None" + ", actual3");
                compare(actual3, expected);
            }
        }
}


TEST_CASE("Invalid sizes")
{
    overlap.setOverlap(1);
    int overlapOffset = overlap.getOverlap() * 2;
    skepu::Matrix<int> input(10, 10), correctSize(10, 10), smallRow(input.total_rows()-1, input.total_cols()), smallCol(input.total_rows(), input.total_cols()-1),
    largeRow(input.total_rows()+1, input.total_cols()), largeCol(input.total_rows(), input.total_cols()+1),
    correctSizeNoneRowWise(input.total_rows(), input.total_cols()-overlapOffset), smallRowNoneRowWise(input.total_rows()-1, input.total_cols()-overlapOffset),
    smallColNoneRowWise(input.total_rows(), input.total_cols()-overlapOffset-1), largeRowNoneRowWise(input.total_rows()+1, input.total_cols()-overlapOffset),
    largeColNoneRowWise(input.total_rows(), input.total_cols()-overlapOffset+1),
    correctSizeNoneColWise(input.total_rows()-overlapOffset, input.total_cols()), smallRowNoneColWise(input.total_rows()-overlapOffset-1, input.total_cols()),
    smallColNoneColWise(input.total_rows()-overlapOffset, input.total_cols()-1), largeRowNoneColWise(input.total_rows()-overlapOffset+1, input.total_cols()),
    largeColNoneColWise(input.total_rows()-overlapOffset, input.total_cols()+1);

    skepu::Edge similarEdgeModes[] = {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    skepu::Overlap overlapModes[] = {skepu::Overlap::RowWise, skepu::Overlap::ColWise};

    for (skepu::Edge edge : similarEdgeModes)
    {
        overlap.setEdgeMode(edge);
        for (skepu::Overlap overlapMode : overlapModes)
        {
            overlap.setOverlapMode(overlapMode);
            CHECK_THROWS_WITH(overlap(smallRow, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
            CHECK_THROWS_WITH(overlap(largeRow, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
            CHECK_THROWS_WITH(overlap(smallCol, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
            CHECK_THROWS_WITH(overlap(largeCol, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
        }
    }

    overlap.setEdgeMode(skepu::Edge::None);
    overlap.setOverlapMode(skepu::Overlap::RowWise);
    CHECK_THROWS_WITH(overlap(smallRowNoneRowWise, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
    CHECK_THROWS_WITH(overlap(largeRowNoneRowWise, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
    CHECK_THROWS_WITH(overlap(smallColNoneRowWise, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
    CHECK_THROWS_WITH(overlap(largeColNoneRowWise, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
    overlap.setOverlap(6);
    CHECK_THROWS_WITH(overlap(correctSizeNoneRowWise, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));

    overlap.setOverlap(1);
    overlap.setOverlapMode(skepu::Overlap::ColWise);
    CHECK_THROWS_WITH(overlap(smallRowNoneColWise, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
    CHECK_THROWS_WITH(overlap(largeRowNoneColWise, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
    CHECK_THROWS_WITH(overlap(smallColNoneColWise, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
    CHECK_THROWS_WITH(overlap(largeColNoneColWise, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
    overlap.setOverlap(6);
    CHECK_THROWS_WITH(overlap(correctSizeNoneColWise, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
}


skepu::multiple<int, int> overlap_sum_2(skepu::Region1D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        val += r(i);
    return skepu::ret(val, val+1);
}

std::vector<std::vector<std::vector<int>>> generate_expected_2(skepu::Matrix<int>& input, int overlap, skepu::Edge edge, skepu::Overlap overlapMode, int pad = 0)
{
    input.flush();
    std::vector<std::vector<int>> res, res2;
    if (edge == skepu::Edge::None && overlapMode == skepu::Overlap::ColWise)
        res.resize(input.total_rows() - overlap * 2);
    else
        res.resize(input.total_rows());
    res2.resize(res.size());
    
    for (int col = 0; col < res.size(); ++col)
    {
        if (edge == skepu::Edge::None && overlapMode == skepu::Overlap::RowWise)
            res[col].resize(input.total_cols() - overlap * 2);
        else
            res[col].resize(input.total_cols());
        res2[col].resize(res[col].size());
    }
    
    int data[overlap*3];
    if (overlapMode == skepu::Overlap::RowWise)
        for (int row = 0; row < res.size(); ++row)
        {
            for (int col = 0; col < res[row].size(); ++col)
            {
                int sum = 0;
                int dataIndex = 0;
                for (int o = -overlap; o <= overlap; ++o)
                {
                    int pos = col + o + ((edge == skepu::Edge::None) ? overlap : 0);
                    if (pos < 0 || pos > input.total_cols() - 1)
                    {
                        switch (edge)
                        {
                            case skepu::Edge::Duplicate:
                                pos = clamp(pos, 0, input.total_cols() - 1);
                                break;
                            case skepu::Edge::Cyclic:
                                pos = (pos + input.total_cols()) % input.total_cols();
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
                        data[dataIndex++] = input(row, pos);
                }
                skepu::Region1D<int> r{overlap, 1, &data[overlap]};
                std::tuple<int, int> output = overlap_sum_2(r);
                res[row][col] = std::get<0>(output);
                res2[row][col] = std::get<1>(output);
            }
        }
    else
        for (int col = 0; col < res[0].size(); ++col)
        {
            for (int row = 0; row < res.size(); ++row)
            {
                int sum = 0;
                int dataIndex = 0;
                for (int o = -overlap; o <= overlap; ++o)
                {
                    int pos = row + o + ((edge == skepu::Edge::None) ? overlap : 0);
                    if (pos < 0 || pos > input.total_rows() - 1)
                    {
                        switch (edge)
                        {
                            case skepu::Edge::Duplicate:
                                pos = clamp(pos, 0, input.total_rows() - 1);
                                break;
                            case skepu::Edge::Cyclic:
                                pos = (pos + input.total_rows()) % input.total_rows();
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
                        data[dataIndex++] = input(pos, col);
                }
                skepu::Region1D<int> r{overlap, 1, &data[overlap]};
                std::tuple<int, int> output = overlap_sum_2(r);
                res[row][col] = std::get<0>(output);
                res2[row][col] = std::get<1>(output);
            }
        }
    return {res, res2};
}

auto overlap2 = skepu::MapOverlap(overlap_sum_2);

TEST_CASE("Multi return")
{
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    std::vector<skepu::Overlap> overlapModes{skepu::Overlap::RowWise, skepu::Overlap::ColWise};
    int pad = 10;
    int overlap_radius = 1;
    overlap2.setOverlap(overlap_radius);
    overlap2.setPad(10);

    for (int rows = 2; rows <= 9; ++rows)
        for (int cols = 2; cols <= 9; ++cols)
    {
        skepu::Matrix<int> input(rows, cols, 1);
        skepu::Matrix<int> actual(input.total_rows(), input.total_cols(), 0), actual2(input.total_rows(), input.total_cols(), 0);
        std::vector<std::vector<std::vector<int>>> expected;
        skepu::external(skepu::read(input), [&]{
        for (int i = 0; i < input.total_rows(); ++i)
            for (int j = 0; j < input.total_cols(); ++j)
            {
                input(i, j) = i * input.total_cols() + j + 1;
            }
        }, skepu::write(input));

        // TODO: Add overlaps larger than input size.

        for (skepu::Overlap overlapMode : overlapModes)
        {
            overlap2.setOverlapMode(overlapMode);
            for (skepu::Edge edge : regularEdges)
            {
                overlap2.setEdgeMode(edge);
                overlap2(actual, actual2, input);
                expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode(), overlap2.getOverlapMode(), pad);
                {
                    INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = " + skepu::to_string(edge) + ", actual");
                    compare(actual, expected[0]);
                }
                {
                    INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = " + skepu::to_string(edge) + ", actual2");
                    compare(actual2, expected[1]);
                }
            }
        }

        overlap2.setEdgeMode(skepu::Edge::None);
        if (cols > overlap_radius * 2)
        {
            skepu::Matrix<int> actual3(input.total_rows(), input.total_cols() - overlap_radius * 2, 0), actual4(input.total_rows(), input.total_cols() - overlap_radius * 2, 0);
            overlap2.setOverlapMode(skepu::Overlap::RowWise);
            overlap2(actual3, actual4, input);
            expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode(), overlap2.getOverlapMode());
            {
                INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = None" + ", actual3");
                compare(actual3, expected[0]);
            }
            {
                INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = None" + ", actual4");
                compare(actual4, expected[1]);
            }
        }

        if (rows > overlap_radius * 2)
        {
            skepu::Matrix<int> actual5(input.total_rows() - overlap_radius * 2, input.total_cols(), 0), actual6(input.total_rows() - overlap_radius * 2, input.total_cols(), 0);
            overlap2.setOverlapMode(skepu::Overlap::ColWise);
            overlap2(actual5, actual6, input);
            expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode(), overlap2.getOverlapMode());
            {
                INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = None" + ", actual5");
                compare(actual5, expected[0]);
            }
            {
                INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = None" + ", actual6");
                compare(actual6, expected[1]);
            }
        }
    }
}

TEST_CASE("Invalid sizes 2 output matrices")
{
    overlap2.setOverlap(1);
    int overlapOffset = overlap2.getOverlap() * 2;
    skepu::Matrix<int> input(10, 10), correctSize(10, 10), smallRow(input.total_rows()-1, input.total_cols()), smallCol(input.total_rows(), input.total_cols()-1),
    largeRow(input.total_rows()+1, input.total_cols()), largeCol(input.total_rows(), input.total_cols()+1),
    correctSizeNoneRowWise(input.total_rows(), input.total_cols()-overlapOffset), smallRowNoneRowWise(input.total_rows()-1, input.total_cols()-overlapOffset),
    smallColNoneRowWise(input.total_rows(), input.total_cols()-overlapOffset-1), largeRowNoneRowWise(input.total_rows()+1, input.total_cols()-overlapOffset),
    largeColNoneRowWise(input.total_rows(), input.total_cols()-overlapOffset+1),
    correctSizeNoneColWise(input.total_rows()-overlapOffset, input.total_cols()), smallRowNoneColWise(input.total_rows()-overlapOffset-1, input.total_cols()),
    smallColNoneColWise(input.total_rows()-overlapOffset, input.total_cols()-1), largeRowNoneColWise(input.total_rows()-overlapOffset+1, input.total_cols()),
    largeColNoneColWise(input.total_rows()-overlapOffset, input.total_cols()+1);

    skepu::Edge similarEdgeModes[] = {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    skepu::Overlap overlapModes[] = {skepu::Overlap::RowWise, skepu::Overlap::ColWise};

    for (skepu::Edge edge : similarEdgeModes)
    {
        overlap2.setEdgeMode(edge);
        for (skepu::Overlap overlapMode : overlapModes)
        {
            CHECK_THROWS_WITH(overlap2(correctSize, smallRow, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
            CHECK_THROWS_WITH(overlap2(correctSize, largeRow, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
            CHECK_THROWS_WITH(overlap2(correctSize, smallCol, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
            CHECK_THROWS_WITH(overlap2(correctSize, largeCol, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
        }
    }

    overlap2.setEdgeMode(skepu::Edge::None);
    overlap2.setOverlapMode(skepu::Overlap::RowWise);
    CHECK_THROWS_WITH(overlap2(correctSizeNoneRowWise, smallRowNoneRowWise, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
    CHECK_THROWS_WITH(overlap2(correctSizeNoneRowWise, largeRowNoneRowWise, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
    CHECK_THROWS_WITH(overlap2(correctSizeNoneRowWise, smallColNoneRowWise, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
    CHECK_THROWS_WITH(overlap2(correctSizeNoneRowWise, largeColNoneRowWise, input), Catch::Matchers::Contains("invalid number of output matrix cols"));

    overlap2.setOverlapMode(skepu::Overlap::ColWise);
    CHECK_THROWS_WITH(overlap2(correctSizeNoneColWise, smallRowNoneColWise, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
    CHECK_THROWS_WITH(overlap2(correctSizeNoneColWise, largeRowNoneColWise, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
    CHECK_THROWS_WITH(overlap2(correctSizeNoneColWise, smallColNoneColWise, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
    CHECK_THROWS_WITH(overlap2(correctSizeNoneColWise, largeColNoneColWise, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
}