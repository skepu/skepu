// Merged from: mapoverlap_1d.cpp, mapoverlap_1d_rowcol.cpp, mapoverlap_2d_new.cpp, mapoverlap_3d_new.cpp, mapoverlap_4d_new.cpp
// (one program instead of one for each: the build of a test program has a cost of its own; each part is in a namespace, and its test cases have its name as a prefix)
#define SKEPU_ENABLE_EXCEPTIONS

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>
#include <test_helper_functions.hpp>
#include <type_traits>


// ======== mapoverlap_1d ========

namespace mapoverlap_1d_part
{
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
                        pos = std::clamp(pos, 0, static_cast<int>(input.size() - 1));
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

TEST_CASE("mapoverlap_1d: Fundamentals")
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

TEST_CASE("mapoverlap_1d: Overlap radius")
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

TEST_CASE("mapoverlap_1d: Invalid sizes")
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
                        pos = std::clamp(pos, 0, static_cast<int>(input.size() - 1));
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

TEST_CASE("mapoverlap_1d: Multi return")
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

TEST_CASE("mapoverlap_1d: Invalid sizes 2 output vectors")
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
}


// ======== mapoverlap_1d_rowcol ========

namespace mapoverlap_1d_rowcol_part
{
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
                                pos = std::clamp(pos, 0, static_cast<int>(input.total_cols() - 1));
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
                                pos = std::clamp(pos, 0, static_cast<int>(input.total_rows() - 1));
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

TEST_CASE("mapoverlap_1d_rowcol: Fundamentals")
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


TEST_CASE("mapoverlap_1d_rowcol: Invalid sizes")
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
                                pos = std::clamp(pos, 0, static_cast<int>(input.total_cols() - 1));
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
                                pos = std::clamp(pos, 0, static_cast<int>(input.total_rows() - 1));
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

TEST_CASE("mapoverlap_1d_rowcol: Multi return")
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

TEST_CASE("mapoverlap_1d_rowcol: Invalid sizes 2 output matrices")
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
}


// ======== mapoverlap_2d_new ========

namespace mapoverlap_2d_new_part
{
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
                                ii = std::clamp(ii, 0, static_cast<int>(input.total_rows() - 1));
                                jj = std::clamp(jj, 0, static_cast<int>(input.total_cols() - 1));
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

template <typename T>
void compare(skepu::Matrix<T>& actual, std::vector<std::vector<T>>& expected)
{
    actual.flush();
    for (int row = 0; row < actual.total_rows(); ++row)
        for (int col = 0; col < actual.total_cols(); ++col)
        {
            INFO("Checking at (" + std::to_string(row) + ", " + std::to_string(col) + "), comparing " + std::to_string(actual(row, col)) + " and " + std::to_string(expected[row][col]));
            if constexpr (std::is_floating_point<T>::value)
                is_similar(actual(row, col), expected[row][col]);
            else
                CHECK(actual(row, col) == expected[row][col]);
        }
}

auto overlap = skepu::MapOverlap(overlap_sum);

TEST_CASE("mapoverlap_2d_new: Fundamentals")
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

TEST_CASE("mapoverlap_2d_new: Overlap ranges")
{
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

TEST_CASE("mapoverlap_2d_new: Invalid sizes")
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
    CHECK_THROWS_WITH(overlap(correctSizeNone, input), Catch::Matchers::Contains("input/output matrix row count mismatch")); // Rows are checked first
}

skepu::multiple<int, float> overlap_sum_2(skepu::Region2D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        for (int j = -r.oj; j <= r.oj; ++j)
            val += r(i, j);
    return skepu::ret(val, val+0.5f);
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<float>>> generate_expected_2(skepu::Matrix<int>& input, std::tuple<int, int> overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    std::vector<std::vector<int>> res;
    std::vector<std::vector<float>> res2;
    int outsize[3];
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
    res2.resize(outsize[0]);
    for (int i = 0; i < res.size(); ++i)
    {
        res[i].resize(outsize[1]);
        res2[i].resize(outsize[1]);
    }

    const size_t elementsInRegion = (std::get<0>(overlap) * 2 + 1) * (std::get<1>(overlap) * 2 + 1);
    int data[elementsInRegion];
    
    skepu::Region2D<int> region(std::get<0>(overlap), std::get<1>(overlap),
                                input.size_i(), input.size_j(),
                                (std::get<1>(overlap) * 2 + 1), &data[elementsInRegion/2]);
    
    for (int i = 0; i < res.size(); ++i)
        for (int j = 0; j < res[i].size(); ++j)
        {
            int pos = 0;
            for (int oi = -std::get<0>(overlap); oi <= std::get<0>(overlap); ++oi)
                for (int oj = -std::get<1>(overlap); oj <= std::get<1>(overlap); ++oj)
                {
                    int ii = i + oi + ((edge == skepu::Edge::None) ? std::get<0>(overlap) : 0);
                    int jj = j + oj + ((edge == skepu::Edge::None) ? std::get<1>(overlap) : 0);
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
            std::tuple<int, float> output = overlap_sum_2(region);
            res[i][j] = std::get<0>(output);
            res2[i][j] = std::get<1>(output);
        }
    return {res, res2};
}

auto overlap2 = skepu::MapOverlap(overlap_sum_2);

TEST_CASE("mapoverlap_2d_new: Multi return")
{
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    int pad = 10;
    int overlapRadius = 1;
    overlap2.setOverlap(overlapRadius, overlapRadius);
    overlap2.setPad(pad);

    for (int rows = 2; rows <= 11; ++rows)
        for (int cols = 2; cols <= 11; ++cols)
        {
            skepu::Matrix<int> input(rows, cols, 1);
            skepu::Matrix<int> actual(input.total_rows(), input.total_cols(), 0);
            skepu::Matrix<float> actual2(input.total_rows(), input.total_cols(), 0);
            std::pair<std::vector<std::vector<int>>, std::vector<std::vector<float>>> expected;
            skepu::external(skepu::read(input), [&]{
                int val = 0;
                for (int ii = 0; ii < input.total_rows(); ++ii)
                    for (int jj = 0; jj < input.total_cols(); ++jj)
                                input(ii, jj) = ++val;
            }, skepu::write(input));

            // TODO: Add overlaps larger than input size.

            for (skepu::Edge edge : regularEdges)
            {
                overlap2.setEdgeMode(edge);
                overlap2(actual, actual2, input);
                expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode(), pad);
                {
                    INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = " + skepu::to_string(edge) + ", actual");
                    compare(actual, expected.first);
                }
                {
                    INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = " + skepu::to_string(edge) + ", actual2");
                    compare(actual2, expected.second);
                }
            }

            if (rows > overlapRadius * 2 &&
                cols > overlapRadius * 2)
            {
                skepu::Matrix<int> actual3(input.total_rows()-overlapRadius*2, input.total_cols()-overlapRadius*2, 0);
                skepu::Matrix<float> actual4(input.total_rows()-overlapRadius*2, input.total_cols()-overlapRadius*2, 0);
                overlap2.setEdgeMode(skepu::Edge::None);
                overlap2(actual3, actual4, input);
                expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode());
                {
                    INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = None" + ", actual3");
                    compare(actual3, expected.first);
                }
                {
                    INFO("size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", edge = None" + ", actual4");
                    compare(actual4, expected.second);
                }
            }
        }
}

TEST_CASE("mapoverlap_2d_new: Invalid sizes 2 output matrices")
{
    overlap2.setOverlap(1, 1);
    std::tuple<int, int> overlapOffsetTemp = overlap2.getOverlap();
    int overlapOffset[2];
    overlapOffset[0] = std::get<0>(overlapOffsetTemp) * 2;
    overlapOffset[1] = std::get<1>(overlapOffsetTemp) * 2;
    skepu::Matrix<int> input(10, 10), correctSize(10, 10), correctSizeNone(input.total_rows()-overlapOffset[0], input.total_cols()-overlapOffset[1]);
    skepu::Matrix<float> smallRow(input.total_rows()-1, input.total_cols()), smallCol(input.total_rows(), input.total_cols()-1),
    largeRow(input.total_rows()+1, input.total_cols()), largeCol(input.total_rows(), input.total_cols()+1),
    smallRowNone(input.total_rows()-overlapOffset[0]-1, input.total_cols()-overlapOffset[1]), smallColNone(input.total_rows()-overlapOffset[0], input.total_cols()-overlapOffset[1]-1),
    largeRowNone(input.total_rows()-overlapOffset[0]+1, input.total_cols()-overlapOffset[1]), largeColNone(input.total_rows()-overlapOffset[0], input.total_cols()-overlapOffset[1]+1);

    skepu::Edge similarEdgeModes[] = {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};

    for (skepu::Edge edge : similarEdgeModes)
    {
        overlap2.setEdgeMode(edge);
        CHECK_THROWS_WITH(overlap2(correctSize, smallRow, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
        CHECK_THROWS_WITH(overlap2(correctSize, largeRow, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
        CHECK_THROWS_WITH(overlap2(correctSize, smallCol, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
        CHECK_THROWS_WITH(overlap2(correctSize, largeCol, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
    }

    overlap2.setEdgeMode(skepu::Edge::None);
    CHECK_THROWS_WITH(overlap2(correctSizeNone, smallRowNone, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
    CHECK_THROWS_WITH(overlap2(correctSizeNone, largeRowNone, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
    CHECK_THROWS_WITH(overlap2(correctSizeNone, smallColNone, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
    CHECK_THROWS_WITH(overlap2(correctSizeNone, largeColNone, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
}
}


// ======== mapoverlap_3d_new ========

namespace mapoverlap_3d_new_part
{
int overlap_sum(skepu::Region3D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        for (int j = -r.oj; j <= r.oj; ++j)
            for (int k = -r.ok; k <= r.ok; ++k)
                    val += r(i, j, k);
    return val;
}

std::vector<std::vector<std::vector<int>>> generate_expected(skepu::Tensor3<int>& input, std::tuple<int, int, int> overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    std::vector<std::vector<std::vector<int>>> res;
    int outsize[3];
    if (edge == skepu::Edge::None)
    {
        outsize[0] = input.size_i() - std::get<0>(overlap) * 2;
        outsize[1] = input.size_j() - std::get<1>(overlap) * 2;
        outsize[2] = input.size_k() - std::get<2>(overlap) * 2;
    }
    else
    {
        outsize[0] = input.size_i();
        outsize[1] = input.size_j();
        outsize[2] = input.size_k();
    }
    res.resize(outsize[0]);
    for (int i = 0; i < res.size(); ++i)
    {
        res[i].resize(outsize[1]);
        for (int j = 0; j < res[i].size(); ++j)
            res[i][j].resize(outsize[2]);
    }

    const size_t elementsInRegion = (std::get<0>(overlap) * 2 + 1) * (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1);
    int data[elementsInRegion];
    
    skepu::Region3D<int> region(std::get<0>(overlap), std::get<1>(overlap), std::get<2>(overlap),
                                input.size_i(), input.size_j(), input.size_k(),
                                (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1),
                                (std::get<2>(overlap) * 2 + 1), &data[elementsInRegion/2]);
    
    for (int i = 0; i < res.size(); ++i)
        for (int j = 0; j < res[i].size(); ++j)
            for (int k = 0; k < res[i][j].size(); ++k)
            {
                int pos = 0;
                for (int oi = -std::get<0>(overlap); oi <= std::get<0>(overlap); ++oi)
                    for (int oj = -std::get<1>(overlap); oj <= std::get<1>(overlap); ++oj)
                        for (int ok = -std::get<2>(overlap); ok <= std::get<2>(overlap); ++ok)
                        {
                            int ii = i + oi + ((edge == skepu::Edge::None) ? std::get<0>(overlap) : 0);
                            int jj = j + oj + ((edge == skepu::Edge::None) ? std::get<1>(overlap) : 0);
                            int kk = k + ok + ((edge == skepu::Edge::None) ? std::get<2>(overlap) : 0);
                            if (ii < 0 || ii > input.size_i() - 1 ||
                                jj < 0 || jj > input.size_j() - 1 ||
                                kk < 0 || kk > input.size_k() - 1)
                            {
                                switch (edge)
                                {
                                    case skepu::Edge::Duplicate:
                                        ii = std::clamp(ii, 0, static_cast<int>(input.size_i() - 1));
                                        jj = std::clamp(jj, 0, static_cast<int>(input.size_j() - 1));
                                        kk = std::clamp(kk, 0, static_cast<int>(input.size_k() - 1));
                                        break;
                                    case skepu::Edge::Cyclic:
                                        ii = (ii + input.size_i()) % input.size_i();
                                        jj = (jj + input.size_j()) % input.size_j();
                                        kk = (kk + input.size_k()) % input.size_k();
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
                                data[pos++] = input(ii, jj, kk);
                        }
                res[i][j][k] = overlap_sum(region);
            }
    return res;
}

template <typename T>
void compare(skepu::Tensor3<T>& actual, std::vector<std::vector<std::vector<T>>>& expected)
{
    actual.flush();
    for (int i = 0; i < actual.size_i(); ++i)
        for (int j = 0; j < actual.size_j(); ++j)
            for (int k = 0; k < actual.size_k(); ++k)
            {
                INFO("Checking at (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + "), comparing " + std::to_string(actual(i, j, k)) + " and " + std::to_string(expected[i][j][k]));
                if constexpr (std::is_floating_point<T>::value)
                    is_similar(actual(i, j, k), expected[i][j][k]);
                else
                    CHECK(actual(i, j, k) == expected[i][j][k]);
            }
}



auto overlap = skepu::MapOverlap(overlap_sum);


TEST_CASE("mapoverlap_3d_new: Fundamentals")
{
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    int overlap_radius[3] = {1, 1, 1};
    overlap.setOverlap(overlap_radius[0], overlap_radius[1], overlap_radius[2]);
    overlap.setPad(1);

    for (int i = 2; i <= 6; ++i)
        for (int j = 2; j <= 6; ++j)
            for (int k = 2; k <= 6; ++k)
            {
                skepu::Tensor3<int> input(i, j, k, "Input");
                input.flush();
                int val = 0;
                for (int ii = 0; ii < i; ++ii)
                    for (int jj = 0; jj < j; ++jj)
                        for (int kk = 0; kk < k; ++kk)
                            input(ii, jj, kk) = ++val;

                skepu::Tensor3<int> actual(input.size_i(), input.size_j(), input.size_k(), "Output", 0);
                std::vector<std::vector<std::vector<int>>> expected;
                
                for (skepu::Edge const& edge : regularEdges)
                {
                    INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + ", edge = " + skepu::to_string(edge));
                    overlap.setEdgeMode(edge);
                    overlap(actual, input);
                    expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), overlap.getPad());
                    compare(actual, expected);
                }
                
                if (i > overlap_radius[0] * 2 &&
                    j > overlap_radius[1] * 2 &&
                    k > overlap_radius[2] * 2)
                {
                    INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + ", edge = None");
                    skepu::Tensor3<int> actual2(input.size_i() - overlap_radius[0] * 2, input.size_j() - overlap_radius[1] * 2, input.size_k() - overlap_radius[2] * 2, "Output", 0);
                    overlap.setEdgeMode(skepu::Edge::None);
                    overlap(actual2, input);
                    expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
                    compare(actual2, expected);
                }
            }
}

TEST_CASE("mapoverlap_3d_new: Overlap ranges")
{
    int inputSize = 10;
    int overlapUpperBound = 3;


    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    skepu::Tensor3<int> input(inputSize, inputSize, inputSize, "Input", 1);
    input.flush();
    int val = 0;
    for (int ii = 0; ii < inputSize; ++ii)
        for (int jj = 0; jj < inputSize; ++jj)
            for (int kk = 0; kk < inputSize; ++kk)
                input(ii, jj, kk) = ++val;

    skepu::Tensor3<int> actual(input.size_i(), input.size_j(), input.size_k(), "Output", 0);
    std::vector<std::vector<std::vector<int>>> expected;
    overlap.setPad(1);

    for (int i = 0; i <= overlapUpperBound; ++i)
        for (int j = 0; j <= overlapUpperBound; ++j)
            for (int k = 0; k <= overlapUpperBound; ++k)
            {
                overlap.setOverlap(i, j, k);
                
                for (skepu::Edge const& edge : regularEdges)
                {
                    overlap.setEdgeMode(edge);
                    overlap(actual, input);
                    expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode(), overlap.getPad());
                    compare(actual, expected);
                }
                
                skepu::Tensor3<int> actual2(input.size_i() - i * 2, input.size_j() - j * 2, input.size_k() - k * 2, "Output", 0);
                overlap.setEdgeMode(skepu::Edge::None);
                overlap(actual2, input);
                expected = generate_expected(input, overlap.getOverlap(), overlap.getEdgeMode());
                compare(actual2, expected);
            }
}

TEST_CASE("mapoverlap_3d_new: Invalid sizes")
{
    overlap.setOverlap(1, 1, 1);
    std::tuple<int, int, int> overlapOffsetTemp = overlap.getOverlap();
    int overlapOffset[3];
    overlapOffset[0] = std::get<0>(overlapOffsetTemp) * 2;
    overlapOffset[1] = std::get<1>(overlapOffsetTemp) * 2;
    overlapOffset[2] = std::get<2>(overlapOffsetTemp) * 2;
    skepu::Tensor3<int> input(10, 10, 10), correctSize(input.size_i(), input.size_j(), input.size_k()),
    small_i(input.size_i()-1, input.size_j(), input.size_k()), small_j(input.size_i(), input.size_j()-1, input.size_k()), small_k(input.size_i(), input.size_j(), input.size_k()-1),
    large_i(input.size_i()+1, input.size_j(), input.size_k()), large_j(input.size_i(), input.size_j()+1, input.size_k()), large_k(input.size_i(), input.size_j(), input.size_k()+1),
    correctSizeNone(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]),
    small_i_none(input.size_i()-overlapOffset[0]-1, input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]),
    small_j_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1]-1, input.size_k()-overlapOffset[2]),
    small_k_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]-1),
    large_i_none(input.size_i()-overlapOffset[0]+1, input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]),
    large_j_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1]+1, input.size_k()-overlapOffset[2]),
    large_k_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]+1);

    skepu::Edge regularEdges[] = {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};

    for (skepu::Edge edge : regularEdges)
    {
        INFO("edge = " + skepu::to_string(edge));
        overlap.setEdgeMode(edge);
        CHECK_THROWS_WITH(overlap(small_i, input), Catch::Matchers::Contains("input/output tensor3 size i mismatch"));
        CHECK_THROWS_WITH(overlap(small_j, input), Catch::Matchers::Contains("input/output tensor3 size j mismatch"));
        CHECK_THROWS_WITH(overlap(small_k, input), Catch::Matchers::Contains("input/output tensor3 size k mismatch"));
        CHECK_THROWS_WITH(overlap(large_i, input), Catch::Matchers::Contains("input/output tensor3 size i mismatch"));
        CHECK_THROWS_WITH(overlap(large_j, input), Catch::Matchers::Contains("input/output tensor3 size j mismatch"));
        CHECK_THROWS_WITH(overlap(large_k, input), Catch::Matchers::Contains("input/output tensor3 size k mismatch"));
    }

    INFO("edge = None");
    overlap.setEdgeMode(skepu::Edge::None);
    CHECK_THROWS_WITH(overlap(small_i_none, input), Catch::Matchers::Contains("input/output tensor3 size i mismatch"));
    CHECK_THROWS_WITH(overlap(small_j_none, input), Catch::Matchers::Contains("input/output tensor3 size j mismatch"));
    CHECK_THROWS_WITH(overlap(small_k_none, input), Catch::Matchers::Contains("input/output tensor3 size k mismatch"));
    CHECK_THROWS_WITH(overlap(large_i_none, input), Catch::Matchers::Contains("input/output tensor3 size i mismatch"));
    CHECK_THROWS_WITH(overlap(large_j_none, input), Catch::Matchers::Contains("input/output tensor3 size j mismatch"));
    CHECK_THROWS_WITH(overlap(large_k_none, input), Catch::Matchers::Contains("input/output tensor3 size k mismatch"));
    overlap.setOverlap(6);
    CHECK_THROWS_WITH(overlap(correctSizeNone, input), Catch::Matchers::Contains("input/output tensor3 size i mismatch"));
}


skepu::multiple<int, float> overlap_sum_2(skepu::Region3D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        for (int j = -r.oj; j <= r.oj; ++j)
            for (int k = -r.ok; k <= r.ok; ++k)
                val += r(i, j, k);
    return skepu::ret(val, val+0.5f);
}

std::pair<std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<std::vector<float>>>> generate_expected_2(skepu::Tensor3<int>& input, std::tuple<int, int, int> overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    std::vector<std::vector<std::vector<int>>> res;
    std::vector<std::vector<std::vector<float>>> res2;
    int outsize[3];
    if (edge == skepu::Edge::None)
    {
        outsize[0] = input.size_i() - std::get<0>(overlap) * 2;
        outsize[1] = input.size_j() - std::get<1>(overlap) * 2;
        outsize[2] = input.size_k() - std::get<2>(overlap) * 2;
    }
    else
    {
        outsize[0] = input.size_i();
        outsize[1] = input.size_j();
        outsize[2] = input.size_k();
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
        }
    }

    const size_t elementsInRegion = (std::get<0>(overlap) * 2 + 1) * (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1);
    int data[elementsInRegion];
    
    skepu::Region3D<int> region(std::get<0>(overlap), std::get<1>(overlap), std::get<2>(overlap),
                                input.size_i(), input.size_j(), input.size_k(),
                                (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1),
                                (std::get<2>(overlap) * 2 + 1), &data[elementsInRegion/2]);
    
    for (int i = 0; i < res.size(); ++i)
        for (int j = 0; j < res[i].size(); ++j)
            for (int k = 0; k < res[i][j].size(); ++k)
            {
                int pos = 0;
                for (int oi = -std::get<0>(overlap); oi <= std::get<0>(overlap); ++oi)
                    for (int oj = -std::get<1>(overlap); oj <= std::get<1>(overlap); ++oj)
                        for (int ok = -std::get<2>(overlap); ok <= std::get<2>(overlap); ++ok)
                        {
                            int ii = i + oi + ((edge == skepu::Edge::None) ? std::get<0>(overlap) : 0);
                            int jj = j + oj + ((edge == skepu::Edge::None) ? std::get<1>(overlap) : 0);
                            int kk = k + ok + ((edge == skepu::Edge::None) ? std::get<2>(overlap) : 0);
                            if (ii < 0 || ii > input.size_i() - 1 ||
                                jj < 0 || jj > input.size_j() - 1 ||
                                kk < 0 || kk > input.size_k() - 1)
                            {
                                switch (edge)
                                {
                                    case skepu::Edge::Duplicate:
                                        ii = std::clamp(ii, 0, static_cast<int>(input.size_i() - 1));
                                        jj = std::clamp(jj, 0, static_cast<int>(input.size_j() - 1));
                                        kk = std::clamp(kk, 0, static_cast<int>(input.size_k() - 1));
                                        break;
                                    case skepu::Edge::Cyclic:
                                        ii = (ii + input.size_i()) % input.size_i();
                                        jj = (jj + input.size_j()) % input.size_j();
                                        kk = (kk + input.size_k()) % input.size_k();
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
                                data[pos++] = input(ii, jj, kk);
                        }
                std::tuple<int, float> output = overlap_sum_2(region);
                res[i][j][k] = std::get<0>(output);
                res2[i][j][k] = std::get<1>(output);
            }
    return {res, res2};
}

auto overlap2 = skepu::MapOverlap(overlap_sum_2);

TEST_CASE("mapoverlap_3d_new: Multi return")
{
    std::vector<skepu::Edge> regularEdges{skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};
    int overlap_radius[3] = {1, 1, 1};
    overlap2.setOverlap(overlap_radius[0], overlap_radius[1], overlap_radius[2]);
    overlap2.setPad(1);

    for (int i = 2; i <= 6; ++i)
        for (int j = 2; j <= 6; ++j)
            for (int k = 2; k <= 6; ++k)
            {
                skepu::Tensor3<int> input(i, j, k, "Input");
                input.flush();
                int val = 0;
                for (int ii = 0; ii < i; ++ii)
                    for (int jj = 0; jj < j; ++jj)
                        for (int kk = 0; kk < k; ++kk)
                                input(ii, jj, kk) = ++val;

                skepu::Tensor3<int> actual(input.size_i(), input.size_j(), input.size_k(), "Output", 0);
                skepu::Tensor3<float> actual2(input.size_i(), input.size_j(), input.size_k(), "Output", 0);
                    
                std::pair<std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<std::vector<float>>>> expected;
                for (skepu::Edge edge : regularEdges)
                {
                    overlap2.setEdgeMode(edge);
                    overlap2(actual, actual2, input);
                    expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode(), overlap2.getPad());
                    {
                        INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + ", edge = " + skepu::to_string(edge) + ", actual");
                        compare(actual, expected.first);
                    }
                    {
                        INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + ", edge = " + skepu::to_string(edge) + ", actual2");
                        compare(actual2, expected.second);
                    }
                }

                if (i > overlap_radius[0] * 2 &&
                    j > overlap_radius[1] * 2 &&
                    k > overlap_radius[2] * 2)
                {
                    skepu::Tensor3<int> actual3(input.size_i()-overlap_radius[0]*2, input.size_j()-overlap_radius[1]*2, input.size_k()-overlap_radius[2]*2, "Output", 0);
                    skepu::Tensor3<float> actual4(input.size_i()-overlap_radius[0]*2, input.size_j()-overlap_radius[1]*2, input.size_k()-overlap_radius[2]*2, "Output", 0);
                    overlap2.setEdgeMode(skepu::Edge::None);
                    overlap2(actual3, actual4, input);
                    expected = generate_expected_2(input, overlap2.getOverlap(), overlap2.getEdgeMode());
                    {
                        INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + ", edge = None" + ", actual3");
                        compare(actual3, expected.first);
                    }
                    {
                        INFO("size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + ", edge = None" + ", actual4");
                        compare(actual4, expected.second);
                    }
                }
            }
}

TEST_CASE("mapoverlap_3d_new: Invalid sizes 2 output tensor3s")
{
    overlap2.setOverlap(1, 1, 1);
    std::tuple<int, int, int> overlapOffsetTemp = overlap2.getOverlap();
    int overlapOffset[3];
    overlapOffset[0] = std::get<0>(overlapOffsetTemp) * 2;
    overlapOffset[1] = std::get<1>(overlapOffsetTemp) * 2;
    overlapOffset[2] = std::get<2>(overlapOffsetTemp) * 2;
    skepu::Tensor3<int> input(10, 10, 10), correctSize(input.size_i(), input.size_j(), input.size_k()), correctSizeNone(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]);
    skepu::Tensor3<float> small_i(input.size_i()-1, input.size_j(), input.size_k()), small_j(input.size_i(), input.size_j()-1, input.size_k()), small_k(input.size_i(), input.size_j(), input.size_k()-1),
    large_i(input.size_i()+1, input.size_j(), input.size_k()), large_j(input.size_i(), input.size_j()+1, input.size_k()), large_k(input.size_i(), input.size_j(), input.size_k()+1),
    small_i_none(input.size_i()-overlapOffset[0]-1, input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]),
    small_j_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1]-1, input.size_k()-overlapOffset[2]),
    small_k_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]-1),
    large_i_none(input.size_i()-overlapOffset[0]+1, input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]),
    large_j_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1]+1, input.size_k()-overlapOffset[2]),
    large_k_none(input.size_i()-overlapOffset[0], input.size_j()-overlapOffset[1], input.size_k()-overlapOffset[2]+1);

    skepu::Edge regularEdges[] = {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad};

    for (skepu::Edge edge : regularEdges)
    {
        INFO("edge = " + skepu::to_string(edge));
        overlap2.setEdgeMode(edge);
        CHECK_THROWS_WITH(overlap2(correctSize, small_i, input), Catch::Matchers::Contains("invalid output tensor3 size i"));
        CHECK_THROWS_WITH(overlap2(correctSize, small_j, input), Catch::Matchers::Contains("invalid output tensor3 size j"));
        CHECK_THROWS_WITH(overlap2(correctSize, small_k, input), Catch::Matchers::Contains("invalid output tensor3 size k"));
        CHECK_THROWS_WITH(overlap2(correctSize, large_i, input), Catch::Matchers::Contains("invalid output tensor3 size i"));
        CHECK_THROWS_WITH(overlap2(correctSize, large_j, input), Catch::Matchers::Contains("invalid output tensor3 size j"));
        CHECK_THROWS_WITH(overlap2(correctSize, large_k, input), Catch::Matchers::Contains("invalid output tensor3 size k"));
    }

    INFO("edge = None");
    overlap2.setEdgeMode(skepu::Edge::None);
    CHECK_THROWS_WITH(overlap2(correctSizeNone, small_i_none, input), Catch::Matchers::Contains("invalid output tensor3 size i"));
    CHECK_THROWS_WITH(overlap2(correctSizeNone, small_j_none, input), Catch::Matchers::Contains("invalid output tensor3 size j"));
    CHECK_THROWS_WITH(overlap2(correctSizeNone, small_k_none, input), Catch::Matchers::Contains("invalid output tensor3 size k"));
    CHECK_THROWS_WITH(overlap2(correctSizeNone, large_i_none, input), Catch::Matchers::Contains("invalid output tensor3 size i"));
    CHECK_THROWS_WITH(overlap2(correctSizeNone, large_j_none, input), Catch::Matchers::Contains("invalid output tensor3 size j"));
    CHECK_THROWS_WITH(overlap2(correctSizeNone, large_k_none, input), Catch::Matchers::Contains("invalid output tensor3 size k"));
}
}


// ======== mapoverlap_4d_new ========

namespace mapoverlap_4d_new_part
{
template <typename T>
using vector4 = std::vector<std::vector<std::vector<std::vector<T>>>>;

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

vector4<int> generate_expected(skepu::Tensor4<int>& input, std::tuple<int, int, int, int> overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    vector4<int> res;
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

    const size_t elementsInRegion = (std::get<0>(overlap) * 2 + 1) * (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1);
    int data[elementsInRegion];
    
    skepu::Region4D<int> region(std::get<0>(overlap), std::get<1>(overlap), std::get<2>(overlap), std::get<3>(overlap),
                                input.size_i(), input.size_j(), input.size_k(), input.size_l(),
                                (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1),
                                (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1),
                                std::get<3>(overlap) * 2 + 1, &data[elementsInRegion/2]);
    
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


TEST_CASE("mapoverlap_4d_new: Fundamentals")
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
                        vector4<int> expected;
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

TEST_CASE("mapoverlap_4d_new: Overlap ranges")
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
    vector4<int> expected;
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

TEST_CASE("mapoverlap_4d_new: Invalid sizes")
{
    overlap.setOverlap(1, 1, 1, 1);
    std::tuple<int, int, int, int> overlapOffsetTemp = overlap.getOverlap();
    int overlapOffset[4];
    overlapOffset[0] = std::get<0>(overlapOffsetTemp) * 2;
    overlapOffset[1] = std::get<1>(overlapOffsetTemp) * 2;
    overlapOffset[2] = std::get<2>(overlapOffsetTemp) * 2;
    overlapOffset[3] = std::get<3>(overlapOffsetTemp) * 2;
    skepu::Tensor4<int> input(10, 10, 10, 10),
    small_i(input.size_i()-1, input.size_j(), input.size_k(), input.size_l()),
    small_j(input.size_i(), input.size_j()-1, input.size_k(), input.size_l()),
    small_k(input.size_i(), input.size_j(), input.size_k()-1, input.size_l()),
    small_l(input.size_i(), input.size_j(), input.size_k(), input.size_l()-1),
    large_i(input.size_i()+1, input.size_j(), input.size_k(), input.size_l()),
    large_j(input.size_i(), input.size_j()+1, input.size_k(), input.size_l()),
    large_k(input.size_i(), input.size_j(), input.size_k()+1, input.size_l()),
    large_l(input.size_i(), input.size_j(), input.size_k(), input.size_l()+1),
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

std::pair<vector4<int>, vector4<float>> generate_expected_2(skepu::Tensor4<int>& input, std::tuple<int, int, int, int> overlap, skepu::Edge edge, int pad = 0)
{
    input.flush();
    vector4<int> res;
    vector4<float> res2;
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

    const size_t elementsInRegion = (std::get<0>(overlap) * 2 + 1) * (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1);
    int data[elementsInRegion];
    
    skepu::Region4D<int> region(std::get<0>(overlap), std::get<1>(overlap), std::get<2>(overlap), std::get<3>(overlap),
                                input.size_i(), input.size_j(), input.size_k(), input.size_l(),
                                (std::get<1>(overlap) * 2 + 1) * (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1),
                                (std::get<2>(overlap) * 2 + 1) * (std::get<3>(overlap) * 2 + 1),
                                std::get<3>(overlap) * 2 + 1, &data[elementsInRegion/2]);
    
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
                    std::tuple<int, float> output = overlap_sum_2(region);
                    res[i][j][k][l] = std::get<0>(output);
                    res2[i][j][k][l] = std::get<1>(output);
                }
    return {res, res2};
}

auto overlap2 = skepu::MapOverlap(overlap_sum_2);

TEST_CASE("mapoverlap_4d_new: Multi return")
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
                        
                    std::pair<vector4<int>, vector4<float>> expected;
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

TEST_CASE("mapoverlap_4d_new: Invalid sizes 2 output tensor4s")
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
}
