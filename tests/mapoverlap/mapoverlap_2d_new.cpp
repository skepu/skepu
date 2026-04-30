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
            std::tuple<int, float> output = overlap_sum_2(region);
            res[i][j] = std::get<0>(output);
            res2[i][j] = std::get<1>(output);
        }
    return {res, res2};
}

auto overlap2 = skepu::MapOverlap(overlap_sum_2);

TEST_CASE("Multi return")
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

TEST_CASE("Invalid sizes 2 output matrices")
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