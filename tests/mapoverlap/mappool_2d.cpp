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

int pool_sum(skepu::Pool2D<int> p)
{
    int val = 0;
    for (int row = 0; row < p.si; ++row)
        for (int col = 0; col < p.sj; ++col)
            val += p(row, col);
    return val;
}

std::vector<std::vector<int>> generate_expected(skepu::Matrix<int>& input, std::tuple<int, int> poolSize, std::tuple<int, int> stride)
{
    input.flush();
    std::vector<std::vector<int>> res;
    int outsize[2];
    outsize[0] = 1 + (input.total_rows() - std::get<0>(poolSize)) / std::get<0>(stride);
    outsize[1] = 1 + (input.total_cols() - std::get<1>(poolSize)) / std::get<1>(stride);
    res.resize(outsize[0]);
    for (int row = 0; row < res.size(); ++row)
        res[row].resize(outsize[1]);

    skepu::Pool2D<int> region{input, std::get<0>(poolSize), std::get<1>(poolSize), skepu::Edge::None, 0};
    
    for (size_t row = 0; row < res.size(); ++row)
        for (size_t col = 0; col < res[row].size(); ++col)
        {
            region.idx = skepu::Index2D{row * std::get<0>(stride), col * std::get<1>(stride)};
            res[row][col] = pool_sum(region);
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

auto pool = skepu::MapPool(pool_sum);

TEST_CASE("Fundamentals")
{
    int poolSize = 2;
    pool.setPoolSizeAndStride(poolSize, poolSize);

    for (int rows = 2; rows <= 20; rows += poolSize)
        for (int cols = 2; cols <= 20; cols += poolSize)
        {
            skepu::Matrix<int> input(rows, cols, 1);
            skepu::Matrix<int> actual(input.total_rows() / poolSize, input.total_cols() / poolSize, 0);
            std::vector<std::vector<int>> expected;
            skepu::external(skepu::read(input), [&]{
                int val = 0;
                for (int row = 0; row < input.total_rows(); ++row)
                    for (int col = 0; col < input.total_cols(); ++col)
                            input(row, col) = ++val;
            }, skepu::write(input));


            INFO("input size = " + std::to_string(rows) + "x" + std::to_string(cols));
            pool(actual, input);
            expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
            compare(actual, expected);
        }
}

TEST_CASE("Stride sizes")
{
    for (int stride_y = 1; stride_y <= 5; ++stride_y)
        for (int stride_x = 1; stride_x <= 5; ++stride_x)
        {
            int pool_size_y = stride_y * 2,
                pool_size_x = stride_x * 2;
            pool.setPoolSize(pool_size_y, pool_size_x);
            pool.setStride(stride_y, stride_x);
            skepu::Matrix<int> input(pool_size_y + stride_y, pool_size_x + stride_x, 1);
            skepu::Matrix<int> actual(1 + (input.total_rows() - pool_size_y) / stride_y, 1 + (input.total_cols() - pool_size_x) / stride_x, 0);
            std::vector<std::vector<int>> expected;
            skepu::external(skepu::read(input), [&]{
                int val = 0;
                for (int row = 0; row < input.total_rows(); ++row)
                    for (int col = 0; col < input.total_cols(); ++col)
                            input(row, col) = ++val;
            }, skepu::write(input));


            INFO("input size = " + std::to_string(input.total_rows()) + "x" + std::to_string(input.total_cols()) + ", output size = " + std::to_string(actual.total_rows()) + "x" + std::to_string(actual.total_cols()));
            pool(actual, input);
            expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
            compare(actual, expected);
        }
}

TEST_CASE("Pool sizes")
{
    int poolSizeUpperBound = 6;
    std::vector<std::vector<int>> expected;

    for (int i = 1; i <= poolSizeUpperBound; ++i)
        for (int j = 1; j <= poolSizeUpperBound; ++j)
        {
            pool.setPoolSizeAndStride(i, j);
            skepu::Matrix<int> input(i*2, j*2, "Input", 1);
            skepu::Matrix<int> actual(input.total_rows() / i, input.total_cols() / j, "Output", 0);
            skepu::external(skepu::read(input), [&]{
                int val = 0;
                for (int row = 0; row < input.total_rows(); ++row)
                    for (int col = 0; col < input.total_cols(); ++col)
                            input(row, col) = ++val;
            }, skepu::write(input));
            
            INFO("poolSize = " + std::to_string(i) + "x" + std::to_string(j) + ", edge = None");
            pool(actual, input);
            expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
            compare(actual, expected);
        }
}

TEST_CASE("Invalid sizes")
{
    pool.setPoolSizeAndStride(1, 1);
    std::tuple<int, int> poolSizeOffsetTemp = pool.getPoolSize();
    int poolSizeOffset[2];
    poolSizeOffset[0] = std::get<0>(poolSizeOffsetTemp);
    poolSizeOffset[1] = std::get<1>(poolSizeOffsetTemp);
    skepu::Matrix<int> input(10, 10), correctSize(input.total_rows()/poolSizeOffset[0], input.total_rows()/poolSizeOffset[1]),
    largeRow(input.total_rows()/poolSizeOffset[0]+1, input.total_rows()/poolSizeOffset[1]), largeCol(input.total_rows()/poolSizeOffset[0], input.total_rows()/poolSizeOffset[1]+1);

    CHECK_THROWS_WITH(pool(largeRow, input), Catch::Matchers::Contains("input/output matrix row count mismatch"));
    CHECK_THROWS_WITH(pool(largeCol, input), Catch::Matchers::Contains("input/output matrix col count mismatch"));
}

skepu::multiple<int, float> pool_sum_2(skepu::Pool2D<int> p)
{
    int val = 0;
    for (int row = 0; row < p.si; ++row)
        for (int col = 0; col < p.sj; ++col)
            val += p(row, col);
    return skepu::ret(val, val+0.5f);
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<float>>> generate_expected_2(skepu::Matrix<int>& input, std::tuple<int, int> poolSize)
{
    input.flush();
    std::vector<std::vector<int>> res;
    std::vector<std::vector<float>> res2;
    int outsize[2];
    outsize[0] = input.total_rows() / std::get<0>(poolSize);
    outsize[1] = input.total_cols() / std::get<1>(poolSize);
    res.resize(outsize[0]);
    res2.resize(outsize[0]);
    for (int row = 0; row < res.size(); ++row)
    {
        res[row].resize(outsize[1]);
        res2[row].resize(outsize[1]);
    }

    skepu::Pool2D<int> region{input, std::get<0>(poolSize), std::get<1>(poolSize), skepu::Edge::None, 0};

    for (size_t row = 0; row < res.size(); ++row)
        for (size_t col = 0; col < res[row].size(); ++col)
        {
            region.idx = skepu::Index2D{row * std::get<0>(poolSize), col * std::get<1>(poolSize)};
            std::tuple<int, float> output = pool_sum_2(region);
            res[row][col] = std::get<0>(output);
            res2[row][col] = std::get<1>(output);
        }
    return {res, res2};
}

auto pool2 = skepu::MapPool(pool_sum_2);

TEST_CASE("Multi return")
{
    int poolSize = 2;
    pool2.setPoolSizeAndStride(poolSize, poolSize);

    for (int rows = 2; rows <= 20; rows += poolSize)
        for (int cols = 2; cols <= 20; cols += poolSize)
        {
            skepu::Matrix<int> input(rows, cols, 1);
            skepu::Matrix<int> actual(input.total_rows() / poolSize, input.total_cols() / poolSize, 0);
            skepu::Matrix<float> actual2(input.total_rows() / poolSize, input.total_cols() / poolSize, 0);
            std::pair<std::vector<std::vector<int>>, std::vector<std::vector<float>>> expected;
            skepu::external(skepu::read(input), [&]{
                int val = 0;
                for (int row = 0; row < input.total_rows(); ++row)
                    for (int col = 0; col < input.total_cols(); ++col)
                            input(row, col) = ++val;
            }, skepu::write(input));


            pool2(actual, actual2, input);
            expected = generate_expected_2(input, pool2.getPoolSize());
            {
                INFO("input size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", actual");
                compare(actual, expected.first);
            }
            {
                INFO("input size = " + std::to_string(rows) + "x" + std::to_string(cols) + ", actual2");
                compare(actual2, expected.second);
            }
        }
}

TEST_CASE("Invalid sizes 2 output matrices")
{
    pool2.setPoolSizeAndStride(1, 1);
    std::tuple<int, int> poolSizeOffsetTemp = pool2.getPoolSize();
    int poolSizeOffset[2];
    poolSizeOffset[0] = std::get<0>(poolSizeOffsetTemp);
    poolSizeOffset[1] = std::get<1>(poolSizeOffsetTemp);
    skepu::Matrix<int> input(10, 10), correctSize(input.total_rows()/poolSizeOffset[0], input.total_rows()/poolSizeOffset[1]);
    skepu::Matrix<float> smallRow(input.total_rows()/poolSizeOffset[0]-1, input.total_rows()/poolSizeOffset[1]), smallCol(input.total_rows()/poolSizeOffset[0], input.total_rows()/poolSizeOffset[1]-1),
    largeRow(input.total_rows()/poolSizeOffset[0]+1, input.total_rows()/poolSizeOffset[1]), largeCol(input.total_rows()/poolSizeOffset[0], input.total_rows()/poolSizeOffset[1]+1);

    CHECK_THROWS_WITH(pool2(correctSize, smallRow, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
    CHECK_THROWS_WITH(pool2(correctSize, largeRow, input), Catch::Matchers::Contains("invalid number of output matrix rows"));
    CHECK_THROWS_WITH(pool2(correctSize, smallCol, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
    CHECK_THROWS_WITH(pool2(correctSize, largeCol, input), Catch::Matchers::Contains("invalid number of output matrix cols"));
}