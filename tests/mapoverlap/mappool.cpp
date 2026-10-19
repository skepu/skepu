// Each part is in its own namespace; its test cases are prefixed with its name.
#define SKEPU_ENABLE_EXCEPTIONS

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>
#include <test_helper_functions.hpp>
#include <type_traits>
#include <algorithm>
#include <vector>

// The output sizes to test in a dimension where the largest output has 'max' elements: all of them when there are few, else the smallest
// sizes, the middle, and the sizes closest to the largest one (an output smaller than the largest one is what the "small output" tests are for;
// every size in between runs the same code, but the number of calls grows with the product of the dimensions)
static std::vector<int> out_sizes(int max)
{
    std::vector<int> sizes;
    if (max <= 5)
        for (int s = 1; s <= max; ++s)
            sizes.push_back(s);
    else
        sizes = {1, 2, 3, (max + 1) / 2, max - 1, max};
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}


// ======== mappool_2d ========

namespace mappool_2d_part
{
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

constexpr int size_upper_bound = 50;

TEST_CASE("mappool_2d: Fundamentals")
{
    int poolSize = 2;
    pool.setPoolSizeAndStride(poolSize, poolSize);

    for (int rows = 2; rows <= size_upper_bound; rows += poolSize)
        for (int cols = 2; cols <= size_upper_bound; cols += poolSize)
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


            INFO("input size = " + std::to_string(rows) + "x" + std::to_string(cols) +
                 ", output size = " + std::to_string(actual.size_i()) + "x" + std::to_string(actual.size_j()) + 
                 ", pool size = " + std::to_string(poolSize) + "x" + std::to_string(poolSize) +
                 ", stride = " + std::to_string(poolSize) + "x" + std::to_string(poolSize));
            pool(actual, input);
            expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
            compare(actual, expected);
        }
}

TEST_CASE("mappool_2d: Stride sizes")
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


            INFO("input size = " + std::to_string(input.total_rows()) + "x" + std::to_string(input.total_cols()) +
                 ", output size = " + std::to_string(actual.total_rows()) + "x" + std::to_string(actual.total_cols()) +
                 ", pool size = " + std::to_string(pool_size_y) + "x" + std::to_string(pool_size_x) + 
                 ", stride = " + std::to_string(stride_y) + "x" + std::to_string(stride_x));
            pool(actual, input);
            expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
            compare(actual, expected);
        }
}

TEST_CASE("mappool_2d: Pool sizes")
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

TEST_CASE("mappool_2d: Invalid sizes")
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

TEST_CASE("mappool_2d: Multi return")
{
    int poolSize = 2;
    pool2.setPoolSizeAndStride(poolSize, poolSize);

    for (int rows = 2; rows <= size_upper_bound; rows += poolSize)
        for (int cols = 2; cols <= size_upper_bound; cols += poolSize)
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

TEST_CASE("mappool_2d: Invalid sizes 2 output matrices")
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
}


// ======== mappool_3d ========

namespace mappool_3d_part
{
template <typename T>
using vector3 = std::vector<std::vector<std::vector<T>>>;

int pool_sum(skepu::Pool3D<int> p)
{
    int val = 0;
    for (int i = 0; i < p.si; ++i)
        for (int j = 0; j < p.sj; ++j)
            for (int k = 0; k < p.sk; ++k)
                    val += p(i, j, k);
    return val;
}

vector3<int> generate_expected(skepu::Tensor3<int>& input, std::tuple<int, int, int> poolSize, std::tuple<int, int, int> stride)
{
    input.flush();
    vector3<int> res;
    int outsize[3];
    outsize[0] = 1 + (input.size_i() - std::get<0>(poolSize)) / std::get<0>(stride);
    outsize[1] = 1 + (input.size_j() - std::get<1>(poolSize)) / std::get<1>(stride);
    outsize[2] = 1 + (input.size_k() - std::get<2>(poolSize)) / std::get<2>(stride);
    res.resize(outsize[0]);
    for (int i = 0; i < res.size(); ++i)
    {
        res[i].resize(outsize[1]);
        for (int j = 0; j < res[i].size(); ++j)
            res[i][j].resize(outsize[2]);
    }

    skepu::Pool3D<int> region{input, std::get<0>(poolSize), std::get<1>(poolSize), std::get<2>(poolSize), skepu::Edge::None, 0};
    
    for (size_t i = 0; i < res.size(); ++i)
        for (size_t j = 0; j < res[i].size(); ++j)
            for (size_t k = 0; k < res[i][j].size(); ++k)
            {
                region.idx = skepu::Index3D{i * std::get<0>(stride), j * std::get<1>(stride), k * std::get<2>(stride)};
                res[i][j][k] = pool_sum(region);
            }
    return res;
}

template <typename T>
void compare(skepu::Tensor3<T>& actual, vector3<T>& expected)
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

auto pool = skepu::MapPool(pool_sum);

constexpr int size_upper_bound = 25;


TEST_CASE("mappool_3d: Fundamentals")
{
    int poolSize = 2;
    pool.setPoolSizeAndStride(poolSize, poolSize, poolSize);

    for (int i = 2; i <= size_upper_bound; i += poolSize)
        for (int j = 2; j <= size_upper_bound; j += poolSize)
            for (int k = 2; k <= size_upper_bound; k += poolSize)
            {
                skepu::Tensor3<int> input(i, j, k, "input", 1);
                skepu::Tensor3<int> actual(1 + (input.size_i() - poolSize) / poolSize,
                                           1 + (input.size_j() - poolSize) / poolSize,
                                           1 + (input.size_k() - poolSize) / poolSize,
                                           "output", 0);
                vector3<int> expected;
                skepu::external(skepu::read(input), [&]{
                    int val = 0;
                    for (int ii = 0; ii < i; ++ii)
                        for (int jj = 0; jj < j; ++jj)
                            for (int kk = 0; kk < k; ++kk)
                                input(ii, jj, kk) = ++val;
                }, skepu::write(input));


                INFO("input size = " + std::to_string(input.size_i()) + "x" + std::to_string(input.size_j()) + "x" + std::to_string(input.size_k()) + 
                    "\nactual size = " + std::to_string(actual.size_i()) + "x" + std::to_string(actual.size_j()) + "x" + std::to_string(actual.size_k()));
                pool(actual, input);
                expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
                compare(actual, expected);
            }
}

TEST_CASE("mappool_3d: Stride sizes")
{
    for (int stride_i = 1; stride_i <= 5; ++stride_i)
        for (int stride_j = 1; stride_j <= 5; ++stride_j)
            for (int stride_k = 1; stride_k <= 5; ++stride_k)
            {
                int pool_size_i = stride_i * 2,
                    pool_size_j = stride_j * 2,
                    pool_size_k = stride_k * 2;
                pool.setPoolSize(pool_size_i, pool_size_j, pool_size_k);
                pool.setStride(stride_i, stride_j, stride_k);
                skepu::Tensor3<int> input(pool_size_i + stride_i, pool_size_j + stride_j, pool_size_k + stride_k, "input", 1);
                skepu::Tensor3<int> actual(1 + (input.size_i() - pool_size_i) / stride_i,
                                           1 + (input.size_j() - pool_size_j) / stride_j,
                                           1 + (input.size_k() - pool_size_k) / stride_k,
                                           "output", 0);
                vector3<int> expected;
                skepu::external(skepu::read(input), [&]{
                    int val = 0;
                    for (int i = 0; i < input.size_i(); ++i)
                        for (int j = 0; j < input.size_j(); ++j)
                            for (int k = 0; k < input.size_k(); ++k)
                                input(i, j, k) = ++val;
                }, skepu::write(input));


                INFO("input size = " + std::to_string(input.size_i()) + "x" + std::to_string(input.size_j()) + "x" + std::to_string(input.size_k()) +
                     "\nactual size = " + std::to_string(actual.size_i()) + "x" + std::to_string(actual.size_j()) + "x" + std::to_string(actual.size_k()) +
                     "\npool size = " + std::to_string(pool_size_i) + "x" + std::to_string(pool_size_j) + "x" + std::to_string(pool_size_k) +
                     "\nstride = " + std::to_string(stride_i) + "x" + std::to_string(stride_j) + "x" + std::to_string(stride_k));
                pool(actual, input);
                expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
                compare(actual, expected);
            }
}

TEST_CASE("mappool_3d: Pool sizes")
{
    int poolSizeUpperBound = 6;
    vector3<int> expected;

    for (int i = 1; i <= poolSizeUpperBound; ++i)
        for (int j = 1; j <= poolSizeUpperBound; ++j)
            for (int k = 1; k <= poolSizeUpperBound; ++k)
            {
                pool.setPoolSizeAndStride(i, j, k);
                skepu::Tensor3<int> input(i*2, j*2, k*2, "input", 1);
                skepu::Tensor3<int> actual(input.size_i() / i, input.size_j() / j, input.size_k() / k, "output", 0);
                skepu::external(skepu::read(input), [&]{
                    int val = 0;
                    for (int i = 0; i < input.size_i(); ++i)
                        for (int j = 0; j < input.size_j(); ++j)
                            for (int k = 0; k < input.size_k(); ++k)
                                input(i, j, k) = ++val;
                }, skepu::write(input));
                
                INFO("poolSize = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k));
                pool(actual, input);
                expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
                compare(actual, expected);
            }
}

TEST_CASE("mappool_3d: Invalid sizes")
{
    pool.setPoolSizeAndStride(1, 1, 1);
    std::tuple<int, int, int> poolSizeOffsetTemp = pool.getPoolSize();
    int poolSizeOffset[3];
    poolSizeOffset[0] = std::get<0>(poolSizeOffsetTemp);
    poolSizeOffset[1] = std::get<1>(poolSizeOffsetTemp);
    poolSizeOffset[2] = std::get<2>(poolSizeOffsetTemp);
    skepu::Tensor3<int> input(10, 10, 10),
    correctSize(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]),
    large_i(input.size_i()/poolSizeOffset[0]+1, input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]),
    large_j(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1]+1, input.size_k()/poolSizeOffset[2]),
    large_k(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]+1);

    CHECK_THROWS_WITH(pool(large_i, input), Catch::Matchers::Contains("input/output tensor3 size i mismatch"));
    CHECK_THROWS_WITH(pool(large_j, input), Catch::Matchers::Contains("input/output tensor3 size j mismatch"));
    CHECK_THROWS_WITH(pool(large_k, input), Catch::Matchers::Contains("input/output tensor3 size k mismatch"));
}

skepu::multiple<int, float> pool_sum_2(skepu::Pool3D<int> p)
{
    int val = 0;
    for (int i = 0; i < p.si; ++i)
        for (int j = 0; j < p.sj; ++j)
            for (int k = 0; k < p.sk; ++k)
                val += p(i, j, k);
    return skepu::ret(val, val+0.5f);
}

std::pair<vector3<int>, vector3<float>> generate_expected_2(skepu::Tensor3<int>& input, std::tuple<int, int, int> poolSize, std::tuple<int, int, int> stride)
{
    input.flush();
    vector3<int> res;
    vector3<float> res2;
    int outsize[3];
    outsize[0] = 1 + (input.size_i() - std::get<0>(poolSize)) / std::get<0>(stride);
    outsize[1] = 1 + (input.size_j() - std::get<1>(poolSize)) / std::get<1>(stride);
    outsize[2] = 1 + (input.size_k() - std::get<2>(poolSize)) / std::get<2>(stride);
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

    skepu::Pool3D<int> region{input, std::get<0>(poolSize), std::get<1>(poolSize), std::get<2>(poolSize), skepu::Edge::None, 0};
    
    for (size_t i = 0; i < res.size(); ++i)
        for (size_t j = 0; j < res[i].size(); ++j)
            for (size_t k = 0; k < res[i][j].size(); ++k)
            {
                region.idx = skepu::Index3D{i * std::get<0>(stride), j * std::get<1>(stride), k * std::get<2>(stride)};
                std::tuple<int, float> output = pool_sum_2(region);
                res[i][j][k] = std::get<0>(output);
                res2[i][j][k] = std::get<1>(output);                
            }
    return {res, res2};
}

auto pool2 = skepu::MapPool(pool_sum_2);

TEST_CASE("mappool_3d: Multi return")
{
    int poolSize = 2;
    pool2.setPoolSizeAndStride(poolSize, poolSize, poolSize);

    for (int i = 2; i <= size_upper_bound; i += poolSize)
        for (int j = 2; j <= size_upper_bound; j += poolSize)
            for (int k = 2; k <= size_upper_bound; k += poolSize)
                {
                    skepu::Tensor3<int> input(i, j, k, "input", 1);
                    skepu::Tensor3<int> actual(input.size_i() / poolSize, input.size_j() / poolSize,
                                               input.size_k() / poolSize, "output", 0);
                    skepu::Tensor3<float> actual2(input.size_i() / poolSize, input.size_j() / poolSize,
                                                  input.size_k() / poolSize, "output", 0);
                    std::pair<vector3<int>, vector3<float>> expected;
                    skepu::external(skepu::read(input), [&]{
                        int val = 0;
                        for (int i = 0; i < input.size_i(); ++i)
                            for (int j = 0; j < input.size_j(); ++j)
                                for (int k = 0; k < input.size_k(); ++k)
                                    input(i, j, k) = ++val;
                    }, skepu::write(input));


                    pool2(actual, actual2, input);
                    expected = generate_expected_2(input, pool2.getPoolSize(), pool2.getStride());
                    {
                        INFO("input size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + ", actual");
                        compare(actual, expected.first);
                    }
                    {
                        INFO("input size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + ", actual2");
                        compare(actual2, expected.second);
                    }
                }
}

TEST_CASE("mappool_3d: Invalid sizes 2 output matrices")
{
    pool2.setPoolSizeAndStride(1, 1, 1);
    std::tuple<int, int, int> poolSizeOffsetTemp = pool2.getPoolSize();
    int poolSizeOffset[3];
    poolSizeOffset[0] = std::get<0>(poolSizeOffsetTemp);
    poolSizeOffset[1] = std::get<1>(poolSizeOffsetTemp);
    poolSizeOffset[2] = std::get<2>(poolSizeOffsetTemp);
    skepu::Tensor3<int> input(10, 10, 10),
    correct_size(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]);
    skepu::Tensor3<float> small_i(input.size_i()/poolSizeOffset[0]-1, input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]),
    small_j(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1]-1, input.size_k()/poolSizeOffset[2]),
    small_k(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]-1),
    large_i(input.size_i()/poolSizeOffset[0]+1, input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]),
    large_j(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1]+1, input.size_k()/poolSizeOffset[2]),
    large_k(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]+1);

    CHECK_THROWS_WITH(pool2(correct_size, small_i, input), Catch::Matchers::Contains("invalid output tensor3 size i"));
    CHECK_THROWS_WITH(pool2(correct_size, small_j, input), Catch::Matchers::Contains("invalid output tensor3 size j"));
    CHECK_THROWS_WITH(pool2(correct_size, small_k, input), Catch::Matchers::Contains("invalid output tensor3 size k"));
    CHECK_THROWS_WITH(pool2(correct_size, large_i, input), Catch::Matchers::Contains("invalid output tensor3 size i"));
    CHECK_THROWS_WITH(pool2(correct_size, large_j, input), Catch::Matchers::Contains("invalid output tensor3 size j"));
    CHECK_THROWS_WITH(pool2(correct_size, large_k, input), Catch::Matchers::Contains("invalid output tensor3 size k"));
}
}


// ======== mappool_4d ========

namespace mappool_4d_part
{
template <typename T>
using vector4 = std::vector<std::vector<std::vector<std::vector<T>>>>;

int pool_sum(skepu::Pool4D<int> p)
{
    int val = 0;
    for (int i = 0; i < p.si; ++i)
        for (int j = 0; j < p.sj; ++j)
            for (int k = 0; k < p.sk; ++k)
                for (int l = 0; l < p.sl; ++l)
                    val += p(i, j, k, l);
    return val;
}

vector4<int> generate_expected(skepu::Tensor4<int>& input, std::tuple<int, int, int, int> poolSize, std::tuple<int, int, int, int> stride)
{
    input.flush();
    vector4<int> res;
    int outsize[4];
    outsize[0] = 1 + (input.size_i() - std::get<0>(poolSize)) / std::get<0>(stride);
    outsize[1] = 1 + (input.size_j() - std::get<1>(poolSize)) / std::get<1>(stride);
    outsize[2] = 1 + (input.size_k() - std::get<2>(poolSize)) / std::get<2>(stride);
    outsize[3] = 1 + (input.size_l() - std::get<3>(poolSize)) / std::get<3>(stride);
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

    skepu::Pool4D<int> region{input, std::get<0>(poolSize), std::get<1>(poolSize), std::get<2>(poolSize), std::get<3>(poolSize), skepu::Edge::None, 0};
    
    for (size_t i = 0; i < res.size(); ++i)
        for (size_t j = 0; j < res[i].size(); ++j)
            for (size_t k = 0; k < res[i][j].size(); ++k)
                for (size_t l = 0; l < res[i][j][k].size(); ++l)
                {
                    region.idx = skepu::Index4D{i * std::get<0>(stride), j * std::get<1>(stride), k * std::get<2>(stride), l * std::get<3>(stride)};
                    res[i][j][k][l] = pool_sum(region);
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

auto pool = skepu::MapPool(pool_sum);
constexpr int size_upper_bound = 12;


TEST_CASE("mappool_4d: Fundamentals")
{
    int poolSize = 2;
    pool.setPoolSizeAndStride(poolSize, poolSize, poolSize, poolSize);

    for (int i = 2; i <= size_upper_bound; i += poolSize)
        for (int j = 2; j <= size_upper_bound; j += poolSize)
            for (int k = 2; k <= size_upper_bound; k += poolSize)
                for (int l = 2; l <= size_upper_bound; l += poolSize)
        {
            skepu::Tensor4<int> input(i, j, k, l, "input", 1);
            skepu::Tensor4<int> actual(1 + (input.size_i() - poolSize) / poolSize,
                                       1 + (input.size_j() - poolSize) / poolSize,
                                       1 + (input.size_k() - poolSize) / poolSize,
                                       1 + (input.size_l() - poolSize) / poolSize,
                                       "output", 0);
            vector4<int> expected;
            skepu::external(skepu::read(input), [&]{
                int val = 0;
                for (int ii = 0; ii < i; ++ii)
                    for (int jj = 0; jj < j; ++jj)
                        for (int kk = 0; kk < k; ++kk)
                            for (int ll = 0; ll < l; ++ll)
                                input(ii, jj, kk, ll) = ++val;
            }, skepu::write(input));


            INFO("input size = " + std::to_string(input.size_i()) + "x" + std::to_string(input.size_j()) + "x" + std::to_string(input.size_k()) + "x" + std::to_string(input.size_l()) + 
                 "\nactual size = " + std::to_string(actual.size_i()) + "x" + std::to_string(actual.size_j()) + "x" + std::to_string(actual.size_k()) + "x" + std::to_string(actual.size_l()));
            pool(actual, input);
            expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
            compare(actual, expected);
        }
}

TEST_CASE("mappool_4d: Stride sizes")
{
    for (int stride_i = 1; stride_i <= 5; ++stride_i)
        for (int stride_j = 1; stride_j <= 5; ++stride_j)
            for (int stride_k = 1; stride_k <= 5; ++stride_k)
                for (int stride_l = 1; stride_l <= 5; ++stride_l)
                {
                    int pool_size_i = stride_i * 2,
                        pool_size_j = stride_j * 2,
                        pool_size_k = stride_k * 2,
                        pool_size_l = stride_l * 2;
                    pool.setPoolSize(pool_size_i, pool_size_j, pool_size_k, pool_size_l);
                    pool.setStride(stride_i, stride_j, stride_k, stride_l);
                    skepu::Tensor4<int> input(pool_size_i + stride_i, pool_size_j + stride_j, pool_size_k + stride_k, pool_size_l + stride_l, "input", 1);
                    skepu::Tensor4<int> actual(1 + (input.size_i() - pool_size_i) / stride_i,
                                               1 + (input.size_j() - pool_size_j) / stride_j,
                                               1 + (input.size_k() - pool_size_k) / stride_k,
                                               1 + (input.size_l() - pool_size_l) / stride_l, "output", 0);
                    vector4<int> expected;
                    skepu::external(skepu::read(input), [&]{
                        int val = 0;
                        for (int i = 0; i < input.size_i(); ++i)
                            for (int j = 0; j < input.size_j(); ++j)
                                for (int k = 0; k < input.size_k(); ++k)
                                    for (int l = 0; l < input.size_l(); ++l)
                                        input(i, j, k, l) = ++val;
                    }, skepu::write(input));


                    INFO("input size = " + std::to_string(input.size_i()) + "x" + std::to_string(input.size_j()) + "x" + std::to_string(input.size_k()) + "x" + std::to_string(input.size_l()) + 
                         "\nactual size = " + std::to_string(actual.size_i()) + "x" + std::to_string(actual.size_j()) + "x" + std::to_string(actual.size_k()) + "x" + std::to_string(actual.size_l()) +
                         "\npool size = " + std::to_string(pool_size_i) + "x" + std::to_string(pool_size_j) + "x" + std::to_string(pool_size_k) + "x" + std::to_string(pool_size_l) +
                         "\nstride = " + std::to_string(stride_i) + "x" + std::to_string(stride_j) + "x" + std::to_string(stride_k) + "x" + std::to_string(stride_l));
                    pool(actual, input);
                    expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
                    compare(actual, expected);
                }
}

TEST_CASE("mappool_4d: Pool sizes")
{
    int poolSizeUpperBound = 6;
    vector4<int> expected;

    for (int i = 1; i <= poolSizeUpperBound; ++i)
        for (int j = 1; j <= poolSizeUpperBound; ++j)
            for (int k = 1; k <= poolSizeUpperBound; ++k)
                for (int l = 1; l <= poolSizeUpperBound; ++l)
                {
                    pool.setPoolSizeAndStride(i, j, k, l);
                    skepu::Tensor4<int> input(i*2, j*2, k*2, l*2, "input", 1);
                    skepu::Tensor4<int> actual(input.size_i() / i, input.size_j() / j, input.size_k() / k, input.size_l() / l, "output", 0);
                    skepu::external(skepu::read(input), [&]{
                        int val = 0;
                        for (int i = 0; i < input.size_i(); ++i)
                            for (int j = 0; j < input.size_j(); ++j)
                                for (int k = 0; k < input.size_k(); ++k)
                                    for (int l = 0; l < input.size_l(); ++l)
                                        input(i, j, k, l) = ++val;
                    }, skepu::write(input));
                    
                    INFO("poolSize = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l));
                    pool(actual, input);
                    expected = generate_expected(input, pool.getPoolSize(), pool.getStride());
                    compare(actual, expected);
                }
}

TEST_CASE("mappool_4d: Invalid sizes")
{
    pool.setPoolSizeAndStride(1, 1, 1, 1);
    std::tuple<int, int, int, int> poolSizeOffsetTemp = pool.getPoolSize();
    int poolSizeOffset[4];
    poolSizeOffset[0] = std::get<0>(poolSizeOffsetTemp);
    poolSizeOffset[1] = std::get<1>(poolSizeOffsetTemp);
    poolSizeOffset[2] = std::get<2>(poolSizeOffsetTemp);
    poolSizeOffset[3] = std::get<3>(poolSizeOffsetTemp);
    skepu::Tensor4<int> input(10, 10, 10, 10),
    correctSize(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]),
    large_i(input.size_i()/poolSizeOffset[0]+1, input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]),
    large_j(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1]+1, input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]),
    large_k(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]+1, input.size_l()/poolSizeOffset[3]),
    large_l(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]+1);

    CHECK_THROWS_WITH(pool(large_i, input), Catch::Matchers::Contains("input/output tensor4 size i mismatch"));
    CHECK_THROWS_WITH(pool(large_j, input), Catch::Matchers::Contains("input/output tensor4 size j mismatch"));
    CHECK_THROWS_WITH(pool(large_k, input), Catch::Matchers::Contains("input/output tensor4 size k mismatch"));
    CHECK_THROWS_WITH(pool(large_l, input), Catch::Matchers::Contains("input/output tensor4 size l mismatch"));
}

skepu::multiple<int, float> pool_sum_2(skepu::Pool4D<int> p)
{
    int val = 0;
    for (int i = 0; i < p.si; ++i)
        for (int j = 0; j < p.sj; ++j)
            for (int k = 0; k < p.sk; ++k)
                for (int l = 0; l < p.sl; ++l)
                    val += p(i, j, k, l);
    return skepu::ret(val, val+0.5f);
}

std::pair<vector4<int>, vector4<float>> generate_expected_2(skepu::Tensor4<int>& input, std::tuple<int, int, int, int> poolSize, std::tuple<int, int, int, int> stride)
{
    input.flush();
    vector4<int> res;
    vector4<float> res2;
    int outsize[4];
    outsize[0] = 1 + (input.size_i() - std::get<0>(poolSize)) / std::get<0>(stride);
    outsize[1] = 1 + (input.size_j() - std::get<1>(poolSize)) / std::get<1>(stride);
    outsize[2] = 1 + (input.size_k() - std::get<2>(poolSize)) / std::get<2>(stride);
    outsize[3] = 1 + (input.size_l() - std::get<3>(poolSize)) / std::get<3>(stride);
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

    skepu::Pool4D<int> region{input, std::get<0>(poolSize), std::get<1>(poolSize), std::get<2>(poolSize), std::get<3>(poolSize), skepu::Edge::None, 0};
    
    for (size_t i = 0; i < res.size(); ++i)
        for (size_t j = 0; j < res[i].size(); ++j)
            for (size_t k = 0; k < res[i][j].size(); ++k)
                for (size_t l = 0; l < res[i][j][k].size(); ++l)
                {
                    region.idx = skepu::Index4D{i * std::get<0>(stride), j * std::get<1>(stride), k * std::get<2>(stride), l * std::get<3>(stride)};
                    std::tuple<int, float> output = pool_sum_2(region);
                    res[i][j][k][l] = std::get<0>(output);
                    res2[i][j][k][l] = std::get<1>(output);                
                }
    return {res, res2};
}

auto pool2 = skepu::MapPool(pool_sum_2);

TEST_CASE("mappool_4d: Multi return")
{
    int poolSize = 2;
    pool2.setPoolSizeAndStride(poolSize, poolSize, poolSize, poolSize);

    for (int i = 2; i <= size_upper_bound; i += poolSize)
        for (int j = 2; j <= size_upper_bound; j += poolSize)
            for (int k = 2; k <= size_upper_bound; k += poolSize)
                for (int l = 2; l <= size_upper_bound; l += poolSize)
                {
                    skepu::Tensor4<int> input(i, j, k, l, "input", 1);
                    skepu::Tensor4<int> actual(input.size_i() / poolSize, input.size_j() / poolSize,
                                               input.size_k() / poolSize, input.size_l() / poolSize, "output", 0);
                    skepu::Tensor4<float> actual2(input.size_i() / poolSize, input.size_j() / poolSize,
                                                  input.size_k() / poolSize, input.size_l() / poolSize, "output", 0);
                    std::pair<vector4<int>, vector4<float>> expected;
                    skepu::external(skepu::read(input), [&]{
                        int val = 0;
                        for (int i = 0; i < input.size_i(); ++i)
                            for (int j = 0; j < input.size_j(); ++j)
                                for (int k = 0; k < input.size_k(); ++k)
                                    for (int l = 0; l < input.size_l(); ++l)
                                        input(i, j, k, l) = ++val;
                    }, skepu::write(input));


                    pool2(actual, actual2, input);
                    expected = generate_expected_2(input, pool2.getPoolSize(), pool2.getStride());
                    {
                        INFO("input size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l) + ", actual");
                        compare(actual, expected.first);
                    }
                    {
                        INFO("input size = " + std::to_string(i) + "x" + std::to_string(j) + "x" + std::to_string(k) + "x" + std::to_string(l) + ", actual2");
                        compare(actual2, expected.second);
                    }
                }
}

TEST_CASE("mappool_4d: Invalid sizes 2 output matrices")
{
    pool2.setPoolSizeAndStride(1, 1, 1, 1);
    std::tuple<int, int, int, int> poolSizeOffsetTemp = pool2.getPoolSize();
    int poolSizeOffset[4];
    poolSizeOffset[0] = std::get<0>(poolSizeOffsetTemp);
    poolSizeOffset[1] = std::get<1>(poolSizeOffsetTemp);
    poolSizeOffset[2] = std::get<2>(poolSizeOffsetTemp);
    poolSizeOffset[3] = std::get<3>(poolSizeOffsetTemp);
    skepu::Tensor4<int> input(10, 10, 10, 10),
    correct_size(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]);
    skepu::Tensor4<float> small_i(input.size_i()/poolSizeOffset[0]-1, input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]),
    small_j(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1]-1, input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]),
    small_k(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]-1, input.size_l()/poolSizeOffset[3]),
    small_l(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]-1),
    large_i(input.size_i()/poolSizeOffset[0]+1, input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]),
    large_j(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1]+1, input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]),
    large_k(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2]+1, input.size_l()/poolSizeOffset[3]),
    large_l(input.size_i()/poolSizeOffset[0], input.size_j()/poolSizeOffset[1], input.size_k()/poolSizeOffset[2], input.size_l()/poolSizeOffset[3]+1);

    CHECK_THROWS_WITH(pool2(correct_size, small_i, input), Catch::Matchers::Contains("invalid output tensor4 size i"));
    CHECK_THROWS_WITH(pool2(correct_size, small_j, input), Catch::Matchers::Contains("invalid output tensor4 size j"));
    CHECK_THROWS_WITH(pool2(correct_size, small_k, input), Catch::Matchers::Contains("invalid output tensor4 size k"));
    CHECK_THROWS_WITH(pool2(correct_size, small_l, input), Catch::Matchers::Contains("invalid output tensor4 size l"));
    CHECK_THROWS_WITH(pool2(correct_size, large_i, input), Catch::Matchers::Contains("invalid output tensor4 size i"));
    CHECK_THROWS_WITH(pool2(correct_size, large_j, input), Catch::Matchers::Contains("invalid output tensor4 size j"));
    CHECK_THROWS_WITH(pool2(correct_size, large_k, input), Catch::Matchers::Contains("invalid output tensor4 size k"));
    CHECK_THROWS_WITH(pool2(correct_size, large_l, input), Catch::Matchers::Contains("invalid output tensor4 size l"));
}
}


// ======== mappool_2d_small_output ========

namespace mappool_2d_small_output_part
{
int poolSum(skepu::Pool2D<int> p)
{
    int val = 0;
    for (int i = 0; i < p.si; ++i)
        for (int j = 0; j < p.sj; ++j)
            val += p(i, j);
    return val;
}

std::vector<std::vector<int>> generate_expected(skepu::Matrix<int>& input, skepu::Matrix<int>& actual, std::tuple<int, int> poolSize, std::tuple<int, int> stride)
{
    input.flush();
    std::vector<std::vector<int>> res;
    int out_size[2];
    out_size[0] = actual.size_i();
    out_size[1] = actual.size_j();
    res.resize(out_size[0]);
    for (int i = 0; i < res.size(); ++i)
        res[i].resize(out_size[1]);

    skepu::Pool2D<int> region{input, std::get<0>(poolSize), std::get<1>(poolSize), skepu::Edge::None, 0};
    
    for (size_t i = 0; i < res.size(); ++i)
        for (size_t j = 0; j < res[i].size(); ++j)
        {
            region.idx = skepu::Index2D{i * std::get<0>(stride), j * std::get<1>(stride)};
            res[i][j] = poolSum(region);
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

auto pool = skepu::MapPool(poolSum);

constexpr int size_upper_bound = 40;
constexpr int size_lower_bound = 40;

TEST_CASE("mappool_2d_small_output: Fundamentals")
{
    for (int in_size_i = size_lower_bound; in_size_i <= size_upper_bound; ++in_size_i)
        for (int in_size_j = size_lower_bound; in_size_j <= size_upper_bound; ++in_size_j)
        {
            skepu::Matrix<int> input(in_size_i, in_size_j, 1);
            skepu::external(skepu::read(input), [&]{
                int val = 0;
                for (int i = 0; i < input.size_i(); ++i)
                    for (int j = 0; j < input.size_j(); ++j)
                        input(i, j) = ++val;
            }, skepu::write(input));

            for (int pool_size = 2; pool_size <= std::min(in_size_i, in_size_j); ++pool_size)
            {
                pool.setPoolSize(pool_size, pool_size);
                for (int stride = 1; stride <= pool_size; ++stride)
                {
                    pool.setStride(stride);
                    for (int out_size_i : out_sizes((input.size_i() - pool_size) / stride + 1))
                        for (int out_size_j : out_sizes((input.size_j() - pool_size) / stride + 1))
                        {
                            skepu::Matrix<int> actual(out_size_i, out_size_j, 0);
                            std::vector<std::vector<int>> expected;

                            INFO("input size = " + std::to_string(in_size_i) + "x" + std::to_string(in_size_j) +
                                "\noutput size = " + std::to_string(out_size_i) + "x" + std::to_string(out_size_j) +
                                "\npool size = " + std::to_string(pool_size) + 
                                "\nstride = " + std::to_string(stride));
                            pool(actual, input);
                            expected = generate_expected(input, actual, pool.getPoolSize(), pool.getStride());
                            compare(actual, expected);
                        }
                }
            }   
        }
}
}


// ======== mappool_3d_small_output ========

namespace mappool_3d_small_output_part
{
template <typename T>
using vector3 = std::vector<std::vector<std::vector<T>>>;

int poolSum(skepu::Pool3D<int> p)
{
    int val = 0;
    for (int i = 0; i < p.si; ++i)
        for (int j = 0; j < p.sj; ++j)
            for (int k = 0; k < p.sk; ++k)
                val += p(i, j, k);
    return val;
}

vector3<int> generate_expected(skepu::Tensor3<int>& input, skepu::Tensor3<int>& actual, std::tuple<int, int, int> poolSize, std::tuple<int, int, int> stride)
{
    input.flush();
    vector3<int> res;
    int out_size[3];
    out_size[0] = actual.size_i();
    out_size[1] = actual.size_j();
    out_size[2] = actual.size_k();
    res.resize(out_size[0]);
    for (int i = 0; i < res.size(); ++i)
    {
        res[i].resize(out_size[1]);
        for (int j = 0; j < res[i].size(); ++j)
            res[i][j].resize(out_size[2]);
    }

    skepu::Pool3D<int> region{input, std::get<0>(poolSize), std::get<1>(poolSize), std::get<2>(poolSize), skepu::Edge::None, 0};
    
    for (size_t i = 0; i < res.size(); ++i)
        for (size_t j = 0; j < res[i].size(); ++j)
            for (size_t k = 0; k < res[i][j].size(); ++k)
            {
                region.idx = skepu::Index3D{i * std::get<0>(stride), j * std::get<1>(stride), k * std::get<2>(stride)};
                res[i][j][k] = poolSum(region);
            }
    return res;
}

void compare(skepu::Tensor3<int>& actual, vector3<int>& expected)
{
    actual.flush();
    for (int i = 0; i < actual.size_i(); ++i)
        for (int j = 0; j < actual.size_j(); ++j)
            for (int k = 0; k < actual.size_k(); ++k)
            {
                INFO("Checking at (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + "), comparing " + std::to_string(actual(i, j, k)) + " and " + std::to_string(expected[i][j][k]));
                CHECK(actual(i, j, k) == expected[i][j][k]);
            }
}

auto pool = skepu::MapPool(poolSum);

constexpr int size_lower_bound = 11;
constexpr int size_upper_bound = 12;

TEST_CASE("mappool_3d_small_output: Fundamentals")
{
    for (int in_size_i = size_lower_bound; in_size_i <= size_upper_bound; ++in_size_i)
        for (int in_size_j = size_lower_bound; in_size_j <= size_upper_bound; ++in_size_j)
            for (int in_size_k = size_lower_bound; in_size_k <= size_upper_bound; ++in_size_k)
            {
                skepu::Tensor3<int> input(in_size_i, in_size_j, in_size_k, "input", 1);
                skepu::external(skepu::read(input), [&]{
                    int val = 0;
                    for (int i = 0; i < input.size_i(); ++i)
                        for (int j = 0; j < input.size_j(); ++j)
                            for (int k = 0; k < input.size_k(); ++k)
                                input(i, j, k) = ++val;
                }, skepu::write(input));

                for (int pool_size = 2; pool_size <= std::min({in_size_i, in_size_j, in_size_k}); ++pool_size)
                {
                    pool.setPoolSize(pool_size, pool_size, pool_size);
                    for (int stride = 1; stride <= pool_size; ++stride)
                    {
                        pool.setStride(stride);
                        for (int out_size_i : out_sizes((input.size_i() - pool_size) / stride + 1))
                            for (int out_size_j : out_sizes((input.size_j() - pool_size) / stride + 1))
                                for (int out_size_k : out_sizes((input.size_k() - pool_size) / stride + 1))
                                {
                                    skepu::Tensor3<int> actual(out_size_i, out_size_j, out_size_k, "actual", 0);
                                    vector3<int> expected;

                                    INFO("input size = " + std::to_string(in_size_i) + "x" + std::to_string(in_size_j) + "x" + std::to_string(in_size_k) +
                                        "\noutput size = " + std::to_string(out_size_i) + "x" + std::to_string(out_size_j) + "x" + std::to_string(out_size_k) +
                                        "\npool size = " + std::to_string(pool_size) + 
                                        "\nstride = " + std::to_string(stride));
                                    pool(actual, input);
                                    expected = generate_expected(input, actual, pool.getPoolSize(), pool.getStride());
                                    compare(actual, expected);
                                }
                    }
                }   
            }
}
}


// ======== mappool_4d_small_output ========

namespace mappool_4d_small_output_part
{
template <typename T>
using vector4 = std::vector<std::vector<std::vector<std::vector<T>>>>;


int poolSum(skepu::Pool4D<int> p)
{
    int val = 0;
    for (int i = 0; i < p.si; ++i)
        for (int j = 0; j < p.sj; ++j)
            for (int k = 0; k < p.sk; ++k)
                for (int l = 0; l < p.sl; ++l)
                    val += p(i, j, k, l);
    return val;
}

vector4<int> generate_expected(skepu::Tensor4<int>& input, skepu::Tensor4<int>& actual, std::tuple<int, int, int, int> poolSize, std::tuple<int, int, int, int> stride)
{
    input.flush();
    vector4<int> res;
    int out_size[4];
    out_size[0] = actual.size_i();
    out_size[1] = actual.size_j();
    out_size[2] = actual.size_k();
    out_size[3] = actual.size_l();
    res.resize(out_size[0]);
    for (int i = 0; i < res.size(); ++i)
    {
        res[i].resize(out_size[1]);
        for (int j = 0; j < res[i].size(); ++j)
        {
            res[i][j].resize(out_size[2]);
            for (int k = 0; k < res[i][j].size(); ++k)
                res[i][j][k].resize(out_size[3]);
        }
    }

    skepu::Pool4D<int> region{input, std::get<0>(poolSize), std::get<1>(poolSize), std::get<2>(poolSize), std::get<3>(poolSize), skepu::Edge::None, 0};
    
    for (size_t i = 0; i < res.size(); ++i)
        for (size_t j = 0; j < res[i].size(); ++j)
            for (size_t k = 0; k < res[i][j].size(); ++k)
                for (size_t l = 0; l < res[i][j][k].size(); ++l)
                {
                    region.idx = skepu::Index4D{i * std::get<0>(stride), j * std::get<1>(stride), k * std::get<2>(stride), l * std::get<3>(stride)};
                    res[i][j][k][l] = poolSum(region);
                }
    return res;
}

void compare(skepu::Tensor4<int>& actual, vector4<int>& expected)
{
    actual.flush();
    for (int i = 0; i < actual.size_i(); ++i)
        for (int j = 0; j < actual.size_j(); ++j)
            for (int k = 0; k < actual.size_k(); ++k)
                for (int l = 0; l < actual.size_l(); ++l)
                {
                    INFO("Checking at (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ", " + std::to_string(l) + "), comparing " + std::to_string(actual(i, j, k, l)) + " and " + std::to_string(expected[i][j][k][l]));
                    CHECK(actual(i, j, k, l) == expected[i][j][k][l]);
                }
}

auto pool = skepu::MapPool(poolSum);

// At 5 this is really fast
// At 6 this takes a around 0.5 seconds on sequential, around 3.5 on OpenMP, and around 2-2.5 on OpenCL/CUDA
// At 7 this takes around 3 seconds on sequential, around 11 on OpenMP, and around 7-9 on OpenCL/CUDA
// At 10 this takes several minutes
constexpr int size_upper_bound = 6;

TEST_CASE("mappool_4d_small_output: Fundamentals")
{
    for (int in_size_i = 1; in_size_i <= size_upper_bound; ++in_size_i)
        for (int in_size_j = 1; in_size_j <= size_upper_bound; ++in_size_j)
            for (int in_size_k = 1; in_size_k <= size_upper_bound; ++in_size_k)
                for (int in_size_l = 1; in_size_l <= size_upper_bound; ++in_size_l)
                {
                    skepu::Tensor4<int> input(in_size_i, in_size_j, in_size_k, in_size_l, "input", 1);
                    skepu::external(skepu::read(input), [&]{
                        int val = 0;
                        for (int i = 0; i < input.size_i(); ++i)
                            for (int j = 0; j < input.size_j(); ++j)
                                for (int k = 0; k < input.size_k(); ++k)
                                    for (int l = 0; l < input.size_l(); ++l)
                                        input(i, j, k, l) = ++val;
                    }, skepu::write(input));

                    for (int pool_size = 2; pool_size <= std::min({in_size_i, in_size_j, in_size_k, in_size_l}); ++pool_size)
                    {
                        pool.setPoolSize(pool_size, pool_size, pool_size, pool_size);
                        for (int stride = 1; stride <= pool_size; ++stride)
                        {
                            pool.setStride(stride);
                            for (int out_size_i : out_sizes((input.size_i() - pool_size) / stride + 1))
                                for (int out_size_j : out_sizes((input.size_j() - pool_size) / stride + 1))
                                    for (int out_size_k : out_sizes((input.size_k() - pool_size) / stride + 1))
                                        for (int out_size_l : out_sizes((input.size_l() - pool_size) / stride + 1))
                                        {
                                            skepu::Tensor4<int> actual(out_size_i, out_size_j, out_size_k, out_size_l, "output", 0);
                                            vector4<int> expected;

                                            INFO("input size = " + std::to_string(in_size_i) + "x" + std::to_string(in_size_j) + "x" + std::to_string(in_size_k) + "x" + std::to_string(in_size_l) +
                                                "\noutput size = " + std::to_string(out_size_i) + "x" + std::to_string(out_size_j) + "x" + std::to_string(out_size_k) + "x" + std::to_string(out_size_l) +
                                                "\npool size = " + std::to_string(pool_size) +
                                                "\nstride = " + std::to_string(stride));
                                            pool(actual, input);
                                            expected = generate_expected(input, actual, pool.getPoolSize(), pool.getStride());
                                            compare(actual, expected);
                                        }
                        }
                    }   
                }
}
}
