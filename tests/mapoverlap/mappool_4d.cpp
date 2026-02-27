#define SKEPU_ENABLE_EXCEPTIONS
#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/io.hpp>
#include <cmath>
#include <type_traits>

template <typename T>
using vector4 = std::vector<std::vector<std::vector<std::vector<T>>>>;

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

TEST_CASE("Fundamentals")
{
    int poolSize = 2;
    pool.setPoolSizeAndStride(poolSize, poolSize, poolSize, poolSize);

    for (int i = 2; i <= 20; i += poolSize)
        for (int j = 2; j <= 20; j += poolSize)
            for (int k = 2; k <= 20; k += poolSize)
                for (int l = 2; l <= 20; l += poolSize)
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

TEST_CASE("Stride sizes")
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

TEST_CASE("Pool sizes")
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

TEST_CASE("Invalid sizes")
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

TEST_CASE("Multi return")
{
    int poolSize = 2;
    pool2.setPoolSizeAndStride(poolSize, poolSize, poolSize, poolSize);

    for (int i = 2; i <= 20; i += poolSize)
        for (int j = 2; j <= 20; j += poolSize)
            for (int k = 2; k <= 20; k += poolSize)
                for (int l = 2; l <= 20; l += poolSize)
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

TEST_CASE("Invalid sizes 2 output matrices")
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