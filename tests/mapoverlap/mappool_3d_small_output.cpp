#define SKEPU_ENABLE_EXCEPTIONS
#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/io.hpp>

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

TEST_CASE("Fundamentals")
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
                        for (int out_size_i = 1; out_size_i <= (input.size_i() - pool_size) / stride + 1; ++out_size_i)
                            for (int out_size_j = 1; out_size_j <= (input.size_j() - pool_size) / stride + 1; ++out_size_j)
                                for (int out_size_k = 1; out_size_k <= (input.size_k() - pool_size) / stride + 1; ++out_size_k)
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