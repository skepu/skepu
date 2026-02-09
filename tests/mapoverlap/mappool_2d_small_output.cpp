#define SKEPU_ENABLE_EXCEPTIONS
#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/io.hpp>

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

constexpr int size_upper_bound = 20;

TEST_CASE("Fundamentals")
{
    for (int in_size_i = 1; in_size_i <= size_upper_bound; ++in_size_i)
        for (int in_size_j = 1; in_size_j <= size_upper_bound; ++in_size_j)
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
                    for (int out_size_i = 1; out_size_i <= (input.size_i() - pool_size) / stride + 1; ++out_size_i)
                        for (int out_size_j = 1; out_size_j <= (input.size_j() - pool_size) / stride + 1; ++out_size_j)
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