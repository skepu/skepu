#define SKEPU_ENABLE_EXCEPTIONS
#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>

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

TEST_CASE("Fundamentals")
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
                            for (int out_size_i = 1; out_size_i <= (input.size_i() - pool_size) / stride + 1; ++out_size_i)
                                for (int out_size_j = 1; out_size_j <= (input.size_j() - pool_size) / stride + 1; ++out_size_j)
                                    for (int out_size_k = 1; out_size_k <= (input.size_k() - pool_size) / stride + 1; ++out_size_k)
                                        for (int out_size_l = 1; out_size_l <= (input.size_l() - pool_size) / stride + 1; ++out_size_l)
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