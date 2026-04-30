#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/io.hpp>

const int size = 10;

int overlap_1d(skepu::Index1D index, skepu::Region1D<int> r)
{
    return index.i;
}

int overlap_2d(skepu::Index2D index, skepu::Region2D<int> r, int size)
{
    return index.row * size + index.col;
}

int overlap_3d(skepu::Index3D index, skepu::Region3D<int> r, int size)
{
    return index.i * size * size + index.j * size + index.k;
}

int overlap_4d(skepu::Index4D index, skepu::Region4D<int> r, int size)
{
    return index.i * size * size * size + index.j * size * size + index.k * size + index.l;
}

auto skel_1d = skepu::MapOverlap(overlap_1d);
auto skel_2d = skepu::MapOverlap(overlap_2d);
auto skel_3d = skepu::MapOverlap(overlap_3d);
auto skel_4d = skepu::MapOverlap(overlap_4d);

TEST_CASE("MapOverlap 1D")
{
    skepu::Vector<int> input(size), output(size);
    skel_1d.setOverlap(1); // The wrap memory used for OpenCL will have a size of 0 otherwise
    skel_1d.setEdgeMode(skepu::Edge::Duplicate);
    skel_1d(output, input);
    skepu::external(skepu::read(output), [&]{
        int expected = 0;
        for (int i = 0; i < size; ++i)
            {
                INFO("MapOverlap1D, index: " + std::to_string(i));
                CHECK(output(i) == expected++);
            }
    });

    skel_1d.setEdgeMode(skepu::Edge::None);
    skepu::Vector<int> output_none(size - 2);
    skel_1d(output_none, input);
    skepu::external(skepu::read(output_none), [&]{
        int expected = 0;
        for (int i = 0; i < output_none.size(); ++i)
            {
                INFO("MapOverlap1D none, index: " + std::to_string(i));
                CHECK(output_none(i) == expected++);
            }
    });
}

TEST_CASE("MapOverlap 1D RowWise")
{
    skepu::Matrix<int> input(size, size), output(size, size);
    skel_1d.setOverlap(1);
    skel_1d.setEdgeMode(skepu::Edge::Duplicate);
    skel_1d.setOverlapMode(skepu::Overlap::RowWise);
    skel_1d(output, input);
    skepu::external(skepu::read(output), [&]{
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
            {
                INFO("MapOverlap2D RoWwise, index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output(i, j) == j);
            }
    });

    skel_1d.setOverlap(1);
    skel_1d.setEdgeMode(skepu::Edge::None);
    skepu::Matrix<int> output_none(size, size - 2);
    skel_1d(output_none, input);
    skepu::external(skepu::read(output_none), [&]{
        for (int i = 0; i < output_none.size_i(); ++i)
            for (int j = 0; j < output_none.size_j(); ++j)
            {
                INFO("MapOverlap2D RoWwise none , index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output_none(i, j) == j);
            }
    });
}

TEST_CASE("MapOverlap 1D ColWise")
{
    skepu::Matrix<int> input(size, size), output(size, size);
    skel_1d.setOverlap(1);
    skel_1d.setEdgeMode(skepu::Edge::Duplicate);
    skel_1d.setOverlapMode(skepu::Overlap::ColWise);
    skel_1d(output, input);
    skepu::external(skepu::read(output), [&]{
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
            {
                INFO("MapOverlap2D ColWise, index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output(i, j) == i);
            }
    });

    skel_1d.setOverlap(1);
    skel_1d.setEdgeMode(skepu::Edge::None);
    skepu::Matrix<int> output_none(size - 2, size);
    skel_1d(output_none, input);
    skepu::external(skepu::read(output_none), [&]{
        for (int i = 0; i < output_none.size_i(); ++i)
            for (int j = 0; j < output_none.size_j(); ++j)
            {
                INFO("MapOverlap1D ColWise none , index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output_none(i, j) == i);
            }
    });
}

TEST_CASE("MapOverlap 2D")
{
    skepu::Matrix<int> input(size, size), output(size, size);
    skel_2d.setOverlap(0, 0);
    skel_2d.setEdgeMode(skepu::Edge::Duplicate);
    skel_2d(output, input, size);
    skepu::external(skepu::read(output), [&]{
        int expected = 0;
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
            {
                INFO("MapOverlap2D, index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output(i, j) == expected++);
            }
    });

    skel_2d.setOverlap(1, 1);
    skel_2d.setEdgeMode(skepu::Edge::None);
    skepu::Matrix<int> output_none(size - 2, size - 2);
    skel_2d(output_none, input, size - 2);
    skepu::external(skepu::read(output_none), [&]{
        int expected = 0;
        for (int i = 0; i < output_none.size_i(); ++i)
            for (int j = 0; j < output_none.size_j(); ++j)
            {
                INFO("MapOverlap2D none , index: (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                CHECK(output_none(i, j) == expected++);
            }
    });
}

TEST_CASE("MapOverlap 3D")
{
    skepu::Tensor3<int> input(size, size, size), output(size, size, size);
    skel_3d.setOverlap(0, 0, 0);
    skel_3d.setEdgeMode(skepu::Edge::Duplicate);
    skel_3d(output, input, size);
    skepu::external(skepu::read(output), [&]{
        int expected = 0;
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                for (int k = 0; k < size; ++k)
                {
                    INFO("MapOverlap3D, index: (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ")");
                    CHECK(output(i, j, k) == expected++);
                }
    });

    skel_3d.setOverlap(1, 1, 1);
    skel_3d.setEdgeMode(skepu::Edge::None);
    skepu::Tensor3<int> output_none(size - 2, size - 2, size - 2);
    skel_3d(output_none, input, size - 2);
    skepu::external(skepu::read(output_none), [&]{
        int expected = 0;
        for (int i = 0; i < output_none.size_i(); ++i)
            for (int j = 0; j < output_none.size_j(); ++j)
                for (int k = 0; k < output_none.size_k(); ++k)
                {
                    INFO("MapOverlap3D none, index: (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ")");
                    CHECK(output_none(i, j, k) == expected++);
                }
    });
}

TEST_CASE("MapOverlap 4D")
{
    skepu::Tensor4<int> input(size, size, size, size), output(size, size, size, size);
    skel_4d.setOverlap(0, 0, 0, 0);
    skel_4d.setEdgeMode(skepu::Edge::Duplicate);
    skel_4d(output, input, size);
    skepu::external(skepu::read(output), [&]{
        int expected = 0;
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                for (int k = 0; k < size; ++k)
                    for (int l = 0; l < size; ++l)
                    {
                        INFO("MapOverlap4D, index: (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ", " + std::to_string(l) + ")");
                        CHECK(output(i, j, k, l) == expected++);
                    }
    });

    skel_4d.setOverlap(1, 1, 1, 1);
    skel_4d.setEdgeMode(skepu::Edge::None);
    skepu::Tensor4<int> output_none(size - 2, size - 2, size - 2, size - 2);
    skel_4d(output_none, input, size - 2);
    skepu::external(skepu::read(output_none), [&]{
        int expected = 0;
        for (int i = 0; i < output_none.size_i(); ++i)
            for (int j = 0; j < output_none.size_j(); ++j)
                for (int k = 0; k < output_none.size_k(); ++k)
                    for (int l = 0; l < output_none.size_l(); ++l)
                    {
                        INFO("MapOverlap4D none, index: (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ", " + std::to_string(l) + ")");
                        CHECK(output_none(i, j, k, l) == expected++);
                    }
    });
}