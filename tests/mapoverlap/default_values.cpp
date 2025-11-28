#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/io.hpp>


int uf_1d(skepu::Region1D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        val += r(i);
    return val;
}

int uf_2d(skepu::Region2D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        for (int j = -r.oj; j <= r.oj; ++j)
            val += r(i, j);
    return val;
}

int uf_3d(skepu::Region3D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        for (int j = -r.oj; j <= r.oj; ++j)
            for (int k = -r.ok; k <= r.ok; ++k)
                val += r(i, j, k);
    return val;
}

int uf_4d(skepu::Region4D<int> r)
{
    int val = 0;
    for (int i = -r.oi; i <= r.oi; ++i)
        for (int j = -r.oj; j <= r.oj; ++j)
            for (int k = -r.ok; k <= r.ok; ++k)
                for (int l = -r.ol; l <= r.ol; ++l)
                    val += r(i, j, k, l);
    return val;
}

auto skel1D = skepu::MapOverlap(uf_1d);
auto skel2D = skepu::MapOverlap(uf_2d);
auto skel3D = skepu::MapOverlap(uf_3d);
auto skel4D = skepu::MapOverlap(uf_4d);

TEST_CASE("MapOverlap 1D")
{
    CHECK(skel1D.getEdgeMode() == skepu::Edge::Duplicate);
    CHECK(skel1D.getOverlapMode() == skepu::Overlap::RowWise);
    CHECK(skel1D.getPad() == int{});
    CHECK(skel1D.getUpdateMode() == skepu::UpdateMode::Normal);
    CHECK(skel1D.getOverlap() == 1);
}

TEST_CASE("MapOverlap 2D")
{
    CHECK(skel2D.getEdgeMode() == skepu::Edge::Duplicate);
    CHECK(skel2D.getUpdateMode() == skepu::UpdateMode::Normal);
    CHECK(skel2D.getPad() == int{});
    std::tuple<int, int> overlap = skel2D.getOverlap();
    CHECK(std::get<0>(overlap) == 1);
    CHECK(std::get<1>(overlap) == 1);
}

TEST_CASE("MapOverlap 3D")
{
    CHECK(skel3D.getEdgeMode() == skepu::Edge::Duplicate);
    CHECK(skel3D.getUpdateMode() == skepu::UpdateMode::Normal);
    CHECK(skel3D.getPad() == int{});
    std::tuple<int, int, int> overlap = skel3D.getOverlap();
    CHECK(std::get<0>(overlap) == 1);
    CHECK(std::get<1>(overlap) == 1);
    CHECK(std::get<2>(overlap) == 1);
}

TEST_CASE("MapOverlap 4D")
{
    CHECK(skel4D.getEdgeMode() == skepu::Edge::Duplicate);
    CHECK(skel4D.getUpdateMode() == skepu::UpdateMode::Normal);
    CHECK(skel4D.getPad() == int{});
    std::tuple<int, int, int, int> overlap = skel4D.getOverlap();
    CHECK(std::get<0>(overlap) == 1);
    CHECK(std::get<1>(overlap) == 1);
    CHECK(std::get<2>(overlap) == 1);
    CHECK(std::get<3>(overlap) == 1);
}