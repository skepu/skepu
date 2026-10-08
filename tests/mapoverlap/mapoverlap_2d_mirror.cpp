#include "../../external/catch2/catch.hpp"

#include <skepu>

// "Mirror mode": the edge mode None with an input that has the size of the output. Only the interior of the output
// (the elements that have the whole overlap in the input) is computed; the other elements keep their values.
int cross(skepu::Region2D<int> r) { return r(-1, 0) + 2 * r(1, 0) - r(0, -1) + r(0, 1) + 3 * r(0, 0) + r(-2, 1) - r(2, -2); }
int cross_indexed(skepu::Index2D i, skepu::Region2D<int> r) { return r(-1, -1) + r(1, 1) + (int)(i.row) * 100 + (int)(i.col); }
skepu::multiple<int, int> cross_two(skepu::Region2D<int> r) { return skepu::ret(r(-1, 0) + r(1, 0), r(0, -1) - r(0, 1)); }

auto cross_skeleton = skepu::MapOverlap(cross);
auto cross_indexed_skeleton = skepu::MapOverlap(cross_indexed);
auto cross_two_skeleton = skepu::MapOverlap(cross_two);

static int input_value(size_t i, size_t j) { return (int)((i * 7 + j * 3) % 13) - 4; }

TEST_CASE("MapOverlap 2D, mirror mode (edge mode None, input of the size of the output)")
{
	for (auto size : {std::pair<size_t, size_t>{9, 7}, {40, 40}, {100, 70}, {17, 300}, {300, 300}})
	{
		const size_t rows = size.first, cols = size.second;
		skepu::Matrix<int> in(rows, cols), out(rows, cols), out_a(rows, cols), out_b(rows, cols), out_i(rows, cols);
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
			{
				in(i, j) = input_value(i, j);
				out(i, j) = 1000 + (int)(i * cols + j);
				out_a(i, j) = -5;
				out_b(i, j) = 77;
				out_i(i, j) = 3;
			}
		for (skepu::Matrix<int> *m : {&in, &out, &out_a, &out_b, &out_i})
			m->invalidateDeviceData();

		cross_skeleton.setOverlap(2, 2);
		cross_skeleton.setEdgeMode(skepu::Edge::None);
		cross_skeleton(out, in);
		cross_indexed_skeleton.setOverlap(1, 1);
		cross_indexed_skeleton.setEdgeMode(skepu::Edge::None);
		cross_indexed_skeleton(out_i, in);
		cross_two_skeleton.setOverlap(1, 1);
		cross_two_skeleton.setEdgeMode(skepu::Edge::None);
		cross_two_skeleton(out_a, out_b, in);
		for (skepu::Matrix<int> *m : {&out, &out_a, &out_b, &out_i})
			m->updateHost();

		INFO("size = " << rows << "x" << cols);
		size_t wrong = 0;
		auto at = [&](long i, long j) { return input_value((size_t)i, (size_t)j); };
		for (long i = 0; i < (long)rows; ++i)
			for (long j = 0; j < (long)cols; ++j)
			{
				const bool inner2 = i >= 2 && i < (long)rows - 2 && j >= 2 && j < (long)cols - 2;
				const bool inner1 = i >= 1 && i < (long)rows - 1 && j >= 1 && j < (long)cols - 1;
				const int expected = inner2 ? at(i - 1, j) + 2 * at(i + 1, j) - at(i, j - 1) + at(i, j + 1) + 3 * at(i, j) + at(i - 2, j + 1) - at(i + 2, j - 2)
					: 1000 + (int)(i * cols + j);
				wrong += out(i, j) != expected;
				wrong += out_i(i, j) != (inner1 ? at(i - 1, j - 1) + at(i + 1, j + 1) + (int)i * 100 + (int)j : 3);
				wrong += out_a(i, j) != (inner1 ? at(i - 1, j) + at(i + 1, j) : -5);
				wrong += out_b(i, j) != (inner1 ? at(i, j - 1) - at(i, j + 1) : 77);
			}
		CHECK(wrong == 0);
	}
}
