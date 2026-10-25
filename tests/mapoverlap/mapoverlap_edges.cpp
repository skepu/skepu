// Each part is in its own namespace; its test cases are prefixed with its name.
#define SKEPU_ENABLE_EXCEPTIONS

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <skepu-lib/randomize.hpp>
#include <algorithm>
#include <vector>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>
#include <array>
#include <utility>


// ======== mapoverlap_2d_mirror ========

namespace mapoverlap_2d_mirror_part
{
// Mirror mode (edge mode None, input of the size of the output): only the interior is computed; the other elements keep their values.
int cross(skepu::Region2D<int> r) { return r(-1, 0) + 2 * r(1, 0) - r(0, -1) + r(0, 1) + 3 * r(0, 0) + r(-2, 1) - r(2, -2); }
int cross_indexed(skepu::Index2D i, skepu::Region2D<int> r) { return r(-1, -1) + r(1, 1) + (int)(i.row) * 100 + (int)(i.col); }
skepu::multiple<int, int> cross_two(skepu::Region2D<int> r) { return skepu::ret(r(-1, 0) + r(1, 0), r(0, -1) - r(0, 1)); }

auto cross_skeleton = skepu::MapOverlap(cross);
auto cross_indexed_skeleton = skepu::MapOverlap(cross_indexed);
auto cross_two_skeleton = skepu::MapOverlap(cross_two);

static int input_value(size_t i, size_t j) { return (int)((i * 7 + j * 3) % 13) - 4; }

TEST_CASE("mapoverlap_2d_mirror: MapOverlap 2D, mirror mode (edge mode None, input of the size of the output)")
{
	for (auto size : std::initializer_list<std::pair<size_t, size_t>>{{9, 7}, {40, 40}, {100, 70}, {17, 300}, {300, 300}})
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
}


// ======== mapoverlap_in_place ========

namespace mapoverlap_in_place_part
{
float update_mapoverlap(skepu::Region1D<float> r)
{
	return (r(-1) + r(1)) / 2;
}

auto update = skepu::MapOverlap(update_mapoverlap);

TEST_CASE("mapoverlap_in_place: MapOverlap in place")
{
	const size_t size{100};
	
	// MapOverlap in place
	// Valid only for Red-Black update pattern with edge mode other than Cyclic
	// and as long as the result container is not also passed as a random-access input

	skepu::Vector<float> domain(size, 0);
	skepu::randomize::uniform(domain, 0, 2);
	domain.flush();   // (the elements are read on the host below)
	
	// The result on the host: the red elements (the odd ones) first, then the black ones (the even ones), each from the latest values
	std::vector<float> expected(size);
	for (size_t i = 0; i < size; ++i)
		expected[i] = domain(i);
	auto at = [&](long i) { return expected[std::min<long>(std::max<long>(i, 0), size - 1)]; };   // (edge mode Duplicate)
	for (size_t i = 1; i < size; i += 2)
		expected[i] = (at(i - 1) + at(i + 1)) / 2;
	for (size_t i = 0; i < size; i += 2)
		expected[i] = (at(i - 1) + at(i + 1)) / 2;
	
	update.setOverlap(1);
	update.setEdgeMode(skepu::Edge::Duplicate);
	update.setUpdateMode(skepu::UpdateMode::RedBlack);
	
	update(domain, domain);
	
	skepu::external(skepu::read(domain), [&]
	{
		for (size_t i = 0; i < size; ++i)
			CHECK(domain(i) == Approx(expected[i]));
	});
}
}


// ======== mapoverlap_updatemode ========

namespace mapoverlap_updatemode_part
{
int updatemode_1d_uf(skepu::Region1D<int> r)
{
	return 1;
}

int updatemode_2d_uf(skepu::Region2D<int> r)
{
	return 1;
}


auto updatemode_1d = skepu::MapOverlap(updatemode_1d_uf);
auto updatemode_2d = skepu::MapOverlap(updatemode_2d_uf);


TEST_CASE("mapoverlap_updatemode: MapOverlap update mode 1D")
{
	const size_t size{17};
  
  skepu::Vector<int> v(size), resR(size, 0), resB(size, 0), resRB(size, 0);
  
	updatemode_1d.setEdgeMode(skepu::Edge::Pad);
  
  updatemode_1d.setUpdateMode(skepu::UpdateMode::Red);
  updatemode_1d(resR, v);
	
  skepu::external(skepu::read(resR), [&]
	{
    for (size_t i = 0; i < size; ++i)
      CHECK(resR(i) == ((i % 2 == 0) ? 0 : 1));
  });
	skepu::io::cout << resR << "\n";
	
	updatemode_1d.setUpdateMode(skepu::UpdateMode::Black);
  updatemode_1d(resB, v);
	
  skepu::external(skepu::read(resB), [&]
	{
    for (size_t i = 0; i < size; ++i)
      CHECK(resB(i) == ((i % 2 == 0) ? 1 : 0));
  });
	skepu::io::cout << resB << "\n";
	
	
  updatemode_1d.setUpdateMode(skepu::UpdateMode::RedBlack);
  updatemode_1d(resRB, v);
	
  skepu::external(skepu::read(resRB), [&]
	{
    for (size_t i = 0; i < size; ++i)
      CHECK(resRB(i) == 1);
  });
	skepu::io::cout << resRB << "\n";
	
}


TEST_CASE("mapoverlap_updatemode: MapOverlap update mode 2D")
{
	const size_t size_i{20}, size_j{31};
  
  skepu::Matrix<int> m(size_i, size_j), resR(size_i, size_j, 0), resB(size_i, size_j, 0), resRB(size_i, size_j, 0);
  
	updatemode_2d.setEdgeMode(skepu::Edge::Pad);
  
  updatemode_2d.setUpdateMode(skepu::UpdateMode::Red);
  updatemode_2d(resR, m);
	
  skepu::external(skepu::read(resR), [&]
	{
    for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
      	CHECK(resR(i, j) == (((i + j) % 2 == 0) ? 0 : 1));
  });
	skepu::io::cout << resR << "\n";
	
	updatemode_2d.setUpdateMode(skepu::UpdateMode::Black);
  updatemode_2d(resB, m);
	
  skepu::external(skepu::read(resB), [&]
	{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
      	CHECK(resB(i, j) == (((i + j) % 2 == 0) ? 1 : 0));
  });
	skepu::io::cout << resB << "\n";
	
	
  updatemode_2d.setUpdateMode(skepu::UpdateMode::RedBlack);
  updatemode_2d(resRB, m);
	
  skepu::external(skepu::read(resRB), [&]
	{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
      	CHECK(resRB(i, j) == 1);
  });
	skepu::io::cout << resRB << "\n";
	
}


// ------------------------------------------------
//   The update modes in all the variants of MapOverlap
// ------------------------------------------------

// The elements of the output that a mode computes: all of them, or the red (odd) or black (even) ones, by the parity of the sum of the indices
static const int sentinel = -77;

static bool computed(skepu::UpdateMode mode, size_t index_sum)
{
	switch (mode)
	{
		case skepu::UpdateMode::Red:   return index_sum % 2 == 1;
		case skepu::UpdateMode::Black: return index_sum % 2 == 0;
		default:                       return true;
	}
}

static const skepu::UpdateMode all_modes[] = {skepu::UpdateMode::Normal, skepu::UpdateMode::Red, skepu::UpdateMode::Black, skepu::UpdateMode::RedBlack};

// The value of an element depends on its index and on its input, so that also the indices and the regions are checked
int um_index_1d(skepu::Index1D i, skepu::Region1D<int> r) { return (int)i.i * 10 + r(0) + r(-1) - r(1); }
int um_index_2d(skepu::Index2D i, skepu::Region2D<int> r) { return (int)(i.row * 100 + i.col) * 10 + r(0, 0) + r(-1, 0) - r(0, 1); }
int um_index_3d(skepu::Index3D i, skepu::Region3D<int> r) { return (int)(i.i * 100 + i.j * 10 + i.k) * 10 + r(0, 0, 0) + r(-1, 0, 0) - r(0, 0, 1); }
int um_index_4d(skepu::Index4D i, skepu::Region4D<int> r) { return (int)(i.i * 1000 + i.j * 100 + i.k * 10 + i.l) * 10 + r(0, 0, 0, 0) + r(-1, 0, 0, 0) - r(0, 0, 0, 1); }

auto um_skeleton_1d = skepu::MapOverlap(um_index_1d);
auto um_skeleton_2d = skepu::MapOverlap(um_index_2d);
auto um_skeleton_3d = skepu::MapOverlap(um_index_3d);
auto um_skeleton_4d = skepu::MapOverlap(um_index_4d);

static int um_input(long i, long j = 0, long k = 0, long l = 0) { return (int)((i * 3 + j * 5 + k * 7 + l * 11) % 17) - 6; }

// The edge mode Duplicate with overlap 1: the input outside is the nearest element
static long um_clamp(long i, long size) { return std::min(std::max(i, 0L), size - 1); }


TEST_CASE("mapoverlap_updatemode: MapOverlap update modes, 1D vector, edge modes Duplicate, None and mirror mode")
{
	for (size_t size : {2, 3, 17, 100, 1031})
		for (skepu::Edge edge : {skepu::Edge::Duplicate, skepu::Edge::None, skepu::Edge::Pad})
			for (bool mirror : {false, true})
			{
				if (mirror && edge != skepu::Edge::None)
					continue;
				if (edge == skepu::Edge::None && !mirror && size < 3)
					continue;
				
				// The edge mode None without mirror mode: the input has the overlap more on each side
				const long in_size = (edge == skepu::Edge::None && !mirror) ? size : size;
				const long out_size = (edge == skepu::Edge::None && !mirror) ? (long)size - 2 : (long)size;
				
				skepu::Vector<int> in(in_size);
				for (long i = 0; i < in_size; ++i)
					in(i) = um_input(i);
				
				for (skepu::UpdateMode mode : all_modes)
				{
					skepu::Vector<int> out(out_size, sentinel);
					um_skeleton_1d.setOverlap(1);
					um_skeleton_1d.setEdgeMode(edge);
					um_skeleton_1d.setPad(5);
					um_skeleton_1d.setUpdateMode(mode);
					um_skeleton_1d(out, in);
					out.flush();
					
					for (long i = 0; i < out_size; ++i)
					{
						const bool interior = !mirror || (i >= 1 && i + 1 < out_size);
						int expected = sentinel;
						if (interior && computed(mode, i))
						{
							const long c = (edge == skepu::Edge::None && !mirror) ? i + 1 : i;  // (the center in the input)
							auto at = [&](long x) { return (x < 0 || x >= in_size) ? (edge == skepu::Edge::Pad ? 5 : in(um_clamp(x, in_size))) : in(x); };
							expected = (int)i * 10 + at(c) + at(c - 1) - at(c + 1);
						}
						INFO("size " << size << ", edge " << (int)edge << ", mirror " << mirror << ", mode " << skepu::to_string(mode) << ", i " << i);
						CHECK(out(i) == expected);
					}
				}
			}
	um_skeleton_1d.setUpdateMode(skepu::UpdateMode::Normal);
}


TEST_CASE("mapoverlap_updatemode: MapOverlap update modes, 1D row-wise and column-wise")
{
	for (auto dims : std::vector<std::pair<size_t, size_t>>{{3, 3}, {7, 10}, {10, 7}, {33, 40}, {40, 33}})
		for (skepu::Overlap overlap_mode : {skepu::Overlap::RowWise, skepu::Overlap::ColWise})
			for (skepu::Edge edge : {skepu::Edge::Duplicate, skepu::Edge::None})
			{
				const long rows = dims.first, cols = dims.second;
				skepu::Matrix<int> in(rows, cols);
				for (long i = 0; i < rows; ++i)
					for (long j = 0; j < cols; ++j)
						in(i, j) = um_input(i, j);
				
				// Row-wise, the index of the user function is the column, and column-wise it is the row
				const long out_rows = (edge == skepu::Edge::None && overlap_mode == skepu::Overlap::ColWise) ? rows - 2 : rows;
				const long out_cols = (edge == skepu::Edge::None && overlap_mode == skepu::Overlap::RowWise) ? cols - 2 : cols;
				
				for (skepu::UpdateMode mode : all_modes)
				{
					skepu::Matrix<int> out(out_rows, out_cols, sentinel);
					um_skeleton_1d.setOverlap(1);
					um_skeleton_1d.setOverlapMode(overlap_mode);
					um_skeleton_1d.setEdgeMode(edge);
					um_skeleton_1d.setUpdateMode(mode);
					um_skeleton_1d(out, in);
					out.flush();
					
					for (long i = 0; i < out_rows; ++i)
						for (long j = 0; j < out_cols; ++j)
						{
							const bool row_wise = overlap_mode == skepu::Overlap::RowWise;
							const long idx = row_wise ? j : i;
							const long shift = (edge == skepu::Edge::None) ? 1 : 0;
							auto at = [&](long x) { return row_wise ? in(i, um_clamp(x + shift, cols)) : in(um_clamp(x + shift, rows), j); };
							const int expected = computed(mode, idx) ? (int)idx * 10 + at(idx) + at(idx - 1) - at(idx + 1) : sentinel;
							INFO("dims " << rows << "x" << cols << ", row-wise " << row_wise << ", edge " << (int)edge << ", mode " << skepu::to_string(mode) << ", i " << i << ", j " << j);
							CHECK(out(i, j) == expected);
						}
				}
			}
	um_skeleton_1d.setOverlapMode(skepu::Overlap::RowWise);
	um_skeleton_1d.setUpdateMode(skepu::UpdateMode::Normal);
}


TEST_CASE("mapoverlap_updatemode: MapOverlap update modes, 2D, edge modes Duplicate and None, and mirror mode")
{
	for (auto dims : std::vector<std::pair<size_t, size_t>>{{3, 3}, {5, 8}, {20, 31}, {41, 40}})
		for (skepu::Edge edge : {skepu::Edge::Duplicate, skepu::Edge::None})
			for (bool mirror : {false, true})
			{
				if (mirror && edge != skepu::Edge::None)
					continue;
				const long rows = dims.first, cols = dims.second;
				const long shift = (edge == skepu::Edge::None && !mirror) ? 1 : 0;
				const long out_rows = rows - 2 * shift, out_cols = cols - 2 * shift;
				skepu::Matrix<int> in(rows, cols);
				for (long i = 0; i < rows; ++i)
					for (long j = 0; j < cols; ++j)
						in(i, j) = um_input(i, j);
				
				for (skepu::UpdateMode mode : all_modes)
				{
					skepu::Matrix<int> out(out_rows, out_cols, sentinel);
					um_skeleton_2d.setOverlap(1, 1);
					um_skeleton_2d.setEdgeMode(edge);
					um_skeleton_2d.setUpdateMode(mode);
					um_skeleton_2d(out, in);
					out.flush();
					
					for (long i = 0; i < out_rows; ++i)
						for (long j = 0; j < out_cols; ++j)
						{
							const bool interior = !mirror || (i >= 1 && i + 1 < out_rows && j >= 1 && j + 1 < out_cols);
							auto at = [&](long x, long y) { return in(um_clamp(x + shift, rows), um_clamp(y + shift, cols)); };
							const int expected = (interior && computed(mode, i + j)) ? (int)(i * 100 + j) * 10 + at(i, j) + at(i - 1, j) - at(i, j + 1) : sentinel;
							INFO("dims " << rows << "x" << cols << ", edge " << (int)edge << ", mirror " << mirror << ", mode " << skepu::to_string(mode) << ", i " << i << ", j " << j);
							CHECK(out(i, j) == expected);
						}
				}
			}
	um_skeleton_2d.setUpdateMode(skepu::UpdateMode::Normal);
}


TEST_CASE("mapoverlap_updatemode: MapOverlap update modes, 3D")
{
	for (auto size : std::vector<std::array<size_t, 3>>{{3, 3, 3}, {5, 6, 7}, {9, 8, 10}, {4, 21, 5}})
		for (skepu::Edge edge : {skepu::Edge::Duplicate, skepu::Edge::None})
			for (bool mirror : {false, true})
			{
				if (mirror && edge != skepu::Edge::None)
					continue;
				const long shift = (edge == skepu::Edge::None && !mirror) ? 1 : 0;
				const long n0 = size[0], n1 = size[1], n2 = size[2];
				skepu::Tensor3<int> in(n0, n1, n2);
				for (long i = 0; i < n0; ++i)
					for (long j = 0; j < n1; ++j)
						for (long k = 0; k < n2; ++k)
							in(i, j, k) = um_input(i, j, k);
				
				for (skepu::UpdateMode mode : all_modes)
				{
					skepu::Tensor3<int> out(n0 - 2 * shift, n1 - 2 * shift, n2 - 2 * shift);
					for (long i = 0; i < n0 - 2 * shift; ++i)
						for (long j = 0; j < n1 - 2 * shift; ++j)
							for (long k = 0; k < n2 - 2 * shift; ++k)
								out(i, j, k) = sentinel;
					
					um_skeleton_3d.setOverlap(1, 1, 1);
					um_skeleton_3d.setEdgeMode(edge);
					um_skeleton_3d.setUpdateMode(mode);
					um_skeleton_3d(out, in);
					out.flush();
					
					for (long i = 0; i < n0 - 2 * shift; ++i)
						for (long j = 0; j < n1 - 2 * shift; ++j)
							for (long k = 0; k < n2 - 2 * shift; ++k)
							{
								const bool interior = !mirror || (i >= 1 && i + 1 < n0 && j >= 1 && j + 1 < n1 && k >= 1 && k + 1 < n2);
								auto at = [&](long x, long y, long z) { return in(um_clamp(x + shift, n0), um_clamp(y + shift, n1), um_clamp(z + shift, n2)); };
								const int expected = (interior && computed(mode, i + j + k)) ? (int)(i * 100 + j * 10 + k) * 10 + at(i, j, k) + at(i - 1, j, k) - at(i, j, k + 1) : sentinel;
								INFO("size " << n0 << "x" << n1 << "x" << n2 << ", edge " << (int)edge << ", mirror " << mirror << ", mode " << skepu::to_string(mode) << ", " << i << " " << j << " " << k);
								CHECK(out(i, j, k) == expected);
							}
				}
			}
	um_skeleton_3d.setUpdateMode(skepu::UpdateMode::Normal);
}


TEST_CASE("mapoverlap_updatemode: MapOverlap update modes, 4D")
{
	for (auto size : std::vector<std::array<size_t, 4>>{{3, 3, 3, 3}, {5, 5, 6, 7}, {4, 4, 5, 30}})
		for (skepu::Edge edge : {skepu::Edge::Duplicate, skepu::Edge::None})
			for (bool mirror : {false, true})
			{
				if (mirror && edge != skepu::Edge::None)
					continue;
				const long shift = (edge == skepu::Edge::None && !mirror) ? 1 : 0;
				const long n0 = size[0], n1 = size[1], n2 = size[2], n3 = size[3];
				skepu::Tensor4<int> in(n0, n1, n2, n3);
				for (long i = 0; i < n0; ++i)
					for (long j = 0; j < n1; ++j)
						for (long k = 0; k < n2; ++k)
							for (long l = 0; l < n3; ++l)
								in(i, j, k, l) = um_input(i, j, k, l);
				
				for (skepu::UpdateMode mode : all_modes)
				{
					skepu::Tensor4<int> out(n0 - 2 * shift, n1 - 2 * shift, n2 - 2 * shift, n3 - 2 * shift);
					for (long i = 0; i < n0 - 2 * shift; ++i)
						for (long j = 0; j < n1 - 2 * shift; ++j)
							for (long k = 0; k < n2 - 2 * shift; ++k)
								for (long l = 0; l < n3 - 2 * shift; ++l)
									out(i, j, k, l) = sentinel;
					
					um_skeleton_4d.setOverlap(1, 1, 1, 1);
					um_skeleton_4d.setEdgeMode(edge);
					um_skeleton_4d.setUpdateMode(mode);
					um_skeleton_4d(out, in);
					out.flush();
					
					for (long i = 0; i < n0 - 2 * shift; ++i)
						for (long j = 0; j < n1 - 2 * shift; ++j)
							for (long k = 0; k < n2 - 2 * shift; ++k)
								for (long l = 0; l < n3 - 2 * shift; ++l)
								{
									const bool interior = !mirror || (i >= 1 && i + 1 < n0 && j >= 1 && j + 1 < n1 && k >= 1 && k + 1 < n2 && l >= 1 && l + 1 < n3);
									auto at = [&](long a, long b, long c, long d) { return in(um_clamp(a + shift, n0), um_clamp(b + shift, n1), um_clamp(c + shift, n2), um_clamp(d + shift, n3)); };
									const int expected = (interior && computed(mode, i + j + k + l))
										? (int)(i * 1000 + j * 100 + k * 10 + l) * 10 + at(i, j, k, l) + at(i - 1, j, k, l) - at(i, j, k, l + 1) : sentinel;
									INFO("size " << n0 << "x" << n1 << "x" << n2 << "x" << n3 << ", edge " << (int)edge << ", mirror " << mirror << ", mode " << skepu::to_string(mode) << ", " << i << " " << j << " " << k << " " << l);
									CHECK(out(i, j, k, l) == expected);
								}
				}
			}
	um_skeleton_4d.setUpdateMode(skepu::UpdateMode::Normal);
}


// ------------------------------------------------
//   Random numbers cannot be combined with the update modes
// ------------------------------------------------

int um_random_1d(skepu::Random<1> &rnd, skepu::Region1D<int> r) { return (int)(rnd.get() % 1000) + r(0); }
int um_random_2d(skepu::Random<1> &rnd, skepu::Region2D<int> r) { return (int)(rnd.get() % 1000) + r(0, 0); }
int um_random_3d(skepu::Random<1> &rnd, skepu::Region3D<int> r) { return (int)(rnd.get() % 1000) + r(0, 0, 0); }
int um_random_4d(skepu::Random<1> &rnd, skepu::Region4D<int> r) { return (int)(rnd.get() % 1000) + r(0, 0, 0, 0); }

auto um_random_skeleton_1d = skepu::MapOverlap(um_random_1d);
auto um_random_skeleton_2d = skepu::MapOverlap(um_random_2d);
auto um_random_skeleton_3d = skepu::MapOverlap(um_random_3d);
auto um_random_skeleton_4d = skepu::MapOverlap(um_random_4d);

TEST_CASE("mapoverlap_updatemode: MapOverlap update modes with random numbers fail with an error")
{
	skepu::Vector<int> v(20, 1), v_out(20, 0);
	skepu::Matrix<int> m(8, 9, 1), m_out(8, 9, 0);
	skepu::Tensor3<int> t3(5, 6, 7), t3_out(5, 6, 7);
	skepu::Tensor4<int> t4(4, 5, 6, 3), t4_out(4, 5, 6, 3);
	skepu::PRNG prng(5);
	um_random_skeleton_1d.setPRNG(prng);
	um_random_skeleton_2d.setPRNG(prng);
	um_random_skeleton_3d.setPRNG(prng);
	um_random_skeleton_4d.setPRNG(prng);
	
	for (skepu::UpdateMode mode : {skepu::UpdateMode::Red, skepu::UpdateMode::Black, skepu::UpdateMode::RedBlack})
	{
		um_random_skeleton_1d.setUpdateMode(mode);
		um_random_skeleton_2d.setUpdateMode(mode);
		um_random_skeleton_3d.setUpdateMode(mode);
		um_random_skeleton_4d.setUpdateMode(mode);
		CHECK_THROWS_WITH(um_random_skeleton_1d(v_out, v), Catch::Contains("update modes"));
		CHECK_THROWS_WITH(um_random_skeleton_2d(m_out, m), Catch::Contains("update modes"));
		CHECK_THROWS_WITH(um_random_skeleton_3d(t3_out, t3), Catch::Contains("update modes"));
		CHECK_THROWS_WITH(um_random_skeleton_4d(t4_out, t4), Catch::Contains("update modes"));
	}
	
	// The mode Normal works
	um_random_skeleton_1d.setUpdateMode(skepu::UpdateMode::Normal);
	um_random_skeleton_2d.setUpdateMode(skepu::UpdateMode::Normal);
	um_random_skeleton_3d.setUpdateMode(skepu::UpdateMode::Normal);
	um_random_skeleton_4d.setUpdateMode(skepu::UpdateMode::Normal);
	CHECK_NOTHROW(um_random_skeleton_1d(v_out, v));
	CHECK_NOTHROW(um_random_skeleton_2d(m_out, m));
	CHECK_NOTHROW(um_random_skeleton_3d(t3_out, t3));
	CHECK_NOTHROW(um_random_skeleton_4d(t4_out, t4));
}
}


// ======== mapoverlap_1d_mirror ========

namespace mapoverlap_1d_mirror_part
{
// Mirror mode (edge mode None, input of the size of the output): only the interior is computed; the other elements keep their values.
// Vectors, and matrices row-wise and column-wise.
int cross(skepu::Region1D<int> r) { return r(-2) + 3 * r(-1) + 5 * r(0) - 7 * r(1) + 11 * r(2); }
int cross_indexed(skepu::Index1D i, skepu::Region1D<int> r) { return r(-1) + r(1) + (int)(i.i) * 100; }
skepu::multiple<int, int> cross_two(skepu::Region1D<int> r) { return skepu::ret(r(-1) + r(1), r(0) - r(1)); }
int random_element(skepu::Random<1> &rnd, skepu::Region1D<int> r) { return (int)(rnd.get() % 1000) + r(0); }

auto cross_skeleton = skepu::MapOverlap(cross);
auto cross_indexed_skeleton = skepu::MapOverlap(cross_indexed);
auto cross_two_skeleton = skepu::MapOverlap(cross_two);
auto random_skeleton = skepu::MapOverlap(random_element);

static int input_value(long i, long j = 0) { return (int)((i * 7 + j * 3) % 13) - 4; }

TEST_CASE("mapoverlap_1d_mirror: MapOverlap 1D, mirror mode on vectors")
{
	for (size_t size : {9, 40, 300, 1000, 5000})
	{
		skepu::Vector<int> in(size), out(size), out_a(size), out_b(size), out_i(size);
		for (size_t i = 0; i < size; ++i)
		{
			in(i) = input_value(i);
			out(i) = 1000 + (int)i;
			out_a(i) = -5;
			out_b(i) = 77;
			out_i(i) = 3;
		}
		for (skepu::Vector<int> *v : {&in, &out, &out_a, &out_b, &out_i})
			v->invalidateDeviceData();

		cross_skeleton.setOverlap(2);
		cross_skeleton.setEdgeMode(skepu::Edge::None);
		cross_skeleton(out, in);
		cross_indexed_skeleton.setOverlap(1);
		cross_indexed_skeleton.setEdgeMode(skepu::Edge::None);
		cross_indexed_skeleton(out_i, in);
		cross_two_skeleton.setOverlap(1);
		cross_two_skeleton.setEdgeMode(skepu::Edge::None);
		cross_two_skeleton(out_a, out_b, in);
		for (skepu::Vector<int> *v : {&out, &out_a, &out_b, &out_i})
			v->updateHost();

		INFO("size = " << size);
		size_t wrong = 0;
		auto at = [&](long i) { return input_value(i); };
		for (long i = 0; i < (long)size; ++i)
		{
			const bool inner2 = i >= 2 && i < (long)size - 2;
			const bool inner1 = i >= 1 && i < (long)size - 1;
			wrong += out(i) != (inner2 ? at(i - 2) + 3 * at(i - 1) + 5 * at(i) - 7 * at(i + 1) + 11 * at(i + 2) : 1000 + (int)i);
			wrong += out_i(i) != (inner1 ? at(i - 1) + at(i + 1) + (int)i * 100 : 3);
			wrong += out_a(i) != (inner1 ? at(i - 1) + at(i + 1) : -5);
			wrong += out_b(i) != (inner1 ? at(i) - at(i + 1) : 77);
		}
		CHECK(wrong == 0);
	}
}

TEST_CASE("mapoverlap_1d_mirror: MapOverlap 1D, mirror mode on matrices, row-wise and column-wise")
{
	for (auto size : std::initializer_list<std::pair<size_t, size_t>>{{9, 7}, {40, 40}, {100, 70}, {17, 300}, {300, 17}})
		for (auto overlapMode : {skepu::Overlap::RowWise, skepu::Overlap::ColWise})
		{
			const size_t rows = size.first, cols = size.second;
			const bool rowwise = overlapMode == skepu::Overlap::RowWise;
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

			auto configure = [&](auto &skeleton)
			{
				skeleton.setEdgeMode(skepu::Edge::None);
				skeleton.setOverlapMode(overlapMode);
			};
			configure(cross_skeleton);
			configure(cross_indexed_skeleton);
			configure(cross_two_skeleton);
			cross_skeleton.setOverlap(2);
			cross_skeleton(out, in);
			cross_indexed_skeleton.setOverlap(1);
			cross_indexed_skeleton(out_i, in);
			cross_two_skeleton.setOverlap(1);
			cross_two_skeleton(out_a, out_b, in);
			for (skepu::Matrix<int> *m : {&out, &out_a, &out_b, &out_i})
				m->updateHost();

			INFO("size = " << rows << "x" << cols << (rowwise ? ", row-wise" : ", column-wise"));
			size_t wrong = 0;
			auto at = [&](long i, long j) { return input_value(i, j); };
			// (along the direction of the overlap; Index1D is the position in it)
			auto neighbour = [&](long i, long j, long d) { return rowwise ? at(i, j + d) : at(i + d, j); };
			for (long i = 0; i < (long)rows; ++i)
				for (long j = 0; j < (long)cols; ++j)
				{
					const long position = rowwise ? j : i, length = rowwise ? (long)cols : (long)rows;
					const bool inner2 = position >= 2 && position < length - 2;
					const bool inner1 = position >= 1 && position < length - 1;
					wrong += out(i, j) != (inner2 ? neighbour(i, j, -2) + 3 * neighbour(i, j, -1) + 5 * neighbour(i, j, 0) - 7 * neighbour(i, j, 1) + 11 * neighbour(i, j, 2) : 1000 + (int)(i * cols + j));
					wrong += out_i(i, j) != (inner1 ? neighbour(i, j, -1) + neighbour(i, j, 1) + (int)position * 100 : 3);
					wrong += out_a(i, j) != (inner1 ? neighbour(i, j, -1) + neighbour(i, j, 1) : -5);
					wrong += out_b(i, j) != (inner1 ? neighbour(i, j, 0) - neighbour(i, j, 1) : 77);
				}
			CHECK(wrong == 0);
		}
}

TEST_CASE("mapoverlap_1d_mirror: MapOverlap 1D, mirror mode and the random numbers")
{
	// The inner elements get the numbers one after the other, and the stream is advanced over all the elements of the output
	const size_t size{103};
	const long overlap{2};
	skepu::Vector<int> in(size, 1), out(size, -1);

	random_skeleton.setOverlap(overlap);
	random_skeleton.setEdgeMode(skepu::Edge::None);
	skepu::PRNG prng(77), reference(77);
	random_skeleton.setPRNG(prng);
	random_skeleton(out, in);
	out.updateHost();

	size_t wrong = 0;
	for (long i = 0; i < (long)size; ++i)
		if (i < overlap || i >= (long)size - overlap)
			wrong += out(i) != -1;
		else
			wrong += out(i) != (int)(reference.get() % 1000) + 1;
	CHECK(wrong == 0);
	reference.forward(2 * overlap);
	CHECK(prng.get() == reference.get());
}

TEST_CASE("mapoverlap_1d_mirror: MapOverlap 1D, mirror mode and a dimension that is not larger than twice the overlap")
{
	// No element has its whole overlap in the input: nothing is computed, the output keeps its values, and the random numbers are used
	random_skeleton.setOverlap(2);
	random_skeleton.setEdgeMode(skepu::Edge::None);

	for (size_t size : {1, 3, 4})
	{
		skepu::Vector<int> in(size, 1), out(size, 7);
		skepu::PRNG prng(5), reference(5);
		random_skeleton.setPRNG(prng);
		random_skeleton(out, in);
		out.updateHost();
		for (size_t i = 0; i < size; ++i)
			CHECK(out(i) == 7);
		reference.forward(size);
		CHECK(prng.get() == reference.get());
	}

	for (auto overlapMode : {skepu::Overlap::RowWise, skepu::Overlap::ColWise})
	{
		const bool rowwise = overlapMode == skepu::Overlap::RowWise;
		const size_t rows = rowwise ? 6 : 4, cols = rowwise ? 4 : 6;   // (the dimension of the overlap is 4)
		skepu::Matrix<int> in(rows, cols, 1), out(rows, cols, 7);
		random_skeleton.setOverlapMode(overlapMode);
		skepu::PRNG prng(5), reference(5);
		random_skeleton.setPRNG(prng);
		random_skeleton(out, in);
		out.updateHost();
		size_t wrong = 0;
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				wrong += out(i, j) != 7;
		CHECK(wrong == 0);
		reference.forward(rows * cols);
		CHECK(prng.get() == reference.get());
	}
}
}


// ======== mapoverlap_2d_edge_modes ========

namespace mapoverlap_2d_edge_modes_part
{
// With an edge mode the input has the size of the output and every output element,
// the border included, is computed (on all backends).
int neighbor_sum(skepu::Region2D<int> r)
{
	int res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			res += r(i, j);
	return res;
}

auto sum3x3 = skepu::MapOverlap(neighbor_sum);

TEST_CASE("mapoverlap_2d_edge_modes: MapOverlap 2D, edge modes with input and output of the same size")
{
	const size_t n{6};

	sum3x3.setOverlap(1, 1);
	sum3x3.setEdgeMode(skepu::Edge::Pad);
	sum3x3.setPad(0);

	skepu::Matrix<int> in(n, n), out(n, n, -1);
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			in(i, j) = 1 + (int)(i * n + j);

	sum3x3(out, in);
	out.updateHost();

	for (int i = 0; i < (int)n; ++i)
		for (int j = 0; j < (int)n; ++j)
		{
			int expected = 0;
			for (int di = -1; di <= 1; ++di)
				for (int dj = -1; dj <= 1; ++dj)
					if (i + di >= 0 && i + di < (int)n && j + dj >= 0 && j + dj < (int)n)
						expected += in(i + di, j + dj);
			REQUIRE(out(i, j) == expected);
		}
}
}


// ======== mapoverlap_3d_mirror ========

namespace mapoverlap_3d_mirror_part
{
// Mirror mode (edge mode None, input of the size of the output): only the interior is computed; the other elements keep their values.
int cross(skepu::Region3D<int> r) { return r(-1, 0, 0) + 2 * r(1, 0, 0) + r(0, -1, 0) + 2 * r(0, 1, 0) + r(0, 0, -1) + 2 * r(0, 0, 1) + 3 * r(0, 0, 0); }
int cross_indexed(skepu::Index3D i, skepu::Region3D<int> r) { return r(-1, -1, -1) + r(1, 1, 1) + (int)(i.i + i.j + i.k) * 100; }
skepu::multiple<int, int> cross_two(skepu::Region3D<int> r) { return skepu::ret(r(-1, 0, 0) + r(1, 0, 0), r(0, 0, -1) - r(0, 0, 1)); }
int random_element(skepu::Random<1> &rnd, skepu::Region3D<int> r) { return (int)(rnd.get() % 1000) + r(0, 0, 0); }

auto cross_skeleton = skepu::MapOverlap(cross);
auto cross_indexed_skeleton = skepu::MapOverlap(cross_indexed);
auto cross_two_skeleton = skepu::MapOverlap(cross_two);
auto random_skeleton = skepu::MapOverlap(random_element);

static int input_value(long i, long j, long k) { return (int)((i * 3 + j * 5 + k * 7) % 13) - 4; }

TEST_CASE("mapoverlap_3d_mirror: MapOverlap 3D, mirror mode (edge mode None, input of the size of the output)")
{
	for (auto size : std::vector<std::array<size_t, 3>>{{5, 6, 7}, {9, 8, 10}, {20, 12, 13}, {4, 40, 5}})
	{
		skepu::Tensor3<int> in(size[0], size[1], size[2]), out(size[0], size[1], size[2]), out_a(size[0], size[1], size[2]), out_b(size[0], size[1], size[2]), out_i(size[0], size[1], size[2]);
		for (long i = 0; i < (long)size[0]; ++i)
			for (long j = 0; j < (long)size[1]; ++j)
				for (long k = 0; k < (long)size[2]; ++k)
					{
						const long flat = ((i) * (long)size[1] + j) * (long)size[2] + k;
						in(i, j, k) = input_value(i, j, k);
						out(i, j, k) = 1000 + (int)flat;
						out_a(i, j, k) = -5;
						out_b(i, j, k) = 77;
						out_i(i, j, k) = 3;
					}
		for (skepu::Tensor3<int> *t : {&in, &out, &out_a, &out_b, &out_i})
			t->invalidateDeviceData();

		cross_skeleton.setOverlap(1, 1, 1);
		cross_skeleton.setEdgeMode(skepu::Edge::None);
		cross_skeleton(out, in);
		cross_indexed_skeleton.setOverlap(1, 1, 1);
		cross_indexed_skeleton.setEdgeMode(skepu::Edge::None);
		cross_indexed_skeleton(out_i, in);
		cross_two_skeleton.setOverlap(1, 1, 1);
		cross_two_skeleton.setEdgeMode(skepu::Edge::None);
		cross_two_skeleton(out_a, out_b, in);
		for (skepu::Tensor3<int> *t : {&out, &out_a, &out_b, &out_i})
			t->updateHost();

		INFO("size = " << size[0] << "x" << size[1] << "x" << size[2]);
		size_t wrong = 0;
		auto at = [&](long i, long j, long k) { return input_value(i, j, k); };
		for (long i = 0; i < (long)size[0]; ++i)
			for (long j = 0; j < (long)size[1]; ++j)
				for (long k = 0; k < (long)size[2]; ++k)
					{
						const bool inner = i >= 1 && i < (long)size[0] - 1 && j >= 1 && j < (long)size[1] - 1 && k >= 1 && k < (long)size[2] - 1;
						const long flat = ((i) * (long)size[1] + j) * (long)size[2] + k;
						wrong += out(i, j, k) != (inner ? at(i-1, j, k) + 2 * at(i+1, j, k) + at(i, j-1, k) + 2 * at(i, j+1, k) + at(i, j, k-1) + 2 * at(i, j, k+1) + 3 * at(i, j, k) : 1000 + (int)flat);
						wrong += out_i(i, j, k) != (inner ? at(i-1, j-1, k-1) + at(i+1, j+1, k+1) + (i + j + k) * 100 : 3);
						wrong += out_a(i, j, k) != (inner ? at(i-1, j, k) + at(i+1, j, k) : -5);
						wrong += out_b(i, j, k) != (inner ? at(i, j, k-1) - at(i, j, k+1) : 77);
					}
		CHECK(wrong == 0);
	}
}

TEST_CASE("mapoverlap_3d_mirror: MapOverlap 3D, mirror mode and the random numbers")
{
	// The inner elements get the numbers one after the other, and the stream is advanced over all the elements of the output
	const std::array<size_t, 3> size{9, 8, 10};
	skepu::Tensor3<int> in(size[0], size[1], size[2], "", 1), out(size[0], size[1], size[2], "", -1);

	random_skeleton.setOverlap(1, 1, 1);
	random_skeleton.setEdgeMode(skepu::Edge::None);
	skepu::PRNG prng(77), reference(77);
	random_skeleton.setPRNG(prng);
	random_skeleton(out, in);
	out.updateHost();

	size_t wrong = 0, total = 0;
	for (long i = 0; i < (long)size[0]; ++i)
		for (long j = 0; j < (long)size[1]; ++j)
			for (long k = 0; k < (long)size[2]; ++k)
				{
					++total;
					if (i < 1 || i >= (long)size[0] - 1 || j < 1 || j >= (long)size[1] - 1 || k < 1 || k >= (long)size[2] - 1)
						wrong += out(i, j, k) != -1;
					else
						wrong += out(i, j, k) != (int)(reference.get() % 1000) + 1;
				}
	CHECK(wrong == 0);
	reference.forward(total - (size[0] - 2)*(size[1] - 2)*(size[2] - 2));
	CHECK(prng.get() == reference.get());
}

TEST_CASE("mapoverlap_3d_mirror: MapOverlap 3D, mirror mode and a dimension that is not larger than twice the overlap")
{
	// No element has its whole overlap in the input: nothing is computed, the output keeps its values, and the random numbers are used
	const std::array<size_t, 3> size{2, 6, 6};
	skepu::Tensor3<int> in(size[0], size[1], size[2], "", 1), out(size[0], size[1], size[2], "", 7);
	random_skeleton.setOverlap(1, 1, 1);
	random_skeleton.setEdgeMode(skepu::Edge::None);
	skepu::PRNG prng(5), reference(5);
	random_skeleton.setPRNG(prng);
	random_skeleton(out, in);
	out.updateHost();
	size_t wrong = 0, total = 0;
	for (size_t i = 0; i < out.size(); ++i, ++total)
		wrong += out.getAddress()[i] != 7;
	CHECK(wrong == 0);
	reference.forward(total);
	CHECK(prng.get() == reference.get());
}
}


// ======== mapoverlap_4d_mirror ========

namespace mapoverlap_4d_mirror_part
{
// Mirror mode (edge mode None, input of the size of the output): only the interior is computed; the other elements keep their values.
int cross(skepu::Region4D<int> r) { return r(-1, 0, 0, 0) + 2 * r(1, 0, 0, 0) + r(0, -1, 0, 0) + 2 * r(0, 1, 0, 0) + r(0, 0, -1, 0) + 2 * r(0, 0, 1, 0) + r(0, 0, 0, -1) + 2 * r(0, 0, 0, 1) + 3 * r(0, 0, 0, 0); }
int cross_indexed(skepu::Index4D i, skepu::Region4D<int> r) { return r(-1, -1, -1, -1) + r(1, 1, 1, 1) + (int)(i.i + i.j + i.k + i.l) * 100; }
skepu::multiple<int, int> cross_two(skepu::Region4D<int> r) { return skepu::ret(r(-1, 0, 0, 0) + r(1, 0, 0, 0), r(0, 0, 0, -1) - r(0, 0, 0, 1)); }
int random_element(skepu::Random<1> &rnd, skepu::Region4D<int> r) { return (int)(rnd.get() % 1000) + r(0, 0, 0, 0); }

auto cross_skeleton = skepu::MapOverlap(cross);
auto cross_indexed_skeleton = skepu::MapOverlap(cross_indexed);
auto cross_two_skeleton = skepu::MapOverlap(cross_two);
auto random_skeleton = skepu::MapOverlap(random_element);

static int input_value(long i, long j, long k, long l) { return (int)((i * 3 + j * 5 + k * 7 + l * 9) % 13) - 4; }

TEST_CASE("mapoverlap_4d_mirror: MapOverlap 4D, mirror mode (edge mode None, input of the size of the output)")
{
	for (auto size : std::vector<std::array<size_t, 4>>{{5, 5, 6, 7}, {8, 9, 6, 5}, {11, 6, 9, 5}, {4, 4, 5, 30}})
	{
		skepu::Tensor4<int> in(size[0], size[1], size[2], size[3]), out(size[0], size[1], size[2], size[3]), out_a(size[0], size[1], size[2], size[3]), out_b(size[0], size[1], size[2], size[3]), out_i(size[0], size[1], size[2], size[3]);
		for (long i = 0; i < (long)size[0]; ++i)
			for (long j = 0; j < (long)size[1]; ++j)
				for (long k = 0; k < (long)size[2]; ++k)
					for (long l = 0; l < (long)size[3]; ++l)
						{
							const long flat = (((i) * (long)size[1] + j) * (long)size[2] + k) * (long)size[3] + l;
							in(i, j, k, l) = input_value(i, j, k, l);
							out(i, j, k, l) = 1000 + (int)flat;
							out_a(i, j, k, l) = -5;
							out_b(i, j, k, l) = 77;
							out_i(i, j, k, l) = 3;
						}
		for (skepu::Tensor4<int> *t : {&in, &out, &out_a, &out_b, &out_i})
			t->invalidateDeviceData();

		cross_skeleton.setOverlap(1, 1, 1, 1);
		cross_skeleton.setEdgeMode(skepu::Edge::None);
		cross_skeleton(out, in);
		cross_indexed_skeleton.setOverlap(1, 1, 1, 1);
		cross_indexed_skeleton.setEdgeMode(skepu::Edge::None);
		cross_indexed_skeleton(out_i, in);
		cross_two_skeleton.setOverlap(1, 1, 1, 1);
		cross_two_skeleton.setEdgeMode(skepu::Edge::None);
		cross_two_skeleton(out_a, out_b, in);
		for (skepu::Tensor4<int> *t : {&out, &out_a, &out_b, &out_i})
			t->updateHost();

		INFO("size = " << size[0] << "x" << size[1] << "x" << size[2] << "x" << size[3]);
		size_t wrong = 0;
		auto at = [&](long i, long j, long k, long l) { return input_value(i, j, k, l); };
		for (long i = 0; i < (long)size[0]; ++i)
			for (long j = 0; j < (long)size[1]; ++j)
				for (long k = 0; k < (long)size[2]; ++k)
					for (long l = 0; l < (long)size[3]; ++l)
						{
							const bool inner = i >= 1 && i < (long)size[0] - 1 && j >= 1 && j < (long)size[1] - 1 && k >= 1 && k < (long)size[2] - 1 && l >= 1 && l < (long)size[3] - 1;
							const long flat = (((i) * (long)size[1] + j) * (long)size[2] + k) * (long)size[3] + l;
							wrong += out(i, j, k, l) != (inner ? at(i-1, j, k, l) + 2 * at(i+1, j, k, l) + at(i, j-1, k, l) + 2 * at(i, j+1, k, l) + at(i, j, k-1, l) + 2 * at(i, j, k+1, l) + at(i, j, k, l-1) + 2 * at(i, j, k, l+1) + 3 * at(i, j, k, l) : 1000 + (int)flat);
							wrong += out_i(i, j, k, l) != (inner ? at(i-1, j-1, k-1, l-1) + at(i+1, j+1, k+1, l+1) + (i + j + k + l) * 100 : 3);
							wrong += out_a(i, j, k, l) != (inner ? at(i-1, j, k, l) + at(i+1, j, k, l) : -5);
							wrong += out_b(i, j, k, l) != (inner ? at(i, j, k, l-1) - at(i, j, k, l+1) : 77);
						}
		CHECK(wrong == 0);
	}
}

TEST_CASE("mapoverlap_4d_mirror: MapOverlap 4D, mirror mode and the random numbers")
{
	// The inner elements get the numbers one after the other, and the stream is advanced over all the elements of the output
	const std::array<size_t, 4> size{8, 9, 6, 5};
	skepu::Tensor4<int> in(size[0], size[1], size[2], size[3], "", 1), out(size[0], size[1], size[2], size[3], "", -1);

	random_skeleton.setOverlap(1, 1, 1, 1);
	random_skeleton.setEdgeMode(skepu::Edge::None);
	skepu::PRNG prng(77), reference(77);
	random_skeleton.setPRNG(prng);
	random_skeleton(out, in);
	out.updateHost();

	size_t wrong = 0, total = 0;
	for (long i = 0; i < (long)size[0]; ++i)
		for (long j = 0; j < (long)size[1]; ++j)
			for (long k = 0; k < (long)size[2]; ++k)
				for (long l = 0; l < (long)size[3]; ++l)
					{
						++total;
						if (i < 1 || i >= (long)size[0] - 1 || j < 1 || j >= (long)size[1] - 1 || k < 1 || k >= (long)size[2] - 1 || l < 1 || l >= (long)size[3] - 1)
							wrong += out(i, j, k, l) != -1;
						else
							wrong += out(i, j, k, l) != (int)(reference.get() % 1000) + 1;
					}
	CHECK(wrong == 0);
	reference.forward(total - (size[0] - 2)*(size[1] - 2)*(size[2] - 2)*(size[3] - 2));
	CHECK(prng.get() == reference.get());
}

TEST_CASE("mapoverlap_4d_mirror: MapOverlap 4D, mirror mode and a dimension that is not larger than twice the overlap")
{
	// No element has its whole overlap in the input: nothing is computed, the output keeps its values, and the random numbers are used
	const std::array<size_t, 4> size{2, 6, 6, 6};
	skepu::Tensor4<int> in(size[0], size[1], size[2], size[3], "", 1), out(size[0], size[1], size[2], size[3], "", 7);
	random_skeleton.setOverlap(1, 1, 1, 1);
	random_skeleton.setEdgeMode(skepu::Edge::None);
	skepu::PRNG prng(5), reference(5);
	random_skeleton.setPRNG(prng);
	random_skeleton(out, in);
	out.updateHost();
	size_t wrong = 0, total = 0;
	for (size_t i = 0; i < out.size(); ++i, ++total)
		wrong += out.getAddress()[i] != 7;
	CHECK(wrong == 0);
	reference.forward(total);
	CHECK(prng.get() == reference.get());
}
}
