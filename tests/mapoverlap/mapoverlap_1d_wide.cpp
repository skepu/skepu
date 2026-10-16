#include "../../external/catch2/catch.hpp"

#include <skepu>

// A row (row-wise) or a column (column-wise) that is longer than a thread block is in several blocks on a GPU backend:
// the elements at the borders of the blocks need the elements of the neighbouring block. All edge modes, with an
// output that is as big as the input (and shorter by the overlap on each side with the edge mode None).
int window_sum(skepu::Region1D<int> r)
{
	int res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		res += (i + 4) * r(i);
	return res;
}

auto window = skepu::MapOverlap(window_sum);

static int input_value(size_t i, size_t j) { return (int)((i * 7 + j * 13) % 29) - 9; }

// The value of the element at 'pos' of a line of 'n' elements, with the edge mode (pad: 'pad')
static int element(const std::vector<int> &line, long pos, skepu::Edge edge, int pad)
{
	const long n = (long)line.size();
	if (pos >= 0 && pos < n)
		return line[pos];
	switch (edge)
	{
	case skepu::Edge::Duplicate: return line[pos < 0 ? 0 : n - 1];
	case skepu::Edge::Cyclic: return line[((pos % n) + n) % n];
	default: return pad;
	}
}

static std::vector<int> expected_line(const std::vector<int> &line, long overlap, skepu::Edge edge, int pad)
{
	const long n = (long)line.size();
	std::vector<int> out;
	const long first = (edge == skepu::Edge::None) ? overlap : 0, last = (edge == skepu::Edge::None) ? n - overlap : n;
	for (long c = first; c < last; ++c)
	{
		int sum = 0;
		for (long o = -overlap; o <= overlap; ++o)
			sum += (int)(o + 4) * element(line, c + o, edge, pad);
		out.push_back(sum);
	}
	return out;
}

TEST_CASE("MapOverlap 1D on matrices with rows and columns longer than a thread block")
{
	const long overlap = 2;
	const int pad = 5;
	window.setOverlap(overlap);
	window.setPad(pad);

	for (auto size : std::initializer_list<std::pair<size_t, size_t>>{{6, 600}, {600, 6}, {4, 1000}, {7, 513}, {300, 300}})
	{
		const size_t rows = size.first, cols = size.second;
		skepu::Matrix<int> in(rows, cols);
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				in(i, j) = input_value(i, j);
		in.invalidateDeviceData();

		for (skepu::Overlap mode : {skepu::Overlap::RowWise, skepu::Overlap::ColWise})
			for (skepu::Edge edge : {skepu::Edge::None, skepu::Edge::Cyclic, skepu::Edge::Duplicate, skepu::Edge::Pad})
			{
				const bool row_wise = mode == skepu::Overlap::RowWise;
				const bool none = edge == skepu::Edge::None;
				if (none && (row_wise ? cols : rows) <= (size_t)(2 * overlap))
					continue; // (no output)
				const size_t out_rows = (none && !row_wise) ? rows - 2 * overlap : rows;
				const size_t out_cols = (none && row_wise) ? cols - 2 * overlap : cols;
				skepu::Matrix<int> out(out_rows, out_cols, -12345);

				window.setOverlapMode(mode);
				window.setEdgeMode(edge);
				window(out, in);
				out.updateHost();

				INFO("size = " << rows << "x" << cols << ", " << (row_wise ? "row-wise" : "column-wise") << ", edge = " << skepu::to_string(edge));
				size_t wrong = 0;
				if (row_wise)
					for (size_t i = 0; i < rows; ++i)
					{
						std::vector<int> line(cols);
						for (size_t j = 0; j < cols; ++j)
							line[j] = input_value(i, j);
						const std::vector<int> expected = expected_line(line, overlap, edge, pad);
						for (size_t j = 0; j < out_cols; ++j)
							wrong += out(i, j) != expected[j];
					}
				else
					for (size_t j = 0; j < cols; ++j)
					{
						std::vector<int> line(rows);
						for (size_t i = 0; i < rows; ++i)
							line[i] = input_value(i, j);
						const std::vector<int> expected = expected_line(line, overlap, edge, pad);
						for (size_t i = 0; i < out_rows; ++i)
							wrong += out(i, j) != expected[i];
					}
				CHECK(wrong == 0);
			}
	}
}
