// The 1D overlap of a matrix, along the rows or along the columns. Same output on a single node and on any number of
// processes, with any backend.

#include <iostream>
#include <skepu>

int smooth(skepu::Region1D<int> r) { return r(-2) + 2 * r(-1) + 3 * r(0) - r(1) + r(2); }
int smooth_indexed(skepu::Index1D i, skepu::Region1D<int> r, int k) { return r(-1) - r(1) + (int)(i.i) * k; }

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

static void matrices(const size_t rows, const size_t cols)
{
	skepu::Matrix<int> in(rows, cols), out(rows, cols);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				in(i, j) = (int)((i * 7 + j * 3) % 13) - 5;
	}, skepu::write(in));
	std::cout << "matrix " << rows << "x" << cols << "\n";

	for (skepu::Overlap policy : {skepu::Overlap::RowWise, skepu::Overlap::ColWise})
	{
		const char *name = policy == skepu::Overlap::RowWise ? "row-wise" : "col-wise";
		for (skepu::Edge e : {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad})
		{
			skepu::MapOverlap(smooth).setOverlap(2).setOverlapMode(policy).setEdgeMode(e).setPad(-7)(out, in);
			std::cout << " " << name << " " << (int)e << ": " << checksum(out) << " " << out(0, 0) << " " << out(rows - 1, cols - 1) << "\n";
		}
		skepu::MapOverlap(smooth_indexed).setOverlap(1).setOverlapMode(policy).setEdgeMode(skepu::Edge::Duplicate)(out, in, 4);
		std::cout << " " << name << " indexed: " << checksum(out) << "\n";
	}

	// shrinking output
	skepu::Matrix<int> fewer_rows(rows - 4, cols), fewer_cols(rows, cols - 4);
	skepu::MapOverlap(smooth).setOverlap(2).setOverlapMode(skepu::Overlap::ColWise).setEdgeMode(skepu::Edge::None)(fewer_rows, in);
	skepu::MapOverlap(smooth).setOverlap(2).setOverlapMode(skepu::Overlap::RowWise).setEdgeMode(skepu::Edge::None)(fewer_cols, in);
	std::cout << " none: " << checksum(fewer_rows) << " " << checksum(fewer_cols) << "\n";
}

int main()
{
	matrices(14, 9);
	matrices(7, 6); // fewer rows than the overlap needs from one process
	return 0;
}
