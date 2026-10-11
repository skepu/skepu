// The "mirror mode" of MapOverlap 1D (vectors, and matrices row-wise and column-wise): the edge mode None and an input of the size of
// the output; only the interior of the output is computed, the elements near the edges keep their values. Same output on a single
// node and on any number of processes.

#include <iostream>
#include <skepu>

int stencil(skepu::Region1D<int> r) { return r(-2) + 3 * r(-1) + 5 * r(0) - 7 * r(1) + 11 * r(2); }
int indexed(skepu::Index1D i, skepu::Region1D<int> r) { return r(-1) + r(1) + (int)(i.i) * 100; }
skepu::multiple<int, int> two(skepu::Region1D<int> r) { return skepu::ret(r(-1) + r(1), r(0) - r(1)); }

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

static void vectors(size_t n)
{
	std::cout << "vector " << n << "\n";
	skepu::Vector<int> in(n), out(n), out2(n);
	skepu::external([&]
	{
		for (size_t i = 0; i < n; ++i)
		{
			in(i) = (int)((i * 7) % 13) - 4;
			out(i) = 1000 + (int)i;
			out2(i) = -5;
		}
	}, skepu::write(in, out, out2));

	skepu::MapOverlap(stencil).setOverlap(2).setEdgeMode(skepu::Edge::None)(out, in);
	std::cout << " one: " << checksum(out) << " " << out(0) << " " << out(n / 2) << "\n";
	skepu::MapOverlap(indexed).setOverlap(1).setEdgeMode(skepu::Edge::None)(out, in);
	std::cout << " indexed: " << checksum(out) << "\n";
	skepu::MapOverlap(two).setOverlap(1).setEdgeMode(skepu::Edge::None)(out, out2, in);
	std::cout << " two: " << checksum(out) << " " << checksum(out2) << "\n";
}

static void matrices(size_t rows, size_t cols, skepu::Overlap mode)
{
	std::cout << "matrix " << rows << " x " << cols << (mode == skepu::Overlap::RowWise ? " row-wise" : " column-wise") << "\n";
	skepu::Matrix<int> in(rows, cols), out(rows, cols), out2(rows, cols);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
			{
				in(i, j) = (int)((i * 7 + j * 3) % 13) - 4;
				out(i, j) = 1000 + (int)(i * cols + j);
				out2(i, j) = -5;
			}
	}, skepu::write(in, out, out2));

	skepu::MapOverlap(stencil).setOverlap(2).setOverlapMode(mode).setEdgeMode(skepu::Edge::None)(out, in);
	std::cout << " one: " << checksum(out) << " " << out(0, 0) << " " << out(rows / 2, cols / 2) << "\n";
	skepu::MapOverlap(indexed).setOverlap(1).setOverlapMode(mode).setEdgeMode(skepu::Edge::None)(out, in);
	std::cout << " indexed: " << checksum(out) << "\n";
	skepu::MapOverlap(two).setOverlap(1).setOverlapMode(mode).setEdgeMode(skepu::Edge::None)(out, out2, in);
	std::cout << " two: " << checksum(out) << " " << checksum(out2) << "\n";
}

int main(int argc, char *argv[])
{
	const std::string be = argc > 1 ? argv[1] : "cpu";
	skepu::BackendSpec spec{be};
	skepu::setGlobalBackendSpec(spec);
	for (size_t n : {9, 16, 5, 4, 3, 40, 103})
		vectors(n);
	for (auto s : {std::pair<size_t, size_t>{9, 7}, {16, 5}, {5, 5}, {2, 6}, {40, 3}})
	{
		matrices(s.first, s.second, skepu::Overlap::RowWise);
		matrices(s.first, s.second, skepu::Overlap::ColWise);
	}
	return 0;
}
