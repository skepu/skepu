// The "mirror mode" of MapOverlap 2D: the edge mode None and an input of the size of the output; only the interior of the output
// is computed, the elements near the edges keep their values. (The node-level skeleton has it with the sequential and OpenMP
// backends.) Same output on a single node and on any number of processes.

#include <iostream>
#include <skepu>

int stencil(skepu::Region2D<int> r) { return r(-1, 0) + 2 * r(1, 0) - r(0, -1) + r(0, 1) + r(0, 0); }
int indexed(skepu::Index2D i, skepu::Region2D<int> r) { return r(-1, -1) + r(1, 1) + (int)(i.row) * 100 + (int)(i.col); }
skepu::multiple<int, int> two(skepu::Region2D<int> r) { return skepu::ret(r(-1, 0) + r(1, 0), r(0, -1) - r(0, 1)); }

static unsigned long long checksum(skepu::Matrix<int> &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

static void run(size_t rows, size_t cols)
{
	std::cout << rows << " x " << cols << "\n";
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

	skepu::MapOverlap(stencil).setOverlap(1, 1).setEdgeMode(skepu::Edge::None)(out, in);
	std::cout << " one: " << checksum(out) << " " << out(0, 0) << " " << out(rows / 2, cols / 2) << "\n";

	skepu::MapOverlap(indexed).setOverlap(1, 1).setEdgeMode(skepu::Edge::None)(out, in);
	std::cout << " indexed: " << checksum(out) << "\n";

	skepu::MapOverlap(two).setOverlap(1, 1).setEdgeMode(skepu::Edge::None)(out, out2, in);
	std::cout << " two: " << checksum(out) << " " << checksum(out2) << "\n";
}

int main(int argc, char *argv[])
{
	const std::string be = argc > 1 ? argv[1] : "cpu";
	skepu::BackendSpec spec{be};
	skepu::setGlobalBackendSpec(spec);
	for (auto s : {std::pair<size_t, size_t>{9, 7}, {16, 5}, {3, 3}, {2, 6}, {40, 3}})
		run(s.first, s.second);
	return 0;
}
