// The "mirror mode" of MapOverlap 4D: the edge mode None and an input of the size of the output; only the interior of the output
// is computed, the elements near the edges keep their values. Same output on a single node and on any number of processes.

#include <array>
#include <iostream>
#include <skepu>

int stencil(skepu::Region4D<int> r) { return r(-1, 0, 0, 0) + 2 * r(1, 0, 0, 0) + r(0, -1, 0, 0) + 2 * r(0, 1, 0, 0) + r(0, 0, -1, 0) + 2 * r(0, 0, 1, 0) + r(0, 0, 0, -1) + 2 * r(0, 0, 0, 1) + r(0, 0, 0, 0); }
int indexed(skepu::Index4D i, skepu::Region4D<int> r) { return r(-1, -1, -1, -1) + r(1, 1, 1, 1) + (int)(i.i + i.j + i.k + i.l) * 100; }
skepu::multiple<int, int> two(skepu::Region4D<int> r) { return skepu::ret(r(-1, 0, 0, 0) + r(1, 0, 0, 0), r(0, 0, 0, -1) - r(0, 0, 0, 1)); }

static unsigned long long checksum(skepu::Tensor4<int> &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

static void run(size_t d0, size_t d1, size_t d2, size_t d3)
{
	std::cout << d0 << " x " << d1 << " x " << d2 << " x " << d3 << "\n";
	skepu::Tensor4<int> in(d0, d1, d2, d3), out(d0, d1, d2, d3), out2(d0, d1, d2, d3);
	skepu::external([&]
	{
		for (size_t i = 0; i < d0; ++i)
			for (size_t j = 0; j < d1; ++j)
				for (size_t k = 0; k < d2; ++k)
					for (size_t l = 0; l < d3; ++l)
						{
							in(i, j, k, l) = (int)((i * 3 + j * 5 + k * 7 + l * 9) % 13) - 4;
							out(i, j, k, l) = 1000 + (int)((((i) * d1 + j) * d2 + k) * d3 + l);
							out2(i, j, k, l) = -5;
						}
	}, skepu::write(in, out, out2));

	skepu::MapOverlap(stencil).setOverlap(1, 1, 1, 1).setEdgeMode(skepu::Edge::None)(out, in);
	std::cout << " one: " << checksum(out) << " " << out(0, 0, 0, 0) << " " << out(d0 / 2, d1 / 2, d2 / 2, d3 / 2) << "\n";

	skepu::MapOverlap(indexed).setOverlap(1, 1, 1, 1).setEdgeMode(skepu::Edge::None)(out, in);
	std::cout << " indexed: " << checksum(out) << "\n";

	skepu::MapOverlap(two).setOverlap(1, 1, 1, 1).setEdgeMode(skepu::Edge::None)(out, out2, in);
	std::cout << " two: " << checksum(out) << " " << checksum(out2) << "\n";
}

int main(int argc, char *argv[])
{
	const std::string be = argc > 1 ? argv[1] : "cpu";
	skepu::BackendSpec spec{be};
	skepu::setGlobalBackendSpec(spec);
	for (auto s : std::vector<std::array<size_t, 4>>{{9, 4, 5, 3}, {16, 3, 3, 4}, {3, 3, 3, 3}, {2, 4, 3, 5}, {12, 3, 4, 3}})
		run(s[0], s[1], s[2], s[3]);
	return 0;
}
