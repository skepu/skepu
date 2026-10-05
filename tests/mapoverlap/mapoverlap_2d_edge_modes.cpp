#include "../../external/catch2/catch.hpp"

#include <skepu>

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

TEST_CASE("MapOverlap 2D, edge modes with input and output of the same size")
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
