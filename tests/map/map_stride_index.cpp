#include "../../external/catch2/catch.hpp"

#include <skepu>

// The index of an element of a strided Map is its number in the strided sequence, on all
// backends (not its position in the vector).
int index_f(skepu::Index1D index, int x)
{
	return index.i * 10 + x;
}

auto strided_index = skepu::Map<1>(index_f);

TEST_CASE("Index of a strided Map")
{
	const size_t n{7};
	for (int stride : {1, 2, 3})
	{
		skepu::Vector<int> in(n * stride), out(n * stride);
		for (size_t i = 0; i < n * stride; ++i)
		{
			in[i] = 1;
			out[i] = -1;
		}
		strided_index.setStride(stride, stride);
		strided_index(out, in);
		out.updateHost();
		for (size_t i = 0; i < n * stride; ++i)
			REQUIRE(out(i) == ((i % stride == 0) ? (int)(i / stride) * 10 + 1 : -1));
	}
}
