#include "../../external/catch2/catch.hpp"

#include <skepu>

// MapPairs in place mode adds the result to the old value of the output element
// (sequential, CPU, OpenMP, OpenCL and CUDA backends).
float product_f(float x, float y)
{
	return x * y;
}

auto outer_product = skepu::MapPairs<1, 1>(product_f);

TEST_CASE("MapPairs in place")
{
	const size_t m{4}, n{3};
	skepu::Vector<float> x(m), y(n);
	skepu::Matrix<float> a(m, n);
	for (size_t i = 0; i < m; ++i) x[i] = i + 1;
	for (size_t j = 0; j < n; ++j) y[j] = 10 * (j + 1);
	for (size_t i = 0; i < m * n; ++i) a[i] = 100;

	outer_product.setInPlace(true);
	outer_product(a, x, y);
	a.updateHost();
	for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			REQUIRE(a(i, j) == 100 + (i + 1) * 10 * (j + 1));

	outer_product.setInPlace(false);
	outer_product(a, x, y);
	a.updateHost();
	for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			REQUIRE(a(i, j) == (i + 1) * 10 * (j + 1));
}
