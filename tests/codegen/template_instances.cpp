#include "../../external/catch2/catch.hpp"

#include <skepu>

// A function template with a skeleton instance, instantiated twice with the same user
// function (the template parameter only changes the type of the scalar argument).
float scale_uf(float x, float factor)
{
	return x * factor;
}

template<typename S>
void scale(skepu::Vector<float> &v, S factor)
{
	auto skel = skepu::Map<1>(scale_uf);
	skel.setStride(1, 1);
	skel(v, v, factor);
}

TEST_CASE("Skeleton in a function template, instantiated twice")
{
	const size_t size{8};
	skepu::Vector<float> v(size);
	for (size_t i = 0; i < size; ++i)
		v[i] = i + 1;

	scale(v, 2.f);   // S = float
	scale(v, 3);     // S = int

	v.updateHost();
	for (size_t i = 0; i < size; ++i)
		REQUIRE(v(i) == 6.f * (i + 1));
}
