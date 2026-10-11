#include "../../external/catch2/catch.hpp"

#include <array>
#include <skepu>

// "Mirror mode": the edge mode None with an input that has the size of the output. Only the interior of the output
// (the elements that have the whole overlap in the input) is computed; the other elements keep their values.
int cross(skepu::Region3D<int> r) { return r(-1, 0, 0) + 2 * r(1, 0, 0) + r(0, -1, 0) + 2 * r(0, 1, 0) + r(0, 0, -1) + 2 * r(0, 0, 1) + 3 * r(0, 0, 0); }
int cross_indexed(skepu::Index3D i, skepu::Region3D<int> r) { return r(-1, -1, -1) + r(1, 1, 1) + (int)(i.i + i.j + i.k) * 100; }
skepu::multiple<int, int> cross_two(skepu::Region3D<int> r) { return skepu::ret(r(-1, 0, 0) + r(1, 0, 0), r(0, 0, -1) - r(0, 0, 1)); }
int random_element(skepu::Random<1> &rnd, skepu::Region3D<int> r) { return (int)(rnd.get() % 1000) + r(0, 0, 0); }

auto cross_skeleton = skepu::MapOverlap(cross);
auto cross_indexed_skeleton = skepu::MapOverlap(cross_indexed);
auto cross_two_skeleton = skepu::MapOverlap(cross_two);
auto random_skeleton = skepu::MapOverlap(random_element);

static int input_value(long i, long j, long k) { return (int)((i * 3 + j * 5 + k * 7) % 13) - 4; }

TEST_CASE("MapOverlap 3D, mirror mode (edge mode None, input of the size of the output)")
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

TEST_CASE("MapOverlap 3D, mirror mode and the random numbers")
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

TEST_CASE("MapOverlap 3D, mirror mode and a dimension that is not larger than twice the overlap")
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
