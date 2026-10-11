#include "../../external/catch2/catch.hpp"

#include <array>
#include <skepu>

// "Mirror mode": the edge mode None with an input that has the size of the output. Only the interior of the output
// (the elements that have the whole overlap in the input) is computed; the other elements keep their values.
int cross(skepu::Region4D<int> r) { return r(-1, 0, 0, 0) + 2 * r(1, 0, 0, 0) + r(0, -1, 0, 0) + 2 * r(0, 1, 0, 0) + r(0, 0, -1, 0) + 2 * r(0, 0, 1, 0) + r(0, 0, 0, -1) + 2 * r(0, 0, 0, 1) + 3 * r(0, 0, 0, 0); }
int cross_indexed(skepu::Index4D i, skepu::Region4D<int> r) { return r(-1, -1, -1, -1) + r(1, 1, 1, 1) + (int)(i.i + i.j + i.k + i.l) * 100; }
skepu::multiple<int, int> cross_two(skepu::Region4D<int> r) { return skepu::ret(r(-1, 0, 0, 0) + r(1, 0, 0, 0), r(0, 0, 0, -1) - r(0, 0, 0, 1)); }
int random_element(skepu::Random<1> &rnd, skepu::Region4D<int> r) { return (int)(rnd.get() % 1000) + r(0, 0, 0, 0); }

auto cross_skeleton = skepu::MapOverlap(cross);
auto cross_indexed_skeleton = skepu::MapOverlap(cross_indexed);
auto cross_two_skeleton = skepu::MapOverlap(cross_two);
auto random_skeleton = skepu::MapOverlap(random_element);

static int input_value(long i, long j, long k, long l) { return (int)((i * 3 + j * 5 + k * 7 + l * 9) % 13) - 4; }

TEST_CASE("MapOverlap 4D, mirror mode (edge mode None, input of the size of the output)")
{
	for (auto size : std::vector<std::array<size_t, 4>>{{5, 5, 6, 7}, {8, 9, 6, 5}, {11, 6, 9, 5}, {4, 4, 5, 30}})
	{
		skepu::Tensor4<int> in(size[0], size[1], size[2], size[3]), out(size[0], size[1], size[2], size[3]), out_a(size[0], size[1], size[2], size[3]), out_b(size[0], size[1], size[2], size[3]), out_i(size[0], size[1], size[2], size[3]);
		for (long i = 0; i < (long)size[0]; ++i)
			for (long j = 0; j < (long)size[1]; ++j)
				for (long k = 0; k < (long)size[2]; ++k)
					for (long l = 0; l < (long)size[3]; ++l)
						{
							const long flat = (((i) * (long)size[1] + j) * (long)size[2] + k) * (long)size[3] + l;
							in(i, j, k, l) = input_value(i, j, k, l);
							out(i, j, k, l) = 1000 + (int)flat;
							out_a(i, j, k, l) = -5;
							out_b(i, j, k, l) = 77;
							out_i(i, j, k, l) = 3;
						}
		for (skepu::Tensor4<int> *t : {&in, &out, &out_a, &out_b, &out_i})
			t->invalidateDeviceData();

		cross_skeleton.setOverlap(1, 1, 1, 1);
		cross_skeleton.setEdgeMode(skepu::Edge::None);
		cross_skeleton(out, in);
		cross_indexed_skeleton.setOverlap(1, 1, 1, 1);
		cross_indexed_skeleton.setEdgeMode(skepu::Edge::None);
		cross_indexed_skeleton(out_i, in);
		cross_two_skeleton.setOverlap(1, 1, 1, 1);
		cross_two_skeleton.setEdgeMode(skepu::Edge::None);
		cross_two_skeleton(out_a, out_b, in);
		for (skepu::Tensor4<int> *t : {&out, &out_a, &out_b, &out_i})
			t->updateHost();

		INFO("size = " << size[0] << "x" << size[1] << "x" << size[2] << "x" << size[3]);
		size_t wrong = 0;
		auto at = [&](long i, long j, long k, long l) { return input_value(i, j, k, l); };
		for (long i = 0; i < (long)size[0]; ++i)
			for (long j = 0; j < (long)size[1]; ++j)
				for (long k = 0; k < (long)size[2]; ++k)
					for (long l = 0; l < (long)size[3]; ++l)
						{
							const bool inner = i >= 1 && i < (long)size[0] - 1 && j >= 1 && j < (long)size[1] - 1 && k >= 1 && k < (long)size[2] - 1 && l >= 1 && l < (long)size[3] - 1;
							const long flat = (((i) * (long)size[1] + j) * (long)size[2] + k) * (long)size[3] + l;
							wrong += out(i, j, k, l) != (inner ? at(i-1, j, k, l) + 2 * at(i+1, j, k, l) + at(i, j-1, k, l) + 2 * at(i, j+1, k, l) + at(i, j, k-1, l) + 2 * at(i, j, k+1, l) + at(i, j, k, l-1) + 2 * at(i, j, k, l+1) + 3 * at(i, j, k, l) : 1000 + (int)flat);
							wrong += out_i(i, j, k, l) != (inner ? at(i-1, j-1, k-1, l-1) + at(i+1, j+1, k+1, l+1) + (i + j + k + l) * 100 : 3);
							wrong += out_a(i, j, k, l) != (inner ? at(i-1, j, k, l) + at(i+1, j, k, l) : -5);
							wrong += out_b(i, j, k, l) != (inner ? at(i, j, k, l-1) - at(i, j, k, l+1) : 77);
						}
		CHECK(wrong == 0);
	}
}

TEST_CASE("MapOverlap 4D, mirror mode and the random numbers")
{
	// The inner elements get the numbers one after the other, and the stream is advanced over all the elements of the output
	const std::array<size_t, 4> size{8, 9, 6, 5};
	skepu::Tensor4<int> in(size[0], size[1], size[2], size[3], "", 1), out(size[0], size[1], size[2], size[3], "", -1);

	random_skeleton.setOverlap(1, 1, 1, 1);
	random_skeleton.setEdgeMode(skepu::Edge::None);
	skepu::PRNG prng(77), reference(77);
	random_skeleton.setPRNG(prng);
	random_skeleton(out, in);
	out.updateHost();

	size_t wrong = 0, total = 0;
	for (long i = 0; i < (long)size[0]; ++i)
		for (long j = 0; j < (long)size[1]; ++j)
			for (long k = 0; k < (long)size[2]; ++k)
				for (long l = 0; l < (long)size[3]; ++l)
					{
						++total;
						if (i < 1 || i >= (long)size[0] - 1 || j < 1 || j >= (long)size[1] - 1 || k < 1 || k >= (long)size[2] - 1 || l < 1 || l >= (long)size[3] - 1)
							wrong += out(i, j, k, l) != -1;
						else
							wrong += out(i, j, k, l) != (int)(reference.get() % 1000) + 1;
					}
	CHECK(wrong == 0);
	reference.forward(total - (size[0] - 2)*(size[1] - 2)*(size[2] - 2)*(size[3] - 2));
	CHECK(prng.get() == reference.get());
}

TEST_CASE("MapOverlap 4D, mirror mode and a dimension that is not larger than twice the overlap")
{
	// No element has its whole overlap in the input: nothing is computed, the output keeps its values, and the random numbers are used
	const std::array<size_t, 4> size{2, 6, 6, 6};
	skepu::Tensor4<int> in(size[0], size[1], size[2], size[3], "", 1), out(size[0], size[1], size[2], size[3], "", 7);
	random_skeleton.setOverlap(1, 1, 1, 1);
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
