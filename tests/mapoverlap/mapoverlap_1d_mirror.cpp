#include "../../external/catch2/catch.hpp"

#include <skepu>

// "Mirror mode": the edge mode None with an input that has the size of the output. Only the interior of the output
// (the elements that have the whole overlap in the input) is computed; the other elements keep their values.
// Vectors, and matrices row-wise and column-wise.
int cross(skepu::Region1D<int> r) { return r(-2) + 3 * r(-1) + 5 * r(0) - 7 * r(1) + 11 * r(2); }
int cross_indexed(skepu::Index1D i, skepu::Region1D<int> r) { return r(-1) + r(1) + (int)(i.i) * 100; }
skepu::multiple<int, int> cross_two(skepu::Region1D<int> r) { return skepu::ret(r(-1) + r(1), r(0) - r(1)); }
int random_element(skepu::Random<1> &rnd, skepu::Region1D<int> r) { return (int)(rnd.get() % 1000) + r(0); }

auto cross_skeleton = skepu::MapOverlap(cross);
auto cross_indexed_skeleton = skepu::MapOverlap(cross_indexed);
auto cross_two_skeleton = skepu::MapOverlap(cross_two);
auto random_skeleton = skepu::MapOverlap(random_element);

static int input_value(long i, long j = 0) { return (int)((i * 7 + j * 3) % 13) - 4; }

TEST_CASE("MapOverlap 1D, mirror mode on vectors")
{
	for (size_t size : {9, 40, 300, 1000, 5000})
	{
		skepu::Vector<int> in(size), out(size), out_a(size), out_b(size), out_i(size);
		for (size_t i = 0; i < size; ++i)
		{
			in(i) = input_value(i);
			out(i) = 1000 + (int)i;
			out_a(i) = -5;
			out_b(i) = 77;
			out_i(i) = 3;
		}
		for (skepu::Vector<int> *v : {&in, &out, &out_a, &out_b, &out_i})
			v->invalidateDeviceData();

		cross_skeleton.setOverlap(2);
		cross_skeleton.setEdgeMode(skepu::Edge::None);
		cross_skeleton(out, in);
		cross_indexed_skeleton.setOverlap(1);
		cross_indexed_skeleton.setEdgeMode(skepu::Edge::None);
		cross_indexed_skeleton(out_i, in);
		cross_two_skeleton.setOverlap(1);
		cross_two_skeleton.setEdgeMode(skepu::Edge::None);
		cross_two_skeleton(out_a, out_b, in);
		for (skepu::Vector<int> *v : {&out, &out_a, &out_b, &out_i})
			v->updateHost();

		INFO("size = " << size);
		size_t wrong = 0;
		auto at = [&](long i) { return input_value(i); };
		for (long i = 0; i < (long)size; ++i)
		{
			const bool inner2 = i >= 2 && i < (long)size - 2;
			const bool inner1 = i >= 1 && i < (long)size - 1;
			wrong += out(i) != (inner2 ? at(i - 2) + 3 * at(i - 1) + 5 * at(i) - 7 * at(i + 1) + 11 * at(i + 2) : 1000 + (int)i);
			wrong += out_i(i) != (inner1 ? at(i - 1) + at(i + 1) + (int)i * 100 : 3);
			wrong += out_a(i) != (inner1 ? at(i - 1) + at(i + 1) : -5);
			wrong += out_b(i) != (inner1 ? at(i) - at(i + 1) : 77);
		}
		CHECK(wrong == 0);
	}
}

TEST_CASE("MapOverlap 1D, mirror mode on matrices, row-wise and column-wise")
{
	for (auto size : {std::pair<size_t, size_t>{9, 7}, {40, 40}, {100, 70}, {17, 300}, {300, 17}})
		for (auto overlapMode : {skepu::Overlap::RowWise, skepu::Overlap::ColWise})
		{
			const size_t rows = size.first, cols = size.second;
			const bool rowwise = overlapMode == skepu::Overlap::RowWise;
			skepu::Matrix<int> in(rows, cols), out(rows, cols), out_a(rows, cols), out_b(rows, cols), out_i(rows, cols);
			for (size_t i = 0; i < rows; ++i)
				for (size_t j = 0; j < cols; ++j)
				{
					in(i, j) = input_value(i, j);
					out(i, j) = 1000 + (int)(i * cols + j);
					out_a(i, j) = -5;
					out_b(i, j) = 77;
					out_i(i, j) = 3;
				}
			for (skepu::Matrix<int> *m : {&in, &out, &out_a, &out_b, &out_i})
				m->invalidateDeviceData();

			auto configure = [&](auto &skeleton)
			{
				skeleton.setEdgeMode(skepu::Edge::None);
				skeleton.setOverlapMode(overlapMode);
			};
			configure(cross_skeleton);
			configure(cross_indexed_skeleton);
			configure(cross_two_skeleton);
			cross_skeleton.setOverlap(2);
			cross_skeleton(out, in);
			cross_indexed_skeleton.setOverlap(1);
			cross_indexed_skeleton(out_i, in);
			cross_two_skeleton.setOverlap(1);
			cross_two_skeleton(out_a, out_b, in);
			for (skepu::Matrix<int> *m : {&out, &out_a, &out_b, &out_i})
				m->updateHost();

			INFO("size = " << rows << "x" << cols << (rowwise ? ", row-wise" : ", column-wise"));
			size_t wrong = 0;
			auto at = [&](long i, long j) { return input_value(i, j); };
			// (along the direction of the overlap; Index1D is the position in it)
			auto neighbour = [&](long i, long j, long d) { return rowwise ? at(i, j + d) : at(i + d, j); };
			for (long i = 0; i < (long)rows; ++i)
				for (long j = 0; j < (long)cols; ++j)
				{
					const long position = rowwise ? j : i, length = rowwise ? (long)cols : (long)rows;
					const bool inner2 = position >= 2 && position < length - 2;
					const bool inner1 = position >= 1 && position < length - 1;
					wrong += out(i, j) != (inner2 ? neighbour(i, j, -2) + 3 * neighbour(i, j, -1) + 5 * neighbour(i, j, 0) - 7 * neighbour(i, j, 1) + 11 * neighbour(i, j, 2) : 1000 + (int)(i * cols + j));
					wrong += out_i(i, j) != (inner1 ? neighbour(i, j, -1) + neighbour(i, j, 1) + (int)position * 100 : 3);
					wrong += out_a(i, j) != (inner1 ? neighbour(i, j, -1) + neighbour(i, j, 1) : -5);
					wrong += out_b(i, j) != (inner1 ? neighbour(i, j, 0) - neighbour(i, j, 1) : 77);
				}
			CHECK(wrong == 0);
		}
}

TEST_CASE("MapOverlap 1D, mirror mode and the random numbers")
{
	// The inner elements get the numbers one after the other, and the stream is advanced over all the elements of the output
	const size_t size{103};
	const long overlap{2};
	skepu::Vector<int> in(size, 1), out(size, -1);

	random_skeleton.setOverlap(overlap);
	random_skeleton.setEdgeMode(skepu::Edge::None);
	skepu::PRNG prng(77), reference(77);
	random_skeleton.setPRNG(prng);
	random_skeleton(out, in);
	out.updateHost();

	size_t wrong = 0;
	for (long i = 0; i < (long)size; ++i)
		if (i < overlap || i >= (long)size - overlap)
			wrong += out(i) != -1;
		else
			wrong += out(i) != (int)(reference.get() % 1000) + 1;
	CHECK(wrong == 0);
	reference.forward(2 * overlap);
	CHECK(prng.get() == reference.get());
}

TEST_CASE("MapOverlap 1D, mirror mode and a dimension that is not larger than twice the overlap")
{
	// No element has its whole overlap in the input: nothing is computed, the output keeps its values, and the random numbers are used
	random_skeleton.setOverlap(2);
	random_skeleton.setEdgeMode(skepu::Edge::None);

	for (size_t size : {1, 3, 4})
	{
		skepu::Vector<int> in(size, 1), out(size, 7);
		skepu::PRNG prng(5), reference(5);
		random_skeleton.setPRNG(prng);
		random_skeleton(out, in);
		out.updateHost();
		for (size_t i = 0; i < size; ++i)
			CHECK(out(i) == 7);
		reference.forward(size);
		CHECK(prng.get() == reference.get());
	}

	for (auto overlapMode : {skepu::Overlap::RowWise, skepu::Overlap::ColWise})
	{
		const bool rowwise = overlapMode == skepu::Overlap::RowWise;
		const size_t rows = rowwise ? 6 : 4, cols = rowwise ? 4 : 6;   // (the dimension of the overlap is 4)
		skepu::Matrix<int> in(rows, cols, 1), out(rows, cols, 7);
		random_skeleton.setOverlapMode(overlapMode);
		skepu::PRNG prng(5), reference(5);
		random_skeleton.setPRNG(prng);
		random_skeleton(out, in);
		out.updateHost();
		size_t wrong = 0;
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				wrong += out(i, j) != 7;
		CHECK(wrong == 0);
		reference.forward(rows * cols);
		CHECK(prng.get() == reference.get());
	}
}
