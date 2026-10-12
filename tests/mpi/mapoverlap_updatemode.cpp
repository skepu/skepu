// The update modes of MapOverlap (Red, Black, RedBlack) in all the variants of MapOverlap: 1D (vector, row-wise and column-wise of a
// matrix), 2D, 3D and 4D, with the edge modes Duplicate and None and with the mirror mode (the edge mode None and an input of the size of the
// output), and in place (the output is the input: the Black pass gets the halos with the elements of the Red pass). Same output on a
// single node and on any number of processes.

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
#include <skepu>

int f1(skepu::Index1D i, skepu::Region1D<int> r) { return (int)i.i * 10 + r(0) + r(-1) - r(1); }
int f2(skepu::Index2D i, skepu::Region2D<int> r) { return (int)(i.row * 100 + i.col) * 10 + r(0, 0) + r(-1, 0) - r(0, 1); }
int f3(skepu::Index3D i, skepu::Region3D<int> r) { return (int)(i.i * 100 + i.j * 10 + i.k) * 10 + r(0, 0, 0) + r(-1, 0, 0) - r(0, 0, 1); }
int f4(skepu::Index4D i, skepu::Region4D<int> r) { return (int)(i.i * 1000 + i.j * 100 + i.k * 10 + i.l) * 10 + r(0, 0, 0, 0) + r(-1, 0, 0, 0) - r(0, 0, 0, 1); }

// A relaxation step in place: the average of the neighbours
float relax1(skepu::Region1D<float> r) { return (r(-1) + r(1)) / 2; }
float relax2(skepu::Region2D<float> r) { return (r(-1, 0) + r(1, 0) + r(0, -1) + r(0, 1)) / 4; }

static const skepu::UpdateMode modes[] = {skepu::UpdateMode::Normal, skepu::UpdateMode::Red, skepu::UpdateMode::Black, skepu::UpdateMode::RedBlack};
static std::string mode_name(skepu::UpdateMode m) { return skepu::to_string(m); }

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

static int value(long i, long j = 0, long k = 0, long l = 0) { return (int)((i * 3 + j * 5 + k * 7 + l * 11) % 17) - 6; }

static const skepu::Edge edges[] = {skepu::Edge::Duplicate, skepu::Edge::None};

static void vector_case(size_t n)
{
	for (skepu::Edge edge : edges)
		for (bool mirror : {false, true})
		{
			if (mirror && edge != skepu::Edge::None)
				continue;
			const size_t shrink = (edge == skepu::Edge::None && !mirror) ? 2 : 0;
			if (n <= shrink)
				continue;
			skepu::Vector<int> in(n), out(n - shrink);
			skepu::external([&] { for (size_t i = 0; i < n; ++i) in(i) = value(i); for (size_t i = 0; i < n - shrink; ++i) out(i) = -77; }, skepu::write(in, out));
			for (skepu::UpdateMode mode : modes)
			{
				auto skel = skepu::MapOverlap(f1);
				skel.setOverlap(1).setEdgeMode(edge).setUpdateMode(mode);
				skel(out, in);
				std::cout << " vector " << n << " edge " << (int)edge << " mirror " << mirror << " " << mode_name(mode) << ": " << checksum(out) << "\n";
			}
		}
}

static void matrix_case(size_t rows, size_t cols)
{
	for (skepu::Edge edge : edges)
		for (bool mirror : {false, true})
		{
			if (mirror && edge != skepu::Edge::None)
				continue;
			const size_t shrink = (edge == skepu::Edge::None && !mirror) ? 2 : 0;
			if (rows <= shrink || cols <= shrink)
				continue;
			skepu::Matrix<int> in(rows, cols), out(rows - shrink, cols - shrink), col_out(rows - shrink, cols), row_out(rows, cols - shrink);
			skepu::Matrix<int> out_full(rows, cols);
			skepu::external([&]
			{
				for (size_t i = 0; i < rows; ++i)
					for (size_t j = 0; j < cols; ++j)
						in(i, j) = value(i, j);
				for (size_t i = 0; i < rows - shrink; ++i)
					for (size_t j = 0; j < cols - shrink; ++j)
						out(i, j) = -77;
				for (size_t i = 0; i < rows - shrink; ++i)
					for (size_t j = 0; j < cols; ++j)
						col_out(i, j) = -77;
				for (size_t i = 0; i < rows; ++i)
					for (size_t j = 0; j < cols - shrink; ++j)
						row_out(i, j) = -77;
			}, skepu::write(in, out, col_out, row_out));
			for (skepu::UpdateMode mode : modes)
			{
				auto skel2 = skepu::MapOverlap(f2);
				skel2.setOverlap(1, 1).setEdgeMode(edge).setUpdateMode(mode);
				skel2(out, in);
				std::cout << " matrix " << rows << "x" << cols << " edge " << (int)edge << " mirror " << mirror << " " << mode_name(mode) << ": " << checksum(out) << "\n";

				auto skel1 = skepu::MapOverlap(f1);
				skel1.setOverlap(1).setEdgeMode(edge).setUpdateMode(mode);
				skel1.setOverlapMode(skepu::Overlap::ColWise);
				skel1(col_out, in);
				std::cout << " column-wise " << rows << "x" << cols << " edge " << (int)edge << " mirror " << mirror << " " << mode_name(mode) << ": " << checksum(col_out) << "\n";
				skel1.setOverlapMode(skepu::Overlap::RowWise);
				skel1(row_out, in);
				std::cout << " row-wise " << rows << "x" << cols << " edge " << (int)edge << " mirror " << mirror << " " << mode_name(mode) << ": " << checksum(row_out) << "\n";
			}
		}
}

static void tensor3_case(size_t a, size_t b, size_t c)
{
	for (skepu::Edge edge : edges)
		for (bool mirror : {false, true})
		{
			if (mirror && edge != skepu::Edge::None)
				continue;
			const size_t shrink = (edge == skepu::Edge::None && !mirror) ? 2 : 0;
			if (a <= shrink || b <= shrink || c <= shrink)
				continue;
			skepu::Tensor3<int> in(a, b, c), out(a - shrink, b - shrink, c - shrink);
			skepu::external([&]
			{
				for (size_t i = 0; i < a; ++i)
					for (size_t j = 0; j < b; ++j)
						for (size_t k = 0; k < c; ++k)
							in(i, j, k) = value(i, j, k);
				for (size_t i = 0; i < a - shrink; ++i)
					for (size_t j = 0; j < b - shrink; ++j)
						for (size_t k = 0; k < c - shrink; ++k)
							out(i, j, k) = -77;
			}, skepu::write(in, out));
			for (skepu::UpdateMode mode : modes)
			{
				auto skel = skepu::MapOverlap(f3);
				skel.setOverlap(1, 1, 1).setEdgeMode(edge).setUpdateMode(mode);
				skel(out, in);
				std::cout << " tensor3 " << a << "x" << b << "x" << c << " edge " << (int)edge << " mirror " << mirror << " " << mode_name(mode) << ": " << checksum(out) << "\n";
			}
		}
}

static void tensor4_case(size_t a, size_t b, size_t c, size_t d)
{
	for (skepu::Edge edge : edges)
		for (bool mirror : {false, true})
		{
			if (mirror && edge != skepu::Edge::None)
				continue;
			const size_t shrink = (edge == skepu::Edge::None && !mirror) ? 2 : 0;
			if (a <= shrink || b <= shrink || c <= shrink || d <= shrink)
				continue;
			skepu::Tensor4<int> in(a, b, c, d), out(a - shrink, b - shrink, c - shrink, d - shrink);
			skepu::external([&]
			{
				for (size_t i = 0; i < a; ++i)
					for (size_t j = 0; j < b; ++j)
						for (size_t k = 0; k < c; ++k)
							for (size_t l = 0; l < d; ++l)
								in(i, j, k, l) = value(i, j, k, l);
				for (size_t i = 0; i < a - shrink; ++i)
					for (size_t j = 0; j < b - shrink; ++j)
						for (size_t k = 0; k < c - shrink; ++k)
							for (size_t l = 0; l < d - shrink; ++l)
								out(i, j, k, l) = -77;
			}, skepu::write(in, out));
			for (skepu::UpdateMode mode : modes)
			{
				auto skel = skepu::MapOverlap(f4);
				skel.setOverlap(1, 1, 1, 1).setEdgeMode(edge).setUpdateMode(mode);
				skel(out, in);
				std::cout << " tensor4 " << a << "x" << b << "x" << c << "x" << d << " edge " << (int)edge << " mirror " << mirror << " " << mode_name(mode) << ": " << checksum(out) << "\n";
			}
		}
}

// In place, a few steps of a relaxation: the Black pass uses the elements that the Red pass has computed, also those of the other processes
static void in_place_case(size_t n, size_t rows, size_t cols)
{
	skepu::Vector<float> v(n);
	skepu::Matrix<float> m(rows, cols);
	skepu::external([&]
	{
		for (size_t i = 0; i < n; ++i) v(i) = (float)((i * 7) % 11);
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				m(i, j) = (float)((i * 5 + j * 3) % 13);
	}, skepu::write(v, m));

	auto step1 = skepu::MapOverlap(relax1);
	step1.setOverlap(1).setEdgeMode(skepu::Edge::Duplicate).setUpdateMode(skepu::UpdateMode::RedBlack);
	auto step2 = skepu::MapOverlap(relax2);
	step2.setOverlap(1, 1).setEdgeMode(skepu::Edge::Duplicate).setUpdateMode(skepu::UpdateMode::RedBlack);
	for (int s = 0; s < 5; ++s)
	{
		step1(v, v);
		step2(m, m);
	}
	v.flush();
	m.flush();
	double sv = 0, sm = 0;
	for (size_t i = 0; i < v.size(); ++i) sv += v.getAddress()[i] * (double)(i + 1);
	for (size_t i = 0; i < m.size(); ++i) sm += m.getAddress()[i] * (double)(i + 1);
	std::cout << " in place " << n << " " << rows << "x" << cols << ": " << (long long)(sv * 1000) << " " << (long long)(sm * 1000) << "\n";
}

int main(int argc, char *argv[])
{
	const std::string be = argc > 1 ? argv[1] : "cpu";
	skepu::BackendSpec spec{be};
	skepu::setGlobalBackendSpec(spec);

	for (size_t n : {3, 4, 17, 50})
		vector_case(n);
	for (auto s : std::vector<std::array<size_t, 2>>{{3, 3}, {8, 5}, {20, 7}, {31, 12}})
		matrix_case(s[0], s[1]);
	for (auto s : std::vector<std::array<size_t, 3>>{{3, 3, 3}, {9, 7, 5}, {16, 5, 4}, {5, 5, 9}})
		tensor3_case(s[0], s[1], s[2]);
	for (auto s : std::vector<std::array<size_t, 4>>{{3, 3, 3, 3}, {9, 5, 4, 3}, {6, 4, 5, 7}})
		tensor4_case(s[0], s[1], s[2], s[3]);
	for (auto s : std::vector<std::array<size_t, 3>>{{20, 9, 8}, {33, 16, 5}, {7, 4, 6}})
		in_place_case(s[0], s[1], s[2]);
	return 0;
}
