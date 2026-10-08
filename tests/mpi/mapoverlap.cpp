// MapOverlap on distributed containers. Same output on a single node and on any number of processes, with any backend.

#include <iostream>
#include <skepu>

int stencil1d(skepu::Region1D<int> r) { return r(-2) + 2 * r(-1) + 3 * r(0) - r(1) + r(2); }
int stencil1d_indexed(skepu::Index1D i, skepu::Region1D<int> r, int k) { return r(-1) - r(1) + (int)(i.i) * k; }
int stencil2d(skepu::Region2D<int> r)
{
	int sum = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			sum += r(i, j) * (i + 2 * j + 3);
	return sum;
}
int stencil2d_indexed(skepu::Index2D i, skepu::Region2D<int> r) { return r(-1, 0) + r(1, 0) - r(0, -1) + r(0, 1) + (int)(i.row) * 3 + (int)(i.col); }
int stencil2d_random_access(skepu::Region2D<int> r, const skepu::Vec<int> w) { return r(0, 0) * w[0] + r(-1, -1) * w[1] + r(1, 1) * w[2]; }
int stencil3d(skepu::Region3D<int> r) { return r(-1, 0, 0) + 2 * r(1, 0, 0) - r(0, -1, 0) + r(0, 1, 0) + 3 * r(0, 0, -1) - r(0, 0, 1) + r(0, 0, 0); }
int stencil3d_indexed(skepu::Index3D i, skepu::Region3D<int> r) { return r(-1, 0, 0) + r(1, 0, 0) + (int)(i.i) * 100 + (int)(i.j) * 10 + (int)(i.k); }
int stencil4d(skepu::Region4D<int> r) { return r(-1, 0, 0, 0) + 2 * r(1, 0, 0, 0) - r(0, -1, 0, 0) + r(0, 0, 1, 0) + r(0, 0, 0, -1) + r(0, 0, 0, 0); }

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

static const char *name(skepu::Edge e)
{
	switch (e)
	{
	case skepu::Edge::None: return "none";
	case skepu::Edge::Cyclic: return "cyclic";
	case skepu::Edge::Duplicate: return "duplicate";
	default: return "pad";
	}
}

static void vectors(const size_t n)
{
	skepu::Vector<int> in(n), out(n);
	skepu::external([&] { for (size_t i = 0; i < n; ++i) in(i) = (int)((i * 5) % 11) - 3; }, skepu::write(in));
	std::cout << "vector " << n << "\n";
	for (skepu::Edge e : {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad})
	{
		skepu::MapOverlap(stencil1d).setOverlap(2).setEdgeMode(e).setPad(7)(out, in);
		std::cout << " " << name(e) << ": " << checksum(out) << " " << out(0) << " " << out(n - 1) << "\n";
	}
	skepu::MapOverlap(stencil1d_indexed).setOverlap(1).setEdgeMode(skepu::Edge::Duplicate)(out, in, 3);
	std::cout << " indexed: " << checksum(out) << "\n";

	skepu::Vector<int> shrunk(n - 4);
	skepu::MapOverlap(stencil1d).setOverlap(2).setEdgeMode(skepu::Edge::None)(shrunk, in);
	std::cout << " none: " << checksum(shrunk) << "\n";
}

static void matrices(const size_t rows, const size_t cols)
{
	skepu::Matrix<int> in(rows, cols), out(rows, cols);
	skepu::Vector<int> w(3);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				in(i, j) = (int)((i * 7 + j * 3) % 13) - 5;
		w(0) = 2; w(1) = -1; w(2) = 3;
	}, skepu::write(in, w));
	std::cout << "matrix " << rows << "x" << cols << "\n";
	for (skepu::Edge e : {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad})
	{
		skepu::MapOverlap(stencil2d).setOverlap(1, 2).setEdgeMode(e).setPad(-9)(out, in);
		std::cout << " " << name(e) << ": " << checksum(out) << " " << out(0, 0) << " " << out(rows - 1, cols - 1) << "\n";
	}
	skepu::MapOverlap(stencil2d).setOverlap(2).setEdgeMode(skepu::Edge::Cyclic)(out, in);
	std::cout << " overlap 2, cyclic: " << checksum(out) << "\n";
	skepu::MapOverlap(stencil2d_indexed).setOverlap(1).setEdgeMode(skepu::Edge::Pad).setPad(1)(out, in);
	std::cout << " indexed: " << checksum(out) << "\n";
	skepu::MapOverlap(stencil2d_random_access).setOverlap(1).setEdgeMode(skepu::Edge::Duplicate)(out, in, w);
	std::cout << " random access: " << checksum(out) << "\n";

	skepu::Matrix<int> shrunk(rows - 2, cols - 4);
	skepu::MapOverlap(stencil2d).setOverlap(1, 2).setEdgeMode(skepu::Edge::None)(shrunk, in);
	std::cout << " none: " << checksum(shrunk) << "\n";
	// (in place is valid only with the red-black update mode, also on a single node)
}

int main()
{
	vectors(103);
	vectors(11); // fewer rows than some of the processes' overlaps allow
	matrices(14, 9);
	matrices(7, 6);

	{
		const size_t si = 8, sj = 5, sk = 4;
		skepu::Tensor3<int> in(si, sj, sk), out(si, sj, sk);
		skepu::external([&]
		{
			for (size_t i = 0; i < si; ++i)
				for (size_t j = 0; j < sj; ++j)
					for (size_t k = 0; k < sk; ++k)
						in(i, j, k) = (int)((i * 5 + j * 3 + k) % 9) - 2;
		}, skepu::write(in));
		for (skepu::Edge e : {skepu::Edge::Duplicate, skepu::Edge::Cyclic, skepu::Edge::Pad})
		{
			skepu::MapOverlap(stencil3d).setOverlap(1).setEdgeMode(e).setPad(4)(out, in);
			std::cout << "tensor3 " << name(e) << ": " << checksum(out) << "\n";
		}
		skepu::MapOverlap(stencil3d_indexed).setOverlap(1).setEdgeMode(skepu::Edge::Duplicate)(out, in);
		std::cout << "tensor3 indexed: " << checksum(out) << "\n";
	}

	{
		const size_t si = 6, sj = 4, sk = 3, sl = 3;
		skepu::Tensor4<int> in(si, sj, sk, sl), out(si, sj, sk, sl);
		skepu::external([&]
		{
			for (size_t i = 0; i < si; ++i)
				for (size_t j = 0; j < sj; ++j)
					for (size_t k = 0; k < sk; ++k)
						for (size_t l = 0; l < sl; ++l)
							in(i, j, k, l) = (int)((i * 7 + j * 5 + k * 3 + l) % 11) - 3;
		}, skepu::write(in));
		for (skepu::Edge e : {skepu::Edge::Duplicate, skepu::Edge::Cyclic})
		{
			skepu::MapOverlap(stencil4d).setOverlap(1).setEdgeMode(e)(out, in);
			std::cout << "tensor4 " << name(e) << ": " << checksum(out) << "\n";
		}
	}
	return 0;
}
