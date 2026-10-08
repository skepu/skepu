// A program written for one node, with matrices and tensors. It must give the same output built for a single node and
// built for the MPI backend (skepu-tool -mpi), on any number of processes.

#include <iostream>
#include <skepu>

int add(int a, int b) { return a + b; }
int maximum(int a, int b) { return a > b ? a : b; }
int scaled(int a, int factor) { return a * factor; }

int index2(skepu::Index2D i, int a) { return (int)(i.row) * 100 + (int)(i.col) + a; }
int reversed_rows(skepu::Index2D i, const skepu::Mat<int> m) { return m(m.rows - 1 - i.row, i.col); }
int row_sum(skepu::Index1D i, const skepu::MatRow<int> r)
{
	int sum = 0;
	for (size_t j = 0; j < r.cols; ++j)
		sum += r[j];
	return sum + int(i.i);
}
int index3(skepu::Index3D i, int a) { return (int)(i.i) * 10000 + (int)(i.j) * 100 + (int)(i.k) + a; }
int reversed_planes(skepu::Index3D i, const skepu::Ten3<int> t) { return t(t.size_i - 1 - i.i, i.j, i.k); }
int index4(skepu::Index4D i, int a) { return (int)(i.i) * 1000 + (int)(i.j) * 100 + (int)(i.k) * 10 + (int)(i.l) + a; }
int reversed_blocks(skepu::Index4D i, const skepu::Ten4<int> t) { return t(t.size_i - 1 - i.i, i.j, i.k, i.l); }

static unsigned long long checksum(const int *data, size_t n)
{
	unsigned long long h = 17;
	for (size_t i = 0; i < n; ++i)
		h = h * 31 + (unsigned long long)(long long)data[i];
	return h;
}

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	return checksum(c.getAddress(), c.size());
}

static void matrices(const size_t rows, const size_t cols)
{
	skepu::Matrix<int> a(rows, cols), b(rows, cols), out(rows, cols);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
			{
				a(i, j) = int((i * 7 + j * 3) % 11) - 4;
				b(i, j) = int((i + j) % 5);
			}
	}, skepu::write(a, b));

	skepu::Map<2>(add)(out, a, b);
	out.flush();
	std::cout << rows << "x" << cols << " add: " << checksum(out) << " " << out(0, 0) << " " << out(rows - 1, cols - 1) << "\n";

	skepu::Map<1>(index2)(out, a);
	out.flush();
	std::cout << " index: " << checksum(out) << " " << out(rows - 1, cols - 1) << "\n";

	skepu::Map<1>(scaled)(out, b, -2);
	std::cout << " scaled: " << checksum(out) << "\n";

	// random access to a matrix
	skepu::Matrix<int> rev(rows, cols);
	skepu::Map<0>(reversed_rows)(rev, a);
	rev.flush();
	std::cout << " reversed: " << checksum(rev) << " " << rev(0, 0) << "\n";

	// a row of a matrix for each element of a vector
	skepu::Vector<int> sums(rows);
	skepu::Map<0>(row_sum)(sums, a);
	sums.flush();
	std::cout << " row sums: " << checksum(sums) << " " << sums(0) << " " << sums(rows - 1) << "\n";

	// the whole matrix
	const int total = skepu::Reduce(add)(a);
	const int biggest = skepu::Reduce(maximum)(b);
	std::cout << " reduce: " << total << " " << biggest << "\n";

	// the host changes elements between skeletons
	skepu::external(skepu::read(a), [&]
	{
		a(rows - 1, 0) = 1000;
		a(0, cols - 1) = -1000;
	}, skepu::write(a));
	const int changed = skepu::Reduce(add)(a);
	skepu::Map<2>(add)(out, a, b);
	std::cout << " changed: " << changed << " " << checksum(out) << "\n";

	out.flush();
	skepu::Matrix<int> copy(out);
	skepu::external([&] { out(0, 0) = 31337; }, skepu::write(out));
	copy.flush();
	out.flush();
	std::cout << " copy: " << copy(0, 0) << " " << out(0, 0) << " " << checksum(copy) << "\n";

	// code that is not parallel
	skepu::Matrix<int> filled(rows, cols);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				filled(i, j) = int((i * i + j) % 13);
	}, skepu::write(filled));
	skepu::external(skepu::read(filled), [&]
	{
		long total = 0;
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				total += filled(i, j);
		std::cout << " external: " << total << "\n";
	});
	skepu::Map<2>(add)(out, filled, filled);
	std::cout << " written: " << checksum(filled) << " " << checksum(out) << "\n";
}

int main()
{
	matrices(12, 7);
	matrices(3, 5); // fewer rows than processes

	{
		const size_t si = 9, sj = 4, sk = 5;
		skepu::Tensor3<int> a(si, sj, sk), out(si, sj, sk), rev(si, sj, sk);
		skepu::external([&]
		{
			for (size_t i = 0; i < si; ++i)
				for (size_t j = 0; j < sj; ++j)
					for (size_t k = 0; k < sk; ++k)
						a(i, j, k) = int((i * 5 + j * 3 + k) % 17) - 3;
		}, skepu::write(a));
		skepu::Map<1>(index3)(out, a);
		out.flush();
		std::cout << "tensor3 index: " << checksum(out) << " " << out(si - 1, sj - 1, sk - 1) << "\n";
		skepu::Map<0>(reversed_planes)(rev, a);
		rev.flush();
		std::cout << "tensor3 reversed: " << checksum(rev) << " " << rev(0, 0, 0) << "\n";
		std::cout << "tensor3 reduce: " << int(skepu::Reduce(add)(a)) << "\n";
		skepu::external(skepu::read(a), [&] { a(0, 0, 0) = 500; }, skepu::write(a));
		skepu::Map<2>(add)(out, a, a);
		std::cout << "tensor3 changed: " << checksum(out) << "\n";
	}

	{
		const size_t si = 5, sj = 3, sk = 4, sl = 2;
		skepu::Tensor4<int> a(si, sj, sk, sl), out(si, sj, sk, sl), rev(si, sj, sk, sl);
		skepu::external([&]
		{
			for (size_t i = 0; i < si; ++i)
				for (size_t j = 0; j < sj; ++j)
					for (size_t k = 0; k < sk; ++k)
						for (size_t l = 0; l < sl; ++l)
							a(i, j, k, l) = int((i * 7 + j * 5 + k * 3 + l) % 19) - 6;
		}, skepu::write(a));
		skepu::Map<1>(index4)(out, a);
		out.flush();
		std::cout << "tensor4 index: " << checksum(out) << " " << out(si - 1, sj - 1, sk - 1, sl - 1) << "\n";
		skepu::Map<0>(reversed_blocks)(rev, a);
		rev.flush();
		std::cout << "tensor4 reversed: " << checksum(rev) << " " << rev(0, 0, 0, 0) << "\n";
		std::cout << "tensor4 reduce: " << int(skepu::Reduce(maximum)(a)) << "\n";
	}
	return 0;
}
