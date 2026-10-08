// Parts of the interface that a program written for one node uses: containers that get their size in an skepu::external
// call (a file that is read), external calls with a description, an element as a uniform argument, container_like, a start
// value that is a braced list, and randomize (the values of a single node). Same output on a single node and on any number
// of processes, with any backend.

#include <iostream>
#include <skepu>

struct Range
{
	int lo, hi;
};

Range to_range(int v) { Range r; r.lo = v; r.hi = v; return r; }
Range merge(Range a, Range b) { Range r; r.lo = a.lo < b.lo ? a.lo : b.lo; r.hi = a.hi > b.hi ? a.hi : b.hi; return r; }
int scaled(int a, int factor) { return a * factor; }
int add(int a, int b) { return a + b; }
int index_sum(skepu::Index2D i, int a) { return a + (int)(i.row) * 100 + (int)(i.col); }

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

int main()
{
	// containers that get their size in an external call (the first process knows it)
	skepu::Matrix<int> m;
	const size_t rows_to_make = 9, cols_to_make = 4;
	skepu::external("Make the matrix", [&]
	{
		m.init(rows_to_make + 1, cols_to_make + 1);
		for (size_t i = 0; i < m.total_rows(); ++i)
			for (size_t j = 0; j < m.total_cols(); ++j)
				m(i, j) = (int)((i * 3 + j) % 7);
	}, skepu::write(m));
	std::cout << "matrix: " << m.total_rows() << "x" << m.total_cols() << " " << checksum(m) << "\n";

	skepu::Vector<int> v;
	skepu::external([&]
	{
		v.init(37);
		for (size_t i = 0; i < 37; ++i)
			v(i) = (int)((i * i) % 11) - 4;
	}, skepu::write(v));
	std::cout << "vector: " << v.size() << " " << checksum(v) << "\n";

	skepu::Tensor3<int> t;
	skepu::external([&]
	{
		t.init(5, 3, 2);
		for (size_t i = 0; i < 5; ++i)
			for (size_t j = 0; j < 3; ++j)
				for (size_t k = 0; k < 2; ++k)
					t(i, j, k) = (int)(i + 2 * j + 3 * k);
	}, skepu::write(t));
	std::cout << "tensor: " << t.size_i() << "x" << t.size_j() << "x" << t.size_k() << " " << checksum(t) << "\n";

	// the shape is used by skeletons
	skepu::Matrix<int> out(m.total_rows(), m.total_cols());
	skepu::Map<1>(index_sum)(out, m);
	std::cout << "map over it: " << checksum(out) << "\n";

	// external calls with a description, in each form
	long total = 0;
	skepu::external("Read", skepu::read(v), [&] { for (size_t i = 0; i < v.size(); ++i) total += v(i); });
	skepu::Vector<int> filled(v.size());
	skepu::external("Write", [&] { for (size_t i = 0; i < filled.size(); ++i) filled(i) = (int)(2 * i); }, skepu::write(filled));
	skepu::external("Read and write", skepu::read(v), [&] { for (size_t i = 0; i < filled.size(); ++i) filled(i) += v(i); }, skepu::write(filled));
	skepu::external("Nothing", [&] { total += 1000; });
	std::cout << "externals: " << checksum(filled) << "\n";

	// an element as a uniform argument
	skepu::Vector<int> times_element(v.size());
	skepu::Map<1>(scaled)(times_element, v, v.elem(3));
	std::cout << "element: " << checksum(times_element) << "\n";

	// a container of the shape of another, of another element type
	auto floats = skepu::container_like<float>(m);
	auto same = skepu::container_like(m);
	std::cout << "like: " << floats.total_rows() << "x" << floats.total_cols() << " " << same.total_rows() << "x" << same.total_cols() << "\n";

	// a start value that is a braced list
	const Range range = skepu::MapReduce<1>(to_range, merge).setStartValue({100, -100})(v);
	std::cout << "range: " << range.lo << " " << range.hi << "\n";

	// randomize: the values of a single node
	skepu::Vector<int> random_vector(101);
	random_vector.randomize(0, 50);
	skepu::Matrix<int> random_matrix(13, 5);
	random_matrix.randomize(-5, 5);
	skepu::Matrix<float> random_real(6, 7);
	random_real.randomizeReal(0, 10);
	random_real.flush();
	double sum = 0;
	for (size_t i = 0; i < random_real.size(); ++i)
		sum += random_real.getAddress()[i];
	skepu::Tensor3<int> random_tensor(4, 3, 5);
	random_tensor.randomize(0, 9);
	std::cout << "randomize: " << checksum(random_vector) << " " << checksum(random_matrix) << " " << (long)(sum * 1000) << " " << checksum(random_tensor) << "\n";
	std::cout << "after: " << rand() << "\n";
	return 0;
}
