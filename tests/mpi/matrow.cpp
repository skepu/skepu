// MatRow arguments on distributed containers: the process has the rows of the matrix that its elements use, not the whole
// matrix. With a matrix that has as many rows as there are elements nothing moves; with more rows the data that moves is
// the least of the elements' and the rows' (see rows.hpp). Same output on a single node and on any number of processes, with any backend.

#include <iostream>
#include <skepu>

int add(int a, int b) { return a + b; }
int row_sum(skepu::Index1D i, const skepu::MatRow<int> r)
{
	int sum = 0;
	for (size_t j = 0; j < r.cols; ++j)
		sum += r[j] * (int)(j + 1);
	return sum + (int)(i.i);
}
int row_with_elwise(int a, const skepu::MatRow<int> r, int k) { return a * k + r[(size_t)(a < 0 ? -a : a) % r.cols]; }
int row_with_two(int a, int b, const skepu::MatRow<int> r, const skepu::MatRow<int> s) { return a - b + r[0] * 3 + s[s.cols - 1]; }
int row_with_vec(skepu::Index1D i, const skepu::MatRow<int> r, const skepu::Vec<int> w) { return r[i.i % r.cols] + w[i.i % w.size]; }
int row_random(skepu::Index1D i, skepu::Random<1> &rnd, const skepu::MatRow<int> r) { return r[i.i % r.cols] + (int)(rnd.get() % 100); }
int row_cell(skepu::Index2D i, int a, const skepu::MatRow<int> r) { return a + r[i.col % r.cols] * 2 + (int)(i.row); }
int pair_row(skepu::Index2D i, int a, int b, const skepu::MatRow<int> r) { return a * b + r[i.col % r.cols] + (int)(i.row) * 5; }
int pair_row_plain(int a, int b, const skepu::MatRow<int> r) { return a + b * r[r.cols - 1]; }
int reduce_row(int a, const skepu::MatRow<int> r) { return a * r[0] + r[r.cols - 1]; }
int reduce_index(skepu::Index1D i, const skepu::MatRow<int> r) { return r[i.i % r.cols] * (int)(i.i + 1); }
int reduce_index2(skepu::Index2D i, const skepu::MatRow<int> r) { return r[i.col % r.cols] + (int)(i.row) * 3; }

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

static void fill(skepu::Matrix<int> &m, size_t rows, size_t cols, int salt)
{
	skepu::external([&] { for (size_t i = 0; i < rows; ++i) for (size_t j = 0; j < cols; ++j) m(i, j) = (int)((i * 7 + j * 3 + salt) % 17) - 5; }, skepu::write(m));
}

static void fill(skepu::Vector<int> &v, size_t n, int salt)
{
	skepu::external([&] { for (size_t i = 0; i < n; ++i) v(i) = (int)((i * 5 + salt) % 13) - 4; }, skepu::write(v));
}

// 'elements' elements and a matrix of 'rows' rows (at least as many as there are elements)
static void run(size_t elements, size_t rows, size_t cols)
{
	std::cout << elements << " elements, matrix " << rows << "x" << cols << "\n";
	skepu::Matrix<int> m(rows, cols), m2(rows, cols);
	fill(m, rows, cols, 1);
	fill(m2, rows, cols, 4);
	skepu::Vector<int> a(elements), b(elements), out(elements), w(7);
	fill(a, elements, 2);
	fill(b, elements, 9);
	fill(w, 7, 3);

	skepu::Map<0>(row_sum)(out, m);
	std::cout << " map row: " << checksum(out) << " " << out(0) << " " << out(elements - 1) << "\n";

	skepu::Map<1>(row_with_elwise)(out, a, m, 3);
	std::cout << " map elwise: " << checksum(out) << "\n";

	skepu::Map<2>(row_with_two)(out, a, b, m, m2);
	std::cout << " map two matrices: " << checksum(out) << "\n";

	skepu::Map<0>(row_with_vec)(out, m, w);
	std::cout << " map vec: " << checksum(out) << "\n";

	skepu::Map<1>(row_with_elwise)(a, a, m, 2); // in place
	std::cout << " map in place: " << checksum(a) << "\n";

	skepu::PRNG prng(77);
	auto random_map = skepu::Map<0>(row_random);
	random_map.setPRNG(prng);
	random_map(out, m);
	std::cout << " map random: " << checksum(out) << "\n";
	random_map(out, m);
	std::cout << " map random again: " << checksum(out) << "\n";

	// A matrix as the result, with a row of the matrix for each of its rows
	skepu::Matrix<int> cells(elements, cols), cin(elements, cols);
	fill(cin, elements, cols, 5);
	skepu::Map<1>(row_cell)(cells, cin, m);
	std::cout << " map cells: " << checksum(cells) << "\n";

	auto mr = skepu::MapReduce<1>(reduce_row, add);
	mr.setStartValue(100);
	std::cout << " mapreduce: " << mr(b, m) << "\n";

	auto mr0 = skepu::MapReduce<0>(reduce_index, add);
	mr0.setDefaultSize(elements);
	mr0.setStartValue(7);
	std::cout << " mapreduce index: " << mr0(m) << "\n";

	auto mr2 = skepu::MapReduce<0>(reduce_index2, add);
	mr2.setDefaultSize(elements, 4);
	std::cout << " mapreduce 2D index: " << mr2(m) << "\n";

	// MapPairs: the rows of the pairs are the elements
	skepu::Vector<int> h(11);
	fill(h, 11, 6);
	skepu::Matrix<int> pairs(elements, 11);
	skepu::MapPairs<1, 1>(pair_row)(pairs, a, h, m);
	std::cout << " mappairs: " << checksum(pairs) << "\n";
	skepu::MapPairs<1, 1>(pair_row_plain)(pairs, b, h, m);
	std::cout << " mappairs plain: " << checksum(pairs) << "\n";
	auto in_place = skepu::MapPairs<1, 1>(pair_row_plain);
	in_place.setInPlace(true);
	in_place(pairs, b, h, m);
	std::cout << " mappairs in place: " << checksum(pairs) << "\n";

	std::cout << " stream: " << prng.get() << "\n";
}

int main()
{
	run(12, 12, 40);  // as many rows as elements: nothing moves
	run(5, 12, 40);   // fewer elements than rows: the elements move
	run(9, 31, 3);    // (a narrow matrix)
	run(2, 40, 17);   // fewer elements than processes
	run(1, 1, 6);
	run(30, 30, 1);
	return 0;
}
