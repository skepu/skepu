// MapPairs on distributed containers. Same output on a single node and on any number of processes, with any backend.

#include <iostream>
#include <skepu>

int pair_product(int a, int b) { return a * b; }
int pair_indexed(skepu::Index2D i, int a, int b) { return a * b + (int)(i.row) * 100 + (int)(i.col); }
int pair_uniform(int a, int b, int factor) { return (a - b) * factor; }
int pair_vec(skepu::Index2D i, int a, int b, const skepu::Vec<int> w) { return a * b + w[i.col % w.size] - w[i.row % w.size]; }
int index_only(skepu::Index2D i) { return (int)(i.row) * 10 + (int)(i.col); }
int accumulate(int a, int b) { return a + b; }

static unsigned long long checksum(skepu::Matrix<int> &m)
{
	m.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < m.total_rows(); ++i)
		for (size_t j = 0; j < m.total_cols(); ++j)
			h = h * 31 + (unsigned long long)(long long)m(i, j);
	return h;
}

static void pairs(const size_t rows, const size_t cols)
{
	skepu::Vector<int> v(rows), h(cols), w(5);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i) v(i) = (int)((i * 3) % 7) + 1;
		for (size_t j = 0; j < cols; ++j) h(j) = (int)((j * 5) % 11) - 3;
		for (size_t k = 0; k < 5; ++k) w(k) = (int)(k * k);
	}, skepu::write(v, h, w));

	skepu::Matrix<int> out(rows, cols);

	skepu::MapPairs<1, 1>(pair_product)(out, v, h);
	std::cout << rows << "x" << cols << " product: " << checksum(out) << " " << out(0, 0) << " " << out(rows - 1, cols - 1) << "\n";

	skepu::MapPairs<1, 1>(pair_indexed)(out, v, h);
	std::cout << " indexed: " << checksum(out) << " " << out(rows - 1, cols - 1) << "\n";

	skepu::MapPairs<1, 1>(pair_uniform)(out, v, h, 4);
	std::cout << " uniform: " << checksum(out) << "\n";

	skepu::MapPairs<1, 1>(pair_vec)(out, v, h, w);
	std::cout << " random access: " << checksum(out) << "\n";

	skepu::MapPairs<0, 0>(index_only)(out);
	std::cout << " index only: " << checksum(out) << " " << out(rows - 1, cols - 1) << "\n";

	// the results are added to what is there
	skepu::MapPairs<1, 1>(pair_product)(out, v, h);
	skepu::Matrix<int> again(rows, cols, 1000);
	skepu::MapPairs<1, 1>(accumulate).setInPlace(true)(again, v, h);
	std::cout << " in place: " << checksum(again) << "\n";
}

int main()
{
	pairs(13, 7);
	pairs(3, 5); // fewer rows than processes
	return 0;
}
