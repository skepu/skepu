// MapPairsReduce on distributed containers. Same output on a single node and on any number of processes, with any
// backend (the OpenCL backend does not apply the start value, also on a single node).

#include <iostream>
#include <skepu>

int pr_map(int a, int b) { return a * b; }
int pr_indexed(skepu::Index2D i, int a, int b) { return a * b + (int)(i.row) * 10 + (int)(i.col); }
int pr_vec(skepu::Index2D i, int a, int b, const skepu::Vec<int> w) { return a * b + w[i.col % w.size] - w[i.row % w.size]; }
int pr_index_only(skepu::Index2D i, int offset) { return (int)(i.row) * 7 + (int)(i.col) + offset; }
int pr_add(int a, int b) { return a + b; }
int pr_max(int a, int b) { return a > b ? a : b; }

static unsigned long long checksum(skepu::Vector<int> &v)
{
	v.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < v.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)v(i);
	return h;
}

static void reductions(const size_t rows, const size_t cols)
{
	skepu::Vector<int> v(rows), h(cols), w(5), by_rows(rows), by_columns(cols);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i) v(i) = (int)((i * 3) % 7) + 1;
		for (size_t j = 0; j < cols; ++j) h(j) = (int)((j * 5) % 11) - 3;
		for (size_t k = 0; k < 5; ++k) w(k) = (int)(k * k);
	}, skepu::write(v, h, w));

	std::cout << rows << "x" << cols << "\n";

	skepu::MapPairsReduce<1, 1>(pr_map, pr_add).setReduceMode(skepu::ReduceMode::RowWise)(by_rows, v, h);
	skepu::MapPairsReduce<1, 1>(pr_map, pr_add).setReduceMode(skepu::ReduceMode::ColWise)(by_columns, v, h);
	std::cout << " add: " << checksum(by_rows) << " " << checksum(by_columns) << "\n";

	skepu::MapPairsReduce<1, 1>(pr_map, pr_max).setReduceMode(skepu::ReduceMode::RowWise)(by_rows, v, h);
	skepu::MapPairsReduce<1, 1>(pr_map, pr_max).setReduceMode(skepu::ReduceMode::ColWise)(by_columns, v, h);
	std::cout << " max: " << checksum(by_rows) << " " << checksum(by_columns) << "\n";

	skepu::MapPairsReduce<1, 1>(pr_indexed, pr_add).setReduceMode(skepu::ReduceMode::RowWise)(by_rows, v, h);
	skepu::MapPairsReduce<1, 1>(pr_indexed, pr_add).setReduceMode(skepu::ReduceMode::ColWise)(by_columns, v, h);
	std::cout << " indexed: " << checksum(by_rows) << " " << checksum(by_columns) << "\n";

	skepu::MapPairsReduce<1, 1>(pr_vec, pr_add).setReduceMode(skepu::ReduceMode::RowWise)(by_rows, v, h, w);
	skepu::MapPairsReduce<1, 1>(pr_vec, pr_add).setReduceMode(skepu::ReduceMode::ColWise)(by_columns, v, h, w);
	std::cout << " random access: " << checksum(by_rows) << " " << checksum(by_columns) << "\n";

	skepu::MapPairsReduce<1, 1>(pr_map, pr_add).setStartValue(1000).setReduceMode(skepu::ReduceMode::RowWise)(by_rows, v, h);
	skepu::MapPairsReduce<1, 1>(pr_map, pr_add).setStartValue(1000).setReduceMode(skepu::ReduceMode::ColWise)(by_columns, v, h);
	std::cout << " start value: " << checksum(by_rows) << " " << checksum(by_columns) << "\n";

	// no containers for the pairs
	skepu::MapPairsReduce<0, 0>(pr_index_only, pr_add).setDefaultSize(rows, cols).setReduceMode(skepu::ReduceMode::RowWise)(by_rows, 3);
	skepu::MapPairsReduce<0, 0>(pr_index_only, pr_add).setDefaultSize(rows, cols).setReduceMode(skepu::ReduceMode::ColWise)(by_columns, 3);
	std::cout << " index only: " << checksum(by_rows) << " " << checksum(by_columns) << "\n";
}

int main()
{
	reductions(13, 7);
	reductions(3, 5); // fewer rows than processes
	return 0;
}
