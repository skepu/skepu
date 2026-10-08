// Reductions of matrices to vectors and to scalars, on distributed containers. Same output on a single node and on any
// number of processes, with any backend.

#include <iostream>
#include <skepu>

int add(int a, int b) { return a + b; }
int maximum(int a, int b) { return a > b ? a : b; }

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
	skepu::Matrix<int> m(rows, cols);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				m(i, j) = (int)((i * 5 + j * 3) % 11) - 4;
	}, skepu::write(m));

	skepu::Vector<int> by_rows(rows), by_columns(cols);
	std::cout << rows << "x" << cols << "\n";

	skepu::Reduce(add).setReduceMode(skepu::ReduceMode::RowWise)(by_rows, m);
	skepu::Reduce(add).setReduceMode(skepu::ReduceMode::ColWise)(by_columns, m);
	std::cout << " add: " << checksum(by_rows) << " " << checksum(by_columns) << "\n";

	skepu::Reduce(maximum).setReduceMode(skepu::ReduceMode::RowWise)(by_rows, m);
	skepu::Reduce(maximum).setReduceMode(skepu::ReduceMode::ColWise)(by_columns, m);
	std::cout << " max: " << checksum(by_rows) << " " << checksum(by_columns) << "\n";

	// the host changes the matrix between reductions
	skepu::external(skepu::read(m), [&] { m(rows - 1, cols - 1) = 500; m(0, 0) = -500; }, skepu::write(m));
	skepu::Reduce(add).setReduceMode(skepu::ReduceMode::ColWise)(by_columns, m);
	skepu::Reduce(add).setReduceMode(skepu::ReduceMode::RowWise)(by_rows, m);
	std::cout << " changed: " << checksum(by_rows) << " " << checksum(by_columns) << "\n";

	// two functions: the rows (or the columns) with the first, their results with the second
	const int row_add_then_max = skepu::Reduce(add, maximum).setReduceMode(skepu::ReduceMode::RowWise)(m);
	const int col_add_then_max = skepu::Reduce(add, maximum).setReduceMode(skepu::ReduceMode::ColWise)(m);
	const int row_max_then_add = skepu::Reduce(maximum, add).setReduceMode(skepu::ReduceMode::RowWise)(m);
	const int col_max_then_add = skepu::Reduce(maximum, add).setReduceMode(skepu::ReduceMode::ColWise)(m);
	std::cout << " two functions: " << row_add_then_max << " " << col_add_then_max << " " << row_max_then_add << " " << col_max_then_add << "\n";

	const int started = skepu::Reduce(add, maximum).setStartValue(10000).setReduceMode(skepu::ReduceMode::RowWise)(m);
	const int started_cols = skepu::Reduce(maximum, add).setStartValue(100).setReduceMode(skepu::ReduceMode::ColWise)(m);
	std::cout << " start value: " << started << " " << started_cols << "\n";
}

int main()
{
	reductions(13, 7);
	reductions(3, 5); // fewer rows than processes
	reductions(1, 9);
	return 0;
}
