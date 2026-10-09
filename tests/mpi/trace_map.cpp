// A program for the trace of the MPI backend (check_trace.py): a few Map calls, a host access that gathers a container,
// a skepu::external that gives one back, and a Map with a MatRow argument for fewer elements than the matrix has rows, which
// makes the elements go to the processes that have the rows. Built with SKEPU_TRACING and run on several processes.

#include <iostream>
#include <skepu>

int increment(int a) { return a + 1; }
int scaled(int a, int factor) { return a * factor; }
int indexed(skepu::Index1D i, int a, int b) { return a + b * (int)(i.i % 7); }
int row_sum(skepu::Index1D i, const skepu::MatRow<int> r)
{
	int sum = 0;
	for (size_t j = 0; j < r.cols; ++j)
		sum += r[j];
	return sum + (int)(i.i);
}

int main()
{
	const size_t n = 1000;
	skepu::Vector<int> a(n, "a"), b(n, "b"), c(n);
	skepu::external([&] { for (size_t i = 0; i < n; ++i) { a(i) = (int)(i % 13); b(i) = (int)(i % 5); } }, skepu::write(a, b));

	auto inc = skepu::Map<1>(increment);
	auto scale = skepu::Map<1>(scaled);
	auto idx = skepu::Map<2>(indexed);
	inc.setLabel("inc");
	scale.setLabel("scale");
	idx.setLabel("idx");

	for (int k = 0; k < 3; ++k)
	{
		inc(c, a);
		scale(a, c, 2);
		idx(c, a, b);
	}

	std::cout << "element: " << c[3] << " " << c[n - 1] << "\n"; // (the host reads: c is gathered)
	skepu::external([&] { c(0) = 42; }, skepu::write(c));      // (and written: c goes back to the owners)

	skepu::Matrix<int> m(12, 40, "m");
	skepu::external([&] { for (size_t i = 0; i < 12; ++i) for (size_t j = 0; j < 40; ++j) m(i, j) = (int)((i + j) % 9); }, skepu::write(m));
	skepu::Vector<int> few(5, "few");
	auto rows = skepu::Map<0>(row_sum);
	rows.setLabel("rows");
	rows(few, m);
	std::cout << "rows: " << few[4] << "\n";
	return 0;
}
