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

int plus(int a, int b) { return a + b; }
int square(int a) { return a * a; }
int index_square(skepu::Index1D i) { return (int)((i.i % 11) * (i.i % 11)); }

int pair_product(int a, int b) { return a * b; }
int smooth(skepu::Region1D<int> r) { return r(-1) + 2 * r(0) - r(1); }
int stencil2d(skepu::Region2D<int> r) { return r(-1, 0) + r(1, 0) - r(0, -1) + r(0, 1) + r(0, 0); }
int stencil3d(skepu::Region3D<int> r) { return r(-1, 0, 0) + r(1, 0, 0) + r(0, 0, 0); }
int stencil4d(skepu::Region4D<int> r) { return r(-1, 0, 0, 0) + r(0, 0, 0, 1) + r(0, 0, 0, 0); }
int pool_sum(skepu::Pool2D<int> p) { return p(0, 0) + p(1, 1); }
int pool_sum3d(skepu::Pool3D<int> p) { return p(0, 0, 0) + p(1, 1, 1); }
int pool_sum4d(skepu::Pool4D<int> p) { return p(0, 0, 0, 0) + p(1, 1, 1, 1); }

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

	// the other skeletons: reductions (of a vector, of the rows and of the columns of a matrix, of a matrix), a map and a
	// reduction, one of the indices, a scan
	auto sum = skepu::Reduce(plus);
	auto sum_rows = skepu::Reduce(plus);
	sum_rows.setReduceMode(skepu::ReduceMode::RowWise);
	auto sum_cols = skepu::Reduce(plus);
	sum_cols.setReduceMode(skepu::ReduceMode::ColWise);
	auto sum_all = skepu::Reduce(plus, plus);
	auto sum_squares = skepu::MapReduce<1>(square, plus);
	auto sum_index = skepu::MapReduce<0>(index_square, plus);
	sum_index.setDefaultSize(n);
	auto prefix = skepu::Scan(plus);
	sum.setLabel("sum");
	sum_rows.setLabel("sum_rows");
	sum_cols.setLabel("sum_cols");
	sum_all.setLabel("sum_all");
	sum_squares.setLabel("sum_squares");
	sum_index.setLabel("sum_index");
	prefix.setLabel("prefix");

	skepu::Vector<int> row_totals(12), col_totals(40), scanned(n);
	const int total = sum(a);
	sum_rows(row_totals, m);
	sum_cols(col_totals, m);
	const int whole = sum_all(m);
	const int squares = sum_squares(a);
	const int by_index = sum_index();
	prefix(scanned, a);
	std::cout << "reductions: " << total << " " << row_totals[3] << " " << col_totals[5] << " " << whole << " " << squares << " " << by_index
		<< " " << scanned[n - 1] << "\n";

	// MapPairs and MapPairsReduce
	skepu::Vector<int> vertical(12, "vertical"), horizontal(40, "horizontal");
	skepu::external([&] { for (size_t i = 0; i < 12; ++i) vertical(i) = (int)(i % 4); for (size_t j = 0; j < 40; ++j) horizontal(j) = (int)(j % 6); },
		skepu::write(vertical, horizontal));
	skepu::Matrix<int> pairs_out(12, 40);
	auto pairs = skepu::MapPairs<1, 1>(pair_product);
	pairs.setLabel("pairs");
	pairs(pairs_out, vertical, horizontal);
	auto pairs_reduce = skepu::MapPairsReduce<1, 1>(pair_product, plus);
	pairs_reduce.setLabel("pairs_reduce");
	pairs_reduce.setReduceMode(skepu::ReduceMode::RowWise)(row_totals, vertical, horizontal);
	pairs_reduce.setReduceMode(skepu::ReduceMode::ColWise)(col_totals, vertical, horizontal);
	pairs_out.flush(); // (operator() reads the host copy)
	std::cout << "pairs: " << pairs_out(11, 39) << " " << row_totals[3] << " " << col_totals[5] << "\n";

	// the overlap and pool skeletons: a vector, the rows and columns of a matrix, matrices, tensors
	skepu::Vector<int> smoothed(n);
	skepu::MapOverlap(smooth).setOverlap(1).setEdgeMode(skepu::Edge::Duplicate)(smoothed, a);
	skepu::Matrix<int> mo(12, 40), mi(12, 40);
	skepu::external([&] { for (size_t i = 0; i < 12; ++i) for (size_t j = 0; j < 40; ++j) mi(i, j) = (int)((i * 3 + j) % 7); }, skepu::write(mi));
	skepu::MapOverlap(smooth).setOverlap(1).setOverlapMode(skepu::Overlap::RowWise).setEdgeMode(skepu::Edge::Cyclic)(mo, mi);
	skepu::MapOverlap(smooth).setOverlap(1).setOverlapMode(skepu::Overlap::ColWise).setEdgeMode(skepu::Edge::Cyclic)(mo, mi);
	skepu::MapOverlap(stencil2d).setOverlap(1).setEdgeMode(skepu::Edge::Duplicate)(mo, mi);
	skepu::Matrix<int> pooled(6, 20);
	skepu::MapPool(pool_sum).setPoolSizeAndStride(2)(pooled, mi);
	skepu::Tensor3<int> t3(6, 5, 4), t3out(6, 5, 4), t3pooled(3, 2, 2);
	skepu::Tensor4<int> t4(6, 3, 4, 2), t4out(6, 3, 4, 2), t4pooled(3, 1, 2, 1);
	skepu::external([&]
	{
		for (size_t i = 0; i < 6; ++i) for (size_t j = 0; j < 5; ++j) for (size_t k = 0; k < 4; ++k) t3(i, j, k) = (int)((i + j * 2 + k) % 5);
		for (size_t i = 0; i < 6; ++i) for (size_t j = 0; j < 3; ++j) for (size_t k = 0; k < 4; ++k) for (size_t l = 0; l < 2; ++l) t4(i, j, k, l) = (int)((i + j + k + l) % 5);
	}, skepu::write(t3, t4));
	skepu::MapOverlap(stencil3d).setOverlap(1).setEdgeMode(skepu::Edge::Duplicate)(t3out, t3);
	skepu::MapOverlap(stencil4d).setOverlap(1).setEdgeMode(skepu::Edge::Duplicate)(t4out, t4);
	skepu::MapPool(pool_sum3d).setPoolSizeAndStride(2)(t3pooled, t3);
	skepu::MapPool(pool_sum4d).setPoolSizeAndStride(2)(t4pooled, t4);
	mo.flush(); pooled.flush(); t3out.flush(); t4out.flush(); t3pooled.flush(); t4pooled.flush();
	std::cout << "windows: " << smoothed[7] << " " << mo(5, 5) << " " << pooled(2, 2) << " " << t3out(3, 2, 1) << " " << t4out(3, 1, 2, 1)
		<< " " << t3pooled(1, 1, 1) << " " << t4pooled(1, 0, 1, 0) << "\n";
	return 0;
}
