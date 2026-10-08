// A program written for one node. It must give the same output built for a single node and built for the MPI
// backend (skepu-tool -mpi), on any number of processes, also with a GPU backend: the host writes are made in
// skepu::external and the host reads follow a flush(), as the GPU backends need.

#include <iostream>
#include <skepu>

int add(int a, int b) { return a + b; }
int maximum(int a, int b) { return a > b ? a : b; }
int index_plus(skepu::Index1D i, int a) { return (int)(i.i) * 3 + a; }
int scaled(int a, int factor) { return a * factor; }
int reversed(skepu::Index1D i, const skepu::Vec<int> table) { return table[table.size - 1 - i.i]; }

auto sum = skepu::Reduce(add);
auto adder = skepu::Map<2>(add);
auto indexer = skepu::Map<1>(index_plus);

static unsigned long long checksum(skepu::Vector<int> &v)
{
	v.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < v.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)v(i);
	return h;
}

int main()
{
	// not a multiple of the usual numbers of processes
#ifndef PROTOTYPE_SIZE
# define PROTOTYPE_SIZE 1003
#endif
	const size_t n = PROTOTYPE_SIZE;

	skepu::Vector<int> a(n), b(n), out(n), rev(n), cumulative(n, 7);
	skepu::external([&]
	{
		for (size_t i = 0; i < n; ++i)
		{
			a(i) = int(i % 17) - 5;
			b(i) = int((i * 7) % 13);
		}
	}, skepu::write(a, b));

	adder(out, a, b);
	std::cout << "add: " << checksum(out) << " " << out(0) << " " << out(n - 1) << "\n";

	indexer(out, a);
	std::cout << "index: " << checksum(out) << " " << out(0) << " " << out(500) << " " << out(n - 1) << "\n";

	// uniform argument, and an anonymous skeleton
	skepu::Map<1>(scaled)(out, b, -3);
	std::cout << "scaled: " << checksum(out) << "\n";

	// random access to a container
	skepu::Map<0>(reversed)(rev, a);
	std::cout << "reversed: " << checksum(rev) << " " << rev(0) << " " << rev(n - 1) << "\n";

	// reductions
	const int total = sum(a);
	const int biggest = skepu::Reduce(maximum)(b);
	const int with_start = skepu::Reduce(add).setStartValue(1000)(a);
	std::cout << "reduce: " << total << " " << biggest << " " << with_start << "\n";

	// the host changes elements between skeletons
	skepu::external(skepu::read(a), [&]
	{
		a(10) = 100000;
		a(n - 1) = -5;
	}, skepu::write(a));
	const int changed = sum(a);
	adder(out, a, b);
	std::cout << "changed: " << changed << " " << checksum(out) << "\n";

	// a copy is a vector of its own
	out.flush();
	skepu::Vector<int> copy(out);
	skepu::external([&] { out(0) = 12345; }, skepu::write(out));
	copy.flush();
	out.flush();
	std::cout << "copy: " << copy(0) << " " << out(0) << " " << checksum(copy) << "\n";

	// code that is not parallel: reads a vector, writes another
	skepu::external(skepu::read(a), [&]
	{
		long external_total = 0;
		for (size_t i = 0; i < n; ++i)
			external_total += a(i);
		std::cout << "external: " << external_total << "\n";
	});

	skepu::Vector<int> filled(n);
	skepu::external([&]
	{
		for (size_t i = 0; i < n; ++i)
			filled(i) = int((i * i) % 101);
	}, skepu::write(filled));
	adder(cumulative, filled, filled);
	std::cout << "written: " << checksum(filled) << " " << checksum(cumulative) << " " << sum(cumulative) << "\n";

	// small vectors, fewer elements than processes
	skepu::Vector<int> tiny{3, 1, 4}, tiny_out(3);
	skepu::Map<1>(scaled)(tiny_out, tiny, 2);
	std::cout << "tiny: " << tiny_out << " " << int(skepu::Reduce(add)(tiny)) << "\n";
	return 0;
}
