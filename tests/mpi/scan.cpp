// Scan on distributed containers. Same output on a single node and on any number of processes, with any backend.

#include <iostream>
#include <skepu>

int add(int a, int b) { return a + b; }
int maximum(int a, int b) { return a > b ? a : b; }

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

static void vectors(const size_t n)
{
	skepu::Vector<int> in(n), out(n);
	skepu::external([&]
	{
		for (size_t i = 0; i < n; ++i)
			in(i) = (int)((i * 7) % 13) - 4;
	}, skepu::write(in));

	std::cout << "vector " << n << "\n";

	skepu::Scan(add)(out, in);
	std::cout << " inclusive add: " << checksum(out) << " " << out(0) << " " << out(n - 1) << "\n";

	skepu::Scan(add).setScanMode(skepu::ScanMode::Exclusive)(out, in);
	std::cout << " exclusive add: " << checksum(out) << " " << out(0) << " " << out(n - 1) << "\n";

	skepu::Scan(add).setScanMode(skepu::ScanMode::Exclusive).setStartValue(1000)(out, in);
	std::cout << " exclusive, start value: " << checksum(out) << " " << out(0) << " " << out(n - 1) << "\n";

	skepu::Scan(add).setScanMode(skepu::ScanMode::Inclusive).setStartValue(1000)(out, in);
	std::cout << " inclusive, start value (not used): " << checksum(out) << "\n";

	skepu::Scan(maximum)(out, in);
	std::cout << " inclusive max: " << checksum(out) << "\n";
	skepu::Scan(maximum).setScanMode(skepu::ScanMode::Exclusive).setStartValue(-100)(out, in);
	std::cout << " exclusive max: " << checksum(out) << "\n";

	// in place
	skepu::Vector<int> copy(in);
	skepu::Scan(add)(copy, copy);
	std::cout << " in place: " << checksum(copy) << "\n";
	// (an exclusive scan in place is not defined on a single node: it reads what it has written)
}

int main()
{
	vectors(1003);
	vectors(3); // fewer elements than processes
	vectors(1);

	// a matrix: the elements in the order of the memory
	const size_t rows = 7, cols = 5;
	skepu::Matrix<int> m(rows, cols), result(rows, cols);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				m(i, j) = (int)((i * 3 + j) % 7) - 2;
	}, skepu::write(m));
	skepu::Scan(add)(result, m);
	std::cout << "matrix inclusive: " << checksum(result) << "\n";
	skepu::Scan(add).setScanMode(skepu::ScanMode::Exclusive).setStartValue(3)(result, m);
	std::cout << "matrix exclusive: " << checksum(result) << "\n";
	return 0;
}
