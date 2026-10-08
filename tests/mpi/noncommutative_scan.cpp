// Scans with a function that is associative but not commutative. SkePU's OpenMP and GPU scans assume a commutative
// function (also on a single node): this is for the sequential backend only.

#include <iostream>
#include <skepu>

int last(int, int b) { return b; }
int first(int a, int) { return a; }

int main()
{
	const size_t n = 1003;
	skepu::Vector<int> a(n);
	skepu::external([&]
	{
		for (size_t i = 0; i < n; ++i)
			a(i) = int(i % 17) - 5;
	}, skepu::write(a));

	// scans: the order of the operands decides
	skepu::Vector<int> out(n);
	auto checksum = [&](skepu::Vector<int> &v)
	{
		v.flush();
		unsigned long long h = 17;
		for (size_t i = 0; i < v.size(); ++i)
			h = h * 31 + (unsigned long long)(long long)v(i);
		return h;
	};
	skepu::Scan(last)(out, a);
	const unsigned long long scan_last = checksum(out);
	skepu::Scan(first)(out, a);
	const unsigned long long scan_first = checksum(out);
	skepu::Scan(last).setScanMode(skepu::ScanMode::Exclusive).setStartValue(99)(out, a);
	const unsigned long long scan_last_exclusive = checksum(out);
	skepu::Scan(first).setScanMode(skepu::ScanMode::Exclusive).setStartValue(99)(out, a);
	const unsigned long long scan_first_exclusive = checksum(out);
	std::cout << "scan: " << scan_last << " " << scan_first << " " << scan_last_exclusive << " " << scan_first_exclusive << "\n";
	return 0;
}
