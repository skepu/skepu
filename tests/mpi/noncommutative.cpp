// Reductions with a function that is associative but not commutative. SkePU's GPU backends assume a commutative
// function (also on a single node), so this is for the CPU backends only (sequential, OpenMP).

#include <iostream>
#include <skepu>

int last(int, int b) { return b; }
int first(int a, int) { return a; }
int identity(int a) { return a; }
int shifted(skepu::Index1D i, int a) { return a + (int)(i.i % 4); }

int main()
{
	const size_t n = 1003;
	skepu::Vector<int> a(n);
	skepu::external([&]
	{
		for (size_t i = 0; i < n; ++i)
			a(i) = int(i % 17) - 5;
	}, skepu::write(a));

	const int final_element = skepu::Reduce(last)(a);
	const int first_element = skepu::Reduce(first)(a);
	const int started = skepu::Reduce(last).setStartValue(99)(a);
	std::cout << "reduce: " << final_element << " " << first_element << " " << started << "\n";

	const int map_last = skepu::MapReduce<1>(identity, last)(a);
	const int map_first = skepu::MapReduce<1>(identity, first)(a);
	const int map_first_started = skepu::MapReduce<1>(shifted, first).setStartValue(99)(a);
	const int map_last_started = skepu::MapReduce<1>(shifted, last).setStartValue(99)(a);
	std::cout << "mapreduce: " << map_last << " " << map_first << " " << map_first_started << " " << map_last_started << "\n";

	return 0;
}
