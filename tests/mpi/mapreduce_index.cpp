// MapReduce without containers that give the elements: over the indices of a default size. Same output on a single node
// and on any number of processes, with any backend.

#include <iostream>
#include <skepu>

int add(int a, int b) { return a + b; }
int maximum(int a, int b) { return a > b ? a : b; }
int f1(skepu::Index1D i, int k) { return (int)((i.i * i.i + k * i.i) % 97) - 40; }
int f2(skepu::Index2D i, int k) { return (int)((i.row * 31 + i.col * 17 + k) % 53) - 20; }
int f3(skepu::Index3D i, int k) { return (int)((i.i * 11 + i.j * 7 + i.k * 3 + k) % 41) - 15; }
int f1_vec(skepu::Index1D i, const skepu::Vec<int> w) { return w[i.i % w.size] * (int)(i.i % 5); }
int last(int, int b) { return b; }

static void sums(const size_t n)
{
	skepu::Vector<int> w(5);
	skepu::external([&] { for (size_t k = 0; k < 5; ++k) w(k) = (int)(k * k) - 3; }, skepu::write(w));

	std::cout << "size " << n << "\n";
	const int plain = skepu::MapReduce<0>(f1, add).setDefaultSize(n)(7);
	const int biggest = skepu::MapReduce<0>(f1, maximum).setDefaultSize(n)(3);
	const int started = skepu::MapReduce<0>(f1, add).setDefaultSize(n).setStartValue(1000)(7);
	const int random_access = skepu::MapReduce<0>(f1_vec, add).setDefaultSize(n)(w);
	std::cout << " 1D: " << plain << " " << biggest << " " << started << " " << random_access << "\n";
}

int main()
{
	sums(1000);
	sums(3); // fewer indices than processes
	sums(1);

	const int rows = 13, cols = 7;
	const int two = skepu::MapReduce<0>(f2, add).setDefaultSize(rows, cols)(5);
	const int two_max = skepu::MapReduce<0>(f2, maximum).setDefaultSize(rows, cols)(1);
	const int two_started = skepu::MapReduce<0>(f2, add).setDefaultSize(rows, cols).setStartValue(500)(5);
	std::cout << "2D: " << two << " " << two_max << " " << two_started << "\n";

	const int three = skepu::MapReduce<0>(f3, add).setDefaultSize(6, 4, 5)(2);
	std::cout << "3D: " << three << "\n";
	return 0;
}
