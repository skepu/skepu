// MapReduce on distributed containers. Same output on a single node and on any number of processes, with any backend.

#include <iostream>
#include <skepu>

int mul(int a, int b) { return a * b; }
int add(int a, int b) { return a + b; }
int maximum(int a, int b) { return a > b ? a : b; }
int square(int a) { return a * a; }
int weighted(skepu::Index1D i, int a, int b) { return a * b + (int)(i.i % 3); }
int scaled_square(int a, int factor) { return a * a * factor; }
int reversed_product(skepu::Index1D i, int a, const skepu::Vec<int> w) { return a * w[w.size - 1 - i.i]; }
int index2(skepu::Index2D i, int a, int b) { return a * b + (int)(i.row) * 10 + (int)(i.col); }
int index3(skepu::Index3D i, int a) { return a + (int)(i.i) * 100 + (int)(i.j) * 10 + (int)(i.k); }

auto dot = skepu::MapReduce<2>(mul, add);

int main()
{
	const size_t n = 1003;
	skepu::Vector<int> a(n), b(n);
	skepu::external([&]
	{
		for (size_t i = 0; i < n; ++i)
		{
			a(i) = int(i % 17) - 5;
			b(i) = int((i * 7) % 13) - 2;
		}
	}, skepu::write(a, b));

	const int d = dot(a, b);
	std::cout << "dot: " << d << "\n";

	// start values
	const int started = skepu::MapReduce<2>(mul, add).setStartValue(1000)(a, b);
	const int started_max = skepu::MapReduce<1>(square, maximum).setStartValue(5)(a);
	const int default_start_max = skepu::MapReduce<1>(square, maximum)(b);
	std::cout << "started: " << started << " " << started_max << " " << default_start_max << "\n";

	// indices, uniforms, random access containers
	const int indexed = skepu::MapReduce<2>(weighted, add)(a, b);
	const int uniform = skepu::MapReduce<1>(scaled_square, add)(a, 3);
	const int random_access = skepu::MapReduce<1>(reversed_product, add)(a, b);
	std::cout << "indexed: " << indexed << " " << uniform << " " << random_access << "\n";

	// the host changes an element between calls
	skepu::external(skepu::read(a), [&] { a(n - 1) = 12345; }, skepu::write(a));
	const int changed = dot(a, b);
	std::cout << "changed: " << changed << "\n";

	// matrices and tensors
	const size_t rows = 11, cols = 6;
	skepu::Matrix<int> m1(rows, cols), m2(rows, cols);
	skepu::external([&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
			{
				m1(i, j) = int((i * 5 + j) % 7) - 3;
				m2(i, j) = int((i + j * 3) % 5);
			}
	}, skepu::write(m1, m2));
	const int mdot = skepu::MapReduce<2>(mul, add)(m1, m2);
	const int mindexed = skepu::MapReduce<2>(index2, add)(m1, m2);
	std::cout << "matrix: " << mdot << " " << mindexed << "\n";

	skepu::Tensor3<int> t(5, 3, 4);
	skepu::external([&]
	{
		for (size_t i = 0; i < 5; ++i)
			for (size_t j = 0; j < 3; ++j)
				for (size_t k = 0; k < 4; ++k)
					t(i, j, k) = int((i * 3 + j * 2 + k) % 9) - 2;
	}, skepu::write(t));
	const int tsum = skepu::MapReduce<1>(index3, add)(t);
	std::cout << "tensor: " << tsum << "\n";

	// fewer elements than processes
	skepu::Vector<int> tiny1{3, 1, 4}, tiny2{2, 7, 1};
	const int tiny = skepu::MapReduce<2>(mul, add).setStartValue(10)(tiny1, tiny2);
	std::cout << "tiny: " << tiny << " " << int(skepu::MapReduce<1>(square, maximum)(tiny1)) << "\n";
	return 0;
}
