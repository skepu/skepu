// Random numbers in skeletons on distributed containers (MapOverlap is in random_overlap.cpp). The elements use the stream of the program in their order, as on
// a single node, so the output is the same (with the sequential backend: the other backends of the node-level library
// split the stream among their threads), and the stream of the program is advanced as on a single node.

#include <iostream>
#include <skepu>

int add(int a, int b) { return a + b; }
int random_element(skepu::Index1D i, skepu::Random<2> &r) { int a = (int)(r.get() % 1000); int b = (int)(r.get() % 500); return a - b + (int)(i.i % 3); }
int random_cell(skepu::Index2D i, skepu::Random<1> &r) { return (int)(r.get() % 100) + (int)(i.row) * 7 + (int)(i.col); }
int random_pair(skepu::Random<1> &r, int a, int b) { return a * b + (int)(r.get() % 100); }
int sample(skepu::Index1D, skepu::Random<2> &r)
{
	int x = (int)(r.get() % 1000);
	int y = (int)(r.get() % 1000);
	return (x * x + y * y <= 1000000) ? 1 : 0;
}
int random_term(skepu::Random<1> &r, int a) { return a + (int)(r.get() % 10); }

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)c.getAddress()[i];
	return h;
}

int main()
{
	skepu::PRNG prng(1234);
	const size_t n = 103;

	skepu::Vector<int> v(n), w(n);
	skepu::Map<0>(random_element).setPRNG(prng)(v);
	std::cout << "map: " << checksum(v) << "\n";
	skepu::Map<0>(random_element).setPRNG(prng)(w);
	std::cout << "map again (the stream goes on): " << checksum(w) << "\n";

	skepu::Matrix<int> m(11, 6);
	skepu::Map<0>(random_cell).setPRNG(prng)(m);
	std::cout << "matrix: " << checksum(m) << "\n";

	skepu::Vector<int> rows(7), cols(5);
	skepu::external([&]
	{
		for (size_t i = 0; i < 7; ++i) rows(i) = (int)i + 1;
		for (size_t j = 0; j < 5; ++j) cols(j) = (int)j - 2;
	}, skepu::write(rows, cols));
	skepu::Matrix<int> pairs(7, 5);
	skepu::MapPairs<1, 1>(random_pair).setPRNG(prng)(pairs, rows, cols);
	std::cout << "pairs: " << checksum(pairs) << "\n";

	const int inside = skepu::MapReduce<0>(sample, add).setDefaultSize(2000).setPRNG(prng)();
	const int sum = skepu::MapReduce<1>(random_term, add).setPRNG(prng)(v);
	std::cout << "mapreduce: " << inside << " " << sum << "\n";

	// the stream of the program is where a single node leaves it
	std::cout << "next: " << prng.get() << " " << prng.get() << "\n";
	return 0;
}
