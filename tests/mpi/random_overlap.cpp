// Random numbers in a MapOverlap on distributed containers, with an edge mode (the skeleton of the node-level library runs
// in a process with the edge mode None, and its OpenMP path uses the random numbers in another order than its
// other paths: same output as a single node with the sequential, OpenCL and hybrid backends only).

#include <iostream>
#include <skepu>

int random_overlap(skepu::Random<1> &r, skepu::Region1D<int> reg) { return reg(-1) + reg(1) + (int)(r.get() % 10); }
int random_element(skepu::Index1D i, skepu::Random<2> &r) { int a = (int)(r.get() % 1000); int b = (int)(r.get() % 500); return a - b + (int)(i.i % 3); }

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
	skepu::Vector<int> v(n);
	skepu::Map<0>(random_element).setPRNG(prng)(v);
	skepu::Vector<int> smoothed(n);
	skepu::MapOverlap(random_overlap).setOverlap(1).setEdgeMode(skepu::Edge::Duplicate).setPRNG(prng)(smoothed, v);
	std::cout << "overlap: " << checksum(smoothed) << "\n";

	skepu::Vector<int> shrunk(n - 2);
	skepu::MapOverlap(random_overlap).setOverlap(1).setEdgeMode(skepu::Edge::None).setPRNG(prng)(shrunk, v);
	std::cout << "overlap, none: " << checksum(shrunk) << "\n";
	std::cout << "next: " << prng.get() << " " << prng.get() << "\n";
	return 0;
}
