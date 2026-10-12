// The random fills of skepu::randomize on distributed containers: the same elements as on a single node, whatever the number of processes, and the stream of
// the program is advanced as on a single node.

#include <iostream>
#include <skepu>
#include <skepu-lib/randomize.hpp>

template<typename C>
static unsigned long long checksum(C &c)
{
	c.flush();
	unsigned long long h = 17;
	for (size_t i = 0; i < c.size(); ++i)
		h = h * 31 + (unsigned long long)(long long)(c.getAddress()[i] * 1000);
	return h;
}

int main()
{
	skepu::PRNG stream(99);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<int> vi(103);
	skepu::randomize::uniform(vi, -20, 20);
	std::cout << "integers: " << checksum(vi) << "\n";

	skepu::Matrix<float> mf(11, 7);
	skepu::randomize::uniform(mf);
	std::cout << "floats: " << checksum(mf) << "\n";

	skepu::Matrix<float> mn(9, 13);
	skepu::randomize::normal(mn, 1.f, 3.f);
	std::cout << "normal: " << checksum(mn) << "\n";

	skepu::Tensor3<int> t3(3, 5, 4);
	skepu::randomize::bernoulli(t3, 0.4);
	std::cout << "bernoulli: " << checksum(t3) << "\n";

	skepu::Tensor4<int> t4(2, 3, 4, 5);
	skepu::randomize::poisson(t4, 6.0);
	std::cout << "poisson: " << checksum(t4) << "\n";

	// the stream goes on as on a single node
	skepu::Vector<int> after(10);
	skepu::randomize::uniform(after, 1000);
	std::cout << "after: " << checksum(after) << "\n";
	return 0;
}
