#include <skepu>
#include <skepu-lib/complex.hpp>

using Complex = skepu::complex::complex<float>;
using InnerType = Complex::value_type; // float

auto complex_dotprod = skepu::MapReduce(
	skepu::complex::mul<Complex>,
	skepu::complex::add<Complex>
);

int main(int argc, char* argv[])
{
	const size_t size{1000};
	skepu::Vector<Complex> v1(size, {1, 1}), v2(size, {1, 0});
	
	auto res = complex_dotprod(v1, v2);
	std::cout << "Result: " << res << "\n";
}
