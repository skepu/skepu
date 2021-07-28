/*
 * Taylor series calculation, natural log(1+x)  sum(1:N) (((-1)^(i+1))/i)*x^i
 */

#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>

float nth_term(skepu::Index1D index, float x)
{
	int k = index.i + 1;
	float temp_x = pow(x, k);
	int sign = (k % 2 == 0) ? -1 : 1;
	return sign * temp_x / k;
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		skepu::io::cout << "Usage: " << argv[0] << " x-value number-of-terms backend\n";
		exit(1);
	}
	
	float x = atof(argv[1]);
	size_t N = std::stoul(argv[2]);
	auto spec = skepu::BackendSpec{argv[3]};
	skepu::setGlobalBackendSpec(spec);
	
	auto taylor = skepu::MapReduce<0>(nth_term, skepu::util::add<float>);
	taylor.setDefaultSize(N);
	
	skepu::io::cout << "Result: ln(" << x << ") = " << taylor(x - 1) << "\n";
	
	return 0;
}