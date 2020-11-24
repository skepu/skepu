/*
 * Taylor series calculation, natural log(1+x)  sum(1:N) (((-1)^(i+1))/i)*x^i
 */

#include <iostream>
#include <cmath>

#include <skepu>


float nth_term(skepu::Index1D index, float x)
{
	int k = index.i + 1;
	float temp_x = pow(x, k);
	int sign = (k % 2 == 0) ? -1 : 1;
	return sign * temp_x / k;
}

float plus(float a, float b)
{
	return a + b;
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		std::cout << "Usage: " << argv[0] << " x-value number-of-terms backend\n";
		exit(1);
	}
	
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[3])};
	skepu::setGlobalBackendSpec(spec);
	float x = atof(argv[1]);
	size_t N = std::stoul(argv[2]);
	
	auto taylor = skepu::MapReduce<0>(nth_term, plus);
	taylor.setDefaultSize(N);
	
	std::cout << "Result: ln(" << x << ") = " << taylor(x - 1) << "\n";
	
	return 0;
}