#include <skepu>

double square(double val)
{
	return val * val;
}

double add(double lhs, float rhs)
{
	return rhs + lhs;
}

auto fold = skepu::Reduce(add);

int main(int argc, char *argv[])
{
	const size_t N = 10;
	
	skepu::Vector<float> vec_float(N, 5.0f);
	skepu::Vector<double> vec_double(N, 5.0);
	skepu::Vector<double> vec_res(N);
	
	float scalar_float = 5.0f;
	double scalar_double = 5.0;
	
	auto skel = skepu::Map(square);
	
	
	// Works: same type
	auto res1 = square(scalar_double);
	
	// Works: implicit conversion
	auto res2 = square(scalar_float);
	
	// Works: same type
	skel(vec_res, vec_double);
	
	// Compile time error: type mismatch
	skel(vec_res, vec_float);
  
	return 0;
}