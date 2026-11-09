// not a kernel builtin, called unqualified
#include <cmath>
#include <skepu>

float call_lgamma(float x)
{
	return lgamma(x);
}

int main()
{
	skepu::Vector<float> in(4, 1.f), out(4);
	skepu::Map<1>(call_lgamma)(out, in);
	return 0;
}
