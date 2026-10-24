// The control of the two programs above: the same program with a constexpr user constant, which skepu-tool must accept.
#include <skepu>

[[skepu::userconstant]] constexpr int valid_constant = 3;

int add_constant(int x)
{
	return x + valid_constant;
}

int main()
{
	skepu::Vector<int> in(4, 1), out(4);
	skepu::Map<1>(add_constant)(out, in);
	return 0;
}
