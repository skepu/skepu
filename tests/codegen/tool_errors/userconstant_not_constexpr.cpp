// skepu-tool must refuse this: a user constant that is not constexpr. The tool puts the constant into the kernels from its initializer, so
// a variable that the program can change would have a value on the CPU and another on a GPU.
#include <skepu>

[[skepu::userconstant]] int not_constexpr = 3;

int add_constant(int x)
{
	return x + not_constexpr;
}

int main()
{
	skepu::Vector<int> in(4, 1), out(4);
	skepu::Map<1>(add_constant)(out, in);
	return 0;
}
