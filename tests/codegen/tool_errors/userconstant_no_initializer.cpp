// skepu-tool must refuse this: a user constant without an initializer (it crashed the tool: there is no value to put into the kernels).
#include <skepu>

[[skepu::userconstant]] extern const int no_initializer;

int add_constant(int x)
{
	return x + no_initializer;
}

int main()
{
	skepu::Vector<int> in(4, 1), out(4);
	skepu::Map<1>(add_constant)(out, in);
	return 0;
}
