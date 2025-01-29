#include <skepu>

float add(float lhs, float rhs) { return rhs + lhs; }
auto fold = skepu::Reduce(add);

int main(int argc, char *argv[])
{
	size_t N = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	float el = 1 / ((float) N);
	skepu::Vector<float> vec(N, el);
	
	float res = fold(vec);
	std::cout << "Result: " << res << "\n";
}
