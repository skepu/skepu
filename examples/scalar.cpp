#include <skepu>
#include <skepu-lib/io.hpp>

template<typename T>
T mult(T a, T b)
{
	return a * b;
}

template<typename T>
T max(T a, T b)
{
	return (a > b) ? a : b;
}


void test()
{
	auto red = skepu::Reduce(max<int>);

	skepu::Vector<int> a(10);
	a.randomize(0, 3);

	skepu::io::cout << a << "\n";

	auto res = red(a);

	res = +res;
	res = -res;

	{ auto test = res + 1; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = res - 1; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = res * 1; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = res / 1; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = res % 1; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = res & 1; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = res | 1; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = res ^ 1; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = res < 1; static_assert(std::is_same<skepu::Scalar<bool>, decltype(test)>::value, "opeator"); }
	{ auto test = res > 1; static_assert(std::is_same<skepu::Scalar<bool>, decltype(test)>::value, "opeator"); }
	{ auto test = res == 1; static_assert(std::is_same<skepu::Scalar<bool>, decltype(test)>::value, "opeator"); }
	{ auto test = res <= 1; static_assert(std::is_same<skepu::Scalar<bool>, decltype(test)>::value, "opeator"); }
	{ auto test = res >= 1; static_assert(std::is_same<skepu::Scalar<bool>, decltype(test)>::value, "opeator"); }
	{ auto test = res << 1; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = res >> 1; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }

	{ auto test = 1 + res; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = 1 - res; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = 1 * res; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = 1 / res; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = 1 % res; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = 1 & res; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = 1 | res; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = 1 ^ res; static_assert(std::is_same<decltype(res), decltype(test)>::value, "opeator"); }
	{ auto test = 1 < res; static_assert(std::is_same<skepu::Scalar<bool>, decltype(test)>::value, "opeator"); }
	{ auto test = 1 > res; static_assert(std::is_same<skepu::Scalar<bool>, decltype(test)>::value, "opeator"); }
	{ auto test = 1 == res; static_assert(std::is_same<skepu::Scalar<bool>, decltype(test)>::value, "opeator"); }
	{ auto test = 1 <= res; static_assert(std::is_same<skepu::Scalar<bool>, decltype(test)>::value, "opeator"); }
	{ auto test = 1 >= res; static_assert(std::is_same<skepu::Scalar<bool>, decltype(test)>::value, "opeator"); }

	res += 1;

	std::cout << res << "\n";
	skepu::io::cout << res << "\n";
}


int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}

	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);

	auto sum = skepu::Reduce([](float a, float b) { return a + b; });
	auto normalize = skepu::Map<1>([](float a, float b) { return a * b; });

	skepu::Vector<float> vec(size, "vec");
	vec.randomize(0, 100);

	skepu::io::cout << vec << "\n";

	auto avg = sum(vec) / size;
	skepu::io::cout << "Avg: " << avg << "\n";
	normalize(vec, vec, 1 / avg);

	skepu::io::cout << vec << "\n";

	return 0;
}
