#include <iostream>
#include <skepu>

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[2])};
	
	auto square = skepu::Map<2>([](float a, float b) { return a * b; });
	square.setBackend(spec);
	
	auto dotprod = skepu::MapReduce<2>(
		[](float a, float b) { return a * b; },
		[](float a, float b) { return a + b; }
	);
	
	
	auto sum =  skepu::Reduce([](float a, float b) { return a + b; });
	sum.setBackend(spec);
	
	auto sum2d =  skepu::Reduce([](float a, float b) { return a + b; }, [](float a, float b) { return a + b; });
	sum2d.setBackend(spec);
	
	auto cum_sum = skepu::Scan([](float a, float b) { return a + b; });
	cum_sum.setBackend(spec);
	
	
	auto conv = skepu::MapOverlap([](skepu::Region1D<float> a)
	{
		float sum = 0;
		for (int i = -a.oi; i <= a.oi; i++)
			sum += a(i);
		return sum;
	});
	conv.setBackend(spec);
	
	auto conv2d = skepu::MapOverlap([](skepu::Region2D<float> m, const skepu::Mat<float> filter)
	{
		float res = 0;
		for (int y = -m.oi; y <= m.oi; ++y)
			for (int x = -m.oj; x <= m.oj; ++x)
				res += m(y, x) * filter.data[(y+m.oi)*m.oj + (x+m.oj)];
		return res;
	});
	conv2d.setBackend(spec);
	
	
	auto caller = skepu::Call([](skepu::Vec<float> r, float a, float b) { r[0] = a * b; });
	caller.setBackend(spec);
	
	
	skepu::Vector<float> v1(size, 3), v2(size, 7), r(size);
	skepu::Matrix<float> m1(size, size, 3);
	std::cout << "v1: " << v1 << "\nv2: " << v2 << "\n";
	
	square(r, v1, v2);
	std::cout << "Map: r = " << r << "\n";
	
	square(r.begin(), r.end(), v1.begin(), v2.begin());
	std::cout << "Map: r = " << r << "\n";
	
	square(r.begin(), r.end(), v1, v2);
	std::cout << "Map: r = " << r << "\n";
	
	square(r, v1.begin(), v2);
	std::cout << "Map: r = " << r << "\n";
	
	
	std::cout << "MapReduce: dotprod(v1, v2) = " << dotprod(v1, v2) << "\n";
	
	std::cout << "Reduce 1D: sum(v1) = " << sum(v1) << "\n";
	
	std::cout << "Reduce 2D: sum2d(m1) = " << sum2d(m1) << "\n";
	
	cum_sum(r, v1);
	std::cout << "Scan: cum_sum(v1) = " << r << "\n";
	
	conv.setOverlap(2);
	conv.setEdgeMode(skepu::Edge::Pad);
	conv.setPad(0);
	conv(r, v1);
	
	std::cout << "MapOverlap1D: conv(r, v1) = " << r << "\n";
	
	int o = 1;
	skepu::Matrix<float> filter(2*o+1, 2*o+1, 1), m(size, size, 5), rm(size - 2*o, size - 2*o);
	
	conv2d.setOverlap(o);
	conv2d.setEdgeMode(skepu::Edge::Pad);
	conv2d.setPad(0);
	conv2d(rm, m, filter);
	
	
	std::cout << "MapOverlap2D: conv2d(rm, m, filter) = " << rm << "\n";
	
	skepu::Vector<float> ans(1);
	caller(ans, 4, 5);
	std::cout << "Call: caller(ans, 4, 5) = " << ans[0] << "\n";
	
	return 0;
}

