#include <iostream>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/blas.hpp>

void containers(size_t size)
{
	skepu::Vector<int> vec1("Vector 1", size); // Label: "Vector 1"
	skepu::Vector<int> vec2(size); // Label: default (data-pointer address)
	vec2.setLabel("Vector 2"); // Label: "Vector 2"

	auto skeleton = skepu::Map(func); // Label: default ("skeleton")
	skeleton.setLabel("Skeleton Call") // Label: "Skeleton Call"
}


void regions(size_t size)
{
	auto initial_work = skepu::Map([](float val) { return val; });
	auto some_work = skepu::Map([](float val) { return val; });
	auto exit_criteria = skepu::MapReduce([](float val, int i) { return (i == 3) ? 1 : 0; }, [](float a, float b) { return a + b; });
	auto some_work = skepu::Map([](float val) { return val; });
	auto final_work = skepu::Map([](float val) { return val; });

	skepu::Vector<float> data(N, "");

	SKEPU_TRACE_REGION_BEGIN("Initial phase");
	initial_work(data, data /*...*/);
	SKEPU_TRACE_REGION_END();

	for (int i = 0; i < ITERATIONS; ++i)
	{
		SKEPU_TRACE_SCOPE("Iteration phase");
		some_work(my_data, ...);
		auto c = exit_criteria(data, data /*...*/);
		if (c < EARLY_RETURN) return;
		more_work(data, data, i /*...*/);
	}

	SKEPU_TRACE_REGION("Final phase", [&] {
		final_work(data, data, /*...*/);
	});

}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::external([&]{ std::cout << "Usage: " << argv[0] << " size backend\n"; });
		exit(1);
	}

	const size_t n = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);

	containers(n);
	regions(n);

	return 0;
}
