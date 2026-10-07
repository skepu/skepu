// Synthetic example for SkeVU: one program that uses the sequential, OpenMP and OpenCL backends, and the
// host, on the same containers. A trace of it shows when the data has to cross between host and device.
// (Illustrative: the trace was made by examples/synthetic_traces.py, this program has not been run.)
#include <iostream>
#include <skepu>
#include <skepu-lib/io.hpp>

using skepu::Backend;
using skepu::BackendSpec;

constexpr size_t N = 1 << 20;

int main()
{
	auto generate = skepu::Map<0>([](skepu::Index1D i) { return std::sin(i.i * 0.001f); });
	auto scale    = skepu::Map<1>([](float x) { return 2.0f * x; });
	auto offset   = skepu::Map<1>([](float x) { return x + 1.0f; });
	auto square   = skepu::Map<1>([](float x) { return x * x; });
	auto total    = skepu::Reduce([](float a, float b) { return a + b; });

	skepu::Vector<float> signal(N, "Signal");
	skepu::Vector<float> scaled(N, "Scaled");
	{ SKEPU_TRACE_SCOPE("Sequential then OpenMP");
		generate.setBackend(BackendSpec(Backend::Type::CPU)); generate(signal);
		scale.setBackend(BackendSpec(Backend::Type::OpenMP));     scale(scaled, signal);
	}

	{ SKEPU_TRACE_SCOPE("OpenCL back to back");
		skepu::Vector<float> stage(N, "Stage");
		skepu::Scalar<float> sum("Sum");
		offset.setBackend(BackendSpec(Backend::Type::OpenCL)); offset(stage, scaled);
		square(stage, stage);
		scale.setBackend(BackendSpec(Backend::Type::OpenCL)); scale(stage, stage);
		total.setBackend(BackendSpec(Backend::Type::OpenCL));
		total(sum, stage);
		std::cout << "sum " << sum << "\n";
	}

	// Alternating backends: every switch moves the vector to the other side.
	auto step_a = skepu::Map<1>([](float x) { return x * 0.5f + 1.0f; });
	auto step_b = skepu::Map<1>([](float x) { return x - std::floor(x); });
	{ SKEPU_TRACE_SCOPE("Alternating backends");
		skepu::Vector<float> state(N, "State", 0.5f);
		step_a.setBackend(BackendSpec(Backend::Type::OpenCL));
		step_b.setBackend(BackendSpec(Backend::Type::OpenMP));
		for (int i = 0; i < 3; ++i)
		{
			SKEPU_TRACE_SCOPE("Iteration");
			step_a(state, state);
			step_b(state, state);
		}
	}

	{ SKEPU_TRACE_SCOPE("One backend");
		skepu::Vector<float> state2(N, "State (OpenCL only)", 0.5f);
		step_b.setBackend(BackendSpec(Backend::Type::OpenCL));
		for (int i = 0; i < 3; ++i)
		{
			SKEPU_TRACE_SCOPE("Iteration");
			step_a(state2, state2);
			step_b(state2, state2);
		}
	}

	auto blur = skepu::Map<1>([](float x) { return 0.9f * x; });
	{ SKEPU_TRACE_SCOPE("Host code between kernels");
		skepu::Vector<float> image(N, "Image", 1.0f);
		blur.setBackend(BackendSpec(Backend::Type::OpenCL));
		blur(image, image);
		host_filter(image);   // plain C++ on the elements, through image[i]
		blur(image, image);
	}

	{ SKEPU_TRACE_SCOPE("Single elements");
		skepu::Vector<float> result(N, "Result");
		generate.setBackend(BackendSpec(Backend::Type::OpenCL)); generate(result);
		float first = result[0];       // reads one element: the whole vector comes back, the device copy stays good
		offset(result, result);       // no transfer in
		result[0] = 0.0f;               // writes one element: the whole vector comes back, the device copy is stale
		offset(result, result);       // the whole vector goes out again
	}

	// The backend chosen by the size: sequential for small data, OpenMP for medium, OpenCL for large.
	auto choose = [](size_t n) { return n < (1 << 12) ? Backend::Type::CPU : n < (1 << 18) ? Backend::Type::OpenMP : Backend::Type::OpenCL; };
	{ SKEPU_TRACE_SCOPE("Backend by size");
		skepu::Vector<float> small(1024, "Small", 1.0f);
		scale.setBackend(BackendSpec(choose(small.size())));
		scale(small, small);
		skepu::Vector<float> medium(65536, "Medium", 1.0f);
		scale.setBackend(BackendSpec(choose(medium.size())));
		scale(medium, medium);
		skepu::Vector<float> large(4194304, "Large", 1.0f);
		scale.setBackend(BackendSpec(choose(large.size())));
		scale(large, large);
	}

	return 0;
}
