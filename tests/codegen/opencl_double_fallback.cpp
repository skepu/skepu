#define SKEPU_ENABLE_EXCEPTIONS
#include "../../external/catch2/catch.hpp"

#include <cstdlib>
#include <skepu>
#include <skepu-lib/util.hpp>
#include <test_helper_automatic_setup.hpp>

// Skeletons with double precision on the OpenCL backend: on a device that has it they run there, on one that does not (as the Apple GPUs) they run on another backend,
// with a message on stderr, and the kernels that cannot be built for such a device are not built. Skeletons without doubles are not affected. With the environment variable
// SKEPU_STRICT_BACKEND set, the skeleton that would change backend is an error instead. Built for OpenCL only (see CMakeLists.txt).

double doubled(double x) { return 2 * x; }
float halved(float x) { return x / 2; }
skepu::multiple<float, double> both(float x) { return skepu::ret(x, (double)x * 3); }

auto double_skeleton = skepu::Map<1>(doubled);
auto float_skeleton = skepu::Map<1>(halved);
auto multiple_skeleton = skepu::Map<1>(both);
auto double_sum = skepu::Reduce(skepu::util::add<double>);

namespace
{
	const skepu::BackendSpec opencl{skepu::Backend::Type::OpenCL};

	// The backend that a skeleton used for its last call. (A skeleton made by the precompiler has the member; the sequential skeleton that the precompiler itself
	// reads the program with does not, and the answer is not used there.)
	template<typename S, typename = void>
	struct has_selected_spec: std::false_type {};

	template<typename S>
	struct has_selected_spec<S, std::void_t<decltype(std::declval<S &>().m_selected_spec)>>: std::true_type {};

	template<typename S>
	skepu::Backend::Type used(S &skeleton)
	{
		if constexpr (has_selected_spec<S>::value)
			return skeleton.m_selected_spec->type();
		else
			return skepu::Backend::Type::CPU;
	}
}

TEST_CASE("A skeleton with doubles runs on OpenCL if a device has them, and on another backend if not")
{
	const size_t n{200};
	skepu::Vector<double> in(n), out(n);
	skepu::Vector<float> fin(n), fout(n);
	for (size_t i = 0; i < n; ++i) { in[i] = (double)i; fin[i] = (float)i; }

	double_skeleton.setBackend(opencl);
	float_skeleton.setBackend(opencl);
	double_skeleton(out, in);
	float_skeleton(fout, fin);
	out.updateHost(); fout.updateHost();
	for (size_t i = 0; i < n; ++i)
	{
		REQUIRE(out(i) == 2.0 * i);
		REQUIRE(fout(i) == 0.5f * i);
	}

	// the float skeleton is on OpenCL; the double one is, if a device can run it
	REQUIRE(used(float_skeleton) == skepu::Backend::Type::OpenCL);
	if (skepu::backend::openclSupportsDouble())
		REQUIRE(used(double_skeleton) == skepu::Backend::Type::OpenCL);
	else
		REQUIRE(used(double_skeleton) != skepu::Backend::Type::OpenCL);
}

TEST_CASE("A double in one result of a skepu::multiple, and in the types of a reduction, is a need of double precision")
{
	const size_t n{100};
	skepu::Vector<float> in(n), floats(n);
	skepu::Vector<double> doubles(n), values(n);
	for (size_t i = 0; i < n; ++i) { in[i] = (float)i; values[i] = 0.5 * i; }

	multiple_skeleton.setBackend(opencl);
	multiple_skeleton(floats, doubles, in);
	double_sum.setBackend(opencl);
	const double total = double_sum(values);

	floats.updateHost(); doubles.updateHost();
	for (size_t i = 0; i < n; ++i)
	{
		REQUIRE(floats(i) == (float)i);
		REQUIRE(doubles(i) == 3.0 * i);
	}
	REQUIRE(total == 0.5 * (n * (n - 1) / 2));

	if (!skepu::backend::openclSupportsDouble())
	{
		REQUIRE(used(multiple_skeleton) != skepu::Backend::Type::OpenCL);
		REQUIRE(used(double_sum) != skepu::Backend::Type::OpenCL);
	}
}

TEST_CASE("With SKEPU_STRICT_BACKEND set, a skeleton that cannot run on the OpenCL devices is an error, and the others are not")
{
	const size_t n{50};
	skepu::Vector<double> in(n, 1.0), out(n);
	skepu::Vector<float> fin(n, 1.f), fout(n);
	double_skeleton.setBackend(opencl);
	float_skeleton.setBackend(opencl);

	setenv("SKEPU_STRICT_BACKEND", "1", 1);
	if (skepu::backend::openclSupportsDouble())
		REQUIRE_NOTHROW(double_skeleton(out, in));                // (nothing to change)
	else
		REQUIRE_THROWS(double_skeleton(out, in));
	REQUIRE_NOTHROW(float_skeleton(fout, fin));

	// 0 is not strict
	setenv("SKEPU_STRICT_BACKEND", "0", 1);
	REQUIRE_NOTHROW(double_skeleton(out, in));
	unsetenv("SKEPU_STRICT_BACKEND");
	REQUIRE_NOTHROW(double_skeleton(out, in));
}
