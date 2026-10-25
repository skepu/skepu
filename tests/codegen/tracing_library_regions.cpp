// A call of a function of a SkePU library (skepu-lib: blas, randomize, dnn) is a region in the trace at the place of the call in the
// program, named after the function; a frontend that is not C++ gives the place with a CallSite object, and a call that the library makes of
// another function of the same name (an overload) is not a region of its own.
#include "../../external/catch2/catch.hpp"
#include "../../external/nlohmann/json.hpp"

#include <fstream>
#include <skepu>
#include <skepu-lib/blas.hpp>
#include <skepu-lib/randomize.hpp>
#include <skepu-lib/ml.hpp>
#include <skepu-lib/benchmark.hpp>

namespace
{
	using json = nlohmann::json;

	json read_trace()
	{
		SKEPU_TRACE_FLUSH();
		std::ifstream file(std::getenv("SKEPU_TRACE_FILE"));
		REQUIRE(file.good());
		json trace = json::parse(file);
		return trace.is_object() ? trace["data"] : trace;
	}

	// The (file, line) of the regions with the label
	std::vector<std::pair<std::string, int>> regions(json const &trace, std::string const &label)
	{
		std::vector<std::pair<std::string, int>> found;
		for (auto const &e : trace)
			if (e["type"] == "region" && e["label"] == label)
				found.push_back({e["file"], e["line"]});
		return found;
	}

	bool is_this_file(std::string const &file)
	{
		const std::string name = "tracing_library_regions.cpp";
		return file.size() >= name.size() && file.compare(file.size() - name.size(), name.size(), name) == 0;
	}
}

TEST_CASE("Calls of the library functions are regions at the lines of the calls")
{
	skepu::Vector<float> x(8, 1.f), y(8, 2.f);
	const int axpy_line = __LINE__ + 1;
	skepu::blas::axpy(8, 2.f, x, 1, y, 1);
	const int uniform_line = __LINE__ + 1;
	skepu::randomize::uniform(x, 0.f, 1.f);
	skepu::Tensor4<float> data(4, 2, 2, 1, "data", 1.f);
	const int take_line = __LINE__ + 1;
	auto first = skepu::dnn::take(data, 2);
	skepu::Vector<int> classes(4, 1);
	const int one_hot_line = __LINE__ + 1;
	auto labels = skepu::dnn::one_hot(classes, 3);   // (calls the overload that fills a tensor: no region of its own)
	(void) first; (void) labels;

	auto trace = read_trace();
	auto one = [&](std::string const &label)
	{
		auto found = regions(trace, label);
		REQUIRE(found.size() == 1);
		REQUIRE(is_this_file(found[0].first));
		return found[0].second;
	};
	REQUIRE(one("blas.axpy") == axpy_line);
	REQUIRE(one("randomize.uniform") == uniform_line);
	REQUIRE(one("dnn.take") == take_line);
	REQUIRE(one("dnn.one_hot") == one_hot_line);
}

TEST_CASE("A frontend gives the place of a call with a CallSite")
{
	skepu::Vector<float> x(8, 1.f), y(8, 2.f);
	{
#ifdef SKEPU_TRACING   // (the tool that precompiles this file does not see tracing)
		skepu::tracing::CallSite site("program.py", 42);
#endif
		skepu::blas::scal(8, 3.f, x, 1);
	}
	skepu::blas::scal(8, 3.f, y, 1);

	auto found = regions(read_trace(), "blas.scal");
	REQUIRE(found.size() == 2);
	REQUIRE(found[0] == std::make_pair(std::string("program.py"), 42));
	REQUIRE(is_this_file(found[1].first));
}

TEST_CASE("The benchmark functions are regions")
{
	int calls = 0;
	const int basic_line = __LINE__ + 1;
	skepu::benchmark::basicBenchmark(3, 8, [&](size_t) { ++calls; });
	skepu::benchmark::measureExecTime([&](int) { ++calls; }, 1);
	REQUIRE(calls == 4);

	auto trace = read_trace();
	auto basic = regions(trace, "benchmark.basicBenchmark");
	REQUIRE(basic.size() == 1);
	REQUIRE(is_this_file(basic[0].first));
	REQUIRE(basic[0].second == basic_line);
	// (one for each of the three repetitions, and the call of the program: they have no place)
	REQUIRE(regions(trace, "benchmark.measureExecTime").size() == 4);
}
