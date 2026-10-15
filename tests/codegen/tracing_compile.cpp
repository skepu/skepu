// The builds of the OpenCL kernels are events in the trace (see tracing_compile in CMakeLists.txt): "kernel_compile", with the id and the label of the
// skeleton instance that the kernel is of (the id is the one of its calls, and the label is the name that the instance gets after it is made), the
// device, the time, and whether the build worked. The kernels of the library (the transpose of a matrix) have no instance.
#include "../../external/catch2/catch.hpp"
#include "../../external/nlohmann/json.hpp"

#include <fstream>
#include <skepu>
#include <skepu-lib/util.hpp>

namespace
{
	using json = nlohmann::json;

	json load_trace()
	{
		SKEPU_TRACE_FLUSH();
		std::ifstream file(std::getenv("SKEPU_TRACE_FILE"));
		REQUIRE(file.good());
		json trace = json::parse(file);
		return trace.is_object() ? trace["data"] : trace;
	}

	json first_of(json const& trace, std::string const& type, std::string const& label)
	{
		for (auto const& e : trace)
			if (e["type"] == type && e["label"] == label)
				return e;
		FAIL("no " << type << " " << label);
		return json();
	}

	int twice(int x) { return 2 * x; }
}

TEST_CASE("The build of a kernel is an event of its skeleton instance")
{
	skepu::Vector<int> numbers(8);
	for (int i = 0; i < 8; ++i) numbers[i] = i;
	auto scale_by_two = skepu::Map<1>(twice);
	auto add_up = skepu::Reduce(skepu::util::add<int>);

	skepu::Vector<int> doubled(8);
	scale_by_two(doubled, numbers);
	const int total = add_up(doubled);
	REQUIRE(total == 56);

	const json trace = load_trace();
	for (std::string label : {"scale_by_two", "add_up"})
	{
		const json compile = first_of(trace, "kernel_compile", label);
		const json call = first_of(trace, "skeleton_call", label);
		INFO(label);
		REQUIRE(compile["object_id"] == call["object_id"]);
		REQUIRE(compile["backend"] == "OpenCL");
		REQUIRE(compile["ok"] == true);
		REQUIRE(compile["device"].get<std::string>().size() > 0);
		REQUIRE(compile["source_bytes"].get<size_t>() > 0);
		REQUIRE(compile["start"].get<unsigned long>() <= compile["end"].get<unsigned long>());
		// the kernel is built before the instance is called
		REQUIRE(compile["end"].get<unsigned long>() <= call["start"].get<unsigned long>());
		REQUIRE_FALSE(compile.contains("error"));
	}
}
