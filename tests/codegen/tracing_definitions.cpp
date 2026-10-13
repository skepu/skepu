// Where the skeleton instances and their user functions are defined: the header of the trace has a table of the
// instances (see tracing_definitions in CMakeLists.txt) with the lines of the declaration and of each user function.
#include "../../external/catch2/catch.hpp"
#include "../../external/nlohmann/json.hpp"
#include "../../external/hash-library/sha256.hpp"

#include <fstream>
#include <skepu>

// (a function for each skeleton: the precompiler does not take a function that two skeletons share)
float twice(float x) { return 2 * x; }
static const int line_twice = __LINE__ - 1;
float spread(float x)
{
	float y = x + 1;
	return y;
}
static const int line_spread = __LINE__ - 5, line_spread_end = __LINE__ - 1;
float mr_map(float a, float b) { return a * b; }
float mr_add(float a, float b) { return a + b; }
float anonymous_f(float x) { return x - 1; }
float unused_f(float x) { return x; }

static const int line_named = __LINE__ + 1;
auto named = skepu::Map<1>(twice);

namespace
{
	using json = nlohmann::json;

	json instances()
	{
		SKEPU_TRACE_FLUSH();
		std::ifstream file(std::getenv("SKEPU_TRACE_FILE"));
		REQUIRE(file.good());
		json trace = json::parse(file);
		REQUIRE(trace.is_object());
		REQUIRE(trace["header"].contains("instances"));
		return trace["header"]["instances"];
	}

	json instance(json const& table, std::string const& label)
	{
		for (auto const& [id, entry] : table.items())
			if (entry["label"] == label)
				return entry;
		FAIL("no instance " << label);
		return {};
	}
}

TEST_CASE("The trace says where the instances and the user functions are defined")
{
	skepu::Vector<float> in(4, 1.f), out(4), sum(1);
	int line_multi_line = 0, line_lambda = 0;

	named(out, in);

	line_multi_line = __LINE__ + 1;
	auto multi_line = skepu::Map<1>(
		spread);
	multi_line(out, in);

	line_lambda = __LINE__ + 1;
	auto with_lambda = skepu::Map<1>([](float x) { return x + 1; });
	with_lambda(out, in);

	auto reduction = skepu::MapReduce<2>(mr_map, mr_add);
	int line_reduction = __LINE__ - 1;
	reduction(in, in);

	int line_anonymous = __LINE__ + 1;
	skepu::Map<1>(anonymous_f)(out, in);

	auto unused = skepu::Map<1>(unused_f); // (an instance that makes no call is not in the table)

	json table = instances();
	std::string file = __FILE__;

	SECTION("a named instance and a function")
	{
		json e = instance(table, "named");
		CHECK(e["instance"]["line"] == line_named);
		CHECK(e["instance"]["end_line"] == line_named);
		CHECK(!e["instance"].contains("anonymous"));
		CHECK(e["instance"]["file"].get<std::string>().find("tracing_definitions") != std::string::npos);
		REQUIRE(e["user_functions"].size() == 1);
		json f = e["user_functions"][0];
		CHECK(f["name"] == "twice");
		CHECK(f["kind"] == "function");
		CHECK(f["line"] == line_twice);
		CHECK(f["end_line"] == line_twice);
	}

	SECTION("a function of several lines, and a declaration of several lines")
	{
		json e = instance(table, "multi_line");
		CHECK(e["instance"]["line"] == line_multi_line);
		CHECK(e["instance"]["end_line"] == line_multi_line + 1);
		json f = e["user_functions"][0];
		CHECK(f["name"] == "spread");
		CHECK(f["line"] == line_spread);
		CHECK(f["end_line"] == line_spread_end);
	}

	SECTION("a lambda is a user function on the line of the instance")
	{
		json e = instance(table, "with_lambda");
		CHECK(e["instance"]["line"] == line_lambda);
		REQUIRE(e["user_functions"].size() == 1);
		json f = e["user_functions"][0];
		CHECK(f["kind"] == "lambda");
		CHECK(f["name"] == "");
		CHECK(f["line"] == line_lambda);
	}

	SECTION("a skeleton with two user functions has both, in order")
	{
		json e = instance(table, "reduction");
		CHECK(e["instance"]["line"] == line_reduction);
		REQUIRE(e["user_functions"].size() == 2);
		CHECK(e["user_functions"][0]["name"] == "mr_map");
		CHECK(e["user_functions"][1]["name"] == "mr_add");
	}

	SECTION("an anonymous instance is on the line of its call")
	{
		bool found = false;
		for (auto const& [id, e] : table.items())
			if (e["instance"].value("anonymous", false))
			{
				found = true;
				CHECK(e["instance"]["line"] == line_anonymous);
				CHECK(e["user_functions"][0]["name"] == "anonymous_f");
			}
		CHECK(found);
	}

	SECTION("an instance that makes no call is not in the table, and an instance is there once")
	{
		int named_entries = 0;
		for (auto const& [id, e] : table.items())
			named_entries += e["label"] == "named";
		CHECK(named_entries == 1);
		for (auto const& [id, e] : table.items())
			CHECK(e["label"] != "unused");
	}
}

// The header has the hash of each source file that the events name (header.files): the precompiler's for the calls of the skeletons
// (also an anonymous instance, which has code generated for it), the tracer's for the events of the program itself (a region)
TEST_CASE("The trace has the hash of each source file that its events name")
{
	skepu::Vector<float> in(4, 1.f), out(4);
	named(out, in);
	skepu::Map<1>(anonymous_f)(out, in);
	{
		SKEPU_TRACE_SCOPE("hashed region");
	}

	SKEPU_TRACE_FLUSH();
	std::ifstream file(std::getenv("SKEPU_TRACE_FILE"));
	REQUIRE(file.good());
	json trace = json::parse(file);
	REQUIRE(trace["header"].contains("files"));

	size_t named_files = 0;
	for (json const& event : trace["data"])
	{
		if (!event.contains("file") || event["file"] == "")
			continue;
		std::string name = event["file"];
		REQUIRE(trace["header"]["files"].contains(name));

		std::ifstream source(name, std::ios::binary);
		REQUIRE(source.good());
		std::ostringstream contents;
		contents << source.rdbuf();
		skepu::tracing::SHA256 sha256;
		CHECK(trace["header"]["files"][name] == sha256(contents.str()));
		++named_files;
	}
	CHECK(named_files >= 3); // two calls and the region
}
