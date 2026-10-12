// The lines of the definitions of an instance and its user function are those that a #line directive says: the code that another
// program generates (SkePY) says where it is in its own source (see tracing_definitions_line in CMakeLists.txt).
#include "../../external/catch2/catch.hpp"
#include "../../external/nlohmann/json.hpp"

#include <fstream>
#include <skepu>

#line 700 "generated/mapped.py"
float mapped_f(float x)
{
	return x * 3;
}
#line 710 "generated/mapped.py"
auto mapped = skepu::Map<1>(mapped_f);
#line 800 "tracing_definitions_line.cpp"

TEST_CASE("The definitions are where the #line directives say")
{
	skepu::Vector<float> in(4, 1.f), out(4);
	mapped(out, in);
	CHECK(out(0) == 3.f);

	SKEPU_TRACE_FLUSH();
	std::ifstream file(std::getenv("SKEPU_TRACE_FILE"));
	REQUIRE(file.good());
	nlohmann::json trace = nlohmann::json::parse(file);
	REQUIRE(trace.is_object());
	nlohmann::json instances = trace["header"]["instances"];
	REQUIRE(instances.size() == 1);
	nlohmann::json e = *instances.begin();
	CHECK(e["label"] == "mapped");
	CHECK(e["instance"]["file"] == "generated/mapped.py");
	CHECK(e["instance"]["line"] == 710);
	REQUIRE(e["user_functions"].size() == 1);
	CHECK(e["user_functions"][0]["file"] == "generated/mapped.py");
	CHECK(e["user_functions"][0]["line"] == 700);
	CHECK(e["user_functions"][0]["end_line"] == 703);
}
