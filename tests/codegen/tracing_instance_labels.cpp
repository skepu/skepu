// A skeleton instance that is named again and again, as the layers of a network that share one name it for their own calls (SkePU-DNN does): every call has the
// name that the instance had when it was made, not the last one. (Naming an instance renames the events of its kernel builds before its first call, nothing later.)
#include "../../external/catch2/catch.hpp"
#include "../../external/nlohmann/json.hpp"

#include <fstream>
#include <skepu>

namespace
{
	using json = nlohmann::json;

	int plus_one(int x) { return x + 1; }
}

TEST_CASE("The calls of an instance that is named again keep the name they were made with")
{
	skepu::Vector<int> in(8, 1), out(8, 0);
	auto step = skepu::Map<1>(plus_one);

	step.setLabel("first name");
	step(out, in);
	step.setLabel("second name");
	step(out, in);
	step(out, in);
	step.setLabel("third name");
	step(out, in);

	SKEPU_TRACE_FLUSH();
	std::ifstream file(std::getenv("SKEPU_TRACE_FILE"));
	REQUIRE(file.good());
	json trace = json::parse(file);
	if (trace.is_object())
		trace = trace["data"];

	std::vector<std::string> names;
	for (auto const &e : trace)
		if (e["type"] == "skeleton_call")
			names.push_back(e["label"]);
	REQUIRE(names == std::vector<std::string>{"first name", "second name", "second name", "third name"});
}
