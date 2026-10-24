// Regions that skepu-tool marks (SKEPU_TRACE_AUTO_REGION) are in the trace only if they have two events or more; the events of one that has
// none or one are as if it were not there, except that they are in the region around it. A region that the program marks is always there.
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <skepu>
#include "../../external/nlohmann/json.hpp"

static int failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::cerr << "FAILED, line " << __LINE__ << ": " #condition "\n"; ++failures; } } while (0)
#define AUTO(name) SKEPU_TRACE_AUTO_REGION(name, __FILE__, __LINE__)
#define EVENT(name) SKEPU_TRACE_ALLOCATION(skepu::UniqueIdentifier::generate(), name, __LINE__)

static nlohmann::json trace()
{
	SKEPU_TRACE_FLUSH();
	std::ifstream in(std::getenv("SKEPU_TRACE_FILE"));
	std::stringstream text;
	text << in.rdbuf();
	return nlohmann::json::parse(text.str())["data"];
}

static const nlohmann::json *find(const nlohmann::json &data, const std::string &label, const std::string &type)
{
	for (const auto &event : data)
		if (event.value("label", "") == label && event.value("type", "") == type)
			return &event;
	return nullptr;
}
static const nlohmann::json *region(const nlohmann::json &data, const std::string &label) { return find(data, label, "region"); }
static const nlohmann::json *event(const nlohmann::json &data, const std::string &label) { return find(data, label, "allocation"); }
static std::string inRegion(const nlohmann::json *e) { return e ? e->value("region", "-") : "missing"; }

int main()
{
	// no event, or one event: the region is not there, and the event is where the region was
	{ AUTO("empty"); }
	{ AUTO("single"); EVENT("e single"); }
	// two events: the region is there
	{ AUTO("pair"); EVENT("e pair 1"); EVENT("e pair 2"); }
	// a region in a region that have one event between them: neither is there
	{ AUTO("outer"); { AUTO("inner"); EVENT("e nested"); } }
	// an event before the inner region: the outer is there with two events, the inner is not
	{ AUTO("outer2"); EVENT("e before"); { AUTO("inner2"); EVENT("e inside"); } }
	// two events in the inner region: both are there, in order
	{ AUTO("outer3"); { AUTO("inner3"); EVENT("e deep 1"); EVENT("e deep 2"); } }
	// a region that the program marks holds a region that is not there
	SKEPU_TRACE_REGION_BEGIN("manual"); { AUTO("hidden"); EVENT("e manual"); } SKEPU_TRACE_REGION_END();
	SKEPU_TRACE_REGION_BEGIN("manual empty"); { AUTO("hidden 2"); } SKEPU_TRACE_REGION_END();
	// a container that is named while its event waits
	{
		AUTO("named");
		const auto id = skepu::UniqueIdentifier::generate();
		SKEPU_TRACE_ALLOCATION(id, "old name", __LINE__);
		SKEPU_TRACE_RELABEL(id, "new name");
	}
	EVENT("e last");

	const nlohmann::json data = trace();
	CHECK(!region(data, "empty"));
	CHECK(!region(data, "single") && inRegion(event(data, "e single")) == "-");
	CHECK(region(data, "pair") && region(data, "pair")->value("region_depth", 0) == 1);
	CHECK(inRegion(event(data, "e pair 1")) == "pair" && inRegion(event(data, "e pair 2")) == "pair");
	CHECK(!region(data, "outer") && !region(data, "inner") && inRegion(event(data, "e nested")) == "-");
	CHECK(region(data, "outer2") && !region(data, "inner2"));
	CHECK(inRegion(event(data, "e before")) == "outer2" && inRegion(event(data, "e inside")) == "outer2");
	CHECK(region(data, "outer3") && region(data, "inner3"));
	CHECK(region(data, "inner3") && region(data, "inner3")->value("region_depth", 0) == 2 && inRegion(region(data, "inner3")) == "outer3");
	CHECK(inRegion(event(data, "e deep 1")) == "inner3" && inRegion(event(data, "e deep 2")) == "inner3");
	CHECK(region(data, "manual") && !region(data, "hidden") && inRegion(event(data, "e manual")) == "manual");
	CHECK(region(data, "manual empty") && !region(data, "hidden 2"));
	CHECK(!event(data, "old name") && !region(data, "named") && inRegion(event(data, "new name")) == "-");
	CHECK(inRegion(event(data, "e last")) == "-");

	// the order of the events is the order they were made in
	std::vector<std::string> order;
	for (const auto &e : data)
		order.push_back(e.value("label", ""));
	const std::vector<std::string> expected{"e single", "pair", "e pair 1", "e pair 2", "e nested", "outer2", "e before", "e inside",
		"outer3", "inner3", "e deep 1", "e deep 2", "manual", "e manual", "manual empty", "new name", "e last"};
	CHECK(order == expected);
	if (order != expected)
		for (const auto &label : order)
			std::cerr << "  " << label << "\n";

	return failures;
}
