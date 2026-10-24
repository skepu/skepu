// skepu-tool puts a trace region at the start of the functions of the program that make events (they use skeletons or containers, or call
// such functions), and not at the others, or at the functions that a user function calls. A region that has fewer than two events is not
// in the trace (see tracing_auto_regions).
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <skepu>
#include "../../external/nlohmann/json.hpp"

static int failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::cerr << "FAILED, line " << __LINE__ << ": " #condition "\n"; ++failures; } } while (0)

static int helper(int a) { return a * 2; }                     // only a user function calls it: no region
static int plain(int a, int b) { return a + b; }               // no events
static int calls_plain(int a) { return plain(a, a); }          // calls a function that makes none

int scale(int a, int b) { return helper(a) + b; }

static auto scale_map = skepu::Map<2>(scale);

// two skeleton calls: a region
static void stage(skepu::Vector<int> &a, skepu::Vector<int> &b)
{
	scale_map(a, a, b);
	scale_map(b, a, b);
}

// one skeleton call: its region is not in the trace
static void single(skepu::Vector<int> &a, skepu::Vector<int> &b)
{
	scale_map(a, a, b);
}

// a function that calls functions that make events
static void pipeline()
{
	skepu::Vector<int> a(100, 1), b(100, 2);
	stage(a, b);
	single(a, b);
	stage(b, a);
}

namespace app
{
	struct Runner
	{
		void run() { pipeline(); }
	};
}

int main()
{
	CHECK(calls_plain(2) == 4);
	app::Runner().run();
	SKEPU_TRACE_FLUSH();

	std::ifstream in(std::getenv("SKEPU_TRACE_FILE"));
	std::stringstream text;
	text << in.rdbuf();
	const nlohmann::json data = nlohmann::json::parse(text.str())["data"];

	auto count = [&](const std::string &label)
	{
		int n = 0;
		for (const auto &event : data)
			if (event.value("type", "") == "region" && event.value("label", "") == label)
				++n;
		return n;
	};
	CHECK(count("pipeline") == 1);
	CHECK(count("stage") == 2);
	// (the regions around a region that is in the trace are in it, also if they hold nothing else; main has none)
	CHECK(count("app::Runner::run") == 1 && count("main") == 0);
	CHECK(count("single") == 0);
	CHECK(count("plain") == 0 && count("calls_plain") == 0 && count("helper") == 0 && count("scale") == 0);

	// the calls of stage are in it, and it is in pipeline
	int stageCalls = 0;
	for (const auto &event : data)
	{
		if (event.value("type", "") == "region" && event.value("label", "") == "stage")
			CHECK(event.value("region", "") == "pipeline" && event.value("region_depth", 0) == 3);
		if (event.value("type", "") == "skeleton_call" && event.value("region", "") == "stage")
			++stageCalls;
	}
	CHECK(stageCalls == 4);
	return failures;
}
