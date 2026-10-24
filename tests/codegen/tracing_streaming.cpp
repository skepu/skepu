// The trace is written as the events are replayed (see tracing_streaming in CMakeLists.txt, which counts the events of the final file):
//  - the file is complete after a flush, with every event so far in the order they were recorded, also when the tracer replayed and wrote
//    events by itself (it does when many wait), and the next flush adds to it;
//  - a container that is renamed gets the name in its events that are among the newest (the window of kEditableEvents events that can still be
//    changed), and not in older ones;
//  - several threads recording at once, while the tracer replays: none lost, and the events of a thread stay in their order.
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <skepu>
#include "../../external/nlohmann/json.hpp"

static int failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::cerr << "FAILED, line " << __LINE__ << ": " #condition "\n"; ++failures; } } while (0)

// The trace file as it is now (it throws if it is not valid JSON)
static nlohmann::json readTrace()
{
	std::ifstream in(std::getenv("SKEPU_TRACE_FILE"));
	std::stringstream text;
	text << in.rdbuf();
	return nlohmann::json::parse(text.str());
}

static const nlohmann::json *find(const nlohmann::json &data, unsigned long id)
{
	for (const auto &event : data)
		if (event.value("object_id", 0UL) == id)
			return &event;
	return nullptr;
}

int main()
{
	// 1. Many events, flushed in the middle of the run
	const int first = 40000;   // (more than the tracer holds before it replays by itself, and than the window)
	for (int i = 0; i < first; ++i)
		SKEPU_TRACE_ALLOCATION(1000000 + i, "event " + std::to_string(i), i);
	SKEPU_TRACE_FLUSH();
	nlohmann::json trace = readTrace();
	CHECK(trace.is_object() && trace.contains("data") && trace.contains("header"));
	CHECK(trace["data"].size() == static_cast<size_t>(first));
	bool inOrder = true;
	for (int i = 0; i < first && i < static_cast<int>(trace["data"].size()); ++i)
		inOrder = inOrder && trace["data"][i].value("object_id", 0UL) == static_cast<unsigned long>(1000000 + i);
	CHECK(inOrder);

	// 2. The file is added to, and a container is renamed among the newest events but not among the old ones
	SKEPU_TRACE_ALLOCATION(2000000, "early name", 1);
	for (int i = 0; i < 500; ++i)
		SKEPU_TRACE_ALLOCATION(3000000 + i, "filler", i);
	SKEPU_TRACE_RELABEL(2000000, "late name");
	SKEPU_TRACE_RELABEL(1000000, "too late");   // (its event is 40000 events back: out of the window)
	SKEPU_TRACE_FLUSH();
	trace = readTrace();
	CHECK(trace["data"].size() == static_cast<size_t>(first + 1 + 500));
	const nlohmann::json *renamed = find(trace["data"], 2000000), *old = find(trace["data"], 1000000);
	CHECK(renamed && renamed->value("label", "") == "late name");
	CHECK(old && old->value("label", "") == "event 0");

	// 3. Several threads at once, while the tracer replays and writes by itself
	const int threads = 4, perThread = 20000;
	std::vector<std::thread> workers;
	for (int t = 0; t < threads; ++t)
		workers.emplace_back([=]
		{
			for (int i = 0; i < perThread; ++i)
				SKEPU_TRACE_ALLOCATION(4000000 + t * 100000 + i, "thread event", i);
		});
	for (auto &worker : workers)
		worker.join();
	SKEPU_TRACE_FLUSH();
	trace = readTrace();
	CHECK(trace["data"].size() == static_cast<size_t>(first + 1 + 500 + threads * perThread));
	std::vector<long> last(threads, -1);
	std::vector<int> count(threads, 0);
	bool threadsInOrder = true;
	for (const auto &event : trace["data"])
	{
		const unsigned long id = event.value("object_id", 0UL);
		if (id < 4000000)
			continue;
		const int t = static_cast<int>((id - 4000000) / 100000);
		const long i = static_cast<long>((id - 4000000) % 100000);
		threadsInOrder = threadsInOrder && i > last[t];
		last[t] = i;
		++count[t];
	}
	CHECK(threadsInOrder);
	for (int t = 0; t < threads; ++t)
		CHECK(count[t] == perThread);

	return failures;   // (the events of the final file are counted by the test that follows)
}
