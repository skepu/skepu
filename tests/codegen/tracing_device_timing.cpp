// The times of the calls of the CUDA and OpenCL backends come from the devices (CUDA events, profiling of OpenCL markers): the host
// does not wait for the device at a call. A kernel that
// takes long is queued several times: the host is back from the calls before the kernels are done, and the trace has the time of
// the device for each call (a time of the host would be that of the launch), one after the other, as they run in the stream.
#include "../../external/catch2/catch.hpp"
#include "../../external/nlohmann/json.hpp"

#include <chrono>
#include <fstream>
#include <skepu>

float heavy(float x)
{
	float y = x;
	for (int i = 0; i < 4000; ++i)
		y = y * 0.999f + sin(y) * 0.001f;
	return y;
}

float light(float x) { return x + 1.0f; }

auto heavy_map = skepu::Map<1>(heavy);
auto light_map = skepu::Map<1>(light);

#ifdef TIMING_OPENCL
static const char *kBackend = "opencl";
#else
static const char *kBackend = "cuda";
#endif

TEST_CASE("the trace has the time of the device for calls that are not waited for")
{
	using clock = std::chrono::steady_clock;
	const size_t n = 1 << 22, calls = 6;
	skepu::Vector<float> in(n, "in", 0.5f), out(n, "out");
	heavy_map.setBackend(skepu::BackendSpec{kBackend});
	heavy_map.setLabel("heavy_map");
	heavy_map(out, in); // (the first use of the device: its context, the code of the kernel)
	out.flush();

	const auto host_start = clock::now();
	for (size_t i = 0; i < calls; ++i)
		heavy_map(out, in);
	const double host_us = std::chrono::duration<double, std::micro>(clock::now() - host_start).count();
	out.flush(); // (the program waits for the device here)
	const double total_us = std::chrono::duration<double, std::micro>(clock::now() - host_start).count();

	skepu::tracing::tracer().flush();
	const char *file = std::getenv("SKEPU_TRACE_FILE");
	REQUIRE(file);
	std::ifstream in_file(file);
	nlohmann::json trace = nlohmann::json::parse(in_file);

	std::vector<std::pair<unsigned long, unsigned long>> intervals;
	for (auto const &e : trace["data"])
		if (e.value("type", "") == "skeleton_call" && e.value("label", "") == "heavy_map")
			intervals.emplace_back(e["start"].get<unsigned long>(), e["end"].get<unsigned long>());
	REQUIRE(intervals.size() == calls + 1);
	intervals.erase(intervals.begin()); // (the first call, which started the device)

	// the host did not wait for the kernels: it is back long before they are done
	CHECK(host_us < total_us / 2);
	double traced_us = 0;
	for (auto const &i : intervals)
	{
		CHECK(i.first <= i.second);
		traced_us += (double)(i.second - i.first);
	}
	// ... and the trace has their time: most of the time that the program waited, not the time of a launch
	CHECK(traced_us > 0.7 * total_us);
	// the kernels run one after the other in the stream (the times are on the clock of the host: a little room for the clocks)
	for (size_t i = 1; i < intervals.size(); ++i)
		CHECK(intervals[i].first + 100 >= intervals[i - 1].second);
}

// Many calls before the trace is written: what the devices keep for the times has a limit, and the times are still right
TEST_CASE("the timers that are not resolved are limited")
{
	const size_t n = 1 << 16, calls = 300, limit = 8;
	skepu::tracing::maxPendingDeviceTimers() = limit;
	skepu::Vector<float> in(n, "in", 0.5f), out(n, "out");
	light_map.setBackend(skepu::BackendSpec{kBackend});
	light_map.setLabel("light_map");
	skepu::tracing::tracer().flush();

	size_t most = 0;
	for (size_t i = 0; i < calls; ++i)
	{
		light_map(out, in);
		most = std::max(most, skepu::tracing::pendingDeviceTimers());
	}
	out.flush();
	CHECK(most <= limit);

	skepu::tracing::tracer().flush();
	CHECK(skepu::tracing::pendingDeviceTimers() == 0);
	const char *file = std::getenv("SKEPU_TRACE_FILE");
	REQUIRE(file);
	std::ifstream in_file(file);
	nlohmann::json trace = nlohmann::json::parse(in_file);
	size_t found = 0;
	unsigned long previous_end = 0;
	for (auto const &e : trace["data"])
		if (e.value("type", "") == "skeleton_call" && e.value("label", "") == "light_map")
		{
			++found;
			const unsigned long start = e["start"].get<unsigned long>(), end = e["end"].get<unsigned long>();
			CHECK(start <= end);
			CHECK(start + 100 >= previous_end);
			previous_end = end;
		}
	CHECK(found == calls);
}
