// Regions of several threads that overlap in time: an event belongs to the region of the thread that
// recorded it (see tracing_regions_threads in CMakeLists.txt, which checks the trace).
#include <atomic>
#include <thread>
#include <vector>
#include <skepu>

int main()
{
	const int threads = 4, per_thread = 50;
	std::atomic<int> started{0};
	std::vector<std::thread> workers;
	for (int t = 0; t < threads; ++t)
		workers.emplace_back([&, t]
		{
			const std::string region = "R" + std::to_string(t);
			skepu::tracing::tracer().beginRegion(region, __FILE__, __LINE__);
			// all regions are open before any thread records an event
			++started;
			while (started < threads)
				std::this_thread::yield();
			for (int i = 0; i < per_thread; ++i)
			{
				SKEPU_TRACE_ALLOCATION(skepu::UniqueIdentifier::generate(), region + " event", i);
				std::this_thread::yield();
			}
			skepu::tracing::tracer().endRegion();
		});
	for (auto &worker : workers)
		worker.join();
	return 0;
}
