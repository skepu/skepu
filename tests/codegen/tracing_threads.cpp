// Events from several threads at once: the tracer takes them all, none lost and nothing broken
// (see tracing_threads in CMakeLists.txt, which counts the events in the trace).
#include <thread>
#include <vector>
#include <skepu>

int main()
{
	const int threads = 8, per_thread = 2000;
	std::vector<std::thread> workers;
	for (int t = 0; t < threads; ++t)
		workers.emplace_back([=]
		{
			for (int i = 0; i < per_thread; ++i)
				SKEPU_TRACE_ALLOCATION(skepu::UniqueIdentifier::generate(), "thread event", i);
		});
	for (auto &worker : workers)
		worker.join();
	return 0;
}
