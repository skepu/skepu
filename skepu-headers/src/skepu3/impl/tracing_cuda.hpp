// The times of the calls of the CUDA backend, from CUDA events. A kernel is started and the host goes on: to find when it was done the
// program would have to wait for the device at every call. Instead the call records an event on the stream before its work and one
// after it, which costs nothing while the device runs, and the times are read from them when the trace is written (the events of the
// trace are replayed then: see Tracer::flush). The times of the device are put on the clock of the host with one event that is recorded
// and waited for once for each device: the time of the host when it had happened, plus the time between that event and the others.
#pragma once

#if defined(SKEPU_TRACING) && defined(SKEPU_CUDA)

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "tracing.h"

namespace skepu
{
	namespace tracing
	{
		class CudaDeviceTimer: public DeviceTimer
		{
		public:
			// Starts the timing on the current device: an event on its default stream (the work of the skeletons is there)
			CudaDeviceTimer()
			{
				cudaGetDevice(&m_device);
				anchor(m_device); // (before the start event)
				if (cudaEventCreate(&m_start_event) != cudaSuccess)
					m_start_event = nullptr;
				else
					cudaEventRecord(m_start_event, 0);
			}

			// The work is done on the device when everything before this event has been done: record one for each device that the call used
			void recordEnd(int device)
			{
				int current = 0;
				cudaGetDevice(&current);
				cudaSetDevice(device);
				anchor(device);
				cudaEvent_t event;
				if (cudaEventCreate(&event) == cudaSuccess)
				{
					cudaEventRecord(event, 0);
					std::lock_guard<std::mutex> lock(m_events_mutex);
					m_ends.emplace_back(device, event);
				}
				cudaSetDevice(current);
			}

			~CudaDeviceTimer() override { release(); }

		protected:
			// (not before the end of the call has been recorded)
			bool ready() override
			{
				std::lock_guard<std::mutex> lock(m_events_mutex);
				if (m_ends.empty())
					return false;
				for (auto &end : m_ends)
					if (cudaEventQuery(end.second) == cudaErrorNotReady)
						return false;
				return true;
			}

			// (this waits for the device if its work is not done: only when the trace is written, or when the host is far ahead of the device)
			void read() override
			{
				std::lock_guard<std::mutex> lock(m_events_mutex);
				if (m_start_event)
					m_start = hostTimeOf(m_device, m_start_event); // (the times of the base class)
				for (auto &end : m_ends)
					if (cudaEventSynchronize(end.second) == cudaSuccess)
						m_end = std::max(m_end, hostTimeOf(end.first, end.second));
				releaseLocked();
			}

		private:
			void release()
			{
				std::lock_guard<std::mutex> lock(m_events_mutex);
				releaseLocked();
			}

			void releaseLocked()
			{
				if (m_start_event) { cudaEventDestroy(m_start_event); m_start_event = nullptr; }
				for (auto &end : m_ends) cudaEventDestroy(end.second);
				m_ends.clear();
			}

			struct Anchor { cudaEvent_t event = nullptr; unsigned long hostTime = 0; };

			// The reference of a device: an event that has happened, and the time of the host just after
			static Anchor &anchor(int device)
			{
				// (not destroyed: the trace is written when the program exits, after the statics of this function)
				static std::mutex &mutex = *new std::mutex;
				static std::map<int, Anchor> &anchors = *new std::map<int, Anchor>;
				std::lock_guard<std::mutex> lock(mutex);
				Anchor &a = anchors[device];
				if (!a.event)
				{
					int current = 0;
					cudaGetDevice(&current);
					cudaSetDevice(device);
					if (cudaEventCreate(&a.event) == cudaSuccess)
					{
						cudaEventRecord(a.event, 0);
						cudaEventSynchronize(a.event);
						a.hostTime = timestamp_now();
					}
					else
						a.event = nullptr;
					cudaSetDevice(current);
				}
				return a;
			}

			// The time of the host (microseconds) of an event of the device; 0 if the device cannot say
			static unsigned long hostTimeOf(int device, cudaEvent_t event)
			{
				const Anchor &a = anchor(device);
				float milliseconds = 0;
				if (!a.event || cudaEventElapsedTime(&milliseconds, a.event, event) != cudaSuccess || milliseconds < 0)
					return 0;
				return a.hostTime + (unsigned long)(milliseconds * 1000.f);
			}

			int m_device = 0;
			cudaEvent_t m_start_event = nullptr;
			std::vector<std::pair<int, cudaEvent_t>> m_ends;
			std::mutex m_events_mutex;
		};

		// The event of a call of a skeleton in the CUDA backend. It is made at the start of the function that runs the call on the devices,
		// and the call is traced when the function returns (by any return): the devices are not waited for, see CudaDeviceTimer. The arguments
		// are those of Tracer::call, found at the start (the identities of the containers and of the random stream do not change in the call).
		class CudaCallScope
		{
		public:
			template<typename Environment>
			CudaCallScope(
				Environment *environment, size_t maxDevices,
				std::string pattern, backend::SkeletonBase *skeleton,
				std::vector<size_t> &&elements,
				std::vector<TraceID> &&elwise_outputs,
				std::vector<TraceID> &&elwise_inputs,
				std::vector<TraceID> &&proxy_inputs,
				std::vector<TraceID> &&scalar_inputs,
				TraceID prng)
			: m_pattern{std::move(pattern)}, m_skeleton{skeleton}, m_elements{std::move(elements)},
			  m_elwise_outputs{std::move(elwise_outputs)}, m_elwise_inputs{std::move(elwise_inputs)},
			  m_proxy_inputs{std::move(proxy_inputs)}, m_scalar_inputs{std::move(scalar_inputs)}, m_prng{prng}
			{
				// (the devices that the call uses: the best one, or the first ones if the call is divided between devices)
				const size_t count = std::min(maxDevices, environment->m_devices_CU.size());
				if (count <= 1)
					m_devices.push_back((int)environment->m_best_cuda_device_id);
				else
					for (size_t d = 0; d < count; ++d) m_devices.push_back((int)d);
				m_handle.m_device = std::make_shared<CudaDeviceTimer>();
				m_handle.m_time = timestamp_now(); // (the start of the call is after the timer exists: the first use of CUDA is not part of the first call)
				registerDeviceTimer(m_handle.m_device);
			}

			CudaCallScope(const CudaCallScope &) = delete;
			CudaCallScope &operator=(const CudaCallScope &) = delete;

			~CudaCallScope()
			{
				auto timer = std::static_pointer_cast<CudaDeviceTimer>(m_handle.m_device);
				for (int device : m_devices)
					timer->recordEnd(device);
				tracer().call(m_handle, std::move(m_pattern), m_skeleton, std::move(m_elements), std::move(m_elwise_outputs),
					std::move(m_elwise_inputs), std::move(m_proxy_inputs), std::move(m_scalar_inputs), m_prng);
			}

		private:
			EventHandle m_handle = tracer().startEvent();
			std::string m_pattern;
			backend::SkeletonBase *m_skeleton;
			std::vector<size_t> m_elements;
			std::vector<TraceID> m_elwise_outputs, m_elwise_inputs, m_proxy_inputs, m_scalar_inputs;
			TraceID m_prng;
			std::vector<int> m_devices;
		};
	}
}

#endif
