// The times of the calls of the OpenCL backend, from the profiling of the devices. A kernel is enqueued and the host goes on: to
// find when it was done the program would have to wait for the device at every call (clFinish, clWaitForEvents), which changes the
// behaviour that the trace is to show. Instead a call enqueues a marker on the queue of each device before its work and one after
// it, which costs almost nothing while the devices run, and the times are read from the markers when the trace is written (the
// events of the trace are replayed then: see Tracer::flush). A marker is done when everything before it in the queue is done (the
// queues are in order), and its profiling gives the time of the device (in nanoseconds, on the clock of the device). That time is
// put on the clock of the host with one marker for each device that is enqueued and waited for once: the time of the host when it
// was done, plus the time between that marker and the others.
//
// The queues have the profiling enabled when the program is traced (see Device_CL).
#pragma once

#if defined(SKEPU_TRACING) && defined(SKEPU_OPENCL)

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

#include "environment.h"
#include "../impl/tracing.h"

namespace skepu
{
	namespace tracing
	{
		class OpenClDeviceTimer: public DeviceTimer
		{
		public:
			// Starts the timing: a marker on the queue of each device (the host time is taken by the caller, after this)
			OpenClDeviceTimer()
			{
				for (backend::Device_CL *device : backend::Environment<int>::getInstance()->m_devices_CL)
				{
					anchor(device);
					m_starts.emplace_back(device, marker(device));
					m_devices.push_back(device);
				}
			}

			// The work is done on a device when everything before the marker has been done
			void recordEnd()
			{
				std::lock_guard<std::mutex> lock(m_events_mutex);
				for (backend::Device_CL *device : m_devices)
					m_ends.emplace_back(device, marker(device));
			}

			~OpenClDeviceTimer() override
			{
				std::lock_guard<std::mutex> lock(m_events_mutex);
				releaseLocked();
			}

		protected:
			// (not before the end of the call has been recorded)
			bool ready() override
			{
				std::lock_guard<std::mutex> lock(m_events_mutex);
				if (m_ends.empty())
					return false;
				for (auto &end : m_ends)
				{
					cl_int status = CL_COMPLETE;
					if (end.second && clGetEventInfo(end.second, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, nullptr) == CL_SUCCESS
						&& status > CL_COMPLETE)
						return false;
				}
				return true;
			}

			// (this waits for the devices if their work is not done: only when the trace is written, or when the host is far ahead of the devices)
			void read() override
			{
				std::lock_guard<std::mutex> lock(m_events_mutex);
				unsigned long earliest = 0;
				for (auto &start : m_starts)
				{
					const unsigned long t = hostTimeOf(start.first, start.second);
					if (t && (!earliest || t < earliest))
						earliest = t;
				}
				m_start = earliest;
				for (auto &end : m_ends)
					m_end = std::max(m_end, hostTimeOf(end.first, end.second));
				releaseLocked();
			}

		private:
			struct Anchor { cl_ulong deviceTime = 0; unsigned long hostTime = 0; bool valid = false; };

			void releaseLocked()
			{
				for (auto &e : m_starts) if (e.second) clReleaseEvent(e.second);
				for (auto &e : m_ends) if (e.second) clReleaseEvent(e.second);
				m_starts.clear();
				m_ends.clear();
			}

			static cl_event marker(backend::Device_CL *device)
			{
				cl_event event = nullptr;
				if (clEnqueueMarkerWithWaitList(device->getQueue(), 0, nullptr, &event) != CL_SUCCESS)
					return nullptr;
				return event;
			}

			// The reference of a device: the time of a marker of it that has happened, and the time of the host just after
			static const Anchor &anchor(backend::Device_CL *device)
			{
				// (not destroyed: the trace is written when the program exits, after the statics of this function)
				static std::mutex &mutex = *new std::mutex;
				static std::map<cl_device_id, Anchor> &anchors = *new std::map<cl_device_id, Anchor>;
				std::lock_guard<std::mutex> lock(mutex);
				Anchor &a = anchors[device->getDeviceID()];
				if (!a.valid)
				{
					if (cl_event event = marker(device))
					{
						clWaitForEvents(1, &event);
						if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &a.deviceTime, nullptr) == CL_SUCCESS)
						{
							a.hostTime = timestamp_now();
							a.valid = true;
						}
						clReleaseEvent(event);
					}
				}
				return a;
			}

			// The profiling times are in nanoseconds, except on macOS, where they are in ticks of the system clock (mach_absolute_time)
			static double nanosecondsPerTick()
			{
#ifdef __APPLE__
				static const double ratio = []{ mach_timebase_info_data_t info; mach_timebase_info(&info); return (double)info.numer / info.denom; }();
				return ratio;
#else
				return 1.0;
#endif
			}

			// The time of the host (microseconds) when a marker was done, or 0 if the device cannot say
			static unsigned long hostTimeOf(backend::Device_CL *device, cl_event event)
			{
				const Anchor &a = anchor(device);
				cl_ulong time = 0;
				if (!a.valid || !event || clWaitForEvents(1, &event) != CL_SUCCESS
					|| clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time, nullptr) != CL_SUCCESS)
					return 0;
				const long long microseconds = (long long)(((double)time - (double)a.deviceTime) * nanosecondsPerTick() / 1000.0);
				return (unsigned long)std::max(1LL, (long long)a.hostTime + microseconds);
			}

			std::vector<backend::Device_CL*> m_devices;
			std::vector<std::pair<backend::Device_CL*, cl_event>> m_starts, m_ends;
			std::mutex m_events_mutex;
		};
	}
}

#endif
