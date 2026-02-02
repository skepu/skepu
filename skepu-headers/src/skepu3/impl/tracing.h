#ifndef TRACING_H
#define TRACING_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <stack>

#ifndef SKEPU_PRECOMPILED
#ifdef SKEPU_TRACING
#warning Tracing has no effect without precompiler processing
#undef SKEPU_TRACING
#endif
#endif

#ifdef SKEPU_TRACING
#include "../../external/nlohmann/json.hpp"
#endif

namespace skepu
{
	namespace backend
	{
		struct SkeletonBase;
	}

	namespace tracing
	{
		using TraceID = UniqueIdentifier::ID;
		static const std::string defaultTraceFilename = "skepu_trace.json";

#ifndef SKEPU_TRACE_FILE
#define SKEPU_TRACE_FILE defaultTraceFilename
#endif

//#define SKEPU_USE_STD_SOURCE_LOCATION

#ifdef SKEPU_TRACING
#define SKEPU_TRACE_START_EVENT(handle_symbol) auto handle_symbol = skepu::tracing::tracer().startEvent();
#define SKEPU_TRACE_CALL(...) skepu::tracing::tracer().call(__VA_ARGS__)
#define SKEPU_TRACE_EXTERNAL(...) skepu::tracing::tracer().external(__VA_ARGS__)
#define SKEPU_TRACE_SCOPE(label) skepu::tracing::Scope scope(label, __LINE__)
#define SKEPU_TRACE_REGION_BEGIN(label) skepu::tracing::tracer().beginRegion(label, __LINE__)
#define SKEPU_TRACE_REGION_END() skepu::tracing::tracer().endRegion()
#define SKEPU_TRACE_REGION(label, work) skepu::tracing::tracer().region(label, __LINE__, work)
#define SKEPU_TRACE_ALLOCATION(...) tracing::tracer().allocation(__VA_ARGS__)
#define SKEPU_TRACE_DEALLOCATION(...) tracing::tracer().deallocation(__VA_ARGS__)
#define SKEPU_TRACE_TRANSFER(...) tracing::tracer().transfer(__VA_ARGS__)
#else
#define SKEPU_TRACE_START_EVENT(handle_symbol)
#define SKEPU_TRACE_CALL(...)
#define SKEPU_TRACE_EXTERNAL(...)
#define SKEPU_TRACE_SCOPE(label)
#define SKEPU_TRACE_REGION_BEGIN(label)
#define SKEPU_TRACE_REGION_END()
#define SKEPU_TRACE_REGION(label, work)
#define SKEPU_TRACE_ALLOCATION(...)
#define SKEPU_TRACE_DEALLOCATION(...)
#define SKEPU_TRACE_TRANSFER(...)
#endif

		struct EventHandle
		{
			EventHandle();

#ifdef SKEPU_TRACING
			UniqueIdentifier::ID m_id;
			unsigned long m_time;
#endif
		};

		struct Scope
		{
			Scope(std::string const& label, int line);
			~Scope();
		};

		struct Tracer
		{

		public:
			Tracer(std::string filename);
			~Tracer();

			EventHandle startEvent();

			void allocation(TraceID id, std::string const& label, int line);
			void deallocation(TraceID id, std::string const& label, int line);
			void transfer(TraceID id, std::string const& label, int line, size_t elements, std::string direction, std::string backend);

			void call(
				EventHandle &h,
				std::string pattern,
				backend::SkeletonBase *skel_instance,
				std::vector<size_t> &&elements,
				std::vector<TraceID> &&elwise_outputs,
				std::vector<TraceID> &&elwise_inputs,
				std::vector<TraceID> &&proxy_inputs = {},
				std::vector<TraceID> &&scalar_inputs = {},
				TraceID prng = UniqueIdentifier::null());

			void call(
				std::string pattern,
				backend::SkeletonBase *skel_instance,
				std::vector<size_t> &&elements,
				std::vector<TraceID> &&elwise_outputs,
				std::vector<TraceID> &&elwise_inputs,
				std::vector<TraceID> &&proxy_inputs,
				std::vector<TraceID> &&scalar_inputs,
				TraceID prng,
				std::function<void(void)> &work_func);

			void external(
				EventHandle &h,
				std::string const& label,
				std::vector<TraceID> &&outputs,
				std::vector<TraceID> &&inputs
#ifdef SKEPU_USE_STD_SOURCE_LOCATION
				, const std::source_location location = std::source_location::current()
#endif
);

			void beginRegion(std::string const& label, int line);
			void endRegion();
			void region(std::string const& label, int line,
				std::function<void(void)> &&work_func);

		private:
#ifdef SKEPU_TRACING
			nlohmann::json m_trace_json;
			std::stack<std::string> m_region_stack;
			std::vector<std::function<void()>> m_captures;
#endif
			std::string m_file_name;
		};

		inline Tracer &tracer();

	}
}

#endif
