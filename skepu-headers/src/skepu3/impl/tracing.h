#ifndef TRACING_H
#define TRACING_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <stack>
#include <mutex>
#include <atomic>

#ifndef SKEPU_PRECOMPILED
#ifdef SKEPU_TRACING
#warning Tracing has no effect without precompiler processing
#undef SKEPU_TRACING
#endif
#endif

// The layout and the function bodies of the tracer depend on SKEPU_TRACING. Translation units
// compiled with and without it can end up in one program (a precompiled source file and a plain
// one that includes <skepu>, for example). The two variants are therefore different types, in
// different inline namespaces: otherwise the linker merges the inline functions of the two
// variants, and an object of one is constructed or destroyed by the code of the other.
#ifdef SKEPU_TRACING
#define SKEPU_TRACING_VARIANT_BEGIN inline namespace enabled {
#else
#define SKEPU_TRACING_VARIANT_BEGIN inline namespace disabled {
#endif
#define SKEPU_TRACING_VARIANT_END }

#if __cplusplus >= 202002L
#include <source_location>
#define SKEPU_USE_STD_SOURCE_LOCATION
#define SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM , const std::source_location loc = std::source_location::current()
#define SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER , const std::source_location loc
#define SKEPU_EXTERNAL_SOURCE_LOCATION_FORWARD_ARG , loc
#define SKEPU_EXTERNAL_SOURCE_LOCATION_ARGS loc.file_name(), loc.line()
#define SKEPU_EXTERNAL_SOURCE_LOCATION_FILE loc.file_name()
#define SKEPU_EXTERNAL_SOURCE_LOCATION_LINE loc.line()
#else
#define SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM
#define SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER
#define SKEPU_EXTERNAL_SOURCE_LOCATION_FORWARD_ARG
#define SKEPU_EXTERNAL_SOURCE_LOCATION_ARGS "", 0
#define SKEPU_EXTERNAL_SOURCE_LOCATION_FILE ""
#define SKEPU_EXTERNAL_SOURCE_LOCATION_LINE 0
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
	SKEPU_TRACING_VARIANT_BEGIN
		using TraceID = UniqueIdentifier::ID;
		static const std::string defaultTraceFilename = "skepu_trace.json";
		static const std::string defaultSnapshotDir = "./";

#ifndef SKEPU_TRACE_FILE
#define SKEPU_TRACE_FILE_NAME defaultTraceFilename
#else
#define SKEPU_TRACE_FILE_NAME SKEPU_STRINGIFY_NESTED(SKEPU_TRACE_FILE)
#endif

#ifndef SKEPU_TRACE_SNAPSHOT_DIR
#define SKEPU_TRACE_SNAPSHOT_DIR_NAME defaultSnapshotDir
#else
#define SKEPU_TRACE_SNAPSHOT_DIR_NAME SKEPU_STRINGIFY_NESTED(SKEPU_TRACE_SNAPSHOT_DIR)
#endif

// Synchronous OpenCL kernels when tracing is enabled
#ifdef SKEPU_TRACING
	#define SKEPU_TRACING_WAIT_CL(event) clWaitForEvents(1, &event);
#else
    #define SKEPU_TRACING_WAIT_CL(event)
#endif

#ifdef SKEPU_TRACING
#define SKEPU_TRACE_START_EVENT(handle_symbol) auto handle_symbol = skepu::tracing::tracer().startEvent();
#define SKEPU_TRACE_CALL(...) skepu::tracing::tracer().call(__VA_ARGS__)
#define SKEPU_TRACE_EXTERNAL(...) skepu::tracing::tracer().external(__VA_ARGS__)
#define SKEPU_TRACE_SCOPE(label) skepu::tracing::Scope scope(label)
#define SKEPU_TRACE_REGION_BEGIN(...) skepu::tracing::tracer().beginRegion(__VA_ARGS__)
#define SKEPU_TRACE_REGION_END() skepu::tracing::tracer().endRegion()
#define SKEPU_TRACE_REGION(...) skepu::tracing::tracer().region(__VA_ARGS__)
#define SKEPU_TRACE_ALLOCATION(...) skepu::tracing::tracer().allocation(__VA_ARGS__)
#define SKEPU_TRACE_DEALLOCATION(...) skepu::tracing::tracer().deallocation(__VA_ARGS__)
#define SKEPU_TRACE_TRANSFER(...) skepu::tracing::tracer().transfer(__VA_ARGS__)
#define SKEPU_TRACE_ELEMENT_ACCESS(...) skepu::tracing::tracer().element_access(__VA_ARGS__)
#define SKEPU_TRACE_SCALAR_IO(...) skepu::tracing::tracer().scalar_io(__VA_ARGS__)
#define SKEPU_TRACE_SNAPSHOT(...) skepu::tracing::tracer().data_snapshot(__VA_ARGS__)
#define SKEPU_TRACE_VIRTUAL_DATASET(...) skepu::tracing::tracer().virtual_dataset(__VA_ARGS__)
#else
#define SKEPU_TRACE_START_EVENT(handle_symbol)
#define SKEPU_TRACE_CALL(...)
#define SKEPU_TRACE_EXTERNAL(...)
#define SKEPU_TRACE_SCOPE(label)
#define SKEPU_TRACE_REGION_BEGIN(label)
#define SKEPU_TRACE_REGION_END()
#define SKEPU_TRACE_REGION(...)
#define SKEPU_TRACE_ALLOCATION(...)
#define SKEPU_TRACE_DEALLOCATION(...)
#define SKEPU_TRACE_TRANSFER(...)
#define SKEPU_TRACE_ELEMENT_ACCESS(...)
#define SKEPU_TRACE_SCALAR_IO(...)
#define SKEPU_TRACE_SNAPSHOT(...)
#define SKEPU_TRACE_VIRTUAL_DATASET(...)
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
			Scope(std::string const& label SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM);
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
			void transfer(EventHandle &h, TraceID id, std::string const& label, int line, size_t elements, std::string direction, std::string backend);
			void element_access(TraceID id, TraceID source_id, size_t index, std::string const& label, std::string const& file, int line);
			void scalar_io(std::vector<TraceID> const& ids, std::string const& label, std::string const& file, int line);

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
				std::string const& file,
				int line,
				std::vector<TraceID> &&outputs,
				std::vector<TraceID> &&inputs);

			template<typename DataContainer>
			void data_snapshot(
			    DataContainer &data
				SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM
			);

			template<typename DataContainer>
			void data_snapshot(DataContainer &data, std::string const& file, int line);

			template<template<typename> typename C1, template<typename> typename C2, typename T1, typename T2>
			void virtual_dataset(
			    std::string const& label,
			    C1<T1> &data1, C2<T2> &data2
				SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM
			);

			template<template<typename> typename C1, template<typename> typename C2, typename T1, typename T2>
			void virtual_dataset(
			    std::string const& label,
			    C1<T1> &data1, C2<T2> &data2,
			    std::string const& file, int line
			);

			void beginRegion(std::string const& label SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM);
			// (with an explicit place, for programs that are not C++: SkePY says where in its source)
			void beginRegion(std::string const& label, std::string const& file, int line);
			void endRegion();
			void region(std::string const& label,
				std::function<void(void)> &&work_func SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM);

		private:
			void push_capture(std::function<void()> &&capture);

#ifdef SKEPU_TRACING
			std::mutex m_mutex; // guards m_captures
			nlohmann::json m_trace_json;
			std::stack<std::string> m_region_stack;
			std::vector<std::function<void()>> m_captures;
#endif
			std::string m_file_name;
		};

		inline Tracer &tracer();

	SKEPU_TRACING_VARIANT_END
	}
}

#endif
