#ifndef TRACING_H
#define TRACING_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <stack>
#include <mutex>
#include <thread>
#include <map>
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
// (for the calls that do not have an event to wait for)
#if defined(SKEPU_TRACING) && defined(SKEPU_OPENCL)
	#define SKEPU_TRACING_FINISH_CL() skepu::backend::finishDevicesCL();
#else
	#define SKEPU_TRACING_FINISH_CL()
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
#define SKEPU_TRACE_COPY(...) skepu::tracing::tracer().copy(__VA_ARGS__)
#define SKEPU_TRACE_SCALAR_IO(...) skepu::tracing::tracer().scalar_io(__VA_ARGS__)
// (a snapshot writes the contents of a container to a file each time: it is taken only if SKEPU_TRACING_SNAPSHOTS is defined too)
#ifdef SKEPU_TRACING_SNAPSHOTS
#define SKEPU_TRACE_SNAPSHOT(...) skepu::tracing::tracer().data_snapshot(__VA_ARGS__)
#else
#define SKEPU_TRACE_SNAPSHOT(...)
#endif
#define SKEPU_TRACE_VIRTUAL_DATASET(...) skepu::tracing::tracer().virtual_dataset(__VA_ARGS__)
#define SKEPU_TRACE_FLUSH() skepu::tracing::tracer().flush()
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
#define SKEPU_TRACE_COPY(...)
#define SKEPU_TRACE_SCALAR_IO(...)
#define SKEPU_TRACE_SNAPSHOT(...)
#define SKEPU_TRACE_VIRTUAL_DATASET(...)
#define SKEPU_TRACE_FLUSH()
#endif

		// The parameters of a skeleton that change what its calls do (the overlap and strides of a MapOverlap,
		// the mode of a Scan, ...). A skeleton describes them in SkeletonBase::traceParameters(), and the
		// tracer records them with every call, as they are when the call is made. Without tracing it is empty.
		struct Parameters
		{
#ifdef SKEPU_TRACING
			nlohmann::json values = nlohmann::json::object();

			// a number, a boolean or a text
			template<typename T>
			void add(std::string const& name, T const& value) { this->values[name] = value; }

			// one value for each dimension
			template<typename T>
			void add(std::string const& name, T const *first, size_t count) { this->values[name] = std::vector<T>(first, first + count); }

			// a value of a type that is a number, if it is and it is not zero (the start value of a
			// reduction can be a struct, and zero is what it is unless it is set)
			template<typename T>
			void addNumber(std::string const& name, T const& value)
			{
				if constexpr (std::is_arithmetic<T>::value)
					if (value != T{})
						this->values[name] = value;
			}
#else
			template<typename... Args>
			void add(Args&&...) {}

			template<typename... Args>
			void addNumber(Args&&...) {}
#endif
		};

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

		// What a recorded event is, for a tracer that the MPI backend has configured (see Tracer::configureMPI). Region: what the program
		// itself marks (regions, snapshots, virtual datasets, the elements it reads)
		enum class CaptureKind { Node, Region, Raw };

		// While a thread has one of these, the events that it records are not dropped by a tracer that is configured to
		// drop the events of the node-level library (the MPI backend records the events of a call itself)
		struct AllowScope
		{
			AllowScope();
			~AllowScope();
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
			// The elements of one container were copied to another (copy constructor and assignment of a container): a
			// data dependence that is not a skeleton call. (A move moves the id with the data and has no event.)
			void copy(EventHandle &h, TraceID source_id, TraceID destination_id, std::string const& label, size_t elements);
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
				TraceID prng = UniqueIdentifier::null(),
				Parameters const *extra = nullptr); // (fields that are added to the event as they are)

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

#ifdef SKEPU_TRACING
			// The MPI backend (skepu3/mpi/trace.hpp). Every process has a tracer. The events of the node-level library (the
			// skeletons and containers of a process work on parts of the containers) are dropped: the backend records the
			// events of the program itself with raw(). On a process that is not the first (lean), everything but raw
			// events is dropped, as the first process records the events of the program that all processes agree on.
			void configureMPI(bool lean) { this->m_suppress_node = true; this->m_lean = lean; }
			void setFileName(std::string const& name) { this->m_file_name = name; }
			std::string const &fileName() const { return this->m_file_name; }
			// An event as it is (it also has the region that it was recorded in, unless lean)
			void raw(nlohmann::json entry);
			// What the file starts with (the trace is then an object { header, data })
			nlohmann::json &header() { return this->m_header; }
#endif

			// Writes the events recorded so far to the trace file (the trace is also written when the
			// tracer is destroyed, at the end of the program; call this to have it earlier, for
			// example from a program that does not end normally or is an extension module of another)
			void flush();

			void beginRegion(std::string const& label SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM);
			// (with an explicit place, for programs that are not C++: SkePY says where in its source)
			void beginRegion(std::string const& label, std::string const& file, int line);
			void endRegion();
			void region(std::string const& label,
				std::function<void(void)> &&work_func SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM);

		private:
			void push_capture(std::function<void()> &&capture, CaptureKind kind = CaptureKind::Node);

#ifdef SKEPU_TRACING
			std::mutex m_mutex; // guards m_captures
			nlohmann::json m_trace_json;
			nlohmann::json m_header = nlohmann::json::object();
			bool m_suppress_node = false, m_lean = false;
			// The events are recorded as closures that run in the order they were recorded when the
			// trace is written (see the destructor). Regions nest per thread: a closure sees the region
			// stack of the thread that recorded it.
			struct Capture { std::thread::id thread; std::function<void()> work; };
			std::vector<Capture> m_captures;
			std::map<std::thread::id, std::stack<std::string>> m_region_stacks;
			std::thread::id m_replaying_thread;
			std::stack<std::string> &m_regions() { return this->m_region_stacks[this->m_replaying_thread]; }
#endif
			std::string m_file_name;
		};

		inline Tracer &tracer();

	SKEPU_TRACING_VARIANT_END
	}
}

#endif
