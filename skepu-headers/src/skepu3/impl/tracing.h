#ifndef TRACING_H
#define TRACING_H

#include <deque>
#include <filesystem>
#include <fstream>
#include <set>
#include <vector>
#include <string>
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
// (before C++20 the place of the call is made by the compiler: __builtin_FILE and __builtin_LINE as default arguments, as for SKEPU_CALL_SITE_PARAM)
#define SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM , const char *skepu_loc_file = __builtin_FILE(), int skepu_loc_line = __builtin_LINE()
#define SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER , const char *skepu_loc_file, int skepu_loc_line
#define SKEPU_EXTERNAL_SOURCE_LOCATION_FORWARD_ARG , skepu_loc_file, skepu_loc_line
#define SKEPU_EXTERNAL_SOURCE_LOCATION_ARGS skepu_loc_file, skepu_loc_line
#define SKEPU_EXTERNAL_SOURCE_LOCATION_FILE skepu_loc_file
#define SKEPU_EXTERNAL_SOURCE_LOCATION_LINE skepu_loc_line
#endif

#ifdef SKEPU_TRACING
#include "../../external/nlohmann/json.hpp"
#include "../../external/hash-library/sha256.hpp"
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
		// (microseconds since the first event, defined in tracing.hpp: the backends read the clock too)
		inline unsigned long timestamp_now();
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
// (what skepu-tool adds at the start of the functions of a program, see AutoRegion)
#define SKEPU_TRACE_AUTO_REGION(label, file, line) skepu::tracing::AutoRegion skepu_trace_auto_region(label, file, line)
#define SKEPU_CALL_SITE_PARAM , const char *skepu_site_file = __builtin_FILE(), int skepu_site_line = __builtin_LINE()
#define SKEPU_CALL_SITE_PARAM_ONLY const char *skepu_site_file = __builtin_FILE(), int skepu_site_line = __builtin_LINE()
// (the place that a function got, to a function of the library that it calls)
#define SKEPU_CALL_SITE_FORWARD_ARG , skepu_site_file, skepu_site_line
#define SKEPU_TRACE_API(label) skepu::tracing::ApiScope skepu_api_scope(label, skepu_site_file, skepu_site_line)
// (for a function that cannot take the place of its call: it has a pack of arguments, or is an operator; the region has no place)
#define SKEPU_TRACE_API_NO_SITE(label) skepu::tracing::ApiScope skepu_api_scope(label, nullptr, 0)
// (the place in the program where the container was declared: the allocation event is where it was made)
#define SKEPU_TRACE_PLACE(...) skepu::tracing::tracer().place(__VA_ARGS__)
#define SKEPU_TRACE_REGION_BEGIN(...) skepu::tracing::tracer().beginRegion(__VA_ARGS__)
#define SKEPU_TRACE_REGION_END() skepu::tracing::tracer().endRegion()
#define SKEPU_TRACE_REGION(...) skepu::tracing::tracer().region(__VA_ARGS__)
#define SKEPU_TRACE_ALLOCATION(...) skepu::tracing::tracer().allocation(__VA_ARGS__)
#define SKEPU_TRACE_DEALLOCATION(...) skepu::tracing::tracer().deallocation(__VA_ARGS__)
#define SKEPU_TRACE_RELABEL(...) skepu::tracing::tracer().relabel(__VA_ARGS__)
#define SKEPU_TRACE_TRANSFER(...) skepu::tracing::tracer().transfer(__VA_ARGS__)
#define SKEPU_TRACE_COMPILE(...) skepu::tracing::tracer().compile(__VA_ARGS__)
#define SKEPU_TRACE_ELEMENT_ACCESS(...) skepu::tracing::tracer().element_access(__VA_ARGS__)
#define SKEPU_TRACE_COPY(...) skepu::tracing::tracer().copy(__VA_ARGS__)
#define SKEPU_TRACE_SCALAR_IO(...) skepu::tracing::tracer().scalar_io(__VA_ARGS__)
#define SKEPU_TRACE_SCALAR_VALUE(...) skepu::tracing::tracer().scalar_value(__VA_ARGS__)
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
#define SKEPU_TRACE_AUTO_REGION(label, file, line)
#define SKEPU_CALL_SITE_PARAM
#define SKEPU_CALL_SITE_PARAM_ONLY
#define SKEPU_CALL_SITE_FORWARD_ARG
#define SKEPU_TRACE_API(label)
#define SKEPU_TRACE_API_NO_SITE(label)
#define SKEPU_TRACE_PLACE(...)
#define SKEPU_TRACE_REGION_BEGIN(label)
#define SKEPU_TRACE_REGION_END()
#define SKEPU_TRACE_REGION(...)
#define SKEPU_TRACE_ALLOCATION(...)
#define SKEPU_TRACE_DEALLOCATION(...)
#define SKEPU_TRACE_RELABEL(...)
#define SKEPU_TRACE_TRANSFER(...)
#define SKEPU_TRACE_COMPILE(...)
#define SKEPU_TRACE_ELEMENT_ACCESS(...)
#define SKEPU_TRACE_COPY(...)
#define SKEPU_TRACE_SCALAR_IO(...)
#define SKEPU_TRACE_SCALAR_VALUE(...)
#define SKEPU_TRACE_SNAPSHOT(...)
#define SKEPU_TRACE_VIRTUAL_DATASET(...)
#define SKEPU_TRACE_FLUSH()
#endif

		// Where the instance of a skeleton and its user functions are defined in the source. The precompiler gives the struct of each
		// user function of an instance a JSON text of both (the member "definition"); a skeleton says which structs are its
		// own in SkeletonBase::traceDefinition(), and the tracer writes the definition once for each instance, in the
		// section "instances" of the header of the trace. Without tracing it is empty.
		struct Definition
		{
#ifdef SKEPU_TRACING
			nlohmann::json values; // null if the user functions of the instance have no definition

			template<typename... UserFunctions>
			void describe() { (this->add<UserFunctions>(), ...); }

		private:
			template<typename UserFunction>
			static auto text(int) -> decltype(std::string(UserFunction::definition)) { return UserFunction::definition; }
			template<typename UserFunction>
			static std::string text(long) { return ""; }

			template<typename UserFunction>
			void add()
			{
				nlohmann::json definition = nlohmann::json::parse(text<UserFunction>(0), nullptr, false);
				if (!definition.is_object() || !definition.contains("function"))
					return;
				if (this->values.is_null())
					this->values = {{"instance", definition["instance"]}, {"user_functions", nlohmann::json::array()}};
				this->values["user_functions"].push_back(definition["function"]);
			}
#else
			template<typename... UserFunctions>
			void describe() {}
#endif
		};

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

#ifdef SKEPU_TRACING
		// The region of a call of a function of the program: skepu-tool puts SKEPU_TRACE_AUTO_REGION at the start of the functions that can
		// make events (they use skeletons or containers, or call such functions), so that the trace has the structure of the program without
		// regions that the programmer has marked. A region that is not needed is not in the trace: see Tracer::beginAutoRegion.
		struct AutoRegion
		{
			AutoRegion(const char *label, const char *file, int line);
			~AutoRegion();
			AutoRegion(const AutoRegion &) = delete;
			AutoRegion &operator=(const AutoRegion &) = delete;
		};

		// The region of a call of a function of the SkePU libraries (skepu-lib: dnn, blas, randomize), at the place of the call in the
		// program: the library function takes its caller's file and line as default arguments (SKEPU_CALL_SITE_PARAM), and its
		// first statement is SKEPU_TRACE_API("name"). A call of the library from another library function with the same name (an
		// overload) is not a region of its own.
		// A frontend that is not C++ (SkePY) says where the call is in its own source with a CallSite object around the call: the
		// first ApiScope that begins while it is there takes that place (not the library's callers inside the call).
		struct ApiScope
		{
			ApiScope(std::string const& label, const char *file, int line);
			~ApiScope();
			ApiScope(const ApiScope &) = delete;
			ApiScope &operator=(const ApiScope &) = delete;
		private:
			bool m_active;
		};

		struct CallSite
		{
			CallSite(const char *file, int line);
			~CallSite();
			CallSite(const CallSite &) = delete;
			CallSite &operator=(const CallSite &) = delete;
		};
#endif

		// What a recorded event is, for a tracer that the MPI backend has configured (see Tracer::configureMPI). Region: what the program
		// itself marks (regions, snapshots, virtual datasets, the elements it reads)
		enum class CaptureKind { Node, Region, Raw };

		// A field of an event, for Tracer::record: the key and a copy of the value, which is written when the event is replayed. It is made
		// by assigning to a key (Schema::Keys, as Key::LABEL = label), or to a key that is conditional (Key::ERROR.when(!ok) = error:
		// the field is left out if the condition is false).
		template<typename V>
		struct [[nodiscard]] Field { const std::string &key; V value; bool present; };

		struct ConditionalKey
		{
			const std::string &name;
			bool condition;
			template<typename V>
			Field<std::decay_t<V>> operator=(V &&value) const { return {this->name, std::forward<V>(value), this->condition}; }
		};

		// Something that is done when the event is replayed, in the order of the fields, that is not a field (then([this]{ ... }))
		template<typename F>
		struct [[nodiscard]] Action { F run; };
		template<typename F>
		Action<std::decay_t<F>> then(F &&function) { return {std::forward<F>(function)}; }

		// The name of a field of an event: a string that makes a Field when it is assigned a value
		struct KeyName : std::string
		{
			using std::string::string;
			template<typename V>
			Field<std::decay_t<V>> operator=(V &&value) const { return {*this, std::forward<V>(value), true}; }
			ConditionalKey when(bool condition) const { return {*this, condition}; }
		};

		// While a thread has one of these, the events that it records are not dropped by a tracer that is configured to
		// drop the events of the node-level library (the MPI backend records the events of a call itself)
		struct AllowScope
		{
			AllowScope();
			~AllowScope();
		};

		// The skeleton instance whose OpenCL kernels are being built, for the events of the builds (SkeletonBase::prepareOpenCL has one while
		// the kernels are built): without one a build is the library's own.
		// A build runs from the start of its program to the start of the next one, or to the end of the scope: the kernel object is made after the
		// build (on some platforms the program is not compiled before that), and the time of it is the time of the build. The events are made
		// when the scope ends (finish() does it earlier, for a build that ends the program).
		struct CompileScope
		{
			CompileScope(TraceID instance, std::string const& label)
#ifdef SKEPU_TRACING
				: m_instance{instance}, m_label{label}, m_outer{current()} { current() = this; }
			~CompileScope() { finish(); current() = m_outer; }
			static CompileScope *&current() { static thread_local CompileScope *scope = nullptr; return scope; }

			struct Build
			{
				unsigned long start;
				std::string qualifier, device, options;
				size_t source_bytes;
				bool ok;
				std::string error;
			};
			void add(Build &&build) { this->m_builds.push_back(std::move(build)); }
			void finish();

			TraceID m_instance;
			std::string m_label;
			CompileScope *m_outer;
			std::vector<Build> m_builds;
#else
				{ (void)instance; (void)label; }
#endif
		};

		struct Tracer
		{

		public:
			Tracer(std::string filename);
			~Tracer();

			EventHandle startEvent();

			void allocation(TraceID id, std::string const& label, int line);
			void deallocation(TraceID id, std::string const& label, int line);
			// A container gets its name after it was allocated (see skepu::trace_label): its events so far get it too. With `instance`, a skeleton
			// instance is named: only its events before its first call (the builds of its kernels) get it
			void relabel(TraceID id, std::string const& label, bool instance = false);
			// The place in the program where a container was declared (see skepu::trace_label): the allocation event, which has the place of the
			// header that made it, gets this place. Like relabel, it reaches only the newest events
			void place(TraceID id, std::string const& file, int line);
			// The build of an OpenCL kernel for a device: `instance` and `label` are the skeleton instance whose kernel it is (null and a name of
			// its own for a kernel of the library), `qualifier` says which one of its kernels (empty if it has one). A failed build is recorded too.
			void compile(unsigned long start, unsigned long end, TraceID instance, std::string const& label, std::string const& qualifier, std::string const& device,
				std::string const& options, size_t source_bytes, bool ok, std::string const& error);
			void transfer(EventHandle &h, TraceID id, std::string const& label, int line, size_t elements, std::string direction, std::string backend);
			void element_access(TraceID id, TraceID source_id, size_t index, std::string const& label, std::string const& file, int line);
			// The elements of one container were copied to another (copy constructor and assignment of a container): a
			// data dependence that is not a skeleton call. (A move moves the id with the data and has no event.)
			void copy(EventHandle &h, TraceID source_id, TraceID destination_id, std::string const& label, size_t elements);
			void scalar_io(std::vector<TraceID> const& ids, std::string const& label, std::string const& file, int line);
			// The value of a scalar (the result of a reduction, a norm, a container element that the program read), when it is made: a number
			// for a type of arithmetic (nothing for the others). A scalar that is made from another one has the same id, and then the later value wins.
			template<typename T>
			void scalar_value(TraceID id, T value);

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
			// A region that skepu-tool has marked (see AutoRegion): it is in the trace only if it has two events or more. The events of one that has
			// none or one are not changed, except that they are in the region around it
			void beginAutoRegion(std::string const& label, std::string const& file, int line);
			void region(std::string const& label,
				std::function<void(void)> &&work_func SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM);

		private:
			void push_capture(std::function<void()> &&capture, CaptureKind kind = CaptureKind::Node);

#ifdef SKEPU_TRACING
			// Every event is recorded with one of these. The event is made when it is replayed, in the order of the trace: its type (one of
			// Schema::Values::Types) and the region that it is in, then the fields. record_with is for an event that needs the state of the tracer
			// when it is replayed (the region stack, the header): the body gets the event after its type and region, and the event is
			// written when the body returns.
			template<typename... Fs>
			void record(CaptureKind kind, const std::string &type, Fs... fields);
			template<typename... Fs>
			void record(const std::string &type, Fs... fields) { this->record(CaptureKind::Node, type, std::move(fields)...); }
			template<typename Body>
			void record_with(CaptureKind kind, const std::string &type, Body body);
#endif

#ifdef SKEPU_TRACING
			std::mutex m_mutex; // guards m_captures
			// The events are written to the file as they are replayed. The newest ones (the window) are kept as JSON objects, as relabel can still rename
			// them; an event that leaves the window is written as one line of text, and its object is gone. The file is complete after each flush: the
			// events of the window and the end of the file (the header) are written after the committed ones, and written again by the next flush.
			static constexpr size_t kEditableEvents = 1024;      // the size of the window
			static constexpr size_t kAutoFlushCaptures = 16384;  // the captured events that make the tracer replay and write them (the closures take memory until then)
			std::deque<nlohmann::json> m_window;
			std::string m_committed_text;           // the events that left the window and are not in the file yet: a line each, with a comma at the end
			std::set<std::string> m_event_files;    // the files that the events name (collectSourceFiles)
			bool m_file_started = false;
			std::streamoff m_commit_pos = 0;        // where the committed events end in the file
			std::uintmax_t m_file_end = 0;          // the size of the file as it was written last
			std::mutex m_flush_mutex;               // one flush at a time (the program, or the tracer when its captures are many)
			void emit(nlohmann::json entry);        // an event, in the order of the trace
			void writeFile();
			nlohmann::json m_header = nlohmann::json::object();
			void collectSourceFiles(); // the files that the events name, with the hashes of their contents, to the header
			// What a call tells the header, when it is replayed: the hash of its source file (made by the precompiler: the program as it was compiled), once
			// per file, and where its instance and the user functions of it are defined, once per instance
			void noteSourceFile(std::string const& file, std::string const& hash);
			void noteInstance(TraceID id, std::string const& label);
			std::map<TraceID, nlohmann::json> m_definitions; // of the instances that have made a call (guarded by m_mutex)
			bool m_suppress_node = false, m_lean = false;
			// The events are recorded as closures that run in the order they were recorded when the
			// trace is written (see the destructor). Regions nest per thread: a closure sees the region
			// stack of the thread that recorded it.
			struct Capture { std::thread::id thread; std::function<void()> work; };
			std::vector<Capture> m_captures;
			// The regions that are open on a thread, innermost last. A region that skepu-tool has marked is not written until it has a second event
			// (a committed region); one that is not committed holds the event that it has had (if any). The ones that are not committed are the
			// innermost ones. An event is in the innermost committed region.
			struct RegionEntry
			{
				std::string label;
				bool committed;
				nlohmann::json begin;      // the event of the region, when it is not committed
				nlohmann::json buffered;   // the first event in the region, null if none (or if it is committed)
				int events = 0;
			};
			std::map<std::thread::id, std::vector<RegionEntry>> m_region_stacks;
			std::thread::id m_replaying_thread;
			std::vector<RegionEntry> &m_regions() { return this->m_region_stacks[this->m_replaying_thread]; }
			void replayBeginRegion(std::string label, std::string file, int line, unsigned long time, bool automatic);
			void replayEndRegion();
			void submit(nlohmann::json entry);   // an event in the region it is in
			// Calls visit with the events that can still be changed, the newest first, until it returns true (for the changes of relabel and place; at replay)
			template<typename Visit>
			void forRecentEvents(Visit visit);
			void commitRegions();                // writes the regions that are not, from the outermost, and the events that they hold
			void settleRegions();                // at the end of the program: the regions that are still open and hold an event

#endif
			std::string m_file_name;
		};

		inline Tracer &tracer();

	SKEPU_TRACING_VARIANT_END
	}
}

#endif
