#ifndef TRACING_HPP
#define TRACING_HPP


#include "tracing.h"
namespace skepu
{
	namespace tracing
	{
	SKEPU_TRACING_VARIANT_BEGIN
		template<size_t... Indices, typename First, typename... Rest>
		std::vector<TraceID> get_all_addresses_tuple_helper(pack_indices<Indices...>, std::tuple<First, Rest...> &tuple)
		{
			return std::vector<TraceID>{ std::get<Indices>(tuple).getObjectID()... };
		}

		template<typename First, typename... Rest>
		std::vector<TraceID> get_all_addresses_tuple(std::tuple<First, Rest...> &tuple)
		{
			auto indices = typename make_pack_indices<std::tuple_size<std::tuple<First, Rest...>>::value, 1>::type{};
			return get_all_addresses_tuple_helper(indices, tuple);
		}

		inline void scalar_labels_helper(std::vector<TraceID> &labels) {}

		template<typename First, typename... Rest>
		void scalar_labels_helper(std::vector<TraceID> &labels, First, Rest... rest)
		{
			scalar_labels_helper(labels, rest...);
		}

		template<typename First, typename... Rest>
		void scalar_labels_helper(std::vector<TraceID> &labels, Scalar<First> first, Rest... rest)
		{
			for (auto label : first.m_ids)
			{
				labels.push_back(label);
			}
			scalar_labels_helper(labels, rest...);
		}

		template<typename... Args>
		std::vector<TraceID> scalar_labels(Args... args)
		{
			std::vector<TraceID> labels{};
			scalar_labels_helper(labels, args...);
			return labels;
		}
		
		template<size_t... Indices, typename... Args>
		void scalar_output_labels_helper(std::vector<TraceID> &labels, future_std::index_sequence<Indices...>, skepu::multiple<Args...> &margs)
		{
			pack_expand((scalar_labels_helper(labels, std::get<Indices>(margs)), 0)...);
		}
		
		template<typename... Args>
		std::vector<TraceID> scalar_output_labels(skepu::multiple<Args...> &margs)
		{
			std::vector<TraceID> labels{};
			scalar_output_labels_helper(labels, typename future_std::make_index_sequence<sizeof...(Args)>::type{}, margs);
			return labels;
		}
		
		template<typename T>
		std::vector<TraceID> scalar_output_labels(skepu::Scalar<T> &arg)
		{
			std::vector<TraceID> labels{};
			scalar_labels_helper(labels, arg);
			return labels;
		}

		inline std::vector<TraceID> uniform_labels()
		{
			return std::vector<TraceID>{};
		}

		// (inline: one origin of time for all translation units of a program)
		inline unsigned long timestamp_now()
		{
			static unsigned long start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
			return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - start_time;
		}



		namespace Schema
		{
			namespace Keys
			{
				static const KeyName OBJECT_ID{"object_id"};
				static const KeyName OBJECT_IDS{"object_ids"};
				static const KeyName VIRTUAL_ID{"virtual_id"};
				static const KeyName TYPE{"type"};
				static const KeyName LABEL{"label"};
				static const KeyName TIME{"time"};
				static const KeyName FILE{"file"};
				static const KeyName LINE{"line"};
				static const KeyName REGION{"region"};
				static const KeyName START{"start"};
				static const KeyName DEPTH{"region_depth"};
				static const KeyName END{"end"};
				static const KeyName ELEMENTS{"elements"};
				static const KeyName ELWISE_INPUTS{"elwise_inputs"};
				static const KeyName PROXY_INPUTS{"proxy_inputs"};
				static const KeyName SCALAR_INPUTS{"scalar_inputs"};
				static const KeyName OUTPUTS{"outputs"};
				static const KeyName PATTERN{"pattern"};
				static const KeyName BACKEND{"backend"};
				static const KeyName THREADS{"threads"};
				static const KeyName DEVICES{"devices"};
				static const KeyName PARAMETERS{"parameters"};
				static const KeyName DIRECTION{"direction"};
				static const KeyName PRNG{"prng"};
				static const KeyName SOURCE{"source"};
				static const KeyName INDEX{"index"};
				static const KeyName DATA_FILE{"data_file"};
				static const KeyName VALUE{"value"};
				static const KeyName QUALIFIER{"qualifier"};
				static const KeyName DEVICE{"device"};
				static const KeyName OPTIONS{"options"};
				static const KeyName SOURCE_BYTES{"source_bytes"};
				static const KeyName OK{"ok"};
				static const KeyName ERROR{"error"};
			}

			namespace Values
			{
				namespace Types
				{
					static const std::string CALL{"skeleton_call"};
					static const std::string ALLOCATION{"allocation"};
					static const std::string DEALLOCATION{"deallocation"};
					static const std::string ELEM_ACCESS{"element_access"};
					static const std::string COPY{"copy"};
					static const std::string TRANSFER{"transfer"};
					static const std::string REGION{"region"};
					static const std::string EXTERNAL{"external"};
					static const std::string SNAPSHOT{"snapshot"};
					static const std::string SCALAR_VALUE{"scalar_value"};
					static const std::string VIRTUAL_DATASET{"virtual_container"};
					static const std::string COMPILE{"kernel_compile"};
				}
			}
		}

		namespace Key = Schema::Keys;
		namespace Type = Schema::Values::Types;

		inline EventHandle::EventHandle()
#ifdef SKEPU_TRACING
		: m_id{UniqueIdentifier::generate()}, m_time{timestamp_now()}
#endif
		{}

		inline Tracer::Tracer(std::string filename): m_file_name{filename}
		{
			const bool defaultName = (this->m_file_name == defaultTraceFilename);
		    if (defaultName && std::getenv("SKEPU_TRACE_FILE"))
			{
				this->m_file_name = std::getenv("SKEPU_TRACE_FILE");
			}

			// Each module of a program that has several (SkePY extension modules, which are built with
			// SKEPU_TRACE_MODULE=name) has a tracer of its own: they write to files of their own, or
			// to the file given by SKEPU_TRACE_FILE with %m replaced by the name of the module
#ifdef SKEPU_TRACE_MODULE
			const std::string module = SKEPU_STRINGIFY_NESTED(SKEPU_TRACE_MODULE);
			if (defaultName && !std::getenv("SKEPU_TRACE_FILE"))
				this->m_file_name = "skepu_trace_" + module + ".json";
#else
			const std::string module = "main";
#endif
			for (size_t at = this->m_file_name.find("%m"); at != std::string::npos; at = this->m_file_name.find("%m", at + module.size()))
				this->m_file_name.replace(at, 2, module);
        }

		inline Tracer::~Tracer()
		{
			this->flush();
#ifdef SKEPU_TRACING
			// (a region of the program that did not end, and has an event that waits)
			this->settleRegions();
			this->writeFile();
#endif
		}

		// The files that the events name, with the hash of the contents of each (SHA-256 of the bytes), in header.files: what a viewer that reads
		// the sources from somewhere else (an editor) needs to know that it has the same version. A call of a skeleton has its hash from the
		// precompiler already; the other events (external, regions, snapshots) are hashed here, from the file as it is when the trace is written.
#ifdef SKEPU_TRACING
		inline void Tracer::collectSourceFiles()
		{
			for (const std::string &name : this->m_event_files)
			{
				if (this->m_header["files"].contains(name))
					continue;
				std::ifstream in(name, std::ios::binary);
				std::ostringstream contents;
				if (!in || !(contents << in.rdbuf()))
					continue; // not readable from here: no hash
				SHA256 sha256;
				this->m_header["files"][name] = sha256(contents.str());
			}
		}
#endif

#ifdef SKEPU_TRACING
		inline void Tracer::noteSourceFile(std::string const& file, std::string const& hash)
		{
			if (!file.empty() && !hash.empty() && !this->m_header["files"].contains(file))
				this->m_header["files"][file] = hash;
		}

		inline void Tracer::noteInstance(TraceID id, std::string const& label)
		{
			const std::string key = std::to_string(id);
			if (this->m_header["instances"].contains(key))
				return;
			nlohmann::json definition;
			{
				std::lock_guard<std::mutex> lock(this->m_mutex);   // (the definitions are made when calls are recorded)
				definition = this->m_definitions[id];
			}
			if (definition.is_null())
				return;
			definition[Schema::Keys::LABEL] = label;
			this->m_header["instances"][key] = std::move(definition);
		}
#endif

		inline void Tracer::flush()
		{
#ifdef SKEPU_TRACING
			std::lock_guard<std::mutex> flushing(this->m_flush_mutex);
			std::vector<Capture> captures;
			{
				std::lock_guard<std::mutex> lock(this->m_mutex);
				captures.swap(this->m_captures);
			}
			for (auto& capture : captures)
			{
				this->m_replaying_thread = capture.thread;
				capture.work();
			}

			this->collectSourceFiles();
			this->writeFile();
#endif
		}

#ifdef SKEPU_TRACING
		// An event in the order of the trace: it stays an object while it is in the window, where relabel can rename it, and is text after that.
		inline void Tracer::emit(nlohmann::json entry)
		{
			auto file = entry.find(Schema::Keys::FILE);
			if (file != entry.end() && file->is_string() && !file->get<std::string>().empty())
				this->m_event_files.insert(file->get<std::string>());
			this->m_window.push_back(std::move(entry));
			if (this->m_window.size() > kEditableEvents)
			{
				this->m_committed_text += this->m_window.front().dump();
				this->m_committed_text += ",\n";
				this->m_window.pop_front();
			}
		}

		// The file is { "data": [ one event on each line ], "header": { ... } }: the header is last, as it is complete only at the end of the run. The
		// committed events are written where the last write left them; after them the events of the window, and the end of the file, which the next
		// write replaces (a write costs the new events, the window and the header, not the whole trace). A reader reads this as any { header, data }.
		inline void Tracer::writeFile()
		{
			if (!this->m_file_started)
			{
				std::ofstream start(this->m_file_name, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
				start << "{\"data\":[\n";
				this->m_commit_pos = start.tellp();
				this->m_file_started = true;
			}
			std::fstream file(this->m_file_name, std::ios::in | std::ios::out | std::ios::binary);
			if (!file)
				return;
			file.seekp(this->m_commit_pos);
			file << this->m_committed_text;
			this->m_commit_pos = file.tellp();
			this->m_committed_text.clear();
			size_t left = this->m_window.size();
			for (const nlohmann::json &entry : this->m_window)
				file << entry.dump() << (--left ? ",\n" : "\n");
			file << "],\"header\":" << this->m_header.dump() << "}\n";
			const std::uintmax_t end = static_cast<std::uintmax_t>(file.tellp());
			file.close();
			if (end < this->m_file_end)
			{
				std::error_code ignored;
				std::filesystem::resize_file(this->m_file_name, end, ignored);   // (what the last write had after this one's end)
			}
			this->m_file_end = end;
		}
#endif


		inline void Tracer::allocation(TraceID id, std::string const& label, int line)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			this->record(Type::ALLOCATION,
				Key::OBJECT_ID = id,
				Key::LABEL = label,
				Key::TIME = now,
				Key::LINE = line);
#endif
		}

		inline void Tracer::relabel(TraceID id, std::string const& label, bool instance)
		{
#ifdef SKEPU_TRACING
			// (the events are written in order, and a container is named right after its declaration: the
			// events of the container are the last ones, which are still in the window of the events that can be changed (kEditableEvents); a
			// container that is named later than that keeps the name that its older events have; a Raw capture is also kept in the traces of the other processes of an MPI program.
			// A skeleton instance can be named again and again, for example by the layers of a network that share one: only the events
			// before its first call, the builds of its kernels, get the name; its calls keep the name that it had at the call)
			auto capture = [=, this]{
			this->forRecentEvents([&](nlohmann::json &event)
			{
				auto object = event.find(Schema::Keys::OBJECT_ID);
				if (object == event.end() || *object != id || !event.contains(Schema::Keys::LABEL))
					return false;
				if (instance && event[Schema::Keys::TYPE] == Schema::Values::Types::CALL)
					return true;
				event[Schema::Keys::LABEL] = label;
				return event[Schema::Keys::TYPE] == Schema::Values::Types::ALLOCATION;
			});
			}; this->push_capture(std::move(capture), CaptureKind::Raw);
#endif
		}

		inline void Tracer::place(TraceID id, std::string const& file, int line)
		{
#ifdef SKEPU_TRACING
			auto capture = [=, this]{
			this->forRecentEvents([&](nlohmann::json &event)
			{
				auto object = event.find(Schema::Keys::OBJECT_ID);
				if (object == event.end() || *object != id || event[Schema::Keys::TYPE] != Schema::Values::Types::ALLOCATION)
					return false;
				event[Schema::Keys::FILE] = file;
				event[Schema::Keys::LINE] = line;
				return true;
			});
			}; this->push_capture(std::move(capture), CaptureKind::Raw);
#endif
		}

#ifdef SKEPU_TRACING
		template<typename Visit>
		inline void Tracer::forRecentEvents(Visit visit)
		{
			// (the events that wait in the regions that are not written are the newest)
			for (auto &stack : this->m_region_stacks)
				for (auto entry = stack.second.rbegin(); entry != stack.second.rend(); ++entry)
					if (!entry->buffered.is_null() && visit(entry->buffered))
						return;
			for (auto it = this->m_window.rbegin(); it != this->m_window.rend(); ++it)
				if (visit(*it))
					return;
		}
#endif

		inline void Tracer::compile(
			unsigned long start,
			unsigned long end,
			TraceID instance,
			std::string const& label,
			std::string const& qualifier,
			std::string const& device,
			std::string const& options,
			size_t source_bytes,
			bool ok,
			std::string const& error)
		{
#ifdef SKEPU_TRACING
			this->record(Type::COMPILE,
				Key::OBJECT_ID.when(instance != UniqueIdentifier::null()) = instance,
				Key::LABEL = label,
				Key::QUALIFIER.when(!qualifier.empty()) = qualifier,
				Key::BACKEND = std::string("OpenCL"),
				Key::DEVICE = device,
				Key::OPTIONS.when(!options.empty()) = options,
				Key::SOURCE_BYTES = source_bytes,
				Key::OK = ok,
				Key::ERROR.when(!ok) = error,
				Key::START = start,
				Key::END = end,
				Key::TIME = end);
#endif
		}

#ifdef SKEPU_TRACING
		inline void CompileScope::finish()
		{
			for (size_t i = 0; i < this->m_builds.size(); ++i)
			{
				Build const& b = this->m_builds[i];
				const unsigned long end = i + 1 < this->m_builds.size() ? this->m_builds[i + 1].start : timestamp_now();
				tracer().compile(b.start, end, this->m_instance, this->m_label, b.qualifier, b.device, b.options, b.source_bytes, b.ok, b.error);
			}
			this->m_builds.clear();
		}
#endif

		inline void Tracer::deallocation(TraceID id, std::string const& label, int line)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			this->record(Type::DEALLOCATION,
				Key::OBJECT_ID = id,
				Key::LABEL = label,
				Key::TIME = now,
				Key::LINE = line);
#endif
		}

		inline void Tracer::element_access(TraceID id, TraceID source_id, size_t index, std::string const& label, std::string const& file, int line)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			this->record(CaptureKind::Region, Type::ELEM_ACCESS,
				Key::OBJECT_ID = id,
				Key::SOURCE = source_id,
				Key::INDEX = index,
				Key::LABEL = label,
				Key::TIME = now,
				Key::FILE = file,
				Key::LINE = line);
#endif
		}

#ifdef SKEPU_TRACING
		// A number for the trace: floating point numbers that are not finite are strings (JSON has no way to write them), a float is the shortest
		// decimal number that reads back as it
		template<typename T>
		inline nlohmann::json scalar_value_json(T value)
		{
			if constexpr (std::is_floating_point<T>::value)
			{
				if (std::isnan(value))
					return "nan";
				if (std::isinf(value))
					return value > 0 ? "inf" : "-inf";
				if constexpr (std::is_same<T, float>::value)
				{
					char text[32];
					for (int digits = 6; digits <= 9; ++digits)
					{
						std::snprintf(text, sizeof text, "%.*g", digits, (double)value);
						if (std::strtof(text, nullptr) == value)
							break;
					}
					return std::strtod(text, nullptr);
				}
				else
					return (double)value;
			}
			else if constexpr (std::is_signed<T>::value)
				return (long long)value;
			else
				return (unsigned long long)value;
		}
#endif

		template<typename T>
		inline void Tracer::scalar_value(TraceID id, T value)
		{
#ifdef SKEPU_TRACING
			if constexpr (std::is_arithmetic<T>::value)
			{
				unsigned long now = timestamp_now();
				nlohmann::json number = scalar_value_json(value);

			this->record(CaptureKind::Region, Type::SCALAR_VALUE,
				Key::OBJECT_ID = id,
				Key::VALUE = number,
				Key::TIME = now);
			}
#endif
		}

		inline void Tracer::copy(EventHandle &h, TraceID source_id, TraceID destination_id, std::string const& label, size_t elements)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			this->record(Type::COPY,
				Key::SOURCE = source_id,
				Key::OBJECT_ID = destination_id,
				Key::LABEL = label,
				Key::ELEMENTS = elements,
				Key::START = h.m_time,
				Key::END = now);
#endif
		}

		inline void Tracer::transfer(
		    EventHandle &h,
			TraceID id,
			std::string const& label,
			int line,
			size_t elements,
			std::string direction,
			std::string backend)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			this->record(Type::TRANSFER,
				Key::OBJECT_ID = id,
				Key::ELEMENTS = elements,
				Key::DIRECTION = direction,
				Key::BACKEND = backend,
				Key::LABEL = label,
				Key::START = h.m_time,
				Key::END = now,
				Key::TIME = now,
				Key::LINE = line);
#endif
		}

		// (the events come from any thread that runs a skeleton or touches a container)
		inline int &allow_depth()
		{
			static thread_local int depth = 0;
			return depth;
		}

		inline AllowScope::AllowScope()
		{
#ifdef SKEPU_TRACING
			++allow_depth();
#endif
		}

		inline AllowScope::~AllowScope()
		{
#ifdef SKEPU_TRACING
			--allow_depth();
#endif
		}

#ifdef SKEPU_TRACING
		template<typename V>
		inline void write_part(nlohmann::json &entry, const Field<V> &field)
		{
			if (field.present)
				entry[field.key] = field.value;
		}

		// The fields of a JSON object, for the ones that an event gets from outside (the parameters of a skeleton): merge(object)
		struct Merge { nlohmann::json fields; };
		inline Merge merge(nlohmann::json fields) { return {std::move(fields)}; }

		inline void write_part(nlohmann::json &entry, const Merge &merge)
		{
			for (auto it = merge.fields.begin(); it != merge.fields.end(); ++it)
				entry[it.key()] = it.value();
		}

		template<typename F>
		inline void write_part(nlohmann::json &, const Action<F> &action)
		{
			action.run();
		}

		template<typename Body>
		inline void Tracer::record_with(CaptureKind kind, const std::string &type, Body body)
		{
			// (type is a constant of Schema::Values::Types: the closure refers to it)
			this->push_capture([this, type = &type, body = std::move(body)]{
				nlohmann::json entry;
				entry[Schema::Keys::TYPE] = *type;
				body(entry);
				this->submit(std::move(entry));
			}, kind);
		}

		template<typename... Fs>
		inline void Tracer::record(CaptureKind kind, const std::string &type, Fs... fields)
		{
			this->record_with(kind, type, [fields = std::make_tuple(std::move(fields)...)](nlohmann::json &entry)
			{
				std::apply([&](const auto &...part) { (write_part(entry, part), ...); }, fields);
			});
		}
#endif

		inline void Tracer::push_capture(std::function<void()> &&capture, CaptureKind kind)
		{
#ifdef SKEPU_TRACING
			if (this->m_lean && kind != CaptureKind::Raw)
				return;
			if (this->m_suppress_node && kind == CaptureKind::Node && allow_depth() == 0)
				return;
			bool many;
			{
				std::lock_guard<std::mutex> lock(this->m_mutex);
				this->m_captures.push_back(Capture{std::this_thread::get_id(), std::move(capture)});
				many = this->m_captures.size() >= kAutoFlushCaptures;
			}
			if (many)
				this->flush();   // (the closures take memory until they are replayed: this keeps it bounded, and the file is complete after it)
#endif
		}

#ifdef SKEPU_TRACING
		inline void Tracer::raw(nlohmann::json entry)
		{
			auto capture = [=, this]{
			this->submit(entry);
			}; this->push_capture(std::move(capture), CaptureKind::Raw);
		}
#endif

		inline EventHandle Tracer::startEvent()
		{
			return EventHandle{};
		}

		inline void Tracer::call(
			EventHandle &h,
			std::string pattern,
			backend::SkeletonBase *skel_instance,
			std::vector<size_t> &&elements,
			std::vector<TraceID> &&elwise_outputs,
			std::vector<TraceID> &&elwise_inputs,
			std::vector<TraceID> &&proxy_inputs,
			std::vector<TraceID> &&scalar_inputs,
			TraceID prng,
			Parameters const *extra)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();
			nlohmann::json extra_fields = extra ? extra->values : nlohmann::json::object();
			std::tuple<std::string, std::string, int> source_location = skel_instance->getSourceLoc();
			const BackendSpec *backend = skel_instance->m_selected_spec;
			UniqueIdentifier::ID id = skel_instance->getObjectID();
			std::string label = skel_instance->getLabel();
			// (the backend that the call used: the type of an automatic choice is the default)
			const Backend::Type type = backend ? backend->type() : Backend::Type::Auto;
			const size_t threads = backend ? backend->CPUThreads() : 0, devices = skel_instance->traceDevices();
			Parameters parameters;
			skel_instance->traceParameters(parameters);
			// (an instance says where it is defined with its first call, also when that event is not recorded: the call
			// of a skeleton that an MPI skeleton makes on its node is not, but the call of the MPI skeleton is)
			{
				std::lock_guard<std::mutex> lock(this->m_mutex);
				if (!this->m_definitions.count(id))
				{
					Definition definition;
					skel_instance->traceDefinition(definition);
					this->m_definitions[id] = std::move(definition.values);
				}
			}
			// a MapPool is a MapOverlap that pools: say so in the name
			if (parameters.values.value("pool", false) && pattern.compare(0, 10, "MapOverlap") == 0)
				pattern.replace(0, 10, "MapPool");

			// (what the skeleton adds to the call, and what runs it: the backend, the threads of the host and the devices that a call used)
			nlohmann::json more = extra_fields;
			if (backend)
			{
				std::stringstream ss;
				ss << type;
				more[Schema::Keys::BACKEND] = ss.str();
				if (type == Backend::Type::CPU)
					more[Schema::Keys::THREADS] = 1;
				else if (type == Backend::Type::OpenMP || type == Backend::Type::Hybrid)
					more[Schema::Keys::THREADS] = threads;
				if (type == Backend::Type::OpenCL || type == Backend::Type::CUDA || type == Backend::Type::Hybrid)
					more[Schema::Keys::DEVICES] = devices;
			}
			const std::string &file = std::get<0>(source_location), &file_hash = std::get<1>(source_location);

			this->record(Type::CALL,
				Key::OBJECT_ID = id,
				Key::PATTERN = pattern,
				Key::LABEL = label,
				Key::START = h.m_time,
				Key::END = now,
				Key::ELEMENTS = elements,
				Key::FILE = file,
				Key::LINE = std::get<2>(source_location),
				Key::ELWISE_INPUTS = elwise_inputs,
				Key::PROXY_INPUTS = proxy_inputs,
				Key::SCALAR_INPUTS = scalar_inputs,
				Key::OUTPUTS = elwise_outputs,
				Key::PRNG.when(prng != UniqueIdentifier::null()) = prng,
				Key::PARAMETERS.when(!parameters.values.empty()) = parameters.values,
				merge(std::move(more)),
				then([this, file, file_hash]{ this->noteSourceFile(file, file_hash); }),
				then([this, id, label]{ this->noteInstance(id, label); }));
#endif
		}

		inline void Tracer::call(
			std::string pattern,
			backend::SkeletonBase *skel_instance,
			std::vector<size_t> &&elements,
			std::vector<TraceID> &&elwise_outputs,
			std::vector<TraceID> &&elwise_inputs,
			std::vector<TraceID> &&proxy_inputs,
			std::vector<TraceID> &&scalar_inputs,
			TraceID prng,
			std::function<void(void)> &work_func)
		{
#ifdef SKEPU_TRACING
			auto h = this->startEvent();
			work_func();
			this->call(h, pattern, skel_instance, std::move(elements), std::move(elwise_outputs), std::move(elwise_inputs), std::move(proxy_inputs), std::move(scalar_inputs), prng);
#endif
		}

		inline void Tracer::external(EventHandle &h,
			std::string const& label,
			std::string const& file,
			int line,
			std::vector<TraceID> &&outputs,
			std::vector<TraceID> &&inputs
		)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			this->record(Type::EXTERNAL,
				Key::LABEL = label,
				Key::START = h.m_time,
				Key::END = now,
				Key::ELWISE_INPUTS = std::vector<TraceID>{},
				Key::PROXY_INPUTS = inputs,
				Key::SCALAR_INPUTS = std::vector<TraceID>{},
				Key::OUTPUTS = outputs,
				Key::FILE = file,
				Key::LINE = line);
#endif
		}
		
		inline void Tracer::scalar_io(std::vector<TraceID> const& ids, std::string const& label, std::string const& file, int line)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			this->record(Type::EXTERNAL,
				Key::LABEL = label,
				Key::START = now,
				Key::END = now,
				Key::ELWISE_INPUTS = std::vector<TraceID>{},
				Key::PROXY_INPUTS = std::vector<TraceID>{},
				Key::SCALAR_INPUTS = ids,
				Key::OUTPUTS = std::vector<TraceID>{},
				Key::FILE = file,
				Key::LINE = line);
#endif
		}

		template<typename C>
		inline void snapshot_serializer(C &data, std::string &fname);

		template<typename T>
		inline void snapshot_serializer(skepu::Vector<T> &data, std::string &fname)
		{
			std::ofstream fs(fname);
			data.flush();
			{
			    fs << 1 << " ";
    			fs << data.size() << "\n";
    			for (size_t i = 0; i < data.size(); ++i)
    			{
    				fs << data(i) << " ";
    			}
    	    }
		}

		template<typename T>
		inline void snapshot_serializer(skepu::Matrix<T> &data, std::string &fname)
		{
			std::ofstream fs(fname);
			data.flush();
			{
			    fs << 2 << " ";
    			fs << data.size_i() << " " << data.size_j() << "\n";
    			for (size_t i = 0; i < data.size_i(); ++i)
    			{
                    for (size_t j = 0; j < data.size_j(); ++j)
        			{
           				fs << data(i, j) << " ";
                    }
    			}
    	    }
		}

		template<typename T>
		inline void snapshot_serializer(skepu::Tensor3<T> &data, std::string &fname)
		{
			std::ofstream fs(fname);
			data.flush();
			fs << 3 << " " << data.size_i() << " " << data.size_j() << " " << data.size_k() << "\n";
			for (size_t i = 0; i < data.size_i(); ++i)
				for (size_t j = 0; j < data.size_j(); ++j)
					for (size_t k = 0; k < data.size_k(); ++k)
						fs << data(i, j, k) << " ";
		}

		template<typename T>
		inline void snapshot_serializer(skepu::Tensor4<T> &data, std::string &fname)
		{
			std::ofstream fs(fname);
			data.flush();
			fs << 4 << " " << data.size_i() << " " << data.size_j() << " " << data.size_k() << " " << data.size_l() << "\n";
			for (size_t i = 0; i < data.size_i(); ++i)
				for (size_t j = 0; j < data.size_j(); ++j)
					for (size_t k = 0; k < data.size_k(); ++k)
						for (size_t l = 0; l < data.size_l(); ++l)
							fs << data(i, j, k, l) << " ";
		}

 		template<typename C
        //, REQUIRES_VALUE(is_skepu_container<C>)
        >
		inline void Tracer::data_snapshot(
		    C &data
			SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER
		)
		{
			// (a string: the overload that takes the place, not this one)
			this->data_snapshot(data, std::string(SKEPU_EXTERNAL_SOURCE_LOCATION_FILE), SKEPU_EXTERNAL_SOURCE_LOCATION_LINE);
		}

 		template<typename C>
		inline void Tracer::data_snapshot(C &data, std::string const& file, int line)
		{
#ifdef SKEPU_TRACING
		    static std::atomic<int> counter{0};
			int counter_value = ++counter;

			auto id = data.getObjectID();
			auto label = data.getLabel();
			unsigned long start = timestamp_now();
			std::stringstream ss_fname;
			ss_fname << SKEPU_TRACE_SNAPSHOT_DIR_NAME << "/";
			ss_fname << label << "@" << counter_value << ".txt";
			auto fname = ss_fname.str();
            snapshot_serializer(data, fname);

            unsigned long end = timestamp_now();
      
			this->record(CaptureKind::Region, Type::SNAPSHOT,
				Key::OBJECT_ID = id,
				Key::LABEL = label,
				Key::START = start,
				Key::END = end,
				Key::DATA_FILE = fname,
				Key::FILE = file,
				Key::LINE = line);
         
#endif
		}

		template<template<typename> typename C1, template<typename> typename C2, typename T1, typename T2
		//, REQUIRES_VALUE(is_skepu_container<C1> && is_skepu_container<C2>)
		>
		inline void Tracer::virtual_dataset(
		    std::string const& label,
		    C1<T1> &data1, C2<T2> &data2
			SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER
		)
		{
			// (a string: the overload that takes the place, not this one)
			this->virtual_dataset(label, data1, data2, std::string(SKEPU_EXTERNAL_SOURCE_LOCATION_FILE), SKEPU_EXTERNAL_SOURCE_LOCATION_LINE);
		}

		template<template<typename> typename C1, template<typename> typename C2, typename T1, typename T2>
		inline void Tracer::virtual_dataset(
		    std::string const& label,
		    C1<T1> &data1, C2<T2> &data2,
		    std::string const& file, int line
		)
		{
#ifdef SKEPU_TRACING
            unsigned long time = timestamp_now();

            std::vector<TraceID> ids{ data1.getObjectID(), data2.getObjectID() };
                
            this->record(CaptureKind::Region, Type::VIRTUAL_DATASET,
                Key::VIRTUAL_ID = UniqueIdentifier::generate(),
                Key::OBJECT_IDS = ids,
                Key::LABEL = label,
                Key::START = time,
                Key::END = time,
                Key::FILE = file,
                Key::LINE = line);
#endif
		}

		inline void Tracer::beginRegion(std::string const& label SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER)
		{
			// (a string: the overload that takes the place, not this one)
			this->beginRegion(label, std::string(SKEPU_EXTERNAL_SOURCE_LOCATION_FILE), SKEPU_EXTERNAL_SOURCE_LOCATION_LINE);
		}

		inline void Tracer::beginRegion(std::string const& label, std::string const& file, int line)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();
			this->push_capture([=, this]{ this->replayBeginRegion(label, file, line, now, false); }, CaptureKind::Region);
#endif
		}

		inline void Tracer::beginAutoRegion(std::string const& label, std::string const& file, int line)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();
			this->push_capture([=, this]{ this->replayBeginRegion(label, file, line, now, true); }, CaptureKind::Region);
#endif
		}

		inline void Tracer::endRegion()
		{
#ifdef SKEPU_TRACING
			this->push_capture([this]{ this->replayEndRegion(); }, CaptureKind::Region);
#endif
		}

#ifdef SKEPU_TRACING
		inline void Tracer::replayBeginRegion(std::string label, std::string file, int line, unsigned long time, bool automatic)
		{
			nlohmann::json entry;
			entry[Schema::Keys::TYPE]  = Schema::Values::Types::REGION;
			entry[Schema::Keys::LABEL] = label;
			entry[Schema::Keys::TIME]  = time;
			entry[Schema::Keys::FILE]  = file;
			entry[Schema::Keys::LINE]  = line;
			if (automatic)
			{
				this->m_regions().push_back(RegionEntry{std::move(label), false, std::move(entry), nullptr});
				return;
			}
			// (the regions around a region that is written are written: the trace has them first)
			this->commitRegions();
			auto &stack = this->m_regions();
			entry[Schema::Keys::DEPTH] = stack.size() + 1;
			if (!stack.empty())
				entry[Schema::Keys::REGION] = stack.back().label;
			this->emit(std::move(entry));
			stack.push_back(RegionEntry{std::move(label), true, nullptr, nullptr});
		}

		inline void Tracer::replayEndRegion()
		{
			auto &stack = this->m_regions();
			if (stack.empty())
				return;
			RegionEntry region = std::move(stack.back());
			stack.pop_back();
			// (the one event of a region that is not written is in the region around it)
			if (!region.committed && !region.buffered.is_null())
				this->submit(std::move(region.buffered));
		}

		inline void Tracer::submit(nlohmann::json entry)
		{
			auto &stack = this->m_regions();
			if (!stack.empty() && !stack.back().committed)
			{
				RegionEntry &region = stack.back();
				if (region.events++ == 0)
				{
					region.buffered = std::move(entry);
					return;
				}
				this->commitRegions();
			}
			if (!stack.empty())
				entry[Schema::Keys::REGION] = stack.back().label;
			this->emit(std::move(entry));
		}

		inline void Tracer::commitRegions()
		{
			auto &stack = this->m_regions();
			for (size_t i = 0; i < stack.size(); ++i)
			{
				RegionEntry &region = stack[i];
				if (region.committed)
					continue;
				region.begin[Schema::Keys::DEPTH] = i + 1;
				if (i > 0)
					region.begin[Schema::Keys::REGION] = stack[i - 1].label;
				this->emit(std::move(region.begin));
				region.begin = nullptr;
				region.committed = true;
				if (!region.buffered.is_null())
				{
					nlohmann::json event = std::move(region.buffered);
					region.buffered = nullptr;
					event[Schema::Keys::REGION] = region.label;
					this->emit(std::move(event));
				}
			}
		}

		inline void Tracer::settleRegions()
		{
			for (auto &stack : this->m_region_stacks)
			{
				this->m_replaying_thread = stack.first;
				for (const RegionEntry &region : stack.second)
					if (!region.committed && !region.buffered.is_null())
					{
						this->commitRegions();
						break;
					}
			}
		}
#endif

		inline void Tracer::region(std::string const& label, std::function<void(void)> &&work_func SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER)
		{
			this->beginRegion(label SKEPU_EXTERNAL_SOURCE_LOCATION_FORWARD_ARG);
			struct End { Tracer &tracer; ~End() { tracer.endRegion(); } } end{*this};   // (also if work_func throws)
			work_func();
		}

		// One default tracer for the whole program (an inline function's local static is shared by
		// all translation units), created on first use: a container that traces its allocation
		// is then destroyed before the tracer, which writes the trace file when it is destroyed.
		// (The trace file name is the one of the translation unit that gets here first.)
		inline Tracer &internal_defaultGlobalTracer()
		{
			static Tracer defaultTracer(SKEPU_TRACE_FILE_NAME);
			return defaultTracer;
		}

		// Enables global tracer across multiple translation units
		inline Tracer **internal_GlobalTracerAccessor()
		{
			static Tracer *local_globalTracer = &internal_defaultGlobalTracer();
			return &local_globalTracer;
		}

		inline void setGlobalTracer(Tracer *spec)
		{
			*internal_GlobalTracerAccessor() = spec;
		}

		inline void restoreDefaultGlobalTracer()
		{
			*internal_GlobalTracerAccessor() = &internal_defaultGlobalTracer();
		}

		inline Tracer &tracer()
		{
			return **internal_GlobalTracerAccessor();
		}


		inline Scope::Scope(std::string const& label SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER)
		{
			tracer().beginRegion(label SKEPU_EXTERNAL_SOURCE_LOCATION_FORWARD_ARG);
		}

		inline Scope::~Scope()
		{
			tracer().endRegion();
		}

#ifdef SKEPU_TRACING
		namespace detail
		{
			// The labels of the library calls that are in progress on this thread, and the place that a frontend gives the next one
			inline std::vector<std::string> &api_calls() { static thread_local std::vector<std::string> calls; return calls; }
			struct CallSiteState { const char *file = nullptr; int line = 0; };
			inline CallSiteState &call_site() { static thread_local CallSiteState state; return state; }
		}

		inline ApiScope::ApiScope(std::string const& label, const char *file, int line): m_active(false)
		{
			auto &calls = detail::api_calls();
			if (!calls.empty() && calls.back() == label)
				return;
			auto &site = detail::call_site();
			if (site.file)
			{
				file = site.file;
				line = site.line;
				site.file = nullptr;   // (the calls inside this one are in the library)
			}
			calls.push_back(label);
			m_active = true;
			tracer().beginRegion(label, file ? file : "", line);
		}

		inline ApiScope::~ApiScope()
		{
			if (!m_active)
				return;
			detail::api_calls().pop_back();
			tracer().endRegion();
		}

		inline AutoRegion::AutoRegion(const char *label, const char *file, int line)
		{
			tracer().beginAutoRegion(label, file, line);
		}

		inline AutoRegion::~AutoRegion()
		{
			tracer().endRegion();
		}

		inline CallSite::CallSite(const char *file, int line)
		{
			auto &site = detail::call_site();
			site.file = file;
			site.line = line;
		}

		inline CallSite::~CallSite()
		{
			detail::call_site().file = nullptr;
		}
#endif
	SKEPU_TRACING_VARIANT_END
	}

	// Names a container in the traces and the debug output, unless the program has named it (a label that the program
	// gives, in the constructor or with setLabel, is the name). skepu-tool adds a call after each declaration of a container
	// variable, with the name of the variable. It does nothing if the program is not traced.
	// The place that skepu-tool gives (the file and line of the declaration, in the program as the programmer wrote it) is the place of the allocation event.
	template<typename Container>
	void trace_label(Container &container, const char *label, const char *file = nullptr, int line = 0)
	{
#ifdef SKEPU_TRACING
		if (!container.hasLabel())
			container.setLabel(label);
		if (file)
			SKEPU_TRACE_PLACE(container.getObjectID(), file, line);
#endif
	}
}

#endif
