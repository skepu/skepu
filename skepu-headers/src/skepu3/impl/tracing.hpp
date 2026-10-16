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
				static const std::string OBJECT_ID{"object_id"};
				static const std::string OBJECT_IDS{"object_ids"};
				static const std::string VIRTUAL_ID{"virtual_id"};
				static const std::string TYPE{"type"};
				static const std::string LABEL{"label"};
				static const std::string TIME{"time"};
				static const std::string FILE{"file"};
				static const std::string LINE{"line"};
				static const std::string REGION{"region"};
				static const std::string START{"start"};
				static const std::string DEPTH{"region_depth"};
				static const std::string END{"end"};
				static const std::string ELEMENTS{"elements"};
				static const std::string ELWISE_INPUTS{"elwise_inputs"};
				static const std::string PROXY_INPUTS{"proxy_inputs"};
				static const std::string SCALAR_INPUTS{"scalar_inputs"};
				static const std::string OUTPUTS{"outputs"};
				static const std::string PATTERN{"pattern"};
				static const std::string BACKEND{"backend"};
				static const std::string THREADS{"threads"};
				static const std::string DEVICES{"devices"};
				static const std::string PARAMETERS{"parameters"};
				static const std::string DIRECTION{"direction"};
				static const std::string PRNG{"prng"};
				static const std::string SOURCE{"source"};
				static const std::string INDEX{"index"};
				static const std::string DATA_FILE{"data_file"};
				static const std::string VALUE{"value"};
				static const std::string QUALIFIER{"qualifier"};
				static const std::string DEVICE{"device"};
				static const std::string OPTIONS{"options"};
				static const std::string SOURCE_BYTES{"source_bytes"};
				static const std::string OK{"ok"};
				static const std::string ERROR{"error"};
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
		}

		// The files that the events name, with the hash of the contents of each (SHA-256 of the bytes), in header.files: what a viewer that reads
		// the sources from somewhere else (an editor) needs to know that it has the same version. A call of a skeleton has its hash from the
		// precompiler already; the other events (external, regions, snapshots) are hashed here, from the file as it is when the trace is written.
#ifdef SKEPU_TRACING
		inline void Tracer::collectSourceFiles()
		{
			if (!this->m_trace_json.is_array())
				return;
			for (const nlohmann::json &entry : this->m_trace_json)
			{
				auto file = entry.find(Schema::Keys::FILE);
				if (file == entry.end() || !file->is_string() || file->get<std::string>().empty())
					continue;
				const std::string name = file->get<std::string>();
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

		inline void Tracer::flush()
		{
#ifdef SKEPU_TRACING
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

			std::ofstream ofs;
			ofs.open(this->m_file_name, std::ofstream::out | std::ofstream::trunc);
			if (this->m_header.empty())
				ofs << this->m_trace_json.dump(4) << std::endl;
			else
			{
				nlohmann::json whole = nlohmann::json::object();
				whole["header"] = this->m_header;
				whole["data"] = this->m_trace_json.is_null() ? nlohmann::json::array() : this->m_trace_json;
				ofs << whole.dump(4) << std::endl;
			}
			ofs.close();
#endif
		}




		inline void Tracer::allocation(TraceID id, std::string const& label, int line)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::OBJECT_ID] = id;
			entry[Schema::Keys::TYPE]  = Schema::Values::Types::ALLOCATION;
			entry[Schema::Keys::LABEL] = label;
			entry[Schema::Keys::TIME]  = now;
			entry[Schema::Keys::LINE]  = line;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();

			this->m_trace_json.push_back(entry);
			}; this->push_capture(std::move(capture));
#endif
		}

		inline void Tracer::relabel(TraceID id, std::string const& label, bool instance)
		{
#ifdef SKEPU_TRACING
			// (the events are written in order, and a container is named right after its declaration: the
			// events of the container are the last ones; a Raw capture is also kept in the traces of the other processes of an MPI program.
			// A skeleton instance can be named again and again, for example by the layers of a network that share one: only the events
			// before its first call, the builds of its kernels, get the name; its calls keep the name that it had at the call)
			auto capture = [=, this]{
			for (auto it = this->m_trace_json.rbegin(); it != this->m_trace_json.rend(); ++it)
			{
				auto object = it->find(Schema::Keys::OBJECT_ID);
				if (object == it->end() || *object != id || !it->contains(Schema::Keys::LABEL))
					continue;
				if (instance && (*it)[Schema::Keys::TYPE] == Schema::Values::Types::CALL)
					break;
				(*it)[Schema::Keys::LABEL] = label;
				if ((*it)[Schema::Keys::TYPE] == Schema::Values::Types::ALLOCATION)
					break;
			}
			}; this->push_capture(std::move(capture), CaptureKind::Raw);
#endif
		}

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
			auto capture = [=, this]{
			nlohmann::json entry;
			if (instance != UniqueIdentifier::null())
				entry[Schema::Keys::OBJECT_ID] = instance;
			entry[Schema::Keys::TYPE]    = Schema::Values::Types::COMPILE;
			entry[Schema::Keys::LABEL]   = label;
			if (!qualifier.empty())
				entry[Schema::Keys::QUALIFIER] = qualifier;
			entry[Schema::Keys::BACKEND] = "OpenCL";
			entry[Schema::Keys::DEVICE]  = device;
			if (!options.empty())
				entry[Schema::Keys::OPTIONS] = options;
			entry[Schema::Keys::SOURCE_BYTES] = source_bytes;
			entry[Schema::Keys::OK]      = ok;
			if (!ok)
				entry[Schema::Keys::ERROR] = error;
			entry[Schema::Keys::START]   = start;
			entry[Schema::Keys::END]     = end;
			entry[Schema::Keys::TIME]    = end;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();

			this->m_trace_json.push_back(entry);
			}; this->push_capture(std::move(capture));
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

			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::OBJECT_ID] = id;
			entry[Schema::Keys::TYPE]  = Schema::Values::Types::DEALLOCATION;
			entry[Schema::Keys::LABEL] = label;
			entry[Schema::Keys::TIME]  = now;
			entry[Schema::Keys::LINE]  = line;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();

			this->m_trace_json.push_back(entry);
			}; this->push_capture(std::move(capture));
#endif
		}

		inline void Tracer::element_access(TraceID id, TraceID source_id, size_t index, std::string const& label, std::string const& file, int line)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::OBJECT_ID] = id;
			entry[Schema::Keys::SOURCE] = source_id;
			entry[Schema::Keys::TYPE]  = Schema::Values::Types::ELEM_ACCESS;
			entry[Schema::Keys::INDEX]  = index;
			entry[Schema::Keys::LABEL] = label;
			entry[Schema::Keys::TIME]  = now;
			entry[Schema::Keys::FILE]  = file;
			entry[Schema::Keys::LINE]  = line;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();

			this->m_trace_json.push_back(entry);
			}; this->push_capture(std::move(capture), CaptureKind::Region);
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

				auto capture = [=, this]{
				nlohmann::json entry;
				entry[Schema::Keys::OBJECT_ID] = id;
				entry[Schema::Keys::TYPE]  = Schema::Values::Types::SCALAR_VALUE;
				entry[Schema::Keys::VALUE] = number;
				entry[Schema::Keys::TIME]  = now;
				if (!this->m_regions().empty())
					entry[Schema::Keys::REGION] = this->m_regions().top();

				this->m_trace_json.push_back(entry);
				}; this->push_capture(std::move(capture), CaptureKind::Region);
			}
#endif
		}

		inline void Tracer::copy(EventHandle &h, TraceID source_id, TraceID destination_id, std::string const& label, size_t elements)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::TYPE]      = Schema::Values::Types::COPY;
			entry[Schema::Keys::SOURCE]    = source_id;
			entry[Schema::Keys::OBJECT_ID] = destination_id;
			entry[Schema::Keys::LABEL]     = label;
			entry[Schema::Keys::ELEMENTS]  = elements;
			entry[Schema::Keys::START]     = h.m_time;
			entry[Schema::Keys::END]       = now;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();

			this->m_trace_json.push_back(entry);
			}; this->push_capture(std::move(capture));
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

			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::OBJECT_ID] = id;
			entry[Schema::Keys::TYPE]  = Schema::Values::Types::TRANSFER;
			entry[Schema::Keys::ELEMENTS]  = elements;
			entry[Schema::Keys::DIRECTION] = direction;
			entry[Schema::Keys::BACKEND] = backend;
			entry[Schema::Keys::LABEL] = label;
			entry[Schema::Keys::START]   = h.m_time;
			entry[Schema::Keys::END]     = now;
			entry[Schema::Keys::TIME]  = now;
			entry[Schema::Keys::LINE]  = line;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();

			this->m_trace_json.push_back(entry);
			}; this->push_capture(std::move(capture));
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

		inline void Tracer::push_capture(std::function<void()> &&capture, CaptureKind kind)
		{
#ifdef SKEPU_TRACING
			if (this->m_lean && kind != CaptureKind::Raw)
				return;
			if (this->m_suppress_node && kind == CaptureKind::Node && allow_depth() == 0)
				return;
			std::lock_guard<std::mutex> lock(this->m_mutex);
			this->m_captures.push_back(Capture{std::this_thread::get_id(), std::move(capture)});
#endif
		}

#ifdef SKEPU_TRACING
		inline void Tracer::raw(nlohmann::json entry)
		{
			const bool lean = this->m_lean;
			auto capture = [=, this]{
			nlohmann::json e = entry;
			if (!lean && !this->m_regions().empty())
				e[Schema::Keys::REGION] = this->m_regions().top();
			this->m_trace_json.push_back(e);
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

			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::OBJECT_ID] = id;
			entry[Schema::Keys::TYPE]    = Schema::Values::Types::CALL;
			entry[Schema::Keys::PATTERN] = pattern;
			entry[Schema::Keys::LABEL]   = label;
			entry[Schema::Keys::START]   = h.m_time;
			entry[Schema::Keys::END]     = now;
			entry[Schema::Keys::ELEMENTS] = elements;
			entry[Schema::Keys::FILE]          = std::get<0>(source_location);
			entry[Schema::Keys::LINE]          = std::get<2>(source_location);
			entry[Schema::Keys::ELWISE_INPUTS] = elwise_inputs;
			entry[Schema::Keys::PROXY_INPUTS]  = proxy_inputs;
			entry[Schema::Keys::SCALAR_INPUTS] = scalar_inputs;
			entry[Schema::Keys::OUTPUTS]       = elwise_outputs;
			if (prng != UniqueIdentifier::null())
				entry[Schema::Keys::PRNG]      		 = prng;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();
			if (!parameters.values.empty())
				entry[Schema::Keys::PARAMETERS] = parameters.values;
			for (auto it = extra_fields.begin(); it != extra_fields.end(); ++it)
				entry[it.key()] = it.value();
			if (backend)
			{
				std::stringstream ss;
				ss << type;
				entry[Schema::Keys::BACKEND]       = ss.str();
				// the threads of the host, and the devices that a call used: what runs a call
				if (type == Backend::Type::CPU)
					entry[Schema::Keys::THREADS]   = 1;
				else if (type == Backend::Type::OpenMP || type == Backend::Type::Hybrid)
					entry[Schema::Keys::THREADS]   = threads;
				if (type == Backend::Type::OpenCL || type == Backend::Type::CUDA || type == Backend::Type::Hybrid)
					entry[Schema::Keys::DEVICES]   = devices;
			}

			// the hash of the source file that the call is in (computed by the precompiler: the program as it was compiled), once per file
			if (!std::get<0>(source_location).empty() && !std::get<1>(source_location).empty())
			{
				std::lock_guard<std::mutex> lock(this->m_mutex);
				if (!this->m_header["files"].contains(std::get<0>(source_location)))
					this->m_header["files"][std::get<0>(source_location)] = std::get<1>(source_location);
			}

			// where the instance and its user functions are defined: once, in the header of the file
			{
				std::lock_guard<std::mutex> lock(this->m_mutex);
				const nlohmann::json &definition = this->m_definitions[id];
				if (!definition.is_null() && !this->m_header["instances"].contains(std::to_string(id)))
				{
					nlohmann::json described = definition;
					described[Schema::Keys::LABEL] = label;
					this->m_header["instances"][std::to_string(id)] = std::move(described);
				}
			}

			this->m_trace_json.push_back(entry);
			}; this->push_capture(std::move(capture));
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

			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::TYPE]    = Schema::Values::Types::EXTERNAL;
			entry[Schema::Keys::LABEL]   = label;
			entry[Schema::Keys::START]   = h.m_time;
			entry[Schema::Keys::END]     = now;
			entry[Schema::Keys::ELWISE_INPUTS] = std::vector<TraceID>{};
			entry[Schema::Keys::PROXY_INPUTS]  = inputs;
			entry[Schema::Keys::SCALAR_INPUTS] = std::vector<TraceID>{};
			entry[Schema::Keys::OUTPUTS]       = outputs;
			entry[Schema::Keys::FILE]          = file;
			entry[Schema::Keys::LINE]          = line;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();

			this->m_trace_json.push_back(entry);
			}; this->push_capture(std::move(capture));
#endif
		}
		
		inline void Tracer::scalar_io(std::vector<TraceID> const& ids, std::string const& label, std::string const& file, int line)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::TYPE]  = Schema::Values::Types::EXTERNAL;
			entry[Schema::Keys::LABEL] = label;
			entry[Schema::Keys::START] = now;
			entry[Schema::Keys::END]   = now;
			entry[Schema::Keys::ELWISE_INPUTS] = std::vector<TraceID>{};
			entry[Schema::Keys::PROXY_INPUTS]  = std::vector<TraceID>{};
			entry[Schema::Keys::SCALAR_INPUTS] = ids;
			entry[Schema::Keys::OUTPUTS]       = std::vector<TraceID>{};
			entry[Schema::Keys::FILE]          = file;
			entry[Schema::Keys::LINE]          = line;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();

			this->m_trace_json.push_back(entry);
			}; this->push_capture(std::move(capture));
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
			this->data_snapshot(data, SKEPU_EXTERNAL_SOURCE_LOCATION_FILE, SKEPU_EXTERNAL_SOURCE_LOCATION_LINE);
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
      
			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::TYPE]      = Schema::Values::Types::SNAPSHOT;
			entry[Schema::Keys::OBJECT_ID] = id;
			entry[Schema::Keys::LABEL]     = label;
			entry[Schema::Keys::START]     = start;
			entry[Schema::Keys::END]       = end;
			entry[Schema::Keys::DATA_FILE] = fname;
			entry[Schema::Keys::FILE]      = file;
			entry[Schema::Keys::LINE]      = line;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();
      
			this->m_trace_json.push_back(entry);
			}; this->push_capture(std::move(capture), CaptureKind::Region);
         
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
			this->virtual_dataset(label, data1, data2, SKEPU_EXTERNAL_SOURCE_LOCATION_FILE, SKEPU_EXTERNAL_SOURCE_LOCATION_LINE);
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
                
            auto capture = [=, this]{
            nlohmann::json entry;
            entry[Schema::Keys::TYPE]       = Schema::Values::Types::VIRTUAL_DATASET;
            entry[Schema::Keys::VIRTUAL_ID] = UniqueIdentifier::generate();
            entry[Schema::Keys::OBJECT_IDS] = ids;
            entry[Schema::Keys::LABEL]      = label;
            entry[Schema::Keys::START]      = time;
            entry[Schema::Keys::END]        = time;
            entry[Schema::Keys::FILE]       = file;
            entry[Schema::Keys::LINE]       = line;
            if (!this->m_regions().empty())
            entry[Schema::Keys::REGION] = this->m_regions().top();
                
            this->m_trace_json.push_back(entry);
            }; this->push_capture(std::move(capture), CaptureKind::Region);
#endif
		}

		inline void Tracer::beginRegion(std::string const& label SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER)
		{
			this->beginRegion(label, SKEPU_EXTERNAL_SOURCE_LOCATION_FILE, SKEPU_EXTERNAL_SOURCE_LOCATION_LINE);
		}

		inline void Tracer::beginRegion(std::string const& label, std::string const& file, int line)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::TYPE]  = Schema::Values::Types::REGION;
			entry[Schema::Keys::LABEL] = label;
			entry[Schema::Keys::TIME]  = now;
			entry[Schema::Keys::FILE]  = file;
			entry[Schema::Keys::LINE]  = line;
			entry[Schema::Keys::DEPTH] = this->m_regions().size() + 1;
			if (!this->m_regions().empty())
				entry[Schema::Keys::REGION] = this->m_regions().top();

			this->m_trace_json.push_back(entry);
			this->m_regions().push(label);
			}; this->push_capture(std::move(capture), CaptureKind::Region);
#endif
		}

		inline void Tracer::endRegion()
		{
#ifdef SKEPU_TRACING
			auto capture = [=, this]{
			this->m_regions().pop();
			}; this->push_capture(std::move(capture), CaptureKind::Region);
#endif
		}

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
	SKEPU_TRACING_VARIANT_END
	}

	// Names a container in the traces and the debug output, unless the program has named it (a label that the program
	// gives, in the constructor or with setLabel, is the name). skepu-tool adds a call after each declaration of a container
	// variable, with the name of the variable. It does nothing if the program is not traced.
	template<typename Container>
	void trace_label(Container &container, const char *label)
	{
#ifdef SKEPU_TRACING
		if (!container.hasLabel())
			container.setLabel(label);
#endif
	}
}

#endif
