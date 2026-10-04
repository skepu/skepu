#ifndef TRACING_HPP
#define TRACING_HPP


#include "tracing.h"
namespace skepu
{
	namespace tracing
	{
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

		static unsigned long timestamp_now()
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
				static const std::string DIRECTION{"direction"};
				static const std::string PRNG{"prng"};
				static const std::string SOURCE{"source"};
				static const std::string INDEX{"index"};
				static const std::string DATA_FILE{"data_file"};
			}

			namespace Values
			{
				namespace Types
				{
					static const std::string CALL{"skeleton_call"};
					static const std::string ALLOCATION{"allocation"};
					static const std::string DEALLOCATION{"deallocation"};
					static const std::string ELEM_ACCESS{"element_access"};
					static const std::string TRANSFER{"transfer"};
					static const std::string REGION{"region"};
					static const std::string EXTERNAL{"external"};
					static const std::string SNAPSHOT{"snapshot"};
					static const std::string VIRTUAL_DATASET{"virtual_container"};
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
		    if (this->m_file_name == defaultTraceFilename && std::getenv("SKEPU_TRACE_FILE"))
			{
				this->m_file_name = std::getenv("SKEPU_TRACE_FILE");
			}
        }

		inline Tracer::~Tracer()
		{
#ifdef SKEPU_TRACING
			for (auto& capture : this->m_captures)
				capture();

			std::cout << "Tracer destructor with filename " << this->m_file_name << " size: " << this->m_trace_json.dump(4).size() << "\n";
			std::ofstream ofs;
			ofs.open(this->m_file_name, std::ofstream::out | std::ofstream::trunc);
			ofs << this->m_trace_json.dump(4) << std::endl;
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
			if (!this->m_region_stack.empty())
				entry[Schema::Keys::REGION] = this->m_region_stack.top();

			this->m_trace_json.push_back(entry);
			}; this->m_captures.push_back(capture);
#endif
		}

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
			if (!this->m_region_stack.empty())
				entry[Schema::Keys::REGION] = this->m_region_stack.top();

			this->m_trace_json.push_back(entry);
			}; this->m_captures.push_back(capture);
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
			entry[Schema::Keys::LINE]  = line;
			if (!this->m_region_stack.empty())
				entry[Schema::Keys::REGION] = this->m_region_stack.top();

			this->m_trace_json.push_back(entry);
			}; this->m_captures.push_back(capture);
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
			if (!this->m_region_stack.empty())
				entry[Schema::Keys::REGION] = this->m_region_stack.top();

			this->m_trace_json.push_back(entry);
			}; this->m_captures.push_back(capture);
#endif
		}

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
			TraceID prng)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();
			std::tuple<std::string, int> source_location = skel_instance->getSourceLoc();
			const BackendSpec *backend = skel_instance->m_selected_spec;
			UniqueIdentifier::ID id = skel_instance->getObjectID();
			std::string label = skel_instance->getLabel();
			size_t threads = backend->CPUThreads();

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
			entry[Schema::Keys::LINE]          = std::get<1>(source_location);
			entry[Schema::Keys::ELWISE_INPUTS] = elwise_inputs;
			entry[Schema::Keys::PROXY_INPUTS]  = proxy_inputs;
			entry[Schema::Keys::SCALAR_INPUTS] = scalar_inputs;
			entry[Schema::Keys::OUTPUTS]       = elwise_outputs;
			if (prng != UniqueIdentifier::null())
				entry[Schema::Keys::PRNG]      		 = prng;
			if (!this->m_region_stack.empty())
				entry[Schema::Keys::REGION] = this->m_region_stack.top();
			if (backend)
			{
				std::stringstream ss;
				ss << backend->getType();
				entry[Schema::Keys::BACKEND]       = ss.str();
				entry[Schema::Keys::THREADS]       = threads;
			}

			this->m_trace_json.push_back(entry);
			}; this->m_captures.push_back(capture);
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
			this->call(h, pattern, skel_instance, std::move(elements), std::move(elwise_outputs), std::move(elwise_inputs), std::move(proxy_inputs), std::move(scalar_inputs));
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
			if (!this->m_region_stack.empty())
				entry[Schema::Keys::REGION] = this->m_region_stack.top();

			this->m_trace_json.push_back(entry);
			}; this->m_captures.push_back(capture);
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
			entry[Schema::Keys::FILE]          = "";
			entry[Schema::Keys::LINE]          = line;
			if (!this->m_region_stack.empty())
				entry[Schema::Keys::REGION] = this->m_region_stack.top();

			this->m_trace_json.push_back(entry);
			}; this->m_captures.push_back(capture);
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

 		template<typename C
        //, REQUIRES_VALUE(is_skepu_container<C>)
        >
		inline void Tracer::data_snapshot(
		    C &data
			SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER
		)
		{
#ifdef SKEPU_TRACING
		    static int counter = 0;
			counter++;

			auto id = data.getObjectID();
			auto label = data.getLabel();
			unsigned long start = timestamp_now();
			std::stringstream ss_fname;
			ss_fname << SKEPU_TRACE_SNAPSHOT_DIR_NAME << "/";
			ss_fname << label << "@" << counter << ".txt";
			auto fname = ss_fname.str();
            std::cout << "Test: " << label << " UID: " << counter << "\n" << "File name: " << fname << "\n";
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
			entry[Schema::Keys::FILE]      = SKEPU_EXTERNAL_SOURCE_LOCATION_FILE;
			entry[Schema::Keys::LINE]      = SKEPU_EXTERNAL_SOURCE_LOCATION_LINE;
			if (!this->m_region_stack.empty())
				entry[Schema::Keys::REGION] = this->m_region_stack.top();
      
			this->m_trace_json.push_back(entry);
			}; this->m_captures.push_back(capture);
         
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
            entry[Schema::Keys::FILE]       = SKEPU_EXTERNAL_SOURCE_LOCATION_FILE;
            entry[Schema::Keys::LINE]       = SKEPU_EXTERNAL_SOURCE_LOCATION_LINE;
            if (!this->m_region_stack.empty())
            entry[Schema::Keys::REGION] = this->m_region_stack.top();
                
            this->m_trace_json.push_back(entry);
            }; this->m_captures.push_back(capture);
#endif
		}

		inline void Tracer::beginRegion(std::string const& label SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER)
		{
#ifdef SKEPU_TRACING
			unsigned long now = timestamp_now();

			auto capture = [=, this]{
			nlohmann::json entry;
			entry[Schema::Keys::TYPE]  = Schema::Values::Types::REGION;
			entry[Schema::Keys::LABEL] = label;
			entry[Schema::Keys::TIME]  = now;
			entry[Schema::Keys::FILE]  = SKEPU_EXTERNAL_SOURCE_LOCATION_FILE;
			entry[Schema::Keys::LINE]  = SKEPU_EXTERNAL_SOURCE_LOCATION_LINE;
			entry[Schema::Keys::DEPTH] = this->m_region_stack.size() + 1;
			if (!this->m_region_stack.empty())
				entry[Schema::Keys::REGION] = this->m_region_stack.top();

			this->m_trace_json.push_back(entry);
			this->m_region_stack.push(label);
			}; this->m_captures.push_back(capture);
#endif
		}

		inline void Tracer::endRegion()
		{
#ifdef SKEPU_TRACING
			auto capture = [=, this]{
			this->m_region_stack.pop();
			}; this->m_captures.push_back(capture);
#endif
		}

		inline void Tracer::region(std::string const& label, std::function<void(void)> &&work_func SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM_INNER)
		{
			this->beginRegion(label SKEPU_EXTERNAL_SOURCE_LOCATION_FORWARD_ARG);
			work_func();
			this->endRegion();
		}

		static Tracer internal_defaultGlobalTracer(SKEPU_TRACE_FILE_NAME);

		// Enables global tracer across multiple translation units
		inline Tracer **internal_GlobalTracerAccessor()
		{
			static Tracer *local_globalTracer = &internal_defaultGlobalTracer;
			return &local_globalTracer;
		}

		inline void setGlobalTracer(Tracer *spec)
		{
			*internal_GlobalTracerAccessor() = spec;
		}

		inline void restoreDefaultGlobalTracer()
		{
			*internal_GlobalTracerAccessor() = &internal_defaultGlobalTracer;
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
	}
}

#endif
