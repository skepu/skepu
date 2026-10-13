/*! \file scan.h
 *  \brief Contains a class declaration for the skeleton base class.
 */

#ifndef SKELETON_BASE_H
#define SKELETON_BASE_H

#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "skepu3/backend/environment.h"
#include "skepu3/impl/chain.hpp"

namespace skepu
{
	namespace backend
	{
		// Whether a user function struct that the precompiler made says that it needs double precision (a double in its signature, or in the signature of a
		// function that it calls): false for one that does not say it
		template<typename F, typename = void>
		struct uf_requires_double: std::false_type {};

		template<typename F>
		struct uf_requires_double<F, std::void_t<decltype(F::requiresDouble)>>: std::integral_constant<bool, F::requiresDouble> {};

		// Whether any of the user functions of a skeleton needs double precision
		template<typename... Fs>
		struct any_uf_requires_double: std::integral_constant<bool, (uf_requires_double<Fs>::value || ...)> {};

		// Whether an OpenCL device can run double precision: the skeletons whose user functions have doubles cannot be run on the OpenCL backend (and their kernels
		// cannot be built) if no device can; they run on another backend
		inline bool openclSupportsDouble()
		{
#ifdef SKEPU_OPENCL
			return Environment<int>::getInstance()->numDoubleDevices_CL() > 0;
#else
			return false;
#endif
		}

		class SkeletonBase
		{
		public:
			void finishAll()
			{
				this->m_environment->finishAll();
			}

			// Transfers ownership of ´plan´
			void setExecPlan(ExecPlan *plan)
			{
				// Clean up old plan
				if (this->m_execPlan != nullptr)
					delete this->m_execPlan;

				this->m_execPlan = plan;
			}

			void setBackend(BackendSpec const& spec)
			{
				this->m_user_spec = new BackendSpec(spec);
			}

			void resetBackend()
			{
				if (this->m_user_spec)
					delete this->m_user_spec;
				this->m_user_spec = nullptr;
			}

			void setLabel(std::string&& label)
			{
#ifdef SKEPU_TRACING
				this->m_label = label;
#endif
			}

			std::string getLabel()
			{
#ifdef SKEPU_TRACING
				if (this->m_label != "")
					return this->m_label;
#endif
				return skepu::debug.getObjectName(this, "Skeleton #" + std::to_string(this->m_object_id));
			}

#ifdef SKEPU_TRACING
			UniqueIdentifier::ID getObjectID()
			{
				return this->m_object_id;
			}
#endif

			// The parameters that change what a call does (overlap, strides, modes, ...): a skeleton with
			// such parameters says what they are now, and the tracer records them with each call
			virtual void traceParameters(tracing::Parameters &) const {}

			// Where the instance and its user functions are defined in the source (the tracer writes it once for each instance)
			virtual void traceDefinition(tracing::Definition &) const {}

			// How many devices a call on a GPU backend uses: the devices that the backend specification asks for, if there are that many
			size_t traceDevices() const
			{
				size_t available = 0;
#ifdef SKEPU_OPENCL
				available = this->m_environment->m_devices_CL.size();
#elif defined(SKEPU_CUDA)
				available = this->m_environment->m_devices_CU.size();
#endif
				return this->m_selected_spec ? std::min(this->m_selected_spec->devices(), available) : available;
			}

			void setSourceLoc(std::string&& file_name, std::string&& file_content_hash, int line_nr)
			{
#ifdef SKEPU_TRACING
				this->m_file_name = file_name;
				this->m_file_content_hash = file_content_hash;
				this->m_line_nr = line_nr;
#endif
			}

#ifdef SKEPU_TRACING
			std::tuple<std::string&, std::string&, int> getSourceLoc()
			{
				return { this->m_file_name, this->m_file_content_hash, this->m_line_nr };
			}
#endif

			const BackendSpec& selectBackend(size_t size = 0)
			{
				if (this->m_user_spec)
					this->m_selected_spec = this->m_user_spec;
				else if (this->m_execPlan)
					this->m_selected_spec = &this->m_execPlan->find(size);
				else
					this->m_selected_spec = &internalGlobalBackendSpecAccessor();

			//	this->m_selected_spec = (this->m_user_spec != nullptr)
			//		? this->m_user_spec
			//		: &this->m_execPlan->find(size);
#ifdef SKEPU_OPENCL
				// A skeleton with double precision cannot run on OpenCL devices that do not have it: it uses the OpenMP (or the CPU) backend instead,
				// and says so once; with the environment variable SKEPU_STRICT_BACKEND set (not 0) it is an error instead
				if (this->m_needs_double && this->m_environment->numDoubleDevices_CL() == 0)
				{
					const Backend::Type type = this->m_selected_spec->type();
					if (type == Backend::Type::OpenCL || type == Backend::Type::Hybrid)
						this->m_selected_spec = this->doubleFallbackSpec(type);
				}
#endif
				return *this->m_selected_spec;
			}

			void setPRNG(PRNG &prng, size_t iterations = 1)
			{
				this->m_prng = &prng;
				this->m_prng->registerInstance(this, iterations);
			}

		protected:
			// Builds the OpenCL kernels of the skeleton, unless it needs double precision and no OpenCL device has it (the build fails for it, and the skeleton does not run there)
			template<typename CLKernel>
			void prepareOpenCL(bool needsDouble)
			{
#ifdef SKEPU_OPENCL
				this->m_needs_double = needsDouble;
				if (!needsDouble || this->m_environment->numDoubleDevices_CL() > 0)
					CLKernel::initialize();
#endif
			}

			SkeletonBase(std::string label): m_label{label}
			{
				this->m_environment = Environment<int>::getInstance();
				this->m_object_id = UniqueIdentifier::generate();
			/*
#if defined(SKEPU_OPENCL)
				BackendSpec bspec(Backend::Type::OpenCL);
				bspec.setDevices(this->m_environment->m_devices_CL.size());
				bspec.setGPUThreads(this->m_environment->m_devices_CL.at(0)->getMaxThreads());
				bspec.setGPUBlocks(this->m_environment->m_devices_CL.at(0)->getMaxBlocks());

#elif defined(SKEPU_CUDA)
				BackendSpec bspec(Backend::Type::CUDA);
				bspec.setDevices(this->m_environment->m_devices_CU.size());
				bspec.setGPUThreads(this->m_environment->m_devices_CU.at(0)->getMaxThreads());
				bspec.setGPUBlocks(this->m_environment->m_devices_CU.at(0)->getMaxBlocks());

#elif defined(SKEPU_OPENMP)
				BackendSpec bspec(Backend::Type::OpenMP);
				bspec.setCPUThreads(omp_get_max_threads());

#else
				BackendSpec bspec(Backend::Type::CPU);
#endif
				ExecPlan *plan = new ExecPlan();
				plan->setCalibrated();
				plan->add(1, MAX_SIZE, bspec);
				setExecPlan(plan);
			*/
			}


			template<size_t randomCount, REQUIRES(randomCount != SKEPU_NO_RANDOM)>
			Random<randomCount> prepareRandom(size_t size)
			{
				PRNG* use_prng = this->m_prng;
				if (this->m_prng == nullptr)
					use_prng = &getGlobalPRNG();
					//SKEPU_ERROR("No random stream set in skeleton instance");
				return use_prng->template asRandom<randomCount>(size);
			}

			template<size_t randomCount, REQUIRES(randomCount != SKEPU_NO_RANDOM)>
			skepu::Vector<Random<randomCount>> prepareRandom(size_t size, size_t copies, size_t atomic_size = 1)
			{
				PRNG* use_prng = this->m_prng;
				if (this->m_prng == nullptr)
					use_prng = &getGlobalPRNG();
				//SKEPU_ERROR("No random stream set in skeleton instance");
				return use_prng->template asRandom<randomCount>(size, copies, atomic_size);
			}

			template<size_t randomCount, REQUIRES(randomCount == SKEPU_NO_RANDOM)>
			PRNG::Placeholder prepareRandom(size_t size = 0, size_t copies = 0, size_t atomic_size = 0)
			{
				return {};
			}
			
#ifdef SKEPU_TRACING
			template<size_t randomCount, REQUIRES(randomCount != SKEPU_NO_RANDOM)>
			UniqueIdentifier::ID randomID()
			{
				PRNG* use_prng = this->m_prng;
				if (this->m_prng == nullptr)
					use_prng = &getGlobalPRNG();
				return use_prng->getObjectID();
			}

			template<size_t randomCount, REQUIRES(randomCount == SKEPU_NO_RANDOM)>
			UniqueIdentifier::ID randomID()
			{
				return UniqueIdentifier::null();
			}
#endif

#ifdef SKEPU_OPENCL

			template<size_t randomCount, REQUIRES(randomCount != SKEPU_NO_RANDOM)>
			skepu::Vector<RandomForCL> prepareRandom_CL(size_t size, size_t copies, size_t atomic_size = 1)
			{
				PRNG* use_prng = this->m_prng;
				if (this->m_prng == nullptr)
					use_prng = &getGlobalPRNG();
				//SKEPU_ERROR("No random stream set in skeleton instance");
				return use_prng->template asRandom_CL<randomCount>(size, copies, atomic_size);
			}

			template<size_t randomCount, REQUIRES(randomCount == SKEPU_NO_RANDOM)>
			PRNG::Placeholder prepareRandom_CL(size_t size = 0, size_t copies = 0, size_t atomic_size = 0)
			{
				return {};
			}

#endif


			Environment<int>* m_environment;

			/*! this is the pointer to execution plan that is active and should be used by implementations to check numOmpThreads and cudaBlocks etc. */
			ExecPlan *m_execPlan = nullptr;

			const BackendSpec *m_user_spec = nullptr;

public: // friend the tracer?
			const BackendSpec *m_selected_spec = nullptr;

			PRNG *m_prng = nullptr;

			// The skeleton needs double precision, which the OpenCL devices may not have (see selectBackend); and what it uses instead of the OpenCL backend, then
			bool m_needs_double = false;
			bool m_double_fallback_logged = false;
			BackendSpec m_double_fallback_spec;

			const BackendSpec *doubleFallbackSpec(Backend::Type type)
			{
				const char *strict = std::getenv("SKEPU_STRICT_BACKEND");
				if (strict && *strict && std::string(strict) != "0")
					SKEPU_ERROR("The skeleton \"" << this->m_label << "\" has user functions with double precision, and no OpenCL device has it (SKEPU_STRICT_BACKEND is set: no other backend is used)");

				BackendSpec spec = *this->m_selected_spec;
#ifdef SKEPU_OPENMP
				const Backend::Type fallback = Backend::Type::OpenMP;
#else
				const Backend::Type fallback = Backend::Type::CPU;
#endif
				spec.setType(fallback);
				this->m_double_fallback_spec = spec;
				if (!this->m_double_fallback_logged)
				{
					this->m_double_fallback_logged = true;
					std::cerr << "[SkePU] The skeleton \"" << this->m_label << "\" has user functions with double precision, which no OpenCL device has: it uses the "
						<< fallback << " backend instead of the " << type << " backend (SKEPU_STRICT_BACKEND=1 makes it an error)\n";
				}
				return &this->m_double_fallback_spec;
			}

			std::string m_label;
			UniqueIdentifier::ID m_object_id;
#ifdef SKEPU_TRACING
			std::string m_file_name;
			std::string m_file_content_hash;
			int m_line_nr = -1;
#endif

		}; // class SkeletonBase

	} // namespace backend

} // namespace skepu


#endif // SKELETON_BASE_H
