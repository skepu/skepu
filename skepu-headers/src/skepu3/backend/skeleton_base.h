/*! \file scan.h
 *  \brief Contains a class declaration for the skeleton base class.
 */

#ifndef SKELETON_BASE_H
#define SKELETON_BASE_H

#include "skepu3/backend/environment.h"

namespace skepu
{
	namespace backend
	{
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
				std::stringstream ss;
				ss << skepu::debug.getObjectName(this);
				return ss.str();
			}

#ifdef SKEPU_TRACING
			UniqueIdentifier::ID getObjectID()
			{
				return this->m_object_id;
			}
#endif

			void setSourceLoc(std::string&& file_name, int line_nr)
			{
#ifdef SKEPU_TRACING
				this->m_file_name = file_name;
				this->m_line_nr = line_nr;
#endif
			}

#ifdef SKEPU_TRACING
			std::tuple<std::string&, int> getSourceLoc()
			{
				return { this->m_file_name, this->m_line_nr };
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
				return *this->m_selected_spec;
			}

			void setPRNG(PRNG &prng, size_t iterations = 1)
			{
				this->m_prng = &prng;
				this->m_prng->registerInstance(this, iterations);
			}

		protected:
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
				std::cout << "In skeleton_base PrepareRandom\n";
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

			std::string m_label;
			UniqueIdentifier::ID m_object_id;
#ifdef SKEPU_TRACING
			std::string m_file_name;
			int m_line_nr = -1;
#endif

		}; // class SkeletonBase

	} // namespace backend

} // namespace skepu


#endif // SKELETON_BASE_H
