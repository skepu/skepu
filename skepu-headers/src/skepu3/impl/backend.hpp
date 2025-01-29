#pragma once

#include <vector>
#include <iostream>
#include <utility>
#include <cassert>
#include <algorithm>
#include <map>


#ifdef SKEPU_OPENMP
#ifndef _OPENMP
#error Fatal: OpenMP backend is activated in SkePU but the backend compiler is not OpenMP-capable.
#endif
#include <omp.h>
#endif

#include "../backend/debug.h"
#include "skepu3/impl/execution_model.hpp"

#ifndef SKEPU_DEFAULT_MAX_DEVICES
#define SKEPU_DEFAULT_MAX_DEVICES 4
#endif

// Set default CPU thread count if not supplied from elsewhere.
// Try to get the number of hardware threads from OpenMP if possible.
#ifndef SKEPU_DEFAULT_CPU_THREADS
#define SKEPU_DEFAULT_CPU_THREADS 16
#endif

// Set default GPU block count if not supplied from elsewhere.
#ifndef SKEPU_DEFAULT_GPU_BLOCKS
#define SKEPU_DEFAULT_GPU_BLOCKS 256
#endif

// Set default GPU thread count if not supplied from elsewhere.
#ifndef SKEPU_DEFAULT_GPU_THREADS
#define SKEPU_DEFAULT_GPU_THREADS 65535
#endif

// Set default hybrid partition ratio if not supplied from elsewhere.
#ifndef SKEPU_DEFAULT_CPU_PARTITION_RATIO
#define SKEPU_DEFAULT_CPU_PARTITION_RATIO 0.2f
#endif

namespace skepu
{
	struct Backend
	{
		enum class Type
		{
			Auto, CPU, OpenMP, OpenCL, CUDA, Hybrid
		};

		enum class Scheduling
		{
			Static, Dynamic, Guided, Auto
		};

		static constexpr size_t chunkSizeDefault = 0;

		static const std::vector<Type> &allTypes()
		{
			static const std::vector<Backend::Type> types
			{
				Backend::Type::CPU, Backend::Type::OpenMP, Backend::Type::OpenCL, Backend::Type::CUDA, Backend::Type::Hybrid
			};

			return types;
		}

		static const std::vector<Type> &availableTypes()
		{
			static const std::vector<Backend::Type> types
			{
				Backend::Type::CPU,
#ifdef SKEPU_OPENMP
				Backend::Type::OpenMP,
#endif
#ifdef SKEPU_OPENCL
				Backend::Type::OpenCL,
#endif
#ifdef SKEPU_CUDA
				Backend::Type::CUDA,
#endif
#ifdef SKEPU_HYBRID
				Backend::Type::Hybrid,
#endif
			};

			return types;
		}

#ifdef SKEPU_OPENMP
		static omp_sched_t convertSchedulingToOpenMP(Scheduling mode)
		{
			switch (mode)
			{
			case Scheduling::Static:  return omp_sched_static;
			case Scheduling::Dynamic: return omp_sched_dynamic;
			case Scheduling::Guided:  return omp_sched_guided;
			case Scheduling::Auto:    return omp_sched_auto;
			default: return omp_sched_static;
			}
		}
#endif

		static Type typeFromString(std::string s)
		{
			std::transform(s.begin(), s.end(), s.begin(), ::tolower);
			if (s == "cpu") return Type::CPU;
			else if (s == "openmp") return Type::OpenMP;
			else if (s == "opencl") return Type::OpenCL;
			else if (s == "cuda") return Type::CUDA;
			else if (s == "hybrid") return Type::Hybrid;
			else if (s == "auto") return Type::CUDA;
			else SKEPU_ERROR("Invalid string for backend type conversion");
		}

		static bool isTypeAvailable(Type type)
		{
			return type == Backend::Type::Auto ||
				std::find(availableTypes().begin(), availableTypes().end(), type) != availableTypes().end();
		}

	};





	inline std::ostream &operator<<(std::ostream &o, Backend::Type b)
	{
		switch (b)
		{
		case Backend::Type::CPU:    o << "CPU"; break;
		case Backend::Type::OpenMP: o << "OpenMP"; break;
		case Backend::Type::OpenCL: o << "OpenCL"; break;
		case Backend::Type::CUDA:   o << "CUDA"; break;
		case Backend::Type::Hybrid:   o << "Hybrid"; break;
		case Backend::Type::Auto:   o << "Auto"; break;
		default: o << ("Invalid backend type");
		}
		return o;
	}

	inline std::ostream &operator<<(std::ostream &o, Backend::Scheduling mode)
	{
		switch (mode)
		{
		case Backend::Scheduling::Static:  o << "static"; break;
		case Backend::Scheduling::Dynamic: o << "dynamic"; break;
		case Backend::Scheduling::Guided:  o << "guided"; break;
		case Backend::Scheduling::Auto:    o << "auto"; break;
		default: o << ("Invalid scheduling mode");
		}
		return o;
	}


	// "Tagged union" structure for specifying backend and parameters in a skeleton invocation
	struct BackendSpec
	{
		static constexpr Backend::Type defaultType
		{
#if   defined(SKEPU_SET_DEFAULT_BACKEND_CPU)
			Backend::Type::CPU
#elif defined(SKEPU_SET_DEFAULT_BACKEND_OPENMP)
			Backend::Type::OpenMP
#elif defined(SKEPU_SET_DEFAULT_BACKEND_OPENCL)
			Backend::Type::OpenCL
#elif defined(SKEPU_SET_DEFAULT_BACKEND_CUDA)
			Backend::Type::CUDA
#elif defined(SKEPU_SET_DEFAULT_BACKEND_HYBRID)
			Backend::Type::Hybrid

// No default backend compiler option present, fall back to internal preference order
#elif defined(SKEPU_OPENCL)
			Backend::Type::OpenCL
#elif defined(SKEPU_CUDA)
			Backend::Type::CUDA
#elif defined(SKEPU_OPENMP)
			Backend::Type::OpenMP
#else
			Backend::Type::CPU
#endif
		};

		static constexpr size_t defaultNumDevices {SKEPU_DEFAULT_MAX_DEVICES};
		static constexpr size_t defaultCPUThreads {SKEPU_DEFAULT_CPU_THREADS};
		static constexpr size_t defaultGPUThreads {SKEPU_DEFAULT_GPU_BLOCKS};
		static constexpr size_t defaultGPUBlocks {SKEPU_DEFAULT_GPU_THREADS};
		static constexpr float defaultCPUPartitionRatio {SKEPU_DEFAULT_CPU_PARTITION_RATIO};

		BackendSpec(Backend::Type type)
		: m_backend(type) {}

		BackendSpec(std::string type)
		: m_backend(Backend::typeFromString(type)) {}

		BackendSpec()
		: m_backend(defaultType) {}

		void setType(Backend::Type type)
		{
			this->m_backend = type;
		}

		Backend::Type getType() const
		{
			return this->m_backend;
		}

		size_t devices() const
		{
			return this->m_devices;
		}

		void setDevices(size_t numDevices)
		{
			this->m_devices = numDevices;
		}


		size_t CPUThreads() const
		{
			return this->m_CPUThreads;
		}

		void setCPUThreads(size_t threads)
		{
			this->m_CPUThreads = threads;
		}

		Backend::Scheduling schedulingMode() const
		{
			return this->m_openmp_scheduling_mode;
		}

		void setSchedulingMode(Backend::Scheduling mode)
		{
			this->m_openmp_scheduling_mode = mode;
		}

		size_t CPUChunkSize() const
		{
			return this->m_openmp_chunk_size;
		}

		void setCPUChunkSize(size_t chunkSize)
		{
			this->m_openmp_chunk_size = chunkSize;
		}


		size_t GPUThreads() const
		{
			return this->m_GPUThreads;
		}

		void setGPUThreads(size_t threads)
		{
			this->m_GPUThreads = threads;
		}


		size_t GPUBlocks() const
		{
			return this->m_blocks;
		}

		void setGPUBlocks(size_t blocks)
		{
			this->m_blocks = blocks;
		}


		/**
		 * The partition ratio for hybrid execution, defining how many percent of the workload should be executed by the CPU.
		 */
		float CPUPartitionRatio() const
		{
			return this->m_cpuPartitionRatio;
		}

		void setCPUPartitionRatio(float percentage)
		{
			assert(percentage >= 0.0f and percentage <= 1.0f);
			this->m_cpuPartitionRatio = percentage;
		}

		Backend::Type type() const
		{
			return (this->m_backend != Backend::Type::Auto) ? this->m_backend : defaultType;
		}

		Backend::Type activateBackend() const
		{
			auto type = this->type();

#ifndef SKEPU_SILENCE_UNAVAILABLE_BACKEND
			if (!Backend::isTypeAvailable(type)) SKEPU_ERROR("Requested backend is not enabled in executable: " << type);
#endif

#ifdef SKEPU_OPENMP
			if (type == Backend::Type::OpenMP || type == Backend::Type::Hybrid)
			{
				DEBUG_TEXT_LEVEL1("Setting OpenMP scheduling mode to " << this->m_openmp_scheduling_mode << " with chunk size " << this->m_openmp_chunk_size);
				omp_set_schedule(Backend::convertSchedulingToOpenMP(this->m_openmp_scheduling_mode), this->m_openmp_chunk_size);
				DEBUG_TEXT_LEVEL1("Setting OpenMP thread count to " << this->CPUThreads());
				omp_set_num_threads(this->CPUThreads());
			}
#endif
			return type;
		}

	private:
		Backend::Type m_backend;

		// GPU parameters
		size_t m_devices {defaultNumDevices};
		size_t m_GPUThreads {defaultGPUThreads};
		size_t m_blocks {defaultGPUBlocks};

		// OpenMP parameters
#ifdef SKEPU_OPENMP
		size_t m_CPUThreads {(size_t)omp_get_max_threads()};
#else
		size_t m_CPUThreads {defaultCPUThreads};
#endif
		Backend::Scheduling m_openmp_scheduling_mode = Backend::Scheduling::Static;
		size_t m_openmp_chunk_size {Backend::chunkSizeDefault};

		// Hybrid parameters
		float m_cpuPartitionRatio {defaultCPUPartitionRatio};

		friend std::ostream &operator<<(std::ostream &o, BackendSpec b);
	};


	inline std::ostream &operator<<(std::ostream &o, BackendSpec b)
	{
		o << "-----------------------------\n SkePU Backend Specification\n-----------------------------\n";
		o << " - Backend Type: " << b.m_backend << "\n";

		if (b.m_backend == Backend::Type::OpenMP)
		{
			o << " - OpenMP Threads:    " << b.m_CPUThreads << "\n";
			o << " - OpenMP Scheduling: " << b.m_openmp_scheduling_mode << "\n";
			o << " - OpenMP Chunk Size: " << b.m_openmp_chunk_size << "\n";
		}
		if (b.m_backend == Backend::Type::OpenCL || b.m_backend == Backend::Type::CUDA || b.m_backend == Backend::Type::Hybrid)
		{
			o << " - GPU Devices: " << b.m_devices << "\n";
			o << " - GPU Blocks:  " << b.m_blocks << "\n";
			o << " - GPU Threads: " << b.m_GPUThreads << "\n";
		}
		if (b.m_backend == Backend::Type::Hybrid)
		{
			o << " - Hybrid Partition Ratio (CPU): " << b.m_cpuPartitionRatio  << "\n";
		}
		o << "-----------------------------";
		return o;
	}

	static const BackendSpec m_defaultGlobalBackendSpec{};

	// Enables global backendspec across multiple translation units
	inline BackendSpec &internalGlobalBackendSpecAccessor()
	{
		static BackendSpec m_globalBackendSpec = m_defaultGlobalBackendSpec;
		return m_globalBackendSpec;
	}

	inline void setGlobalBackendSpec(BackendSpec &spec)
	{
		internalGlobalBackendSpecAccessor() = spec;
	}

	inline void restoreDefaultGlobalBackendSpec()
	{
		internalGlobalBackendSpecAccessor() = m_defaultGlobalBackendSpec;
	}


	/*!
	*  \class ExecPlan
	*
	*  \brief A class that describes an execution plan
	*
	*  This class is used to specifiy execution parameters. For the GPU back ends
	*  you can set both the block size (maxThreads) and the grid size (maxBlocks).
	*  For OpenMP the number of threads is parameterized (numOmpThreads).
	*
	*  It is also possible to specify which backend should be used for a certain
	*  data size. This is done by adding a lowBound and a highBound of data sizes
	*  and a backend that should be used for that range to a list. The skeletons
	*  will use this list when deciding which back end to use.
	*/
	class ExecPlan
	{
	public:
		ExecPlan() : m_calibrated(false)
		{
			this->m_cacheEntry.first = 0;
			cpuModel = nullptr;
			gpuModel = nullptr;
		}

		ExecPlan(skepu::ExecutionTimeModel* _cpuModel, skepu::ExecutionTimeModel* _gpuModel) : cpuModel{_cpuModel}, gpuModel{_gpuModel}, m_calibrated(false)
		{
			this->m_cacheEntry.first = 0;
		}

		~ExecPlan()
		{
			if(cpuModel)
				delete cpuModel;
			if(gpuModel)
				delete gpuModel;
		}

		void add(size_t lowBound, size_t highBound, Backend::Type backend, size_t gs, size_t bs)
		{
			BackendSpec bspec(backend);
			bspec.setGPUThreads(bs);
			bspec.setGPUBlocks(gs);
			this->m_sizePlan.insert(std::make_pair(std::make_pair(lowBound, highBound), bspec));
		}

		void add(size_t lowBound, size_t highBound, Backend::Type backend, size_t numOmpThreads)
		{
			BackendSpec bspec(backend);
			bspec.setCPUThreads(numOmpThreads);
			this->m_sizePlan.insert(std::make_pair(std::make_pair(lowBound, highBound), bspec));
		}

		void add(size_t lowBound, size_t highBound, BackendSpec bspec)
		{
			this->m_sizePlan.insert(std::make_pair(std::make_pair(lowBound, highBound), bspec));
		}

		void add(size_t lowBound, size_t highBound, Backend::Type backend)
		{
			BackendSpec bspec(backend);
			this->m_sizePlan.insert(std::make_pair(std::make_pair(lowBound, highBound), bspec));
		}


		void setCPUThreads(size_t size, size_t maxthreads)
		{
			if (this->m_sizePlan.empty())
				SKEPU_ERROR("Empty execution plan!");

			this->find(size).setCPUThreads(maxthreads);
		}

		size_t CPUThreads(size_t size)
		{
			if (this->m_sizePlan.empty())
				SKEPU_ERROR("Empty execution plan!");

			return this->find(size).CPUThreads();
		}


		void setGPUThreads(size_t size, size_t maxthreads)
		{
			if (this->m_sizePlan.empty())
				SKEPU_ERROR("Empty execution plan!");

			this->find(size).setGPUThreads(maxthreads);
		}

		size_t GPUThreads(size_t size)
		{
			if (this->m_sizePlan.empty())
				SKEPU_ERROR("Empty execution plan!");

			return this->find(size).GPUThreads();
		}


		void setGPUBlocks(size_t size, size_t maxblocks)
		{
			if (this->m_sizePlan.empty())
				SKEPU_ERROR("Empty execution plan!");

			this->find(size).setGPUBlocks(maxblocks);
		}

		size_t GPUblocks(size_t size)
		{
			if (this->m_sizePlan.empty())
				SKEPU_ERROR("Empty execution plan!");

			return this->find(size).GPUBlocks();
		}


		void setDevices(size_t size, size_t maxdevices)
		{
			if (this->m_sizePlan.empty())
				SKEPU_ERROR("Empty execution plan!");

			this->find(size).setDevices(maxdevices);
		}

		size_t devices(size_t size)
		{
			if (this->m_sizePlan.empty())
				SKEPU_ERROR("Empty execution plan!");

			return this->find(size).devices();
		}


		bool isTrainedFor(size_t size)
		{
			if (this->m_sizePlan.empty())
				return false;

			for (auto plan : this->m_sizePlan)
			{
				if (size >= plan.first.first && size <= plan.first.second)
					return true;
			}
			return false;
		}

		bool isCalibrated()
		{
			return this->m_calibrated;
		}

		void setCalibrated()
		{
			this->m_calibrated = true;
		}

		BackendSpec &find(size_t size)
		{
			if (this->m_sizePlan.empty())
				SKEPU_ERROR("Empty execution plan!");

			if (this->m_cacheEntry.first == size)
				return m_cacheEntry.second;

			for (auto plan : this->m_sizePlan)
			{
				if (size >= plan.first.first && size <= plan.first.second)
				{
					if(cpuModel and gpuModel)
					{
						float cpuRatio = ExecutionTimeModel::predictCPUPartitionRatio(*cpuModel, *gpuModel, size);
						plan.second.setCPUPartitionRatio(cpuRatio);
					}

					this->m_cacheEntry = std::make_pair(size, plan.second);
					return this->m_cacheEntry.second;
				}
			}

			if(cpuModel and gpuModel)
			{
				float cpuRatio = ExecutionTimeModel::predictCPUPartitionRatio(*cpuModel, *gpuModel, size);
				this->m_sizePlan.rbegin()->second.setCPUPartitionRatio(cpuRatio);
			}
			this->m_cacheEntry = std::make_pair(size, this->m_sizePlan.rbegin()->second);
			return this->m_cacheEntry.second;
		}

		void clear()
		{
			this->m_sizePlan.clear();
		}

	private:
		std::pair<size_t, BackendSpec> m_cacheEntry;
		std::map<std::pair<size_t, size_t>, BackendSpec> m_sizePlan;

		skepu::ExecutionTimeModel* cpuModel;
		skepu::ExecutionTimeModel* gpuModel;

		/*! boolean field to specify if this exec plan is properly initialized or not */
		bool m_calibrated;
	};

}
