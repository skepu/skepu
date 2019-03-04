#pragma once

#include <skepu2/backend/benchmark.h>
#include <skepu2/backend/tuner.h>

#define HYBRID_MEASURE_REPEATS 5
#define HYBRID_ARG_NUM_STEPS 10
#define HYBRID_ARG_SIZE_MIN 1024
#define HYBRID_ARG_SIZE_MAX (1 << 20)

namespace skepu2
{	
	namespace backend
	{
		namespace tuner
		{
			
#if defined(SKEPU_PRECOMPILED) && defined(SKEPU_HYBRID)
			
			template<typename Skeleton, size_t... ResultIdx, size_t... ElwiseIdx, size_t... ContainerIdx, size_t... UniformIdx>
			void hybrid_tune_impl(Skeleton& instance, const int numThreads, const size_t numDevices, const size_t min, const size_t max, const size_t steps, const size_t repeats,
				future_std::index_sequence<ResultIdx...>,    future_std::index_sequence<ElwiseIdx...>,
				future_std::index_sequence<ContainerIdx...>, future_std::index_sequence<UniformIdx...>)
			{
				std::cout << "Start hybrid tuning." << std::endl;
				// Tuple with container for the output container (can be empty)
				typename select_if<Skeleton::prefers_matrix,
					typename add_container_layer<Matrix, typename Skeleton::ResultArg>::type,
					typename add_container_layer<Vector, typename Skeleton::ResultArg>::type>::type resultArg;
				
				// Tuple with containers for the element-wise arguments (can be empty)
				typename select_if<Skeleton::prefers_matrix,
					typename add_container_layer<Matrix, typename Skeleton::ElwiseArgs>::type,
					typename add_container_layer<Vector, typename Skeleton::ElwiseArgs>::type>::type elwiseArgs;
				
				// Tuple with containers for the random-access container arguments (can be different container types, can be empty)
				typename container_args_tuple_transformer<typename Skeleton::ContainerArgs>::type containerArgs;
				
				// Tuple with uniform arguments (can be empty)
				typename Skeleton::UniformArgs uniformArgs;
				
				// Reserve memory for the maximum input size in advance
				reserve_all_in_tuple(resultArg,  max);
				reserve_all_in_tuple(elwiseArgs, max);
				
				ExecutionTimeModel* cpuModel = new ExecutionTimeModel();
				ExecutionTimeModel* gpuModel = new ExecutionTimeModel();
				
				// Run tests for all input sizes
				const size_t stepSize = (max - min) / (steps-1);
				for (size_t i = min, prev_i = 0; i <= max; prev_i = i, i += stepSize)
				{
					std::cout << "Make measurement on size: " << i << std::endl;
					resize_all_in_tuple(resultArg,  i, instance, true);
					resize_all_in_tuple(elwiseArgs, i, instance);
					resize_all_in_tuple(containerArgs, i, instance);
					
					// TODO: Set default size on Map and MapReduce skeletons only
// 					instance.setDefaultSize(i);
					
					BackendSpec cpuSpec(skepu2::Backend::Type::OpenMP);
					cpuSpec.setCPUThreads(numThreads-1); // One will be used for GPU
					instance.setBackend(cpuSpec);
					
					auto durationCPU = benchmark::basicBenchmark(
						repeats, i,
						[&] (size_t) {
							instance(
								std::get<ResultIdx>(resultArg)...,
								std::get<ElwiseIdx>(elwiseArgs)...,
								std::get<ContainerIdx>(containerArgs)...,
								std::get<UniformIdx>(uniformArgs)...);
						},
						[&](benchmark::TimeSpan duration)
						{
							skepu2::containerutils::updateHostAndInvalidateDevice(
								std::get<ResultIdx>(resultArg)...,
								std::get<ElwiseIdx>(elwiseArgs)...,
								std::get<ContainerIdx>(containerArgs)...);
						}
					);
					double timeCPU = (double) durationCPU.count();
					cpuModel->addDataPoint(i, timeCPU);
					std::cout << "    CPU time was: " << timeCPU << std::endl;
					
#ifdef SKEPU_HYBRID_USE_CUDA
					BackendSpec gpuSpec(skepu2::Backend::Type::CUDA);
#else
					BackendSpec gpuSpec(skepu2::Backend::Type::OpenCL);
#endif
					gpuSpec.setDevices(numDevices);
					instance.setBackend(gpuSpec);
						
					auto durationGPU = benchmark::basicBenchmark(
						MEASURE_REPEATS, i,
						[&] (size_t) {
							instance(
								std::get<ResultIdx>(resultArg)...,
								std::get<ElwiseIdx>(elwiseArgs)...,
								std::get<ContainerIdx>(containerArgs)...,
								std::get<UniformIdx>(uniformArgs)...);
						},
						[&](benchmark::TimeSpan duration)
						{
							skepu2::containerutils::updateHostAndInvalidateDevice(
								std::get<ResultIdx>(resultArg)...,
								std::get<ElwiseIdx>(elwiseArgs)...,
								std::get<ContainerIdx>(containerArgs)...);
						}
					);
					double timeGPU = (double) durationGPU.count();
					gpuModel->addDataPoint(i, timeGPU);
					std::cout << "    GPU time was: " << timeGPU << std::endl;
				}
				
				cpuModel->fitModel();
				gpuModel->fitModel();
				
				std::cout << "CPU model: " << cpuModel->getA() << "x + " << cpuModel->getB() << std::endl; 
				std::cout << "GPU model: " << gpuModel->getA() << "x + " << gpuModel->getB() << std::endl; 
				std::cout << "CPU R2 error: " << cpuModel->getR2Error() << std::endl;
				std::cout << "GPU R2 error: " << gpuModel->getR2Error() << std::endl;
				
				// The plan which will be generated
				BackendSpec bspec(skepu2::Backend::Type::Hybrid);
				bspec.setCPUThreads(numThreads);
				bspec.setDevices(numDevices);
				ExecPlan *plan = new ExecPlan(cpuModel, gpuModel);
				plan->setCalibrated();
				plan->add(0, HYBRID_ARG_SIZE_MAX, bspec);
				
				instance.setExecPlan(std::move(plan));
			}
			
			
			template<typename Skeleton>
			void hybridTune(Skeleton& instance, size_t numThreads = CPU_THREADS, size_t numDevices = 1, size_t minSize = HYBRID_ARG_SIZE_MIN, size_t maxSize = HYBRID_ARG_SIZE_MAX, size_t numSteps = HYBRID_ARG_NUM_STEPS)
			{
				hybrid_tune_impl(instance, numThreads, numDevices, minSize, maxSize, numSteps, HYBRID_MEASURE_REPEATS,
					typename future_std::make_index_sequence<std::tuple_size<typename Skeleton::ResultArg>::value>::type(),
					typename future_std::make_index_sequence<std::tuple_size<typename Skeleton::ElwiseArgs>::value>::type(),
					typename future_std::make_index_sequence<std::tuple_size<typename Skeleton::ContainerArgs>::value>::type(),
					typename future_std::make_index_sequence<std::tuple_size<typename Skeleton::UniformArgs>::value>::type());
			}
			
#else
			
			template<typename S>
			void hybridTune(S& instance, size_t numThreads = CPU_THREADS, size_t numDevices = 1, size_t minSize = HYBRID_ARG_SIZE_MIN, size_t maxSize = HYBRID_ARG_SIZE_MAX)
			{
				// Do nothing
			}
			
#endif
		}
	}
}
