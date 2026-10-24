/*! \file mapoverlap.h
 *  \brief Contains a class declaration for the MapOverlap skeleton.
 */

#ifndef MAPOVERLAP_PAR_H
#define MAPOVERLAP_PAR_H

#include "skepu3/impl/region.hpp"
#include "skepu3/mapoverlap_base.hpp"

namespace skepu
{

	namespace backend
	{
        template <typename MapOverlapFunc, SkeletonType skeletonType>
        class MapOverlapPar: public impl::MapOverlapBase<typename region_type<typename parameter_type<(MapOverlapFunc::indexed ? 1 : 0) + (MapOverlapFunc::usesPRNG ? 1 : 0), decltype(&MapOverlapFunc::CPU)>::type>::type>
        {
        protected:
            using Ret = typename MapOverlapFunc::Ret;
            static constexpr bool isPool = MapOverlapFunc::isPool;
			using RegionType = typename parameter_type<(MapOverlapFunc::indexed ? 1 : 0) + (MapOverlapFunc::usesPRNG ? 1 : 0), decltype(&MapOverlapFunc::CPU)>::type;
            using T = typename region_type<RegionType>::type;
            using F = ConditionalIndexForwarder<MapOverlapFunc::indexed, MapOverlapFunc::usesPRNG, decltype(&MapOverlapFunc::CPU)>;
            // The parameters of the overlap, for the trace: the overlap, strides and edge, or the window and stride of a pool
            void describeOverlap(tracing::Parameters &parameters, size_t dimensions) const
            {
                int overlap[4], strides[4];
                for (size_t d = 0; d < dimensions; ++d)
                {
                    overlap[d] = this->m_overlap[d];
                    strides[d] = this->m_strides[d];
                }
                if (isPool)
                {
                    parameters.add("pool", true);
                    parameters.add("pool_size", overlap, dimensions);
                    parameters.add("stride", strides, dimensions);
                    return;
                }
                parameters.add("overlap", overlap, dimensions);
                if (std::any_of(strides, strides + dimensions, [](int s) { return s != 1; }))
                    parameters.add("stride", strides, dimensions);
                if (this->m_edge != Edge::Duplicate)     // (the default)
                    parameters.add("edge", to_string(this->m_edge));
                if (this->m_edge == Edge::Pad)
                    parameters.addNumber("pad", this->m_pad);
                if (this->m_updateMode != UpdateMode::Normal)
                    parameters.add("update_mode", to_string(this->m_updateMode));
            }

            // The backend of the call, checked: mirror mode (the edge mode None with an input of the size of the output; only the interior of the
            // output is computed) is not implemented for the Hybrid backend.
            Backend::Type checkMirrorBackend(Backend::Type type, const char *skeleton)
            {
                if (this->m_mirror_mode && type == Backend::Type::Hybrid)
                    SKEPU_ERROR(skeleton << " with the edge mode None and an input of the size of the output (mirror mode) is not implemented for the Hybrid backend");
                return type;
            }

            size_t expectedInputSize(size_t size, size_t dim) const
			{
				if (isPool) return this->m_overlap[dim] + this->m_strides[dim] * (size - 1);
				else if (this->m_edge == Edge::None) return size + 2 * this->m_overlap[dim];
				else return size;
			}

            bool isInputSizeValid(size_t out_size, size_t in_size, size_t dim) const
			{
                // input must be greater than or equal to the area that will be traversed
                // to obtain the outputs.
                if (this->isPool) 
                    return in_size >= this->getSmallestAllowedInputSizePool(out_size, dim);
                
                // input must be sized exactly so that there is no edge handling
                else if (this->m_edge == skepu::Edge::None)
                    return in_size == this->getAllowedInputSizeNone(out_size, dim);

                // difference between in_size and out_size must be even
                return this->isRegularInputSizeValid(in_size, out_size);
			}

			template <typename T, typename U, unsigned N, unsigned M>
			void configurePoolKernelParameters(size_t max_threads_per_block, const std::array<U*, N>& num_threads, const std::array<U*, N>& num_blocks,
											   bool should_use_global_memory, size_t max_shared_memory_per_block, const std::array<size_t*, M>& shared_memory, const std::array<size_t, M>& in_size)
			{
				while (true)
				{
					if (should_use_global_memory)
						for (size_t* shared_memory : shared_memory)
							*shared_memory = 0;
					else if (N == M)
						for (int n = N-1; n >= 0; --n)
							*shared_memory[N-n-1] = std::min(in_size[N-n-1], in_size[N-n-1]/(*num_blocks[n]) + this->m_overlap[n] - this->m_strides[n]);

					U total_threads = 1;
					for (U* numThread : num_threads)
						total_threads *= *numThread;

					size_t total_shared_memory = sizeof(T);
					for (size_t* shared_mem : shared_memory)
						total_shared_memory *= *shared_mem;

					if (total_threads <= max_threads_per_block && total_shared_memory <= max_shared_memory_per_block)
						break;

					bool modifiedSomething = false;
					for (int prime : first_20_primes)
					{
						for (int n = N - 1; n >= 0; --n)
							if (*num_threads[n] % prime == 0)
							{
								*num_threads[n] /= prime;
								*num_blocks[n] *= prime;
								modifiedSomething = true;
								goto configure_pool_kernel_parameters_while_end;
							}
					}
					configure_pool_kernel_parameters_while_end:
					if (!modifiedSomething)
					{
						std::string dim = std::to_string(M);
						SKEPU_ERROR("Could not find a valid setup of threads and blocks when running MapPool" << dim << "D Kernel");
					}
				}
			}

        public:
            using ResultArg = std::tuple<T>;
            using ElwiseArgs = std::tuple<T>;
            using ContainerArgs = typename MapOverlapFunc::ContainerArgs;
            using UniformArgs = typename MapOverlapFunc::UniformArgs;
            static constexpr bool prefers_matrix = false;

            static constexpr size_t InArity = 1;
            static constexpr size_t OutArity = MapOverlapFunc::outArity;
			static constexpr size_t numArgs = MapOverlapFunc::totalArity - (MapOverlapFunc::indexed ? 1 : 0) - (MapOverlapFunc::usesPRNG ? 1 : 0) + OutArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;

            static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
            static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
            static constexpr typename make_pack_indices<InArity + anyArity + OutArity, InArity + OutArity>::type any_indices{};
            static constexpr typename make_pack_indices<numArgs, InArity + anyArity + OutArity>::type const_indices{};

        }; // MapOverlapPar

    } // backend

} // skepu

#endif // MAPOVERLAP_PAR_H