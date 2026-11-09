#ifdef SKEPU_CUDA

#include <cuda.h>

namespace skepu
{
	namespace backend
	{
		template<typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPool1D<MapFunc, CUDAKernel, CLKernel>
		::CUDA(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CUDA MapPool1D: size = " << size << ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			SKEPU_TRACE_CALL_SCOPE_CU(this->m_environment, 1, "MapPool 1D", this, {size},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...),
				this->template randomID<MapFunc::randomCount>());

			if (size == 0)
				return;

			const size_t deviceID = this->m_environment->m_best_cuda_device_id;
			auto aArgs = std::forward_as_tuple(get<AI>(std::forward<CallArgs>(args)...)...);
			auto scArgs = std::forward_as_tuple(get<CI>(std::forward<CallArgs>(args)...)...);
			static constexpr auto proxy_tags = typename MapFunc::ProxyTags{};

			const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), size);
			const size_t numBlocks = std::max<size_t>(1, std::min(size / numThreads + (size % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks()));

			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CU(get<OI>(std::forward<CallArgs>(args)...).getAddress(), size, deviceID, AccessMode::Write)...);
			auto inMemP  = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CU(get<EI>(std::forward<CallArgs>(args)...).getAddress(), inputSize(size), deviceID, AccessMode::Read)...);
			auto anyMemP = std::make_tuple(std::get<AI-1-outArity>(aArgs).getParent().cudaProxy(deviceID, MapFunc::anyAccessMode[AI-1-outArity], std::get<AI-1-outArity>(proxy_tags), Index1D{0})...);

			size_t prng_threads = std::min<size_t>(size, numBlocks * numThreads);
			auto random = this->template prepareRandom<MapFunc::randomCount>(size, prng_threads);
			auto randomMemP = random.updateDevice_CU(random.getAddress(), prng_threads, deviceID, AccessMode::ReadWrite);

			this->m_cuda_kernel<<<numBlocks, numThreads>>>
			(
				std::get<OI>(outMemP)->getDeviceDataPointer()...,
				randomMemP->getDeviceDataPointer(),
				std::get<EI-outArity>(inMemP)->getDeviceDataPointer()...,
				std::get<AI-1-outArity>(anyMemP).second...,
				std::get<CI-1-anyArity-outArity>(scArgs)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_j(),
				get<0>(std::forward<CallArgs>(args)...).getParent().size_k(),
				get<0>(std::forward<CallArgs>(args)...).getParent().size_l(),
				size, 0, this->m_pool_size, this->m_stride
			);

			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			pack_expand((std::get<AI-1-outArity>(anyMemP).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-1-outArity])), 0)...);
		}
	}
}

#endif // SKEPU_CUDA
