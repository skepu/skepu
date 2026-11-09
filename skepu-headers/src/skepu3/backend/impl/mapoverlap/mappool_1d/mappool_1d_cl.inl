#ifdef SKEPU_OPENCL

namespace skepu
{
	namespace backend
	{
		template<typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPool1D<MapFunc, CUDAKernel, CLKernel>
		::CL(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL MapPool1D: size = " << size << ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());

			if (size == 0)
				return;

			Device_CL *device = this->m_environment->m_devices_CL[0];
			const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), size);
			const size_t numBlocks = std::min(size / numThreads + (size % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks());

			auto inMemP  = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress(), inputSize(size), device, true)...);
			auto anyMemP = std::make_tuple(cl_helpers::randomAccessArg(get<AI>(std::forward<CallArgs>(args)...).getParent(), device, hasReadAccess(MapFunc::anyAccessMode[AI-1-outArity]))...);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress(), size, device, false)...);

			size_t threads = std::min<size_t>(size, numBlocks * numThreads);
			auto random = this->template prepareRandom_CL<MapFunc::randomCount>(size, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);

			SKEPU_TRACE_START_EVENT(trace_handle);
			SKEPU_TRACE_DEVICE_BEGIN_CL(trace_handle);

			CLKernel::mapPool(
				nullptr, 0, numThreads, numBlocks * numThreads,
				std::get<OI>(outMemP)...,
				randomMemP,
				std::get<EI-outArity>(inMemP)...,
				std::get<AI-1-outArity>(anyMemP)...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
				size, iterator_index_base(get<0>(std::forward<CallArgs>(args)...)), this->m_pool_size, this->m_stride
			);

			pack_expand((std::get<1>(std::get<AI-1-outArity>(anyMemP))->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-1-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);

			SKEPU_TRACE_DEVICE_END_CL(trace_handle);
			SKEPU_TRACE_CALL(trace_handle, "MapPool 1D", this, {size},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...),
				this->template randomID<MapFunc::randomCount>()
			);
		}
	}
}

#endif // SKEPU_OPENCL
