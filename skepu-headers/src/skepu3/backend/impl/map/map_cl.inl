/*! \file map_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the Map skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu
{

	void synchronize()
	{
		for (size_t i = 0; i < skepu::backend::Environment<int>::getInstance()->m_devices_CL.size(); ++i)
		{
			clFinish(skepu::backend::Environment<int>::getInstance()->m_devices_CL[i]->getQueue());
		}
	}

	namespace backend
	{
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::mapNumDevices_CL(size_t startIdx, size_t numDevices, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			constexpr size_t numKernelArgs = numArgs + 3;
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;

			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t baseIndex = startIdx + i * numElemPerSlice;

				pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, numElem * abs(this->m_strides[EI]), device, false), 0)...);
				pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(), get<AI>(std::forward<CallArgs>(args)...).size(), device, false), 0)...);
				pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, numElem * abs(this->m_strides[OI]), device, false), 0)...);
			}

			DEBUG_TEXT_LEVEL1("OpenCL Map: numDevices " << numDevices);
			SKEPU_TRACE_START_EVENT(trace_handle);

			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), numElem);
				const size_t numBlocks = std::min(numElem / numThreads + (numElem % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks());
				const size_t baseIndex = startIdx + i * numElemPerSlice;

				DEBUG_TEXT_LEVEL1("OpenCL Map: label = '" << this->getLabel() << "'  device " << i << ", numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);

				// Copies the elements to the device
				auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, numElem * abs(this->m_strides[EI]), device, true)...);
				auto anyMemP    = std::make_tuple(cl_helpers::randomAccessArg(get<AI>(std::forward<CallArgs>(args)...).getParent(), device, hasReadAccess(MapFunc::anyAccessMode[AI-arity-outArity]))...);
				auto outMemP    = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, numElem * abs(this->m_strides[OI]), device, abs(this->m_strides[OI]) != 1 /* upload if not unit stride */)...);

				size_t threads = std::min<size_t>(size, numBlocks * numThreads);
				auto random = this->template prepareRandom_CL<MapFunc::randomCount>(size, threads);
				auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);

				CLKernel::map(
					i, numThreads, numBlocks * numThreads,
					std::get<OI>(outMemP)...,
					randomMemP,
					std::get<EI-outArity>(elwiseMemP)...,
					std::get<AI-arity-outArity>(anyMemP)...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
					numElem, baseIndex, this->m_strides
				);

				// Make sure the data is marked as changed by the device
				pack_expand((std::get<1>(std::get<AI-arity-outArity>(anyMemP))->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
				pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			}

			SKEPU_TRACE_CALL(trace_handle, "Map", this, {size},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);

			synchronize();
		}


		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::CL(size_t startIdx, size_t size, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL Map: label = '" << this->getLabel() << "'  size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());

			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());

			// Same method is invoked no matter how many GPUs we use
			this->mapNumDevices_CL(startIdx, numDevices, size, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}

	} // namespace backend
} // namespace skepu

#endif
