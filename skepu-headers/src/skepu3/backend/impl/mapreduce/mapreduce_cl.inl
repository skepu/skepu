/*! \file mapreduce_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the MapReduce skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu
{
	namespace backend
	{
		template<typename T>
		DeviceMemPointer_CL<T> makeTempReduceMemPointer(T *res, size_t numBlocks, Device_CL *device)
		{
			return DeviceMemPointer_CL<T>(res, numBlocks, device);
		}


		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		scalar_wrapper_t<typename MapFunc::Ret> MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::mapReduceSingle_CL(size_t deviceID, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			Ret startValue = res;
			auto aArgs  = std::forward_as_tuple(get<AI>(std::forward<CallArgs>(args)...)...);
			
			const size_t numThreads = this->m_selected_spec->GPUThreads();
			const size_t numBlocks = std::max<size_t>(1, std::min(size / numThreads, this->m_selected_spec->GPUBlocks()));
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			
			DEBUG_TEXT_LEVEL1("OpenCL MapReduce: numThreads = " << numThreads << ", numBlocks = " << numBlocks);
			
			// Copies the elements to the device
			auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress() + startIdx, size * abs(this->m_strides[EI]), device, true)...);
			auto anyMemP    = std::make_tuple(std::get<AI-arity>(aArgs).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).size(), device, hasReadAccess(MapFunc::anyAccessMode[AI-arity]))...);
			
			// Create the output memory
		//	auto outMemP = std::make_tuple(makeTempReduceMemPointer(&get_or_return<OI>(res), numBlocks, device)...);
			auto outMemP = std::make_tuple(
				new DeviceMemPointer_CL<typename return_type_index<typename MapFunc::Ret, OI>::type>(&get_or_return<OI>(res), numBlocks, device
			)...);
#ifdef SKEPU_TRACING
			std::get<0>(outMemP)->m_label = "Internal (reduce-temp)";
			std::get<0>(outMemP)->m_container_id = UniqueIdentifier::generate();
#endif
			
			size_t threads = std::min<size_t>(size, numBlocks * numThreads);
			auto random = this->template prepareRandom_CL<MapFunc::randomCount>(size, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			CLKernel::mapReduce (
				deviceID, numThreads, numBlocks * numThreads,
				std::get<OI>(outMemP)...,
				randomMemP,
				device_mem_pointer_const_cast(std::get<EI>(elwiseMemP))...,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				size_info(defaultDim{}, this->default_size_i, this->default_size_j, this->default_size_k, this->default_size_l, get<EI>(std::forward<CallArgs>(args)...)...),
				size, startIdx, this->m_strides,
				sizeof(Ret) * numThreads
			);
			
			CLKernel::reduceOnly (
				deviceID, numThreads, numThreads,
				std::get<OI>(outMemP)...,
			//	std::get<OI>(outMemP)...,
				numBlocks, sizeof(Ret) * numThreads
			);
			
			// Copy back result
			pack_expand((std::get<AI-arity>(anyMemP)->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			pack_expand((std::get<OI>(outMemP)->copyDeviceToHost(1), 0)...);
			pack_expand((delete std::get<OI>(outMemP), 0)...);

		//	return ReduceFunc::CPU(res, startValue);
			pack_expand((get_or_return<OI>(res) = ReduceFunc::CPU(get_or_return<OI>(res), get_or_return<OI>(startValue)), 0)...);
			
			auto result = scalar_wrapper_t<typename MapFunc::Ret>(res);
			SKEPU_TRACE_CALL(trace_handle, "MapReduce", this, {size}, tracing::scalar_output_labels(result),
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
			return result;
		}
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		scalar_wrapper_t<typename MapFunc::Ret> MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::mapReduceNumDevices_CL(size_t numDevices, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			auto aArgs  = std::forward_as_tuple(get<AI>(std::forward<CallArgs>(args)...)...);
			
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			Ret result[numDevices];//[sizeof...(OI)];
			std::vector<decltype(std::make_tuple(makeTempReduceMemPointer(&get_or_return<OI>(res), 0, nullptr)...))> outMemP;
			SKEPU_TRACE_START_EVENT(trace_handle);

			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + (i == numDevices - 1 ? rest : 0);
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const size_t numBlocks = std::max<size_t>(1, std::min(numElem / numThreads, this->m_selected_spec->GPUBlocks()));
				const size_t baseIndex = startIdx + i * numElemPerSlice;
				
				DEBUG_TEXT_LEVEL1("OpenCL MapReduce: device = " << i << ", numThreads = " << numThreads << ", numBlocks = " << numBlocks);
				
				// Copies the elements to the device
				auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, numElem * abs(this->m_strides[EI]), device, true)...);
				auto anyMemP    = std::make_tuple(std::get<AI-arity>(aArgs).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
					get<AI>(std::forward<CallArgs>(args)...).size(), device, hasReadAccess(MapFunc::anyAccessMode[AI-arity]))...);
			
				// Create the output memory
		//		outMemP.emplace_back(&result[i], numBlocks, device);
				pack_expand((outMemP.push_back(std::make_tuple(makeTempReduceMemPointer(&get_or_return<OI>(result[i]), numBlocks, device)...)), 0));

				size_t threads = std::min<size_t>(size, numBlocks * numThreads);
				auto random = this->template prepareRandom_CL<MapFunc::randomCount>(size, threads);
				auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
				
				CLKernel::mapReduce (
					i, numThreads, numBlocks * numThreads,
					&std::get<OI>(outMemP[i])...,
					randomMemP,
					device_mem_pointer_const_cast(std::get<EI>(elwiseMemP))...,
					std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity>(anyMemP))...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					size_info(defaultDim{}, this->default_size_i, this->default_size_j, this->default_size_k, this->default_size_l, get<EI>(std::forward<CallArgs>(args)...)...),
					numElem, baseIndex, this->m_strides,
					sizeof(Ret) * numThreads
				);
				
				CLKernel::reduceOnly (
					i, numThreads, numThreads,
					&std::get<OI>(outMemP[i])...,
				//	&std::get<OI>(outMemP[i])...,
					numBlocks,
					sizeof(Ret) * numThreads
				);
				
				// Copy back result
				pack_expand((std::get<AI-arity>(anyMemP)->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
				pack_expand((std::get<OI>(outMemP[i]).changeDeviceData(), 0)...);
			}
			
			// Reduces results from each device on the CPU to yield the total result.
			for (size_t i = 0; i < numDevices; ++i)
			{
				pack_expand((std::get<OI>(outMemP[i]).copyDeviceToHost(1), 0)...);
			//	res = ReduceFunc::CPU(res, result[i]);
				pack_expand((get_or_return<OI>(res) = ReduceFunc::CPU(get_or_return<OI>(res), get_or_return<OI>(result[i])), 0)...);
			}

			auto result_w = scalar_wrapper_t<typename MapFunc::Ret>(res);
			SKEPU_TRACE_CALL(trace_handle, "MapReduce", this, {size}, tracing::scalar_output_labels(result_w),
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
			return result_w;
		}
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		scalar_wrapper_t<typename MapFunc::Ret> MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::CL(size_t startIdx, size_t size, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL MapReduce: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return mapReduceSingle_CL(0, startIdx, size, oi, ei, ai, ci, res, std::forward<CallArgs>(args)...);

#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL

			return mapReduceNumDevices_CL(numDevices, startIdx, size, oi, ei, ai, ci, res, std::forward<CallArgs>(args)...);
		}
		
	} // end namespace backend
} // end namespace skepu

#endif
