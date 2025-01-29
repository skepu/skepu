/*! \file scan_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the Scan skeleton.
 */

namespace skepu
{
	namespace backend
	{
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		template<typename OutIterator, typename InIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		::CPU(size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial)
		{
			// Make sure we are properly synched with device data
			res.getParent().invalidateDeviceData();
			arg.getParent().updateHost();
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			// Process first element here
			*res = (mode == ScanMode::Inclusive) ? *arg++ : initial;
			
			for (size_t i = 1; i < size; ++i)
				res(i) = ScanFunc::CPU(res(i-1), arg(i-1));
			
			SKEPU_TRACE_CALL(trace_handle, "Scan", this, {size}, {res.getParent().getObjectID()}, {arg.getParent().getObjectID()}, {}, {});
		}
		
	}
}
