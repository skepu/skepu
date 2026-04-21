/*! \file device_cu.h
 *  \brief Contains a class declaration for the object that represents a CUDA device.
 */

#ifndef DEVICE_CU_H
#define DEVICE_CU_H

#ifdef SKEPU_CUDA

#include <iostream>
#include <cuda.h>

#include "globals.h"



namespace skepu
{
namespace backend
{

/*!
 *  \ingroup helpers
 */

/*!
 *  \class Device_CU
 *
 *  \brief A class representing a CUDA device.
 *
 *  This class represents one CUDA device which can be used to execute the skeletons on if CUDA
 *  is used as backend. Stores various properties about the device and provides functions that return them.
 */
class Device_CU
{

public:
   cudaStream_t m_streams[MAX_POSSIBLE_CUDA_STREAMS_PER_GPU];

private:
   unsigned int m_deviceID;
   cudaDeviceProp m_deviceProp;
   size_t m_maxThreads;
   size_t m_maxBlocks;

   // Check out commit 8e4a4e2560c2f8ad32c0cfb12b625f564b659418
   // for what this was previously doing. To put it simply,
   // we used a manual hash table using major+minor version as key
   // and the number of concurrent kernels support as value.
   // Modern Nvidia GPUs have settled on a limit of 32 so we have
   // hardcoded that value instead of extending the table whenever
   // new versions are released.
   const unsigned int m_num_concurrent_kernels_supported = 32;

   /*!
    *  Run once during construction to get all the device properties.
    *
    *  \param device Integer specifying the device to fetch properties for.
    */
   void initDeviceProps(unsigned int device)
   {
      cudaError_t err;
      err = cudaGetDeviceProperties(&m_deviceProp, device);
      if (err != cudaSuccess)
      {
         SKEPU_ERROR("getDeviceProps failed!\n");
      }
   }

public:

   /*!
    *  The constructor creates a device from an ID and gets all its properties.
    *
    *  \param id Device ID for the device that is to be created.
    */
   Device_CU(unsigned int id)
   {
      m_deviceID = id;

      cudaSetDevice(m_deviceID);

      initDeviceProps(id);

#ifdef USE_PINNED_MEMORY
      for(unsigned int i = 0; i < m_num_concurrent_kernels_supported; i++)
            cudaStreamCreate(&(m_streams[i]));
#endif

      if(m_deviceProp.major == 1 && m_deviceProp.minor < 2)
      {
         m_maxThreads = 256;
      }
      else
      {
         m_maxThreads = m_deviceProp.maxThreadsPerBlock;
      }

      m_maxBlocks = m_deviceProp.maxGridSize[0];
   }

   /*!
    * \brief The destructor.
    */
   ~Device_CU()
   {
      // Explicitly destroys and cleans up all resources associated with the current device in the current process.
      // Any subsequent API call to this device will reinitialize the device.
      cudaSetDevice(m_deviceID);
      cudaDeviceReset();
   };


   /*!
    * Returns whether the device supports overlap (memory,kernel) operation or not
    */
   /*bool isOverlapSupported()
   {
      return m_deviceProp.deviceOverlap;
   }*/

   /*!
    *  \return The maximum block size.
    */
   size_t getMaxBlockSize() const
   {
      return m_deviceProp.maxThreadsPerBlock;
   }

   /*!
    *  \return The major version.
    */
   int getMajorVersion() const
   {
      return m_deviceProp.major;
   }

   /*!
    *  \return The minor version.
    */
   int getMinorVersion() const
   {
      return m_deviceProp.minor;
   }

   /*!
    *  \return The name of current GPU.
    */
   std::string getDeviceName() const
   {
      return m_deviceProp.name;
   }

   /*!
    * test
    * 
    *  \return The clock rate of current GPU.
    */
   /*int getClockRate() const
   {
      return m_deviceProp.clockRate;
   }*/

   /*!
    *  \return The integer specifying whether the overlap is support between pinned memory transfer and kernel launches (value>0 if supported)
    *          and/or between pinned memory HTD and DTH (value=2 if supported).
    */
   int getAsyncEngineCount() const
   {
      return m_deviceProp.asyncEngineCount;
   }

   /*!
    *  \return The boolean specifying whether the concurrent kernels are supported on this GPU or not. If supported maximum kernels count is 16.
    */
   bool IsConcurrentKernels() const
   {
      return m_deviceProp.concurrentKernels;
   }

   /*!
    *  \return The integer specifying the max number of concurrent kernels supported on this GPU (if not supprted then return 1).
    */
   unsigned int getNoConcurrentKernels() const
   {
        return m_num_concurrent_kernels_supported;
   }

   /*!
    *  \return The maximum number of compute units available.
    */
   int getNumComputeUnits() const
   {
      return m_deviceProp.multiProcessorCount;
   }

   /*!
    *  \return The global memory size.
    */
   size_t getGlobalMemSize() const
   {
      return m_deviceProp.totalGlobalMem;
   }

   /*!
    *  \return The shared memory size.
    */
   size_t getSharedMemPerBlock() const
   {
      return m_deviceProp.sharedMemPerBlock;
   }

   /*!
    *  \return The maximum number of threads per block or group.
    */
   size_t getMaxThreads() const
   {
#ifdef SKEPU_MAX_GPU_THREADS
      return SKEPU_MAX_GPU_THREADS;
#else
      return m_maxThreads;
#endif
   }

   /*!
    *  \return The maximum number of blocks or groups for a kernel launch.
    */
   size_t getMaxBlocks() const
   {
#ifdef SKEPU_MAX_GPU_BLOCKS
      return SKEPU_MAX_GPU_BLOCKS;
#else
      return m_maxBlocks;
#endif
   }

   /*!
    *  \return Device ID.
    */
   unsigned int getDeviceID() const
   {
      return m_deviceID;
   }
};

}
}

#endif

#endif


