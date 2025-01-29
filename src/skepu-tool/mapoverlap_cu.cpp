#include <algorithm>

#include "code_gen.h"
#include "code_gen_cu.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

/*!
 *
 *  OpenCL MapOverlap kernel for vector. It uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static const std::string MapOverlapKernel_CU = R"~~~(
__global__ void {{KERNEL_NAME}}_MapOverlapKernel_CU({{KERNEL_PARAMS}}
	{{MAPOVERLAP_INPUT_TYPE}}* wrap, size_t n,
	size_t out_offset, size_t out_numelements,
	skepu::Edge edgeMode, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap
)
{
   extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];
   size_t skepu_tid = threadIdx.x;
   size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
 	size_t skepu_global_prng_id = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   while (skepu_i < n + overlap)
   {
      //Copy data to shared memory
      if (edgeMode == skepu::Edge::Pad || edgeMode == skepu::Edge::None)
      {
         {{SHARED_BUFFER}}[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[skepu_i] : pad;

         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = (skepu_i<overlap) ? pad : skepu_input[skepu_i-overlap];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = (skepu_i+overlap < n) ? skepu_input[skepu_i+overlap] : pad;
         }
      }
      else if (edgeMode == skepu::Edge::Cyclic)
      {
         if (skepu_i < n)
         {
            {{SHARED_BUFFER}}[overlap+skepu_tid] = skepu_input[skepu_i];
         }
         else
         {
            {{SHARED_BUFFER}}[overlap + skepu_tid] = wrap[overlap + (skepu_i - n)];
         }

         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = (skepu_i < overlap) ? wrap[skepu_tid] : skepu_input[skepu_i - overlap];
         }

         if (skepu_tid >= (blockDim.x - overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid + 2 * overlap] = (skepu_i + overlap < n) ? skepu_input[skepu_i + overlap] : wrap[overlap + (skepu_i + overlap - n)];
         }
      }
      else if (edgeMode == skepu::Edge::Duplicate)
      {
         {{SHARED_BUFFER}}[overlap + skepu_tid] = (skepu_i < n) ? skepu_input[skepu_i] : skepu_input[n - 1];

         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = (skepu_i < overlap) ? skepu_input[0] : skepu_input[skepu_i - overlap];
         }

         if (skepu_tid >= (blockDim.x - overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = (skepu_i + overlap < n) ? skepu_input[skepu_i + overlap] : skepu_input[n - 1];
         }
      }

      __syncthreads();
			
			bool edgeModeNoneCheck = (edgeMode != skepu::Edge::None) ? true : (skepu_i >= out_offset + overlap) && (skepu_i < out_offset + out_numelements - overlap);

      //Compute and store data
      if ( (skepu_i >= out_offset) && (skepu_i < out_offset + out_numelements) && edgeModeNoneCheck )
			{
				skepu_i = skepu_i - out_offset;
				const size_t skepu_base = 0;
				{{INDEX_INITIALIZER}}
				{{PROXIES_UPDATE}}
				auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
				{{OUTPUT_BINDINGS}}
			}

      skepu_i += gridSize;

      __syncthreads();
   }
}
)~~~";


/*!
 *
 *  OpenCL MapOverlap kernel for applying row-wise overlap on matrix operands. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static const std::string MapOverlapKernel_CU_Matrix_Row = R"~~~(
__global__ void {{KERNEL_NAME}}_MapOverlapKernel_CU_Matrix_Row({{KERNEL_PARAMS}}
	{{MAPOVERLAP_INPUT_TYPE}}* wrap, size_t n, size_t out_offset, size_t out_numelements,
	skepu::Edge edgeMode, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap, size_t blocksPerRow, size_t rowWidth
)
{
   extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];
   size_t skepu_tid = threadIdx.x;
   size_t skepu_i = blockIdx.x * blockDim.x + skepu_tid;
 	size_t skepu_global_prng_id = blockIdx.x * blockDim.x + threadIdx.x;

   size_t wrapIndex = 2 * overlap * (int)(blockIdx.x / blocksPerRow);
   size_t tmp = (blockIdx.x % blocksPerRow);
   size_t tmp2 = (blockIdx.x / blocksPerRow);


   // Copy data to shared memory
   if (edgeMode == skepu::Edge::Pad || edgeMode == skepu::Edge::None)
   {
      {{SHARED_BUFFER}}[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[skepu_i] : pad;

      if (skepu_tid < overlap)
      {
         {{SHARED_BUFFER}}[skepu_tid] = (tmp==0) ? pad : skepu_input[skepu_i-overlap];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         {{SHARED_BUFFER}}[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x - 1 && (skepu_i+overlap < n) && tmp != (blocksPerRow - 1)) ? skepu_input[skepu_i + overlap] : pad;
      }
   }
   else if (edgeMode == skepu::Edge::Cyclic)
   {
      if (skepu_i < n)
      {
         {{SHARED_BUFFER}}[overlap + skepu_tid] = skepu_input[skepu_i];
      }
      else if (skepu_i - n < overlap)
      {
         {{SHARED_BUFFER}}[overlap + skepu_tid] = wrap[(overlap + (skepu_i - n)) + wrapIndex];
      }
      else
      {
         {{SHARED_BUFFER}}[overlap + skepu_tid] = pad;
      }

      if (skepu_tid < overlap)
      {
         {{SHARED_BUFFER}}[skepu_tid] = (tmp == 0) ? wrap[skepu_tid + wrapIndex] : skepu_input[skepu_i - overlap];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         {{SHARED_BUFFER}}[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && skepu_i+overlap < n && tmp!=(blocksPerRow-1)) ? skepu_input[skepu_i+overlap] : wrap[overlap+wrapIndex+(skepu_tid+overlap-blockDim.x)];
      }
   }
   else if (edgeMode == skepu::Edge::Duplicate)
   {
      {{SHARED_BUFFER}}[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[skepu_i] : skepu_input[n-1];

      if (skepu_tid < overlap)
      {
         {{SHARED_BUFFER}}[skepu_tid] = (tmp==0) ? skepu_input[tmp2*rowWidth] : skepu_input[skepu_i-overlap];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         {{SHARED_BUFFER}}[skepu_tid + 2 * overlap] = (blockIdx.x != gridDim.x-1 && (skepu_i + overlap < n) && (tmp != (blocksPerRow - 1))) ? skepu_input[skepu_i + overlap] : skepu_input[(tmp2 + 1) * rowWidth - 1];
      }
   }

   __syncthreads();

  //Compute and store data
  if ( (skepu_i >= out_offset) && (skepu_i < out_offset + out_numelements) )
	{
  //	skepu_output[skepu_i - out_offset] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		skepu_i = skepu_i - out_offset;
		const size_t skepu_base = 0;
		const size_t saved_skepu_i = skepu_i;
		skepu_i = skepu_i % rowWidth;
		{{INDEX_INITIALIZER}}
		{{PROXIES_UPDATE}}
		skepu_i = saved_skepu_i;
		auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_BINDINGS}}
	}
}
)~~~";



/*!
 *
 *  OpenCL MapOverlap kernel for applying column-wise overlap on matrix operands. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static const std::string MapOverlapKernel_CU_Matrix_Col = R"~~~(
__global__ void {{KERNEL_NAME}}_MapOverlapKernel_CU_Matrix_Col({{KERNEL_PARAMS}}
	{{MAPOVERLAP_INPUT_TYPE}}* wrap, size_t n, size_t out_offset, size_t out_numelements,
	skepu::Edge edgeMode, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap, size_t blocksPerCol, size_t rowWidth, size_t colWidth
)
{
   extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];
   size_t skepu_tid = threadIdx.x;
   size_t skepu_i = blockIdx.x * blockDim.x + skepu_tid;
 	size_t skepu_global_prng_id = blockIdx.x * blockDim.x + threadIdx.x;

   size_t wrapIndex= 2 * overlap * (int)(blockIdx.x/blocksPerCol);
   size_t tmp= (blockIdx.x % blocksPerCol);
   size_t tmp2= (blockIdx.x / blocksPerCol);

   size_t arrInd = (threadIdx.x + tmp*blockDim.x)*rowWidth + ((blockIdx.x)/blocksPerCol);

   //Copy data to shared memory
   if (edgeMode == skepu::Edge::Pad || edgeMode == skepu::Edge::None)
   {
      {{SHARED_BUFFER}}[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[arrInd] : pad;

      if (skepu_tid < overlap)
      {
         {{SHARED_BUFFER}}[skepu_tid] = (tmp==0) ? pad : skepu_input[(arrInd-(overlap*rowWidth))];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         {{SHARED_BUFFER}}[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+(overlap*rowWidth))] : pad;
      }
   }
   else if (edgeMode == skepu::Edge::Cyclic)
   {
      if (skepu_i < n)
      {
         {{SHARED_BUFFER}}[overlap+skepu_tid] = skepu_input[arrInd];
      }
      else if (skepu_i-n < overlap)
      {
         {{SHARED_BUFFER}}[overlap+skepu_tid] = wrap[(overlap+(skepu_i-n))+ wrapIndex];
      }
      else
      {
         {{SHARED_BUFFER}}[overlap+skepu_tid] = pad;
      }

      if (skepu_tid < overlap)
      {
         {{SHARED_BUFFER}}[skepu_tid] = (tmp==0) ? wrap[skepu_tid+wrapIndex] : skepu_input[(arrInd-(overlap*rowWidth))];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         {{SHARED_BUFFER}}[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+(overlap*rowWidth))] : wrap[overlap+wrapIndex+(skepu_tid+overlap-blockDim.x)];
      }
   }
   else if (edgeMode == skepu::Edge::Duplicate)
   {
      {{SHARED_BUFFER}}[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[arrInd] : skepu_input[n-1];

      if (skepu_tid < overlap)
      {
         {{SHARED_BUFFER}}[skepu_tid] = (tmp==0) ? skepu_input[tmp2] : skepu_input[(arrInd-(overlap*rowWidth))];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         {{SHARED_BUFFER}}[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+(overlap*rowWidth))] : skepu_input[tmp2+(colWidth-1)*rowWidth];
      }
   }

   __syncthreads();

	//Compute and store data
	if ( (arrInd >= out_offset) && (arrInd < out_offset+out_numelements) )
	{
		//  skepu_output[arrInd-out_offset] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		size_t skepu_i = arrInd - out_offset;
		const size_t skepu_base = 0;
		const size_t saved_skepu_i = skepu_i;
		skepu_i = skepu_i / rowWidth;
		{{INDEX_INITIALIZER}}
		{{PROXIES_UPDATE}}
		skepu_i = saved_skepu_i;
		auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_BINDINGS}}
	}
}
)~~~";



/*!
 *
 *  OpenCL MapOverlap kernel for applying column-wise overlap on matrix operands when using multiple GPUs. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static const std::string MapOverlapKernel_CU_Matrix_ColMulti = R"~~~(
__global__ void {{KERNEL_NAME}}_MapOverlapKernel_CU_Matrix_ColMulti({{KERNEL_PARAMS}}
	{{MAPOVERLAP_INPUT_TYPE}}* wrap, size_t n, size_t in_offset, size_t out_numelements,
	skepu::Edge edgeMode, int deviceType, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap, size_t blocksPerCol, size_t rowWidth, size_t colWidth
)
{
   extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];

   size_t skepu_tid = threadIdx.x;
   size_t skepu_i = blockIdx.x * blockDim.x + skepu_tid;
 	 size_t skepu_global_prng_id = blockIdx.x * blockDim.x + threadIdx.x;

   size_t tmp= (blockIdx.x % blocksPerCol);
   size_t tmp2= (blockIdx.x / blocksPerCol);

   size_t arrInd = (threadIdx.x + tmp*blockDim.x)*rowWidth + tmp2; //((blockIdx.x)/blocksPerCol);

   if (edgeMode == skepu::Edge::Pad || edgeMode == skepu::Edge::None)
   {
      {{SHARED_BUFFER}}[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[arrInd+in_offset] : pad; // in_offset

      if (deviceType == -1) // first device, i.e. in_offset=0
      {
         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = (tmp==0) ? pad : skepu_input[(arrInd-(overlap*rowWidth))];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 0) // middle device
      {
         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 1) // last device
      {
         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+in_offset+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+in_offset+(overlap*rowWidth))] : pad;
         }
      }
   }
   else if (edgeMode == skepu::Edge::Cyclic)
   {
      {{SHARED_BUFFER}}[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[arrInd+in_offset] : ((skepu_i-n < overlap) ? wrap[(skepu_i-n)+ (overlap * tmp2)] : pad);

      if (deviceType == -1) // first device, i.e. in_offset=0
      {
         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = (tmp==0) ? wrap[skepu_tid+(overlap * tmp2)] : skepu_input[(arrInd-(overlap*rowWidth))];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 0) // middle device
      {
         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 1) // last device
      {
         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+in_offset+(overlap*rowWidth))] : wrap[(overlap * tmp2)+(skepu_tid+overlap-blockDim.x)];
         }
      }
   }
   else if (edgeMode == skepu::Edge::Duplicate)
   {
      {{SHARED_BUFFER}}[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[arrInd+in_offset] : skepu_input[n+in_offset-1];

      if (deviceType == -1) // first device, i.e. in_offset=0
      {
         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = (tmp==0) ? skepu_input[tmp2] : skepu_input[(arrInd-(overlap*rowWidth))];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 0) // middle device
      {
         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 1) // last device
      {
         if (skepu_tid < overlap)
         {
            {{SHARED_BUFFER}}[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            {{SHARED_BUFFER}}[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+in_offset+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+in_offset+(overlap*rowWidth))] : skepu_input[tmp2+in_offset+(colWidth-1)*rowWidth];
         }
      }
   }

   __syncthreads();

	// Compute and store data
	if ( arrInd < out_numelements )
	{
	//	skepu_output[arrInd] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		size_t skepu_i = arrInd;
		const size_t skepu_base = 0;
		const size_t saved_skepu_i = skepu_i;
		skepu_i = skepu_i / rowWidth;
		{{INDEX_INITIALIZER}}
		{{PROXIES_UPDATE}}
		skepu_i = saved_skepu_i;
		auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_BINDINGS}}
	}
}
)~~~";



/*!
* The mapoverlap OpenCL kernel to apply a user function on neighbourhood of each element in the matrix.
*/
static const std::string MatrixConvol2D_CU = R"~~~(
__global__ void {{KERNEL_NAME}}_conv_cuda_2D_kernel({{KERNEL_PARAMS}}
	const size_t skepu_in_rows, const size_t skepu_in_cols,
	const size_t skepu_out_rows, const size_t skepu_out_cols,
	size_t skepu_overlap_y, size_t skepu_overlap_x,
	size_t skepu_in_pitch, size_t skepu_out_pitch,
	const size_t skepu_sharedRows, const size_t skepu_sharedCols,
	skepu::Edge skepu_edge, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad
)
{
  extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];
	size_t skepu_xx = blockIdx.x * blockDim.x;
	size_t skepu_yy = blockIdx.y * blockDim.y;

	size_t skepu_x = skepu_xx + threadIdx.x;
	size_t skepu_y = skepu_yy + threadIdx.y;
	
	
	if (skepu_x < skepu_out_cols + skepu_overlap_x * 2 && skepu_y < skepu_out_rows + skepu_overlap_y * 2)
	{
		size_t skepu_shared_x = threadIdx.x;
		size_t skepu_shared_y = threadIdx.y;
		while (skepu_shared_y < skepu_sharedRows)
		{
			while (skepu_shared_x < skepu_sharedCols)
			{
				size_t skepu_sharedIdx = skepu_shared_y * skepu_sharedCols + skepu_shared_x;
				int skepu_global_x = (skepu_xx + skepu_shared_x - skepu_overlap_x);
				int skepu_global_y = (skepu_yy + skepu_shared_y - skepu_overlap_y);
				
				if ((skepu_global_y >= 0 && skepu_global_y < skepu_in_rows) && (skepu_global_x >= 0 && skepu_global_x < skepu_in_cols))
					{{SHARED_BUFFER}}[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[skepu_global_y * skepu_in_cols + skepu_global_x];
				else
				{
					if (skepu_edge == skepu::Edge::Pad)
						{{SHARED_BUFFER}}[skepu_sharedIdx] = skepu_pad;
					else if (skepu_edge == skepu::Edge::Duplicate)
					{
						{{SHARED_BUFFER}}[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
							skepu::cuda::clamp(skepu_global_y, 0, (int)skepu_in_rows - 1) * skepu_in_cols +
							skepu::cuda::clamp(skepu_global_x, 0, (int)skepu_in_cols - 1)];
					}
					else if (skepu_edge == skepu::Edge::Cyclic)
					{
						{{SHARED_BUFFER}}[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
							((skepu_global_y + skepu_in_rows) % skepu_in_rows) * skepu_in_cols +
							((skepu_global_x + skepu_in_cols) % skepu_in_cols)];
					}
				}
				
				skepu_shared_x += blockDim.x;
			}
			skepu_shared_x  = threadIdx.x;
			skepu_shared_y += blockDim.y;
		}
	}

	__syncthreads();
	
	{{PROXIES_INIT}}

	if (skepu_x < skepu_out_cols && skepu_y < skepu_out_rows)
	{
		size_t skepu_w2 = skepu_out_cols;
		size_t skepu_i = skepu_y * skepu_out_cols + skepu_x;
		size_t skepu_global_prng_id = skepu_i;
		size_t skepu_base = 0;
		{{INDEX_INITIALIZER}}
		{{PROXIES_UPDATE}}
		auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_BINDINGS}}
	}
}
)~~~";



/*!
* The mapoverlap OpenCL kernel to apply a user function on neighbourhood of each element in the matrix.
*/
static const std::string MatrixConvol3D_CU = R"~~~(
__global__ void {{KERNEL_NAME}}_conv_cuda_3D_kernel({{KERNEL_PARAMS}}
	const size_t skepu_in_size_i, const size_t skepu_in_size_j, const size_t skepu_in_size_k, 
	const size_t skepu_out_size_i, const size_t skepu_out_size_j, const size_t skepu_out_size_k, 
	size_t skepu_overlap_i, size_t skepu_overlap_j, size_t skepu_overlap_k,
	const size_t skepu_shared_size_i, const size_t skepu_shared_size_j, const size_t skepu_shared_size_k,
	skepu::Edge skepu_edge, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad
)
{
  extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];
	size_t skepu_kk = blockIdx.x * blockDim.x;
	size_t skepu_jj = blockIdx.y * blockDim.y;
	size_t skepu_ii = blockIdx.z * blockDim.z;

	size_t skepu_k = skepu_kk + threadIdx.x;
	size_t skepu_j = skepu_jj + threadIdx.y;
	size_t skepu_i = skepu_ii + threadIdx.z;
	
	
	if (skepu_i < skepu_out_size_i + skepu_overlap_i * 2 && skepu_j < skepu_out_size_j + skepu_overlap_j * 2 && skepu_k < skepu_out_size_k + skepu_overlap_k * 2)
	{
		size_t skepu_shared_k = threadIdx.x;
		size_t skepu_shared_j = threadIdx.y;
		size_t skepu_shared_i = threadIdx.z;
		while (skepu_shared_i < skepu_shared_size_i)
		{
			while (skepu_shared_j < skepu_shared_size_j)
			{
				while (skepu_shared_k < skepu_shared_size_k)
				{
					size_t skepu_sharedIdx = skepu_shared_i * skepu_shared_size_j * skepu_shared_size_k + skepu_shared_j * skepu_shared_size_k + skepu_shared_k;
					int skepu_global_k = (skepu_kk + skepu_shared_k - skepu_overlap_k);
					int skepu_global_j = (skepu_jj + skepu_shared_j - skepu_overlap_j);
					int skepu_global_i = (skepu_ii + skepu_shared_i - skepu_overlap_i);
					
					if ((skepu_global_i >= 0 && skepu_global_i < skepu_in_size_i) && (skepu_global_j >= 0 && skepu_global_j < skepu_in_size_j) && (skepu_global_k >= 0 && skepu_global_k < skepu_in_size_k))
						{{SHARED_BUFFER}}[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[skepu_global_i * skepu_in_size_j * skepu_in_size_k + skepu_global_j * skepu_in_size_k + skepu_global_k];
					else
					{
						if (skepu_edge == skepu::Edge::Pad)
							{{SHARED_BUFFER}}[skepu_sharedIdx] = skepu_pad;
						else if (skepu_edge == skepu::Edge::Duplicate)
						{
							{{SHARED_BUFFER}}[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
								skepu::cuda::clamp(skepu_global_i, 0, (int)skepu_in_size_i - 1) * skepu_in_size_j * skepu_in_size_k +
								skepu::cuda::clamp(skepu_global_j, 0, (int)skepu_in_size_j - 1) * skepu_in_size_k +
								skepu::cuda::clamp(skepu_global_k, 0, (int)skepu_in_size_k - 1)];
						}
						else if (skepu_edge == skepu::Edge::Cyclic)
						{
							{{SHARED_BUFFER}}[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
								((skepu_global_i + skepu_in_size_i) % skepu_in_size_i) * skepu_in_size_j * skepu_in_size_k +
								((skepu_global_j + skepu_in_size_j) % skepu_in_size_j) * skepu_in_size_k +
								((skepu_global_k + skepu_in_size_k) % skepu_in_size_k)];
						}
					}
					
					skepu_shared_k += blockDim.x;
				}
				skepu_shared_k  = threadIdx.x;
				skepu_shared_j += blockDim.y;
			}
			skepu_shared_j  = threadIdx.y;
			skepu_shared_i += blockDim.z;
		}
	}

	__syncthreads();
	
	{{PROXIES_INIT}}

	if (skepu_i < skepu_out_size_i && skepu_j < skepu_out_size_j && skepu_k < skepu_out_size_k)
	{
	//	size_t skepu_w2 = skepu_out_size_?;
		skepu_i = skepu_i * skepu_out_size_j * skepu_out_size_k + skepu_j * skepu_out_size_k + skepu_k;
		size_t skepu_global_prng_id = skepu_i;
		size_t skepu_base = 0;
		{{INDEX_INITIALIZER}}
		{{PROXIES_UPDATE}}
		auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_BINDINGS}}
	}
}
)~~~";

static const std::string MatrixConvol4D_CU = R"~~~()~~~";


std::string createMapOverlapKernelProgramHelper_CU(SkeletonInstance &instance, UserFunction &mapOverlapFunc, int dim, std::string dir, std::string kernelSource, std::string kernelTag)
{
	std::stringstream SSMapOverlapFuncArgs, SSKernelParamList;
	IndexCodeGen indexInfo = indexInitHelper_CU(mapOverlapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapOverlapFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CU(mapOverlapFunc, SSKernelParamList /*, "y * out_pitch + x"*/);
	handleRandomParam_CU(mapOverlapFunc, SSMapOverlapFuncArgs, SSKernelParamList, first);
	if (!first) { SSMapOverlapFuncArgs << ", "; }
	first = false;
	
	std::string sdataName = "sdata_" + instance;
	if (dim == 1)
		SSMapOverlapFuncArgs << "{(int)overlap, 1, &" << sdataName << "[skepu_tid + overlap]}";
	else if (dim == 2)
		SSMapOverlapFuncArgs << "{(int)skepu_overlap_y, (int)skepu_overlap_x, skepu_sharedCols, &" << sdataName << "[(threadIdx.y + skepu_overlap_y) * skepu_sharedCols + (threadIdx.x + skepu_overlap_x)]}";
	else if (dim == 3)
		SSMapOverlapFuncArgs
			<< "{(int)skepu_overlap_i, (int)skepu_overlap_j, (int)skepu_overlap_k, skepu_shared_size_j * skepu_shared_size_k, skepu_shared_size_k, &"
			<< sdataName << "[(threadIdx.z + skepu_overlap_i) * skepu_shared_size_j * skepu_shared_size_k + (threadIdx.y + skepu_overlap_j) * skepu_shared_size_k + (threadIdx.x + skepu_overlap_k)]}";
	
	SSKernelParamList << mapOverlapFunc.regionParam->templateInstantiationType() << " *skepu_input, ";
	
	auto argsInfo = handleRandomAccessAndUniforms_CU(mapOverlapFunc, SSMapOverlapFuncArgs, SSKernelParamList, first);
	
	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_" + kernelTag + "_" + mapOverlapFunc.uniqueName;
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(kernelSource,
	{
		{"{{MAPOVERLAP_INPUT_TYPE}}",    mapOverlapFunc.regionParam->templateInstantiationType()},
		{"{{KERNEL_NAME}}",              kernelName},
		{"{{FUNCTION_NAME_MAPOVERLAP}}", mapOverlapFunc.funcNameCUDA()},
		{"{{INPUT_PARAM_NAME}}",         "skepu_input"},
		{"{{KERNEL_PARAMS}}",            SSKernelParamList.str()},
		{"{{MAPOVERLAP_ARGS}}",          SSMapOverlapFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}",        indexInfo.indexInit},
		{"{{OUTPUT_BINDINGS}}",          multiOutputAssign},
		{"{{PROXIES_UPDATE}}",           argsInfo.proxyInitializerInner},
		{"{{PROXIES_INIT}}",             argsInfo.proxyInitializer},
		{"{{SHARED_BUFFER}}",            sdataName}
	});
	return kernelName;
}

std::string createMapOverlap1DKernelProgram_CU(SkeletonInstance &instance, UserFunction &mapOverlapFunc, std::string dir)
{
	return createMapOverlapKernelProgramHelper_CU(instance, mapOverlapFunc, 1, dir, 
		MapOverlapKernel_CU + MapOverlapKernel_CU_Matrix_Row + MapOverlapKernel_CU_Matrix_Col + MapOverlapKernel_CU_Matrix_ColMulti, 
		"Overlap1DKernel");
}

std::string createMapOverlap2DKernelProgram_CU(SkeletonInstance &instance, UserFunction &mapOverlapFunc, std::string dir)
{
	return createMapOverlapKernelProgramHelper_CU(instance, mapOverlapFunc, 2, dir, MatrixConvol2D_CU, "Overlap2DKernel");
}

std::string createMapOverlap3DKernelProgram_CU(SkeletonInstance &instance, UserFunction &mapOverlapFunc, std::string dir)
{
	return createMapOverlapKernelProgramHelper_CU(instance, mapOverlapFunc, 3, dir, MatrixConvol3D_CU, "Overlap3DKernel");
}

std::string createMapOverlap4DKernelProgram_CU(SkeletonInstance &instance, UserFunction &mapOverlapFunc, std::string dir)
{
	return createMapOverlapKernelProgramHelper_CU(instance, mapOverlapFunc, 4, dir, MatrixConvol4D_CU, "Overlap4DKernel");
}






