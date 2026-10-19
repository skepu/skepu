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
	skepu::Edge edgeMode, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap, int skepu_parity, int skepu_mirror
)
{
	extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];
	const size_t skepu_tid = threadIdx.x;

	// The output is cut into tiles of blockDim.x elements. A block loads the elements of its tile and the overlap on both sides, applying the
	// edge mode, then computes the tile. Mirror mode (the edge mode None with an input of the size of the output): the halo is loaded as for
	// Duplicate and only the elements that have the whole overlap in the input are computed. With the edge mode None the output is
	// shorter than the input, by the overlap on each side.
	const bool none = edgeMode == skepu::Edge::None && !skepu_mirror;
	const size_t shift = none ? overlap : 0;
	const size_t shared_size = blockDim.x + 2 * overlap;

	for (size_t skepu_tile = blockIdx.x * blockDim.x; skepu_tile < out_numelements; skepu_tile += (size_t)gridDim.x * blockDim.x)
	{
		const long skepu_base = (long)(skepu_tile + out_offset + shift) - (long)overlap;

		for (size_t k = skepu_tid; k < shared_size; k += blockDim.x)
		{
			long p = skepu_base + (long)k;
			{{MAPOVERLAP_INPUT_TYPE}} value;
			if (p >= 0 && p < (long)n)
				value = skepu_input[p];
			else if (edgeMode == skepu::Edge::Pad)
				value = pad;
			else if (edgeMode == skepu::Edge::Cyclic)
				value = (p < 0) ? wrap[p + (long)overlap] : wrap[overlap + (size_t)p - n];
			else
				value = skepu_input[(p < 0) ? 0 : n - 1]; // (Duplicate, mirror mode; the edge mode None only outside of the tile that is computed)
			{{SHARED_BUFFER}}[k] = value;
		}

		__syncthreads();

		const size_t skepu_j = skepu_tile + skepu_tid;
		const size_t skepu_center = skepu_j + out_offset + shift;
		if (skepu_j < out_numelements && (!skepu_mirror || (skepu_center >= overlap && skepu_center + overlap < n))
			&& (skepu_parity < 0 || (int)(skepu_j & 1) == skepu_parity))
		{
			size_t skepu_i = skepu_j;
			// (the number of an element is its position in the output; in mirror mode the computed elements are numbered one after the other)
			const size_t skepu_global_prng_id = skepu_mirror ? skepu_center - overlap : skepu_j;
			{{INDEX_INITIALIZER}}
			{{PROXIES_UPDATE}}
			auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
			{{OUTPUT_BINDINGS}}
		}

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
	skepu::Edge edgeMode, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap, size_t blocksPerRow, size_t rowWidth, int skepu_parity, int skepu_mirror
)
{
	extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];
	const size_t skepu_tid = threadIdx.x;

	// Every row is cut into tiles of blockDim.x output elements. A block loads the elements of its tile and the overlap on both sides,
	// applying the edge mode (the wrap array is not needed), then computes the tile. Mirror mode: see the vector kernel.
	const size_t numRows = n / rowWidth;
	const bool none = edgeMode == skepu::Edge::None && !skepu_mirror;
	const size_t out_rowWidth = none ? rowWidth - overlap * 2 : rowWidth;
	const size_t shared_size = blockDim.x + 2 * overlap;

	for (size_t skepu_work = blockIdx.x; skepu_work < numRows * blocksPerRow; skepu_work += gridDim.x)
	{
		const size_t skepu_row = skepu_work / blocksPerRow;
		const size_t skepu_tile = (skepu_work % blocksPerRow) * blockDim.x;
		const long skepu_base = (long)skepu_tile + (none ? (long)overlap : 0L) - (long)overlap;

		for (size_t k = skepu_tid; k < shared_size; k += blockDim.x)
		{
			long p = skepu_base + (long)k;
			if (p < 0 || p >= (long)rowWidth)
			{
				if (edgeMode == skepu::Edge::Cyclic) p = ((p % (long)rowWidth) + (long)rowWidth) % (long)rowWidth;
				else if (edgeMode == skepu::Edge::Pad) p = -1;
				else p = (p < 0) ? 0 : (long)rowWidth - 1; // (Duplicate, mirror mode; the edge mode None only outside of the tile that is computed)
			}
			{{SHARED_BUFFER}}[k] = (p < 0) ? pad : skepu_input[skepu_row * rowWidth + p];
		}

		__syncthreads();

		const size_t skepu_col = skepu_tile + skepu_tid;
		if (skepu_col < out_rowWidth && (!skepu_mirror || (skepu_col >= overlap && skepu_col + overlap < rowWidth))
			&& (skepu_parity < 0 || (int)(skepu_col & 1) == skepu_parity))
		{
			size_t skepu_i = skepu_col;
			{{INDEX_INITIALIZER}}
			{{PROXIES_UPDATE}}
			skepu_i = skepu_row * out_rowWidth + skepu_col;
			const size_t skepu_global_prng_id = skepu_mirror ? skepu_row * (rowWidth - 2 * overlap) + skepu_col - overlap : skepu_i;
			auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
			{{OUTPUT_BINDINGS}}
		}

		__syncthreads();
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
	skepu::Edge edgeMode, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap, size_t blocksPerCol, size_t rowWidth, size_t colWidth, int skepu_parity, int skepu_mirror
)
{
	extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];
	const size_t skepu_tid = threadIdx.x;

	// Every column (rowWidth is the number of columns) is cut into tiles of blockDim.x output elements, as in the row-wise kernel.
	const bool none = edgeMode == skepu::Edge::None && !skepu_mirror;
	const size_t out_colWidth = none ? colWidth - overlap * 2 : colWidth;
	const size_t shared_size = blockDim.x + 2 * overlap;

	for (size_t skepu_work = blockIdx.x; skepu_work < rowWidth * blocksPerCol; skepu_work += gridDim.x)
	{
		const size_t skepu_col = skepu_work / blocksPerCol;
		const size_t skepu_tile = (skepu_work % blocksPerCol) * blockDim.x;
		const long skepu_base = (long)skepu_tile + (none ? (long)overlap : 0L) - (long)overlap;

		for (size_t k = skepu_tid; k < shared_size; k += blockDim.x)
		{
			long p = skepu_base + (long)k;
			if (p < 0 || p >= (long)colWidth)
			{
				if (edgeMode == skepu::Edge::Cyclic) p = ((p % (long)colWidth) + (long)colWidth) % (long)colWidth;
				else if (edgeMode == skepu::Edge::Pad) p = -1;
				else p = (p < 0) ? 0 : (long)colWidth - 1; // (Duplicate, mirror mode; the edge mode None only outside of the tile that is computed)
			}
			{{SHARED_BUFFER}}[k] = (p < 0) ? pad : skepu_input[(size_t)p * rowWidth + skepu_col];
		}

		__syncthreads();

		const size_t skepu_orow = skepu_tile + skepu_tid;
		if (skepu_orow < out_colWidth && (!skepu_mirror || (skepu_orow >= overlap && skepu_orow + overlap < colWidth))
			&& (skepu_parity < 0 || (int)(skepu_orow & 1) == skepu_parity))
		{
			size_t skepu_i = skepu_orow;
			{{INDEX_INITIALIZER}}
			{{PROXIES_UPDATE}}
			skepu_i = skepu_orow * rowWidth + skepu_col;
			// (the numbers of the elements of the output go down the columns, one column after the other)
			const size_t skepu_global_prng_id = skepu_mirror ? skepu_col * (colWidth - 2 * overlap) + skepu_orow - overlap : skepu_col * out_colWidth + skepu_orow;
			auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
			{{OUTPUT_BINDINGS}}
		}

		__syncthreads();
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
   size_t skepu_stride_y, size_t skepu_stride_x,
	size_t skepu_in_pitch, size_t skepu_out_pitch,
	const size_t skepu_sharedRows, const size_t skepu_sharedCols,
	int skepu_is_pool, skepu::Edge skepu_edge, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad, int skepu_parity
)
{
  extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];
	size_t skepu_xx = blockIdx.x * blockDim.x * skepu_stride_x;
	size_t skepu_yy = blockIdx.y * blockDim.y * skepu_stride_y;

	size_t skepu_x = skepu_xx/skepu_stride_x + threadIdx.x;
	size_t skepu_y = skepu_yy/skepu_stride_y + threadIdx.y;

    if (skepu_is_pool)
        if ((skepu_overlap_x == 0 || skepu_overlap_x <= skepu_stride_x) &&
            (skepu_overlap_y == 0 || skepu_overlap_y <= skepu_stride_y))
        {
            if (skepu_x < skepu_out_cols && skepu_y < skepu_out_rows)
            {
                {{INDEX_INITIALIZER}}
                size_t skepu_global_prng_id = skepu_y * skepu_out_cols + skepu_x;
                auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{PERFECT_MAPPOOL_ARGS}});
        
                size_t skepu_i = skepu_global_prng_id;
                {{OUTPUT_BINDINGS}}
            }
            return;
        }
	
    size_t skepu_offset_x = skepu_is_pool ? 0 : ((skepu_out_cols - skepu_in_cols) / 2 + skepu_overlap_x);
	size_t skepu_offset_y = skepu_is_pool ? 0 : ((skepu_out_rows - skepu_in_rows) / 2 + skepu_overlap_y);

	
	if (skepu_x < skepu_out_cols + skepu_overlap_x * 2 && skepu_y < skepu_out_rows + skepu_overlap_y * 2)
	{
		size_t skepu_shared_x = threadIdx.x;
		size_t skepu_shared_y = threadIdx.y;
		while (skepu_shared_y < skepu_sharedRows)
		{
			while (skepu_shared_x < skepu_sharedCols)
			{
				size_t skepu_sharedIdx = skepu_shared_y * skepu_sharedCols + skepu_shared_x;
				int skepu_global_x = (skepu_xx + skepu_shared_x - skepu_offset_x);
				int skepu_global_y = (skepu_yy + skepu_shared_y - skepu_offset_y);
				
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

	// Mirror mode (the edge mode None and an input of the size of the output): only the interior of the output is computed, the
	// elements near the edges keep their values
	const bool skepu_mirror = !skepu_is_pool && skepu_edge == skepu::Edge::None && skepu_out_rows == skepu_in_rows && skepu_out_cols == skepu_in_cols;
	// (update modes: skepu_parity is -1 for all the elements, else the parity of the sum of the indices of the elements to compute)
	if (skepu_x < skepu_out_cols && skepu_y < skepu_out_rows && (skepu_parity < 0 || (int)((skepu_x + skepu_y) & 1) == skepu_parity)
		&& (!skepu_mirror || (skepu_x >= skepu_overlap_x && skepu_x + skepu_overlap_x < skepu_out_cols && skepu_y >= skepu_overlap_y && skepu_y + skepu_overlap_y < skepu_out_rows)))
	{
		size_t skepu_i = skepu_y * skepu_out_cols + skepu_x;
		// (in mirror mode the computed elements are numbered one after the other)
		size_t skepu_global_prng_id = skepu_mirror ? (skepu_y - skepu_overlap_y) * (skepu_out_cols - 2 * skepu_overlap_x) + (skepu_x - skepu_overlap_x) : skepu_i;
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
    size_t skepu_stride_i, size_t skepu_stride_j, size_t skepu_stride_k,
	const size_t skepu_shared_size_i, const size_t skepu_shared_size_j, const size_t skepu_shared_size_k,
	int skepu_is_pool, skepu::Edge skepu_edge, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad, int skepu_parity
)
{
  extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];
	size_t skepu_kk = blockIdx.x * blockDim.x * skepu_stride_k;
	size_t skepu_jj = blockIdx.y * blockDim.y * skepu_stride_j;
	size_t skepu_ii = blockIdx.z * blockDim.z * skepu_stride_i;

	size_t skepu_k = skepu_kk/skepu_stride_k + threadIdx.x;
	size_t skepu_j = skepu_jj/skepu_stride_j + threadIdx.y;
	size_t skepu_i = skepu_ii/skepu_stride_i + threadIdx.z;

    if (skepu_is_pool)
        if ((skepu_overlap_i == 0 || skepu_overlap_i <= skepu_stride_i) &&
            (skepu_overlap_j == 0 || skepu_overlap_j <= skepu_stride_j) &&
            (skepu_overlap_k == 0 || skepu_overlap_k <= skepu_stride_k))
        {
            if (skepu_i < skepu_out_size_i && skepu_j < skepu_out_size_j && skepu_k < skepu_out_size_k)
            {
                {{INDEX_INITIALIZER}}
                const size_t skepu_global_prng_id = skepu_i * skepu_out_size_j * skepu_out_size_k + skepu_j * skepu_out_size_k + skepu_k;
                auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{PERFECT_MAPPOOL_ARGS}});
        
                skepu_i = skepu_global_prng_id;
                {{OUTPUT_BINDINGS}}
            }
            return;
        }

    size_t skepu_offset_i = skepu_is_pool ? 0 : ((skepu_out_size_i - skepu_in_size_i) / 2 + skepu_overlap_i);
    size_t skepu_offset_j = skepu_is_pool ? 0 : ((skepu_out_size_j - skepu_in_size_j) / 2 + skepu_overlap_j);
    size_t skepu_offset_k = skepu_is_pool ? 0 : ((skepu_out_size_k - skepu_in_size_k) / 2 + skepu_overlap_k);
	
	
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
					int skepu_global_k = (skepu_kk + skepu_shared_k - skepu_offset_k);
					int skepu_global_j = (skepu_jj + skepu_shared_j - skepu_offset_j);
					int skepu_global_i = (skepu_ii + skepu_shared_i - skepu_offset_i);
					
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

	// Mirror mode (the edge mode None and an input of the size of the output): only the interior of the output is computed, the
	// elements near the edges keep their values
	const bool skepu_mirror = !skepu_is_pool && skepu_edge == skepu::Edge::None && skepu_out_size_i == skepu_in_size_i && skepu_out_size_j == skepu_in_size_j && skepu_out_size_k == skepu_in_size_k;
	// (update modes: skepu_parity is -1 for all the elements, else the parity of the sum of the indices of the elements to compute)
	if (skepu_i < skepu_out_size_i && skepu_j < skepu_out_size_j && skepu_k < skepu_out_size_k && (skepu_parity < 0 || (int)((skepu_i + skepu_j + skepu_k) & 1) == skepu_parity)
		&& (!skepu_mirror || (skepu_i >= skepu_overlap_i && skepu_i + skepu_overlap_i < skepu_out_size_i && skepu_j >= skepu_overlap_j && skepu_j + skepu_overlap_j < skepu_out_size_j
			&& skepu_k >= skepu_overlap_k && skepu_k + skepu_overlap_k < skepu_out_size_k)))
	{
		// (in mirror mode the computed elements are numbered one after the other)
		const size_t skepu_mirror_id = skepu_mirror ? ((skepu_i - skepu_overlap_i) * (skepu_out_size_j - 2 * skepu_overlap_j) + (skepu_j - skepu_overlap_j)) * (skepu_out_size_k - 2 * skepu_overlap_k) + (skepu_k - skepu_overlap_k) : 0;
        {{INDEX_INITIALIZER}}
		skepu_i = skepu_i * skepu_out_size_j * skepu_out_size_k + skepu_j * skepu_out_size_k + skepu_k;
		size_t skepu_global_prng_id = skepu_mirror ? skepu_mirror_id : skepu_i;
		{{PROXIES_UPDATE}}
		auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_BINDINGS}}
	}
}
)~~~";

static const std::string MatrixConvol4D_CU = R"~~~(
__global__ void {{KERNEL_NAME}}_conv_cuda_4D_kernel({{KERNEL_PARAMS}}
	const size_t skepu_in_size_i, const size_t skepu_in_size_j, const size_t skepu_in_size_k, size_t skepu_in_size_l,
	const size_t skepu_out_size_i, const size_t skepu_out_size_j, const size_t skepu_out_size_k, size_t skepu_out_size_l,
	size_t skepu_overlap_i, size_t skepu_overlap_j, size_t skepu_overlap_k, size_t skepu_overlap_l,
    size_t skepu_stride_i, size_t skepu_stride_j, size_t skepu_stride_k, size_t skepu_stride_l,
	size_t skepu_shared_size_i, size_t skepu_shared_size_j, size_t skepu_shared_size_k, size_t skepu_shared_size_l,
	size_t skepu_numThreads_0a,
    int skepu_is_pool, skepu::Edge skepu_edge, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad, int skepu_parity
)
{
    extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} {{SHARED_BUFFER}}[];

    size_t skepu_numThreads_0b = blockDim.x / skepu_numThreads_0a;
   
    size_t skepu_l = (blockIdx.x * blockDim.x + threadIdx.x) % skepu_numThreads_0a;
    size_t skepu_k = (blockIdx.x * blockDim.x + threadIdx.x) / skepu_numThreads_0a;
	size_t skepu_j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t skepu_i = blockIdx.z * blockDim.z + threadIdx.z;

    size_t skepu_ll = ((size_t)(skepu_l / skepu_numThreads_0a)) * skepu_numThreads_0a;
	size_t skepu_kk = ((size_t)(skepu_k / skepu_numThreads_0b)) * skepu_numThreads_0b;
	size_t skepu_jj = ((size_t)(skepu_j / blockDim.y)) * blockDim.y;
	size_t skepu_ii = ((size_t)(skepu_i / blockDim.z)) * blockDim.z;


    if (skepu_is_pool)
    {
        if (skepu_i < skepu_out_size_i && skepu_j < skepu_out_size_j && skepu_k < skepu_out_size_k && skepu_l < skepu_out_size_l)
        {
            {{INDEX_INITIALIZER}}
            const size_t skepu_global_prng_id = skepu_i * skepu_out_size_j * skepu_out_size_k * skepu_out_size_l + skepu_j * skepu_out_size_k * skepu_out_size_l + skepu_k * skepu_out_size_l + skepu_l;
            auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{PERFECT_MAPPOOL_ARGS}});
    
            skepu_i = skepu_global_prng_id;
            {{OUTPUT_BINDINGS}}
        }
        return;
    }
   
   size_t skepu_increment_l = skepu_numThreads_0a; //blockDim.x / skepu_out_size_l;
   size_t skepu_increment_k = blockDim.x / skepu_increment_l;

   size_t skepu_offset_i = (skepu_out_size_i - skepu_in_size_i) / 2 + skepu_overlap_i;
	size_t skepu_offset_j = (skepu_out_size_j - skepu_in_size_j) / 2 + skepu_overlap_j;
	size_t skepu_offset_k = (skepu_out_size_k - skepu_in_size_k) / 2 + skepu_overlap_k;
	size_t skepu_offset_l = (skepu_out_size_l - skepu_in_size_l) / 2 + skepu_overlap_l;

   if (skepu_overlap_i == 0) skepu_offset_i = 0;
	if (skepu_overlap_j == 0) skepu_offset_j = 0;
	if (skepu_overlap_k == 0) skepu_offset_k = 0;
	if (skepu_overlap_l == 0) skepu_offset_l = 0;
	
	if (skepu_i < skepu_out_size_i + skepu_overlap_i * 2
       && skepu_j < skepu_out_size_j + skepu_overlap_j * 2
       && skepu_k < skepu_out_size_k + skepu_overlap_k * 2
       && skepu_l < skepu_out_size_l + skepu_overlap_l * 2)
	{
      size_t skepu_shared_l = threadIdx.x % skepu_numThreads_0a;
		size_t skepu_shared_k = threadIdx.x / skepu_numThreads_0a;
		size_t skepu_shared_j = threadIdx.y;
		size_t skepu_shared_i = threadIdx.z;
		while (skepu_shared_i < skepu_shared_size_i)
		{
			while (skepu_shared_j < skepu_shared_size_j)
			{
				while (skepu_shared_k < skepu_shared_size_k)
				{
               while (skepu_shared_l < skepu_shared_size_l)
               {
					   size_t skepu_sharedIdx =
                  skepu_shared_i * skepu_shared_size_j * skepu_shared_size_k * skepu_shared_size_l +
                  skepu_shared_j * skepu_shared_size_k * skepu_shared_size_l +
                  skepu_shared_k * skepu_shared_size_l +
                  skepu_shared_l;

					   int skepu_global_l = (skepu_ll + skepu_shared_l - skepu_offset_l);
                  int skepu_global_k = (skepu_kk + skepu_shared_k - skepu_offset_k);
					   int skepu_global_j = (skepu_jj + skepu_shared_j - skepu_offset_j);
					   int skepu_global_i = (skepu_ii + skepu_shared_i - skepu_offset_i);

					   if ((skepu_global_i >= 0 && skepu_global_i < skepu_in_size_i) && (skepu_global_j >= 0 && skepu_global_j < skepu_in_size_j)
                   && (skepu_global_k >= 0 && skepu_global_k < skepu_in_size_k) && (skepu_global_l >= 0 && skepu_global_l < skepu_in_size_l))
                  {
						   {{SHARED_BUFFER}}[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
                        skepu_global_i * skepu_in_size_j * skepu_in_size_k * skepu_in_size_l +
                        skepu_global_j * skepu_in_size_k * skepu_in_size_l +
                        skepu_global_k * skepu_in_size_l +
                        skepu_global_l];
                  }
					   else
					   {
						   if (skepu_edge == skepu::Edge::Pad)
							   {{SHARED_BUFFER}}[skepu_sharedIdx] = skepu_pad;
						   else if (skepu_edge == skepu::Edge::Duplicate)
						   {
							   {{SHARED_BUFFER}}[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
								   skepu::cuda::clamp(skepu_global_i, 0, (int)skepu_in_size_i - 1) * skepu_in_size_j * skepu_in_size_k * skepu_in_size_l +
								   skepu::cuda::clamp(skepu_global_j, 0, (int)skepu_in_size_j - 1) * skepu_in_size_k * skepu_in_size_l +
								   skepu::cuda::clamp(skepu_global_k, 0, (int)skepu_in_size_k - 1) * skepu_in_size_l +
                           skepu::cuda::clamp(skepu_global_l, 0, (int)skepu_in_size_l - 1)];
						   }
						   else if (skepu_edge == skepu::Edge::Cyclic)
						   {
							   {{SHARED_BUFFER}}[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
								   ((skepu_global_i + skepu_in_size_i) % skepu_in_size_i) * skepu_in_size_j * skepu_in_size_k * skepu_in_size_l +
								   ((skepu_global_j + skepu_in_size_j) % skepu_in_size_j) * skepu_in_size_k * skepu_in_size_l +
								   ((skepu_global_k + skepu_in_size_k) % skepu_in_size_k) * skepu_in_size_l +
                           ((skepu_global_l + skepu_in_size_l) % skepu_in_size_l)];
						   }
					   }
                  
                  skepu_shared_l += skepu_increment_l;
				   }
               skepu_shared_l = threadIdx.x % skepu_out_size_l;
				   skepu_shared_k += skepu_increment_k;
			   }
            skepu_shared_k = threadIdx.x / skepu_out_size_l;
            skepu_shared_j += blockDim.y;
		   }
         skepu_shared_j = threadIdx.y;
         skepu_shared_i += blockDim.z;
      }
	}
   
	__syncthreads();
	
	{{PROXIES_INIT}}

	// Mirror mode (the edge mode None and an input of the size of the output): only the interior of the output is computed, the
	// elements near the edges keep their values
	const bool skepu_mirror = !skepu_is_pool && skepu_edge == skepu::Edge::None && skepu_out_size_i == skepu_in_size_i && skepu_out_size_j == skepu_in_size_j && skepu_out_size_k == skepu_in_size_k && skepu_out_size_l == skepu_in_size_l;
	// (update modes: skepu_parity is -1 for all the elements, else the parity of the sum of the indices of the elements to compute)
	if (skepu_i < skepu_out_size_i && skepu_j < skepu_out_size_j && skepu_k < skepu_out_size_k && skepu_l < skepu_out_size_l
		&& (skepu_parity < 0 || (int)((skepu_i + skepu_j + skepu_k + skepu_l) & 1) == skepu_parity)
		&& (!skepu_mirror || (skepu_i >= skepu_overlap_i && skepu_i + skepu_overlap_i < skepu_out_size_i && skepu_j >= skepu_overlap_j && skepu_j + skepu_overlap_j < skepu_out_size_j
			&& skepu_k >= skepu_overlap_k && skepu_k + skepu_overlap_k < skepu_out_size_k && skepu_l >= skepu_overlap_l && skepu_l + skepu_overlap_l < skepu_out_size_l)))
	{
		// (in mirror mode the computed elements are numbered one after the other)
		const size_t skepu_mirror_id = skepu_mirror ? (((skepu_i - skepu_overlap_i) * (skepu_out_size_j - 2 * skepu_overlap_j) + (skepu_j - skepu_overlap_j)) * (skepu_out_size_k - 2 * skepu_overlap_k)
			+ (skepu_k - skepu_overlap_k)) * (skepu_out_size_l - 2 * skepu_overlap_l) + (skepu_l - skepu_overlap_l) : 0;
		{{INDEX_INITIALIZER}}
		skepu_i = skepu_i * skepu_out_size_j * skepu_out_size_k * skepu_out_size_l + skepu_j * skepu_out_size_k * skepu_out_size_l + skepu_k * skepu_out_size_l + skepu_l;
		size_t skepu_global_prng_id = skepu_mirror ? skepu_mirror_id : skepu_i;
		{{PROXIES_UPDATE}}
		auto skepu_res = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_BINDINGS}}
	}
}
)~~~";


std::string createMapOverlapKernelProgramHelper_CU(SkeletonInstance &instance, UserFunction &mapOverlapFunc, int dim, std::string dir, std::string kernelSource, std::string kernelTag)
{
	std::stringstream SSMapOverlapFuncArgs, SSKernelParamList, ss_perfect_mappool_args;
	IndexCodeGen indexInfo = indexInitHelper_CU(mapOverlapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapOverlapFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CU(mapOverlapFunc, SSKernelParamList /*, "y * out_pitch + x"*/);
	handleRandomParam_CU(mapOverlapFunc, SSMapOverlapFuncArgs, SSKernelParamList, first);
	if (!first) { SSMapOverlapFuncArgs << ", "; }
	first = false;

    std::string indexInit = "";
    if (mapOverlapFunc.indexed1D)
		indexInit = "skepu::Index1D skepu_index{.i = skepu_i};";
    else if (mapOverlapFunc.indexed2D)
		indexInit = "skepu::Index2D skepu_index{.row = skepu_y, .col = skepu_x};";
    else if (mapOverlapFunc.indexed3D)
		indexInit = "skepu::Index3D skepu_index{.i = skepu_i, .j = skepu_j, .k = skepu_k};";
    else if (mapOverlapFunc.indexed4D)
		indexInit = "skepu::Index4D skepu_index{.i = skepu_i, .j = skepu_j, .k = skepu_k, .l = skepu_l};";
	
	std::string sdataName = "sdata_" + instance;
	if (dim == 1)
		SSMapOverlapFuncArgs << "{(int)overlap, 1, &" << sdataName << "[skepu_tid + overlap]}";
	else if (dim == 2)
    {
        if (mapOverlapFunc.regionParam->isPool)
        {
            SSMapOverlapFuncArgs << "{(int)skepu_overlap_y, (int)skepu_overlap_x, skepu_sharedCols, &" << sdataName << "[(threadIdx.y * skepu_stride_y) * skepu_sharedCols + (threadIdx.x * skepu_stride_x)]}";
            ss_perfect_mappool_args << "{(int)skepu_overlap_y, (int)skepu_overlap_x,"
                                    <<  "skepu_in_cols,"
                                    <<  "&skepu_input[skepu_y * skepu_stride_y * skepu_in_cols +"
                                                 <<  "skepu_x * skepu_stride_x]}";
        }
        else
        SSMapOverlapFuncArgs << "{(int)skepu_overlap_y, (int)skepu_overlap_x, skepu_sharedCols, &" << sdataName << "[(threadIdx.y + skepu_overlap_y) * skepu_sharedCols + (threadIdx.x + skepu_overlap_x)]}";
    }
	else if (dim == 3)
    {
        if (mapOverlapFunc.regionParam->isPool)
        {
            SSMapOverlapFuncArgs
			<< "{(int)skepu_overlap_i, (int)skepu_overlap_j, (int)skepu_overlap_k, skepu_shared_size_j * skepu_shared_size_k, skepu_shared_size_k, &"
			<< sdataName << "[(threadIdx.z * skepu_stride_i) * skepu_shared_size_j * skepu_shared_size_k + (threadIdx.y * skepu_stride_j) * skepu_shared_size_k + (threadIdx.x * skepu_stride_k)]}";

            ss_perfect_mappool_args << "{(int)skepu_overlap_i, (int)skepu_overlap_j, (int)skepu_overlap_k,"
                                    <<  "skepu_in_size_j * skepu_in_size_k,"
                                    <<  "skepu_in_size_k,"
                                    <<  "&skepu_input[skepu_i * skepu_stride_i * skepu_in_size_j * skepu_in_size_k +"
                                                 <<  "skepu_j * skepu_stride_j * skepu_in_size_k +"
                                                 <<  "skepu_k * skepu_stride_k]}";
        }
        else
            SSMapOverlapFuncArgs
                << "{(int)skepu_overlap_i, (int)skepu_overlap_j, (int)skepu_overlap_k, skepu_shared_size_j * skepu_shared_size_k, skepu_shared_size_k, &"
                << sdataName << "[(threadIdx.z + skepu_overlap_i) * skepu_shared_size_j * skepu_shared_size_k + (threadIdx.y + skepu_overlap_j) * skepu_shared_size_k + (threadIdx.x + skepu_overlap_k)]}";
    }
   else if (dim == 4)
   {
        if (mapOverlapFunc.regionParam->isPool)
        {
            SSMapOverlapFuncArgs
                << "{(int)skepu_overlap_i, (int)skepu_overlap_j, (int)skepu_overlap_k, (int)skepu_overlap_l, skepu_shared_size_j * skepu_shared_size_k * skepu_shared_size_l, skepu_shared_size_k * skepu_shared_size_l, skepu_shared_size_l, &"
                << sdataName << "[(threadIdx.z * skepu_stride_i) * skepu_shared_size_j * skepu_shared_size_k * skepu_shared_size_l + (threadIdx.y * skepu_stride_j) * skepu_shared_size_k * skepu_shared_size_l + ((threadIdx.x / skepu_numThreads_0a) * skepu_stride_k) * skepu_shared_size_l + ((threadIdx.x % skepu_numThreads_0a) * skepu_stride_l)]}";
            ss_perfect_mappool_args << "{(int)skepu_overlap_i, (int)skepu_overlap_j, (int)skepu_overlap_k, (int)skepu_overlap_l,"
                                    <<  "skepu_in_size_j * skepu_in_size_k * skepu_in_size_l,"
                                    <<  "skepu_in_size_k * skepu_in_size_l,"
                                    <<  "skepu_in_size_l,"
                                    <<  "&skepu_input[skepu_i * skepu_stride_i * skepu_in_size_j * skepu_in_size_k * skepu_in_size_l +"
                                                 <<  "skepu_j * skepu_stride_j * skepu_in_size_k * skepu_in_size_l +"
                                                 <<  "skepu_k * skepu_stride_k * skepu_in_size_l +"
                                                 <<  "skepu_l * skepu_stride_l]}";
        }
        else
        {
            SSMapOverlapFuncArgs
                << "{(int)skepu_overlap_i, (int)skepu_overlap_j, (int)skepu_overlap_k, (int)skepu_overlap_l, skepu_shared_size_j * skepu_shared_size_k * skepu_shared_size_l, skepu_shared_size_k * skepu_shared_size_l, skepu_shared_size_l, &"
                << sdataName << "[(threadIdx.z + skepu_overlap_i) * skepu_shared_size_j * skepu_shared_size_k * skepu_shared_size_l + (threadIdx.y + skepu_overlap_j) * skepu_shared_size_k * skepu_shared_size_l + (threadIdx.x / skepu_numThreads_0a + skepu_overlap_k) * skepu_shared_size_l + (threadIdx.x % skepu_numThreads_0a + skepu_overlap_l)]}";
        }

   }
	
	SSKernelParamList << mapOverlapFunc.regionParam->templateInstantiationType() << " *skepu_input, ";
	
	auto argsInfo = handleRandomAccessAndUniforms_CU(mapOverlapFunc, SSMapOverlapFuncArgs, SSKernelParamList, first);

    if (!mapOverlapFunc.regionParam->isPool)
        ss_perfect_mappool_args << SSMapOverlapFuncArgs.str();
	
	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_" + kernelTag + "_" + mapOverlapFunc.uniqueName;
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(kernelSource,
	{
		{"{{MAPOVERLAP_INPUT_TYPE}}",    mapOverlapFunc.regionParam->templateInstantiationType()},
		{"{{KERNEL_NAME}}",              kernelName},
		{"{{FUNCTION_NAME_MAPOVERLAP}}", mapOverlapFunc.funcNameCUDA()},
		{"{{INPUT_PARAM_NAME}}",         "skepu_input"},
		{"{{KERNEL_PARAMS}}",            SSKernelParamList.str()},
		{"{{MAPOVERLAP_ARGS}}",          SSMapOverlapFuncArgs.str()},
        {"{{PERFECT_MAPPOOL_ARGS}}",     ss_perfect_mappool_args.str()},
		{"{{INDEX_INITIALIZER}}",        indexInit},
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






