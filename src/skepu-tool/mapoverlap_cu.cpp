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
	int poly, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap
)
{
   extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} sdata[];
   size_t skepu_tid = threadIdx.x;
   size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   while(skepu_i<(n+overlap-1))
   {
      //Copy data to shared memory
      if (poly == 0) // constant policy
      {
         sdata[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[skepu_i] : pad;

         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = (skepu_i<overlap) ? pad : skepu_input[skepu_i-overlap];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            sdata[skepu_tid+2*overlap] = (skepu_i+overlap < n) ? skepu_input[skepu_i+overlap] : pad;
         }
      }
      else if (poly == 1)
      {
         if (skepu_i < n)
         {
            sdata[overlap+skepu_tid] = skepu_input[skepu_i];
         }
         else
         {
            sdata[overlap + skepu_tid] = wrap[overlap + (skepu_i - n)];
         }

         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = (skepu_i < overlap) ? wrap[skepu_tid] : skepu_input[skepu_i - overlap];
         }

         if (skepu_tid >= (blockDim.x - overlap))
         {
            sdata[skepu_tid + 2 * overlap] = (skepu_i + overlap < n) ? skepu_input[skepu_i + overlap] : wrap[overlap + (skepu_i + overlap - n)];
         }
      }
      else if (poly == 2) // DUPLICATE
      {
         sdata[overlap + skepu_tid] = (skepu_i < n) ? skepu_input[skepu_i] : skepu_input[n - 1];

         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = (skepu_i < overlap) ? skepu_input[0] : skepu_input[skepu_i - overlap];
         }

         if (skepu_tid >= (blockDim.x - overlap))
         {
            sdata[skepu_tid+2*overlap] = (skepu_i + overlap < n) ? skepu_input[skepu_i + overlap] : skepu_input[n - 1];
         }
      }

      __syncthreads();

      //Compute and store data
      if ( (skepu_i >= out_offset) && (skepu_i < out_offset + out_numelements) )
			{
      //	skepu_output[skepu_i-out_offset] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
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
	int poly, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap, size_t blocksPerRow, size_t rowWidth
)
{
   extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} sdata[];
   size_t skepu_tid = threadIdx.x;
   size_t skepu_i = blockIdx.x * blockDim.x + skepu_tid;

   size_t wrapIndex = 2 * overlap * (int)(blockIdx.x / blocksPerRow);
   size_t tmp = (blockIdx.x % blocksPerRow);
   size_t tmp2 = (blockIdx.x / blocksPerRow);


   // Copy data to shared memory
   if (poly == 0)
   {
      sdata[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[skepu_i] : pad;

      if (skepu_tid < overlap)
      {
         sdata[skepu_tid] = (tmp==0) ? pad : skepu_input[skepu_i-overlap];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         sdata[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x - 1 && (skepu_i+overlap < n) && tmp != (blocksPerRow - 1)) ? skepu_input[skepu_i + overlap] : pad;
      }
   }
   else if (poly == 1)
   {
      if (skepu_i < n)
      {
         sdata[overlap + skepu_tid] = skepu_input[skepu_i];
      }
      else if (skepu_i - n < overlap)
      {
         sdata[overlap + skepu_tid] = wrap[(overlap + (skepu_i - n)) + wrapIndex];
      }
      else
      {
         sdata[overlap + skepu_tid] = pad;
      }

      if (skepu_tid < overlap)
      {
         sdata[skepu_tid] = (tmp == 0) ? wrap[skepu_tid + wrapIndex] : skepu_input[skepu_i - overlap];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         sdata[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && skepu_i+overlap < n && tmp!=(blocksPerRow-1)) ? skepu_input[skepu_i+overlap] : wrap[overlap+wrapIndex+(skepu_tid+overlap-blockDim.x)];
      }
   }
   else if (poly == 2)
   {
      sdata[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[skepu_i] : skepu_input[n-1];

      if (skepu_tid < overlap)
      {
         sdata[skepu_tid] = (tmp==0) ? skepu_input[tmp2*rowWidth] : skepu_input[skepu_i-overlap];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         sdata[skepu_tid + 2 * overlap] = (blockIdx.x != gridDim.x-1 && (skepu_i + overlap < n) && (tmp != (blocksPerRow - 1))) ? skepu_input[skepu_i + overlap] : skepu_input[(tmp2 + 1) * rowWidth - 1];
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
	int poly, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap, size_t blocksPerCol, size_t rowWidth, size_t colWidth
)
{
   extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} sdata[];
   size_t skepu_tid = threadIdx.x;
   size_t skepu_i = blockIdx.x * blockDim.x + skepu_tid;

   size_t wrapIndex= 2 * overlap * (int)(blockIdx.x/blocksPerCol);
   size_t tmp= (blockIdx.x % blocksPerCol);
   size_t tmp2= (blockIdx.x / blocksPerCol);

   size_t arrInd = (threadIdx.x + tmp*blockDim.x)*rowWidth + ((blockIdx.x)/blocksPerCol);

   //Copy data to shared memory
   if (poly == 0)
   {
      sdata[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[arrInd] : pad;

      if (skepu_tid < overlap)
      {
         sdata[skepu_tid] = (tmp==0) ? pad : skepu_input[(arrInd-(overlap*rowWidth))];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         sdata[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+(overlap*rowWidth))] : pad;
      }
   }
   else if (poly == 1)
   {
      if (skepu_i < n)
      {
         sdata[overlap+skepu_tid] = skepu_input[arrInd];
      }
      else if (skepu_i-n < overlap)
      {
         sdata[overlap+skepu_tid] = wrap[(overlap+(skepu_i-n))+ wrapIndex];
      }
      else
      {
         sdata[overlap+skepu_tid] = pad;
      }

      if (skepu_tid < overlap)
      {
         sdata[skepu_tid] = (tmp==0) ? wrap[skepu_tid+wrapIndex] : skepu_input[(arrInd-(overlap*rowWidth))];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         sdata[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+(overlap*rowWidth))] : wrap[overlap+wrapIndex+(skepu_tid+overlap-blockDim.x)];
      }
   }
   else if (poly == 2)
   {
      sdata[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[arrInd] : skepu_input[n-1];

      if (skepu_tid < overlap)
      {
         sdata[skepu_tid] = (tmp==0) ? skepu_input[tmp2] : skepu_input[(arrInd-(overlap*rowWidth))];
      }

      if (skepu_tid >= (blockDim.x-overlap))
      {
         sdata[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+(overlap*rowWidth))] : skepu_input[tmp2+(colWidth-1)*rowWidth];
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
	int poly, int deviceType, {{MAPOVERLAP_INPUT_TYPE}} pad, size_t overlap, size_t blocksPerCol, size_t rowWidth, size_t colWidth
)
{
   extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} sdata[];

   size_t skepu_tid = threadIdx.x;
   size_t skepu_i = blockIdx.x * blockDim.x + skepu_tid;

   size_t tmp= (blockIdx.x % blocksPerCol);
   size_t tmp2= (blockIdx.x / blocksPerCol);

   size_t arrInd = (threadIdx.x + tmp*blockDim.x)*rowWidth + tmp2; //((blockIdx.x)/blocksPerCol);

   if (poly == 0) //IF overlap policy is CONSTANT
   {
      sdata[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[arrInd+in_offset] : pad; // in_offset

      if (deviceType == -1) // first device, i.e. in_offset=0
      {
         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = (tmp==0) ? pad : skepu_input[(arrInd-(overlap*rowWidth))];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            sdata[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 0) // middle device
      {
         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            sdata[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 1) // last device
      {
         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            sdata[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+in_offset+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+in_offset+(overlap*rowWidth))] : pad;
         }
      }
   }
   else if (poly == 1) //IF overlap policy is CYCLIC
   {
      sdata[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[arrInd+in_offset] : ((skepu_i-n < overlap) ? wrap[(skepu_i-n)+ (overlap * tmp2)] : pad);

      if (deviceType == -1) // first device, i.e. in_offset=0
      {
         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = (tmp==0) ? wrap[skepu_tid+(overlap * tmp2)] : skepu_input[(arrInd-(overlap*rowWidth))];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            sdata[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 0) // middle device
      {
         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            sdata[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 1) // last device
      {
         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            sdata[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+in_offset+(overlap*rowWidth))] : wrap[(overlap * tmp2)+(skepu_tid+overlap-blockDim.x)];
         }
      }
   }
   else if (poly == 2) //IF overlap policy is DUPLICATE
   {
      sdata[overlap+skepu_tid] = (skepu_i < n) ? skepu_input[arrInd+in_offset] : skepu_input[n+in_offset-1];

      if (deviceType == -1) // first device, i.e. in_offset=0
      {
         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = (tmp==0) ? skepu_input[tmp2] : skepu_input[(arrInd-(overlap*rowWidth))];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            sdata[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 0) // middle device
      {
         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            sdata[skepu_tid+2*overlap] = skepu_input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if (deviceType == 1) // last device
      {
         if (skepu_tid < overlap)
         {
            sdata[skepu_tid] = skepu_input[arrInd];
         }

         if (skepu_tid >= (blockDim.x-overlap))
         {
            sdata[skepu_tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+in_offset+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? skepu_input[(arrInd+in_offset+(overlap*rowWidth))] : skepu_input[tmp2+in_offset+(colWidth-1)*rowWidth];
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
	const size_t out_rows, const size_t out_cols,
	size_t overlap_y, size_t overlap_x,
	size_t in_pitch, size_t out_pitch,
	const size_t sharedRows, const size_t sharedCols
)
{
  extern __shared__ {{MAPOVERLAP_INPUT_TYPE}} sdata[]; // will also contain extra (overlap data)
	size_t xx = blockIdx.x * blockDim.x;
	size_t yy = blockIdx.y * blockDim.y;

	size_t x = xx + threadIdx.x;
	size_t y = yy + threadIdx.y;

	if (x < out_cols + overlap_x * 2 && y < out_rows + overlap_y * 2)
	{
		sdata[threadIdx.y * sharedCols + threadIdx.x] = skepu_input[y * in_pitch + x];

		// To load data in shared memory including neighbouring elements...
		for (size_t shared_y = threadIdx.y; shared_y < sharedRows; shared_y += blockDim.y)
		{
			for (size_t shared_x = threadIdx.x; shared_x < sharedCols; shared_x += blockDim.x)
			{
				sdata[shared_y * sharedCols + shared_x] = skepu_input[(yy + shared_y) * in_pitch + xx + shared_x];
			}
		}
	}

	__syncthreads();
	
	{{PROXIES_INIT}}

	if (x < out_cols && y < out_rows)
	{
		size_t skepu_w2 = out_cols;
		size_t skepu_i = y * out_cols + x;
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
static const std::string MatrixConvol3D_CU = R"~~~()~~~";

static const std::string MatrixConvol4D_CU = R"~~~()~~~";


std::string createMapOverlapKernelProgramHelper_CU(UserFunction &mapOverlapFunc, int dim, std::string dir, std::string kernelSource, std::string kernelTag)
{
	std::stringstream SSMapOverlapFuncArgs, SSKernelParamList;
	IndexCodeGen indexInfo = indexInitHelper_CU(mapOverlapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapOverlapFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CU(mapOverlapFunc, SSKernelParamList /*, "y * out_pitch + x"*/);
	if (!first) { SSMapOverlapFuncArgs << ", "; }
	first = false;
	
	if (dim == 1)
		SSMapOverlapFuncArgs << "{(int)overlap, 1, &sdata[skepu_tid + overlap]}";
	else if (dim == 2)
		SSMapOverlapFuncArgs << "{(int)overlap_x, (int)overlap_y, sharedCols, &sdata[(threadIdx.y + overlap_y) * sharedCols + (threadIdx.x + overlap_x)]}";
	
	SSKernelParamList << mapOverlapFunc.regionParam->templateInstantiationType() << " *skepu_input, ";
	
	auto argsInfo = handleRandomAccessAndUniforms_CU(mapOverlapFunc, SSMapOverlapFuncArgs, SSKernelParamList, first);
	
	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_" + kernelTag + "_" + mapOverlapFunc.uniqueName;
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << templateString(kernelSource,
	{
		{"{{MAPOVERLAP_INPUT_TYPE}}",    mapOverlapFunc.regionParam->templateInstantiationType()},
		{"{{KERNEL_NAME}}",              kernelName},
		{"{{FUNCTION_NAME_MAPOVERLAP}}", mapOverlapFunc.funcNameCUDA()},
		{"{{KERNEL_PARAMS}}",            SSKernelParamList.str()},
		{"{{MAPOVERLAP_ARGS}}",          SSMapOverlapFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}",        indexInfo.indexInit},
		{"{{OUTPUT_BINDINGS}}",          multiOutputAssign},
		{"{{PROXIES_UPDATE}}",           argsInfo.proxyInitializerInner},
		{"{{PROXIES_INIT}}",             argsInfo.proxyInitializer}
	});
	return kernelName;
}

std::string createMapOverlap1DKernelProgram_CU(UserFunction &mapOverlapFunc, std::string dir)
{
	return createMapOverlapKernelProgramHelper_CU(mapOverlapFunc, 1, dir, 
		MapOverlapKernel_CU + MapOverlapKernel_CU_Matrix_Row + MapOverlapKernel_CU_Matrix_Col + MapOverlapKernel_CU_Matrix_ColMulti, 
		"Overlap1DKernel");
}

std::string createMapOverlap2DKernelProgram_CU(UserFunction &mapOverlapFunc, std::string dir)
{
	return createMapOverlapKernelProgramHelper_CU(mapOverlapFunc, 2, dir, MatrixConvol2D_CU, "Overlap2DKernel");
}

std::string createMapOverlap3DKernelProgram_CU(UserFunction &mapOverlapFunc, std::string dir)
{
	return createMapOverlapKernelProgramHelper_CU(mapOverlapFunc, 3, dir, MatrixConvol3D_CU, "Overlap3DKernel");
}

std::string createMapOverlap4DKernelProgram_CU(UserFunction &mapOverlapFunc, std::string dir)
{
	return createMapOverlapKernelProgramHelper_CU(mapOverlapFunc, 4, dir, MatrixConvol4D_CU, "Overlap4DKernel");
}






