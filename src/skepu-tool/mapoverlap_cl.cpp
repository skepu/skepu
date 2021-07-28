#include <algorithm>

#include "code_gen.h"
#include "code_gen_cl.h"

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
static const std::string MapOverlapKernel_CL = R"~~~(
__kernel void {{KERNEL_NAME}}_Vector({{KERNEL_PARAMS}}
	__global {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* skepu_wrap, size_t skepu_n, size_t skepu_overlap, size_t out_offset,
	size_t out_numelements, int skepu_poly, {{MAPOVERLAP_INPUT_TYPE_OPENCL}} skepu_pad, __local {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* sdata
)
{
	size_t skepu_global_prng_id = get_global_id(0);
	size_t skepu_tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	{{CONTAINER_PROXIES}}
	{{CONTAINER_PROXIE_INNER}}

	if (skepu_poly == SKEPU_EDGE_PAD)
	{
		sdata[skepu_overlap + skepu_tid] = (skepu_i < skepu_n) ? {{INPUT_PARAM_NAME}}[skepu_i] : skepu_pad;
		if (skepu_tid < skepu_overlap)
			sdata[skepu_tid] = (get_group_id(0) == 0) ? skepu_pad : {{INPUT_PARAM_NAME}}[skepu_i - skepu_overlap];

		if (skepu_tid >= get_local_size(0) - skepu_overlap)
			sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != get_num_groups(0) - 1 && skepu_i + skepu_overlap < skepu_n) ? {{INPUT_PARAM_NAME}}[skepu_i + skepu_overlap] : skepu_pad;
	}
	else if (skepu_poly == SKEPU_EDGE_CYCLIC)
	{
		if (skepu_i < skepu_n)
			sdata[skepu_overlap + skepu_tid] = {{INPUT_PARAM_NAME}}[skepu_i];
		else if (skepu_i - skepu_n < skepu_overlap)
			sdata[skepu_overlap + skepu_tid] = skepu_wrap[skepu_overlap + skepu_i - skepu_n];
		else
			sdata[skepu_overlap + skepu_tid] = skepu_pad;

		if (skepu_tid < skepu_overlap)
			sdata[skepu_tid] = (get_group_id(0) == 0) ? skepu_wrap[skepu_tid] : {{INPUT_PARAM_NAME}}[skepu_i - skepu_overlap];

		if (skepu_tid >= get_local_size(0) - skepu_overlap)
			sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != get_num_groups(0) - 1 && skepu_i + skepu_overlap < skepu_n) ? {{INPUT_PARAM_NAME}}[skepu_i + skepu_overlap] : skepu_wrap[skepu_overlap + skepu_i + skepu_overlap - skepu_n];
	}
	else if (skepu_poly == SKEPU_EDGE_DUPLICATE)
	{
		sdata[skepu_overlap+skepu_tid] = (skepu_i < skepu_n) ? {{INPUT_PARAM_NAME}}[skepu_i] : {{INPUT_PARAM_NAME}}[skepu_n-1];
		if (skepu_tid < skepu_overlap)
			sdata[skepu_tid] = (get_group_id(0) == 0) ? {{INPUT_PARAM_NAME}}[0] : {{INPUT_PARAM_NAME}}[skepu_i-skepu_overlap];

		if (skepu_tid >= get_local_size(0) - skepu_overlap)
			sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != get_num_groups(0) - 1 && skepu_i + skepu_overlap < skepu_n) ? {{INPUT_PARAM_NAME}}[skepu_i + skepu_overlap] : {{INPUT_PARAM_NAME}}[skepu_n - 1];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (skepu_i >= out_offset && skepu_i < out_offset + out_numelements)
	{
		skepu_i = skepu_i - out_offset;
		const size_t skepu_base = 0;
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_i] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
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
static const std::string MapOverlapKernel_CL_Matrix_Row = R"~~~(
__kernel void {{KERNEL_NAME}}_MatRowWise({{KERNEL_PARAMS}}
	__global {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* skepu_wrap, size_t skepu_n, size_t skepu_overlap, size_t out_offset, size_t out_numelements,
	int skepu_poly, {{MAPOVERLAP_INPUT_TYPE_OPENCL}} skepu_pad, size_t blocksPerRow, size_t rowWidth, __local {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* sdata
)
{
	size_t skepu_global_prng_id = get_global_id(0);
	size_t skepu_tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex = 2 * skepu_overlap * (int)(get_group_id(0) / blocksPerRow);
	size_t skepu_tmp  = (get_group_id(0) % blocksPerRow);
	size_t skepu_tmp2 = (get_group_id(0) / blocksPerRow);
	{{CONTAINER_PROXIES}}
	{{CONTAINER_PROXIE_INNER}}

	if (skepu_poly == SKEPU_EDGE_PAD)
	{
		sdata[skepu_overlap + skepu_tid] = (skepu_i < skepu_n) ? {{INPUT_PARAM_NAME}}[skepu_i] : skepu_pad;
		if (skepu_tid < skepu_overlap)
			sdata[skepu_tid] = (skepu_tmp == 0) ? skepu_pad : {{INPUT_PARAM_NAME}}[skepu_i - skepu_overlap];

		if (skepu_tid >= (get_local_size(0) - skepu_overlap))
			sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (skepu_i + skepu_overlap < skepu_n) && skepu_tmp != (blocksPerRow - 1)) ? {{INPUT_PARAM_NAME}}[skepu_i+skepu_overlap] : skepu_pad;
	}
	else if (skepu_poly == SKEPU_EDGE_CYCLIC)
	{
		if (skepu_i < skepu_n)
			sdata[skepu_overlap + skepu_tid] = {{INPUT_PARAM_NAME}}[skepu_i];
		else if (skepu_i-skepu_n < skepu_overlap)
			sdata[skepu_overlap + skepu_tid] = skepu_wrap[(skepu_overlap + (skepu_i - skepu_n)) + wrapIndex];
		else
			sdata[skepu_overlap + skepu_tid] = skepu_pad;

		if (skepu_tid < skepu_overlap)
			sdata[skepu_tid] = (skepu_tmp == 0) ? skepu_wrap[skepu_tid + wrapIndex] : {{INPUT_PARAM_NAME}}[skepu_i - skepu_overlap];

		if (skepu_tid >= (get_local_size(0)-skepu_overlap))
			sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != (get_num_groups(0) - 1) && skepu_i+skepu_overlap < skepu_n && skepu_tmp != (blocksPerRow - 1))
				? {{INPUT_PARAM_NAME}}[skepu_i + skepu_overlap] : skepu_wrap[skepu_overlap + wrapIndex + (skepu_tid + skepu_overlap - get_local_size(0))];
	}
	else if (skepu_poly == SKEPU_EDGE_DUPLICATE)
	{
		sdata[skepu_overlap + skepu_tid] = (skepu_i < skepu_n) ? {{INPUT_PARAM_NAME}}[skepu_i] : {{INPUT_PARAM_NAME}}[skepu_n - 1];
		if (skepu_tid < skepu_overlap)
			sdata[skepu_tid] = (skepu_tmp == 0) ? {{INPUT_PARAM_NAME}}[skepu_tmp2 * rowWidth] : {{INPUT_PARAM_NAME}}[skepu_i - skepu_overlap];

		if (skepu_tid >= (get_local_size(0) - skepu_overlap))
			sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != (get_num_groups(0) - 1) && (skepu_i + skepu_overlap < skepu_n) && (skepu_tmp != (blocksPerRow - 1)))
				? {{INPUT_PARAM_NAME}}[skepu_i + skepu_overlap] : {{INPUT_PARAM_NAME}}[(skepu_tmp2 + 1)*rowWidth - 1];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if ((skepu_i >= out_offset) && (skepu_i < out_offset + out_numelements))
	{
		skepu_i = skepu_i - out_offset;
		const size_t skepu_base = 0;
		const size_t saved_skepu_i = skepu_i;
		skepu_i = skepu_i % rowWidth;
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
		skepu_i = saved_skepu_i;
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_i] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
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
static const std::string MapOverlapKernel_CL_Matrix_Col = R"~~~(
__kernel void {{KERNEL_NAME}}_MatColWise({{KERNEL_PARAMS}}
	__global {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* skepu_wrap, size_t skepu_n, size_t skepu_overlap, size_t out_offset, size_t out_numelements,
	int skepu_poly, {{MAPOVERLAP_INPUT_TYPE_OPENCL}} skepu_pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth, __local {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* sdata
	)
{
	size_t skepu_global_prng_id = get_global_id(0);
	size_t skepu_tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex = 2 * skepu_overlap * (int)(get_group_id(0) / blocksPerCol);
	size_t skepu_tmp = (get_group_id(0) % blocksPerCol);
	size_t skepu_tmp2 = (get_group_id(0) / blocksPerCol);
	size_t arrInd = (skepu_tid + skepu_tmp * get_local_size(0)) * rowWidth + skepu_tmp2;
	{{CONTAINER_PROXIES}}
	{{CONTAINER_PROXIE_INNER}}

	if (skepu_poly == SKEPU_EDGE_PAD)
	{
		sdata[skepu_overlap+skepu_tid] = (skepu_i < skepu_n) ? {{INPUT_PARAM_NAME}}[arrInd] : skepu_pad;
		if (skepu_tid < skepu_overlap)
			sdata[skepu_tid] = (skepu_tmp == 0) ? skepu_pad : {{INPUT_PARAM_NAME}}[(arrInd-(skepu_overlap*rowWidth))];

		if (skepu_tid >= (get_local_size(0)-skepu_overlap))
			sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd + (skepu_overlap * rowWidth)) < skepu_n && (skepu_tmp != (blocksPerCol - 1))) ? {{INPUT_PARAM_NAME}}[(arrInd+(skepu_overlap*rowWidth))] : skepu_pad;
	}
	else if (skepu_poly == SKEPU_EDGE_CYCLIC)
	{
		if (skepu_i < skepu_n)
			sdata[skepu_overlap + skepu_tid] = {{INPUT_PARAM_NAME}}[arrInd];
		else if (skepu_i-skepu_n < skepu_overlap)
			sdata[skepu_overlap+skepu_tid] = skepu_wrap[(skepu_overlap+(skepu_i-skepu_n))+ wrapIndex];
		else
			sdata[skepu_overlap+skepu_tid] = skepu_pad;

		if (skepu_tid < skepu_overlap)
			sdata[skepu_tid] = (skepu_tmp == 0) ? skepu_wrap[skepu_tid + wrapIndex] : {{INPUT_PARAM_NAME}}[(arrInd - (skepu_overlap * rowWidth))];

		if (skepu_tid >= (get_local_size(0)-skepu_overlap))
			sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != (get_num_groups(0) - 1) && (arrInd+(skepu_overlap * rowWidth)) < skepu_n && (skepu_tmp != (blocksPerCol - 1)))
				? {{INPUT_PARAM_NAME}}[(arrInd + (skepu_overlap * rowWidth))] : skepu_wrap[skepu_overlap + wrapIndex + (skepu_tid + skepu_overlap - get_local_size(0))];
	}
	else if (skepu_poly == SKEPU_EDGE_DUPLICATE)
	{
		sdata[skepu_overlap+skepu_tid] = (skepu_i < skepu_n) ? {{INPUT_PARAM_NAME}}[arrInd] : {{INPUT_PARAM_NAME}}[skepu_n - 1];
		if (skepu_tid < skepu_overlap)
			sdata[skepu_tid] = (skepu_tmp == 0) ? {{INPUT_PARAM_NAME}}[skepu_tmp2] : {{INPUT_PARAM_NAME}}[(arrInd - (skepu_overlap * rowWidth))];

		if (skepu_tid >= (get_local_size(0) - skepu_overlap))
			sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd + (skepu_overlap * rowWidth)) < skepu_n && (skepu_tmp != (blocksPerCol - 1)))
				? {{INPUT_PARAM_NAME}}[(arrInd + (skepu_overlap * rowWidth))] : {{INPUT_PARAM_NAME}}[skepu_tmp2 + (colWidth - 1) * rowWidth];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if ((arrInd >= out_offset) && (arrInd < out_offset + out_numelements))
	{
		skepu_i = arrInd - out_offset;
		const size_t skepu_base = 0;
		const size_t saved_skepu_i = skepu_i;
		skepu_i = skepu_i / rowWidth;
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
		skepu_i = saved_skepu_i;
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_i] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
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
static const std::string MapOverlapKernel_CL_Matrix_ColMulti = R"~~~(
__kernel void {{KERNEL_NAME}}_MatColWiseMulti({{KERNEL_PARAMS}}
	__global {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* skepu_wrap, size_t skepu_n, size_t skepu_overlap, size_t in_offset, size_t out_numelements,
	int skepu_poly, int deviceType, {{MAPOVERLAP_INPUT_TYPE_OPENCL}} skepu_pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
	__local {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* sdata
)
{
	size_t skepu_global_prng_id = get_global_id(0);
	size_t skepu_tid = get_local_id(0);
	size_t skepu_i   = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex = 2 * skepu_overlap * (int)(get_group_id(0)/blocksPerCol);
	size_t skepu_tmp  = (get_group_id(0) % blocksPerCol);
	size_t skepu_tmp2 = (get_group_id(0) / blocksPerCol);
	size_t arrInd = (skepu_tid + skepu_tmp * get_local_size(0)) * rowWidth + skepu_tmp2;
	{{CONTAINER_PROXIES}}
	{{CONTAINER_PROXIE_INNER}}

	if (skepu_poly == SKEPU_EDGE_PAD)
	{
		sdata[skepu_overlap + skepu_tid] = (skepu_i < skepu_n) ? {{INPUT_PARAM_NAME}}[arrInd + in_offset] : skepu_pad;
		if (deviceType == -1)
		{
			if (skepu_tid < skepu_overlap)
				sdata[skepu_tid] = (skepu_tmp == 0) ? skepu_pad : {{INPUT_PARAM_NAME}}[(arrInd - (skepu_overlap * rowWidth))];

			if (skepu_tid >= (get_local_size(0)-skepu_overlap))
				sdata[skepu_tid + 2 * skepu_overlap] = {{INPUT_PARAM_NAME}}[(arrInd + in_offset + (skepu_overlap * rowWidth))];
		}
		else if (deviceType == 0)
		{
			if (skepu_tid < skepu_overlap)
				sdata[skepu_tid] = {{INPUT_PARAM_NAME}}[arrInd];

			if (skepu_tid >= (get_local_size(0) - skepu_overlap))
				sdata[skepu_tid + 2 * skepu_overlap] = {{INPUT_PARAM_NAME}}[(arrInd + in_offset + (skepu_overlap * rowWidth))];
		}
		else if (deviceType == 1)
		{
			if (skepu_tid < skepu_overlap)
				sdata[skepu_tid] = {{INPUT_PARAM_NAME}}[arrInd];

			if (skepu_tid >= (get_local_size(0)-skepu_overlap))
				sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != (get_num_groups(0) - 1) && (arrInd + (skepu_overlap * rowWidth)) < skepu_n && (skepu_tmp != (blocksPerCol - 1)))
					? {{INPUT_PARAM_NAME}}[(arrInd + in_offset + (skepu_overlap * rowWidth))] : skepu_pad;
		}
	}
	else if (skepu_poly == SKEPU_EDGE_CYCLIC)
	{
		sdata[skepu_overlap + skepu_tid] = (skepu_i < skepu_n) ? {{INPUT_PARAM_NAME}}[arrInd + in_offset] : ((skepu_i-skepu_n < skepu_overlap) ? skepu_wrap[(skepu_i - skepu_n) + (skepu_overlap * skepu_tmp2)] : skepu_pad);
		if (deviceType == -1)
		{
			if (skepu_tid < skepu_overlap)
				sdata[skepu_tid] = (skepu_tmp == 0) ? skepu_wrap[skepu_tid+(skepu_overlap * skepu_tmp2)] : {{INPUT_PARAM_NAME}}[(arrInd - (skepu_overlap * rowWidth))];

			if (skepu_tid >= (get_local_size(0)-skepu_overlap))
				sdata[skepu_tid + 2 * skepu_overlap] = {{INPUT_PARAM_NAME}}[(arrInd + in_offset + (skepu_overlap * rowWidth))];
		}
		else if (deviceType == 0)
		{
			if (skepu_tid < skepu_overlap)
				sdata[skepu_tid] = {{INPUT_PARAM_NAME}}[arrInd];

			if (skepu_tid >= (get_local_size(0)-skepu_overlap))
				sdata[skepu_tid + 2 * skepu_overlap] = {{INPUT_PARAM_NAME}}[(arrInd+in_offset+(skepu_overlap*rowWidth))];
		}
		else if (deviceType == 1)
		{
			if (skepu_tid < skepu_overlap)
				sdata[skepu_tid] = {{INPUT_PARAM_NAME}}[arrInd];

			if (skepu_tid >= (get_local_size(0) - skepu_overlap))
				sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != (get_num_groups(0) - 1) && (arrInd + (skepu_overlap * rowWidth)) < skepu_n && (skepu_tmp != (blocksPerCol - 1)))
					? {{INPUT_PARAM_NAME}}[(arrInd + in_offset + (skepu_overlap*rowWidth))] : skepu_wrap[(skepu_overlap * skepu_tmp2) + (skepu_tid + skepu_overlap - get_local_size(0))];
		}
	}
	else if (skepu_poly == SKEPU_EDGE_DUPLICATE)
	{
		sdata[skepu_overlap+skepu_tid] = (skepu_i < skepu_n) ? {{INPUT_PARAM_NAME}}[arrInd + in_offset] : {{INPUT_PARAM_NAME}}[skepu_n + in_offset - 1];
		if (deviceType == -1)
		{
			if (skepu_tid < skepu_overlap)
				sdata[skepu_tid] = (skepu_tmp == 0) ? {{INPUT_PARAM_NAME}}[skepu_tmp2] : {{INPUT_PARAM_NAME}}[arrInd - skepu_overlap * rowWidth];

			if (skepu_tid >= get_local_size(0) - skepu_overlap)
				sdata[skepu_tid + 2 * skepu_overlap] = {{INPUT_PARAM_NAME}}[arrInd + in_offset + skepu_overlap * rowWidth];
		}
		else if (deviceType == 0)
		{
			if (skepu_tid < skepu_overlap)
				sdata[skepu_tid] = {{INPUT_PARAM_NAME}}[arrInd];

			if (skepu_tid >= get_local_size(0) - skepu_overlap)
				sdata[skepu_tid + 2 * skepu_overlap] = {{INPUT_PARAM_NAME}}[arrInd + in_offset + skepu_overlap * rowWidth];
		}
		else if (deviceType == 1)
		{
			if (skepu_tid < skepu_overlap)
				sdata[skepu_tid] = {{INPUT_PARAM_NAME}}[arrInd];

			if (skepu_tid >= get_local_size(0) - skepu_overlap)
				sdata[skepu_tid + 2 * skepu_overlap] = (get_group_id(0) != get_num_groups(0) - 1 && (arrInd + skepu_overlap * rowWidth < skepu_n) && (skepu_tmp != blocksPerCol - 1))
					? {{INPUT_PARAM_NAME}}[arrInd + in_offset + skepu_overlap * rowWidth] : {{INPUT_PARAM_NAME}}[skepu_tmp2 + in_offset + (colWidth - 1) * rowWidth];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if (arrInd < out_numelements)
	{
		skepu_i = arrInd;
		const size_t skepu_base = 0;
		const size_t saved_skepu_i = skepu_i;
		skepu_i = skepu_i / rowWidth;
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
		skepu_i = saved_skepu_i;
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_i] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
	}
}
)~~~";


const std::string Constructor1D = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	enum
	{
		KERNEL_VECTOR = 0,
		KERNEL_MATRIX_ROW,
		KERNEL_MATRIX_COL,
		KERNEL_MATRIX_COL_MULTI,
		KERNEL_COUNT
	};

	static cl_kernel kernels(size_t deviceID, size_t kerneltype, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8][KERNEL_COUNT]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID][kerneltype] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID][kerneltype];
	}

	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;

		std::string source = skepu::backend::cl_helpers::replaceSizeT(R"###({{OPENCL_KERNEL}})###");

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel_vector = clCreateKernel(program, "{{KERNEL_NAME}}_Vector", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D vector kernel '{{KERNEL_NAME}}'");

			cl_kernel kernel_matrix_row = clCreateKernel(program, "{{KERNEL_NAME}}_MatRowWise", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix row-wise kernel '{{KERNEL_NAME}}'");

			cl_kernel kernel_matrix_col = clCreateKernel(program, "{{KERNEL_NAME}}_MatColWise", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix col-wise kernel '{{KERNEL_NAME}}'");

			cl_kernel kernel_matrix_col_multi = clCreateKernel(program, "{{KERNEL_NAME}}_MatColWiseMulti", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix col-wise multi kernel '{{KERNEL_NAME}}'");

			kernels(counter, KERNEL_VECTOR,           &kernel_vector);
			kernels(counter, KERNEL_MATRIX_ROW,       &kernel_matrix_row);
			kernels(counter, KERNEL_MATRIX_COL,       &kernel_matrix_col);
			kernels(counter, KERNEL_MATRIX_COL_MULTI, &kernel_matrix_col_multi);
			counter++;
		}

		initialized = true;
	}

	{{TEMPLATE_HEADER}}
	static void mapOverlapVector
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		{{HOST_KERNEL_PARAMS}} {{SIZES_TUPLE_PARAM}}
		skepu::backend::DeviceMemPointer_CL<{{MAPOVERLAP_INPUT_TYPE}}> *skepu_wrap,
		size_t skepu_n, size_t skepu_overlap, size_t out_offset, size_t out_numelements, int skepu_poly, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_VECTOR);
		skepu::backend::cl_helpers::setKernelArgs(kernel, {{KERNEL_ARGS}} {{SIZE_ARGS}}
			skepu_wrap->getDeviceDataPointer(), skepu_n, skepu_overlap, out_offset, out_numelements, skepu_poly, skepu_pad);
		clSetKernelArg(kernel, {{KERNEL_ARG_COUNT}} + 7, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D vector kernel");
	}

	{{TEMPLATE_HEADER}}
	static void mapOverlapMatrixRowWise
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		{{HOST_KERNEL_PARAMS}} {{SIZES_TUPLE_PARAM}}
		skepu::backend::DeviceMemPointer_CL<{{MAPOVERLAP_INPUT_TYPE}}> *skepu_wrap,
		size_t skepu_n, size_t skepu_overlap, size_t out_offset, size_t out_numelements, int skepu_poly, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad, size_t blocksPerRow, size_t rowWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_ROW);
		skepu::backend::cl_helpers::setKernelArgs(kernel, {{KERNEL_ARGS}} {{SIZE_ARGS}}
			skepu_wrap->getDeviceDataPointer(), skepu_n, skepu_overlap, out_offset, out_numelements, skepu_poly, skepu_pad, blocksPerRow, rowWidth);
		clSetKernelArg(kernel, {{KERNEL_ARG_COUNT}} + 9, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix row-wise kernel");
	}

	{{TEMPLATE_HEADER}}
	static void mapOverlapMatrixColWise
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		{{HOST_KERNEL_PARAMS}} {{SIZES_TUPLE_PARAM}}
		skepu::backend::DeviceMemPointer_CL<{{MAPOVERLAP_INPUT_TYPE}}> *skepu_wrap,
		size_t skepu_n, size_t skepu_overlap, size_t out_offset, size_t out_numelements, int skepu_poly, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_COL);
		skepu::backend::cl_helpers::setKernelArgs(kernel, {{KERNEL_ARGS}} {{SIZE_ARGS}}
			skepu_wrap->getDeviceDataPointer(), skepu_n, skepu_overlap, out_offset, out_numelements, skepu_poly, skepu_pad, blocksPerCol, rowWidth, colWidth);
		clSetKernelArg(kernel, {{KERNEL_ARG_COUNT}} + 10, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix col-wise kernel");
	}

	{{TEMPLATE_HEADER}}
	static void mapOverlapMatrixColWiseMulti
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		{{HOST_KERNEL_PARAMS}} {{SIZES_TUPLE_PARAM}}
		skepu::backend::DeviceMemPointer_CL<{{MAPOVERLAP_INPUT_TYPE}}> *skepu_wrap,
		size_t skepu_n, size_t skepu_overlap, size_t in_offset, size_t out_numelements, int skepu_poly, int deviceType, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_COL_MULTI);
		skepu::backend::cl_helpers::setKernelArgs(kernel, {{KERNEL_ARGS}} {{SIZE_ARGS}}
			skepu_wrap->getDeviceDataPointer(), skepu_n, skepu_overlap, in_offset, out_numelements, skepu_poly, deviceType, skepu_pad, blocksPerCol, rowWidth, colWidth);
		clSetKernelArg(kernel, {{KERNEL_ARG_COUNT}} + 11, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix col-wise multi kernel");
	}
};
)~~~";

std::string createMapOverlap1DKernelProgram_CL(SkeletonInstance &instance, UserFunction &mapOverlapFunc, std::string dir)
{
	std::stringstream sourceStream, SSMapOverlapFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs;
	IndexCodeGen indexInfo = indexInitHelper_CL(mapOverlapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapOverlapFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CL(mapOverlapFunc, SSHostKernelParamList, SSKernelParamList, SSKernelArgs);
	handleRandomParam_CL(mapOverlapFunc, sourceStream, SSMapOverlapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	
	UserFunction::RegionParam& overlapParam = *mapOverlapFunc.regionParam;
	if (!first) { SSMapOverlapFuncArgs << ", "; }
	first = false;
	SSMapOverlapFuncArgs << "skepu_region";
	SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<" << overlapParam.resolvedTypeName << "> *skepu_input, ";
	SSKernelArgs << "skepu_input->getDeviceDataPointer(), ";
	SSKernelParamList << "__global " << overlapParam.typeNameOpenCL() << "* " << overlapParam.name << ", "; 
	
	std::string proxy = "skepu_region1d_" + transformToCXXIdentifier(overlapParam.resolvedTypeName) + " skepu_region = { .data = &sdata[skepu_tid+skepu_overlap], .oi = skepu_overlap, .stride = 1 };\n";
	
	handleUserTypesConstantsAndPrecision_CL({&mapOverlapFunc}, sourceStream);
	sourceStream << generateOpenCLRegion(1, overlapParam);
	auto argsInfo = handleRandomAccessAndUniforms_CL(mapOverlapFunc, SSMapOverlapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	proxyCodeGenHelper_CL(argsInfo.containerProxyTypes, sourceStream);
	sourceStream << generateUserFunctionCode_CL(mapOverlapFunc)
	             << MapOverlapKernel_CL << MapOverlapKernel_CL_Matrix_Row
	             << MapOverlapKernel_CL_Matrix_Col << MapOverlapKernel_CL_Matrix_ColMulti;

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_OverlapKernel_" + mapOverlapFunc.uniqueName;
	std::stringstream SSKernelArgCount;
	SSKernelArgCount << (mapOverlapFunc.numKernelArgsCL());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor1D,
	{
		{"{{OPENCL_KERNEL}}",            sourceStream.str()},
		{"{{MAPOVERLAP_INPUT_TYPE}}",    overlapParam.resolvedTypeName},
		{"{{MAPOVERLAP_INPUT_TYPE_OPENCL}}", overlapParam.typeNameOpenCL()},
		{"{{MAPOVERLAP_RESULT_TYPE}}",   mapOverlapFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",              kernelName},
		{"{{INPUT_PARAM_NAME}}",         overlapParam.name},
		{"{{FUNCTION_NAME_MAPOVERLAP}}", mapOverlapFunc.uniqueName},
		{"{{KERNEL_PARAMS}}",            SSKernelParamList.str()},
		{"{{MAPOVERLAP_ARGS}}",          SSMapOverlapFuncArgs.str()},
		{"{{HOST_KERNEL_PARAMS}}",       SSHostKernelParamList.str()},
		{"{{KERNEL_CLASS}}",             "CLWrapperClass_" + kernelName},
		{"{{KERNEL_ARGS}}",              SSKernelArgs.str()},
		{"{{KERNEL_ARG_COUNT}}",         SSKernelArgCount.str()},
		{"{{CONTAINER_PROXIES}}",        argsInfo.proxyInitializer + proxy},
		{"{{CONTAINER_PROXIE_INNER}}",   argsInfo.proxyInitializerInner},
		{"{{INDEX_INITIALIZER}}",        indexInfo.indexInit},
		{"{{SIZE_PARAMS}}",              indexInfo.sizeParams},
		{"{{SIZE_ARGS}}",                indexInfo.sizeArgs},
		{"{{SIZES_TUPLE_PARAM}}",        indexInfo.sizesTupleParam},
		{"{{TEMPLATE_HEADER}}",          indexInfo.templateHeader},
		{"{{MULTI_TYPE}}",               mapOverlapFunc.multiReturnTypeNameGPU()},
		{"{{USE_MULTIRETURN}}",          (mapOverlapFunc.multipleReturnTypes.size() > 0) ? "1" : "0"},
		{"{{OUTPUT_ASSIGN}}",            multiOutputAssign}
	});

	return kernelName;
}




/*!
* The mapoverlap OpenCL kernel to apply a user function on neighbourhood of each element in the matrix.
*/
static const std::string MatrixConvol2D_CL = R"~~~(
__kernel void {{KERNEL_NAME}}({{KERNEL_PARAMS}}
	size_t skepu_out_rows, size_t skepu_out_cols,
	size_t skepu_overlap_y, size_t skepu_overlap_x,
	size_t skepu_in_rows, size_t skepu_in_cols,
	size_t skepu_sharedRows, size_t skepu_sharedCols,
	int skepu_edge, {{MAPOVERLAP_INPUT_TYPE_OPENCL}} skepu_pad, __global {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* skepu_wrap,
	__local {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* skepu_sdata)
{
	size_t skepu_global_prng_id = get_global_id(0);
	size_t skepu_xx = ((size_t)(get_global_id(0) / get_local_size(0))) * get_local_size(0);
	size_t skepu_yy = ((size_t)(get_global_id(1) / get_local_size(1))) * get_local_size(1);
	size_t skepu_x = get_global_id(0);
	size_t skepu_y = get_global_id(1);
	{{CONTAINER_PROXIES}}
	{{CONTAINER_PROXIE_INNER}}

	if (skepu_x < skepu_out_cols + skepu_overlap_x * 2 && skepu_y < skepu_out_rows + skepu_overlap_y * 2)
	{
		size_t skepu_shared_x = get_local_id(0);
		size_t skepu_shared_y = get_local_id(1);
		while (skepu_shared_y < skepu_sharedRows)
		{
			while (skepu_shared_x < skepu_sharedCols)
			{
				size_t skepu_sharedIdx = skepu_shared_y * skepu_sharedCols + skepu_shared_x;
				int skepu_global_x = (skepu_xx + skepu_shared_x - skepu_overlap_x);
				int skepu_global_y = (skepu_yy + skepu_shared_y - skepu_overlap_y);
				
				if ((skepu_global_y >= 0 && skepu_global_y < skepu_in_rows) && (skepu_global_x >= 0 && skepu_global_x < skepu_in_cols))
					skepu_sdata[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[skepu_global_y * skepu_in_cols + skepu_global_x];
				else
				{
					if (skepu_edge == SKEPU_EDGE_PAD)
						skepu_sdata[skepu_sharedIdx] = skepu_pad;
					else if (skepu_edge == SKEPU_EDGE_DUPLICATE)
					{
						skepu_sdata[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
							clamp(skepu_global_y, 0, (int)skepu_in_rows - 1) * skepu_in_cols +
							clamp(skepu_global_x, 0, (int)skepu_in_cols - 1)];
					}
					else if (skepu_edge == SKEPU_EDGE_CYCLIC)
					{
						skepu_sdata[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
							((skepu_global_y + skepu_in_rows) % skepu_in_rows) * skepu_in_cols +
							((skepu_global_x + skepu_in_cols) % skepu_in_cols)];
					}
				}
				
				skepu_shared_x += get_local_size(0);
			}
			skepu_shared_x  = get_local_id(0);
			skepu_shared_y += get_local_size(1);
		}
	}
	
	/*
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (skepu_x == 0 && skepu_y == 0)
	{
		for (size_t yi = 0; yi < skepu_sharedCols; ++yi)
		{
			for (size_t xi = 0; xi < skepu_sharedRows; ++xi)
			{
				printf("%d\t", skepu_sdata[yi * skepu_sharedCols + xi]);
			}
			printf("\n");
		}
	}
	*/

	barrier(CLK_LOCAL_MEM_FENCE);
	if (skepu_x < skepu_out_cols && skepu_y < skepu_out_rows)
	{
		size_t skepu_i = skepu_y * skepu_out_cols + skepu_x;
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_i] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
	}
}
)~~~";


const std::string Constructor2D = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	static cl_kernel kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
	}

	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;

		std::string source = skepu::backend::cl_helpers::replaceSizeT(R"###({{OPENCL_KERNEL}})###");

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "{{KERNEL_NAME}}", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 2D kernel '{{KERNEL_NAME}}'");

			kernels(counter++, &kernel);
		}

		initialized = true;
	}

	{{TEMPLATE_HEADER}}
	static void mapOverlap2D
	(
		size_t deviceID, size_t localSize[2], size_t globalSize[2],
		{{HOST_KERNEL_PARAMS}} {{SIZES_TUPLE_PARAM}}
		size_t out_rows, size_t out_cols, size_t skepu_overlap_y, size_t skepu_overlap_x,
		size_t in_rows, size_t in_cols, size_t sharedRows, size_t sharedCols,
		int skepu_edge, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad,
		skepu::backend::DeviceMemPointer_CL<{{MAPOVERLAP_INPUT_TYPE}}> *skepu_wrap,
		size_t sharedMemSize
	)
	{
		skepu::backend::cl_helpers::setKernelArgs(kernels(deviceID), {{KERNEL_ARGS}}
			out_rows, out_cols, skepu_overlap_y, skepu_overlap_x, in_rows, in_cols, sharedRows, sharedCols,
			skepu_edge, skepu_pad, skepu_wrap->getDeviceDataPointer());
		clSetKernelArg(kernels(deviceID), {{KERNEL_ARG_COUNT}} + 11, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernels(deviceID), 2, NULL, globalSize, localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 2D kernel");
	}
};
)~~~";

/*
handleRegion_CL(UserFunction &mapOverlapFunc, std::stringstream &SSMapOverlapFuncArgs, std::stringstream &SSHostKernelParamList, std::stringstream &SSKernelParamList, std::stringstream &SSKernelArgs, first)
{
	
}*/

std::string createMapOverlap2DKernelProgram_CL(SkeletonInstance &instance, UserFunction &mapOverlapFunc, std::string dir)
{
	std::stringstream sourceStream, SSMapOverlapFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs;
	std::string indexInit = "";
	if (mapOverlapFunc.indexed2D)
	{
		indexInit = "index2_t skepu_index = { .row = x, .col = y };";
		SSMapOverlapFuncArgs << "skepu_index";
	}
	IndexCodeGen indexInfo = indexInitHelper_CL(mapOverlapFunc);
	bool first = !indexInfo.hasIndex;
	std::string multiOutputAssign = handleOutputs_CL(mapOverlapFunc, SSHostKernelParamList, SSKernelParamList, SSKernelArgs);
	handleRandomParam_CL(mapOverlapFunc, sourceStream, SSMapOverlapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	
	UserFunction::RegionParam& overlapParam = *mapOverlapFunc.regionParam;
	if (!first) { SSMapOverlapFuncArgs << ", "; }
	first = false;
	SSMapOverlapFuncArgs << "skepu_region";
	SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<" << overlapParam.resolvedTypeName << "> *skepu_input, ";
	SSKernelArgs << "skepu_input->getDeviceDataPointer(), ";
	SSKernelParamList << "__global " << overlapParam.typeNameOpenCL() << "* " << overlapParam.name << ", "; 
	
	std::string proxy = "skepu_region2d_" + transformToCXXIdentifier(overlapParam.resolvedTypeName) + " skepu_region = { .data = &skepu_sdata[(get_local_id(1) + skepu_overlap_y) * skepu_sharedCols + (get_local_id(0) + skepu_overlap_x)], .oi = skepu_overlap_y, .oj = skepu_overlap_x, .stride = skepu_sharedCols };\n";
	
	handleUserTypesConstantsAndPrecision_CL({&mapOverlapFunc}, sourceStream);
	sourceStream << generateOpenCLRegion(2, overlapParam);
	auto argsInfo = handleRandomAccessAndUniforms_CL(mapOverlapFunc, SSMapOverlapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	proxyCodeGenHelper_CL(argsInfo.containerProxyTypes, sourceStream);
	sourceStream << generateUserFunctionCode_CL(mapOverlapFunc) << MatrixConvol2D_CL;

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_Overlap2DKernel_" + mapOverlapFunc.uniqueName;
	std::stringstream SSKernelArgCount;
	SSKernelArgCount << (mapOverlapFunc.numKernelArgsCL());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor2D,
	{
		{"{{OPENCL_KERNEL}}",            sourceStream.str()},
		{"{{MAPOVERLAP_INPUT_TYPE}}",    overlapParam.resolvedTypeName},
		{"{{MAPOVERLAP_INPUT_TYPE_OPENCL}}", overlapParam.typeNameOpenCL()},
		{"{{MAPOVERLAP_RESULT_TYPE}}",   mapOverlapFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",              kernelName},
		{"{{INPUT_PARAM_NAME}}",         overlapParam.name},
		{"{{FUNCTION_NAME_MAPOVERLAP}}", mapOverlapFunc.uniqueName},
		{"{{KERNEL_PARAMS}}",            SSKernelParamList.str()},
		{"{{MAPOVERLAP_ARGS}}",          SSMapOverlapFuncArgs.str()},
		{"{{HOST_KERNEL_PARAMS}}",       SSHostKernelParamList.str()},
		{"{{KERNEL_CLASS}}",             "CLWrapperClass_" + kernelName},
		{"{{KERNEL_ARGS}}",              SSKernelArgs.str()},
		{"{{KERNEL_ARG_COUNT}}",         SSKernelArgCount.str()},
		{"{{CONTAINER_PROXIES}}",        argsInfo.proxyInitializer + proxy},
		{"{{CONTAINER_PROXIE_INNER}}",   argsInfo.proxyInitializerInner},
		{"{{INDEX_INITIALIZER}}",        indexInit},
		{"{{SIZE_PARAMS}}",              indexInfo.sizeParams},
		{"{{SIZE_ARGS}}",                indexInfo.sizeArgs},
		{"{{SIZES_TUPLE_PARAM}}",        indexInfo.sizesTupleParam},
		{"{{TEMPLATE_HEADER}}",          indexInfo.templateHeader},
		{"{{MULTI_TYPE}}",               mapOverlapFunc.multiReturnTypeNameGPU()},
		{"{{USE_MULTIRETURN}}",          (mapOverlapFunc.multipleReturnTypes.size() > 0) ? "1" : "0"},
		{"{{OUTPUT_ASSIGN}}",            multiOutputAssign}
	});

	return kernelName;
}







/*!
* The mapoverlap OpenCL kernel to apply a user function on neighbourhood of each element in the matrix.
*/
static const std::string MatrixConvol3D_CL = R"~~~(
__kernel void {{KERNEL_NAME}}({{KERNEL_PARAMS}}
	size_t skepu_out_size_i, size_t skepu_out_size_j, size_t skepu_out_size_k,
	size_t skepu_overlap_i, size_t skepu_overlap_j, size_t skepu_overlap_k,
	size_t skepu_in_size_i, size_t skepu_in_size_j, size_t skepu_in_size_k,
	size_t skepu_shared_size_i, size_t skepu_shared_size_j, size_t skepu_shared_size_k,
	int skepu_edge, {{MAPOVERLAP_INPUT_TYPE_OPENCL}} skepu_pad, __global {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* skepu_wrap,
	__local {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* skepu_sdata)
{
	size_t skepu_global_prng_id = get_global_id(0);
	size_t skepu_kk = ((size_t)(get_global_id(0) / get_local_size(0))) * get_local_size(0);
	size_t skepu_jj = ((size_t)(get_global_id(1) / get_local_size(1))) * get_local_size(1);
	size_t skepu_ii = ((size_t)(get_global_id(2) / get_local_size(2))) * get_local_size(2);
	size_t skepu_k = get_global_id(0);
	size_t skepu_j = get_global_id(1);
	size_t skepu_i = get_global_id(2);
	{{CONTAINER_PROXIES}}
	{{CONTAINER_PROXIE_INNER}}

	if (skepu_i < skepu_out_size_i + skepu_overlap_i * 2 && skepu_j < skepu_out_size_j + skepu_overlap_j * 2 && skepu_k < skepu_out_size_k + skepu_overlap_k * 2)
	{
		size_t skepu_shared_k = get_local_id(0);
		size_t skepu_shared_j = get_local_id(1);
		size_t skepu_shared_i = get_local_id(2);
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
						skepu_sdata[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[skepu_global_i * skepu_in_size_j * skepu_in_size_k + skepu_global_j * skepu_in_size_k + skepu_global_k];
					else
					{
						if (skepu_edge == SKEPU_EDGE_PAD)
							skepu_sdata[skepu_sharedIdx] = skepu_pad;
						else if (skepu_edge == SKEPU_EDGE_DUPLICATE)
						{
							skepu_sdata[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
								clamp(skepu_global_i, 0, (int)skepu_in_size_i - 1) * skepu_in_size_j * skepu_in_size_k +
								clamp(skepu_global_j, 0, (int)skepu_in_size_j - 1) * skepu_in_size_k +
								clamp(skepu_global_k, 0, (int)skepu_in_size_k - 1)];
						}
						else if (skepu_edge == SKEPU_EDGE_CYCLIC)
						{
							skepu_sdata[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
								((skepu_global_i + skepu_in_size_i) % skepu_in_size_i) * skepu_in_size_j * skepu_in_size_k +
								((skepu_global_j + skepu_in_size_j) % skepu_in_size_j) * skepu_in_size_k +
								((skepu_global_k + skepu_in_size_k) % skepu_in_size_k)];
						}
					}
					
					skepu_shared_k += get_local_size(0);
				}
				skepu_shared_k  = get_local_id(0);
				skepu_shared_j += get_local_size(1);
			}
			skepu_shared_j  = get_local_id(1);
			skepu_shared_i += get_local_size(2);
		}
	}
	
/*
	barrier(CLK_LOCAL_MEM_FENCE);

	if (skepu_i == 0 && skepu_j == 0 && skepu_k == 0)
	{
		for (size_t dbg_j = 0; dbg_j < skepu_shared_size_j; ++dbg_j)
		{
			for (size_t dbg_i = 0; dbg_i < skepu_shared_size_i; ++dbg_i)
			{
				for (size_t dbg_k = 0; dbg_k < skepu_shared_size_k; ++dbg_k)
				{
					printf("%d\t", skepu_sdata[dbg_i * skepu_shared_size_j * skepu_shared_size_k + dbg_j * skepu_shared_size_k + dbg_k]);
				}
				printf(" | ");
			}
			printf("\n");
		}
	}
	*/
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if (skepu_i < skepu_out_size_i && skepu_j < skepu_out_size_j && skepu_k < skepu_out_size_k)
	{
		skepu_i = skepu_i * skepu_out_size_j * skepu_out_size_k + skepu_j * skepu_out_size_k + skepu_k;
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_i] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
	}
}
)~~~";


const std::string Constructor3D = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	static cl_kernel kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
	}

	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;

		std::string source = skepu::backend::cl_helpers::replaceSizeT(R"###({{OPENCL_KERNEL}})###");

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "{{KERNEL_NAME}}", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 3D kernel '{{KERNEL_NAME}}'");

			kernels(counter++, &kernel);
		}

		initialized = true;
	}

	{{TEMPLATE_HEADER}}
	static void mapOverlap3D
	(
		size_t skepu_deviceID, size_t skepu_localSize[3], size_t skepu_globalSize[3],
		{{HOST_KERNEL_PARAMS}} {{SIZES_TUPLE_PARAM}}
		size_t skepu_out_i, size_t skepu_out_j, size_t skepu_out_k,
		size_t skepu_overlap_i, size_t skepu_overlap_j, size_t skepu_overlap_k,
		size_t skepu_in_size_i, size_t skepu_in_size_j, size_t skepu_in_size_k,
		size_t skepu_shared_size_i, size_t skepu_shared_size_j, size_t skepu_shared_size_k,
		int skepu_edge, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad,
		skepu::backend::DeviceMemPointer_CL<{{MAPOVERLAP_INPUT_TYPE}}> *skepu_wrap,
		size_t skepu_sharedMemSize
	)
	{
		skepu::backend::cl_helpers::setKernelArgs(kernels(skepu_deviceID), {{KERNEL_ARGS}}
			skepu_out_i, skepu_out_j, skepu_out_k,
			skepu_overlap_i, skepu_overlap_j, skepu_overlap_k,
	 		skepu_in_size_i, skepu_in_size_j, skepu_in_size_k,
			skepu_shared_size_i, skepu_shared_size_j, skepu_shared_size_k,
			skepu_edge, skepu_pad, skepu_wrap->getDeviceDataPointer()
		);
		clSetKernelArg(kernels(skepu_deviceID), {{KERNEL_ARG_COUNT}} + 15, skepu_sharedMemSize, NULL);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(),
			kernels(skepu_deviceID), 3, NULL, skepu_globalSize, skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching MapOverlap 3D kernel");
	}
};
)~~~";

/*
handleRegion_CL(UserFunction &mapOverlapFunc, std::stringstream &SSMapOverlapFuncArgs, std::stringstream &SSHostKernelParamList, std::stringstream &SSKernelParamList, std::stringstream &SSKernelArgs, first)
{
	
}*/

std::string createMapOverlap3DKernelProgram_CL(SkeletonInstance &instance, UserFunction &mapOverlapFunc, std::string dir)
{
	std::stringstream sourceStream, SSMapOverlapFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs;
	std::string indexInit = "";
	if (mapOverlapFunc.indexed3D)
	{
		indexInit = "index3_t skepu_index = { .i = i, .j = j, .k = k };";
		SSMapOverlapFuncArgs << "skepu_index";
	}
	IndexCodeGen indexInfo = indexInitHelper_CL(mapOverlapFunc);
	bool first = !indexInfo.hasIndex;
	std::string multiOutputAssign = handleOutputs_CL(mapOverlapFunc, SSHostKernelParamList, SSKernelParamList, SSKernelArgs);
	handleRandomParam_CL(mapOverlapFunc, sourceStream, SSMapOverlapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	
	UserFunction::RegionParam& overlapParam = *mapOverlapFunc.regionParam;
	if (!first) { SSMapOverlapFuncArgs << ", "; }
	first = false;
	SSMapOverlapFuncArgs << "skepu_region";
	SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<" << overlapParam.resolvedTypeName << "> *skepu_input, ";
	SSKernelArgs << "skepu_input->getDeviceDataPointer(), ";
	SSKernelParamList << "__global " << overlapParam.typeNameOpenCL() << "* " << overlapParam.name << ", "; 
	
	std::string proxy = "skepu_region3d_" + transformToCXXIdentifier(overlapParam.resolvedTypeName) + " skepu_region = { .data = &skepu_sdata[(get_local_id(2) + skepu_overlap_i) * skepu_shared_size_j * skepu_shared_size_k + (get_local_id(1) + skepu_overlap_j) * skepu_shared_size_k + (get_local_id(0) + skepu_overlap_k)], .oi = skepu_overlap_i, .oj = skepu_overlap_j, .ok = skepu_overlap_k, .stride1 = skepu_shared_size_j, .stride2 = skepu_shared_size_k };\n";
	
	handleUserTypesConstantsAndPrecision_CL({&mapOverlapFunc}, sourceStream);
	sourceStream << generateOpenCLRegion(3, overlapParam);
	auto argsInfo = handleRandomAccessAndUniforms_CL(mapOverlapFunc, SSMapOverlapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	proxyCodeGenHelper_CL(argsInfo.containerProxyTypes, sourceStream);
	sourceStream << generateUserFunctionCode_CL(mapOverlapFunc) << MatrixConvol3D_CL;

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_Overlap3DKernel_" + mapOverlapFunc.uniqueName;
	std::stringstream SSKernelArgCount;
	SSKernelArgCount << (mapOverlapFunc.numKernelArgsCL());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor3D,
	{
		{"{{OPENCL_KERNEL}}",            sourceStream.str()},
		{"{{MAPOVERLAP_INPUT_TYPE}}",    overlapParam.resolvedTypeName},
		{"{{MAPOVERLAP_INPUT_TYPE_OPENCL}}", overlapParam.typeNameOpenCL()},
		{"{{MAPOVERLAP_RESULT_TYPE}}",   mapOverlapFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",              kernelName},
		{"{{INPUT_PARAM_NAME}}",         overlapParam.name},
		{"{{FUNCTION_NAME_MAPOVERLAP}}", mapOverlapFunc.uniqueName},
		{"{{KERNEL_PARAMS}}",            SSKernelParamList.str()},
		{"{{MAPOVERLAP_ARGS}}",          SSMapOverlapFuncArgs.str()},
		{"{{HOST_KERNEL_PARAMS}}",       SSHostKernelParamList.str()},
		{"{{KERNEL_CLASS}}",             "CLWrapperClass_" + kernelName},
		{"{{KERNEL_ARGS}}",              SSKernelArgs.str()},
		{"{{KERNEL_ARG_COUNT}}",         SSKernelArgCount.str()},
		{"{{CONTAINER_PROXIES}}",        argsInfo.proxyInitializer + proxy},
		{"{{CONTAINER_PROXIE_INNER}}",   argsInfo.proxyInitializerInner},
		{"{{INDEX_INITIALIZER}}",        indexInit},
		{"{{SIZE_PARAMS}}",              indexInfo.sizeParams},
		{"{{SIZE_ARGS}}",                indexInfo.sizeArgs},
		{"{{SIZES_TUPLE_PARAM}}",        indexInfo.sizesTupleParam},
		{"{{TEMPLATE_HEADER}}",          indexInfo.templateHeader},
		{"{{MULTI_TYPE}}",               mapOverlapFunc.multiReturnTypeNameGPU()},
		{"{{USE_MULTIRETURN}}",          (mapOverlapFunc.multipleReturnTypes.size() > 0) ? "1" : "0"},
		{"{{OUTPUT_ASSIGN}}",            multiOutputAssign}
	});

	return kernelName;
}















/*!
* The mapoverlap OpenCL kernel to apply a user function on neighbourhood of each element in the matrix.
*/
static const std::string MatrixConvol4D_CL = R"~~~(
__kernel void {{KERNEL_NAME}}({{KERNEL_PARAMS}}
	size_t skepu_out_size_i, size_t skepu_out_size_j, size_t skepu_out_size_k, size_t skepu_out_size_l,
	size_t skepu_overlap_i, size_t skepu_overlap_j, size_t skepu_overlap_k, size_t skepu_overlap_l,
	size_t skepu_in_size_i, size_t skepu_in_size_j, size_t skepu_in_size_k, size_t skepu_in_size_l,
	size_t skepu_shared_size_i, size_t skepu_shared_size_j, size_t skepu_shared_size_k, size_t skepu_shared_size_l,
	int skepu_edge, {{MAPOVERLAP_INPUT_TYPE_OPENCL}} skepu_pad, __global {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* skepu_wrap,
	__local {{MAPOVERLAP_INPUT_TYPE_OPENCL}}* skepu_sdata)
{
	size_t skepu_global_prng_id = get_global_id(0);
	const size_t skepu_ll = ((size_t)(get_global_id(0) / get_local_size(0))) * get_local_size(0);
	const size_t skepu_kk = ((size_t)(get_global_id(0) / get_local_size(0))) * get_local_size(0);
	const size_t skepu_jj = ((size_t)(get_global_id(1) / get_local_size(1))) * get_local_size(1);
	const size_t skepu_ii = ((size_t)(get_global_id(2) / get_local_size(2))) * get_local_size(2);
	
	const size_t skepu_l = get_global_id(0) % skepu_out_size_l;
	const size_t skepu_k = get_global_id(0) / skepu_out_size_l;
	const size_t skepu_j = get_global_id(1);
	      size_t skepu_i = get_global_id(2);
	
	const size_t skepu_increment_l = get_local_size(0) / skepu_out_size_l;
	const size_t skepu_increment_k = get_local_size(0) / skepu_increment_l;
	
//	if (get_group_id(0) != 0 || get_group_id(1) != 0 || get_group_id(2) != 0)
//		return;
	
	{{CONTAINER_PROXIES}}
	{{CONTAINER_PROXIE_INNER}}

	if (skepu_i < skepu_out_size_i + skepu_overlap_i * 2 && skepu_j < skepu_out_size_j + skepu_overlap_j * 2 && skepu_k < skepu_out_size_k + skepu_overlap_k * 2 && skepu_l < skepu_out_size_l + skepu_overlap_l * 2)
	{
		size_t skepu_shared_l = get_local_id(0) % skepu_out_size_l;
		size_t skepu_shared_k = get_local_id(0) / skepu_out_size_l;
		size_t skepu_shared_j = get_local_id(1);
		size_t skepu_shared_i = get_local_id(2);
		while (skepu_shared_i < skepu_shared_size_i)
		{
			while (skepu_shared_j < skepu_shared_size_j)
			{
				while (skepu_shared_k < skepu_shared_size_k)
				{
					while (skepu_shared_l < skepu_shared_size_l)
					{
						size_t skepu_sharedIdx = skepu_shared_i * skepu_shared_size_j * skepu_shared_size_k * skepu_shared_size_l + skepu_shared_j * skepu_shared_size_k * skepu_shared_size_l + skepu_shared_k * skepu_shared_size_l + skepu_shared_l;
						int skepu_global_l = (skepu_ll + skepu_shared_l - skepu_overlap_l);
						int skepu_global_k = (skepu_kk + skepu_shared_k - skepu_overlap_k);
						int skepu_global_j = (skepu_jj + skepu_shared_j - skepu_overlap_j);
						int skepu_global_i = (skepu_ii + skepu_shared_i - skepu_overlap_i);
						
						if ((skepu_global_i >= 0 && skepu_global_i < skepu_in_size_i) && (skepu_global_j >= 0 && skepu_global_j < skepu_in_size_j)
						 && (skepu_global_k >= 0 && skepu_global_k < skepu_in_size_k) && (skepu_global_l >= 0 && skepu_global_l < skepu_in_size_l))
							skepu_sdata[skepu_sharedIdx] = get_group_id(0);//{{INPUT_PARAM_NAME}}[skepu_global_i * skepu_in_size_j * skepu_in_size_k * skepu_in_size_l + skepu_global_j * skepu_in_size_k * skepu_in_size_l + skepu_global_k * skepu_in_size_l + skepu_global_l];
						else
						{
							if (skepu_edge == SKEPU_EDGE_PAD)
								skepu_sdata[skepu_sharedIdx] = skepu_pad;
							else if (skepu_edge == SKEPU_EDGE_DUPLICATE)
							{
								skepu_sdata[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
									clamp(skepu_global_i, 0, (int)skepu_in_size_i - 1) * skepu_in_size_j * skepu_in_size_k * skepu_in_size_l +
									clamp(skepu_global_j, 0, (int)skepu_in_size_j - 1) * skepu_in_size_k * skepu_in_size_l +
									clamp(skepu_global_k, 0, (int)skepu_in_size_k - 1) * skepu_in_size_l +
									clamp(skepu_global_l, 0, (int)skepu_in_size_l - 1)];
							}
							else if (skepu_edge == SKEPU_EDGE_CYCLIC)
							{
								skepu_sdata[skepu_sharedIdx] = {{INPUT_PARAM_NAME}}[
									((skepu_global_i + skepu_in_size_i) % skepu_in_size_i) * skepu_in_size_j * skepu_in_size_k * skepu_in_size_l +
									((skepu_global_j + skepu_in_size_j) % skepu_in_size_j) * skepu_in_size_k * skepu_in_size_l +
									((skepu_global_k + skepu_in_size_k) % skepu_in_size_k) * skepu_in_size_l +
									((skepu_global_l + skepu_in_size_l) % skepu_in_size_l)];
							}
						}
						
						skepu_shared_l += skepu_increment_l;
					}
					skepu_shared_l = get_local_id(0) % skepu_out_size_l;
					skepu_shared_k += skepu_increment_k;
				}
				skepu_shared_k  = get_local_id(0) / skepu_out_size_l;
				skepu_shared_j += get_local_size(1);
			}
			skepu_shared_j  = get_local_id(1);
			skepu_shared_i += get_local_size(2);
		}
	}
	
/*
	barrier(CLK_LOCAL_MEM_FENCE);

	if (skepu_i == 0 && skepu_j == 0 && skepu_k == 0 && skepu_l == 0)
	{
		for (size_t dbg_i = 0; dbg_i < skepu_shared_size_i; ++dbg_i)
		{
			for (size_t dbg_k = 0; dbg_k < skepu_shared_size_k; ++dbg_k)
			{
				for (size_t dbg_j = 0; dbg_j < skepu_shared_size_j; ++dbg_j)
				{
					for (size_t dbg_l = 0; dbg_l < skepu_shared_size_l; ++dbg_l)
					{
						printf("%d\t", skepu_sdata[dbg_i * skepu_shared_size_j * skepu_shared_size_k * skepu_shared_size_l + dbg_j * skepu_shared_size_k * skepu_shared_size_l + dbg_k * skepu_shared_size_l + dbg_l]);
					}
					printf(" | ");
				}
				printf("\n");
			}
			printf("---------------------------------\n");
		}
	}
	*/
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if (skepu_i < skepu_out_size_i && skepu_j < skepu_out_size_j && skepu_k < skepu_out_size_k && skepu_l < skepu_out_size_l)
	{
		skepu_i = skepu_i * skepu_out_size_j * skepu_out_size_k * skepu_out_size_l + skepu_j * skepu_out_size_k * skepu_out_size_l + skepu_k * skepu_out_size_l + skepu_l;
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_i] = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAPOVERLAP}}({{MAPOVERLAP_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
	}
}
)~~~";


const std::string Constructor4D = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	static cl_kernel kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
	}

	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;

		std::string source = skepu::backend::cl_helpers::replaceSizeT(R"###({{OPENCL_KERNEL}})###");

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "{{KERNEL_NAME}}", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 4D kernel '{{KERNEL_NAME}}'");

			kernels(counter++, &kernel);
		}

		initialized = true;
	}

	{{TEMPLATE_HEADER}}
	static void mapOverlap4D
	(
		size_t skepu_deviceID, size_t skepu_localSize[3], size_t skepu_globalSize[3],
		{{HOST_KERNEL_PARAMS}} {{SIZES_TUPLE_PARAM}}
		size_t skepu_out_i, size_t skepu_out_j, size_t skepu_out_k, size_t skepu_out_l,
		size_t skepu_overlap_i, size_t skepu_overlap_j, size_t skepu_overlap_k, size_t skepu_overlap_l,
		size_t skepu_in_size_i, size_t skepu_in_size_j, size_t skepu_in_size_k, size_t skepu_in_size_l,
		size_t skepu_shared_size_i, size_t skepu_shared_size_j, size_t skepu_shared_size_k, size_t skepu_shared_size_l,
		int skepu_edge, {{MAPOVERLAP_INPUT_TYPE}} skepu_pad,
		skepu::backend::DeviceMemPointer_CL<{{MAPOVERLAP_INPUT_TYPE}}> *skepu_wrap,
		size_t skepu_sharedMemSize
	)
	{
		skepu::backend::cl_helpers::setKernelArgs(kernels(skepu_deviceID), {{KERNEL_ARGS}}
			skepu_out_i, skepu_out_j, skepu_out_k, skepu_out_l,
			skepu_overlap_i, skepu_overlap_j, skepu_overlap_k, skepu_overlap_l,
	 		skepu_in_size_i, skepu_in_size_j, skepu_in_size_k, skepu_in_size_l,
			skepu_shared_size_i, skepu_shared_size_j, skepu_shared_size_k, skepu_shared_size_l,
			skepu_edge, skepu_pad, skepu_wrap->getDeviceDataPointer()
		);
		clSetKernelArg(kernels(skepu_deviceID), {{KERNEL_ARG_COUNT}} + 19, skepu_sharedMemSize, NULL);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(),
			kernels(skepu_deviceID), 3, NULL, skepu_globalSize, skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching MapOverlap 4D kernel");
	}
};
)~~~";

/*
handleRegion_CL(UserFunction &mapOverlapFunc, std::stringstream &SSMapOverlapFuncArgs, std::stringstream &SSHostKernelParamList, std::stringstream &SSKernelParamList, std::stringstream &SSKernelArgs, first)
{
	
}*/

std::string createMapOverlap4DKernelProgram_CL(SkeletonInstance &instance, UserFunction &mapOverlapFunc, std::string dir)
{
	std::stringstream sourceStream, SSMapOverlapFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs;
	std::string indexInit = "";
	if (mapOverlapFunc.indexed4D)
	{
		indexInit = "index4_t skepu_index = { .i = i, .j = j, .k = k, .l = l };";
		SSMapOverlapFuncArgs << "skepu_index";
	}
	IndexCodeGen indexInfo = indexInitHelper_CL(mapOverlapFunc);
	bool first = !indexInfo.hasIndex;
	std::string multiOutputAssign = handleOutputs_CL(mapOverlapFunc, SSHostKernelParamList, SSKernelParamList, SSKernelArgs);
	handleRandomParam_CL(mapOverlapFunc, sourceStream, SSMapOverlapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	
	UserFunction::RegionParam& overlapParam = *mapOverlapFunc.regionParam;
	if (!first) { SSMapOverlapFuncArgs << ", "; }
	first = false;
	SSMapOverlapFuncArgs << "skepu_region";
	SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<" << overlapParam.resolvedTypeName << "> *skepu_input, ";
	SSKernelArgs << "skepu_input->getDeviceDataPointer(), ";
	SSKernelParamList << "__global " << overlapParam.typeNameOpenCL() << "* " << overlapParam.name << ", "; 
	
	std::string proxy = "skepu_region4d_" + transformToCXXIdentifier(overlapParam.resolvedTypeName) + " skepu_region = { .data = &skepu_sdata [(get_local_id(2) + skepu_overlap_i) * skepu_shared_size_j * skepu_shared_size_k * skepu_shared_size_l + (get_local_id(1) + skepu_overlap_j) * skepu_shared_size_k * skepu_shared_size_l + ((get_local_id(0) / skepu_out_size_l) + skepu_overlap_k) * skepu_shared_size_l + ((get_local_id(0) % skepu_out_size_l) + skepu_overlap_l)], .oi = skepu_overlap_i, .oj = skepu_overlap_j, .ok = skepu_overlap_k, .ol = skepu_overlap_l, .stride1 = skepu_shared_size_j, .stride2 = skepu_shared_size_k, .stride3 = skepu_shared_size_l };\n";
	
	handleUserTypesConstantsAndPrecision_CL({&mapOverlapFunc}, sourceStream);
	sourceStream << generateOpenCLRegion(4, overlapParam);
	auto argsInfo = handleRandomAccessAndUniforms_CL(mapOverlapFunc, SSMapOverlapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	proxyCodeGenHelper_CL(argsInfo.containerProxyTypes, sourceStream);
	sourceStream << generateUserFunctionCode_CL(mapOverlapFunc) << MatrixConvol4D_CL;

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_Overlap4DKernel_" + mapOverlapFunc.uniqueName;
	std::stringstream SSKernelArgCount;
	SSKernelArgCount << (mapOverlapFunc.numKernelArgsCL());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor4D,
	{
		{"{{OPENCL_KERNEL}}",            sourceStream.str()},
		{"{{MAPOVERLAP_INPUT_TYPE}}",    overlapParam.resolvedTypeName},
		{"{{MAPOVERLAP_INPUT_TYPE_OPENCL}}", overlapParam.typeNameOpenCL()},
		{"{{MAPOVERLAP_RESULT_TYPE}}",   mapOverlapFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",              kernelName},
		{"{{INPUT_PARAM_NAME}}",         overlapParam.name},
		{"{{FUNCTION_NAME_MAPOVERLAP}}", mapOverlapFunc.uniqueName},
		{"{{KERNEL_PARAMS}}",            SSKernelParamList.str()},
		{"{{MAPOVERLAP_ARGS}}",          SSMapOverlapFuncArgs.str()},
		{"{{HOST_KERNEL_PARAMS}}",       SSHostKernelParamList.str()},
		{"{{KERNEL_CLASS}}",             "CLWrapperClass_" + kernelName},
		{"{{KERNEL_ARGS}}",              SSKernelArgs.str()},
		{"{{KERNEL_ARG_COUNT}}",         SSKernelArgCount.str()},
		{"{{CONTAINER_PROXIES}}",        argsInfo.proxyInitializer + proxy},
		{"{{CONTAINER_PROXIE_INNER}}",   argsInfo.proxyInitializerInner},
		{"{{INDEX_INITIALIZER}}",        indexInit},
		{"{{SIZE_PARAMS}}",              indexInfo.sizeParams},
		{"{{SIZE_ARGS}}",                indexInfo.sizeArgs},
		{"{{SIZES_TUPLE_PARAM}}",        indexInfo.sizesTupleParam},
		{"{{TEMPLATE_HEADER}}",          indexInfo.templateHeader},
		{"{{MULTI_TYPE}}",               mapOverlapFunc.multiReturnTypeNameGPU()},
		{"{{USE_MULTIRETURN}}",          (mapOverlapFunc.multipleReturnTypes.size() > 0) ? "1" : "0"},
		{"{{OUTPUT_ASSIGN}}",            multiOutputAssign}
	});

	return kernelName;
}