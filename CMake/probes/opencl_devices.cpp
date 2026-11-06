// Configure-time probe: the GPUs and accelerators of the first two platforms (as environment.inl), and how many have double precision (as device_cl.h).
// Prints "gpu=<n> fp64=<n>".
#define CL_SILENCE_DEPRECATION
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <cstdio>
#include <cstring>

int main()
{
	cl_platform_id platforms[2];
	cl_uint platformCount = 0;
	if (clGetPlatformIDs(2, platforms, &platformCount) != CL_SUCCESS)
		platformCount = 0;
	if (platformCount > 2)
		platformCount = 2;

	int gpus = 0, fp64 = 0;
	for (cl_uint p = 0; p < platformCount; ++p)
	{
		cl_device_id devices[16];
		cl_uint count = 0;
		if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 16, devices, &count) != CL_SUCCESS)
			continue;
		for (cl_uint d = 0; d < count; ++d)
		{
			++gpus;
			cl_device_fp_config config = 0;
			bool has = clGetDeviceInfo(devices[d], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof config, &config, nullptr) == CL_SUCCESS && config != 0;
			if (!has)
			{
				char extensions[4096] = "";
				clGetDeviceInfo(devices[d], CL_DEVICE_EXTENSIONS, sizeof extensions - 1, extensions, nullptr);
				has = std::strstr(extensions, "cl_khr_fp64") || std::strstr(extensions, "cl_amd_fp64");
			}
			if (has)
				++fp64;
		}
	}
	std::printf("gpu=%d fp64=%d\n", gpus, fp64);
	return 0;
}
