#pragma once

#include <string>
#include <unordered_set>

// Library functions that a user function may call: kernels have one of the same name (OpenCL C builtin, CUDA device function).
inline const std::unordered_set<std::string> &kernelBuiltinFunctions()
{
	static const std::unordered_set<std::string> names
	{
		"exp", "exp2", "exp2f",
		"sqrt", "copysign", "copysignf",
		"abs", "fabs",
		"max", "fmax",
		"min", "fmin",
		"pow",
		"log", "log2", "log10",
		"sin", "sinh", "asin", "asinh",
		"cos", "cosh", "acos", "acosh",
		"tan", "tanh", "atan", "atanh",
		"round", "ceil", "floor",
		"fmod", "trunc",
		"erf",
		"nan", "isnan", "isinf",
		"printf",
	};
	return names;
}

inline std::string openCLBuiltinName(const std::string &name, bool floatingArgument)
{
	return (name == "abs" && floatingArgument) ? "fabs" : name;
}

// OpenCL C reserved names beyond the builtins above
inline const std::unordered_set<std::string> &openCLReservedNames()
{
	static const std::unordered_set<std::string> names
	{
	// types
	"half", "double", "float", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "size_t", "ptrdiff_t", "intptr_t",
	"uintptr_t", "void", "bool", "event_t", "sampler_t", "image2d_t", "image3d_t", "image1d_t",
	// common, geometric and relational
	"clamp", "degrees", "radians", "mix", "step", "smoothstep", "sign", "cross", "dot", "distance", "length", "normalize",
	"fast_distance", "fast_length", "fast_normalize", "any", "all", "bitselect", "select", "isequal", "isnotequal", "isgreater",
	"isgreaterequal", "isless", "islessequal", "islessgreater", "isfinite", "isnormal", "isordered", "isunordered", "signbit",
	// math
	"cbrt", "erfc", "expm1", "fdim", "fma", "fract", "frexp", "hypot", "ilogb", "ldexp", "lgamma", "lgamma_r", "log1p", "logb", "mad",
	"maxmag", "minmag", "modf", "nextafter", "pown", "powr", "remainder", "remquo", "rint", "rootn", "rsqrt", "sincos", "sinpi",
	"cospi", "tanpi", "asinpi", "acospi", "atanpi", "atan2", "atan2pi", "tgamma", "native_sin", "native_cos", "native_tan",
	"native_exp", "native_log", "native_sqrt", "native_rsqrt", "native_divide", "native_recip", "native_powr",
	"half_sin", "half_cos", "half_tan", "half_exp", "half_log", "half_sqrt", "half_rsqrt", "half_divide", "half_recip", "half_powr",
	// integer
	"abs_diff", "add_sat", "sub_sat", "hadd", "rhadd", "clz", "mad24", "mul24", "mad_hi", "mad_sat", "mul_hi", "rotate", "upsample",
	"popcount",
	// work-items, synchronization, memory
	"get_work_dim", "get_global_size", "get_global_id", "get_local_size", "get_local_id", "get_num_groups", "get_group_id",
	"get_global_offset", "barrier", "mem_fence", "read_mem_fence", "write_mem_fence", "async_work_group_copy", "wait_group_events",
	"prefetch", "vload2", "vload3", "vload4", "vload8", "vload16", "vstore2", "vstore3", "vstore4", "vstore8", "vstore16",
	"atomic_add", "atomic_sub", "atomic_inc", "atomic_dec", "atomic_xchg", "atomic_cmpxchg", "atomic_min", "atomic_max",
	"atomic_and", "atomic_or", "atomic_xor",
	// images
	"read_imagef", "read_imagei", "read_imageui", "write_imagef", "write_imagei", "write_imageui", "get_image_width",
	"get_image_height", "get_image_depth",
	// conversions and reinterpretations
	"as_char", "as_uchar", "as_short", "as_ushort", "as_int", "as_uint", "as_long", "as_ulong", "as_float",
	"convert_char", "convert_uchar", "convert_short", "convert_ushort", "convert_int", "convert_uint", "convert_long",
	"convert_ulong", "convert_float",
};
	return names;
}

// such user functions are prefixed in kernels
inline bool isKernelReservedName(const std::string &name)
{
	return kernelBuiltinFunctions().count(name) || openCLReservedNames().count(name);
}
