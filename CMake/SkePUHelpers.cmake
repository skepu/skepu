#	skepu_add_library(<name> [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL]
#		SKEPUSRC ssrc1 [ssrc2 ...]
#		[SRC	src1 [src2 ...]])
#
#	
function(skepu_add_library a_name)

endfunction(skepu_add_library)

#	skepu_add_executable(<name> [EXCLUDE_FROM_ALL]
#		[CUDA] [OpenCL] [OpenMP] [OpenMPI]
#		SKEPUSRC ssrc1 [ssrc2 ...]
#		[SRC osrc1 [osrc2 ...]])
#
#	A wrapper function to add_executable.
#	Creates an executable target. Source files lister after SKEPUSRC will be
#	precompiled with skepu-tool. OTHERSRC will be redirected together with the
#	precompiled source to add_executable(). The target will be linked with the
#	skepu headers.
function(skepu_add_executable a_target)

endfunction(skepu_add_executable)
