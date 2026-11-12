# System include directories for the precompiler. skepu-tool is a libclang tool: it does not find the
# system headers of the compiler that builds the project, so they are passed explicitly.
# On macOS they are taken from the active developer directory (Xcode or the Command Line Tools),
# not from a fixed Xcode.app location.
# Clang's headers (SkePU::clang-headers) go between the C++ library headers and the SDK headers: the SDK's stddef.h relies on Clang's being found first.
set(SKEPU_TOOL_SYSTEM_INCLUDE_FLAGS -I /usr/local/include)
set(SKEPU_TOOL_SYSTEM_INCLUDE_FLAGS_AFTER_CLANG)
if(APPLE)
	execute_process(COMMAND xcrun --show-sdk-path
		OUTPUT_VARIABLE _skepu_sdk OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
	execute_process(COMMAND xcode-select -p
		OUTPUT_VARIABLE _skepu_devdir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
	if(_skepu_sdk)
		list(APPEND SKEPU_TOOL_SYSTEM_INCLUDE_FLAGS
			-I ${_skepu_sdk}/usr/include/c++/v1)
		list(APPEND SKEPU_TOOL_SYSTEM_INCLUDE_FLAGS_AFTER_CLANG
			-I ${_skepu_sdk}/usr/include
			-I ${_skepu_devdir}/usr/include)
	else()
		message(WARNING "[SkePU] Could not determine the macOS SDK "
			"(xcrun failed); skepu-tool may not find the system headers")
	endif()
endif()

# Function to filter sources and build options from argument list.
macro(skepu_filter_args)
	set(_fnames_arg OFF)
	set(_skepu_src_arg OFF)
	set(_src_arg OFF)
	set(_precompile_arg OFF)
	foreach(arg ${ARGN})
		if("${arg}" STREQUAL "STATIC"
				OR "${arg}" STREQUAL "SHARED"
				OR "${arg}" STREQUAL "MODULE")
			set(_lib_type ${arg})
		elseif("${arg}" STREQUAL "EXCLUDE_FROM_ALL")
			set(_exclude_from_all ${arg})
		elseif("${arg}" STREQUAL "CUDA")
			set(_skepu_cuda ON)
		elseif("${arg}" STREQUAL "MPI")
			set(_skepu_mpi ON)
			set(_skepu_openmp ON)
		elseif("${arg}" STREQUAL "OpenCL")
			set(_skepu_opencl ON)
		elseif("${arg}" STREQUAL "OpenMP")
			set(_skepu_openmp ON)
		elseif("${arg}" STREQUAL "FNAMES")
			set(_fnames_arg ON)
			set(_src_arg OFF)
			set(_skepu_src_arg OFF)
			set(_precompile_arg OFF)
		elseif("${arg}" STREQUAL "SKEPUSRC")
			set(_fnames_arg OFF)
			set(_skepu_src_arg ON)
			set(_src_arg OFF)
			set(_precompile_arg OFF)
		elseif("${arg}" STREQUAL "SRC")
			set(_fnames_arg OFF)
			set(_src_arg ON)
			set(_skepu_src_arg OFF)
			set(_precompile_arg OFF)
		elseif("${arg}" STREQUAL "PRECOMPILE_OPTIONS")
			set(_fnames_arg OFF)
			set(_src_arg OFF)
			set(_skepu_src_arg OFF)
			set(_precompile_arg ON)
		else()
			if(_fnames_arg)
				list(APPEND _skepu_fnames ${arg})
			elseif(_precompile_arg)
				list(APPEND _precompile_options ${arg})
			elseif(_skepu_src_arg)
				list(APPEND _skepu_src ${arg})
			elseif(_src_arg)
				list(APPEND _src ${arg})
			else()
				message(FATAL_ERROR
					"[SkePU] Unknown argument: ${arg}.")
			endif()
		endif()
	endforeach()

	if(_skepu_mpi AND _skepu_cuda)
		message(FATAL_ERROR
			"[SkePU] The MPI backend does not have support for CUDA")
	endif()

	# Check that there are some SkePU sources in the argument list
	list(LENGTH _skepu_src _ll)
	if(_ll EQUAL 0)
		message(FATAL_ERROR
			"[SkePU] No SkePU sources found in argument list.")
	endif()
endmacro(skepu_filter_args)

# Macro to generate target CXXFLAGS and library information for a target. The
# file extension is also taken care of in this function.
macro(skepu_configure)
	set(_skepu_ext ".cpp")
	set(_target_libs SkePU::SkePU)

	if(_skepu_cuda)
		if(NOT CMAKE_CUDA_COMPILER)
			message(FATAL_ERROR "[SkePU] No CUDA compiler enabled")
		endif()
		list(APPEND _skepu_backends "-cuda")
		set(_skepu_ext ".cu")
		# nvcc does not call the constexpr host overloads of the standard library, such as fmod(float, float) of libstdc++, from device code without this
		list(APPEND _target_cxxflags "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
	endif()

	if(_skepu_mpi)
		if(NOT MPI_CXX_FOUND)
			find_package(MPI REQUIRED)
		endif()
		list(APPEND _skepu_backends "-mpi")
		list(APPEND _target_libs MPI::MPI_CXX)
	endif()

	if(_skepu_opencl)
		if(NOT OpenCL_FOUND)
			find_package(OpenCL REQUIRED)
		endif()
		list(APPEND _skepu_backends "-opencl")
		list(APPEND _target_libs OpenCL::OpenCL)
	endif()

	if(_skepu_openmp)
		if(NOT OpenMP_FOUND)
			find_package(OpenMP REQUIRED)
		endif()
		list(APPEND _skepu_backends "-openmp")

		# With CUDA enabled the OpenMP flag is passed to the host compiler
		# (-Xcompiler).
		if(_skepu_cuda)
			list(APPEND _target_cxxflags "SHELL:-Xcompiler -fopenmp")
		endif()
		list(APPEND _target_libs OpenMP::OpenMP_CXX)
	endif()
endmacro(skepu_configure)

# target_link_libraries and target_include_directories are propagated to the
# skepu-tool command, so that the include directories that the precompilation
# needs are in the command.
macro(skepu_generate_include_generators)
	# skepu-tool needs these.
	set(_clang_prop
		$<TARGET_PROPERTY:SkePU::clang-headers,INTERFACE_INCLUDE_DIRECTORIES>)

	# Make sure that target_link_libraries and target_include_directories
	# is propagated onto the skepu-tool target.
	set(_id_prop $<TARGET_PROPERTY:${name},INCLUDE_DIRECTORIES>)
	set(_iid_prop $<TARGET_PROPERTY:${name},INTERFACE_INCLUDE_DIRECTORIES>)
	set(_isid_prop
		$<TARGET_PROPERTY:${name},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>)

	# The generators will create one -I<include> for every item in the target
	# property. Note that the $<BOOL:...> is there to make sure that no empty -I
	# is generated by the generators, should the prop be empty.
	set(_clang_generator
		$<$<BOOL:${_clang_prop}>:-I$<JOIN:${_clang_prop}, -I>>)
	set(_include_generators
		$<$<BOOL:${_id_prop}>:-I$<JOIN:${_id_prop}, -I>>)
	set(_include_generators ${_include_generators}
		$<$<BOOL:${_iid_prop}>:-I$<JOIN:${_iid_prop}, -I>>)
	set(_include_generators ${_include_generators}
		$<$<BOOL:${_isid_prop}>:-I$<JOIN:${_isid_prop}, -I>>)
endmacro(skepu_generate_include_generators)

# Precompiles the SkePU sources of skepu_add_library/skepu_add_executable with skepu-tool.
# Expects the results of skepu_filter_args and skepu_configure; sets _precompiled_src,
# _skepu_targets and _output_dir.
macro(skepu_precompile)
	# The precompiled sources go to their own directory.
	set(_output_dir ${CMAKE_CURRENT_BINARY_DIR}/skepu_precompiled)
	if(NOT EXISTS ${_output_dir})
		file(MAKE_DIRECTORY ${_output_dir})
	endif()

	# A custom target for each SkePU source file precompiles it.
	foreach(_file IN LISTS _skepu_src)
		get_filename_component(_file_name ${_file} NAME_WE)
		get_filename_component(_file_abs ${_file} ABSOLUTE
			BASE_DIR ${CMAKE_CURRENT_LIST_DIR})
		set(_target_name "${name}_${_file_name}_precompiled")
		set(_target_byprod "${name}_${_file_name}_precompiled${_skepu_ext}")
		skepu_generate_include_generators()
		add_custom_command(OUTPUT ${_output_dir}/${_target_byprod}
			COMMAND
				${SKEPU_EXECUTABLE}
					${_skepu_backends}
					-silent
					-name ${_target_name}
					-dir=${_output_dir}
					-fnames="${_skepu_fnames}"
					${_file_abs}
					--
					${SKEPU_TOOL_SYSTEM_INCLUDE_FLAGS}
					"${_clang_generator}"
					${SKEPU_TOOL_SYSTEM_INCLUDE_FLAGS_AFTER_CLANG}
					${_precompile_options}
					"${_include_generators}"
			DEPENDS ${_file_abs} ${SKEPU_EXECUTABLE} ${SKEPU_HEADER_FILES}
			BYPRODUCTS ${_output_dir}/${_target_byprod}
			COMMAND_EXPAND_LISTS
			)
		set_source_files_properties(${_output_dir}/${_target_byprod}
			PROPERTIES
				GENERATED TRUE)
		add_custom_target(${_target_name}
			DEPENDS ${_output_dir}/${_target_byprod})
		add_dependencies(${_target_name}
			SkePU::skepu-tool SkePU::clang-headers SkePU::SkePU)
		list(APPEND _precompiled_src ${_output_dir}/${_target_byprod})
		list(APPEND _skepu_targets ${_target_name})
	endforeach()
endmacro(skepu_precompile)

#	skepu_add_library(<name> [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL]
#		[[CUDA] [OpenCL] [OpenMP] | [MPI]]
#		SKEPUSRC ssrc1 [ssrc2 ...]
#		[SRC	src1 [src2 ...]]
#		[PRECOMPILE_OPTIONS opt1 [opt2 ...]])
#
#	A wrapper function to add_library
#	Creates a library target. Source files listed after SKEPUSRC will be
#	precompiled with skepu-tool. Other sources, listed after SRC, will be
#	redirected to add_library. SkePU headers will be included automatically.
#	The library type defaults to STATIC, as for add_library; MODULE is what a
#	plugin or a Python extension module needs. Options listed after
#	PRECOMPILE_OPTIONS are passed on to the skepu-tool compiler invocation only.
# Note that the user is resposible for enabling CUDA,OpenCL,OpenMP, and OpenMPI
# within their cmake scripts.
function(skepu_add_library name)
	skepu_filter_args(${ARGN})
	skepu_configure()
	skepu_precompile()

	add_library(${name} ${_lib_type} ${_exclude_from_all} ${_precompiled_src} ${_src})
	target_include_directories(${name} PRIVATE ${_output_dir})
	target_link_libraries(${name} PRIVATE ${_target_libs})
	target_compile_options(${name} PRIVATE ${_target_cxxflags})
	add_dependencies(${name} ${_skepu_targets})
endfunction(skepu_add_library)

#	skepu_add_executable(<name> [EXCLUDE_FROM_ALL]
#		[[CUDA] [OpenCL] [OpenMP] | [MPI]]
#		SKEPUSRC ssrc1 [ssrc2 ...]
#		[SRC src1 [src2 ...]]
#		[PRECOMPILE_OPTIONS opt1 [opt2 ...]])
#
#	A wrapper function to add_executable.
#	Creates an executable target. Source files listed after SKEPUSRC will be
#	precompiled with skepu-tool. SRC will be redirected together with the
#	precompiled source to add_executable(). The function will automatically
#	include the SkePU headers. Source files may be given as absolute paths or
#	relative to the current CMake list directory. Options listed after
#	PRECOMPILE_OPTIONS (for example -D definitions) are passed on to the
#	skepu-tool compiler invocation only; use target_compile_options for the
#	actual build.
# Note that the user is resposible for enabling CUDA,OpenCL,OpenMP, and OpenMPI
# within their cmake scripts.
function(skepu_add_executable name)
	# The SkePU headers are needed for the build.
	skepu_filter_args(${ARGN})
	skepu_configure()
	skepu_precompile()

	# The executable target has the precompiled source and the C++ sources (if
	# any). CMake takes the language, CUDA or C++, from the file extension.
	add_executable(${name} ${_exclude_from_all} ${_precompiled_src} ${_src})
	target_include_directories(${name} PRIVATE ${_output_dir})
	target_link_libraries(${name} PRIVATE ${_target_libs})
	target_compile_options(${name} PRIVATE ${_target_cxxflags})
	add_dependencies(${name} ${_skepu_targets})
endfunction(skepu_add_executable)
