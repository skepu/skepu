# Function to filter sources and build options from argument list.
macro(skepu_filter_args)
	set(_fnames_arg OFF)
	set(_skepu_src_arg OFF)
	set(_src_arg OFF)
	foreach(arg ${ARGN})
		if(${arg} STREQUAL "STATIC"
				OR ${arg} STREQUAL "SHARED"
				OR ${arg} STREQUAL "MODULE")
			set(_lib_type ${arg})
		elseif(${arg} STREQUAL "EXCLUDE_FROM_ALL")
			set(_exclude_from_all ${arg})
		elseif(${arg} STREQUAL "CUDA")
			set(_skepu_cuda ON)
		elseif(${arg} STREQUAL "MPI")
			set(_skepu_mpi ON)
			set(_skepu_openmp ON)
		elseif(${arg} STREQUAL "OpenCL")
			set(_skepu_opencl ON)
		elseif(${arg} STREQUAL "OpenMP")
			set(_skepu_openmp ON)
		elseif(${arg} STREQUAL "FNAMES")
			set(_fnames_arg ON)
			set(_src_arg OFF)
			set(_skepu_src_arg OFF)
		elseif(${arg} STREQUAL "SKEPUSRC")
			set(_fnames_arg OFF)
			set(_skepu_src_arg ON)
			set(_src_arg OFF)
		elseif(${arg} STREQUAL "SRC")
			set(_fnames_arg OFF)
			set(_src_arg ON)
			set(_skepu_src_arg OFF)
		else()
			if(_fnames_arg)
				list(APPEND _skepu_fnames ${arg})
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

	if(_skepu_mpi AND (_skepu_cuda OR _skepu_opencl))
		message(FATAL_ERROR
			"[SkePU] OpenMPI cannot be enabled with other backends except OpenMP.")
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
			message(FATAL_ERROR "[SKEPU] No CUDA compiler enabled")
		endif()
		list(APPEND _skepu_backends "-cuda")
		set(_skepu_ext ".cu")
	endif()

	if(_skepu_mpi)
		if(NOT STARPU_FOUND)
			find_package(MPI REQUIRED)
			find_package(PkgConfig REQUIRED)
			pkg_check_modules(STARPU REQUIRED IMPORTED_TARGET
				starpu-1.3 starpumpi-1.3)
		endif()
		list(APPEND _target_cxxflags -DSKEPU_MPI_STARPU)
		list(APPEND _target_libs OpenMP::OpenMP_CXX MPI::MPI_CXX PkgConfig::STARPU)
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

		# We need to be a bit careful with the OpenMP flags and libraries if CUDA is
		# enabled...
		if(_skepu_cuda)
			list(APPEND _target_cxxflags
				-Xcompiler $<JOIN:${OpenMP_CXX_FLAGS}, -Xcompiler >)
			list(APPEND _target_libs ${OpenMP_CXX_LIBRARIES})
		else()
			list(APPEND _target_libs OpenMP::OpenMP_CXX)
		endif()
	endif()
endmacro(skepu_configure)

# We need to make sure that target_link_libraries and target_include_directories
# is propagated to the skepu-tool command, so that any extra include directories
# needed during precompilation is in the command.
macro(skepu_generate_include_generators)
	# We need these to run skepu-tool.
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
	set(_include_generators
		$<$<BOOL:${_clang_prop}>:-I$<JOIN:${_clang_prop}, -I>>)
	set(_include_generators ${_include_generators}
		$<$<BOOL:${_id_prop}>:-I$<JOIN:${_id_prop}, -I>>)
	set(_include_generators ${_include_generators}
		$<$<BOOL:${_iid_prop}>:-I$<JOIN:${_iid_prop}, -I>>)
	set(_include_generators ${_include_generators}
		$<$<BOOL:${_isid_prop}>:-I$<JOIN:${_isid_prop}, -I>>)
endmacro(skepu_generate_include_generators)

#	skepu_add_library(<name> [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL]
#		[[CUDA] [OpenCL] [OpenMP] | [MPI]]
#		SKEPUSRC ssrc1 [ssrc2 ...]
#		[SRC	src1 [src2 ...]])
#
#	A wrapper function to add_library
#	Creates a library target. Source files listed after SKEPUSRC will be
#	precompiled with skepu-tool. Other sources, listed after SRC, will be
#	redirected to add_library. SkePU headers will be included automatically.
# Note that the user is resposible for enabling CUDA,OpenCL,OpenMP, and OpenMPI
# within their cmake scripts.
function(skepu_add_library name)

endfunction(skepu_add_library)

#	skepu_add_executable(<name> [EXCLUDE_FROM_ALL]
#		[[CUDA] [OpenCL] [OpenMP] | [MPI]]
#		SKEPUSRC ssrc1 [ssrc2 ...]
#		[SRC src1 [src2 ...]])
#
#	A wrapper function to add_executable.
#	Creates an executable target. Source files listed after SKEPUSRC will be
#	precompiled with skepu-tool. SRC will be redirected together with the
#	precompiled source to add_executable(). The function will automatically
#	include the SkePU headers.
# Note that the user is resposible for enabling CUDA,OpenCL,OpenMP, and OpenMPI
# within their cmake scripts.
function(skepu_add_executable name)
	# We need the skepu headers when building.
	skepu_filter_args(${ARGN})
	skepu_configure()

	# Just keeps the output directory a bit tidier.
	set(_output_dir ${CMAKE_CURRENT_BINARY_DIR}/skepu_precompiled)
	if(NOT EXISTS ${_output_dir})
		file(MAKE_DIRECTORY ${_output_dir})
	endif()

	# Add a custom target (per SkePU source file) so that cmake can procompile
	# SkePU sources for us.
	foreach(_file IN LISTS _skepu_src)
		get_filename_component(_file_name ${_file} NAME_WE)
		set(_target_name "${name}_${_file_name}_precompiled")
		set(_target_byprod "${name}_${_file_name}_precompiled${_skepu_ext}")
		set(_skepu_fnames "${_skepu_fnames}")
		skepu_generate_include_generators()
		add_custom_command(OUTPUT ${_output_dir}/${_target_byprod}
			COMMAND
				${SKEPU_EXECUTABLE}
					${_skepu_backends}
					-silent
					-name ${_target_name}
					-dir=${_output_dir}
					-fnames="${_skepu_fnames}"
					${CMAKE_CURRENT_LIST_DIR}/${_file}
					--
					-std=c++11
					"${_include_generators}"
			DEPENDS ${CMAKE_CURRENT_LIST_DIR}/${_file}
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

	# Finally add an executable target with both the procompiled source and C++
	# source (if any). Cmake will use the file extension to figure out if we are
	# dealing with CUDA or C++, so we do not have to care about that.
	add_executable(${name} ${_exclude_from_all} ${_precompiled_src} ${_src})
	target_include_directories(${name} PRIVATE ${_output_dir})
	target_link_libraries(${name} PRIVATE ${_target_libs})
	target_compile_options(${name} PRIVATE ${_target_cxxflags})
	add_dependencies(${name} ${_skepu_targets})
endfunction(skepu_add_executable)
