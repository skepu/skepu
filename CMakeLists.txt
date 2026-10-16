cmake_minimum_required(VERSION 3.13)
find_package(Patch REQUIRED)

project(SkePU VERSION 3.0.0.1
	DESCRIPTION
		"A C++11/CUDA/OpenCL skeleton framework with smart containers"
	LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Initialize CMAKE_CUDA_ARCHITECTURES automatically if cmake version >= 3.18
if(POLICY CMP0104)
	cmake_policy(SET CMP0104 NEW)
endif()

include(CMakePackageConfigHelpers)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/CMake)
include(skepu-options)
include(SkePUHelpers)

# The skepu-tool sources are shared with the LLVM tree by symbolic links. They are made here and not by the patches,
# since a patch cannot be relied on to create symbolic links (some versions of patch write them as text files).
# A link is replaced when something else than a link is where it should be.
function(skepu_link link target)
	if(NOT IS_SYMLINK "${link}")
		file(REMOVE_RECURSE "${link}")
		file(CREATE_LINK "${target}" "${link}" SYMBOLIC RESULT result)
		if(NOT result EQUAL 0)
			message(FATAL_ERROR "[SkePU] Could not link ${link} to ${target}: ${result}")
		endif()
	endif()
endfunction()

# Whether a patch is applied is seen on a file that it changes.
function(skepu_patched file marker result)
	set(${result} FALSE PARENT_SCOPE)
	if(EXISTS "${file}")
		file(READ "${file}" content)
		if(content MATCHES "${marker}")
			set(${result} TRUE PARENT_SCOPE)
		endif()
	endif()
endfunction()

# We need to apply our patches to the llvm work tree, if it has not already been
# done.
skepu_patched(${CMAKE_CURRENT_LIST_DIR}/llvm/clang/tools/CMakeLists.txt "skepu-tool" SkePU_LLVM_PATCHED)
if(NOT SkePU_LLVM_PATCHED)
	message(STATUS "[SkePU] Patching llvm")
	set(SkePU_LLVM_PATCH_RESULT 0)
	execute_process(
		COMMAND
			${Patch_EXECUTABLE} -Np1 -i ${CMAKE_CURRENT_LIST_DIR}/llvm.patch
		WORKING_DIRECTORY
			${CMAKE_CURRENT_LIST_DIR}/llvm
		RESULT_VARIABLE
			SkePU_LLVM_PATCH_RESULT)
	if(NOT SkePU_LLVM_PATCH_RESULT EQUAL 0)
		message(FATAL_ERROR "[SkePU] Patching llvm failed")
	endif()
	unset(SkePU_LLVM_PATCH_RESULT)
else()
	message(STATUS "[SkePU] LLVM patched and ready")
endif()
skepu_link(${CMAKE_CURRENT_LIST_DIR}/llvm/clang/tools/skepu-tool ../../../src/skepu-tool)

# The SkePU language server (a clangd with SkePU support, see README.md) is built in the same LLVM build
# tree as skepu-tool, so that the two share all the compiled LLVM and Clang libraries.
if(SKEPU_LSP)
	skepu_patched(${CMAKE_CURRENT_LIST_DIR}/llvm/clang-tools-extra/clangd/CMakeLists.txt "skepu.cpp" SkePU_CLANGD_PATCHED)
	if(NOT SkePU_CLANGD_PATCHED)
		message(STATUS "[SkePU] Patching clangd")
		set(SkePU_CLANGD_PATCH_RESULT 0)
		execute_process(
			COMMAND
				${Patch_EXECUTABLE} -Np1 -i ${CMAKE_CURRENT_LIST_DIR}/clangd.patch
			WORKING_DIRECTORY
				${CMAKE_CURRENT_LIST_DIR}/llvm
			RESULT_VARIABLE
				SkePU_CLANGD_PATCH_RESULT)
		if(NOT SkePU_CLANGD_PATCH_RESULT EQUAL 0)
			message(FATAL_ERROR "[SkePU] Patching clangd failed")
		endif()
		unset(SkePU_CLANGD_PATCH_RESULT)
	else()
		message(STATUS "[SkePU] clangd patched and ready")
	endif()
	foreach(file data_structures.cpp data_structures.h globals.h skepu.cpp visitor.cpp visitor.h)
		skepu_link(${CMAKE_CURRENT_LIST_DIR}/llvm/clang-tools-extra/clangd/${file} ../../../src/skepu-tool/${file})
	endforeach()
	foreach(file skepu_lsp.cpp skepu_lsp.h)
		skepu_link(${CMAKE_CURRENT_LIST_DIR}/llvm/clang-tools-extra/clangd/${file} ../../../src/skepu-lsp/${file})
	endforeach()
	set(SKEPU_LLVM_PROJECTS "clang|clang-tools-extra")
	set(SKEPU_LLVM_TARGETS skepu-tool clangd)
	set(SKEPU_LLVM_BYPRODUCTS llvm/bin/skepu-tool llvm/bin/clangd)
else()
	set(SKEPU_LLVM_PROJECTS clang)
	set(SKEPU_LLVM_TARGETS skepu-tool)
	set(SKEPU_LLVM_BYPRODUCTS llvm/bin/skepu-tool)
endif()

# Setting some convenience variables
set(SKEPU_TOOL_LLVM_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm")
set(SKEPU_TOOL_BINARY_DIR "${SKEPU_TOOL_LLVM_DIR}/bin")
set(SKEPU_TOOL_TARGETS_DIR
	"tools/clang/tools/skepu-tool/CMakeFiles/Export/lib/cmake/SkePU")

# We need to build SkePU-tool as an external project for these reasons:
# 1.	If llvm is added as a subproject, all it's targets would be added to make
#			all.
# 2.	If llvm is added as a subproject but EXCLUDE_FROM_ALL is set, than we
#			can't whitelist the skepu-tool target again on cmake version 3.10, which
#			our build system uses, and probably most Ubuntu 18.04 systems.
# 3.	If llvm is one subproject, and skepu-tool another, all linking information
#			from llvm is lost and skepu-tool will not be able to build at all.
include(ExternalProject)
if(SKEPU_TOOL_STATIC)
	set(SKEPU_STATIC_FLAGS "-DCMAKE_EXE_LINKER_FLAGS=\"-static\"")
endif()
ExternalProject_Add(skepu-tool-llvm
	LIST_SEPARATOR |
	SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/llvm/llvm
	BINARY_DIR ${SKEPU_TOOL_LLVM_DIR}
	CMAKE_ARGS
		-DLLVM_ENABLE_PROJECTS=${SKEPU_LLVM_PROJECTS}
		-DCMAKE_BUILD_TYPE=Release
		${SKEPU_STATIC_FLAGS}
		-DSKEPU_HASH_LIB_DIR=${CMAKE_CURRENT_LIST_DIR}/external/hash-library
	STEP_TARGETS configure
	BUILD_COMMAND $(MAKE) ${SKEPU_LLVM_TARGETS}
	BUILD_ALWAYS ON
	BUILD_BYPRODUCTS ${SKEPU_LLVM_BYPRODUCTS}
	INSTALL_COMMAND "")

if(SKEPU_LSP)
	# The language server on its own (the default target builds it together with skepu-tool). The binary is
	# ${SKEPU_TOOL_LLVM_DIR}/bin/clangd. The configure step comes first, to make the LLVM build tree have clang-tools-extra.
	add_custom_target(clangd
		COMMAND $(MAKE) -C ${SKEPU_TOOL_LLVM_DIR} clangd
		COMMENT "Building the SkePU language server (clangd)"
		USES_TERMINAL)
	add_dependencies(clangd skepu-tool-llvm-configure)
endif()

# The SkePUHelpers cmake script, which is used to generate targets for the
# examples and the tests, depends on this variable being set. It is set if SkePU
# has been installed, but we need to set it for the build as well.
set(SKEPU_EXECUTABLE ${CMAKE_BINARY_DIR}/llvm/bin/skepu-tool)

# The precompilation of a SkePU source is redone when the precompiler or any of the SkePU headers
# changes (see skepu_precompile). Not set in an installation; the precompiler is then a program
# that is not rebuilt by the project.
file(GLOB_RECURSE SKEPU_HEADER_FILES CONFIGURE_DEPENDS
	${CMAKE_CURRENT_LIST_DIR}/skepu-headers/src/*)

add_executable(skepu-tool IMPORTED GLOBAL)
add_dependencies(skepu-tool skepu-tool-llvm)
add_executable(SkePU::skepu-tool IMPORTED GLOBAL)
add_dependencies(SkePU::skepu-tool skepu-tool-llvm)
# (the program is built by the external project: the targets are there for the order of the build, and a location is what CMP0111 wants)
set_target_properties(skepu-tool SkePU::skepu-tool PROPERTIES IMPORTED_LOCATION ${SKEPU_EXECUTABLE})
# We don't have a native skepu-tool target, but the binary position is known,
# so we install it as a program.
install(PROGRAMS ${SKEPU_TOOL_BINARY_DIR}/skepu-tool
	DESTINATION bin)

# The tracing support (SKEPU_TRACING) includes ../../external/nlohmann/json.hpp from
# skepu3/impl/tracing.h, which is include/external/nlohmann in an installation (and the hashes of the source files, external/hash-library).
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/external/nlohmann ${CMAKE_CURRENT_LIST_DIR}/external/hash-library
	DESTINATION include/external)

# And the target exports are also available, so we install them as well for good
# measure. Older CMake versions write the export files to a fixed directory
# (SKEPU_TOOL_TARGETS_DIR), newer ones to a directory named by a hash, so look
# for them at install time instead of relying on one location.
install(CODE "
	file(GLOB_RECURSE _skepu_tool_exports
		\"${SKEPU_TOOL_LLVM_DIR}/tools/clang/tools/skepu-tool/CMakeFiles/Export/*skepu-toolTargets*.cmake\")
	if(NOT _skepu_tool_exports)
		message(FATAL_ERROR \"Could not find the skepu-tool export files in ${SKEPU_TOOL_LLVM_DIR}\")
	endif()
	file(INSTALL \${_skepu_tool_exports}
		DESTINATION \"\${CMAKE_INSTALL_PREFIX}/lib/cmake/SkePU\")
")

# Now that we don't have a llvm subproject, we can actually use the target name
# clang-headers. So we do. This target is used in the functions skepu_add_*
# functions.
add_library(clang-headers INTERFACE)
add_library(SkePU::clang-headers ALIAS clang-headers)
install(TARGETS clang-headers
	EXPORT clang-headersTargets
	DESTINATION lib)
install(EXPORT clang-headersTargets
	DESTINATION lib/cmake/SkePU
	NAMESPACE SkePU::)
# To make sure that -I<path_to_clang-headers> is added when linking with
# SkePU::clang-headers
target_include_directories(clang-headers INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/llvm/clang/lib/Headers>
	$<INSTALL_INTERFACE:lib/SkePU/clang-headers>)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/llvm/clang/lib/Headers/
	DESTINATION lib/SkePU/clang-headers/)
# TODO: We should probably install the clang licence in that directory.

# Enable testing first.
if(SKEPU_ENABLE_TESTING)
	enable_testing()
endif()

# The SkePU headers (skepu-headers/src): a library of headers only, that everything that uses SkePU links with.
add_library(SkePU INTERFACE)
add_library(SkePU::SkePU ALIAS SkePU)
target_compile_features(SkePU INTERFACE
	cxx_attributes
	cxx_auto_type
	cxx_constexpr
	cxx_decltype
	cxx_lambdas
	cxx_range_for
	cxx_rvalue_references
	cxx_variadic_templates
	cxx_template_template_parameters)
target_include_directories(SkePU INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/skepu-headers/src>
	$<INSTALL_INTERFACE:include>)
install(
	DIRECTORY
		${CMAKE_CURRENT_LIST_DIR}/skepu-headers/src/skepu3
		${CMAKE_CURRENT_LIST_DIR}/skepu-headers/src/skepu-lib
	DESTINATION include)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/skepu-headers/src/skepu
	DESTINATION include)
install(TARGETS SkePU
	EXPORT SkePUTargets
	DESTINATION lib)
install(EXPORT SkePUTargets
	NAMESPACE SkePU::
	DESTINATION lib/cmake/SkePU)

# Now add the tests
if(SKEPU_ENABLE_TESTING)
	add_subdirectory(tests)
endif()


# This is for the future when the utility functions has been added.
if(SKEPU_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

configure_package_config_file(
	${CMAKE_CURRENT_LIST_DIR}/CMake/skepu-toolConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/skepu-toolConfig.cmake
	INSTALL_DESTINATION lib/cmake/SkePU)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/skepu-toolConfigVersion.cmake
	COMPATIBILITY SameMajorVersion)
install(
	FILES
		${CMAKE_CURRENT_BINARY_DIR}/skepu-toolConfig.cmake
		${CMAKE_CURRENT_BINARY_DIR}/skepu-toolConfigVersion.cmake
	DESTINATION
		lib/cmake/SkePU)
install(
	FILES ${CMAKE_CURRENT_LIST_DIR}/CMake/SkePUConfig.cmake.in
	DESTINATION lib/cmake/SkePU
	RENAME SkePUConfig.cmake)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/SkePUConfigVersion.cmake
	COMPATIBILITY SameMajorVersion)
install(
	FILES ${CMAKE_CURRENT_BINARY_DIR}/SkePUConfigVersion.cmake
	DESTINATION lib/cmake/SkePU)
install(
	FILES ${CMAKE_CURRENT_LIST_DIR}/CMake/SkePUHelpers.cmake
	DESTINATION lib/cmake/SkePU)

skepu_print_config()
