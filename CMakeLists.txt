cmake_minimum_required(VERSION 3.13)
find_package(Patch REQUIRED)

project(SkePU VERSION 3.0.0.1
	DESCRIPTION
		"A C++11/CUDA/OpenCL skeleton framework with smart containers"
	LANGUAGES CXX)

# Initialize CMAKE_CUDA_ARCHITECTURES automatically if cmake version >= 3.18
if(POLICY CMP0104)
	cmake_policy(SET CMP0104 NEW)
endif()

include(CMakePackageConfigHelpers)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/CMake)
include(skepu-options)
include(SkePUHelpers)

# We need to apply our patches to the llvm work tree, if it has not alredy been
# done.
if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/llvm/clang/tools/skepu-tool)
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
	SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/llvm/llvm
	BINARY_DIR ${SKEPU_TOOL_LLVM_DIR}
	CMAKE_ARGS
		-DLLVM_ENABLE_PROJECTS=clang
		-DCMAKE_BUILD_TYPE=Release
		${SKEPU_STATIC_FLAGS}
	BUILD_COMMAND $(MAKE) skepu-tool
	BUILD_ALWAYS ON
	BUILD_BYPRODUCTS llvm/bin/skepu-tool
	INSTALL_COMMAND "")

# The SkePUHelpers cmake script, which is used to generate targets for the
# examples and the tests, depends on this variable being set. It is set if SkePU
# has been installed, but we need to set it for the build as well.
set(SKEPU_EXECUTABLE ${CMAKE_BINARY_DIR}/llvm/bin/skepu-tool)

add_executable(skepu-tool IMPORTED GLOBAL)
add_dependencies(skepu-tool skepu-tool-llvm)
add_executable(SkePU::skepu-tool IMPORTED GLOBAL)
add_dependencies(SkePU::skepu-tool skepu-tool-llvm)
# We don't have a native skepu-tool target, but the binary position is known,
# so we install it as a program.
install(PROGRAMS ${SKEPU_TOOL_BINARY_DIR}/skepu-tool
	DESTINATION bin)

# And the target exports are also available, so we install them as well for good
# measure.
install(
	DIRECTORY
		${SKEPU_TOOL_LLVM_DIR}/${SKEPU_TOOL_TARGETS_DIR}/
	DESTINATION lib/cmake/SkePU/
	FILES_MATCHING PATTERN "*.cmake")

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

# Enable testing first in case skepu headers tests are enabled.
if(SKEPU_ENABLE_TESTING)
	enable_testing()
endif()

# We need the SkePU headers.
add_subdirectory(skepu-headers)

# Now add the skepu-tool tests
if(SKEPU_ENABLE_TESTING)
	find_package(Catch2)

	# If skepu headers tests were not enabled, we add the catch2_main target here.
	if(NOT TARGET catch2_main)
		add_subdirectory(${CMAKE_SOURCE_DIR}/skepu-headers/tests/catch2_main)
	endif()

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
install(
	FILES ${CMAKE_CURRENT_LIST_DIR}/CMake/SkePUHelpers.cmake
	DESTINATION lib/cmake/SkePU)

skepu_print_config()
skepu_headers_print_config()
