cmake_minimum_required(VERSION 3.13)

if(POLICY CMP0104)
	cmake_policy(SET CMP0104 OLD)
endif()

if(CMAKE_PROJECT_NAME)
	set(SKEPU_HEADERS_SUBPROJECT ${CMAKE_PROJECT_NAME})
endif()

project(SkePU-headers
	VERSION 3.0.0
	LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

include(CMakePackageConfigHelpers)

# All options available for the SkePU headers shall be listed in
# CMake/skepu-headers-options.cmake
include(skepu-headers-options)

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
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
	$<INSTALL_INTERFACE:include>)

install(
	DIRECTORY
		${CMAKE_CURRENT_LIST_DIR}/src/skepu3
		${CMAKE_CURRENT_LIST_DIR}/src/skepu-lib
	DESTINATION include)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/src/skepu
	DESTINATION include)

if(SKEPU_HEADERS_ENABLE_TESTING)
	enable_testing()
	add_subdirectory(tests)
else()
	file(REMOVE ${CMAKE_BINARY_DIR}/CTestTestfile.cmake)
endif()

install(TARGETS SkePU
	EXPORT skepu-headersTargets
	DESTINATION lib)
install(EXPORT skepu-headersTargets
	NAMESPACE SkePU::
	DESTINATION lib/cmake/SkePU)

configure_package_config_file(
	${CMAKE_CURRENT_LIST_DIR}/CMake/skepu-headersConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/skepu-headersConfig.cmake
	INSTALL_DESTINATION lib/cmake/SkePU)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/skepu-headersConfigVersion.cmake
	COMPATIBILITY SameMajorVersion)
install(
	FILES
		${CMAKE_CURRENT_BINARY_DIR}/skepu-headersConfig.cmake
		${CMAKE_CURRENT_BINARY_DIR}/skepu-headersConfigVersion.cmake
	DESTINATION lib/cmake/SkePU)

if(NOT SKEPU_HEADERS_SUBPROJECT)
	skepu_headers_print_config()
endif()

# vi:ts=2 sw=2 noet
