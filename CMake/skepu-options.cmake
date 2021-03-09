# Better handling of build type in ccmake and other such tools.
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Debug")
else()
	if(NOT (CMAKE_BUILD_TYPE STREQUAL "Debug"
					OR CMAKE_BUILD_TYPE STREQUAL "Release"))
		message(FATAL_ERROR "The build type must be either Debug or Release")
	endif()
endif()
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE
	STRING "Debug or Release build." FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
	"Debug" "Release")

# In case this is a subproject, make sure that SKEPU_ENABLE_TESTING is set.
# The default value is on iff we are in Debug mode, else the default is off.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	option(SKEPU_ENABLE_TESTING
		"Enable the SkePU test bed."
		ON)
else()
	option(SKEPU_ENABLE_TESTING
		"Enable the SkePU test bed."
		OFF)
endif()

option(SKEPU_TOOL_STATIC
	"Static linking of skepu-tool."
	OFF)
mark_as_advanced(SKEPU_TOOL_STATIC)

# Check for languages and options
include(CheckLanguage)
check_language(CUDA)
set(SKEPU_CUDA OFF)
if(CMAKE_CUDA_COMPILER)
	set(SKEPU_CUDA ON)
endif()

find_package(MPI)
set(SKEPU_MPI OFF)
if(MPI_FOUND)
	find_package(PkgConfig)
	if(PkgConfig_FOUND)
		pkg_check_modules(STARPU IMPORTED_TARGET
			starpu-1.3 starpumpi-1.3)
		if(STARPU_FOUND)
			set(SKEPU_MPI ON)
		endif()
	endif()
endif()

find_package(OpenCL)
set(SKEPU_OPENCL OFF)
if(OpenCL_FOUND)
	set(SKEPU_OPENCL ON)
endif()

find_package(OpenMP)
set(SKEPU_OPENMP OFF)
if(OpenCL_FOUND)
	set(SKEPU_OPENMP ON)
endif()

# Example directory options
option(SKEPU_BUILD_EXAMPLES
	"Build the SkePU example programs."
	OFF)
option(SKEPU_EXAMPLES_SEQ
	"If building examples, build sequential examples."
	ON)
if(SKEPU_CUDA OR SKEPU_OPENCL OR SKEPU_OPENMP)
	option(SKEPU_EXAMPLES_PAR
		"If building examples, Build parallel examples (CUDA, OpenCL, and OpenMP)."
		ON)
else()
	option(SKEPU_EXAMPLES_PAR
		"If building examples, Build parallel examples (CUDA, OpenCL, and OpenMP)."
		OFF)
endif()
option(SKEPU_EXAMPLES_MPI
	"If building examples, build MPI examples."
	${SKEPU_MPI})

macro(skepu_print_config)
	message("
    +=====================+
    | SkePU configuration |
    +=====================+

    Buid type           ${CMAKE_BUILD_TYPE}
    Install prefix      ${CMAKE_INSTALL_PREFIX}
    Build examples      ${SKEPU_BUILD_EXAMPLES}
    Test suite enabled  ${SKEPU_ENABLE_TESTING}")

	if(SKEPU_BUILD_EXAMPLES OR SKEPU_ENABLE_TESTING)
		message("
    Backends for examples and test suite
    ------------------------------------
    CUDA        ${SKEPU_CUDA}
    OpenCL      ${SKEPU_OPENCL}
    OpenMP      ${SKEPU_OPENMP}
    StarPU-MPI  ${SKEPU_MPI}")
	endif()

	if(SKEPU_BUILD_EXAMPLES)
		message("
    Examples
    --------
    Sequential  ${SKEPU_EXAMPLES_SEQ}
    Parallel    ${SKEPU_EXAMPLES_PAR}
    StarPU-MPI  ${SKEPU_EXAMPLES_MPI}")
	endif()
	message("")
endmacro(skepu_print_config)
