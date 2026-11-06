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

option(SKEPU_LSP
	"Also build the SkePU language server (clangd with SkePU support) in the LLVM build tree. Patches clang-tools-extra with clangd.patch."
	OFF)

# The backends: SKEPU_CUDA, SKEPU_OPENCL, SKEPU_OPENMP and SKEPU_MPI are what the tests and the examples read. A backend is on if it is found and not turned
# off with SKEPU_ENABLE_<BACKEND>; SKEPU_<BACKEND>_WHY says why one is off. (A project that uses SkePU's helpers chooses its own backends: see skepu_add_executable.)
option(SKEPU_ENABLE_CUDA "Use the CUDA backend, if there is a CUDA compiler." ON)
option(SKEPU_ENABLE_OPENCL "Use the OpenCL backend, if OpenCL is found." ON)
option(SKEPU_ENABLE_OPENMP "Use the OpenMP backend, if OpenMP is found." ON)
option(SKEPU_ENABLE_MPI "Use the MPI backend, if MPI is found. It is built with OpenMP, and needs it." ON)

include(CheckLanguage)
set(SKEPU_CUDA OFF)
if(NOT SKEPU_ENABLE_CUDA)
	set(SKEPU_CUDA_WHY "turned off (SKEPU_ENABLE_CUDA is OFF)")
else()
	check_language(CUDA)
	if(CMAKE_CUDA_COMPILER)
		set(SKEPU_CUDA ON)
	else()
		set(SKEPU_CUDA_WHY "no CUDA compiler was found")
	endif()
endif()

set(SKEPU_OPENCL OFF)
if(NOT SKEPU_ENABLE_OPENCL)
	set(SKEPU_OPENCL_WHY "turned off (SKEPU_ENABLE_OPENCL is OFF)")
else()
	find_package(OpenCL)
	if(OpenCL_FOUND)
		set(SKEPU_OPENCL ON)
	else()
		set(SKEPU_OPENCL_WHY "OpenCL was not found")
	endif()
endif()

set(SKEPU_OPENMP OFF)
if(NOT SKEPU_ENABLE_OPENMP)
	set(SKEPU_OPENMP_WHY "turned off (SKEPU_ENABLE_OPENMP is OFF)")
else()
	find_package(OpenMP)
	if(OpenMP_FOUND)
		set(SKEPU_OPENMP ON)
	else()
		set(SKEPU_OPENMP_WHY "OpenMP was not found")
	endif()
endif()

set(SKEPU_MPI OFF)
if(NOT SKEPU_ENABLE_MPI)
	set(SKEPU_MPI_WHY "turned off (SKEPU_ENABLE_MPI is OFF)")
elseif(NOT SKEPU_OPENMP)
	set(SKEPU_MPI_WHY "the MPI backend needs OpenMP: ${SKEPU_OPENMP_WHY}")
else()
	find_package(MPI)
	if(MPI_FOUND)
		set(SKEPU_MPI ON)
	else()
		set(SKEPU_MPI_WHY "MPI was not found")
	endif()
endif()

# Hybrid OpenMP+GPU backends
set(SKEPU_HYBRID_CUDA OFF)
set(SKEPU_HYBRID_OPENCL OFF)

if(SKEPU_OPENMP AND SKEPU_CUDA)
	set(SKEPU_HYBRID_CUDA ON)
endif()

if(SKEPU_OPENMP AND SKEPU_OPENCL)
	set(SKEPU_HYBRID_OPENCL ON)
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

    Build type           ${CMAKE_BUILD_TYPE}
    Install prefix      ${CMAKE_INSTALL_PREFIX}
    Build examples      ${SKEPU_BUILD_EXAMPLES}
    Test suite enabled  ${SKEPU_ENABLE_TESTING}
    Language server     ${SKEPU_LSP}")

	if(SKEPU_BUILD_EXAMPLES OR SKEPU_ENABLE_TESTING)
		message("
    Backends for examples and test suite
    ------------------------------------
    OpenMP         ${SKEPU_OPENMP}
    CUDA           ${SKEPU_CUDA}
    OpenCL         ${SKEPU_OPENCL}
    Hybrid-CUDA    ${SKEPU_HYBRID_CUDA}
    Hybrid-OpenCL  ${SKEPU_HYBRID_OPENCL}
    MPI            ${SKEPU_MPI}")
	endif()

	if(SKEPU_ENABLE_TESTING OR SKEPU_BUILD_EXAMPLES)
		get_property(_skepu_capabilities_text GLOBAL PROPERTY SKEPU_CAPABILITIES_TEXT)
		if(_skepu_capabilities_text)
			message("${_skepu_capabilities_text}")
		endif()
	endif()

	if(SKEPU_BUILD_EXAMPLES)
		message("
    Examples
    --------
    Sequential  ${SKEPU_EXAMPLES_SEQ}
    Parallel    ${SKEPU_EXAMPLES_PAR}
    MPI         ${SKEPU_EXAMPLES_MPI}")
		get_property(_skepu_examples_skipped_text GLOBAL PROPERTY SKEPU_EXAMPLES_SKIPPED_TEXT)
		if(_skepu_examples_skipped_text)
			message("${_skepu_examples_skipped_text}")
		endif()
	endif()
	message("")
endmacro(skepu_print_config)
