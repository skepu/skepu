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

# Example directory options
option(SKEPU_BUILD_EXAMPLES
	"Build the SkePU example programs."
	OFF)
option(SKEPU_EXAMPLES_SEQ
	"If building examples, build sequential examples."
	ON)
option(SKEPU_EXAMPLES_PAR
	"If building examples, Build parallel examples (CUDA, OpenCL, and OpenMP)."
	ON)
option(SKEPU_EXAMPLES_MPI
	"If building examples, build MPI examples."
	ON)
