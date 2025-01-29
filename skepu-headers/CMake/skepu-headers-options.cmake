# Better build type handling.
if(NOT SKEPU_HEADERS_SUBPROJECT)
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
endif()

if(SKEPU_HEADERS_SUBPROJECT STREQUAL "SkePU")
	option(SKEPU_HEADERS_ENABLE_TESTING
		"Enable SkePU headers tests."
		 ${SKEPU_ENABLE_TESTING})
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
	option(SKEPU_HEADERS_ENABLE_TESTING
		"Enable SkePU headers tests."
		 ON)
else()
	option(SKEPU_HEADERS_ENABLE_TESTING
		"Enable SkePU headers tests."
		 OFF)
endif()

if(SKEPU_HEADERS_SUBPROJECT STREQUAL "SkePU")
	set(SKEPU_HEADERS_CUDA ${SKEPU_CUDA} PARENT_SCOPE)
	set(SKEPU_HEADERS_OPENCL ${SKEPU_OPENCL} PARENT_SCOPE)
	set(SKEPU_HEADERS_OPENMP ${SKEPU_OPENMP} PARENT_SCOPE)
	set(SKEPU_HEADERS_MPI ${SKEPU_MPI} PARENT_SCOPE)
else()
	set(SKEPU_HEADERS_CUDA OFF)
	set(SKEPU_HEADERS_OPENCL OFF)
	set(SKEPU_HEADERS_OPENMP OFF)
	set(SKEPU_HEADERS_MPI OFF)

	include(CheckLanguage)
	check_language(CUDA)
	if(CMAKE_CUDA_COMPILER)
		set(SKEPU_HEADERS_CUDA ON)
	endif()

	find_package(MPI)
	if(MPI_FOUND)
		find_package(PkgConfig)
		if(PkgConfig_FOUND)
			pkg_check_modules(STARPU IMPORTED_TARGET
				starpumpi-1.3)
			if(STARPU_FOUND)
				set(SKEPU_HEADERS_MPI ON)
				if(${CMAKE_VERSION} VERSION_LESS 3.18)
					#message(STATUS "Fixing MPI + CUDA issue...")
					get_target_property(_flags MPI::MPI_CXX INTERFACE_LINK_LIBRARIES)

					# In versions previous to 3.18, this causes an issue when CMake tries
					# to link cuda device code, so we disable -pthread.
					string(REPLACE "-pthread" "" _flags "${_flags}")

					# As a precaution, we do not search for " -pthread", but then we have
					# to deal with a possible " ;" situation which CMake does not like.
					string(REPLACE " ;" ";" _flags "${_flags}")
					set_property(TARGET MPI::MPI_CXX PROPERTY
						INTERFACE_LINK_LIBRARIES ${_flags})
					unset(_flags)
				endif()
			endif()
		endif()
	endif()

	find_package(OpenCL)
	if(OpenCL_FOUND)
		set(SKEPU_HEADERS_OPENCL ON)
	endif()

	find_package(OpenMP)
	if(OpenMP_FOUND)
		set(SKEPU_HEADERS_OPENMP ON)
	endif()
endif()

if(SKEPU_HEADERS_CUDA)
	enable_language(CUDA)
endif()

macro(skepu_headers_print_config)
	message("
   +=============================+
   | SkePU Headers configuration |
   +=============================+
   Testing  ${SKEPU_HEADERS_ENABLE_TESTING}")

	if(SKEPU_HEADERS_ENABLE_TESTING)
		message("
   Testing backends
   ----------------
   CUDA    ${SKEPU_HEADERS_CUDA}
   OpenCL  ${SKEPU_HEADERS_OPENCL}
   OpenMP  ${SKEPU_HEADERS_OPENMP}
   MPI     ${SKEPU_HEADERS_MPI}")
	endif()

	message("")
endmacro(skepu_headers_print_config)
