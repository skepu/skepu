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
