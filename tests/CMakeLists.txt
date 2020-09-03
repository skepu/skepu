if(SKEPU_CUDA)
	enable_language(CUDA)
endif()

add_subdirectory(backend)
add_subdirectory(codegen)
add_subdirectory(containers)
add_subdirectory(map)
add_subdirectory(reduce)
