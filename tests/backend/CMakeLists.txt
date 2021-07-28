# ------------------------------------------------
#   Backend selection
# ------------------------------------------------

if(SKEPU_OPENMP)
	skepu_add_executable(selection_test OpenMP SKEPUSRC selection.cpp)
	target_link_libraries(selection_test PRIVATE catch2_main)
	add_test(selection selection_test)
endif()

# ------------------------------------------------
#   BackendSpec options
# ------------------------------------------------

if(SKEPU_OPENMP)
	skepu_add_executable(backendspec_test OpenMP SKEPUSRC backendspec.cpp)
	target_link_libraries(backendspec_test PRIVATE catch2_main)
	add_test(backendspec backendspec_test)
endif()


