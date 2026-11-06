# ------------------------------------------------
#   Backend selection
# ------------------------------------------------

skepu_gate(_gate_ok selection openmp)
if(_gate_ok)
	skepu_add_executable(selection_test OpenMP SKEPUSRC selection.cpp)
	target_link_libraries(selection_test PRIVATE catch2_main)
	add_test(NAME selection COMMAND selection_test)
endif()

# ------------------------------------------------
#   BackendSpec options
# ------------------------------------------------

skepu_gate(_gate_ok backendspec openmp)
if(_gate_ok)
	skepu_add_executable(backendspec_test OpenMP SKEPUSRC backendspec.cpp)
	target_link_libraries(backendspec_test PRIVATE catch2_main)
	add_test(NAME backendspec COMMAND backendspec_test)
endif()


