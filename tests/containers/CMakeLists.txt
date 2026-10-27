
# ------------------------------------------------
#   Container proxy access, flat map, and the life cycle of the containers
# ------------------------------------------------

skepu_add_test(containers BACKENDS sequential cpu)


# ------------------------------------------------
#   Swapping
# ------------------------------------------------

skepu_add_test(swap ALSO opencl_multi)

# ------------------------------------------------
#   skepu::external flushes the containers it reads
# ------------------------------------------------

skepu_add_test(external BACKENDS sequential cpu)


# ------------------------------------------------
#   A probe of the data coherence with both CUDA and OpenCL in the program (not a test: see the source)
# ------------------------------------------------

if(SKEPU_CUDA AND SKEPU_OPENCL)
	skepu_add_executable(coherence_probe CUDA OpenCL OpenMP SKEPUSRC coherence_probe.cpp)
	target_link_libraries(coherence_probe PRIVATE catch2_main)

	# The same probe with the debug log (level 1) and the trace: the transfers of a switch of backend are shown
	skepu_add_executable(coherence_probe_logged CUDA OpenCL OpenMP SKEPUSRC coherence_probe.cpp)
	target_link_libraries(coherence_probe_logged PRIVATE catch2_main)
	target_compile_definitions(coherence_probe_logged PRIVATE SKEPU_DEBUG=1 SKEPU_TRACING SKEPU_TRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/coherence_probe_logged_trace.json)
endif()
