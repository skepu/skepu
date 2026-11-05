# ------------------------------------------------
#   The names of the source files in a trace (skepu3/impl/tracing_paths.hpp)
# ------------------------------------------------

skepu_add_test(trace_paths BACKENDS sequential)

# ------------------------------------------------
#   The list of the compiler's directories outlives the objects that read it when they are destroyed
# ------------------------------------------------

add_executable(trace_paths_order trace_paths_order.cpp)
target_compile_features(trace_paths_order PRIVATE cxx_std_17)
add_test(trace_paths_order trace_paths_order)
