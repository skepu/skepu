# ------------------------------------------------
#   Container proxy access
# ------------------------------------------------

skepu_add_executable(lifecycle_test SKEPUSRC lifecycle.cpp)
target_link_libraries(lifecycle_test PRIVATE catch2_main)
add_test(lifecycle lifecycle_test)

# ------------------------------------------------
#   Container proxy access
# ------------------------------------------------

skepu_add_executable(containers_test SKEPUSRC containers.cpp)
target_link_libraries(containers_test PRIVATE catch2_main)
add_test(containers containers_test)

# ------------------------------------------------
#   Flat map
# ------------------------------------------------

skepu_add_executable(flatmap_test SKEPUSRC flatmap.cpp)
target_link_libraries(flatmap_test PRIVATE catch2_main)
add_test(flatmap flatmap_test)


