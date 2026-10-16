
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
