include $(dir $(lastword $(MAKEFILE_LIST)))../../examples/Makefile

EM_FLAGS = -std=c++17 -I$(SKEPU_INCLUDE_PATH) \
           -s MODULARIZE=1 -s EXPORT_NAME='createSkePUModule' \
           -s INVOKE_RUN=0 -s EXIT_RUNTIME=1 \
           -s EXPORTED_RUNTIME_METHODS='["callMain","FS"]' \
           -s ALLOW_MEMORY_GROWTH=1 -O2

DEMOS = map_squaring  # add more here: mapoverlap reduce …

.PHONY: all-wasm clean
all-wasm: $(addprefix wasm/, $(addsuffix .js, $(DEMOS)))

wasm/%.js: %.cpp | wasm/
	$(DBGR) $(SKEPU_TOOL_PATH) -name $*_precompiled $<  -dir ./ $(SKEPU_FLAGS)
	em++ $*_precompiled.cpp -o $@ $(EM_FLAGS) -DSKEPU_TRACING -DSKEPU_TRACE_FILE="trace.json"

wasm/:
	mkdir -p $@

clean:
	rm -rf wasm/
