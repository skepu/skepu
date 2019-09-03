TARGET_NAME=test

SRC_FILES := $(wildcard ./*.cpp)
PRE_FILES := $(patsubst ./%.cpp,./%_precompiled.cu,$(SRC_FILES))
OBJ_FILES := $(patsubst ./%.cpp,./%.o,$(SRC_FILES))

all: $(TARGET_NAME)

$(TARGET_NAME): $(OBJ_FILES)
	@echo "Linking..."
	@STARPU_COMM_STATS=1 mpic++ -o $@ $^ -g -O0			\
-DSKEPU_MPI_STARPU -std=c++11 -DSKEPU_MPI_STARPU -I ../include -I ./	\
-fopenmp \
$$(pkg-config starpu-1.3 --cflags)						\
$$(pkg-config starpu-1.3 --libs) $$(pkg-config --cflags								\
starpumpi-1.3) $$(pkg-config --libs starpumpi-1.3)										\
#


./%.o: ./%.cpp
	@echo "Compiling $<"
	@../build/bin/skepu-tool -silent -name $(basename $<)_precompiled -dir . $< -openmp  -- -std=c++11 -Wno-expansion-to-defined -I ../clang/lib/Headers -I ../include
	@STARPU_COMM_STATS=1 mpic++ $(basename $<)_precompiled.cpp  -c -o $@ -g -O0			\
-DSKEPU_MPI_STARPU -std=c++11 -DSKEPU_MPI_STARPU -I ../include -I ./	\
$$(pkg-config starpu-1.3 --cflags)						\
$$(pkg-config starpu-1.3 --libs) $$(pkg-config --cflags								\
starpumpi-1.3) $$(pkg-config --libs starpumpi-1.3)

clean:
	mv main.o main.catch
	rm $(TARGET_NAME) ./*_precompiled* ./*.o ./*.csv 2>/dev/null || true
	mv main.catch main.o

run: $(TARGET_NAME)
	@echo "Running tests..."
	@srun -N5 -l ./$(TARGET_NAME) --use-colour yes  | \
		grep -v "UCX  WARN" | \
		grep -v "===============================================================================" | \
		grep -v -E ".: $$"
	@srun -N4 -l ./$(TARGET_NAME) --use-colour yes  | \
		grep -v "UCX  WARN" | \
		grep -v "===============================================================================" | \
		grep -v -E ".: $$"
	@srun -N3 -l ./$(TARGET_NAME) --use-colour yes  | \
		grep -v "UCX  WARN" | \
		grep -v "===============================================================================" | \
		grep -v -E ".: $$"
	@srun -N2 -l ./$(TARGET_NAME) --use-colour yes  | \
		grep -v "UCX  WARN" | \
		grep -v "===============================================================================" | \
		grep -v -E ".: $$"
	@srun -N1 -l ./$(TARGET_NAME) --use-colour yes  | \
		grep -v "UCX  WARN" | \
		grep -v "===============================================================================" | \
		grep -v -E ".: $$"
