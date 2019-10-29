.DEFAULT_GOAL=all

EXEC=test
SRC=$(wildcard *.cpp)
PREC=$(SRC:.cpp=_precompiled.cpp)
OBJ=$(SRC:.cpp=.o)
CLANG_HEADERS=../llvm/clang/lib/Headers 
SKEPU_HEADERS=../include/src

all: $(EXEC)

test: $(OBJ)
	@echo "Linking $@"
	@STARPU_COMM_STATS=1 mpic++ -o $@ $^ \
		-std=c++11 -fopenmp -g -O0 \
		-DSKEPU_MPI_STARPU \
		-I$(SKEPU_HEADERS) -I ./ \
		$$(pkg-config starpu-1.3 --cflags --libs) \
		$$(pkg-config starpumpi-1.3 --cflags --libs)

.cpp.o:
	@echo "Precompiling $(<:.cpp=)"
	@../build/bin/skepu-tool --name $(<:.cpp=)_precompiled $< \
		-dir=. -openmp -silent \
		-- -std=c++11 -Wno-expansion-to-defined \
			-I$(CLANG_HEADERS) -I$(SKEPU_HEADERS)
	@echo "Compiling $(<:.cpp=)"
	@STARPU_COMM_STATS=1 mpic++ $(<:.cpp=)_precompiled.cpp  -o $@ \
		-c -g -O0 -std=c++11 \
		-DSKEPU_MPI_STARPU \
		-I$(SKEPU_HEADERS) -I ./ \
		$$(pkg-config starpu-1.3 --cflags --libs) \
		$$(pkg-config starpumpi-1.3 --cflags --libs)

clean:
	$(RM) -f $(EXEC) $(PREC) $(OBJ) *.csv

run: $(EXEC)
	@echo "Running tests..."
	@srun -N5 -l ./$(EXEC) --use-colour yes  | \
		grep -v "UCX  WARN" | \
		grep -v "===============================================================================" | \
		grep -v -E ".: $$"
	@srun -N4 -l ./$(EXEC) --use-colour yes  | \
		grep -v "UCX  WARN" | \
		grep -v "===============================================================================" | \
		grep -v -E ".: $$"
	@srun -N3 -l ./$(EXEC) --use-colour yes  | \
		grep -v "UCX  WARN" | \
		grep -v "===============================================================================" | \
		grep -v -E ".: $$"
	@srun -N2 -l ./$(EXEC) --use-colour yes  | \
		grep -v "UCX  WARN" | \
		grep -v "===============================================================================" | \
		grep -v -E ".: $$"
	@srun -N1 -l ./$(EXEC) --use-colour yes  | \
		grep -v "UCX  WARN" | \
		grep -v "===============================================================================" | \
		grep -v -E ".: $$"
