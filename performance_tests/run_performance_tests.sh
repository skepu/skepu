#!/bin/bash

SKEPU_DEBUG=0
SKEPU_TOOL=~/clang-llvm/build/bin/skepu-tool

# On Excess:
# NVCC=/usr/local/cuda-8.0/bin/nvcc
# CLANG_INCLUDE=(-I /usr/lib/gcc/x86_64-linux-gnu/5.4.1/include -I ~/skepu_hybrid_execution/skepu-preview/include)
# GCC_INCLUDE="-I ../include/ -I . -I /usr/local/cuda-8.0/include/"


# On Triolith:
NVCC=/usr/local/cuda-7.5/bin/nvcc
CLANG_INCLUDE=(-I /software/apps/gcc/4.9.2/build01/include/c++/4.9.2/x86_64-unknown-linux-gnu/ -I /software/apps/gcc/4.9.2/build01/include/c++/4.9.2 -I ~/clang-llvm/llvm/tools/clang/lib/Headers -I ~/skepu_hybrid_execution/skepu-preview/include)
GCC_INCLUDE="-I ../include -I . -I /usr/local/cuda-7.5/include/ -L /usr/local/cuda-7.5/lib64"

OPT_FLAG=-O0

mkdir output
mkdir tmp

applications=( cma dotproduct gaussian mandelbrot psnr ppmcc taylor ) # coulombic nbody
# not interesting: mvmult 



for app in "${applications[@]}"
do 
	$SKEPU_TOOL -name ${app}_out -dir tmp -openmp -cuda $app.cpp -- -std=c++11 "${CLANG_INCLUDE[@]}"
done



for app in "${applications[@]}"
do 
# 	g++ -std=c++11 tmp/${app}_out.cpp -o output/${app} $GCC_INCLUDE $OPT_FLAG -fopenmp -lOpenCL -DSKEPU_DEBUG=$SKEPU_DEBUG
	$NVCC -std=c++11 tmp/${app}_out.cu -o output/${app} $GCC_INCLUDE $OPT_FLAG -arch=sm_30 --disable-warnings -Xcompiler -fopenmp -DSKEPU_DEBUG=$SKEPU_DEBUG
done

rm -rf tmp
rm speedup_openmp.dat
rm speedup_cuda.dat
rm speedup_oracle.dat
rm speedup_hybrid.dat

printf "dummy 0\n" > speedup_openmp.dat
printf "dummy 0\n" > speedup_cuda.dat
printf "dummy 0\n" > speedup_oracle.dat
printf "dummy 0\n" > speedup_hybrid.dat

echo "### Running all tests"


for app in "${applications[@]}"
do 
	./output/${app}
done
