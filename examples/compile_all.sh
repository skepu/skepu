#!/bin/bash

SKEPU_DEBUG=0
SKEPU_TOOL=~/clang-llvm/build/bin/skepu-tool

# On Triolith:
NVCC=/usr/local/cuda-7.5/bin/nvcc
CLANG_INCLUDE=(-I /software/apps/gcc/4.9.2/build01/include/c++/4.9.2/x86_64-unknown-linux-gnu/ -I /software/apps/gcc/4.9.2/build01/include/c++/4.9.2 -I ~/clang-llvm/llvm/tools/clang/lib/Headers -I ~/skepu_hybrid_execution/skepu-preview/include)
GCC_INCLUDE="-I ../include -I . -I /usr/local/cuda-7.5/include/ -L /usr/local/cuda-7.5/lib64"

mkdir output
mkdir tmp


applications=( map mapoverlap mapreduce reduce scan cma coulombic dotproduct lambda mandelbrot mmmult mvmult nbody ppmcc psnr taylor writeback ) # median 

for app in "${applications[@]}"
do 
# 	$SKEPU_TOOL -name ${app}_out -dir tmp -openmp -opencl $app.cpp -- -std=c++11 "${CLANG_INCLUDE[@]}"
	$SKEPU_TOOL -name ${app}_out -dir tmp -openmp -cuda $app.cpp -- -std=c++11 "${CLANG_INCLUDE[@]}"
done


for app in "${applications[@]}"
do 
# 	g++ -std=c++11 tmp/$app_out.cpp -o output/$app $GCC_INCLUDE -fopenmp -lOpenCL -DSKEPU_DEBUG=$SKEPU_DEBUG
	$NVCC -std=c++11 tmp/${app}_out.cu -o output/$app $GCC_INCLUDE -arch=sm_30 --disable-warnings -Xcompiler -fopenmp -DSKEPU_DEBUG=$SKEPU_DEBUG
done


rm -rf tmp
