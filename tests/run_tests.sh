#!/bin/bash

SKEPU_DEBUG=0
SKEPU_TOOL=~/clang-llvm/build/bin/skepu-tool


# On Excess:
# NVCC=/usr/local/cuda-7.5/bin/nvcc
# CLANG_INCLUDE=(-I /usr/include/ -I ~/clang-llvm/llvm/tools/clang/lib/Headers/ -I ../include/)
# GCC_INCLUDE="-ccbin /usr/bin/g++-4.9 -I /usr/include/ -I ../include/ -I . -I /usr/local/cuda-7.5/include/"

# On Triolith:
NVCC=/usr/local/cuda-7.5/bin/nvcc
CLANG_INCLUDE=(-I /software/apps/gcc/4.9.2/build01/include/c++/4.9.2/x86_64-unknown-linux-gnu/ -I /software/apps/gcc/4.9.2/build01/include/c++/4.9.2 -I ~/clang-llvm/llvm/tools/clang/lib/Headers -I ~/skepu_hybrid_execution/skepu2-preview/include)
GCC_INCLUDE="-I ../include -I . -I /usr/local/cuda-7.5/include/ -L /usr/local/cuda-7.5/lib64"

OPT_FLAG=-O0

mkdir output
mkdir tmp

skeletons=( map reduce mapreduce mapoverlap scan )

for skel in "${skeletons[@]}"
do 
	$SKEPU_TOOL -name ${skel}_test_out -dir tmp -openmp -cuda performance_${skel}_test.cpp -- -std=c++11 "${CLANG_INCLUDE[@]}"
done

# $SKEPU_TOOL -name autotune_test_out -dir tmp -openmp -cuda autotune_test.cpp -- -std=c++11 "${CLANG_INCLUDE[@]}"

for skel in "${skeletons[@]}"
do 
	# g++ -std=c++11 tmp/${skel}_test_out.cpp -o output/${skel}_test $GCC_INCLUDE $OPT_FLAG -lOpenCL -fopenmp -DSKEPU_DEBUG=$SKEPU_DEBUG
	$NVCC -std=c++11 tmp/${skel}_test_out.cu -o output/${skel}_test $GCC_INCLUDE $OPT_FLAG -arch=sm_30 --disable-warnings -Xcompiler -fopenmp -DSKEPU_DEBUG=$SKEPU_DEBUG
done
# $NVCC -std=c++11 tmp/autotune_test_out.cu -o output/autotune_test $GCC_INCLUDE -arch=sm_30 --disable-warnings -Xcompiler -fopenmp -DSKEPU_DEBUG=$SKEPU_DEBUG

rm -rf tmp

for skel in "${skeletons[@]}"
do 
	echo ""
	echo "### Test ${skel} ###"
	./output/${skel}_test
done

# echo ""
# echo "### Test Auto-tuner ###"
# ./output/autotune_test
