#!/bin/bash -x

SKEPU_TOOL=../build/bin/skepu-tool
# Path to cuda installation
CUDA_DIR=/usr/local/cuda-9.2/

# path to clang source files
CLANG_SOURCE_DIR=../clang

OPT_FLAG=-O0
SKEPU_DEBUG=0

mkdir -p output
mkdir -p tmp

skeletons=( map )
for skel in "${skeletons[@]}"
do 
    ./run_test.sh ${skel} $SKEPU_TOOL $CUDA_DIR $CLANG_SOURCE_DIR $OPT_FLAG $SKEPU_DEBUG
done

# skeletons=( map reduce mapreduce mapoverlap scan )

# for skel in "${skeletons[@]}"
# do 
# 	$SKEPU_TOOL -name ${skel}_test_out -dir tmp -openmp -cuda performance_${skel}_test.cpp -- -std=c++11 "${CLANG_INCLUDE[@]}"
# done

# # $SKEPU_TOOL -name autotune_test_out -dir tmp -openmp -cuda autotune_test.cpp -- -std=c++11 "${CLANG_INCLUDE[@]}"

# for skel in "${skeletons[@]}"
# do 
# 	# g++ -std=c++11 tmp/${skel}_test_out.cpp -o output/${skel}_test $GCC_INCLUDE $OPT_FLAG -lOpenCL -fopenmp -DSKEPU_DEBUG=$SKEPU_DEBUG
# 	$NVCC -std=c++11 tmp/${skel}_test_out.cu -o output/${skel}_test $GCC_INCLUDE $OPT_FLAG -arch=sm_30 --disable-warnings -Xcompiler -fopenmp -DSKEPU_DEBUG=$SKEPU_DEBUG
# done
# # $NVCC -std=c++11 tmp/autotune_test_out.cu -o output/autotune_test $GCC_INCLUDE -arch=sm_30 --disable-warnings -Xcompiler -fopenmp -DSKEPU_DEBUG=$SKEPU_DEBUG

rm -rf tmp

# for skel in "${skeletons[@]}"
# do 
# 	echo ""
# 	echo "### Test ${skel} ###"
# 	./output/${skel}_test
# done

# # echo ""
# # echo "### Test Auto-tuner ###"
# # ./output/autotune_test
