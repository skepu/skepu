#!/bin/bash 

skel=$1
SKEPU_TOOL=$2
CUDA_DIR=$3
CLANG_SOURCE_DIR=$4

OPT_FLAG=$5
SKEPU_DEBUG=$6

NVCC=$CUDA_DIR/bin/nvcc
SKEPU_INCLUDE="-I . -I ../include/"

# On Excess:
CLANG_INCLUDE="-I $CLANG_SOURCE_DIR/lib/Headers/"
CUDA_INCLUDE="-I $CUDA_DIR/include/"

# # On Triolith:
# CLANG_INCLUDE=(-I /software/apps/gcc/4.9.2/build01/include/c++/4.9.2/x86_64-unknown-linux-gnu/ -I /software/apps/gcc/4.9.2/build01/include/c++/4.9.2 -I $CLANG_SOURCE_DIR/lib/Headers -I ~/skepu_hybrid_execution/skepu-preview/include)
# CUDA_INCLUDE="-I $CUDA_DIR/include/ -L $CUDA_DIR/lib64"


echo ---- : Running test for ${skel}


$SKEPU_TOOL -name ${skel}_test_out -dir tmp -openmp -cuda performance_${skel}_test.cpp -- -std=c++11 $SKEPU_INCLUDE $CLANG_INCLUDE

if [ $? -eq 0 ]; then
    echo ---- : ${skel} : SKEPU COMPILATION OK
else
    echo ---- : ${skel} : SKEPU COMPILATION FAILED. Skpipping further processing of this test file.
    exit 1
fi

# g++ -std=c++11 tmp/${skel}_test_out.cpp -o output/${skel}_test $CUDA_INCLUDE $OPT_FLAG -lOpenCL -fopenmp -DSKEPU_DEBUG=$SKEPU_DEBUG
$NVCC -std=c++11 tmp/${skel}_test_out.cu -o output/${skel}_test $SKEPU_INCLUDE $CUDA_INCLUDE $OPT_FLAG -arch=sm_30 --disable-warnings -Xcompiler -fopenmp -DSKEPU_DEBUG=$SKEPU_DEBUG

if [ $? -eq 0 ]; then
    echo ---- : ${skel} : NVCC COMPILATION OK
else
    echo ---- : ${skel} : NVCC COMPILATION FAILED. Skpipping further processing of this test file.
    exit 1
fi

echo ""
echo "### Test ${skel} ###"
./output/${skel}_test