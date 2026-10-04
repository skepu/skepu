#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED

static skepu::PrecompilerMarker startOf_ML_HPP;

#define SKEPU_DNN_NO_PRETRAINED_PARAMS "NO PRETRAINED PARAMS"
#define SKEPU_DNN_USE_MASKED_POOL 1
#define DNN_CONST /*const&*/

#include <skepu>
#include <skepu-lib/io.hpp>

#include <skepu-lib/dnn/csv_io.hpp>

// These library files will be injected by the precompiler
#ifndef SKEPU_PRECOMPILED
#include "dnn/dnn_support.hpp"
#include "dnn/layer.hpp"
#include "dnn/activation.hpp"
#include "dnn/dense.hpp"
#include "dnn/dropout.hpp"
#include "dnn/flatten.hpp"
#include "dnn/conv1d.hpp"
#include "dnn/conv2d.hpp"
#include "dnn/pool2d.hpp"
#include "dnn/sequential_model.hpp"
#endif // SKEPU_PRECOMPILED


static skepu::PrecompilerMarker endOf_ML_HPP;
#endif // SKEPU_PRECOMPILED
