#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include <skepu>
#ifndef SKEPU_PRECOMPILED

static skepu::PrecompilerMarker startOf_DNN_HPP;


#define SKEPU_ENABLE_EXCEPTIONS
#define SKEPU_DNN_NO_PRETRAINED_PARAMS "NO PRETRAINED PARAMS"
#define SKEPU_DNN_USE_MASKED_POOL 1
#define DNN_CONST /*const&*/
// #define DNN_DEBUG 1

#include <skepu>
#include <skepu-lib/io.hpp>

#include "csv_io.hpp"

#include "dnn/dnn_support.hpp"
#include "dnn/sequential_model.hpp"





static skepu::PrecompilerMarker endOfD_NN_HPP;