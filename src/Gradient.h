/*
 * gradient.h
 *      Author: reid
 */

#pragma once

#include <cuda_runtime_api.h>
#include <string>
#include "util.h"
using std::pair;

template <typename T> class Gradient {

public:
	static __host__ __device__ pair<CuMatrix<T>&,CuMatrix<T>&> grad(const CuMatrix<T>&);
};
