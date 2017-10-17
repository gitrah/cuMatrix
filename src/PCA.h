/*
 * PCA.h
 *
 *  Created on: Apr 21, 2016
 *      Author: reid
 */

#pragma once

#include "CuMatrix.h"

template<typename T> class PCA {
	static __host__ __device__ CuMatrix<T> pca( const CuMatrix<T>& src,  int k );
};
