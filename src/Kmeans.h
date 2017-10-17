/*
 * Kmeans.h
 *
 *  Created on: Apr 24, 2014
 *      Author: reid
 */
#pragma once

#include "CuMatrix.h"

template <typename T> class Kmeans {
public:
	static __host__ CUDART_DEVICE void findClosest(IndexArray& indices, const CuMatrix<T>& means, const CuMatrix<T>& x);
	static __host__ CUDART_DEVICE void calcMeans(IndexArray& indices, CuMatrix<T>& means, const CuMatrix<T>& x);
	static __host__ CUDART_DEVICE T distortion(IndexArray& indices, CuMatrix<T>& means, const CuMatrix<T>& x);
	static __host__ CUDART_DEVICE void calcMeansColThread(IndexArray& indices, CuMatrix<T>& means, const CuMatrix<T>& x);
};

