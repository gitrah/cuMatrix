/*
 * PCA.h
 *
 *  Created on: Apr 21, 2016
 *      Author: reid
 */

#ifndef PCA_H_
#define PCA_H_

#include "CuMatrix.h"

template<typename T> class PCA {

	static __host__ __device__ CuMatrix<T> pca( const CuMatrix<T>& src,  int k );
};


#endif /* PCA_H_ */
