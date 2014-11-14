/*
 * CuMatrixProfile.cu
 *
 *  Created on: Jun 14, 2014
 *      Author: reid
 */


#include "CuMatrix.h"
#include "debug.h"

template <typename T> __host__ __device__ float CuMatrix<T>::flow(int iterations, int iterationMemoryFactor, float exeTime) {
	float ret = iterationMemoryFactor * 1000. * this->size / Giga / (exeTime/ iterations);
	flprintf("ret %f iterations %d, iterationMemoryFactor %d, exeTime %f\n",ret, iterations, iterationMemoryFactor, exeTime);
	// iterationMemoryFactor eg how many reads and or writes of mat.size
	return ret;
}


#include "CuMatrixInster.cu"
