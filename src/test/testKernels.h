/*
 * testKernels.h
 *
 *  Created on: Dec 18, 2013
 *      Author: reid
 */

#ifndef TESTKERNELS_H_
#define TESTKERNELS_H_
#include "../debug.h"
#include "../CuFunctor.h"
template<typename T> __global__ void testShuffleKernel();
template<typename T> void launchTestShuffleKernel();

template<typename T> __global__ void testCdpKernel(T* res, CuMatrix<T> mat);
template<typename T> void launchTestCdpKernel(T* res, CuMatrix<T> mat);

__global__ void testSetLastErrorKernel(CuMatrixException);

template<typename T> void launchRedux(T* res, CuMatrix<T> mat);
template<typename T> __global__ void launchReduxD(T* res, CuMatrix<T> mat);

template<typename T, typename BinaryOp> __global__ void kFunctor(BinaryOp op);
template<typename T, typename BinaryOp> void launchKFunctor(BinaryOp op);

struct kfoo {
	void * pointierre;
	int somint;
};
template<typename T> struct kbar {
	T * pointierre;
	T somt;

	__host__ __device__ T operator()(T input) const {
		return somt * input;
	}
};
__global__ void kFoo(kfoo foo);
void launchKfoo(kfoo foo);
template<typename T> __global__ void kBar(kbar<T> bar);
template<typename T>void launchKbar(kbar<T> bar);

template<typename T, int StateDim> __global__ void constFillKrnle( UnaryOpIndexF<T,StateDim> fill );
template<typename T>void constFillKrnleL( );

#endif /* TESTKERNELS_H_ */
