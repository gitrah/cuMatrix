/*
 * CuMatrixRand.cu
 *
 *      Author: reid
 */
#include "CuMatrix.h"
#include "caps.h"
#include "Kernels.h"


template <typename T> void CuMatrix<T>::initRand(int height, int width) {
	checkCudaError( cudaMalloc (( void **) & devStates, height * width *  sizeof ( curandState ))) ;
	uint blockW = MIN(width, ExecCaps::currCaps()->maxBlock.x);
	uint blockH = MIN( ExecCaps::currCaps()->thrdPerBlock/blockW, MIN(height, ExecCaps::currCaps()->maxBlock.y));
	dim3 block(blockW, blockH);
	dim3 grid(DIV_UP(width, blockW), DIV_UP(height,blockH));
	setup_kernel<<<grid, block >>>( devStates, width );
}

template <typename T> void CuMatrix<T>::freeRand() {
	cudaFree(devStates);
	devStates = null;
}

template <typename T> cudaError_t CuMatrix<T>::randn(DMatrix<T>& d_ret, T epsilon) {
	if(devStates) {
		freeRand();
	}
	initRand(d_ret.m, d_ret.n);
	uint blockW = MIN(d_ret.n, ExecCaps::currCaps()->maxBlock.x);
	uint blockH = MIN( ExecCaps::currCaps()->thrdPerBlock/blockW, MIN(d_ret.m, ExecCaps::currCaps()->maxBlock.y));
	dim3 block(blockW, blockH);
	dim3 grid(DIV_UP(d_ret.n, blockW), DIV_UP(d_ret.m, blockH));
	outln("rand exec " << b_util::pxd(grid,block));
	generate_uniform_kernel<<<grid,block>>>(devStates, d_ret.elements, epsilon, d_ret.m, d_ret.n);
	freeRand();
	return cudaDeviceSynchronize();
}

//template cudaError_t CuMatrix<float>::randn(DMatrix<float>&, float);
//template cudaError_t CuMatrix<double>::randn(DMatrix<double>&, double);

#include "CuMatrixInster.cu"
