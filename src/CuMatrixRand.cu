/*
 * CuMatrixRand.cu
 *
 *      Author: reid
 */
#include "CuMatrix.h"
#include "caps.h"
#include "util.h"
#include "Kernels.h"


template <typename T> void CuMatrix<T>::initRand(int height, int width, int pitch) {
	checkCudaError( cudaMalloc (( void **) & devStates, height * width *  sizeof ( curandState ))) ;
	uint blockW = MIN(MIN( b_util::pad(width,32), b_util::pad(ExecCaps::currCaps()->maxBlock.x, 32)), b_util::maxBlockThreads((void*)setup_kernel));
	uint blockH = MIN( ExecCaps::currCaps()->thrdPerBlock/blockW, MIN(height, ExecCaps::currCaps()->maxBlock.y));
	dim3 block(blockW, blockH);
	outln("block was " << b_util::pd3(block));
	b_util::validLaunch(block,(void*)setup_kernel );
	outln("block now " << b_util::pd3(block));
	dim3 grid(DIV_UP(width, blockW), DIV_UP(height,blockH));
	bool valid = b_util::validLaunchQ((void*)setup_kernel,grid, block);
	b_util::pFuncPtrAtts((void *)setup_kernel);
	outln("setup_kernel exec " << b_util::pxd(grid,block) << " valid " << valid);
	assert(valid);
	setup_kernel<<<grid, block >>>( devStates, width, pitch );
}

template <typename T> void CuMatrix<T>::freeRand() {
	cudaFree(devStates);
	devStates = null;
}

template <typename T> cudaError_t CuMatrix<T>::randn(DMatrix<T>& d_ret, T epsilon) {
	if(devStates) {
		freeRand();
	}
	initRand(d_ret.m, d_ret.n, d_ret.p);
	uint blockW = MIN(b_util::pad(d_ret.n,32), b_util::pad(ExecCaps::currCaps()->maxBlock.x,32));
	uint blockH = MIN( ExecCaps::currCaps()->thrdPerBlock/blockW, MIN(d_ret.m, ExecCaps::currCaps()->maxBlock.y));
	dim3 block(blockW, blockH);
	dim3 grid(DIV_UP(d_ret.n, blockW), DIV_UP(d_ret.m, blockH));
	bool valid = b_util::validLaunchQ((void*)generate_uniform_kernel<T>,grid, block);
	b_util::pFuncPtrAtts((void *)generate_uniform_kernel<T>);
	outln("generate_uniform_kernel exec " << b_util::pxd(grid,block) << " valid " << valid);

	generate_uniform_kernel<<<grid,block>>>(devStates, d_ret.elements, epsilon, d_ret.m, d_ret.n,d_ret.p);
	freeRand();
	return cudaDeviceSynchronize();
}

template <typename T> cudaError_t CuMatrix<T>::randmod(DMatrix<T>& d_ret, int mod, T epsilon) {
	if(devStates) {
		freeRand();
	}
	initRand(d_ret.m, d_ret.n, d_ret.p);
	uint blockW = MIN(b_util::pad(d_ret.n,32), b_util::pad(ExecCaps::currCaps()->maxBlock.x,32));
	uint blockH = MIN( ExecCaps::currCaps()->thrdPerBlock/blockW, MIN(d_ret.m, ExecCaps::currCaps()->maxBlock.y));
	dim3 block(blockW, blockH);
	dim3 grid(DIV_UP(d_ret.n, blockW), DIV_UP(d_ret.m, blockH));
	bool valid = b_util::validLaunchQ((void*)generate_uniform_kernel<T>,grid, block);
	b_util::pFuncPtrAtts((void *)generate_uniform_kernel<T>);
	outln("generate_uniform_kernel exec " << b_util::pxd(grid,block) << " valid " << valid);

	generate_uniform_kernel_mod<<<grid,block>>>(devStates, d_ret.elements, epsilon, d_ret.m, d_ret.n,d_ret.p, mod);
	freeRand();
	return cudaDeviceSynchronize();
}

//template cudaError_t CuMatrix<float>::randn(DMatrix<float>&, float);
//template cudaError_t CuMatrix<double>::randn(DMatrix<double>&, double);

#include "CuMatrixInster.cu"
