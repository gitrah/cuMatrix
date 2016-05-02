/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "CuMatrix.h"
#include "caps.h"
#include "debug.h"
#include "MemMgr.h"
#include "Kernels.h"
#include "DMatrix.h"
#include <typeinfo>

/*
 * expects a nonsqr block and a grid of (sqr) blocks to cover x (and pdens)
 * x is input sampleCount*featureCount
 * pdens is sampleCount*featureProbability
 */
template<typename T> __global__ void multivariateGaussianFeaturesKernel(
		DMatrix<T> d_pdens, const DMatrix<T> d_x, const DMatrix<T> d_sigmaSquared, const DMatrix<T> d_mu) {
	int col = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	uint rowStart = blockIdx.y * blockDim.x + threadIdx.y; //
	uint rowsRemaining = MIN(d_x.m - rowStart, blockDim.x);
	uint xRowOff,pdensRowOff;
	T x_n, sigmaSquared_n, mu_n;
	if(col < d_x.n && rowStart < d_x.m) {
		sigmaSquared_n = d_sigmaSquared.elements[col];
		mu_n = d_mu.elements[col];
		xRowOff = rowStart * d_x.p;
		pdensRowOff = rowStart * d_pdens.p;
		for (int row = 0; row < rowsRemaining; row += blockDim.y) {
			x_n = d_x.elements[xRowOff + row * d_x.p + col];
			d_pdens.elements[pdensRowOff + row * d_pdens.p + col] =
					(ONE_OVER_SQR_2PI / sqrt((double)sigmaSquared_n)) / exp(  ( (x_n - mu_n )*( x_n - mu_n ) / (2.0 * sigmaSquared_n)));
		}
	}
}


template<typename T>  void CuMatrix<T>::multivariateGaussianFeatures( DMatrix<T>& d_pden, const DMatrix<T>& d_x, const DMatrix<T>& d_sqrdSigmas, const DMatrix<T>& d_mu) {
	dim3 block(32,8);
	dim3 grid(DIV_UP( d_x.n,block.x), DIV_UP(d_x.m,block.x));
	outln("multivariateGaussianFeatures on d_x " << util<T>::pdm(d_x) );
	outln("multivariateGaussianFeatures with grid " << b_util::pd3(grid) << " block " << b_util::pd3(block));
	//outln("rows " << d_x.m);
	multivariateGaussianFeaturesKernel<<<grid,block,0>>>(d_pden, d_x, d_sqrdSigmas, d_mu);
}


/*
 * converts probability density matrix to probability vector
 * assumes block as wide as pdens.n
 */
template<typename T> __global__ void mvGaussianVectorFromFeaturesKernel(
		DMatrix<T> d_pvec, const DMatrix<T> d_pdens) {
	int col = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	int row = blockIdx.y * blockDim.y + threadIdx.y; //
	uint tileIdx = threadIdx.x + threadIdx.y * blockDim.x;
	// load smem
	T* tile = SharedMemory<T>();
	T currP;
	if(col < d_pdens.n && row < d_pdens.m) {
		// fill tile with block of pdens
		tile[tileIdx] = d_pdens.elements[col + row * d_pdens.p];
		__syncthreads();
		if(threadIdx.x == 0) {
			currP = tile[tileIdx];
			for(int icol = 1; icol < d_pdens.n; icol++) {
				currP *= tile[tileIdx +icol];
			}
			d_pvec.elements[row* d_pvec.p] = currP;
		}
	}
}

template<typename T>  void CuMatrix<T>::mvGaussianVectorFromFeatures( DMatrix<T>& d_pvec, const DMatrix<T>& d_pdens) {
	//uint blockX =MAX(32, d_pdens.n);
	//dim3 block(blockX,ExecCaps::currCaps()->maxTsPerBlock<T>()/blockX);
	//uint smem = block.x * block.y * sizeof(T);
	uint blockX = d_pdens.n;
	uint blockY = MIN(d_pdens.m, ExecCaps::currCaps()->maxTsPerBlock<T>()/blockX);
	outln("blockY by max smem " << blockY);
	if(blockY < 1) {
		dthrow(notEnoughSmem());
	}
	blockY = MIN(blockY, ExecCaps::currCaps()->thrdPerBlock/blockX);
	outln("blockY by max thread/block" << blockY);
	uint smem = blockX * blockY * sizeof(T);
	dim3 block(blockX,blockY);
	dim3 grid(1, DIV_UP(d_pdens.m,block.y));
	outln("mvGaussianVectorFromFeatures with grid " << b_util::pd3(grid) << " block " << b_util::pd3(block) << ", smem " << smem);
	//outln("rows " << d_x.m);
	mvGaussianVectorFromFeaturesKernel<<<grid,block,smem>>>(d_pvec, d_pdens);
}

/*
 *
 * d_x.n must < max block dim otherwise two-step it
 * rows is how many rows of d_x.n cols will fit in smem
 *
 */

template<typename T> __global__ void multivariateGaussianVectorKernel(
		DMatrix<T> d_pvec, const DMatrix<T> d_x, const DMatrix<T> d_sigmaSquared, const DMatrix<T> d_mu) {
	int col = threadIdx.x; // index into column
	int row = blockIdx.y * blockDim.y + threadIdx.y; //
	uint xRowOff;
	uint tileIdx = threadIdx.x + threadIdx.y * blockDim.x;
	T x_n, sigmaSquared_n, mu_n;
	T* tile = SharedMemory<T>();
	if(col < d_x.n && row < d_x.m) {
		sigmaSquared_n = d_sigmaSquared.elements[col];
		mu_n = d_mu.elements[col];
		xRowOff = row * d_x.p;
		// fill as many rows as will fit in smem with features probs
		x_n = d_x.elements[xRowOff + col];
		tile[tileIdx] =
				(ONE_OVER_SQR_2PI / sqrt((double)sigmaSquared_n)) / exp(  ( (x_n - mu_n )*( x_n - mu_n ) / (2.0 * sigmaSquared_n)));
		__syncthreads();
		// now find the sample prob (== product of feature probs)
		if(threadIdx.x == 0) {
			x_n = tile[tileIdx] ;
			for(int icol = 1; icol < blockDim.x; icol++) {
				x_n *= tile[tileIdx + icol];
			}
			d_pvec.elements[row * d_pvec.p] = x_n;
		}
	}
}


template<typename T>  void CuMatrix<T>::multivariateGaussianVector(  DMatrix<T>& d_pvec, const DMatrix<T>& d_x, const DMatrix<T>& d_sqrdSigmas, const DMatrix<T>& d_mu) {
	uint blockX = d_x.n;
	ExecCaps* caps = ExecCaps::currCaps();
    struct cudaFuncAttributes funcAttrib;
    T startRegisterHeadroomFactor = REGISTER_HEADROOM_FACTOR;
    cudaError_t err = cudaSuccess;


    cherr(cudaFuncGetAttributes(&funcAttrib, multivariateGaussianVectorKernel<T>));
 /*   printf("unmangled %s numRegs=%d\n", typeid( multivariateGaussianVectorKernel<T>).name(),funcAttrib.numRegs);
    printf("%s numRegs=%d\n", b_util::unmangl(typeid( multivariateGaussianVectorKernel<T>).name()).c_str(),funcAttrib.numRegs);
*/
    do {
		uint blockY = MIN( caps->thrdPerBlock/blockX, MIN(d_x.m, caps->maxTsPerBlock<T>()/blockX));
		blockY = MIN(blockY, startRegisterHeadroomFactor * caps->regsPerBlock / (blockX * funcAttrib.numRegs));
		outln("blockX " << blockX);
		outln("blockY " << blockY);
		outln("threadsPerBlock " << (blockX*blockY));
		uint totalRegs = blockX*blockY*funcAttrib.numRegs;
		outln("totalRegs " << totalRegs);
		T regHeadroom = 100.0 * ( 1.0*( caps->regsPerBlock -totalRegs)/caps->regsPerBlock);
		outln("reg headroom " <<  regHeadroom);
		outln("max regsPerBlock " << caps->regsPerBlock);
		outln("max regsPerSM " << caps->regsPerSM);
		outln("ExecCaps::currCaps()->maxTsPerBlock<T>() " << ExecCaps::currCaps()-> maxTsPerBlock<T>());

		if(blockY < 1) {
			dthrow(notEnoughSmem());
		}
		uint smem = blockX * blockY * sizeof(T);
		outln("smem " << smem);

		dim3 block(blockX,blockY);
		dim3 grid(1, DIV_UP( d_x.m,block.y));
		uint totalSmem = smem * grid.x * grid.y;
		outln("totalSmem " << totalSmem);
		outln("totalThreads " << grid.x* grid.y * blockX * blockY);

		outln("multivariateGaussianVector on d_x " << util<T>::pdm(d_x));
		outln("multivariateGaussianVector with grid " << b_util::pd3(grid) << " of blocks " << b_util::pd3(block) << " with smem " << smem);
		//outln("rows " << d_x.m);
		multivariateGaussianVectorKernel<<<grid,block,smem>>>(d_pvec, d_x, d_sqrdSigmas,d_mu);
		err = cudaGetLastError();
		startRegisterHeadroomFactor *= .99;
		outln("err " << err << ", startFactor " << startRegisterHeadroomFactor);
    } while(err == cudaErrorLaunchOutOfResources);
}

#include "CuMatrixInster.cu"

