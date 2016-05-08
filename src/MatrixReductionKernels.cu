/*
 * MatrixReductionKernels.cu
 *
 *  Created on: Oct 19, 2013
 *      Author: reid
 */

#include <typeinfo>

#include "Kernels.h"
#include "util.h"
#include "caps.h"
#include "CuMatrix.h"
#include "debug.h"

/*
 * initial reduction is between matrices
 */
//template<typename T, template <typename> class Function>
#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class BinaryOp1,
template <typename> class  BinaryOp2>
__global__ void combineReduceOpKernel(const T* g_idata1, const T* g_idata2,
		T* g_odata, long n, BinaryOp1<T> mop, BinaryOp2<T> bop, T start)
#else
template<typename T, uint blockSize, bool nIsPow2, int MopDim, int BopDim>
__global__ void combineReduceOpKernel(const T* g_idata1, const T* g_idata2,
		T* g_odata, long n, BinaryOpF<T, MopDim> mop, BinaryOpF<T,BopDim> bop, T start)
#endif
{
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	uint tid = threadIdx.x;
	long i = blockIdx.x * blockSize * 2 + threadIdx.x;
	uint gridSize = blockSize * 2 * gridDim.x;
	T myReduction = start;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n) {
		myReduction = bop(myReduction, mop(g_idata1[i], g_idata2[i]));

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction = bop(myReduction,
					mop(blockSize[i + g_idata1], blockSize[i + g_idata2])); // a[x +b] parses to *(a + x + b)
		if(checkDebug(debugRedux) && i + blockSize == n -1 ) {
			flprintf("max i reading g_idata1(%p) and g_idata2(%p)\n", g_idata1 + i + blockSize, g_idata2+i+blockSize);
		}
		i += gridSize;
	}
	// each thread puts its local sum into shared memory
	sdata[tid] = myReduction;
	__syncthreads();

	/* sdata-a */

	// do reduction in shared mem
	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] = myReduction = bop(myReduction, sdata[tid + 512]);
		}

		__syncthreads();
	}

	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] = myReduction = bop(myReduction, sdata[tid + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] = myReduction = bop(myReduction, sdata[tid + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] = myReduction = bop(myReduction, sdata[tid + 64]);
		}

		__syncthreads();
	}

	if (tid < 32) {
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn'float reorder stores to it and induce incorrect behavior.
		//volatile T* smem = sdata;

		if (blockSize >= 64) {
			myReduction = bop(myReduction, sdata[tid + 32]);
		}

		if (blockSize >= 32) {
			myReduction =bop(myReduction,  shfl(myReduction, tid + 16));
		}

		if (blockSize >= 16) {
			myReduction =bop(myReduction,  shfl(myReduction, tid + 8));
		}

		if (blockSize >= 8) {
			myReduction =bop(myReduction,  shfl(myReduction, tid + 4));
		}

		if (blockSize >= 4) {
			myReduction =bop(myReduction,  shfl(myReduction, tid + 2));
		}

		if (blockSize >= 2) {
			myReduction =bop(myReduction,  shfl(myReduction, tid + 1));
		}

	}

	if (tid == 0)
		g_odata[blockIdx.x] = myReduction;
}


// input pass works across two matrices via BinaryOp1 'mop' (global to local), subsequent passes are regular self-reductions using binaryop 'op'
#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp1, template <typename> class BinaryOp2>
__host__ CUDART_DEVICE T combineReduceOpLauncher(T* d_odata, const T* d_idata1, const T* d_idata2, long n, BinaryOp1<T> mop, BinaryOp2<T> bop, T start, cudaStream_t stream )
#else
template<typename T, int MopDim, int BopDim>
__host__ CUDART_DEVICE T combineReduceOpLauncher(T* d_odata, const T* d_idata1, const T* d_idata2, long n, BinaryOpF<T,MopDim> mop, BinaryOpF<T,BopDim> bop, T start, cudaStream_t stream )
#endif
{
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		if( checkDebug(debugRedux)) flprintf("mop.fn %p, bop.fn %p\n", mop.fn, bop.fn);
	#else
		if( checkDebug(debugRedux)) flprintf("mop.operation %p, bop.operation %p\n", mop.operation, bop.operation);
	#endif
#endif

#ifndef __CUDA_ARCH__
	if(checkDebug(debugRedux)){outln("mop type " << b_util::unmangl(typeid(mop).name()));}
	T gpu_result = 0;
#endif
	//printf("combineReduceOpLauncher\n");

	// sum partial block sums on GPU
	int blocks,threads;
	bool powOf2Q;
	bool firstRedux = true;
	dim3 dimBlock;
	dim3 dimGrid;

	while ( n > 1) {
		powOf2Q = b_util::isPow2(n);
		getReductionExecContext(blocks,threads, n);
		int smemSize =
				(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
		dimBlock.x = threads;
		dimGrid.x = blocks;
		if( checkDebug(debugExec)) {
			flprintf("n %d\n",n);
			prlocf("dimGrid "); b_util::prd3(dimGrid);
			prlocf("dimBlock "); b_util::prd3(dimBlock);
			flprintf("smemSize %d\n",smemSize);
			flprintf("smem Ts %d\n",smemSize/sizeof(T));
		}
		if(firstRedux) {
			if (powOf2Q) {
				switch (threads) {
#ifdef  CuMatrix_Enable_KTS
					case 1024:
					combineReduceOpKernel<T, 1024, true, BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 512:
					combineReduceOpKernel<T, 512, true, BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 256:
					combineReduceOpKernel<T, 256, true,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 128:
					combineReduceOpKernel<T, 128, true,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 64:
					combineReduceOpKernel<T, 64, true,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 32:
					combineReduceOpKernel<T, 32, true,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop,bop, start); break;
					case 16:
					combineReduceOpKernel<T, 16, true,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 8:
					combineReduceOpKernel<T, 8, true,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 4:
					combineReduceOpKernel<T, 4, true,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 2:
					combineReduceOpKernel<T, 2, true,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 1:
					combineReduceOpKernel<T, 1, true,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>( d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
#else
					case 1024:
					combineReduceOpKernel<T, 1024, true, MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 512:
					combineReduceOpKernel<T, 512, true, MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 256:
					combineReduceOpKernel<T, 256, true,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 128:
					combineReduceOpKernel<T, 128, true,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 64:
					combineReduceOpKernel<T, 64, true,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 32:
					combineReduceOpKernel<T, 32, true,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop,bop, start); break;
					case 16:
					combineReduceOpKernel<T, 16, true,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 8:
					combineReduceOpKernel<T, 8, true,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 4:
					combineReduceOpKernel<T, 4, true,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 2:
					combineReduceOpKernel<T, 2, true,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 1:
					combineReduceOpKernel<T, 1, true,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>( d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
#endif
					}
				if(checkDebug(debugRedux)){prlocf("launched power-of-2 combineReduceOpKernel\n");}
			}else {
				switch (threads) {
#ifdef  CuMatrix_Enable_KTS
					case 1024:
					combineReduceOpKernel<T, 1024, false, BinaryOp1, BinaryOp2> <<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 512:
					combineReduceOpKernel<T, 512, false, BinaryOp1, BinaryOp2> <<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 256:
					combineReduceOpKernel<T, 256, false,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 128:
					combineReduceOpKernel<T, 128, false,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 64:
					combineReduceOpKernel<T, 64, false,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 32:
					combineReduceOpKernel<T, 32, false,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 16:
					combineReduceOpKernel<T, 16, false,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 8:
					combineReduceOpKernel<T, 8, false,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 4:
					combineReduceOpKernel<T, 4, false,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 2:
					combineReduceOpKernel<T, 2, false,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 1:
					combineReduceOpKernel<T, 1, false,BinaryOp1, BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>( d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
#else
					case 1024:
					combineReduceOpKernel<T, 1024, false, MopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 512:
					combineReduceOpKernel<T, 512, false, MopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 256:
					combineReduceOpKernel<T, 256, false,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 128:
					combineReduceOpKernel<T, 128, false,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 64:
					combineReduceOpKernel<T, 64, false,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 32:
					combineReduceOpKernel<T, 32, false,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 16:
					combineReduceOpKernel<T, 16, false,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 8:
					combineReduceOpKernel<T, 8, false,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 4:
					combineReduceOpKernel<T, 4, false,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 2:
					combineReduceOpKernel<T, 2, false,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
					case 1:
					combineReduceOpKernel<T, 1, false,MopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>( d_idata1, d_idata2, d_odata, n, mop, bop, start); break;
#endif
					}
				if(checkDebug(debugRedux)){prlocf("launched non-power-of-2 combineReduceOpKernel\n");}
			}
		}
		else
		{
			if (powOf2Q) {
				switch (threads) {
#ifdef  CuMatrix_Enable_KTS
				case 1024:
					//reduceOpKernel(DMatrix<T> out, const DMatrix<T> src, T* g_odata, long n, BinaryOp2 op, T start)
					reduceOpKernel<T, 1024, true, BinaryOp2> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,bop, start); break;
				case 512:
					reduceOpKernel<T, 512, true, BinaryOp2> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 256:
					reduceOpKernel<T, 256, true,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 128:
					reduceOpKernel<T, 128, true,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 64:
					reduceOpKernel<T, 64, true,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 32:
					reduceOpKernel<T, 32, true,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 16:
					reduceOpKernel<T, 16, true,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 8:
					reduceOpKernel<T, 8, true,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 4:
					reduceOpKernel<T, 4, true,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 2:
					reduceOpKernel<T, 2, true,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 1:
					reduceOpKernel<T, 1, true,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
#else
				case 1024:
					//reduceOpKernel(DMatrix<T> out, const DMatrix<T> src, T* g_odata, long n, BinaryOp2 op, T start)
					reduceOpKernel<T, 1024, true, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,bop, start); break;
				case 512:
					reduceOpKernel<T, 512, true, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 256:
					reduceOpKernel<T, 256, true,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 128:
					reduceOpKernel<T, 128, true,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 64:
					reduceOpKernel<T, 64, true,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 32:
					reduceOpKernel<T, 32, true,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 16:
					reduceOpKernel<T, 16, true,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 8:
					reduceOpKernel<T, 8, true,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 4:
					reduceOpKernel<T, 4, true,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 2:
					reduceOpKernel<T, 2, true,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 1:
					reduceOpKernel<T, 1, true,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
#endif
					}
				if(checkDebug(debugRedux))prlocf("launched power-of-2 reduceOpKernel\n");
			} else {
				switch (threads) {
#ifdef  CuMatrix_Enable_KTS
				case 1024:
					reduceOpKernel<T, 1024, false, BinaryOp2> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 512:
					reduceOpKernel<T, 512, false, BinaryOp2> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 256:
					reduceOpKernel<T, 256, false,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 128:
					reduceOpKernel<T, 128, false,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 64:
					reduceOpKernel<T, 64, false,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 32:
					reduceOpKernel<T, 32, false,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 16:
					reduceOpKernel<T, 16, false,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 8:
					reduceOpKernel<T, 8, false,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 4:
					reduceOpKernel<T, 4, false,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 2:
					reduceOpKernel<T, 2, false,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 1:
					reduceOpKernel<T, 1, false,BinaryOp2><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
#else
				case 1024:
					reduceOpKernel<T, 1024, false, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 512:
					reduceOpKernel<T, 512, false, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 256:
					reduceOpKernel<T, 256, false,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 128:
					reduceOpKernel<T, 128, false,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 64:
					reduceOpKernel<T, 64, false,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 32:
					reduceOpKernel<T, 32, false,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 16:
					reduceOpKernel<T, 16, false,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 8:
					reduceOpKernel<T, 8, false,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 4:
					reduceOpKernel<T, 4, false,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 2:
					reduceOpKernel<T, 2, false,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
				case 1:
					reduceOpKernel<T, 1, false,BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata,d_odata, n,bop, start); break;
#endif
					}
				if(checkDebug(debugRedux))prlocf("launched non-power-of-2 reduceOpKernel\n");
			}
		}
#ifndef __CUDA_ARCH__
		if(stream!=null) {
			cudaStreamSynchronize(stream);
			if(checkDebug(debugRedux))flprintf("h synced stream %p\n", stream);
		} else  {
			cudaDeviceSynchronize();
			if(checkDebug(debugRedux))prlocf("h synced null stream\n");
		}
		checkCudaError(cudaGetLastError());
#else
		cherr(cudaDeviceSynchronize());
		if(checkDebug(debugRedux))prlocf("d synced null stream\n");
#endif
		if(firstRedux) {
			firstRedux = false;
		}
		n = blocks;
	}

	// copy final sum from device to host
#ifndef __CUDA_ARCH__
	if(checkDebug(debugRedux)){prlocf("host copying result to gpu_result and retoining\n");}
	cudaError_t cuda_error = cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost);
	checkCudaError(cuda_error);
	return gpu_result;
#else
	if(checkDebug(debugRedux)){prlocf("dev return d_odata[0]\n");}
	return d_odata[0];
#endif
}

#ifdef  CuMatrix_Enable_KTS

template __host__ CUDART_DEVICE long combineReduceOpLauncher<long, equalsBinaryOp, plusBinaryOp>(long*, long const*, long const*, long, equalsBinaryOp<long>, plusBinaryOp<long>, long, CUstream_st*);
template __host__ CUDART_DEVICE  float combineReduceOpLauncher<float, equalsBinaryOp, andBinaryOp>(float*, float const*, float const*, long, equalsBinaryOp<float>, andBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE  unsigned long combineReduceOpLauncher<unsigned long, multBinaryOp, plusBinaryOp>(unsigned long*, unsigned long const*, unsigned long const*, long, multBinaryOp<unsigned long>, plusBinaryOp<unsigned long>, unsigned long, CUstream_st*);
template __host__ CUDART_DEVICE  unsigned int combineReduceOpLauncher<unsigned int, diffSquaredBinaryOp, plusBinaryOp>(unsigned int*, unsigned int const*, unsigned int const*, long, diffSquaredBinaryOp<unsigned int>, plusBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE  double combineReduceOpLauncher<double, diffSquaredBinaryOp, plusBinaryOp>(double*, double const*, double const*, long, diffSquaredBinaryOp<double>, plusBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE  int combineReduceOpLauncher<int, almostEqualsBinaryOp, andBinaryOp>(int*, int const*, int const*, long, almostEqualsBinaryOp<int>, andBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE  unsigned long combineReduceOpLauncher<unsigned long, equalsBinaryOp, andBinaryOp>(unsigned long*, unsigned long const*, unsigned long const*, long, equalsBinaryOp<unsigned long>, andBinaryOp<unsigned long>, unsigned long, CUstream_st*);
template __host__ CUDART_DEVICE  unsigned int combineReduceOpLauncher<unsigned int, equalsBinaryOp, andBinaryOp>(unsigned int*, unsigned int const*, unsigned int const*, long, equalsBinaryOp<unsigned int>, andBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE  double combineReduceOpLauncher<double, equalsBinaryOp, andBinaryOp>(double*, double const*, double const*, long, equalsBinaryOp<double>, andBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE  float combineReduceOpLauncher<float, diffSquaredBinaryOp, plusBinaryOp>(float*, float const*, float const*, long, diffSquaredBinaryOp<float>, plusBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE  double combineReduceOpLauncher<double, multBinaryOp, plusBinaryOp>(double*, double const*, double const*, long, multBinaryOp<double>, plusBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE  int combineReduceOpLauncher<int, equalsBinaryOp, plusBinaryOp>(int*, int const*, int const*, long, equalsBinaryOp<int>, plusBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE  unsigned long combineReduceOpLauncher<unsigned long, diffSquaredBinaryOp, plusBinaryOp>(unsigned long*, unsigned long const*, unsigned long const*, long, diffSquaredBinaryOp<unsigned long>, plusBinaryOp<unsigned long>, unsigned long, CUstream_st*);
template __host__ CUDART_DEVICE  unsigned int combineReduceOpLauncher<unsigned int, equalsBinaryOp, plusBinaryOp>(unsigned int*, unsigned int const*, unsigned int const*, long, equalsBinaryOp<unsigned int>, plusBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE  double combineReduceOpLauncher<double, equalsBinaryOp, plusBinaryOp>(double*, double const*, double const*, long, equalsBinaryOp<double>, plusBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE  long combineReduceOpLauncher<long, multBinaryOp, plusBinaryOp>(long*, long const*, long const*, long, multBinaryOp<long>, plusBinaryOp<long>, long, CUstream_st*);
template __host__ CUDART_DEVICE  long combineReduceOpLauncher<long, equalsBinaryOp, andBinaryOp>(long*, long const*, long const*, long, equalsBinaryOp<long>, andBinaryOp<long>, long, CUstream_st*);
template __host__ CUDART_DEVICE  float combineReduceOpLauncher<float, almostEqualsBinaryOp, andBinaryOp>(float*, float const*, float const*, long, almostEqualsBinaryOp<float>, andBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE  int combineReduceOpLauncher<int, diffSquaredBinaryOp, plusBinaryOp>(int*, int const*, int const*, long, diffSquaredBinaryOp<int>, plusBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE  unsigned int combineReduceOpLauncher<unsigned int, almostEqualsBinaryOp, andBinaryOp>(unsigned int*, unsigned int const*, unsigned int const*, long, almostEqualsBinaryOp<unsigned int>, andBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE  long combineReduceOpLauncher<long, diffSquaredBinaryOp, plusBinaryOp>(long*, long const*, long const*, long, diffSquaredBinaryOp<long>, plusBinaryOp<long>, long, CUstream_st*);
template __host__ CUDART_DEVICE  float combineReduceOpLauncher<float, equalsBinaryOp, plusBinaryOp>(float*, float const*, float const*, long, equalsBinaryOp<float>, plusBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE  int combineReduceOpLauncher<int, equalsBinaryOp, andBinaryOp>(int*, int const*, int const*, long, equalsBinaryOp<int>, andBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE  unsigned long combineReduceOpLauncher<unsigned long, equalsBinaryOp, plusBinaryOp>(unsigned long*, unsigned long const*, unsigned long const*, long, equalsBinaryOp<unsigned long>, plusBinaryOp<unsigned long>, unsigned long, CUstream_st*);
template __host__ CUDART_DEVICE  float combineReduceOpLauncher<float, multBinaryOp, plusBinaryOp>(float*, float const*, float const*, long, multBinaryOp<float>, plusBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE  unsigned long combineReduceOpLauncher<unsigned long, almostEqualsBinaryOp, andBinaryOp>(unsigned long*, unsigned long const*, unsigned long const*, long, almostEqualsBinaryOp<unsigned long>, andBinaryOp<unsigned long>, unsigned long, CUstream_st*);
template __host__ CUDART_DEVICE  double combineReduceOpLauncher<double, almostEqualsBinaryOp, andBinaryOp>(double*, double const*, double const*, long, almostEqualsBinaryOp<double>, andBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE  unsigned int combineReduceOpLauncher<unsigned int, multBinaryOp, plusBinaryOp>(unsigned int*, unsigned int const*, unsigned int const*, long, multBinaryOp<unsigned int>, plusBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE  long combineReduceOpLauncher<long, almostEqualsBinaryOp, andBinaryOp>(long*, long const*, long const*, long, almostEqualsBinaryOp<long>, andBinaryOp<long>, long, CUstream_st*);
template __host__ CUDART_DEVICE  int combineReduceOpLauncher<int, multBinaryOp, plusBinaryOp>(int*, int const*, int const*, long, multBinaryOp<int>, plusBinaryOp<int>, int, CUstream_st*);

#else
template __host__ CUDART_DEVICE float combineReduceOpLauncher<float, 0, 1>(float*, float const*, float const*, long, BinaryOpF<float, 0>, BinaryOpF<float, 1>, float, CUstream_st*);
template __host__ CUDART_DEVICE float combineReduceOpLauncher<float, 1, 1>(float*, float const*, float const*, long, BinaryOpF<float, 1>, BinaryOpF<float, 1>, float, CUstream_st*);
template __host__ CUDART_DEVICE double combineReduceOpLauncher<double, 0, 1>(double*, double const*, double const*, long, BinaryOpF<double, 0>, BinaryOpF<double, 1>, double, CUstream_st*);
template __host__ CUDART_DEVICE double combineReduceOpLauncher<double, 1, 1>(double*, double const*, double const*, long, BinaryOpF<double, 1>, BinaryOpF<double, 1>, double, CUstream_st*);
template __host__ CUDART_DEVICE int combineReduceOpLauncher<int, 0, 1>(int*, int const*, int const*, long, BinaryOpF<int, 0>, BinaryOpF<int, 1>, int, CUstream_st*);
template __host__ CUDART_DEVICE int combineReduceOpLauncher<int, 1, 1>(int*, int const*, int const*, long, BinaryOpF<int, 1>, BinaryOpF<int, 1>, int, CUstream_st*);
template __host__ CUDART_DEVICE uint combineReduceOpLauncher<uint, 0, 1>(uint*, uint const*, uint const*, long, BinaryOpF<uint, 0>, BinaryOpF<uint, 1>, uint, CUstream_st*);
template __host__ CUDART_DEVICE uint combineReduceOpLauncher<uint, 1, 1>(uint*, uint const*, uint const*, long, BinaryOpF<uint, 1>, BinaryOpF<uint, 1>, uint, CUstream_st*);
template __host__ CUDART_DEVICE long combineReduceOpLauncher<long, 0, 1>(long*, long const*, long const*, long, BinaryOpF<long, 0>, BinaryOpF<long, 1>, long, CUstream_st*);
template __host__ CUDART_DEVICE long combineReduceOpLauncher<long, 1, 1>(long*, long const*, long const*, long, BinaryOpF<long, 1>, BinaryOpF<long, 1>, long, CUstream_st*);
template __host__ CUDART_DEVICE ulong combineReduceOpLauncher<ulong, 0, 1>(ulong*, ulong const*, ulong const*, long, BinaryOpF<ulong, 0>, BinaryOpF<ulong, 1>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE ulong combineReduceOpLauncher<ulong, 1, 1>(ulong*, ulong const*, ulong const*, long, BinaryOpF<ulong, 1>, BinaryOpF<ulong, 1>, ulong, CUstream_st*);
#endif


#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class BinaryOp>
__global__ void reduceOpKernel( T* g_odata, const T* g_idata, long n,
		BinaryOp<T> op, T start, uint stride, uint offset)
#else
template<typename T, uint blockSize, bool nIsPow2, int StateDim>
__global__ void reduceOpKernel( T* g_odata, const T* g_idata, long n,
		BinaryOpF<T,StateDim> op, T start, uint stride, uint offset)
#endif
{
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	uint tid = threadIdx.x;
	ulong i = blockIdx.x * blockSize * 2 + threadIdx.x;
	uint gridSize = blockSize * 2 * gridDim.x;
	T myReduction = start;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n) {
		//printf("op(%f, %f): ", myReduction, g_idata[i] );
		myReduction = op(myReduction, g_idata[offset + i * stride]);
		//printf("%f\n", myReduction  );
		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction = op(myReduction, g_idata[offset + (i + blockSize) * stride]);
		i += gridSize;
	}
	// each thread puts its local sum into shared memory
	sdata[tid] = myReduction;
	__syncthreads();

	/* sdata-a */

	// do reduction in shared mem
	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] = myReduction = op(myReduction, sdata[tid + 512]);
		}

		__syncthreads();
	}

	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] = myReduction = op(myReduction, sdata[tid + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] = myReduction = op(myReduction, sdata[tid + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] = myReduction = op(myReduction, sdata[tid + 64]);
		}

		__syncthreads();
	}

	if (tid < 32) {
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn'float reorder stores to it and induce incorrect behavior.
		if (blockSize >= 64) {
				myReduction = op(myReduction, sdata[tid + 32]);
		}

		if (blockSize >= 32) {
			myReduction =op(myReduction,  shfl(myReduction, tid + 16));
		}

		if (blockSize >= 16) {
			myReduction =op(myReduction,  shfl(myReduction, tid + 8));
		}

		if (blockSize >= 8) {
			myReduction =op(myReduction,  shfl(myReduction, tid + 4));
		}

		if (blockSize >= 4) {
			myReduction =op(myReduction,  shfl(myReduction, tid + 2));
		}

		if (blockSize >= 2) {
			myReduction =op(myReduction,  shfl(myReduction, tid + 1));
		}

	}

	if (tid == 0)
		g_odata[blockIdx.x] = myReduction;
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class BinaryOp>
__global__ void reduceOpKernel(DMatrix<T> out, const DMatrix<T> src, long n, BinaryOp<T> op, T start, uint stride, uint offset)
#else
template<typename T, uint blockSize, bool nIsPow2, int StateDim>
__global__ void reduceOpKernel(DMatrix<T> out, const DMatrix<T> src, long n, BinaryOpF<T,StateDim> op, T start, uint stride, uint offset)
#endif
{

	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	uint tid = threadIdx.x;
	ulong i = blockIdx.x * blockSize * 2 + threadIdx.x ;
	uint gridSize = blockSize * 2 * gridDim.x;

	T myReduction = start;

	FirstThread {
		if(checkDebug(debugRedux))
			flprintf("in reduceOpKernel blockSize %d n %d, src.elements %p last %p\n",blockSize,  n, src.elements + offset, src.elements + (n-1) * stride);
	}

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n) {
		myReduction = op(myReduction, get(src, offset + i )); // in effect, there may be two pitches; one for matrix itself, and the other for the column if stride > 1
		if(checkDebug(debugRedux) && i == n - 1 )
	 			flprintf("i == n - 1!, i %lu reading src(%p)\n", i, src.elements + i);
		//if(tid==0)printf("reduceOpKernel i %d op(%f, %f) \n", i, myReduction,get(src,i));
		//printf("%f\n", myReduction  );
		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 ||i + blockSize < n)
			myReduction = op(myReduction, get(src,offset + (i + blockSize)));
		if(checkDebug(debugRedux) && i + blockSize== n - 1 )
			flprintf("i + blockSize== n - 1!, i %lu reading src(%p)\n", i, src.elements + offset + (i + blockSize) * stride);
		i += gridSize;
	}
	// each thread puts its local sum into shared memory
	sdata[tid] = myReduction;
	__syncthreads();

	/* sdata-a */

	// do reduction in shared mem
	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] = myReduction = op(myReduction, sdata[tid + 512]);
		}
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] = myReduction = op(myReduction, sdata[tid + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] = myReduction = op(myReduction, sdata[tid + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] = myReduction = op(myReduction, sdata[tid + 64]);
		}

		__syncthreads();
	}

	if (tid < 32) {
		if (blockSize >= 64) {
				myReduction = op(myReduction, sdata[tid + 32]);
		}

		if (blockSize >= 32) {
			myReduction =op(myReduction,  shfl(myReduction, tid + 16));
		}

		if (blockSize >= 16) {
			myReduction =op(myReduction,  shfl(myReduction, tid + 8));
		}

		if (blockSize >= 8) {
			myReduction =op(myReduction,  shfl(myReduction, tid + 4));
		}

		if (blockSize >= 4) {
			myReduction =op(myReduction,  shfl(myReduction, tid + 2));
		}

		if (blockSize >= 2) {
			myReduction =op(myReduction,  shfl(myReduction, tid + 1));
		}

	}

	if (tid == 0) {
		out.elements[blockIdx.x] = myReduction;
		if(checkDebug(debugRedux))
			flprintf("reduceOpKernel setting %p[%d] to %f\n", out.elements, blockIdx.x, myReduction);
	}
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __host__ CUDART_DEVICE void reduceLauncher(T* result, DMatrix<T> buff, long n, const DMatrix<T> src,
		BinaryOp<T> op, T start, int stride, int offset, cudaStream_t stream)
#else
template<typename T, int StateDim> __host__ CUDART_DEVICE void reduceLauncher(T* result, DMatrix<T> buff, long n, const DMatrix<T> src,
		MonoidF<T,StateDim> op, T start, int stride, int offset, cudaStream_t stream)
#endif
{
 	if(!result) {
		setLastError(nullPointerEx);
	}
	cherr(cudaPeekAtLastError());
	DMatrix<T> rSrc(src);
	cherr(cudaDeviceSynchronize());
	bool powOf2Q;
	bool firstRedux = true;
	dim3 dimBlock;
	dim3 dimGrid;
	int threads, blocks;
	while ( n > 1 ) {
		powOf2Q = b_util::isPow2(n);
		if(checkDebug(debugRedux)) flprintf("n %d b_util::isPow2(n) %s\n", n, tOrF(b_util::isPow2(n) ));
		getReductionExecContext(blocks,threads, n);
		int smemSize =
				(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
		dimBlock.x = threads;
		dimGrid.x = blocks;
		if(checkDebug(debugRedux)) {
			flprintf("reduceLauncher smemSize %d buff %p buff.elems %p bl %d th %d n %u stride %u, src.elements %p last %p\n",
				smemSize, &buff,buff.elements, blocks,threads,n, stride, src.elements, src.elements - 1 + n);
		}
#ifndef __CUDA_ARCH__
		checkCudaError(cudaGetLastError());
		flush(cout);
#endif
		if (powOf2Q) {
			if(checkDebug(debugRedux)) {prlocf("power of 2\n");}
			switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
				//reduceOpKernel(DMatrix<T> out, const DMatrix<T> src, T* g_odata, long n, BinaryOp op, T start)
				reduceOpKernel<T, 1024, true, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 512:
				reduceOpKernel<T, 512, true, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 256:
				reduceOpKernel<T, 256, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 128:
				reduceOpKernel<T, 128, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 64:
				reduceOpKernel<T, 64, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 32:
				reduceOpKernel<T, 32, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
#else
			case 1024:
				//reduceOpKernel(DMatrix<T> out, const DMatrix<T> src, T* g_odata, long n, BinaryOp op, T start)
				reduceOpKernel<T, 1024, true, StateDim> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 512:
				reduceOpKernel<T, 512, true, StateDim> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 256:
				reduceOpKernel<T, 256, true, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 128:
				reduceOpKernel<T, 128, true, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 64:
				reduceOpKernel<T, 64, true, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 32:
				reduceOpKernel<T, 32, true, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
#endif
			case 16:
			case 8:
			case 4:
			case 2:
			case 1:
				//shuffle(T* res, const T* a, int len, BinaryOpF bop)
				// only skepticism until tested
				if(checkDebug(debugRedux)) {flprintf("shuffln curr n %d \n" , n);}
#ifdef __CUDA_ARCH__
				int idx = threadIdx.x;
				int len2 = b_util::nextPowerOf2(n);
				int currLen = len2/2;
				T s = idx < n ? rSrc.elements[idx]: start;
				T s0 = s;
				while(currLen > 0 ) {
					if(checkDebug(debugRedux)&&threadIdx.x == 0 && blockIdx.x == 0){flprintf("currLen %d\n", currLen);}
					int lane = idx + currLen;
					//s = lane < len ? bop(s, __shfl(s, lane)) : s;
					s = op(s, shfl<T>(s, lane));
					//s = bop(s, lane < len ? __shfl(s, lane) : bop.identity);
					if(checkDebug(debugRedux)&&threadIdx.x == 0 && blockIdx.x == 0){flprintf("idx: %d, lane %d, s0 %f, s %f\n", idx, lane,  s0, s);}
					s0 += s;
					currLen >>= 1;
				}
				if(idx == 0) {
					buff.elements[0] = s;
				}
#else
	#ifdef  CuMatrix_Enable_KTS
				*result = shuffle<T>( (const T*) rSrc.elements, n, op);
	#else
				*result = shuffle<T,StateDim>( (const T*) rSrc.elements, n, op);
	#endif

#endif
				return;
			}
		} else {
			if(checkDebug(debugRedux)){flprintf("!power of 2 threads %d, buff.elements %p, rSrc.elements %p\n", threads,buff.elements, rSrc.elements);}
			cherr(cudaDeviceSynchronize());
			cherr(cudaPeekAtLastError());
			switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
				reduceOpKernel<T, 1024, false, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 512:
				reduceOpKernel<T, 512, false, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 256:
				reduceOpKernel<T, 256, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 128:
				reduceOpKernel<T, 128, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 64:
				reduceOpKernel<T, 64, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 32:
				reduceOpKernel<T, 32, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 16:
				reduceOpKernel<T, 16, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 8:
				reduceOpKernel<T, 8, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 4:
				reduceOpKernel<T, 4, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 2:
				reduceOpKernel<T, 2, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 1:
				reduceOpKernel<T, 1, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
#else
			case 1024:
				reduceOpKernel<T, 1024, false, StateDim> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 512:
				reduceOpKernel<T, 512, false, StateDim> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 256:
				reduceOpKernel<T, 256, false, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 128:
				reduceOpKernel<T, 128, false, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 64:
				reduceOpKernel<T, 64, false, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 32:
				reduceOpKernel<T, 32, false, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 16:
				reduceOpKernel<T, 16, false, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 8:
				reduceOpKernel<T, 8, false, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 4:
				reduceOpKernel<T, 4, false, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 2:
				reduceOpKernel<T, 2, false, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
			case 1:
				reduceOpKernel<T, 1, false, StateDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,op, start,stride, offset); break;
#endif
			}
		}

		stride =1; // only meaningful for first pass
		offset =0;
#ifndef __CUDA_ARCH__
		checkCudaError(cudaGetLastError());
		if(stream!=null){
			checkCudaError(cudaStreamSynchronize(stream));
		} else  {
			checkCudaError(cudaDeviceSynchronize());
		}
#else
		cherr(cudaDeviceSynchronize());
#endif
		if(firstRedux) {
			if(checkDebug(debugRedux))prlocf("reduceLauncher after first redux, setting input to output\n");
			if(checkDebug(debugRedux)) printArray(buff.elements,20);
			rSrc.elements  = buff.elements;

			rSrc.m = buff.m;
			rSrc.p = buff.p;
			rSrc.n = buff.n;
			firstRedux = false;
		}
		if(checkDebug(debugRedux)){flprintf("reduceLauncher old n %lu\n", n);}
		n = blocks; // (n + (threads * 2 - 1)) / (threads * 2);
		if(checkDebug(debugRedux)){flprintf("reduceLauncher new n %lu\n", n);}
	}

	if(checkDebug(debugRedux)){
		prlocf("reduceLauncher after reduceOpKernel\n");
	}
#ifdef __CUDA_ARCH__
	if(checkDebug(debugRedux)){
		flprintf(" buff 0x%p buff[0] %f buff[1] %f \n",buff.elements,buff.elements[0],buff.elements[1]);
	}
#endif
	// copy final sum from device to host	cudaDeviceSynchronize();

#ifndef __CUDA_ARCH__

	checkCudaError(cudaGetLastError());
	struct cudaPointerAttributes ptrAtts;
	//ptrAtts.memoryType = dev ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
	checkCudaError( cudaPointerGetAttributes(&ptrAtts, buff.elements));
	if(checkDebug(debugRedux)) {
		outln("result " << result );
		outln("buff.elements " << buff.elements << ", ptrAtts dev " << ptrAtts.device);
	}
	T localRes;
	T *devLocal;
	checkCudaError(cudaMalloc(&devLocal,sizeof(T)));
	checkCudaError(
			cudaMemcpy/*Async*/(&localRes, devLocal, sizeof(T), cudaMemcpyDeviceToHost/*, stream*/));
	checkCudaError(
			cudaMemcpy/*Async*/(&localRes, buff.elements, sizeof(T), cudaMemcpyDeviceToHost/*, stream*/));

	checkCudaError(cudaFree(devLocal));

	if(checkDebug(debugCopyDh))outln("debugCopyDh ::reduceLauncher localRes " << localRes);
	*result=localRes;
	CuMatrix<T>::DHCopied++;
	CuMatrix<T>::MemDhCopied += sizeof(T);
#else
	__syncthreads();
	memcpy(result, buff.elements, sizeof(T));
#endif

}

#ifdef  CuMatrix_Enable_KTS

template __host__ CUDART_DEVICE void reduceLauncher<float,plusBinaryOp>( float*, DMatrix<float>,  long, const DMatrix<float>, plusBinaryOp<float>, float, int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<double,plusBinaryOp>(double*, DMatrix<double>,  long, const DMatrix<double>, plusBinaryOp<double>, double,int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<ulong,plusBinaryOp>(ulong*, DMatrix<ulong>,  long, const DMatrix<ulong>, plusBinaryOp<ulong>, ulong,int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<long, plusBinaryOp>(long*, DMatrix<long>, long, DMatrix<long>, plusBinaryOp<long>, long, int,int, CUstream_st*);

template __host__ CUDART_DEVICE void reduceLauncher<float,multBinaryOp>(float*, DMatrix<float>,  long, const DMatrix<float>,multBinaryOp<float>,float, int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<double,multBinaryOp>(double*, DMatrix<double>,  long, const DMatrix<double>, multBinaryOp<double>, double,int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<ulong, multBinaryOp>(ulong*, DMatrix<ulong>, long, DMatrix<ulong>, multBinaryOp<ulong>, ulong, int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<long, multBinaryOp>(long*, DMatrix<long>, long, DMatrix<long>, multBinaryOp<long>, long, int,int, CUstream_st*);

template __host__ CUDART_DEVICE void reduceLauncher<float,maxBinaryOp>(float*, DMatrix<float>,  long, const DMatrix<float>, maxBinaryOp<float>,float, int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<double,maxBinaryOp>(double*, DMatrix<double>,  long, const DMatrix<double>, maxBinaryOp<double>, double,int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<ulong,maxBinaryOp>(ulong*, DMatrix<ulong>,  long, const DMatrix<ulong>, maxBinaryOp<ulong>, ulong,int,int,cudaStream_t);

template __host__ CUDART_DEVICE void reduceLauncher<float,minBinaryOp>(float*, DMatrix<float>,  long, const DMatrix<float>, minBinaryOp<float>,float, int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<double,minBinaryOp>(double*, DMatrix<double>,  long, const DMatrix<double>, minBinaryOp<double>, double,int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<ulong,minBinaryOp>(ulong*, DMatrix<ulong>,  long, const DMatrix<ulong>, minBinaryOp<ulong>, ulong,int,int,cudaStream_t);

template __host__ CUDART_DEVICE void reduceLauncher<float, andBinaryOp>(float*, DMatrix<float>, long, DMatrix<float>, andBinaryOp<float>, float, int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<double, andBinaryOp>(double*, DMatrix<double>, long, DMatrix<double>, andBinaryOp<double>, double,int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<ulong, andBinaryOp>(ulong*, DMatrix<ulong>, long, DMatrix<ulong>, andBinaryOp<ulong>, ulong,int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<long, andBinaryOp>(long*, DMatrix<long>, long, DMatrix<long>, andBinaryOp<long>, long, int, int, CUstream_st*);

template __host__ CUDART_DEVICE void reduceLauncher<float, orBinaryOp>(float*, DMatrix<float>, long, DMatrix<float>, orBinaryOp<float>, float, int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<double, orBinaryOp>(double*, DMatrix<double>, long, DMatrix<double>, orBinaryOp<double>, double,int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<ulong, orBinaryOp>(ulong*, DMatrix<ulong>, long, DMatrix<ulong>, orBinaryOp<ulong>, ulong,int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<long, orBinaryOp>(long*, DMatrix<long>, long, DMatrix<long>, orBinaryOp<long>, long,int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<int, orBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, orBinaryOp<int>, int, int, int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<unsigned int, orBinaryOp>(unsigned int*, DMatrix<unsigned int>, long, DMatrix<unsigned int>, orBinaryOp<unsigned int>, unsigned int, int, int, CUstream_st*);

template __host__ CUDART_DEVICE void reduceLauncher<int, maxBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, maxBinaryOp<int>, int, int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<int, minNotZeroBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, minNotZeroBinaryOp<int>, int, int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<long, maxBinaryOp>(long*, DMatrix<long>, long, DMatrix<long>, maxBinaryOp<long>, long, int, int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<long, minBinaryOp>(long*, DMatrix<long>, long, DMatrix<long>, minBinaryOp<long>, long, int, int, CUstream_st*);

template __host__ CUDART_DEVICE void reduceLauncher<int, plusBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, plusBinaryOp<int>, int, int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<unsigned int, plusBinaryOp>(unsigned int*, DMatrix<unsigned int>, long, DMatrix<unsigned int>, plusBinaryOp<unsigned int>, uint,int, int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<int, multBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, multBinaryOp<int>, int, int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<unsigned int, multBinaryOp>(unsigned int*, DMatrix<unsigned int>, long, DMatrix<unsigned int>, multBinaryOp<unsigned int>, uint,int, int, CUstream_st*);

template __host__ CUDART_DEVICE void reduceLauncher<int, andBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, andBinaryOp<int>, int, int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<unsigned int, andBinaryOp>(unsigned int*, DMatrix<unsigned int>, long, DMatrix<unsigned int>, andBinaryOp<unsigned int>, uint,int, int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<int, minBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, minBinaryOp<int>, int, int,int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<unsigned int, maxBinaryOp>(unsigned int*, DMatrix<unsigned int>, long, DMatrix<unsigned int>, maxBinaryOp<unsigned int>, uint,int, int, CUstream_st*);
template __host__ CUDART_DEVICE void reduceLauncher<unsigned int, minBinaryOp>(unsigned int*, DMatrix<unsigned int>, long, DMatrix<unsigned int>, minBinaryOp<unsigned int>, uint,int, int, CUstream_st*);

#else
template __host__ CUDART_DEVICE void reduceLauncher<float,1>(float*, DMatrix<float>,  long, const DMatrix<float>, MonoidF<float,1>, float, int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<double,1>(double*, DMatrix<double>,  long, const DMatrix<double>, MonoidF<double,1>, double, int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<long,1>(long*, DMatrix<long>,  long, const DMatrix<long>, MonoidF<long,1>, long, int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<ulong,1>(ulong*, DMatrix<ulong>,  long, const DMatrix<ulong>, MonoidF<ulong,1>, ulong, int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<int,1>(int*, DMatrix<int>,  long, const DMatrix<int>, MonoidF<int,1>, int, int,int,cudaStream_t);
template __host__ CUDART_DEVICE void reduceLauncher<uint,1>(uint*, DMatrix<uint>,  long, const DMatrix<uint>, MonoidF<uint,1>, uint, int,int,cudaStream_t);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __global__ void reduceLauncherG(T* result, DMatrix<T> buff, long n, const DMatrix<T> src,
		BinaryOp<T> op, T start, uint stride, cudaStream_t stream)
#else
template<typename T, int StateDim> __global__ void reduceLauncherG(T* result, DMatrix<T> buff, long n, const DMatrix<T> src,
		BinaryOpF<T,StateDim> op, T start, uint stride, cudaStream_t stream)
#endif
{
	reduceLauncher(result, buff, n, op, start, stride, 0, stream);
}

#ifdef CuMatrix_Enable_Cdp
	#ifdef  CuMatrix_Enable_KTS
	template<typename T, template <typename> class BinaryOp>  __global__ void  reduceLauncherCount(T* result, DMatrix<T> buff, long nP, const DMatrix<T> src, BinaryOp<T> op, T start, int count)
	#else
	template<typename T, int StateDim>  __global__ void  reduceLauncherCount(T* result, DMatrix<T> buff, long nP, const DMatrix<T> src, MonoidF<T,StateDim> op, T start, int count)
	#endif
	{
		if(threadIdx.x == 0 && blockIdx.x == 0)
		{
				for(int i = 0; i < count; i++) {
					reduceLauncher(result, buff, nP, src, op, start);
				}
		}
	}
	#ifdef  CuMatrix_Enable_KTS
		template __global__ void reduceLauncherCount<float, plusBinaryOp>(float*, DMatrix<float>, long, DMatrix<float>, plusBinaryOp<float>, float, int);
		template __global__ void reduceLauncherCount<double, plusBinaryOp>(double*, DMatrix<double>, long, DMatrix<double>, plusBinaryOp<double>, double, int);
		template __global__ void reduceLauncherCount<unsigned long, plusBinaryOp>(unsigned long*, DMatrix<unsigned long>, long, DMatrix<unsigned long>, plusBinaryOp<unsigned long>, unsigned long, int);
	#else
		template __global__ void reduceLauncherCount<float, 1>(float*, DMatrix<float>, long, DMatrix<float>, MonoidF<float,1>, float, int);
		template __global__ void reduceLauncherCount<double, 1>(double*, DMatrix<double>, long, DMatrix<double>, MonoidF<double,1>, double, int);
		template __global__ void reduceLauncherCount<ulong, 1>(ulong*, DMatrix<ulong>, long, DMatrix<ulong>, MonoidF<ulong, 1>, ulong, int);
	#endif
#endif

