/*
 * CuMatrixIndexedReductions.cu
 *
 *      Author: reid
 */
#include "CuMatrix.h"
#include "Kernels.h"
#include "caps.h"
#include "CuDefs.h"

/*
 * fixme
 * reduces via BinaryOp values that are generated from the thread index via IndexUnaryOp
 *
 */
#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class IndexUnaryOp, template <typename> class BinaryOp>
__global__ void indexReduceOpKernel( T* g_odata, long n,
		IndexUnaryOp<T> idxOp, BinaryOp<T> op, T start)
#else
template<typename T, uint blockSize, bool nIsPow2, int IopDim, int BopDim>
__global__ void indexReduceOpKernel( T* g_odata, long n,
		UnaryOpIndexF<T,IopDim> idxOp, MonoidF<T,BopDim> op, T start)
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
		myReduction = op(myReduction, idxOp(i));

		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction = op(myReduction, idxOp(i+ blockSize));

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
		volatile T* smem = sdata;

		if (blockSize >= 64) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 32]);
			__syncthreads();
		}

		if (blockSize >= 32) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 16]);
			__syncthreads();
		}

		if (blockSize >= 16) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 8]);
			__syncthreads();
		}

		if (blockSize >= 8) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 4]);
			__syncthreads();
		}

		if (blockSize >= 4) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 2]);
			__syncthreads();
		}

		if (blockSize >= 2) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 1]);
			__syncthreads();
		}

	}

	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class IndexUnaryOp,template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::indexReduceLauncher(T* d_odata, long n, IndexUnaryOp<T> idxOp, BinaryOp<T> op, T start, cudaStream_t stream)
#else
template<typename T> template<int IopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::indexReduceLauncher(T* d_odata, long n, UnaryOpIndexF<T,IopDim> idxOp, MonoidF<T,BopDim> op, T start, cudaStream_t stream)
#endif
{
	T gpu_result = 0;
	gpu_result = 0;
	int blocks,threads;
	getReductionExecContext(blocks,threads,n);
	// sum partial block sums on GPU
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize =
			(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	bool powOf2Q;

	powOf2Q = b_util::isPow2(n);
	if(checkDebug(debugRedux))flprintf("n %d dimGrid", n);
	if(checkDebug(debugRedux))b_util::prd3(dimGrid);
	if(checkDebug(debugRedux))prlocf("dimBlock ");
	if(checkDebug(debugRedux))b_util::prd3(dimBlock);
	if(checkDebug(debugRedux))flprintf("smemSize %d\n" , smemSize);
	if (powOf2Q) {
		switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
			indexReduceOpKernel<T, 1024, true, IndexUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 512:
			indexReduceOpKernel<T, 512, true, IndexUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 256:
			indexReduceOpKernel<T, 256, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 128:
			indexReduceOpKernel<T, 128, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 64:
			indexReduceOpKernel<T, 64, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 32:
			indexReduceOpKernel<T, 32, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 16:
			indexReduceOpKernel<T, 16, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 8:
			indexReduceOpKernel<T, 8, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 4:
			indexReduceOpKernel<T, 4, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 2:
			indexReduceOpKernel<T, 2, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 1:
			indexReduceOpKernel<T, 1, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
#else
			case 1024:
			indexReduceOpKernel<T, 1024, true, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 512:
			indexReduceOpKernel<T, 512, true, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 256:
			indexReduceOpKernel<T, 256, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 128:
			indexReduceOpKernel<T, 128, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 64:
			indexReduceOpKernel<T, 64, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 32:
			indexReduceOpKernel<T, 32, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 16:
			indexReduceOpKernel<T, 16, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 8:
			indexReduceOpKernel<T, 8, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 4:
			indexReduceOpKernel<T, 4, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 2:
			indexReduceOpKernel<T, 2, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 1:
			indexReduceOpKernel<T, 1, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
#endif
		}
	} else {
		switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
			indexReduceOpKernel<T, 1024, false, IndexUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 512:
			indexReduceOpKernel<T, 512, false, IndexUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 256:
			indexReduceOpKernel<T, 256, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 128:
			indexReduceOpKernel<T, 128, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 64:
			indexReduceOpKernel<T, 64, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 32:
			indexReduceOpKernel<T, 32, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 16:
			indexReduceOpKernel<T, 16, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 8:
			indexReduceOpKernel<T, 8, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 4:
			indexReduceOpKernel<T, 4, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 2:
			indexReduceOpKernel<T, 2, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 1:
			indexReduceOpKernel<T, 1, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
#else
			case 1024:
			indexReduceOpKernel<T, 1024, false, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 512:
			indexReduceOpKernel<T, 512, false, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 256:
			indexReduceOpKernel<T, 256, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 128:
			indexReduceOpKernel<T, 128, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 64:
			indexReduceOpKernel<T, 64, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 32:
			indexReduceOpKernel<T, 32, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 16:
			indexReduceOpKernel<T, 16, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 8:
			indexReduceOpKernel<T, 8, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 4:
			indexReduceOpKernel<T, 4, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 2:
			indexReduceOpKernel<T, 2, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
			case 1:
			indexReduceOpKernel<T, 1, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start); break;
#endif
			}
#ifndef __CUDA_ARCH__
		if(stream!=null)checkCudaError(cudaStreamSynchronize(stream)); else  checkCudaError(cudaDeviceSynchronize());
#else
		cherr(cudaDeviceSynchronize());
#endif
		n = DIV_UP(n, 2*threads);
	}

	// copy final sum from device to host
#ifndef __CUDA_ARCH__
	checkCudaError(
			cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
	if(checkDebug(debugCopyDh))outln("debugCopyDh " << "CuMatrix<T>::indexReduceLauncher");
	CuMatrix<T>::DHCopied++;
	CuMatrix<T>::MemDhCopied +=sizeof(T);
#else
	memcpy(&gpu_result, d_odata, sizeof(T));
#endif
	return gpu_result;
}

#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE float CuMatrix<float>::indexReduceLauncher<sequenceFiller, multBinaryOp>(float*,ulong, sequenceFiller<float>, multBinaryOp<float>, float, cudaStream_t);
template __host__ CUDART_DEVICE double CuMatrix<double>::indexReduceLauncher<sequenceFiller, multBinaryOp>(double*,ulong, sequenceFiller<double>, multBinaryOp<double>, double, cudaStream_t);
#else
#endif

template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::factorial(int val) {
	if(val < 3) {
		return val;
	}
	int threads;
	int blocks;
	getReductionExecContext(blocks,threads, val);
	CuMatrix<T> res(blocks, 1, false, true);
	DMatrix<T> d_Res;
	res.tile0(d_Res,false);
	sequenceFiller<T> seq = Functory<T, sequenceFiller>::pinch(1);
	T total = indexReduceLauncher( res.tiler.currBuffer(), val, seq, Functory<T,multBinaryOp>::pinch(), 1.0);
	return total;
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, typename IndexBoolUnaryOp, template <typename> class BinaryOp>
__global__ void indexedReduceOpKernel(T* g_odata, const T* g_idata, long n,
		IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start)
#else
template<typename T, uint blockSize, bool nIsPow2, int IopDim, int BopDim>
__global__ void indexedReduceOpKernel(T* g_odata, const T* g_idata, long n,
		UnaryOpIndexF<T,IopDim> idxOp, BinaryOpF<T,BopDim> op, T start)
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
		myReduction = idxOp(i) ? op(myReduction, g_idata[i]) : myReduction;

		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction =idxOp(i + blockSize) ? op(myReduction, g_idata[i + blockSize]) : myReduction;

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
		volatile T* smem = sdata;

		if (blockSize >= 64) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 32]);
			__syncthreads();
		}

		if (blockSize >= 32) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 16]);
			__syncthreads();
		}

		if (blockSize >= 16) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 8]);
			__syncthreads();
		}

		if (blockSize >= 8) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 4]);
			__syncthreads();
		}

		if (blockSize >= 4) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 2]);
			__syncthreads();
		}

		if (blockSize >= 2) {
			smem[tid] = myReduction = op(myReduction, sdata[tid + 1]);
			__syncthreads();
		}

	}

	if (tid == 0) {
		if(checkDebug(debugRedux))flprintf("indexedReduceOpKernel setting %p[%d] to %f\n", g_odata, blockIdx.x, sdata[0]);
		g_odata[blockIdx.x] = sdata[0];
	}
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<typename IndexBoolUnaryOp,template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduceLauncher(DMatrix<T> res, const T* d_idata, long n,	IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start, cudaStream_t stream)
#else
template<typename T> template<int IopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduceLauncher(DMatrix<T> res, const T* d_idata, long n,	UnaryOpIndexF<T,IopDim> idxOp, MonoidF<T,BopDim> op, T start, cudaStream_t stream)
#endif
{
	int blocks,threads;
	getReductionExecContext(blocks,threads,n);
	// sum partial block sums on GPU
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize =
			(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	bool powOf2Q;

	powOf2Q = b_util::isPow2(n);
	if(checkDebug(debugRedux))flprintf("n %d dimGrid", n);
	if(checkDebug(debugRedux))b_util::prd3(dimGrid);
	if(checkDebug(debugRedux))prlocf("dimBlock ");
	if(checkDebug(debugRedux))b_util::prd3(dimBlock);
	if(checkDebug(debugRedux))flprintf("smemSize %d\n" , smemSize);
	if (powOf2Q) {
		switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
			indexedReduceOpKernel<T, 1024, true, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 512:
			indexedReduceOpKernel<T, 512, true, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 256:
			indexedReduceOpKernel<T, 256, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 128:
			indexedReduceOpKernel<T, 128, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 64:
			indexedReduceOpKernel<T, 64, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 32:
			indexedReduceOpKernel<T, 32, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 16:
			indexedReduceOpKernel<T, 16, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 8:
			indexedReduceOpKernel<T, 8, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 4:
			indexedReduceOpKernel<T, 4, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 2:
			indexedReduceOpKernel<T, 2, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 1:
			indexedReduceOpKernel<T, 1, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
#else
			case 1024:
			indexedReduceOpKernel<T, 1024, true, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 512:
			indexedReduceOpKernel<T, 512, true, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 256:
			indexedReduceOpKernel<T, 256, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 128:
			indexedReduceOpKernel<T, 128, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 64:
			indexedReduceOpKernel<T, 64, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 32:
			indexedReduceOpKernel<T, 32, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 16:
			indexedReduceOpKernel<T, 16, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 8:
			indexedReduceOpKernel<T, 8, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 4:
			indexedReduceOpKernel<T, 4, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 2:
			indexedReduceOpKernel<T, 2, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 1:
			indexedReduceOpKernel<T, 1, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
#endif
		}
	} else {
		switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
			indexedReduceOpKernel<T, 1024, false, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 512:
			indexedReduceOpKernel<T, 512, false, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 256:
			indexedReduceOpKernel<T, 256, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 128:
			indexedReduceOpKernel<T, 128, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 64:
			indexedReduceOpKernel<T, 64, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 32:
			indexedReduceOpKernel<T, 32, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 16:
			indexedReduceOpKernel<T, 16, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 8:
			indexedReduceOpKernel<T, 8, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 4:
			indexedReduceOpKernel<T, 4, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 2:
			indexedReduceOpKernel<T, 2, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 1:
			indexedReduceOpKernel<T, 1, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
#else
			case 1024:
			indexedReduceOpKernel<T, 1024, false, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 512:
			indexedReduceOpKernel<T, 512, false, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 256:
			indexedReduceOpKernel<T, 256, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 128:
			indexedReduceOpKernel<T, 128, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 64:
			indexedReduceOpKernel<T, 64, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 32:
			indexedReduceOpKernel<T, 32, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 16:
			indexedReduceOpKernel<T, 16, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 8:
			indexedReduceOpKernel<T, 8, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 4:
			indexedReduceOpKernel<T, 4, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 2:
			indexedReduceOpKernel<T, 2, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
			case 1:
			indexedReduceOpKernel<T, 1, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start); break;
#endif
		}
	}

#ifndef __CUDA_ARCH__
		if(stream!=null)cudaStreamSynchronize(stream); else  cudaDeviceSynchronize();
#else
		cudaDeviceSynchronize();
#endif
	n = DIV_UP(n, 2*threads);
	if(checkDebug(debugRedux)){
		prlocf("now reducing indexReduceOpKernel results\n");
		util<T>::prdmln(res);
	}

	return CuMatrix<T>::reduce(res, op, start, stream  );
}

#ifdef  CuMatrix_Enable_KTS
template float CuMatrix<float>::indexedReduceLauncher<isColumnFiller<float>, plusBinaryOp>(DMatrix<float>,const float*,ulong, isColumnFiller<float>, plusBinaryOp<float>, float, cudaStream_t);
template double CuMatrix<double>::indexedReduceLauncher<isColumnFiller<float>, plusBinaryOp>(DMatrix<double>,const double*,ulong, isColumnFiller<float>, plusBinaryOp<double>, double, cudaStream_t);
template float CuMatrix<float>::indexedReduceLauncher<isRowFiller<float>, plusBinaryOp>(DMatrix<float>,const float*,ulong, isRowFiller<float>, plusBinaryOp<float>, float, cudaStream_t);
template double CuMatrix<double>::indexedReduceLauncher<isRowFiller<float>, plusBinaryOp>(DMatrix<double>,const double*,ulong, isRowFiller<float>, plusBinaryOp<double>, double, cudaStream_t);
#else
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, typename IndexBoolUnaryOp, typename UnaryOp,
		typename BinaryOp>
__global__ void indexedGloloReduceOpKernel(const T* g_idata, T* g_odata,
		long n, IndexBoolUnaryOp idxOp, UnaryOp gop, BinaryOp lop, T start)
#else
template<typename T, uint blockSize, bool nIsPow2, int IopDim, int UopDim, int BopDim>
__global__ void indexedGloloReduceOpKernel(const T* g_idata, T* g_odata,
		long n, UnaryOpIndexF<T,IopDim> idxOp, UnaryOpF<T,UopDim> gop, BinaryOpF<T,BopDim> lop, T start)
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
		myReduction = idxOp(i) ? lop(myReduction, gop(g_idata[i])) : myReduction;

		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction = idxOp(i + blockSize) ? lop(myReduction, gop(g_idata[i + blockSize])) : myReduction;

		i += gridSize;
	}
	// each thread puts its local sum into shared memory
	sdata[tid] = myReduction;
	__syncthreads();

	/* sdata-a */

	// do reduction in shared mem
	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] = myReduction = lop(myReduction, sdata[tid + 512]);
		}

		__syncthreads();
	}

	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] = myReduction = lop(myReduction, sdata[tid + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] = myReduction = lop(myReduction, sdata[tid + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] = myReduction = lop(myReduction, sdata[tid + 64]);
		}

		__syncthreads();
	}

	if (tid < 32) {
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn'float reorder stores to it and induce incorrect behavior.
		volatile T* smem = sdata;

		if (blockSize >= 64) {
			smem[tid] = myReduction = lop(myReduction, sdata[tid + 32]);
			__syncthreads();
		}

		if (blockSize >= 32) {
			smem[tid] = myReduction = lop(myReduction, sdata[tid + 16]);
			__syncthreads();
		}

		if (blockSize >= 16) {
			smem[tid] = myReduction = lop(myReduction, sdata[tid + 8]);
			__syncthreads();
		}

		if (blockSize >= 8) {
			smem[tid] = myReduction = lop(myReduction, sdata[tid + 4]);
			__syncthreads();
		}

		if (blockSize >= 4) {
			//smem[tid] = myReduction = op( myReduction, sdata[tid + 2] == 0 ? start : sdata[tid + 2]);
			smem[tid] = myReduction = lop(myReduction, sdata[tid + 2]);
			__syncthreads();
		}

		if (blockSize >= 2) {
			smem[tid] = myReduction = lop(myReduction, sdata[tid + 1]);
			__syncthreads();
		}

	}

	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	//
	// write result for this block to global mem
}

template <typename T> inline __device__ bool tabs(T val) {
	return val < 0 ? -val : val;
}

template<typename T> __global__ void onesColumnQRowMajorKernel(const T* g_idata, T* g_odata, int rows, int pitch, T epsilon) {
    ulong i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < rows && tabs(1 - g_idata[i * pitch]) < epsilon) {
    	g_odata[0] = 1;
    }
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<typename IndexBoolUnaryOp,template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduceL(const DMatrix<T>& d_M, IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start, cudaStream_t stream) const
#else
template<typename T> template<int IopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduceL(const DMatrix<T>& d_M, UnaryOpIndexF<T, IopDim> idxOp, MonoidF<T, BopDim> op, T start, cudaStream_t stream) const
#endif
{
	long nP = d_M.m * d_M.n;
	int threads;
	int blocks;
	getReductionExecContext(blocks,threads, nP);
	CuMatrix<T> res(blocks, 1, false, true);
	DMatrix<T> d_Res;
	res.tile0(d_Res,false);
	T total = indexedReduceLauncher( d_Res, d_M.elements, nP, idxOp, op, start, stream);
	return total;
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<typename IndexBoolUnaryOp, template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduce(IndexBoolUnaryOp idxOp,BinaryOp<T> op, T start, cudaStream_t stream) const
#else
template<typename T> template<int IopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduce(UnaryOpIndexF<T, IopDim> idxOp, MonoidF<T, BopDim> op, T start, cudaStream_t stream) const
#endif
{
	DMatrix<T> d_A;
	tiler.tile0(d_A,true);
	T res = indexedReduceL(d_A, idxOp, op, start, stream);
	return res;
}


#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::columnReduceIndexed(BinaryOp<T> op, int column, T start, cudaStream_t stream ) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE
T CuMatrix<T>::columnReduceIndexed(MonoidF<T,StateDim> op, int column, T start, cudaStream_t stream ) const
#endif
{
	if(!validColQ(column)) {
		setLastError(columnOutOfBoundsEx);
	}
/*
#ifdef  CuMatrix_Enable_KTS
	isColumnFiller<T> idxOp;
	idxOp.column() = column;
	idxOp.pitch() = p;
#else
*/
	isColumnFiller<T> idxOp = Functory<T,isColumnFiller>::pinch(p, column);
//#endif
	return indexedReduce(idxOp, op, 0,stream);
}

template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::columnSum(int column, cudaStream_t stream ) const {
	return columnReduceIndexed( Functory<T,plusBinaryOp>::pinch(), column, 0, stream);
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::rowReduce(BinaryOp<T> op, int row, T start , cudaStream_t stream) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE
T CuMatrix<T>::rowReduce(MonoidF<T,StateDim> op, int row, T start , cudaStream_t stream) const
#endif
{
	if(!validRowQ(row)) {
		setLastError(rowOutOfBoundsEx);
	}
	isRowFiller<T> idxOp = Functory<T,isRowFiller>::pinch(p, row);
	return indexedReduce(idxOp, op, 0, stream);
}

template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::rowSum(int row, cudaStream_t stream) const {
#ifdef  CuMatrix_Enable_KTS
	return rowReduce(plusBinaryOp<T>(), row, 0, stream);
#else
	return rowReduce(Functory<T,plusBinaryOp>::pinch(), row, 0, stream);
#endif
}

#include "CuMatrixInster.cu"
