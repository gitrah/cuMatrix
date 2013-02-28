/*
 * CuMatrixIndexedReductions.cu
 *
 *  Created on: Dec 18, 2012
 *      Author: reid
 */
#include "CuMatrix.h"
#include "functors.h"


template<typename T, uint blockSize, bool nIsPow2, typename IndexBoolUnaryOp, typename BinaryOp>
__global__ void indexedReduceOpKernel(const T* g_idata, T* g_odata, ulong n,
		IndexBoolUnaryOp idxOp, BinaryOp op, T start) {
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

	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

template<typename T, uint blockSize, bool nIsPow2, typename IndexBoolUnaryOp, typename UnaryOp,
		typename BinaryOp>
__global__ void indexedGloloReduceOpKernel(const T* g_idata, T* g_odata,
		ulong n, IndexBoolUnaryOp idxOp, UnaryOp gop, BinaryOp lop, T start) {
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

template<typename T> __global__ void onesColumnQRowMajorKernel(const T* g_idata, T* g_odata, uint rows, uint pitch, T epsilon) {
    ulong i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < rows && tabs(1 - g_idata[i * pitch]) < epsilon) {
    	g_odata[0] = 1;
    }
}

template<typename T> template<typename IndexBoolUnaryOp,typename BinaryOp> T CuMatrix<T>::indexedReduceLauncher(
		T* d_odata, const T* d_idata, ulong n,	IndexBoolUnaryOp idxOp, BinaryOp op, T start) {
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
	bool powOf2Q = b_util::isPow2(n);
	if(debugExec)outln("n " << n);
	if(debugExec)outln("dimGrid " << b_util::pd3(dimGrid));
	if(debugExec)outln("dimBlock " << b_util::pd3(dimBlock));
	if(debugExec)outln("smemSize " << smemSize);
	if(debugExec)outln("smem Ts " << smemSize/sizeof(T));
	if (powOf2Q) {
		switch (threads) {
			case 1024:
			indexedReduceOpKernel<T, 1024, true, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 512:
			indexedReduceOpKernel<T, 512, true, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 256:
			indexedReduceOpKernel<T, 256, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 128:
			indexedReduceOpKernel<T, 128, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 64:
			indexedReduceOpKernel<T, 64, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 32:
			indexedReduceOpKernel<T, 32, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 16:
			indexedReduceOpKernel<T, 16, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 8:
			indexedReduceOpKernel<T, 8, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 4:
			indexedReduceOpKernel<T, 4, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 2:
			indexedReduceOpKernel<T, 2, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 1:
			indexedReduceOpKernel<T, 1, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
		}
	} else {
		switch (threads) {
			case 1024:
			indexedReduceOpKernel<T, 1024, false, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 512:
			indexedReduceOpKernel<T, 512, false, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 256:
			indexedReduceOpKernel<T, 256, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 128:
			indexedReduceOpKernel<T, 128, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 64:
			indexedReduceOpKernel<T, 64, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 32:
			indexedReduceOpKernel<T, 32, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 16:
			indexedReduceOpKernel<T, 16, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 8:
			indexedReduceOpKernel<T, 8, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 4:
			indexedReduceOpKernel<T, 4, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 2:
			indexedReduceOpKernel<T, 2, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
			case 1:
			indexedReduceOpKernel<T, 1, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, op, start); break;
		}

	}
	cudaError_t cuda_error = syncHappy ? cudaDeviceSynchronize() : cudaSuccess;
	checkCudaError(cuda_error);

	if(cuda_error == cudaSuccess)
			partialReduceLauncher(d_odata, blocks, op,start);

	// copy final sum from device to host
	checkCudaError(
			cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));

	return gpu_result;
}

template float CuMatrix<float>::indexedReduceLauncher<isColumnUnaryOp, plusBinaryOp<float> >(float*,const float*,ulong, isColumnUnaryOp, plusBinaryOp<float>, float);
template double CuMatrix<double>::indexedReduceLauncher<isColumnUnaryOp, plusBinaryOp<double> >(double*,const double*,ulong, isColumnUnaryOp, plusBinaryOp<double>, double);
template float CuMatrix<float>::indexedReduceLauncher<isRowUnaryOp, plusBinaryOp<float> >(float*,const float*,ulong, isRowUnaryOp, plusBinaryOp<float>, float);
template double CuMatrix<double>::indexedReduceLauncher<isRowUnaryOp, plusBinaryOp<double> >(double*,const double*,ulong, isRowUnaryOp, plusBinaryOp<double>, double);

template<typename T> template<typename IndexBoolUnaryOp, typename UnaryOp, typename BinaryOp> T CuMatrix<T>::indexedGloloReduceOpLauncher(
		T* d_odata, const T* d_idata, ulong n, IndexBoolUnaryOp idxOp, UnaryOp gop, BinaryOp lop,
		T start) {
	T gpu_result = 0;
	gpu_result = 0;

	// sum partial block sums on GPU
	int blocks,threads;
	getReductionExecContext(blocks,threads,n);
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	bool powOf2Q = b_util::isPow2(n);
	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize =
			(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	if(debugExec)outln("n " << n);
	if(debugExec)outln("dimGrid " << b_util::pd3(dimGrid));
	if(debugExec)outln("dimBlock " << b_util::pd3(dimBlock));
	if(debugExec)outln("smemSize " << smemSize);
	if(debugExec)outln("smem Ts " << smemSize/sizeof(T));
	if (powOf2Q) {
		switch (threads) {
			case 1024:
			indexedGloloReduceOpKernel<T, 1024, true, IndexBoolUnaryOp, UnaryOp,
					BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 512:
			indexedGloloReduceOpKernel<T, 512, true, IndexBoolUnaryOp, UnaryOp,
					BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 256:
			indexedGloloReduceOpKernel<T, 256, true, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 128:
			indexedGloloReduceOpKernel<T, 128, true, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 64:
			indexedGloloReduceOpKernel<T, 64, true, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 32:
			indexedGloloReduceOpKernel<T, 32, true, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 16:
			indexedGloloReduceOpKernel<T, 16, true, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 8:
			indexedGloloReduceOpKernel<T, 8, true, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 4:
			indexedGloloReduceOpKernel<T, 4, true, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 2:
			indexedGloloReduceOpKernel<T, 2, true, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 1:
			indexedGloloReduceOpKernel<T, 1, true, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, n, idxOp, gop, lop, start); break;
		}
	} else {
		switch (threads) {
			case 1024:
			indexedGloloReduceOpKernel<T, 1024, false, IndexBoolUnaryOp, UnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 512:
			indexedGloloReduceOpKernel<T, 512, false, IndexBoolUnaryOp, UnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 256:
			indexedGloloReduceOpKernel<T, 256, false, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 128:
			indexedGloloReduceOpKernel<T, 128, false, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 64:
			indexedGloloReduceOpKernel<T, 64, false, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 32:
			indexedGloloReduceOpKernel<T, 32, false, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 16:
			indexedGloloReduceOpKernel<T, 16, false, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 8:
			indexedGloloReduceOpKernel<T, 8, false, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 4:
			indexedGloloReduceOpKernel<T, 4, false, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 2:
			indexedGloloReduceOpKernel<T, 2, false, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n, idxOp, gop, lop, start); break;
			case 1:
			indexedGloloReduceOpKernel<T, 1, false, IndexBoolUnaryOp, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, n, idxOp, gop, lop, start); break;
		}

	}
	cudaError_t cuda_error = syncHappy ? cudaDeviceSynchronize() : cudaSuccess;
	checkCudaError(cuda_error);

	if(cuda_error == cudaSuccess)
			partialReduceLauncher(d_odata, blocks, lop,start);

	checkCudaError(
			cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));

	return gpu_result;
}

template float CuMatrix<float>::indexedGloloReduceOpLauncher<isColumnUnaryOp, almostEqualsBoolUnaryOp<float>, andBinaryOp<float> >(float*,const float*,ulong, isColumnUnaryOp, almostEqualsBoolUnaryOp<float>, andBinaryOp<float>, float);
template double CuMatrix<double>::indexedGloloReduceOpLauncher<isColumnUnaryOp, almostEqualsBoolUnaryOp<double>, andBinaryOp<double> >(double*,const double*,ulong, isColumnUnaryOp, almostEqualsBoolUnaryOp<double>, andBinaryOp<double>, double);

#include "CuMatrixInster.cu"
