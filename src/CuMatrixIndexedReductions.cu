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
		IndexUnaryOp<T> idxOp, BinaryOp<T> op, T start, ulong startIdx)
#else
template<typename T, uint blockSize, bool nIsPow2, int IopDim, int BopDim>
__global__ void indexReduceOpKernel( T* g_odata, long n,
		UnaryOpIndexF<T,IopDim> idxOp, MonoidF<T,BopDim> op, T start, ulong startIdx)
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
		myReduction = op(myReduction, idxOp(i + startIdx));

		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction = op(myReduction, idxOp(i + startIdx + blockSize));

		i += gridSize;
	}
	// each thread puts its local sum into shared memory
	sdata[tid] = myReduction;
	__syncthreads();

	shreduceShared(g_odata, sdata, myReduction, blockSize, tid, blockIdx.x, op);
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class IndexUnaryOp,template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::indexReduceLauncher(
		T* d_odata, long n, IndexUnaryOp<T> idxOp, BinaryOp<T> op, T start, ulong startIdx, cudaStream_t stream)
#else
template<typename T> template<int IopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::indexReduceLauncher(
		T* d_odata, long n, UnaryOpIndexF<T,IopDim> idxOp, MonoidF<T,BopDim> op, T start, ulong startIdx, cudaStream_t stream)
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
			indexReduceOpKernel<T, 1024, true, IndexUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 512:
			indexReduceOpKernel<T, 512, true, IndexUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 256:
			indexReduceOpKernel<T, 256, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 128:
			indexReduceOpKernel<T, 128, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 64:
			indexReduceOpKernel<T, 64, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 32:
			indexReduceOpKernel<T, 32, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 16:
			indexReduceOpKernel<T, 16, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 8:
			indexReduceOpKernel<T, 8, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 4:
			indexReduceOpKernel<T, 4, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 2:
			indexReduceOpKernel<T, 2, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 1:
			indexReduceOpKernel<T, 1, true, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
#else
			case 1024:
			indexReduceOpKernel<T, 1024, true, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 512:
			indexReduceOpKernel<T, 512, true, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 256:
			indexReduceOpKernel<T, 256, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 128:
			indexReduceOpKernel<T, 128, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 64:
			indexReduceOpKernel<T, 64, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 32:
			indexReduceOpKernel<T, 32, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 16:
			indexReduceOpKernel<T, 16, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 8:
			indexReduceOpKernel<T, 8, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 4:
			indexReduceOpKernel<T, 4, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 2:
			indexReduceOpKernel<T, 2, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 1:
			indexReduceOpKernel<T, 1, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
#endif
		}
	} else {
		switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
			indexReduceOpKernel<T, 1024, false, IndexUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 512:
			indexReduceOpKernel<T, 512, false, IndexUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 256:
			indexReduceOpKernel<T, 256, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 128:
			indexReduceOpKernel<T, 128, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 64:
			indexReduceOpKernel<T, 64, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 32:
			indexReduceOpKernel<T, 32, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 16:
			indexReduceOpKernel<T, 16, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 8:
			indexReduceOpKernel<T, 8, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 4:
			indexReduceOpKernel<T, 4, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 2:
			indexReduceOpKernel<T, 2, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 1:
			indexReduceOpKernel<T, 1, false, IndexUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
#else
			case 1024:
			indexReduceOpKernel<T, 1024, false, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 512:
			indexReduceOpKernel<T, 512, false, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 256:
			indexReduceOpKernel<T, 256, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 128:
			indexReduceOpKernel<T, 128, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 64:
			indexReduceOpKernel<T, 64, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 32:
			indexReduceOpKernel<T, 32, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 16:
			indexReduceOpKernel<T, 16, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 8:
			indexReduceOpKernel<T, 8, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 4:
			indexReduceOpKernel<T, 4, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 2:
			indexReduceOpKernel<T, 2, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
			case 1:
			indexReduceOpKernel<T, 1, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, n, idxOp, op, start, startIdx); break;
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
	CuTimer timer;
	timer.start();
	checkCudaError(
			cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
	//CuMatrix<T>::incDhCopy("CuMatrix<T>::indexReduceLauncher",sizeof(T),timer.stop());
	if(checkDebug(debugCopyDh))outln("debugCopyDh " << "CuMatrix<T>::indexReduceLauncher");
	CuMatrix<T>::DHCopied++;
	CuMatrix<T>::MemDhCopied +=sizeof(T);
#else
	memcpy(&gpu_result, d_odata, sizeof(T));
#endif
	return gpu_result;
}
#ifdef  CuMatrix_Enable_KTS
//template __host__ CUDART_DEVICE float CuMatrix<float>::indexReduceLauncher<sequenceFiller, multBinaryOp>(float*,ulong, sequenceFiller<float>, multBinaryOp<float>, float, cudaStream_t);
//template __host__ CUDART_DEVICE double CuMatrix<double>::indexReduceLauncher<sequenceFiller, multBinaryOp>(double*,ulong, sequenceFiller<double>, multBinaryOp<double>, double, cudaStream_t);
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
	sequenceFiller<T> seq = Functory<T, sequenceFiller>::pinch(1,1);
	T total = indexReduceLauncher( res.tiler.currBuffer(), val, seq, Functory<T,multBinaryOp>::pinch(), 1.0);
	return total;
}

/*
 * fixme
 * reduces via BinaryOp values that are selected from g_idata when the thread that indexes them also passes IndexBoolUnaryOp
 *
 */
#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, typename IndexBoolUnaryOp, template <typename> class BinaryOp>
__global__ void indexedReduceOpKernel(T* g_odata, const T* g_idata, long n,
		IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start, ulong idxStart)
#else
template<typename T, uint blockSize, bool nIsPow2, int IopDim, int BopDim>
__global__ void indexedReduceOpKernel(T* g_odata, const T* g_idata, long n,
		UnaryOpIndexF<T,IopDim> idxOp, BinaryOpF<T,BopDim> op, T start, ulong idxStart)
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
		myReduction = idxOp(i + idxStart) ? op(myReduction, g_idata[i]) : myReduction;
		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction =idxOp(i + blockSize + idxStart) ? op(myReduction, g_idata[i + blockSize]) : myReduction;
		i += gridSize;
	}
	// each thread puts its local sum into shared memory
	sdata[tid] = myReduction;
	__syncthreads();

	shreduceShared(g_odata, sdata, myReduction, blockSize, tid, blockIdx.x, op);

}

__inline__ ulong spitch(ulong idx, int pitch, int rows, int cols) {
	long rowi = idx / cols;
	int coli = idx - rowi * rows;
	return rowi * pitch + coli;
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, typename IndexBoolUnaryOp, template <typename> class BinaryOp>
__global__ void indexedReduceOpKernel(T* g_odata, const T* g_idata, size_t pitch, size_t cols, size_t rows,
		IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start, ulong idxStart)
#else
template<typename T, uint blockSize, bool nIsPow2, int IopDim, int BopDim>
__global__ void indexedReduceOpKernel(T* g_odata, const T* g_idata, size_t pitch, size_t cols, size_t rows,
		UnaryOpIndexF<T,IopDim> idxOp, BinaryOpF<T,BopDim> op, T start, ulong idxStart)
#endif
{
	T* sdata = SharedMemory<T>();

	uint tid = threadIdx.x;
	ulong _i = blockIdx.x * blockSize * 2 + threadIdx.x;

	ulong n = rows * pitch;
	long rowi = _i / cols;
	int coli = _i - rowi * rows;

	ulong i = rowi * pitch + coli;

	uint gridSize = blockSize * 2 * gridDim.x;

	T myReduction = start;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n) {
		myReduction = idxOp(i + idxStart) ? op(myReduction, g_idata[i]) : myReduction;

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction =idxOp(i + blockSize + idxStart) ? op(myReduction, g_idata[i + blockSize]) : myReduction;

		i += gridSize;
	}
	// each thread puts its local sum into shared memory
	sdata[tid] = myReduction;
	__syncthreads();

	shreduceShared(g_odata, sdata, myReduction, blockSize, tid, blockIdx.x, op);

}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<typename IndexBoolUnaryOp,template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduceLauncher(
		DMatrix<T> res, const T* d_idata, long n, IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start, ulong idxStart, cudaStream_t stream)
#else
template<typename T> template<int IopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduceLauncher(
		DMatrix<T> res, const T* d_idata, long n, UnaryOpIndexF<T,IopDim> idxOp, MonoidF<T,BopDim> op, T start, ulong idxStart, cudaStream_t stream)
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
			indexedReduceOpKernel<T, 1024, true, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 512:
			indexedReduceOpKernel<T, 512, true, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 256:
			indexedReduceOpKernel<T, 256, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 128:
			indexedReduceOpKernel<T, 128, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 64:
			indexedReduceOpKernel<T, 64, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 32:
			indexedReduceOpKernel<T, 32, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 16:
			indexedReduceOpKernel<T, 16, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 8:
			indexedReduceOpKernel<T, 8, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 4:
			indexedReduceOpKernel<T, 4, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 2:
			indexedReduceOpKernel<T, 2, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 1:
			indexedReduceOpKernel<T, 1, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
#else
			case 1024:
			indexedReduceOpKernel<T, 1024, true, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 512:
			indexedReduceOpKernel<T, 512, true, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 256:
			indexedReduceOpKernel<T, 256, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 128:
			indexedReduceOpKernel<T, 128, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 64:
			indexedReduceOpKernel<T, 64, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 32:
			indexedReduceOpKernel<T, 32, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 16:
			indexedReduceOpKernel<T, 16, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 8:
			indexedReduceOpKernel<T, 8, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 4:
			indexedReduceOpKernel<T, 4, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 2:
			indexedReduceOpKernel<T, 2, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 1:
			indexedReduceOpKernel<T, 1, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
#endif
		}
	} else {
		switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
			indexedReduceOpKernel<T, 1024, false, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 512:
			indexedReduceOpKernel<T, 512, false, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 256:
			indexedReduceOpKernel<T, 256, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 128:
			indexedReduceOpKernel<T, 128, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 64:
			indexedReduceOpKernel<T, 64, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 32:
			indexedReduceOpKernel<T, 32, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 16:
			indexedReduceOpKernel<T, 16, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 8:
			indexedReduceOpKernel<T, 8, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 4:
			indexedReduceOpKernel<T, 4, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 2:
			indexedReduceOpKernel<T, 2, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 1:
			indexedReduceOpKernel<T, 1, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
#else
			case 1024:
			indexedReduceOpKernel<T, 1024, false, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 512:
			indexedReduceOpKernel<T, 512, false, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 256:
			indexedReduceOpKernel<T, 256, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 128:
			indexedReduceOpKernel<T, 128, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 64:
			indexedReduceOpKernel<T, 64, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 32:
			indexedReduceOpKernel<T, 32, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 16:
			indexedReduceOpKernel<T, 16, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 8:
			indexedReduceOpKernel<T, 8, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 4:
			indexedReduceOpKernel<T, 4, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 2:
			indexedReduceOpKernel<T, 2, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
			case 1:
			indexedReduceOpKernel<T, 1, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, n, idxOp, op, start,idxStart); break;
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
template<typename T> template<typename IndexBoolUnaryOp,template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduceLauncherPitch(
		T* d_odata, size_t pitch, size_t cols, size_t rows, IndexUnaryOp<T> idxOp, BinaryOp<T> op, T start, ulong idxStart, cudaStream_t stream)
#else
template<typename T> template<int IopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduceLauncherPitch(
		DMatrix<T> res, const T* d_idata, size_t pitch, size_t cols, size_t rows, UnaryOpIndexF<T,IopDim> idxOp, MonoidF<T,BopDim> op, T start, ulong idxStart, cudaStream_t stream )
#endif
{
	int blocks,threads;
	long n = rows * cols;
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
	if(checkDebug(debugRedux))flprintf("nP %d dimGrid", n);
	if(checkDebug(debugRedux))b_util::prd3(dimGrid);
	if(checkDebug(debugRedux))prlocf("dimBlock ");
	if(checkDebug(debugRedux))b_util::prd3(dimBlock);
	if(checkDebug(debugRedux))flprintf("smemSize %d\n" , smemSize);
	if (powOf2Q) {
		switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
			indexedReduceOpKernel<T, 1024, true, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 512:
			indexedReduceOpKernel<T, 512, true, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 256:
			indexedReduceOpKernel<T, 256, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 128:
			indexedReduceOpKernel<T, 128, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 64:
			indexedReduceOpKernel<T, 64, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 32:
			indexedReduceOpKernel<T, 32, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 16:
			indexedReduceOpKernel<T, 16, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 8:
			indexedReduceOpKernel<T, 8, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 4:
			indexedReduceOpKernel<T, 4, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 2:
			indexedReduceOpKernel<T, 2, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 1:
			indexedReduceOpKernel<T, 1, true, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
#else
			case 1024:
			indexedReduceOpKernel<T, 1024, true, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 512:
			indexedReduceOpKernel<T, 512, true, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 256:
			indexedReduceOpKernel<T, 256, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 128:
			indexedReduceOpKernel<T, 128, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 64:
			indexedReduceOpKernel<T, 64, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 32:
			indexedReduceOpKernel<T, 32, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 16:
			indexedReduceOpKernel<T, 16, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 8:
			indexedReduceOpKernel<T, 8, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 4:
			indexedReduceOpKernel<T, 4, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 2:
			indexedReduceOpKernel<T, 2, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 1:
			indexedReduceOpKernel<T, 1, true, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
#endif
		}
	} else {
		switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
			indexedReduceOpKernel<T, 1024, false, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 512:
			indexedReduceOpKernel<T, 512, false, IndexBoolUnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 256:
			indexedReduceOpKernel<T, 256, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 128:
			indexedReduceOpKernel<T, 128, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 64:
			indexedReduceOpKernel<T, 64, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 32:
			indexedReduceOpKernel<T, 32, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 16:
			indexedReduceOpKernel<T, 16, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 8:
			indexedReduceOpKernel<T, 8, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 4:
			indexedReduceOpKernel<T, 4, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 2:
			indexedReduceOpKernel<T, 2, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 1:
			indexedReduceOpKernel<T, 1, false, IndexBoolUnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
#else
			case 1024:
			indexedReduceOpKernel<T, 1024, false, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 512:
			indexedReduceOpKernel<T, 512, false, IopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 256:
			indexedReduceOpKernel<T, 256, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 128:
			indexedReduceOpKernel<T, 128, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 64:
			indexedReduceOpKernel<T, 64, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 32:
			indexedReduceOpKernel<T, 32, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 16:
			indexedReduceOpKernel<T, 16, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 8:
			indexedReduceOpKernel<T, 8, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 4:
			indexedReduceOpKernel<T, 4, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 2:
			indexedReduceOpKernel<T, 2, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
			case 1:
			indexedReduceOpKernel<T, 1, false, IopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(res.elements, d_idata, pitch, cols, rows, idxOp, op, start,idxStart); break;
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
/*
template float CuMatrix<float>::indexedReduceLauncher<isColumnFiller<float>, plusBinaryOp>(DMatrix<float>,const float*,ulong, isColumnFiller<float>, plusBinaryOp<float>, float, cudaStream_t);
template double CuMatrix<double>::indexedReduceLauncher<isColumnFiller<float>, plusBinaryOp>(DMatrix<double>,const double*,ulong, isColumnFiller<float>, plusBinaryOp<double>, double, cudaStream_t);
template float CuMatrix<float>::indexedReduceLauncher<isRowFiller<float>, plusBinaryOp>(DMatrix<float>,const float*,ulong, isRowFiller<float>, plusBinaryOp<float>, float, cudaStream_t);
template double CuMatrix<double>::indexedReduceLauncher<isRowFiller<float>, plusBinaryOp>(DMatrix<double>,const double*,ulong, isRowFiller<float>, plusBinaryOp<double>, double, cudaStream_t);
*/
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

template <typename T> inline __device__ T tabs(T val) {
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
T CuMatrix<T>::indexedReduceL(const DMatrix<T>& d_M, IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start, ulong idxStart, cudaStream_t stream) const
#else
template<typename T> template<int IopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduceL(const DMatrix<T>& d_M, UnaryOpIndexF<T, IopDim> idxOp, MonoidF<T, BopDim> op, T start, ulong idxStart, cudaStream_t stream) const
#endif
{
	long nP = d_M.m * d_M.n;
	int threads;
	int blocks;
	getReductionExecContext(blocks,threads, nP);
	CuMatrix<T> res(blocks, 1, false, true);
	DMatrix<T> d_Res;
	res.tile0(d_Res,false);
	T total = -1;
	if(d_M.n == d_M.p) {
		total = indexedReduceLauncher( d_Res, d_M.elements, nP, idxOp, op, start, idxStart, stream);
	}else {
		assert(false);
		total = indexedReduceLauncherPitch(d_Res, d_M.elements, d_M.p, d_M.n, d_M.m, idxOp,  op, start, idxStart, stream );
	}
	return total;
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<typename IndexBoolUnaryOp, template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduce(IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start, cudaStream_t stream) const
#else
template<typename T> template<int IopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::indexedReduce(UnaryOpIndexF<T, IopDim> idxOp, MonoidF<T, BopDim> op, T start, cudaStream_t stream) const
#endif
{
	DMatrix<T> d_A;

	int tileCount = tiler.getTileCount();
	if(tileCount == 1) {
#ifndef __CUDA_ARCH__
		outln("tiler.getTileCount() " << tiler.getTileCount());
#endif
		tiler.tile0(d_A,true);
		T res = indexedReduceL(d_A, idxOp, op, start, 0, stream);
		return res;
	} else {
		int roff=0, coff=0, tileM = 0, tileN = 0, tileP = 0;
		T* resA;
		T res;
	#ifndef __CUDA_ARCH__
		cherr(cudaHostAlloc(&resA,tileCount*sizeof(T),0));
	#else
		resA = (T*) malloc(tileCount*sizeof(T));
	#endif

		int lastGpu = -1;
		int orgDevice = ExecCaps::currDev();
		int gpuCount = tiler.countGpus();
		if(checkDebug(debugRedux) ) flprintf("orgDev %d gpuCount %d\n",orgDevice, gpuCount);
		cudaStream_t* streams = null;

		if(gpuCount > 1) {
			assert(!stream);
			cudaStream_t* streams = (cudaStream_t* ) malloc(gpuCount * sizeof(cudaStream_t));
			for(int i =0 ; i < gpuCount; i++) {
				lastGpu = tiler.nextGpu(lastGpu);
				if(gpuCount> 1)
					ExecCaps_setDevice(lastGpu);
				//cherr(cudaSetDevice(lastGpu));
				cherr(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));
			}
		}
		lastGpu = -1;
		if(checkDebug(debugRedux) ) flprintf("m %d, n %d, p %d, _tileP %d, tiler.getTileCount() %d tileDir %s\n",m,n,p,_tileP, tileCount, b_util::tileDir(tdRows));
		for(int tile = 0; tile < tileCount; tile++) {
			if(gpuCount> 1)
				ExecCaps_setDevice(lastGpu);
			tiler.tile1D( d_A,roff,coff,tileM, tileN, tileP, tile, tdRows,lastMod == mod_host, lastGpu,gpuCount > 1 ? streams[tile] : stream);
			if(checkDebug(debugRedux) ) util<T>::prdm("d_A", d_A);
			//indexedReduceLauncher( d_Res, d_M.elements, nP, idxOp, op, start, stream);
			//T res = indexedReduceL(d_A, idxOp, op, start, stream);
			resA[tile] = indexedReduceL(d_A, idxOp, op, start, roff * _tileP + coff, gpuCount > 1 ? streams[tile] : stream);
					//reduce(d_A, op, start, gpuCount > 1 ? streams[tile] : stream);
		}
		if(gpuCount > 1) {
			for(int i =0 ; i < gpuCount; i++) {
				cherr(cudaStreamDestroy(streams[i]));
			}
			free(streams);
		}
		if(tileCount > 1) {
			if(checkDebug(debugRedux) ) flprintf("reduce tileCount %d\n",tileCount);
			// reduce across tile reductions
			T* dres = null;
	#ifndef __CUDA_ARCH__
			cherr(cudaMalloc(&dres, tileCount *sizeof(T)));
			cherr(cudaMemcpy(dres, resA, tileCount*sizeof(T), cudaMemcpyHostToDevice));
	#else
			dres = resA;
	#endif
			d_A.elements = dres;
			d_A.m = tileCount;
			d_A.n = 1; d_A.p = 1;
			res = reduce(d_A, op, start, stream);
	#ifndef __CUDA_ARCH__
			cudaFree(dres);
	#endif
		} else {
			if(checkDebug(debugRedux) ) flprintf("single tile reduction -> %f\n", (float) resA[0]);
			res =  resA[0];
		}
	#ifndef __CUDA_ARCH__
		if(checkDebug(debugDestr))flprintf("freeing host resA %p\n", resA);
		cherr(cudaFreeHost(resA));
	#else
		free(resA);
	#endif

		return res;

	}

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
	//#endif
	//if(checkDebug(debugRedux| debugRefcount))flprintf("colredxIndxed of column %d on mat %s\n", column, toss().c_str());

	isColumnFiller<T> idxOp = Functory<T,isColumnFiller>::pinch(_tileP, column);

	return indexedReduce(idxOp, op, 0,stream);
}

template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::columnSum(int column, cudaStream_t stream ) const {
	if(checkDebug(debugRedux| debugRefcount))flprintf("columnSum for col %d\n", column);
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
	isRowFiller<T> idxOp = Functory<T,isRowFiller>::pinch(_tileP, row);
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
