/*
 * CuMatrixGloloReductions.cu
 *
 *      Author: reid
 *
 *      allows different functors for the reduction from global mem
 *      and then of those results (stored in local mem)
 *
 */

#include "CuMatrix.h"
#include "Kernels.h"
#include "caps.h"

#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class UnaryOp,
template <typename> class BinaryOp>
__global__ void gloloReduceOpKernel(DMatrix<T> out, const DMatrix<T> src,
		long n, UnaryOp<T> gop, BinaryOp<T> lop, T start)
#else
template<typename T, uint blockSize, bool nIsPow2, int UopDim, int BopDim>
__global__ void gloloReduceOpKernel(DMatrix<T> out, const DMatrix<T> src,
		long n, UnaryOpF<T, UopDim> gop, MonoidF<T, BopDim> lop, T start)
#endif
		{
	T* sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	uint tid = threadIdx.x;
	ulong i = blockIdx.x * blockSize * 2 + threadIdx.x;
	uint gridSize = blockSize * 2 * gridDim.x;
	FirstThread {
		if (checkDebug(debugRedux))
			flprintf(
					"in gloloReduceOpKernel n %lu, src.elements %p, gridSize %u blockSize %d\n",
					n, src.elements, gridSize, blockSize);
	}

	T myReduction = start;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n) {
		myReduction = lop(myReduction, gop(get(src, i)));
		if (checkDebug(debugRedux) && i == n - 1)
			flprintf("gloloReduceOpKernel i == n - 1!, i %lu reading src(%p)\n",
					i, src.elements + i);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction = lop(myReduction, gop(get(src, i + blockSize)));
		if (checkDebug(debugRedux) && i + blockSize == n - 1) {
			flprintf(
					"gloloReduceOpKernel i + blockSize== n -1: i + blckSize %lu, reading src(%p)\n",
					i + blockSize, (src.elements + i + blockSize));
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
		out.elements[blockIdx.x] = sdata[0];
	//
	// write result for this block to global mem
}

//
// input pass from global to local via unaryop 'gop', subsequent passes over local are regular self-reductions using binaryop 'lop'
#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class UnaryOp, template <typename> class BinaryOp> __host__ CUDART_DEVICE void gloloReduceOpLauncher(T* result,
		DMatrix<T> buff, long n, const DMatrix<T> src, UnaryOp<T> gop, BinaryOp<T> lop,
		T start, cudaStream_t stream )
#else
template<typename T, int UopDim, int BopDim> __host__ CUDART_DEVICE void gloloReduceOpLauncher(
		T* result, DMatrix<T> buff, long n, const DMatrix<T> src,
		UnaryOpF<T, UopDim> gop, MonoidF<T, BopDim> lop, T start,
		cudaStream_t stream)
#endif
		{

	int blocks, threads;

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	bool powOf2Q;
	bool firstRedux = true;
	dim3 dimBlock;
	dim3 dimGrid;
	DMatrix<T> rSrc(src);

	while (n > 1) {
		powOf2Q = b_util::isPow2(n);
		getReductionExecContext(blocks, threads, n);
		int smemSize =
				(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
		dimBlock.x = threads;
		dimGrid.x = blocks;
		if (checkDebug(debugRedux)) {
			flprintf("n %d src.elements %p, lastelement @ %p\n", n,
					rSrc.elements, rSrc.elements + n - 1);
#ifndef __CUDA_ARCH__
			MemMgr<T>::checkValid(rSrc.elements, "rSrc.elements");
			MemMgr<T>::checkValid(rSrc.elements + n -1, "rSrc.elements + n -1");
#endif
			prlocf("dimGrid ");
			b_util::prd3(dimGrid);
			prlocf("dimBlock ");
			b_util::prd3(dimBlock);
			flprintf("threads %d\n", threads);
			flprintf("smemSize %d\n", smemSize);
			flprintf("smem Ts %d\n", smemSize / sizeof(T));
			flprintf("after block n %d src.elements %p, lastelement @ %p\n", n,
					rSrc.elements, rSrc.elements + n - 1);
		}
		if (firstRedux) {
			if (powOf2Q) {
				if (checkDebug(debugRedux))
					prlocf("powOf2Q\n");
				switch (threads) {
#ifdef  CuMatrix_Enable_KTS
				case 1024:
				gloloReduceOpKernel<T, 1024, true, UnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 512:
				gloloReduceOpKernel<T, 512, true, UnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 256:
				gloloReduceOpKernel<T, 256, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 128:
				gloloReduceOpKernel<T, 128, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 64:
				gloloReduceOpKernel<T, 64, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 32:
				gloloReduceOpKernel<T, 32, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 16:
				gloloReduceOpKernel<T, 16, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 8:
				gloloReduceOpKernel<T, 8, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 4:
				gloloReduceOpKernel<T, 4, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 2:
				gloloReduceOpKernel<T, 2, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 1:
				gloloReduceOpKernel<T, 1, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>( buff, rSrc, n,gop, lop, start); break;
#else
				case 1024:
					gloloReduceOpKernel<T, 1024, true, UopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 512:
					gloloReduceOpKernel<T, 512, true, UopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 256:
					gloloReduceOpKernel<T, 256, true,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 128:
					gloloReduceOpKernel<T, 128, true,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 64:
					gloloReduceOpKernel<T, 64, true,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 32:
					gloloReduceOpKernel<T, 32, true,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 16:
					gloloReduceOpKernel<T, 16, true,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 8:
					gloloReduceOpKernel<T, 8, true,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 4:
					gloloReduceOpKernel<T, 4, true,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 2:
					gloloReduceOpKernel<T, 2, true,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 1:
					gloloReduceOpKernel<T, 1, true,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>( buff, rSrc, n,gop, lop, start); break;
#endif
				}
			} else {
				if (checkDebug(debugRedux)) {
					prlocf("!powOf2Q\n");
					flprintf("buff %p rSrc.elements %p\n", buff.elements,
							rSrc.elements);
				}
				switch (threads) {
#ifdef  CuMatrix_Enable_KTS
				case 1024:
				gloloReduceOpKernel<T, 1024, false, UnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 512:
				gloloReduceOpKernel<T, 512, false, UnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 256:
				gloloReduceOpKernel<T, 256, false, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 128:
				gloloReduceOpKernel<T, 128, false, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 64:
				gloloReduceOpKernel<T, 64, false, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 32:
				gloloReduceOpKernel<T, 32, false, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 16:
				gloloReduceOpKernel<T, 16, false, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 8:
				gloloReduceOpKernel<T, 8, false, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 4:
				gloloReduceOpKernel<T, 4, false, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 2:
				gloloReduceOpKernel<T, 2, false, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 1:
				gloloReduceOpKernel<T, 1, false, UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>( buff, rSrc, n,gop, lop, start); break;
#else
				case 1024:
					gloloReduceOpKernel<T, 1024, false, UopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 512:
					gloloReduceOpKernel<T, 512, false, UopDim, BopDim> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 256:
					gloloReduceOpKernel<T, 256, false,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 128:
					gloloReduceOpKernel<T, 128, false,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 64:
					gloloReduceOpKernel<T, 64, false,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 32:
					gloloReduceOpKernel<T, 32, false,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 16:
					gloloReduceOpKernel<T, 16, false,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 8:
					gloloReduceOpKernel<T, 8, false,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 4:
					gloloReduceOpKernel<T, 4, false,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 2:
					gloloReduceOpKernel<T, 2, false,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>(buff, rSrc, n,gop, lop, start); break;
				case 1:
					gloloReduceOpKernel<T, 1, false,UopDim, BopDim><<< dimGrid, dimBlock, smemSize, stream >>>( buff, rSrc, n,gop, lop, start); break;
#endif
				}
			}

		}

#ifndef __CUDA_ARCH__
		if(checkDebug(debugRedux)) flprintf("stream %p\n", stream);
		cherr(cudaStreamSynchronize(stream));
#else
		cherr(cudaDeviceSynchronize());
#endif
		if (firstRedux) {
			if (checkDebug(debugRedux))
				prlocf(
						"gloloReduceLauncher after first redux, setting input to output\n");
			rSrc.elements = buff.elements;
			rSrc.m = buff.m;
			rSrc.p = buff.p;
			rSrc.n = buff.n;
			firstRedux = false;
		}
		n = blocks;
		if (n > 1) {
			reduceLauncher(result, buff, n, buff, lop, start, 0, stream);
			return;
		}
	}
	// copy final sum from device to host
#ifndef __CUDA_ARCH__
	CuTimer timer;
	timer.start();
	checkCudaError(
			cudaMemcpy(result, buff.elements, sizeof(T), cudaMemcpyDeviceToHost));
	//CuMatrix<T>::incDhCopy("gloloReduceOpLauncher",sizeof(T),timer.stop());
	if(checkDebug(debugCopyDh))outln("debugCopyDh " << "CuMatrix<T>::gloloReduceOpLauncher");
	CuMatrix<T>::DHCopied++;
	CuMatrix<T>::MemDhCopied +=sizeof(T);
#else
	memcpy(result, buff.elements, sizeof(T));
#endif
}

// for primeQ
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void gloloReduceOpLauncher<int, divisibleUnaryOp, maxBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, divisibleUnaryOp<int>, maxBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void gloloReduceOpLauncher<int, mutuallyDivisibleUnaryOp, maxBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, mutuallyDivisibleUnaryOp<int>, maxBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void gloloReduceOpLauncher<int, divisibleUnaryOp, minNotZeroBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, divisibleUnaryOp<int>, minNotZeroBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void gloloReduceOpLauncher<int, mutuallyDivisibleUnaryOp, minNotZeroBinaryOp>(int*, DMatrix<int>, long, DMatrix<int>, mutuallyDivisibleUnaryOp<int>, minNotZeroBinaryOp<int>, int, CUstream_st*);
#else
template __host__ CUDART_DEVICE void gloloReduceOpLauncher<int, 1, 1>(int*,
		DMatrix<int>, long, DMatrix<int>, UnaryOpF<int, 1>, MonoidF<int, 1>,
		int, CUstream_st*);
template __host__ CUDART_DEVICE void gloloReduceOpLauncher<int, 2, 1>(int*,
		DMatrix<int>, long, DMatrix<int>, UnaryOpF<int, 2>, MonoidF<int, 1>,
		int, CUstream_st*);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class UnaryOp, template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::gloloReduceL(const DMatrix<T>& d_M, UnaryOp<T> gop, BinaryOp<T> lop,
		T start, cudaStream_t stream ) const
#else
template<typename T> template<int IopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::gloloReduceL(const DMatrix<T>& d_M, UnaryOpF<T, IopDim> gop,
		MonoidF<T, BopDim> lop, T start, cudaStream_t stream) const
#endif
		{
	long nP = d_M.m * d_M.n;
	int threads;
	int blocks;
	::getReductionExecContext(blocks, threads, nP);
	CuMatrix<T> res(1, blocks, true, true);
	DMatrix<T> d_Res;
	res.tile0(d_Res, false);
	T total;
	gloloReduceOpLauncher(&total, d_Res, nP, d_M, gop, lop, start, stream);
	return total;
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class UnaryOp, template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::gloloReduce(UnaryOp<T> gop, BinaryOp<T> lop, T start, cudaStream_t stream) const
#else
template<typename T> template<int UopDim, int BopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::gloloReduce(UnaryOpF<T, UopDim> gop, MonoidF<T, BopDim> lop,
		T start, cudaStream_t stream) const
#endif
		{
	DMatrix<T> d_A;

	cherr(cudaPeekAtLastError());
	T* resA;
	T res;
	int tileCount = tiler.getTileCount();

#ifndef __CUDA_ARCH__
	cherr(cudaHostAlloc(&resA,tileCount*sizeof(T),0));
	int roff,coff, tileM = 0, tileN=0, tileP=0;
	int lastGpu = -1;
	int gpuCount = tiler.countGpus();
	int orgDevice = ExecCaps::currDev();
	cudaStream_t* streams = null;
	if(gpuCount > 1) {
		assert(!stream);
		cudaStream_t* streams = (cudaStream_t* ) malloc(gpuCount * sizeof(cudaStream_t));
		for(int i =0; i < gpuCount; i++) {
			lastGpu = tiler.nextGpu(lastGpu);
			if(gpuCount > 1)
			ExecCaps_setDevice(lastGpu);
			cherr(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));
		}
	}
	lastGpu = -1;
	for(int tile = 0; tile < tileCount; tile++) {
		lastGpu = tiler.nextGpu(lastGpu);
		if(gpuCount > 1)
		ExecCaps_setDevice(lastGpu);
		tiler.tile1D( d_A,roff,coff,tileM, tileN, tileP, tile, tdRows, tileCount > 1 || lastMod == mod_host, lastGpu,gpuCount > 1 ? streams[tile] : stream);
		resA[tile] = gloloReduceL(d_A, gop, lop, start,gpuCount > 1 ? streams[tile] : stream);
		if(checkDebug(debugRedux) ) flprintf("gloloreducing tile %d\n",tile );
	}

	if(gpuCount > 1) {
		for(int i =0; i < gpuCount; i++) {
			cherr(cudaStreamDestroy(streams[i]));
		}
		free(streams);
	}

	if(tileCount > 1) {
		if(checkDebug(debugRedux) ) flprintf("reduce across %d tile gloloreductions %d\n",tileCount);
		// reduce across tile reductions
		T* dres = null;
		cherr(cudaMalloc(&dres, tileCount * sizeof(T)));
		cherr(cudaMemcpy(dres, resA, tileCount * sizeof(T), cudaMemcpyHostToDevice));
		d_A.elements = dres;
		d_A.m = tileCount;
		d_A.n = 1; d_A.p = 1;
		res = reduce(d_A, lop, start, stream);
		cudaFree(dres);
	} else {
		if(checkDebug(debugRedux) ) flprintf("single tile gloloreduction -> %f\n", (float) resA[0]);
		res = resA[0];
	}
	if(checkDebug(debugDestr))flprintf("freeing host %p\n", resA);
	cherr(cudaFreeHost(resA));
#else
	res = gloloReduceL(d_A, gop, lop, start, stream);
#endif

	return res;
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<oneOrZeroUnaryOp, andBinaryOp>(oneOrZeroUnaryOp<float>, andBinaryOp<float>, float,cudaStream_t) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<oneOrZeroUnaryOp, andBinaryOp>(oneOrZeroUnaryOp<double>, andBinaryOp<double>, double,cudaStream_t) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<oneOrZeroUnaryOp, andBinaryOp>(oneOrZeroUnaryOp<ulong>, andBinaryOp<ulong>, ulong,cudaStream_t) const;
template __host__ CUDART_DEVICE long CuMatrix<long>::gloloReduce<oneOrZeroUnaryOp, andBinaryOp>(oneOrZeroUnaryOp<long>, andBinaryOp<long>, long, CUstream_st*) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<almostEqUnaryOp, andBinaryOp>(almostEqUnaryOp<float>, andBinaryOp<float>, float,cudaStream_t) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<almostEqUnaryOp, andBinaryOp>(almostEqUnaryOp<double>, andBinaryOp<double>, double,cudaStream_t) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<almostEqUnaryOp, andBinaryOp>(almostEqUnaryOp<ulong>, andBinaryOp<ulong>, ulong,cudaStream_t) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<sqrUnaryOp, plusBinaryOp>(sqrUnaryOp<float>, plusBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<sqrUnaryOp, plusBinaryOp>(sqrUnaryOp<double>, plusBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<sqrUnaryOp, plusBinaryOp>(sqrUnaryOp<ulong>, plusBinaryOp<ulong>, ulong, CUstream_st*) const;
template __host__ CUDART_DEVICE long CuMatrix<long>::gloloReduce<sqrUnaryOp, plusBinaryOp>(sqrUnaryOp<long>, plusBinaryOp<long>, long, CUstream_st*) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<almostEqUnaryOp, plusBinaryOp>(almostEqUnaryOp<float>, plusBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<almostEqUnaryOp, plusBinaryOp>(almostEqUnaryOp<double>, plusBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<almostEqUnaryOp, plusBinaryOp>(almostEqUnaryOp<ulong>, plusBinaryOp<ulong>, ulong, CUstream_st*) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<neqUnaryOp, plusBinaryOp>(neqUnaryOp<float>, plusBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<neqUnaryOp, plusBinaryOp>(neqUnaryOp<double>, plusBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<neqUnaryOp, plusBinaryOp>(neqUnaryOp<ulong>, plusBinaryOp<ulong>, ulong, CUstream_st*) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<notAlmostEqUnaryOp, plusBinaryOp>(notAlmostEqUnaryOp<float>, plusBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<notAlmostEqUnaryOp, plusBinaryOp>(notAlmostEqUnaryOp<double>, plusBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<notAlmostEqUnaryOp, plusBinaryOp>(notAlmostEqUnaryOp<ulong>, plusBinaryOp<ulong>, ulong, CUstream_st*) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<ltUnaryOp, plusBinaryOp>(ltUnaryOp<float>, plusBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<ltUnaryOp, plusBinaryOp>(ltUnaryOp<double>, plusBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<ltUnaryOp, plusBinaryOp>(ltUnaryOp<ulong>, plusBinaryOp<ulong>, ulong, CUstream_st*) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<ltUnaryOp, andBinaryOp>(ltUnaryOp<float>, andBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<ltUnaryOp, orBinaryOp>(ltUnaryOp<float>, orBinaryOp<float>, float, CUstream_st*) const;

template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<ltUnaryOp, andBinaryOp>(ltUnaryOp<double>, andBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<ltUnaryOp, orBinaryOp>(ltUnaryOp<double>, orBinaryOp<double>, double, CUstream_st*) const;

template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<ltUnaryOp, andBinaryOp>(ltUnaryOp<ulong>, andBinaryOp<ulong>, ulong, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<ltUnaryOp, orBinaryOp>(ltUnaryOp<ulong>, orBinaryOp<ulong>, ulong, CUstream_st*) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<gtUnaryOp, andBinaryOp>(gtUnaryOp<float>, andBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<gtUnaryOp, andBinaryOp>(gtUnaryOp<double>, andBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<gtUnaryOp, andBinaryOp>(gtUnaryOp<ulong>, andBinaryOp<ulong>, ulong, CUstream_st*) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<gtUnaryOp, orBinaryOp>(gtUnaryOp<float>, orBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<gtUnaryOp, orBinaryOp>(gtUnaryOp<double>, orBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<gtUnaryOp, orBinaryOp>(gtUnaryOp<ulong>, orBinaryOp<ulong>, ulong, CUstream_st*) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<almostEqUnaryOp, orBinaryOp>(almostEqUnaryOp<float>, orBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<almostEqUnaryOp, orBinaryOp>(almostEqUnaryOp<double>, orBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<almostEqUnaryOp, orBinaryOp>(almostEqUnaryOp<ulong>, orBinaryOp<ulong>, ulong, CUstream_st*) const;

template __host__ CUDART_DEVICE int CuMatrix<int>::gloloReduce<almostEqUnaryOp, andBinaryOp>(almostEqUnaryOp<int>, andBinaryOp<int>, int, CUstream_st*) const;
template __host__ CUDART_DEVICE int CuMatrix<int>::gloloReduce<oneOrZeroUnaryOp, andBinaryOp>(oneOrZeroUnaryOp<int>, andBinaryOp<int>, int, CUstream_st*) const;
template __host__ CUDART_DEVICE uint CuMatrix<uint>::gloloReduce<almostEqUnaryOp, andBinaryOp>(almostEqUnaryOp<uint>, andBinaryOp<uint>, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE uint CuMatrix<uint>::gloloReduce<oneOrZeroUnaryOp, andBinaryOp>(oneOrZeroUnaryOp<uint>, andBinaryOp<uint>, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE long CuMatrix<long>::gloloReduce<almostEqUnaryOp, andBinaryOp>(almostEqUnaryOp<long>, andBinaryOp<long>, long, CUstream_st*) const;

template __host__ CUDART_DEVICE int CuMatrix<int>::gloloReduce<sqrUnaryOp, plusBinaryOp>(sqrUnaryOp<int>, plusBinaryOp<int>, int, CUstream_st*) const;
template __host__ CUDART_DEVICE uint CuMatrix<uint>::gloloReduce<sqrUnaryOp, plusBinaryOp>(sqrUnaryOp<uint>, plusBinaryOp<uint>, uint, CUstream_st*) const;


template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<idx1DblockAlmostEqUnaryOp, plusBinaryOp>(idx1DblockAlmostEqUnaryOp<ulong>, plusBinaryOp<ulong>, ulong, CUstream_st*) const;
template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<idx1DblockAlmostEqUnaryOp, plusBinaryOp>(idx1DblockAlmostEqUnaryOp<float>, plusBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<idx1DblockAlmostEqUnaryOp, plusBinaryOp>(idx1DblockAlmostEqUnaryOp<double>, plusBinaryOp<double>, double, CUstream_st*) const;


#else
template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<0, 1>(
		UnaryOpF<float, 0>, MonoidF<float, 1>, float, cudaStream_t) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<0, 1>(
		UnaryOpF<double, 0>, MonoidF<double, 1>, double, cudaStream_t) const;
template __host__ CUDART_DEVICE int CuMatrix<int>::gloloReduce<0, 1>(
		UnaryOpF<int, 0>, MonoidF<int, 1>, int, cudaStream_t) const;
template __host__ CUDART_DEVICE uint CuMatrix<uint>::gloloReduce<0, 1>(
		UnaryOpF<uint, 0>, MonoidF<uint, 1>, uint, cudaStream_t) const;
template __host__ CUDART_DEVICE long CuMatrix<long>::gloloReduce<0, 1>(
		UnaryOpF<long, 0>, MonoidF<long, 1>, long, cudaStream_t) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<0, 1>(
		UnaryOpF<ulong, 0>, MonoidF<ulong, 1>, ulong, cudaStream_t) const;
template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<1, 1>(
		UnaryOpF<float, 1>, MonoidF<float, 1>, float, cudaStream_t) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<1, 1>(
		UnaryOpF<double, 1>, MonoidF<double, 1>, double, cudaStream_t) const;
template __host__ CUDART_DEVICE long CuMatrix<long>::gloloReduce<1, 1>(
		UnaryOpF<long, 1>, MonoidF<long, 1>, long, cudaStream_t) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<1, 1>(
		UnaryOpF<ulong, 1>, MonoidF<ulong, 1>, ulong, cudaStream_t) const;
template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<2, 1>(
		UnaryOpF<float, 2>, MonoidF<float, 1>, float, cudaStream_t) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<2, 1>(
		UnaryOpF<double, 2>, MonoidF<double, 1>, double, cudaStream_t) const;
template __host__ CUDART_DEVICE long CuMatrix<long>::gloloReduce<2, 1>(
		UnaryOpF<long, 2>, MonoidF<long, 1>, long, cudaStream_t) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<2, 1>(
		UnaryOpF<ulong, 2>, MonoidF<ulong, 1>, ulong, cudaStream_t) const;
template __host__ CUDART_DEVICE int CuMatrix<int>::gloloReduce<2, 1>(
		UnaryOpF<int, 2>, MonoidF<int, 1>, int, cudaStream_t) const;
template __host__ CUDART_DEVICE uint CuMatrix<uint>::gloloReduce<2, 1>(
		UnaryOpF<uint, 2>, MonoidF<uint, 1>, uint, CUstream_st*) const;

template __host__ CUDART_DEVICE float CuMatrix<float>::gloloReduce<3, 1>(
		UnaryOpF<float, 3>, MonoidF<float, 1>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::gloloReduce<3, 1>(
		UnaryOpF<double, 3>, MonoidF<double, 1>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE int CuMatrix<int>::gloloReduce<3, 1>(
		UnaryOpF<int, 3>, MonoidF<int, 1>, int, CUstream_st*) const;
template __host__ CUDART_DEVICE long CuMatrix<long>::gloloReduce<3, 1>(
		UnaryOpF<long, 3>, MonoidF<long, 1>, long, CUstream_st*) const;
template __host__ CUDART_DEVICE uint CuMatrix<uint>::gloloReduce<3, 1>(
		UnaryOpF<uint, 3>, MonoidF<uint, 1>, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::gloloReduce<3, 1>(
		UnaryOpF<ulong, 3>, MonoidF<ulong,1>, ulong, CUstream_st*) const;
#endif

#include "CuMatrixInster.cu"
