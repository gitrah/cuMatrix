#include "CuMatrix.h"
#include "functors.h"


template<typename T, uint blockSize, bool nIsPow2, typename UnaryOp,
		typename BinaryOp>
__global__ void gloloReduceOpKernel(const T* g_idata, T* g_odata,
		ulong n, UnaryOp gop, BinaryOp lop, T start) {
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
		myReduction = lop(myReduction, gop(g_idata[i]));

		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction = lop(myReduction, gop(g_idata[i + blockSize]));

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


template<typename T, uint blockSize, bool nIsPow2, typename MatBinaryOp,
		typename BinaryOp>
__global__ void matrixReduceOpKernel(const T* g_idata1, const T* g_idata2,
		T* g_odata, ulong n, MatBinaryOp mop, BinaryOp op, T start) {
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
		myReduction = op(myReduction, mop(g_idata1[i], g_idata2[i]));

		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction = op(myReduction,
					mop(g_idata1[i + blockSize], g_idata2[i + blockSize]));

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
			//smem[tid] = myReduction = op( myReduction, sdata[tid + 2] == 0 ? start : sdata[tid + 2]);
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
	//
	// write result for this block to global mem
}

//
// input pass works across two matrices via 'MatBinaryOp', subsequent passes are regular self-reductions
template<typename T> template<typename MatBinaryOp, typename BinaryOp> T CuMatrix<
		T>::matrixReduceOpLauncher(T* d_odata, const T* d_idata1, const T* d_idata2, ulong n, MatBinaryOp mop, BinaryOp op, T start) {
	T gpu_result = 0;
	gpu_result = 0;

	// sum partial block sums on GPU
	int blocks,threads;
	getReductionExecContext(blocks,threads, n);
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	bool powOf2Q = b_util::isPow2(n);
	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize =
			(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	if(debugExec)outln("matredop n " << n);
	if(debugExec)outln("matredop dimGrid " << b_util::pd3(dimGrid));
	if(debugExec)outln("matredop dimBlock " << b_util::pd3(dimBlock));
	if(debugExec)outln("matredop smemSize " << smemSize);
	if(debugExec)outln("matredop smem Ts " << smemSize/sizeof(T));
	if (powOf2Q) {
		switch (threads) {
			case 1024:
			matrixReduceOpKernel<T, 1024, true, MatBinaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			case 512:
			matrixReduceOpKernel<T, 512, true, MatBinaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			case 256:
			matrixReduceOpKernel<T, 256, true,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			case 128:
			matrixReduceOpKernel<T, 128, true,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			case 64:
			matrixReduceOpKernel<T, 64, true,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			case 32:
			matrixReduceOpKernel<T, 32, true,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			case 16:
			matrixReduceOpKernel<T, 16, true,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			case 8:
			matrixReduceOpKernel<T, 8, true,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			case 4:
			matrixReduceOpKernel<T, 4, true,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			case 2:
			matrixReduceOpKernel<T, 2, true,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			case 1:
			matrixReduceOpKernel<T, 1, true,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>( d_idata1, d_idata2, d_odata, n,mop,op, start); break;
		}
	}else {
			switch (threads) {
				case 1024:
				matrixReduceOpKernel<T, 1024, false, MatBinaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
				case 512:
				matrixReduceOpKernel<T, 512, false, MatBinaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
				case 256:
				matrixReduceOpKernel<T, 256, false,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
				case 128:
				matrixReduceOpKernel<T, 128, false,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
				case 64:
				matrixReduceOpKernel<T, 64, false,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
				case 32:
				matrixReduceOpKernel<T, 32, false,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
				case 16:
				matrixReduceOpKernel<T, 16, false,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
				case 8:
				matrixReduceOpKernel<T, 8, false,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
				case 4:
				matrixReduceOpKernel<T, 4, false,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
				case 2:
				matrixReduceOpKernel<T, 2, false,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2, d_odata, n,mop,op, start); break;
				case 1:
				matrixReduceOpKernel<T, 1, false,MatBinaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>( d_idata1, d_idata2, d_odata, n,mop,op, start); break;
			}

	}
	cudaError_t cuda_error = syncHappy ? cudaDeviceSynchronize() : cudaSuccess;
	checkCudaError(cuda_error);
	T* res = null;
	if(blocks == 2) {
		cudaHostAlloc((void**) & res, 2 * sizeof(T),0);
		cudaMemcpy(res,d_odata,2*sizeof(T), cudaMemcpyDeviceToHost);
		cout << "res[0] " << res[0] << ", [1] " << res[1] << endl;
		cudaFreeHost(res);
	}
	if(cuda_error == cudaSuccess)
		partialReduceLauncher( d_odata, blocks, op,start);

	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
	// copy final sum from device to host
	cuda_error = cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost);
	if(cuda_error != cudaSuccess) {
		outln("d_odata  " << d_odata);
		checkCudaError(cuda_error);
	}
	//outln("gpu_result " << gpu_result);
	return gpu_result;
}
template float CuMatrix<float>::matrixReduceOpLauncher<multBinaryOp<float>, plusBinaryOp<float> >(float*, const float*, const float*,ulong, multBinaryOp<float>, plusBinaryOp<float>, float);
template double CuMatrix<double>::matrixReduceOpLauncher<multBinaryOp<double>, plusBinaryOp<double> >(double*, const double*, const double*, ulong, multBinaryOp<double>, plusBinaryOp<double>, double);
template float CuMatrix<float>::matrixReduceOpLauncher<diffSquaredBinaryOp<float>, plusBinaryOp<float> >(float*, const float*, const float*,ulong, diffSquaredBinaryOp<float>, plusBinaryOp<float>, float);
template double CuMatrix<double>::matrixReduceOpLauncher<diffSquaredBinaryOp<double>, plusBinaryOp<double> >(double*, const double*, const double*, ulong, diffSquaredBinaryOp<double>, plusBinaryOp<double>, double);
template float CuMatrix<float>::matrixReduceOpLauncher<equalsBinaryOp<float>, plusBinaryOp<float> >(float*, const float*, const float*,ulong, equalsBinaryOp<float>, plusBinaryOp<float>, float);
template double CuMatrix<double>::matrixReduceOpLauncher<equalsBinaryOp<double>, plusBinaryOp<double> >(double*, const double*, const double*, ulong, equalsBinaryOp<double>, plusBinaryOp<double>, double);
template float CuMatrix<float>::matrixReduceOpLauncher<almostEqualsBinaryOp<float>, andBinaryOp<float> >(float*, const float*, const float*,ulong, almostEqualsBinaryOp<float>, andBinaryOp<float>, float);
template double CuMatrix<double>::matrixReduceOpLauncher<almostEqualsBinaryOp<double>, andBinaryOp<double> >(double*, const double*, const double*, ulong, almostEqualsBinaryOp<double>, andBinaryOp<double>, double);
template float CuMatrix<float>::matrixReduceOpLauncher<equalsBinaryOp<float>, andBinaryOp<float> >(float*, const float*, const float*,ulong, equalsBinaryOp<float>, andBinaryOp<float>, float);
template double CuMatrix<double>::matrixReduceOpLauncher<equalsBinaryOp<double>, andBinaryOp<double> >(double*, const double*, const double*, ulong,  equalsBinaryOp<double>, andBinaryOp<double>, double);


//
// input pass works across two matrices via 'MatBinaryOp', subsequent passes are regular self-reductions
template<typename T> template<typename UnaryOp, typename BinaryOp> T CuMatrix<T>::gloloReduceOpLauncher(
		T* d_odata, const T* d_idata, ulong n,	UnaryOp gop, BinaryOp lop,
		T start)  {
	T gpu_result = 0;
	gpu_result = 0;

	int blocks,threads;
	getReductionExecContext(blocks,threads, n);
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
			gloloReduceOpKernel<T, 1024, true, UnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 512:
			gloloReduceOpKernel<T, 512, true, UnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 256:
			gloloReduceOpKernel<T, 256, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 128:
			gloloReduceOpKernel<T, 128, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 64:
			gloloReduceOpKernel<T, 64, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 32:
			gloloReduceOpKernel<T, 32, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 16:
			gloloReduceOpKernel<T, 16, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 8:
			gloloReduceOpKernel<T, 8, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 4:
			gloloReduceOpKernel<T, 4, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 2:
			gloloReduceOpKernel<T, 2, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 1:
			gloloReduceOpKernel<T, 1, true,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, n,gop, lop, start); break;
		}
	} else {
		switch (threads) {
			case 1024:
			gloloReduceOpKernel<T, 1024, false, UnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 512:
			gloloReduceOpKernel<T, 512, false, UnaryOp, BinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 256:
			gloloReduceOpKernel<T, 256, false,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 128:
			gloloReduceOpKernel<T, 128, false,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 64:
			gloloReduceOpKernel<T, 64, false,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 32:
			gloloReduceOpKernel<T, 32, false,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 16:
			gloloReduceOpKernel<T, 16, false,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 8:
			gloloReduceOpKernel<T, 8, false,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 4:
			gloloReduceOpKernel<T, 4, false,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 2:
			gloloReduceOpKernel<T, 2, false,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n,gop, lop, start); break;
			case 1:
			gloloReduceOpKernel<T, 1, false,UnaryOp, BinaryOp><<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, n,gop, lop, start); break;
		}
	}

	cudaError_t cuda_error = syncHappy ? cudaDeviceSynchronize() : cudaSuccess;
	checkCudaError(cuda_error);
	if(cuda_error == cudaSuccess)
			partialReduceLauncher( d_odata, blocks, lop,start);

	// copy final sum from device to host
	checkCudaError(
			cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));

	return gpu_result;
}

template float CuMatrix<float>::gloloReduceOpLauncher<oneOrZeroBoolUnaryOp<float>, andBinaryOp<float> >(float*, const float*, ulong, oneOrZeroBoolUnaryOp<float>, andBinaryOp<float>, float);
template double CuMatrix<double>::gloloReduceOpLauncher<oneOrZeroBoolUnaryOp<double>, andBinaryOp<double> >(double*, const double*, ulong, oneOrZeroBoolUnaryOp<double>, andBinaryOp<double>, double);
template float CuMatrix<float>::gloloReduceOpLauncher<sqrUnaryOp<float>, plusBinaryOp<float> >(float*, const float*, ulong, sqrUnaryOp<float>, plusBinaryOp<float>, float);
template double CuMatrix<double>::gloloReduceOpLauncher<sqrUnaryOp<double>, plusBinaryOp<double> >(double*, const double*, ulong, sqrUnaryOp<double>, plusBinaryOp<double>, double);
template float CuMatrix<float>::gloloReduceOpLauncher<almostEqualsBoolUnaryOp<float>, andBinaryOp<float> >(float*, const float*, ulong, almostEqualsBoolUnaryOp<float>, andBinaryOp<float>, float);
template double CuMatrix<double>::gloloReduceOpLauncher<almostEqualsBoolUnaryOp<double>, andBinaryOp<double> >(double*, const double*, ulong, almostEqualsBoolUnaryOp<double>, andBinaryOp<double>, double);

#include "CuMatrixInster.cu"
