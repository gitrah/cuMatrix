#include "CuMatrix.h"
#include "functors.h"

template<typename T, uint blockSize, bool nIsPow2, typename BinaryOp>
__global__ void reduceOpKernel( T* g_odata, const T* g_idata, ulong n,
		BinaryOp op, T start) {
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
		myReduction = op(myReduction, g_idata[i]);
		//printf("%f\n", myReduction  );
		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction = op(myReduction, g_idata[i + blockSize]);

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
}

template<typename T> inline __device__ T get(const DMatrix<T>& dm, uint l) {
//	if(dm.n == dm.p) {
//		return dm.elements[l];
//	}
	uint div = l /dm.n;
	uint idx = div * dm.p;
	idx += l - div * dm.n;
	//printf("offset l %u -> %u\n",l ,idx);
	return dm.elements[idx ];
}

template<typename T, uint blockSize, bool nIsPow2, typename BinaryOp>
__global__ void reduceOpMdKernel(DMatrix<T> out, const DMatrix<T> src, ulong n, BinaryOp op, T start) {
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
		myReduction = op(myReduction, get(src,i));
		//printf("%f\n", myReduction  );
		// ensure we don'float read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myReduction = op(myReduction, get(src,i + blockSize));

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
		out.elements[blockIdx.x] = sdata[0];
}

template<typename T> template<typename BinaryOp> cudaError_t CuMatrix<T>::partialReduceLauncher(T* d_odata, uint n,
		BinaryOp op, T start, cudaStream_t stream)  {
	// sum partial block sums on GPU
	cudaError_t cuda_error = cudaSuccess;
	int blocks,threads;
	if(debugExec)outln("partial n " << n);
	while (cuda_error == cudaSuccess && n > 1) {
		getReductionExecContext(blocks, threads, n);
		int smemSize =
				(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
		dim3 dimGrid(blocks), dimBlock(threads);
		if(debugExec)outln("partial dimGrid " << b_util::pd3(dimGrid));
		if(debugExec)outln("partial dimBlock " << b_util::pd3(dimBlock));
		if(debugExec)outln("partial smemSize " << smemSize);
		if(debugExec)outln("partial smem Ts " << smemSize/sizeof(T));
		bool powOf2Q = b_util::isPow2(n);
		if (powOf2Q) {
			switch (threads) {
				case 1024:
				reduceOpKernel<T, 1024, true, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 512:
				reduceOpKernel<T, 512, true, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 256:
				reduceOpKernel<T, 256, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 128:
				reduceOpKernel<T, 128, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 64:
				reduceOpKernel<T, 64, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 32:
				reduceOpKernel<T, 32, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 16:
				reduceOpKernel<T, 16, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 8:
				reduceOpKernel<T, 8, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 4:
				reduceOpKernel<T, 4, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 2:
				reduceOpKernel<T, 2, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 1:
				reduceOpKernel<T, 1, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
			}

		} else {
			switch (threads) {
				case 1024:
				reduceOpKernel<T, 1024, false, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 512:
				reduceOpKernel<T, 512, false, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 256:
				reduceOpKernel<T, 256, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 128:
				reduceOpKernel<T, 128, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 64:
				reduceOpKernel<T, 64, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 32:
				reduceOpKernel<T, 32, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 16:
				reduceOpKernel<T, 16, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 8:
				reduceOpKernel<T, 8, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_odata, n,op, start); break;
				case 4:
				reduceOpKernel<T, 4, false,BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n,op, start); break;
				case 2:
				reduceOpKernel<T, 2, false,BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n,op, start); break;
				case 1:
				reduceOpKernel<T, 1, false,BinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_odata, n,op, start); break;
			}
		}
		cudaError_t ret = syncHappy ? cudaDeviceSynchronize() : cudaSuccess;
		checkCudaError(cuda_error);

		n = (n + (threads * 2 - 1)) / (threads * 2);
	}
	return cuda_error;
}

template cudaError_t CuMatrix<float>::partialReduceLauncher<andBinaryOp<float> >( float*, uint,andBinaryOp<float>, float, cudaStream_t);
template cudaError_t CuMatrix<double>::partialReduceLauncher<andBinaryOp<double> >(double*, uint, andBinaryOp<double>, double, cudaStream_t);

template<typename T> template<typename BinaryOp> T CuMatrix<T>::reduceLauncher(T* d_odata, const T* d_idata,
		ulong n, BinaryOp op, T start, cudaStream_t stream)  {
	T gpu_result = 0;
	gpu_result = 0;

	int blocks,threads;
	getReductionExecContext(blocks, threads, n);

	// sum partial block sums on GPU
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	bool powOf2Q = b_util::isPow2(n);
	if(debugExec)outln("reduceLauncher n " << n);
	if(debugExec)outln("reduceLauncher dimGrid " << b_util::pd3(dimGrid));
	if(debugExec)outln("reduceLauncher dimBlock " << b_util::pd3(dimBlock));
	if(debugExec)outln("reduceLauncher smemSize " << smemSize);
	if(debugExec)outln("reduceLauncher smem Ts " << smemSize/sizeof(T));
	if (powOf2Q) {
		switch (threads) {
		case 1024:
			reduceOpKernel<T, 1024, true, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 512:
			reduceOpKernel<T, 512, true, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 256:
			reduceOpKernel<T, 256, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 128:
			reduceOpKernel<T, 128, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 64:
			reduceOpKernel<T, 64, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 32:
			reduceOpKernel<T, 32, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 16:
			reduceOpKernel<T, 16, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 8:
			reduceOpKernel<T, 8, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 4:
			reduceOpKernel<T, 4, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 2:
			reduceOpKernel<T, 2, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 1:
			reduceOpKernel<T, 1, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		}
	} else {
		switch (threads) {
		case 1024:
			reduceOpKernel<T, 1024, false, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 512:
			reduceOpKernel<T, 512, false, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 256:
			reduceOpKernel<T, 256, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 128:
			reduceOpKernel<T, 128, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 64:
			reduceOpKernel<T, 64, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 32:
			reduceOpKernel<T, 32, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 16:
			reduceOpKernel<T, 16, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 8:
			reduceOpKernel<T, 8, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 4:
			reduceOpKernel<T, 4, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 2:
			reduceOpKernel<T, 2, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		case 1:
			reduceOpKernel<T, 1, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(d_odata, d_idata, n,op, start); break;
		}
	}
	cudaError_t cuda_error = syncHappy ? cudaDeviceSynchronize() : cudaSuccess;
	checkCudaError(cuda_error);

	if(cuda_error == cudaSuccess)
			partialReduceLauncher(d_odata, blocks, op,start, stream);

	// copy final sum from device to host
	if(debugCopy) {
		outln("cudaMemcpyDeviceToHost " << d_odata << " to " << &gpu_result);
	}
	checkCudaError(
			cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));

	return gpu_result;
}
template float CuMatrix<float>::reduceLauncher<sqrPlusBinaryOp<float> >( float*, const float*, ulong, sqrPlusBinaryOp<float>, float,cudaStream_t);
template double CuMatrix<double>::reduceLauncher<sqrPlusBinaryOp<double> >(double*, const double*, ulong, sqrPlusBinaryOp<double>, double,cudaStream_t);
template float CuMatrix<float>::reduceLauncher<plusBinaryOp<float> >( float*, const float*, ulong, plusBinaryOp<float>, float,cudaStream_t);
template double CuMatrix<double>::reduceLauncher<plusBinaryOp<double> >(double*, const double*, ulong, plusBinaryOp<double>, double,cudaStream_t);
template float CuMatrix<float>::reduceLauncher<multBinaryOp<float> >(float*, const float*, ulong, multBinaryOp<float>, float,cudaStream_t);
template double CuMatrix<double>::reduceLauncher<multBinaryOp<double> >(double*, const double*, ulong, multBinaryOp<double>, double,cudaStream_t);
template float CuMatrix<float>::reduceLauncher<maxBinaryOp<float> >(float*, const float*, ulong, maxBinaryOp<float>, float,cudaStream_t);
template double CuMatrix<double>::reduceLauncher<maxBinaryOp<double> >(double*, const double*, ulong, maxBinaryOp<double>, double,cudaStream_t);
template float CuMatrix<float>::reduceLauncher<minBinaryOp<float> >(float*, const float*, ulong, minBinaryOp<float>, float,cudaStream_t);
template double CuMatrix<double>::reduceLauncher<minBinaryOp<double> >(double*, const double*, ulong, minBinaryOp<double>, double,cudaStream_t);

template<typename T> template<typename BinaryOp> T CuMatrix<T>::reduceLauncherDm(DMatrix<T>& buff, const DMatrix<T>& src,
		ulong n, BinaryOp op, T start, cudaStream_t stream) {
	T gpu_result = 0;

	int blocks,threads;
	getReductionExecContext(blocks, threads, n);
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize =
			(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	bool powOf2Q = b_util::isPow2(n);
	if(debugExec)outln("reduceLauncherDm n " << n);
	if(debugExec)outln("reduceLauncherDm dimGrid " << b_util::pd3(dimGrid));
	if(debugExec)outln("reduceLauncherDm dimBlock " << b_util::pd3(dimBlock));
	if(debugExec)outln("reduceLauncherDm smemSize " << smemSize);
	if(debugExec)outln("reduceLauncherDm smem Ts " << smemSize/sizeof(T));
	if (powOf2Q) {
		switch (threads) {
		case 1024:
			//reduceOpMdKernel(DMatrix<T> out, const DMatrix<T> src, T* g_odata, ulong n, BinaryOp op, T start)
			reduceOpMdKernel<T, 1024, true, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 512:
			reduceOpMdKernel<T, 512, true, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 256:
			reduceOpMdKernel<T, 256, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 128:
			reduceOpMdKernel<T, 128, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 64:
			reduceOpMdKernel<T, 64, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 32:
			reduceOpMdKernel<T, 32, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 16:
			reduceOpMdKernel<T, 16, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 8:
			reduceOpMdKernel<T, 8, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 4:
			reduceOpMdKernel<T, 4, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 2:
			reduceOpMdKernel<T, 2, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 1:
			reduceOpMdKernel<T, 1, true,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		}
	} else {
		switch (threads) {
		case 1024:
			reduceOpMdKernel<T, 1024, false, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 512:
			reduceOpMdKernel<T, 512, false, BinaryOp> <<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 256:
			reduceOpMdKernel<T, 256, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 128:
			reduceOpMdKernel<T, 128, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 64:
			reduceOpMdKernel<T, 64, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 32:
			reduceOpMdKernel<T, 32, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 16:
			reduceOpMdKernel<T, 16, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 8:
			reduceOpMdKernel<T, 8, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 4:
			reduceOpMdKernel<T, 4, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 2:
			reduceOpMdKernel<T, 2, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		case 1:
			reduceOpMdKernel<T, 1, false,BinaryOp><<< dimGrid, dimBlock, smemSize, stream >>>(buff, src, n,op, start); break;
		}
	}
	cudaError_t cuda_error = syncHappy ? cudaDeviceSynchronize() : cudaSuccess;
	checkCudaError(cuda_error);

	if(cuda_error == cudaSuccess)
			partialReduceLauncher(buff.elements, blocks, op, start);

	// copy final sum from device to host
	if(debugCopy) {
		outln("cudaMemcpyDeviceToHost " <<  buff.elements << " to " << &gpu_result);
	}
	checkCudaError(	cudaMemcpy(&gpu_result, buff.elements, sizeof(T), cudaMemcpyDeviceToHost));

	return gpu_result;
}
//																										ulong n, BinaryOp op, T start, cudaStream_t
template float CuMatrix<float>::reduceLauncherDm<sqrPlusBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, ulong, sqrPlusBinaryOp<float>, float,cudaStream_t);
template double CuMatrix<double>::reduceLauncherDm<sqrPlusBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, ulong, sqrPlusBinaryOp<double>, double,cudaStream_t);
template float CuMatrix<float>::reduceLauncherDm<plusBinaryOp<float> >( DMatrix<float>&, const DMatrix<float>&, ulong, plusBinaryOp<float>, float,cudaStream_t);
template double CuMatrix<double>::reduceLauncherDm<plusBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, ulong, plusBinaryOp<double>, double,cudaStream_t);
template float CuMatrix<float>::reduceLauncherDm<multBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&,ulong, multBinaryOp<float>,float,cudaStream_t);
template double CuMatrix<double>::reduceLauncherDm<multBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, ulong, multBinaryOp<double>, double,cudaStream_t);
template float CuMatrix<float>::reduceLauncherDm<maxBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, ulong, maxBinaryOp<float>,float,cudaStream_t);
template double CuMatrix<double>::reduceLauncherDm<maxBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, ulong, maxBinaryOp<double>, double,cudaStream_t);
template float CuMatrix<float>::reduceLauncherDm<minBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, ulong, minBinaryOp<float>,float,cudaStream_t);
template double CuMatrix<double>::reduceLauncherDm<minBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, ulong, minBinaryOp<double>, double,cudaStream_t);

#include "CuMatrixInster.cu"
