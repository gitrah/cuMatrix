/*
 * CuMatrixProductReduction.cu
 *
 *      plagiarist: reid
 */

#include "CuMatrix.h"
#include "caps.h"
#include <typeinfo>
#include "Kernels.h"
#include "CuDefs.h"

__host__ __device__ inline bool ilIsPow2(uint x) {
    return ((x&(x-1))==0);
}


#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class MatBinaryOp,
template <typename> class BinaryOp>
__global__ void combineReduceOpKernel2(const T* g_idata1, const T* g_idata2,
		T* g_odata, long n, MatBinaryOp<T> mop, BinaryOp<T> op, T start, uint grow, uint gcol)
#else
template<typename T, uint blockSize, bool nIsPow2, int MopDim,int BopDim>
__global__ void combineReduceOpKernel2(const T* g_idata1, const T* g_idata2,
		T* g_odata, long n, BinaryOpF<T,MopDim> mop, BinaryOpF<T,BopDim> op, T start, uint grow, uint gcol)
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
		g_odata[blockIdx.x] = myReduction;
		if(gcol == 0 && grow == 0) {
			printf("blockIdx.x %d\n", blockIdx.x);
		}
	}
	//
	// write result for this block to global mem
}


#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class BinaryOp>
__global__ void reduceOpKernel2( T* g_odata, const T* g_idata, long n,
		BinaryOp<T> op, T start)
#else
template<typename T, uint blockSize, bool nIsPow2, int StateDim>
__global__ void reduceOpKernel2( T* g_odata, const T* g_idata, long n,
		BinaryOpF<T,StateDim> op, T start)
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

	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}
}



#ifdef CuMatrix_Enable_Cdp

template __global__ void matrixProductReductionTxdBKernel(DMatrix<float> C,
		const DMatrix<float> A, const DMatrix<float> B, int stepsDummy);
template __global__ void matrixProductReductionTxdBKernel(DMatrix<double> C,
		const DMatrix<double> A, const DMatrix<double> B, int stepsDummy);
template __global__ void matrixProductReductionTxdBKernel(DMatrix<ulong> C,
		const DMatrix<ulong> A, const DMatrix<ulong> B, int stepsDummy);
template __global__ void matrixProductReductionTxdBKernel(DMatrix<long> C,
		const DMatrix<long> A, const DMatrix<long> B, int stepsDummy);
template __global__ void matrixProductReductionTxdBKernel(DMatrix<uint> C,
		const DMatrix<uint> A, const DMatrix<uint> B, int stepsDummy);
template __global__ void matrixProductReductionTxdBKernel(DMatrix<int> C,
		const DMatrix<int> A, const DMatrix<int> B, int stepsDummy);


template<typename T> __global__ void matrixProductReductionTxdBKernel(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int stepsDummy) {
	// Block row and column
	const int blockRow = blockIdx.y;
	const int blockCol = blockIdx.x;
	// Each thread block computes one sub-matrix Csub of C,
	// each thread launches a product-sum reduction for each element of Csub that maps a row of A and a row (transposed col) of B
	// take steps of blockDim.x cols across row of A and steps of blockDim.y rows across col of B for each pair of strips that
	// sum-of-products reduce to get each Cvalue
	DMatrix<T> Csub = GetSubMatrix(C, blockRow, blockCol, blockDim);
	// Thread row and column within Csub (and shared mem matrices)
	const int row = threadIdx.y;
	const int col = threadIdx.x;
	const int grow = blockRow * blockDim.y + row;
	const int gcol = blockCol * blockDim.x + col;
	// locate Asub (row of A at grow) and Bsub (row (txd col) of B at gcol)
	DMatrix<T> Asub(&A.elements[A.p * grow], 1, A.n, A.p);
	DMatrix<T> Bsub(&B.elements[B.p * gcol], 1, B.n, B.p);

	long n = A.n;
	int threads;
	int blocks;
	getReductionExecContext(blocks,threads, n,256, 64);
	const int resSize = blocks * sizeof(T);
	T* d_res = null;
	cudaMalloc((void**) &d_res, resSize);
#ifdef  CuMatrix_Enable_KTS
	multBinaryOp<T>  mop;
	plusBinaryOp<T> op;
#else
	multBinaryOp<T>  mop = Functory<T, multBinaryOp>::pinch();
	plusBinaryOp<T> op = Functory<T, plusBinaryOp>::pinch();
#endif
	// sum partial block sums on GPU
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	T start = 0;
	bool powOf2Q = ilIsPow2(n);
	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize =
			(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	if(grow == 0 && gcol == 0) {
		printf("b %d t %d sm %d\n", blocks, threads, smemSize);
	}
	if(checkDebug(debugMatProd) && row == 0 && col == 0) {
		printf("Asub gr %d gc %d el %p m %d n %d p %d\n", grow,gcol, Asub.elements, Asub.m, Asub.n, Asub.p);
		printf("Bsub gr %d gc %d el %p m %d n %d p %d\n", grow,gcol, Bsub.elements, Bsub.m, Bsub.n, Bsub.p);
	}

	if (powOf2Q) {
		switch (threads) {
#ifdef  CuMatrix_Enable_KTS
			case 1024:
			combineReduceOpKernel2<T, 1024, true, multBinaryOp, plusBinaryOp> <<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 512:
			combineReduceOpKernel2<T, 512, true, multBinaryOp, plusBinaryOp> <<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 256:
			combineReduceOpKernel2<T, 256, true, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 128:
			combineReduceOpKernel2<T, 128, true, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 64:
			combineReduceOpKernel2<T, 64, true, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 32:
			combineReduceOpKernel2<T, 32, true, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 16:
			combineReduceOpKernel2<T, 16, true, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 8:
			combineReduceOpKernel2<T, 8, true, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 4:
			combineReduceOpKernel2<T, 4, true, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 2:
			combineReduceOpKernel2<T, 2, true, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 1:
			combineReduceOpKernel2<T, 1, true, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>( Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
#else
			case 1024:
			combineReduceOpKernel2<T, 1024, true, 1,1> <<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 512:
			combineReduceOpKernel2<T, 512, true, 1,1> <<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 256:
			combineReduceOpKernel2<T, 256, true, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 128:
			combineReduceOpKernel2<T, 128, true, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 64:
			combineReduceOpKernel2<T, 64, true, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 32:
			combineReduceOpKernel2<T, 32, true, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 16:
			combineReduceOpKernel2<T, 16, true, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 8:
			combineReduceOpKernel2<T, 8, true, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 4:
			combineReduceOpKernel2<T, 4, true, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 2:
			combineReduceOpKernel2<T, 2, true, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
			case 1:
			combineReduceOpKernel2<T, 1, true, 1,1><<< dimGrid, dimBlock, smemSize >>>( Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
#endif
		}
	}else {
			switch (threads) {
#ifdef  CuMatrix_Enable_KTS
				case 1024:
				combineReduceOpKernel2<T, 1024, false, multBinaryOp, plusBinaryOp> <<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 512:
				combineReduceOpKernel2<T, 512, false, multBinaryOp, plusBinaryOp> <<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 256:
				combineReduceOpKernel2<T, 256, false, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 128:
				combineReduceOpKernel2<T, 128, false, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 64:
				combineReduceOpKernel2<T, 64, false, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 32:
				combineReduceOpKernel2<T, 32, false, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 16:
				combineReduceOpKernel2<T, 16, false, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 8:
				combineReduceOpKernel2<T, 8, false, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 4:
				combineReduceOpKernel2<T, 4, false, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 2:
				combineReduceOpKernel2<T, 2, false, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 1:
				combineReduceOpKernel2<T, 1, false, multBinaryOp, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>( Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
#else
				case 1024:
				combineReduceOpKernel2<T, 1024, false, 1,1> <<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 512:
				combineReduceOpKernel2<T, 512, false, 1,1> <<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 256:
				combineReduceOpKernel2<T, 256, false, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 128:
				combineReduceOpKernel2<T, 128, false, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 64:
				combineReduceOpKernel2<T, 64, false, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 32:
				combineReduceOpKernel2<T, 32, false, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 16:
				combineReduceOpKernel2<T, 16, false, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 8:
				combineReduceOpKernel2<T, 8, false, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 4:
				combineReduceOpKernel2<T, 4, false, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 2:
				combineReduceOpKernel2<T, 2, false, 1,1><<< dimGrid, dimBlock, smemSize >>>(Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
				case 1:
				combineReduceOpKernel2<T, 1, false, 1,1><<< dimGrid, dimBlock, smemSize >>>( Asub.elements, Bsub.elements, d_res, n,mop,op, start, grow, gcol); break;
#endif
			}

	}
	if(grow == 0 && gcol == 0) {
		printf("grow %d\n", grow);
	}
	cudaError_t cuda_error = cudaSuccess;
	__syncthreads();

	if(grow == 0 && gcol == 0) {
		printf("blocks %d\n", blocks);
	}
	n= blocks;
	while (cuda_error == cudaSuccess && n > 1) {
		getReductionExecContext(blocks, threads, n,256, 64);
		int smemSize =
				(threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
		dim3 dimGrid(blocks), dimBlock(threads);

		bool powOf2Q = ilIsPow2(n);
		if (powOf2Q) {
			switch (threads) {
#ifdef  CuMatrix_Enable_KTS
				case 1024:
				reduceOpKernel2<T, 1024, true, plusBinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 512:
				reduceOpKernel2<T, 512, true, plusBinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 256:
				reduceOpKernel2<T, 256, true, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 128:
				reduceOpKernel2<T, 128, true, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 64:
				reduceOpKernel2<T, 64, true, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 32:
				reduceOpKernel2<T, 32, true, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 16:
				reduceOpKernel2<T, 16, true, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 8:
				reduceOpKernel2<T, 8, true, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 4:
				reduceOpKernel2<T, 4, true, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 2:
				reduceOpKernel2<T, 2, true, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 1:
				reduceOpKernel2<T, 1, true, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
#else
				case 1024:
				reduceOpKernel2<T, 1024, true, 1> <<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 512:
				reduceOpKernel2<T, 512, true, 1> <<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 256:
				reduceOpKernel2<T, 256, true, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 128:
				reduceOpKernel2<T, 128, true, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 64:
				reduceOpKernel2<T, 64, true, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 32:
				reduceOpKernel2<T, 32, true, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 16:
				reduceOpKernel2<T, 16, true, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 8:
				reduceOpKernel2<T, 8, true, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 4:
				reduceOpKernel2<T, 4, true, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 2:
				reduceOpKernel2<T, 2, true, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 1:
				reduceOpKernel2<T, 1, true, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
#endif
			}

		} else {
			switch (threads) {
#ifdef  CuMatrix_Enable_KTS
				case 1024:
				reduceOpKernel2<T, 1024, false, plusBinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 512:
				reduceOpKernel2<T, 512, false, plusBinaryOp> <<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 256:
				reduceOpKernel2<T, 256, false, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 128:
				reduceOpKernel2<T, 128, false, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 64:
				reduceOpKernel2<T, 64, false, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 32:
				reduceOpKernel2<T, 32, false, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 16:
				reduceOpKernel2<T, 16, false, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 8:
				reduceOpKernel2<T, 8, false, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 4:
				reduceOpKernel2<T, 4, false, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 2:
				reduceOpKernel2<T, 2, false, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 1:
				reduceOpKernel2<T, 1, false, plusBinaryOp><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
#else
				case 1024:
				reduceOpKernel2<T, 1024, false, 1> <<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 512:
				reduceOpKernel2<T, 512, false, 1> <<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 256:
				reduceOpKernel2<T, 256, false, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 128:
				reduceOpKernel2<T, 128, false, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 64:
				reduceOpKernel2<T, 64, false, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 32:
				reduceOpKernel2<T, 32, false, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 16:
				reduceOpKernel2<T, 16, false, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 8:
				reduceOpKernel2<T, 8, false, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 4:
				reduceOpKernel2<T, 4, false, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 2:
				reduceOpKernel2<T, 2, false, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
				case 1:
				reduceOpKernel2<T, 1, false, 1><<< dimGrid, dimBlock, smemSize >>>(d_res, d_res, n,op, start); break;
#endif
			}
		}
		//cherr(cuda_error);
		 //cudaDeviceSynchronize() ;
		__syncthreads();
		n = DIV_UP(n, 2*threads);
	}
	// copy final sum from device to host
	//outln("gpu_result " << gpu_result);
	cudaDeviceSynchronize();
	//__syncthreads();
	// Write Csub to device memory
	// Each thread writes one element
	//if (col + blockIdx.x * blockDim.x < C.n
	//		&& row + blockIdx.y * blockDim.y < C.m)
	if(row == 0 && col == 0) {
		printf("res at %d, %d %f\n", grow, gcol, d_res[0]);
	}
	if(col < Csub.n && row < Csub.m)
		SetElement(Csub, row, col, d_res[0]);
	cudaFree(d_res);
}
#endif

#include "CuMatrixInster.cu"
