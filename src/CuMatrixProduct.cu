/*
 * CuMatrixProduct.cu
 *
 *      plagiarist: reid
 */

#include "CuMatrix.h"
#include "caps.h"
#include "Kernels.h"
#include <typeinfo>
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.p + col)
// Get a matrix element
template<typename T> inline __device__ float GetElement( volatile const DMatrix<T>& A, volatile  int row,
		volatile int col) {
	return (row < A.m && col < A.n) ? A.elements[row * A.p + col] : 0;
	//return A.elements[row * A.p + col];
}
template<typename T> __device__ float GetElementNI( volatile const DMatrix<T>& A, volatile  int row,
		volatile int col) {
	return (row < A.m && col < A.n) ? A.elements[row * A.p + col] : 0;
	//return A.elements[row * A.p + col];
}
// Set a matrix element
/*
template<typename T> inline __device__ void SetElement(DMatrix<T>& A, int row, int col,
		float value) {
	A.elements[row * A.p + col] = value;
}
*/
// Get the dimBlock.y x dimBLock.x sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
/*
template<typename T> inline __device__ DMatrix<T> GetSubMatrix( const DMatrix<T>& A,
		int row, int col, const dim3& dimBlock) {
	DMatrix<T> Asub;
	Asub.n = MIN(dimBlock.x, A.n - col * dimBlock.x);
	Asub.m = MIN(dimBlock.y, A.m - row * dimBlock.y);
	Asub.p = A.p;
	Asub.elements = &A.elements[A.p * dimBlock.y * row + dimBlock.x * col];
	return Asub;
}
*/

template<typename T> inline __device__ void GetSubMatrix(DMatrix<T>& sub, const DMatrix<T>& A,
		int row, int col, const dim3& dimBlock) {
	sub.n = MIN(dimBlock.x, A.n - col * dimBlock.x);
	sub.m = MIN(dimBlock.y, A.m - row * dimBlock.y);
	sub.elements =&A.elements[A.p * dimBlock.y * row + dimBlock.x * col];
}


template<typename T> __global__ void matrixProductBandwidthKernel(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int steps) {
	// Shared memory used to store Asub and Bsub respectively
	// Block row and column
// Each thread block computes one sub-matrix Csub of C
	//DMatrix<T> Csub = GetSubMatrix(C, blockRow, blockCol, blockDim);
// Each thread computes one element of Csub
// by accumulating results into Cvalue
	T Cvalue = 0;
// Thread row and column within Csub
	int off = ( blockDim.y * blockIdx.y +  threadIdx.y) * A.p +  blockDim.x * blockIdx.x + threadIdx.x;
// take steps of blockDim.x cols across row of A and steps of blockDim.y rows across col of B for each pair of strips that
// sum-of-products reduce to get each Cvalue
	//int gcol, grow;
	int asize = A.m * A.p;
	int bsize = B.m * B.p;
	int csize = C.m * C.p;
	for (int m = 0; m < steps; ++m) {
/*
		grow = m * blockDim.y;
		gcol = m * blockDim.x;
		if(grow > A.m || grow > B.m || gcol > A.n || gcol > B.n) {
			continue;
		}
*/
		// if m puts sub outside of A or B just continue


// Get sub-matrix Asub of A
		//  &A.elements[A.p * dimBlock.y * row + dimBlock.x * col]
//		DMatrix<T> Asub = GetSubMatrix(A, blockRow, m, blockDim);
// Get sub-matrix Bsub of B
	//	DMatrix<T> Bsub = GetSubMatrix(B, m, blockCol, blockDim);
// Load Asub and Bsub from device memory
		if(off < asize && off < bsize) {
			Cvalue += A.elements[off] * B.elements[off];
		}
		__syncthreads();
	}
// Write Csub to device memory
// Each thread writes one element
	if (off < csize)
		C.elements[off] = Cvalue;
}

template<typename T> __global__ void matrixProductKernel(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int steps) {
	// Shared memory used to store Asub and Bsub respectively
	T* As = SharedMemory<T>();
	T* Bs = As + blockDim.y * blockDim.x; // rows x cols, sized to traverse A and B in equal steps
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	// one sub-matrix Csub of C per thread block
	DMatrix<T> Csub = GetSubMatrix(C, blockRow, blockCol, blockDim);
	// Each thread computes one element of Csub by iterating through row of a and col of b by accumulating into Cvalue
	T Cvalue = 0;
// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;
// idx into a and b tiles in sm
	int smIdx = row * blockDim.y + col;
// take steps of blockDim.x cols across row of A and steps of blockDim.y rows across col of B for each pair of strips that
// sum-of-products reduce to get each Cvalue
	for (int m = 0; m < steps; ++m) {
/*
		grow = m * blockDim.y;
		gcol = m * blockDim.x;
		if(grow > A.m || grow > B.m || gcol > A.n || gcol > B.n) {
			continue;
		}
*/

// Get sub-matrix Asub of A
		DMatrix<T> Asub = GetSubMatrix(A, blockRow, m, blockDim);
// Get sub-matrix Bsub of B
		DMatrix<T> Bsub = GetSubMatrix(B, m, blockCol, blockDim);
// Load Asub and Bsub from device memory to shared memory
// Each thread loads one element of each sub-matrix
		// (row < A.m && col < A.n) ? A.elements[row * A.p + col] : 0
		As[smIdx] = GetElement(Asub, row, col);
		Bs[smIdx] = GetElement(Bsub, row, col);

// Synchronize to make sure the sub-matrices are loaded
// before starting the reduction
		__syncthreads();
// Multiply Asub and Bsub together
		for (int e = 0; e < blockDim.x; ++e) {
			Cvalue += As[row * blockDim.y + e] * Bs[e * blockDim.y + col];
		}

// Synchronize to make sure that the preceding
// computation is done before loading two new
// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
// Write Csub to device memory
// Each thread writes one element
	//if(row < A.m && col < A.n)
	if (col + blockIdx.x * blockDim.x < C.n
			&& row + blockIdx.y * blockDim.y < C.m)
		SetElement(Csub, row, col, Cvalue);
}

template<typename T> __global__ void matrixProductKernel2(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int steps) {
	// Shared memory used to store Asub and Bsub respectively
	T* As = SharedMemory<T>();
	//T* Bs = As + blockDim.y * blockDim.x; // rows x cols, sized to traverse A and B in equal steps
	// Block row and column
	const int blockRow = blockIdx.y;
	const int blockCol = blockIdx.x;
	// Each thread block computes one sub-matrix Csub of C, one element of Csub per thread
	// take steps of blockDim.x cols across row of A and steps of blockDim.y rows across col of B for each pair of strips that
	// sum-of-products reduce to get each Cvalue
	DMatrix<T> Csub = GetSubMatrix(C, blockRow, blockCol, blockDim);
	T Cvalue = 0;
	// Thread row and column within Csub (and shared mem matrices)
	const int row = threadIdx.y;
	const int col = threadIdx.x;
	if(blockCol == 0 && col == 0 && checkDebug(debugMatProd)){
		printf("\nmatrixProductKernel2 enter\n\n");
	}
	// locate Asub and Bsub at step 0 in their respective matrices
	//DMatrix<T> Asub(&A.elements[A.p * blockDim.y * blockRow], MIN(blockDim.y, A.m - blockRow * blockDim.y),0,A.p);
	//DMatrix<T> Bsub(&B.elements[blockDim.x * blockCol], 0, MIN(blockDim.x, B.n - blockCol * blockDim.x), B.p);
	uint Asubn = 0;
	uint Asubm = MIN(blockDim.y, A.m - blockRow * blockDim.y);
	T* AsubElements = A.elements;
	AsubElements += A.p * blockDim.y * blockRow;
	uint Bsubn = MIN(blockDim.x, B.n - blockCol * blockDim.x);
	uint Bsubm = 0;
	T* BsubElements = B.elements;
	BsubElements += blockDim.x * blockCol;

	const int bStepDelta = B.p * blockDim.y;
	// index into smem copies of Asub and Bsub; each same per thread
	const int smemIdx = row * blockDim.y + col;
	const int aOff = row * A.p + col;
	const int bOff = row * B.p + col;


	for (int m = 0; m < steps; ++m) {
		// Get sub-matrices Asub and Bsub, checking boundaries
		Asubn = MIN(blockDim.x, A.n - m * blockDim.x);
		Bsubm = MIN(blockDim.y, B.m - m * blockDim.y);
		// populate smem copies
		As[smemIdx] = (row < Asubm && col < Asubn) ? *(AsubElements + aOff) : 0;
		As[blockDim.y * blockDim.x + smemIdx] = (row < Bsubm && col < Bsubn) ? *(BsubElements + bOff) : 0;
		__syncthreads();
		// accum step's worth of Cvalue
		for (int e = 0; e < blockDim.x; ++e) {
			Cvalue += As[row * blockDim.y + e] * As[blockDim.y * blockDim.x + e * blockDim.y + col];
		}

		__syncthreads();
		// move Asub and Bsub a block step
		AsubElements += blockDim.x;
		BsubElements += bStepDelta;
	}
	// Write Csub to device memory
	// Each thread writes one element
	//if (col + blockIdx.x * blockDim.x < C.n
	//		&& row + blockIdx.y * blockDim.y < C.m)
	if(col < Csub.n && row < Csub.m)
		SetElement(Csub, row, col, Cvalue);
}


template<typename T> __global__ void matrixProductKernel3(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int steps) {
	// Shared memory used to store Asub and Bsub respectively
	T* As = SharedMemory<T>();
	T* Bs = As + blockDim.y * blockDim.x; // rows x cols, sized to traverse A and B in equal steps
	// Block row and column
	const int blockRow = blockIdx.y;
	const int blockCol = blockIdx.x;
	// Each thread block computes one sub-matrix Csub of C, one element of Csub per thread
	// take steps of blockDim.x cols across row of A and steps of blockDim.y rows across col of B for each pair of strips that
	// sum-of-products reduce to get each Cvalue
	DMatrix<T> Csub = GetSubMatrix(C, blockRow, blockCol, blockDim);
	T Cvalue = 0;
	// Thread row and column within Csub (and shared mem matrices)
	const int row = threadIdx.y;
	const int col = threadIdx.x;
	// locate Asub and Bsub at step 0 in their respective matrices
	DMatrix<T> Asub(&A.elements[A.p * blockDim.y * blockRow], MIN(blockDim.y, A.m - blockRow * blockDim.y),0,A.p);
	DMatrix<T> Bsub(&B.elements[blockDim.x * blockCol], 0, MIN(blockDim.x, B.n - blockCol * blockDim.x), B.p);
	const int bStepDelta = B.p * blockDim.y;
	// index into smem copies of Asub and Bsub; each same per thread
	const int smemIdx = row * blockDim.y + col;
	const int aOff = row * A.p + col;
	const int bOff = row * B.p + col;

		for (int m = 0; m < steps; ++m) {
			// Get sub-matrices Asub and Bsub, checking boundaries
			Asub.n = MIN(blockDim.x, A.n - m * blockDim.x);
			Bsub.m = MIN(blockDim.y, B.m - m * blockDim.y);
			// populate smem copies
			As[smemIdx] =  (row < Asub.m && col < Asub.n)? Asub.elements[aOff] : 0;
			Bs[smemIdx] =  (row < Bsub.m && col < Bsub.n)? Bsub.elements[bOff] : 0;
			__syncthreads();
			// accum step's worth of Cvalue
			for (int e = 0; e < blockDim.x; ++e) {
				Cvalue += As[row * blockDim.y + e] * Bs[e * blockDim.y + col];
			}

			__syncthreads();
			// move Asub and Bsub a block step
			Asub.elements += blockDim.x;
			Bsub.elements += bStepDelta;
		}
		// Write Csub to device memory
		// Each thread writes one element
		if(col < Csub.n && row < Csub.m) {
			SetElement(Csub, row, col, Cvalue);
		}
}

template<typename T> __global__ void matrixProductKernel4(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int steps) {
	// Shared memory used to store Asub and Bsub respectively
	T* As = SharedMemory<T>();
	T* Bs = As + blockDim.y * blockDim.x; // rows x cols, sized to traverse A and B in equal steps
	// Block row and column
	const int blockRow = blockIdx.y;
	const int blockCol = blockIdx.x;
	// Each thread block computes one sub-matrix Csub of C, one element of Csub per thread
	// take steps of blockDim.x cols across row of A and steps of blockDim.y rows across col of B for each pair of strips that
	// sum-of-products reduce to get each Cvalue
	DMatrix<T> Csub = GetSubMatrix(C, blockRow, blockCol, blockDim);
	T Cvalue = 0;
	// Thread row and column within Csub (and shared mem matrices)
	const int row = threadIdx.y;
	const int col = threadIdx.x;
	// locate Asub and Bsub at step 0 in their respective matrices
	DMatrix<T> Asub(&A.elements[A.p * blockDim.y * blockRow], MIN(blockDim.y, A.m - blockRow * blockDim.y),0,A.p);
	DMatrix<T> Bsub(&B.elements[blockDim.x * blockCol], 0, MIN(blockDim.x, B.n - blockCol * blockDim.x), B.p);
	const int bStepDelta = B.p * blockDim.y;
	// index into smem copies of Asub and Bsub; each same per thread
	const int asOff = row * blockDim.y;
	const int smemIdx = asOff + col;
	const int aOff = row * A.p + col;
	const int bOff = row * B.p + col;
	int asubN = A.n;
	int bsubM = B.m;
		for (int m = 0; m < steps; ++m) {
			// Get sub-matrices Asub and Bsub, checking boundaries
			Asub.n = MIN(blockDim.x, asubN );
			Bsub.m = MIN(blockDim.y, bsubM);
			// populate smem copies
			As[smemIdx] =  (row < Asub.m && col < Asub.n)? Asub.elements[aOff] : 0;
			//As[smemIdx] = Asub.elements[aOff];
			Bs[smemIdx] =  (row < Bsub.m && col < Bsub.n)? Bsub.elements[bOff] : 0;
			//Bs[smemIdx] =  Bsub.elements[bOff];
			__syncthreads();
			// accum step's worth of Cvalue
			int  bsOff = col;
			for (int e = 0; e < blockDim.x; ++e) {
				Cvalue += As[asOff + e] * Bs[bsOff];
				 bsOff += blockDim.y;
			}

			__syncthreads();
			// move Asub and Bsub a block step
			Asub.elements += blockDim.x;
			Bsub.elements += bStepDelta;
			asubN -= blockDim.x;
			bsubM -= blockDim.y;
		}
		// Write Csub to device memory
		// Each thread writes one element
		if(col < Csub.n && row < Csub.m) {
			SetElement(Csub, row, col, Cvalue);
		}
}

template<typename T> cudaError_t CuMatrix<T>::matrixProductL2(DMatrix<T>& d_res,
		const DMatrix<T>& d_A, const DMatrix<T>& d_B, dim3* block) {
	if(d_A.n != d_B.m) {
		dthrow(matricesOfIncompatibleShape());
	}
	if(d_res.m != d_A.m) {
		dthrow(rowDimsDisagree());
	}
	if(d_res.n != d_B.n) {
		dthrow(columnDimsDisagree());
	}
	if (debugMatProd && block)
		outln("blockSize " << b_util::pd3(*block));
	// block dim (bases?) should be warp-divisible
	// but constrained by smem
	if (block == null) {
		block = &DefaultMatProdBlock;
	}
	int columnBlocks = DIV_UP( d_A.n, block->x); //(d_A.n + block->x - 1) /block->x;
	int rowBlocks = DIV_UP(d_B.m, block->y ); //d_B.m + block->y - 1) / block->y;

	const int stepsPerResultBlock = MAX(columnBlocks,rowBlocks);
	if(checkDebug(debugMatProd))
		outln("columnBlocks " << columnBlocks << " rowBlocks " << rowBlocks);
	dim3 resultGrid(DIV_UP(d_res.n,block->x), DIV_UP(d_res.m, block->y));
	if (debugMatProd)
		outln("resultGrid " << b_util::pd3(resultGrid));
	uint smem = block->x * block->y * sizeof(T) * 2;
	if (smem > ExecCaps::currCaps()->memSharedPerBlock) {
		dthrow( new smemExceeded());
	}
	if (debugMatProd)
		outln( "launching matPrdKrnl resultGrid "<< b_util::pd3(resultGrid) << ", block " << b_util::pd3(*block) << ", smem " << smem);
	matrixProductKernel2<<<resultGrid, *block, smem>>>(d_res, d_A, d_B , stepsPerResultBlock);
	cudaError_t ret =  cudaDeviceSynchronize();
	checkCudaError(ret);
	return ret;
}
template<typename T> cudaError_t CuMatrix<T>::matrixProductL3(DMatrix<T>& d_res,
		const DMatrix<T>& d_A, const DMatrix<T>& d_B, dim3* block) {
	if(d_A.n != d_B.m) {
		dthrow(matricesOfIncompatibleShape());
	}
	if(d_res.m != d_A.m) {
		dthrow(rowDimsDisagree());
	}
	if(d_res.n != d_B.n) {
		dthrow(columnDimsDisagree());
	}
	if (checkDebug(debugMatProd) && block)
		outln("blockSize " << b_util::pd3(*block));
	// block dim (bases?) should be warp-divisible
	// but constrained by smem
	if (block == null) {
		block = &DefaultMatProdBlock;
	}
	int columnBlocks = DIV_UP( d_A.n, block->x); //(d_A.n + block->x - 1) /block->x;
	int rowBlocks = DIV_UP(d_B.m, block->y ); //d_B.m + block->y - 1) / block->y;

	const int stepsPerResultBlock = MAX(columnBlocks,rowBlocks);
	if(checkDebug(debugMatProd))
		outln("columnBlocks " << columnBlocks << " rowBlocks " << rowBlocks);
	dim3 resultGrid(DIV_UP(d_res.n,block->x), DIV_UP(d_res.m, block->y));
	if (debugMatProd)
		outln("resultGrid " << b_util::pd3(resultGrid));
	uint smem = block->x * block->y * sizeof(T) * 2;
	if (smem > ExecCaps::currCaps()->memSharedPerBlock) {
		dthrow( new smemExceeded());
	}
	if (debugMatProd)
		outln( "launching matPrdKrnl resultGrid "<< b_util::pd3(resultGrid) << ", block " << b_util::pd3(*block) << ", smem " << smem);

	matrixProductKernel4<<<resultGrid, *block, smem>>>(d_res, d_A, d_B , stepsPerResultBlock);
	// for the side effect of genearting the
	if(1==3) matrixProductKernel3<<<resultGrid, *block, smem>>>(d_res, d_A, d_B , stepsPerResultBlock);
	if(1 == 2) matrixProductBandwidthKernel<<<resultGrid, *block, smem>>>(d_res, d_A, d_B , stepsPerResultBlock);
	cudaError_t ret =  cudaDeviceSynchronize();
	checkCudaError(ret);
	return ret;
}

/*
 * this kernel assumes B is transposed, which enables coalesced reads from both A and B
 * (each is now a row strip)
 */
template<typename T> __global__ void matrixProductKernelTxdB(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int steps) {
	// Shared memory used to store Asub and Bsub respectively
	T* As = SharedMemory<T>();
	T* Bs = As + blockDim.y * blockDim.x; // rows x cols, sized to traverse A and B in equal steps
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
// Each thread block computes one sub-matrix Csub of C
	DMatrix<T> Csub = GetSubMatrix(C, blockRow, blockCol, blockDim);
// Each thread computes one element of Csub
// by accumulating results into Cvalue
	T Cvalue = 0;
// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;
/*
	DMatrix<T> Asub;
	DMatrix<T> Bsub;
	Asub.p = A.p;
	Asub.n = blockDim.x;
	Asub.m = blockDim.y;
	Bsub.p = B.p;
	Bsub.n = blockDim.x;
	Bsub.m = blockDim.y;
*/

	int smIdx = row * blockDim.y + col;
	/*Asub.elements = A.elements + A.p * blockDim.y * blockRow;
	Bsub.elements = B.elements + B.p * blockDim.x * blockCol;
	int aOff = row * A.p + col;
	int bOff = row * B.p + col;
	bool aIn = row < A.m && col < A.n;
	bool bIn = row < B.m && col < B.n;*/
// Loop over all the sub-matrices of A and B that are
// required to compute Csub
// Multiply each pair of sub-matrices together
// and accumulate the results
	for (int m = 0; m < steps; ++m) {
// Get sub-matrix Asub of A
		DMatrix<T> Asub = GetSubMatrix(A, blockRow, m, blockDim);
		//GetSubMatrix(Asub,A, blockRow, m, blockDim);
		//  &A.elements[A.p * dimBlock.y * row + dimBlock.x * col];
		//Asub.elements += blockDim.x;
// Get sub-matrix Bsub of B (pre-transposed)
		//  &A.elements[A.p * dimBlock.y * row + dimBlock.x * col];
		// [B.p * dimBlock.y * blockCol + dimBlock.x * m;
		DMatrix<T> Bsub = GetSubMatrix(B, blockCol, m, blockDim);
		//Bsub.elements += blockDim.y;
// Load Asub and Bsub from device memory to shared memory
// Each thread loads one element of each sub-matrix
		// (row < A.m && col < A.n) ? A.elements[row * A.p + col] : 0;
		As[smIdx] =  GetElement(Asub, row, col); // aIn ? Asub.elements[aOff] : 0; //
		Bs[smIdx] =  GetElement(Bsub, row, col); // bIn ? Bsub.elements[bOff] : 0; //

// Synchronize to make sure the sub-matrices are loaded
// before starting the reduction
		__syncthreads();
// Multiply Asub and Bsub together
		for (int e = 0; e < blockDim.x; ++e) {
			Cvalue += As[row * blockDim.y + e] * Bs[col * blockDim.y + e];
		}

// Synchronize to make sure that the preceding
// computation is done before loading two new
// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
// Write Csub to device memory
// Each thread writes one element
	//if(row < A.m && col < A.n)
	if (col + blockIdx.x * blockDim.x < C.n
			&& row + blockIdx.y * blockDim.y < C.m)
		SetElement(Csub, row, col, Cvalue);
}

/*
 * this kernel assumes B is transposed, which enables coalesced reads from both A and B
 * (each is now a row strip)
 */
template<typename T> __global__ void matrixProductKernelTxdB2(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int steps) {
	// Shared memory used to store Asub and Bsub respectively
	T* As = SharedMemory<T>();
	T* Bs = As + blockDim.y * blockDim.x; // rows x cols, sized to traverse A and B in equal steps
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
// Each thread block computes one sub-matrix Csub of C
	DMatrix<T> Csub = GetSubMatrix(C, blockRow, blockCol, blockDim);
// Each thread computes one element of Csub
// by accumulating results into Cvalue
	T Cvalue = 0;
// Thread row and column within Csub
	const int row = threadIdx.y;
	const int col = threadIdx.x;
	DMatrix<T> Asub(&A.elements[A.p * blockDim.y * blockRow], MIN(blockDim.y, A.m - blockRow * blockDim.y),0,A.p);
	DMatrix<T> Bsub(&B.elements[B.p * blockDim.x * blockCol], MIN(blockDim.y, B.m - blockCol * blockDim.y),0,B.p);
	const int smemIdx = row * blockDim.y + col;
	const int aOff = row * A.p + col;
	const int bOff = row * B.p + col;

	for (int m = 0; m < steps; ++m) {
		Asub.n = MIN(blockDim.x, A.n - m * blockDim.x);
		Bsub.n = MIN(blockDim.x, B.n - m * blockDim.x);
// Load Asub and Bsub from device memory to shared memory
		As[smemIdx] =  (row < Asub.m && col < Asub.n) ? Asub.elements[aOff] : 0;
		Bs[smemIdx] =  (row < Bsub.m && col < Bsub.n) ? Bsub.elements[bOff] : 0;

// Synchronize to make sure the sub-matrices are loaded
// before starting the reduction
		__syncthreads();
// Multiply Asub and Bsub together
		for (int e = 0; e < blockDim.x; ++e) {
			Cvalue += As[row * blockDim.y + e] * Bs[col * blockDim.y + e];
		}

// Synchronize to make sure that the preceding
// computation is done before loading two new
// sub-matrices of A and B in the next iteration
		__syncthreads();
		Asub.elements += blockDim.x;
		Bsub.elements += blockDim.x;
	}
// Write Csub to device memory
// Each thread writes one element
	//if(row < A.m && col < A.n)
	if (col + blockIdx.x * blockDim.x < C.n
			&& row + blockIdx.y * blockDim.y < C.m)
		SetElement(Csub, row, col, Cvalue);
}




template<typename T> cudaError_t CuMatrix<T>::matrixProductTxdbL(DMatrix<T>& d_res,
		const DMatrix<T>& d_A, const DMatrix<T>& d_B, dim3* block) {
	//const uint l = d_A.m * d_B.n;
	if(d_A.n != d_B.n) {
		dthrow(matricesOfIncompatibleShape());
	}
	if(d_res.m != d_A.m) {
		dthrow(rowDimsDisagree());
	}
	if(d_res.n != d_B.m) {
		dthrow(columnDimsDisagree());
	}
	if (debugMatProd && block)
		outln("blockSize " << b_util::pd3(*block));
	// block dim (bases?) should be warp-divisible
	// but constrained by smem
	if (block == null) {
		block = &DefaultMatProdBlock;
	}
	int columnBlocks = DIV_UP( d_A.n, block->x);
	int rowBlocks = DIV_UP(d_B.n, block->y );

	const int steps = MAX(columnBlocks,rowBlocks); // add extra step when A or B not of integral size
	if(checkDebug(debugMatProd))
		outln("columnBlocks " << columnBlocks << " rowBlocks " << rowBlocks);

	if (debugMatProd)
		outln("steps " << steps);
	dim3 resultGrid(DIV_UP(d_res.n,block->x), DIV_UP(d_res.m, block->y));
	if (debugMatProd)
		outln("resultGrid " << b_util::pd3(resultGrid));
	uint smem = block->x * block->y * sizeof(T) * 2;
	if (smem > ExecCaps::currCaps()->memSharedPerBlock) {
		dthrow( new smemExceeded());
	}
	if (debugMatProd)
		outln("launching matPrdKrnl resultGrid "<< b_util::pd3(resultGrid) << ", block " << b_util::pd3(*block) << ", smem " << smem);
	matrixProductKernelTxdB<<<resultGrid, *block, smem>>>(d_res, d_A, d_B , steps);

	cudaError_t ret = cudaDeviceSynchronize();
	checkCudaError(ret);
	return ret;
}


template<typename T> cudaError_t CuMatrix<T>::matrixProductReduxL(DMatrix<T>& d_res,
		const DMatrix<T>& d_A, const DMatrix<T>& d_B, dim3* block) {
	const uint l = d_A.m * d_B.n;
	bool notImpld = true;
	if(notImpld)
		dthrow(notImplemented());

	if (debugMatProd && block)
		outln("blockSize " << b_util::pd3(*block));
	// block dim (bases?) should be warp-divisible
	// but constrained by smem
	uint maxSamples = ExecCaps::currCaps()->memSharedPerBlock / sizeof(T);
	if (debugMatProd)
		outln("maxSamples " << maxSamples);
	if (debugMatProd)
		outln("d_A.m * d_B.n (l) = " << l);
	if (debugMatProd)
		outln("maxSamples / l " << (maxSamples / l));
	if (debugMatProd)
		outln(" l / maxSamples " << (l / maxSamples ));
	if (debugMatProd)
		outln("ExecCaps::currCaps()->thrdPerBlock " << ExecCaps::currCaps()->thrdPerBlock);
	uint orgDim = MAX(b_util::nextPowerOf2(d_A.m), b_util::nextPowerOf2(d_B.n));
	uint blockDim = 0;
	if (debugMatProd)
		outln("orgDim " << orgDim);
	uint i = 1;
	do {
		blockDim = orgDim / i;
		i++;
	} while (blockDim * blockDim > maxSamples
			|| blockDim * blockDim > ExecCaps::currCaps()->thrdPerBlock);
	if (debugMatProd)
		outln("blockDim " << blockDim << ",  i " << i);
	dim3 _block(blockDim, blockDim);
	if (debugMatProd)
		outln("_block " << b_util::pd3(_block));
	if (block == null)
		block = &_block;
	const int steps = MAX((d_A.n + block->x - 1) / block->x,
			(d_B.m + block->y - 1) / block->y); // add extra step when A or B not of integral size
	if (debugMatProd)
		outln("(d_A.n + block->x - 1) / block->x " << ((d_A.n + block->x - 1) / block->x) );
	if (debugMatProd)
		outln("(d_B.m + block->y - 1) / block->y " << ((d_B.m + block->y - 1) / block->y) );
	if (debugMatProd)
		outln("steps " << steps);
	dim3 dimGrid(DIV_UP(d_B.n, blockDim ),
			DIV_UP( d_A.m, blockDim));
	if (debugMatProd)
		outln("dimGrid " << b_util::pd3(dimGrid));
/*
	std::pair<uint, uint> remdrs(dimGrid.x * blockDim - d_B.n,
			dimGrid.y * blockDim - d_A.m);
	if (debugMatProd)
		outln("remdrs " << pp(remdrs));
	if (debugMem)
		outln(
				"d_A.e " << d_A.elements << ", d_B.e " << d_B.elements << ", d_res.e " << d_res.elements);
*/
	uint smem = blockDim * blockDim * sizeof(T) * 2;
	if (smem > ExecCaps::currCaps()->memSharedPerBlock) {
		dthrow( new smemExceeded());
	}
	if (debugMatProd)
		outln(
				"launching matPrdKrnl dimGrid "<< b_util::pd3(dimGrid) << ", block " << b_util::pd3(*block) << ", smem " << smem);
	//matrixProductKernel<<<dimGrid, dimBlock, smem>>>( d_A, d_B, d_C,blockSize);
	matrixProductKernel<<<dimGrid, *block, smem>>>(d_res, d_A, d_B , steps);
	int a = 2;
	if(a == 3) matrixProductKernelTxdB2<<<dimGrid, *block, smem>>>(d_res, d_A, d_B , steps);
//	if(a == 3) matrixProductReductionTxdBKernel<<<dimGrid, *block, smem>>>(d_res, d_A, d_B , steps);

	cudaError_t ret =  cudaDeviceSynchronize() ;
	checkCudaError(ret);
	return ret;
}

// want to write (RM) rows of res to coalesce.
// want to read (RM) rows of d_aRM to coalesce.
// want to read (CM) cols of d_bCM to coalesce
template<typename T> __global__ void matrixProductBCMKernel(DMatrix<T> res,
		const DMatrix<T> d_aRM, const DMatrix<T> d_bCM) {
	// Shared memory used to store Asub and Bsub respectively
	T* As = SharedMemory<T>();
}



#include "CuMatrixInster.cu"
