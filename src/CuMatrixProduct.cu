#include "CuMatrix.h"
#include "caps.h"

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.p + col)
// Get a matrix element
template<typename T> __device__ float GetElement(const DMatrix<T>& A, int row,
		int col) {
	return A.elements[row * A.p + col];
}
// Set a matrix element
template<typename T> __device__ void SetElement(DMatrix<T>& A, int row, int col,
		float value) {
	A.elements[row * A.p + col] = value;
}
// Get the dimBlock.y x dimBLock.x sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
template<typename T> __device__ DMatrix<T> GetSubMatrix(const DMatrix<T>& A,
		int row, int col, const dim3& dimBlock, bool reverse) {
	DMatrix<T> Asub;
	Asub.n = reverse ? dimBlock.y : dimBlock.x;
	Asub.m = reverse ? dimBlock.x : dimBlock.y;
	Asub.p = A.p;
	Asub.elements = &A.elements[A.p * Asub.m * row + Asub.n * col];
	return Asub;
}

template<typename T> __global__ void matrixProductKernel(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B) {
	// Shared memory used to store Asub and Bsub respectively
	T* As = SharedMemory<T>();
	T* Bs = As + blockDim.y * blockDim.x; // rows x cols, sized to traverse A and B in equal steps
	const int steps = max((A.n + blockDim.x - 1) / blockDim.x,
			(B.m + blockDim.y - 1) / blockDim.y); // add extra step when A or B not of integral size
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
// Each thread block computes one sub-matrix Csub of C
	DMatrix<T> Csub = GetSubMatrix(C, blockRow, blockCol, blockDim, false);
// Each thread computes one element of Csub
// by accumulating results into Cvalue
	T Cvalue = 0;
// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;
	int grow;
	int gcol;
// Loop over all the sub-matrices of A and B that are
// required to compute Csub
// Multiply each pair of sub-matrices together
// and accumulate the results
	for (int m = 0; m < steps; ++m) {
// Get sub-matrix Asub of A
		DMatrix<T> Asub = GetSubMatrix(A, blockRow, m, blockDim, false);
// Get sub-matrix Bsub of B
		DMatrix<T> Bsub = GetSubMatrix(B, m, blockCol, blockDim, false);
// Load Asub and Bsub from device memory to shared memory
// Each thread loads one element of each sub-matrix
		grow = m * blockDim.y + row;
		gcol = m * blockDim.x + col;
		As[row * blockDim.y + col] =
				(col >= A.n || gcol >= A.n || row >= A.m) ?
						0 : GetElement(Asub, row, col); // A.elements[row * A.p + col];
		Bs[row * blockDim.y + col] =
				(col >= B.n || grow >= B.m || row >= B.m) ?
						0 : GetElement(Bsub, row, col);

// Synchronize to make sure the sub-matrices are loaded
// before starting the computation
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

template<typename T> cudaError_t CuMatrix<T>::matrixProductL(DMatrix<T>& d_res,
		const DMatrix<T>& d_A, const DMatrix<T>& d_B, dim3* block) {
	const uint l = d_A.m * d_B.n;
	if(d_A.n != d_B.m) {
		dthrow(matricesOfIncompatibleShape());
	}
	if(d_res.m != d_A.m) {
		dthrow(rowDimsDontAgree());
	}
	if(d_res.n != d_B.n) {
		dthrow(columnDimsDontAgree());
	}
	if (syncHappy)
		checkCudaError(cudaDeviceSynchronize());
	if (debugMatProd && block)
		outln("blockSize " << b_util::pd3(*block));
	// block dim (bases?) should be warp-divisible
	// but constrained by smem
	uint maxSamples = caps.memSharedPerBlock / sizeof(T);
	if (debugMatProd)
		outln("maxSamples " << maxSamples);
	if (debugMatProd)
		outln("d_A.m * d_B.n (l) = " << l);
	if (debugMatProd)
		outln("caps.thrdPerBlock " << caps.thrdPerBlock);
	uint orgDim = max(b_util::nextPowerOf2(d_A.m), b_util::nextPowerOf2(d_B.n));
	uint blockDim = 0;
	if (debugMatProd)
		outln("orgDim " << orgDim);
	uint i = 1;
	do {
		blockDim = orgDim / i;
		//outln(i << ": blockDim^2 == " << (blockDim * blockDim));
		i++;
	} while (blockDim * blockDim > maxSamples
			|| blockDim * blockDim > caps.thrdPerBlock);
	if (debugMatProd)
		outln("blockDim " << blockDim << ",  i " << i);
	dim3 _block(blockDim, blockDim);
	if (debugMatProd)
		outln("_block " << b_util::pd3(_block));
	if (block == null) {
		block = &DefaultMatProdBlock;
	}
	const int steps = max(  enuff(block->y, d_A.m),enuff(block->y, d_B.n)); // add extra step when A or B not of integral size
	if (debugMatProd)
		outln(" enuff(block->y, d_A.m) " << enuff(block->y, d_A.m) );
	if (debugMatProd)
		outln(" enuff(block->y, d_B.n) " << enuff(block->y, d_B.n) );
	if (debugMatProd)
		outln("steps " << steps);
	dim3 dimGrid(b_util::enough(block->x, d_B.n), b_util::enough(block->x, d_A.m));
	//dim3 dimGrid(b_util::enough(blockDim, d_B.n), b_util::enough(blockDim, d_A.m ));
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
	uint smem = block->x * block->y * sizeof(T) * 2;
	if (smem > caps.memSharedPerBlock) {
		dthrow( new smemExceeded());
	}
	if (debugMatProd)
		outln(
				"launching matPrdKrnl dimGrid "<< b_util::pd3(dimGrid) << ", block " << b_util::pd3(*block) << ", smem " << smem);
	//matrixProductKernel<<<dimGrid, dimBlock, smem>>>( d_A, d_B, d_C,blockSize);
	matrixProductKernel<<<dimGrid, *block, smem>>>(d_res, d_A, d_B );

	cudaError_t ret = syncHappy ? cudaDeviceSynchronize() : cudaSuccess;
	checkCudaError(ret);
	return ret;
}


template<typename T> cudaError_t CuMatrix<T>::matrixProductReduxL(DMatrix<T>& d_res,
		const DMatrix<T>& d_A, const DMatrix<T>& d_B, dim3* block) {
	const uint l = d_A.m * d_B.n;
	if (syncHappy)
		checkCudaError(cudaDeviceSynchronize());
	if (debugMatProd && block)
		outln("blockSize " << b_util::pd3(*block));
	// block dim (bases?) should be warp-divisible
	// but constrained by smem
	uint maxSamples = caps.memSharedPerBlock / sizeof(T);
	if (debugMatProd)
		outln("maxSamples " << maxSamples);
	if (debugMatProd)
		outln("d_A.m * d_B.n (l) = " << l);
	if (debugMatProd)
		outln("maxSamples / l " << (maxSamples / l));
	if (debugMatProd)
		outln(" l / maxSamples " << (l / maxSamples ));
	if (debugMatProd)
		outln("caps.thrdPerBlock " << caps.thrdPerBlock);
	uint orgDim = max(b_util::nextPowerOf2(d_A.m), b_util::nextPowerOf2(d_B.n));
	uint blockDim = 0;
	if (debugMatProd)
		outln("orgDim " << orgDim);
	uint i = 1;
	do {
		blockDim = orgDim / i;
		i++;
	} while (blockDim * blockDim > maxSamples
			|| blockDim * blockDim > caps.thrdPerBlock);
	if (debugMatProd)
		outln("blockDim " << blockDim << ",  i " << i);
	dim3 _block(blockDim, blockDim);
	if (debugMatProd)
		outln("_block " << b_util::pd3(_block));
	if (block == null)
		block = &_block;
	const int steps = max((d_A.n + block->x - 1) / block->x,
			(d_B.m + block->y - 1) / block->y); // add extra step when A or B not of integral size
	if (debugMatProd)
		outln("(d_A.n + block->x - 1) / block->x " << ((d_A.n + block->x - 1) / block->x) );
	if (debugMatProd)
		outln("(d_B.m + block->y - 1) / block->y " << ((d_B.m + block->y - 1) / block->y) );
	if (debugMatProd)
		outln("steps " << steps);
	dim3 dimGrid(b_util::enough(blockDim, d_B.n),
			b_util::enough(blockDim, d_A.m));
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
	if (smem > caps.memSharedPerBlock) {
		dthrow( new smemExceeded());
	}
	if (debugMatProd)
		outln(
				"launching matPrdKrnl dimGrid "<< b_util::pd3(dimGrid) << ", block " << b_util::pd3(*block) << ", smem " << smem);
	//matrixProductKernel<<<dimGrid, dimBlock, smem>>>( d_A, d_B, d_C,blockSize);
	matrixProductKernel<<<dimGrid, *block, smem>>>(d_res, d_A, d_B );

	cudaError_t ret = syncHappy ? cudaDeviceSynchronize() : cudaSuccess;
	checkCudaError(ret);
	return ret;
}


#include "CuMatrixInster.cu"
