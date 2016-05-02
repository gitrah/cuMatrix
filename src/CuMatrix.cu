/*
 * CuMatrix.cu
 *
 *      Author: reid
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "CuMatrix.h"
#include "caps.h"
#include "debug.h"
#include "MemMgr.h"
#include "Kernels.h"
#include "DMatrix.h"

__constant__ uint D_MaxRowsDisplayed;
__constant__ uint D_MaxColsDisplayed;
cublasHandle_t g_handle;
bool g_useCublas = false;

template <typename T> __host__ void CuMatrix<T>::setMaxRowsDisplayed(int rows) {
	MaxRowsDisplayed = rows;
	//checkCudaError(cudaMemcpyToSymbol(D_MaxRowsDisplayed,  (void*) &CuMatrix<T>::MaxRowsDisplayed, sizeof(uint)));
}
template <typename T> __host__ void CuMatrix<T>::setMaxColsDisplayed(int cols) {
	MaxColsDisplayed = cols;
	//checkCudaError(cudaMemcpyToSymbol(D_MaxColsDisplayed,  (void*) &CuMatrix<T>::MaxColsDisplayed, sizeof(uint)));
}

template <typename T> typename MatProd<T>::MatProdKptr CuMatrix<T>::g_matrix_product_kernel = null;

template<typename T> int CuMatrix<T>::maxSlices(int n) {
	return ExecCaps::currCaps()->memSharedPerBlock / (2. * n * sizeof(T));
}

template<typename T, typename UnaryOp> __global__ void unaryOp1dKernel(
		T* trg, const T* src, UnaryOp op, ulong len) {
	ulong i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		trg[i] = op(src[i]);
	}
}

template<typename T, typename UnaryOp> __global__ void unaryOpDmKernel(
		DMatrix<T> trg, const DMatrix<T> src, UnaryOp op ) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.x + threadIdx.y;
	uint srcOff = y * src.p + x;
	uint trgOff = y * trg.p + x;
	for(int i = 0; i < blockDim.x; i+=blockDim.y) {
		if(x < src.n && y + i < src.m) {
			trg.elements[trgOff + i * trg.p] = op(src.elements[srcOff + i * src.p]);
		}
	}
}

// TODO implement column and row-wise reductions and replace these
template<typename T> __global__ void varianceMusKernel(DMatrix<T> sigmas, const DMatrix<T> x,
		const DMatrix<T> mus) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	T curr;
	T sqrDiff = 0;
	if (i < x.n) {
		const T currMu = mus.elements[i];
		//printf("i %d -> avg %f\n", i, currMu);
		for (int row = 0; row < x.m; row++) {
			curr = x.elements[row * x.p + i] - currMu;
			sqrDiff += curr * curr;
		}
		sigmas.elements[i] = sqrDiff / x.m;
	}
}

template<typename T> __global__ void varianceKernel(DMatrix<T> sigmas, DMatrix<T> mus, const DMatrix<T> x ) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	T avgSum = 0;
	T sqrDiff = 0;
	T curr;
	int row;
	// compute column (feature) averages
	if (i < x.n) {
		avgSum = x.elements[i];
		for (row = 1; row < x.m; row++) {
			curr = x.elements[row * x.p + i];
			avgSum += curr;
		}
		avgSum /= x.m; // now column avg
		mus.elements[i] = avgSum;
		//printf("i %d -> avg %f\n", i, avgSum);
		for (row = 0; row < x.m; row++) {
			curr = x.elements[row * x.p + i] - avgSum;
			sqrDiff += curr * curr;
		}
		sigmas.elements[i] = sqrDiff / x.m;
	}
}


template<typename T> void CuMatrix<T>::varianceL( DMatrix<T>& d_Sigmas, const DMatrix<T>& d_X,
		const DMatrix<T>& d_Mus) {
	dim3 block(MIN(512, b_util::nextPowerOf2(d_X.n)));
	dim3 grid(DIV_UP( d_X.n,block.x));
	outln("varianceL(&,const&,const&) for " << util<T>::pdm(d_X) << " with mus " << util<T>::pdm(d_Mus) << " have exctx grid " << b_util::pd3(grid) << " or blk " <<  b_util::pd3(block));
	varianceMusKernel<<<grid,block,0>>>(d_Sigmas,d_X,d_Mus);
}

template<typename T> void CuMatrix<T>::varianceAndMeanL(DMatrix<T>& d_Sigmas,  DMatrix<T>& d_Mus, const DMatrix<T>& d_X) {
	dim3 block(MIN(512, b_util::nextPowerOf2(d_X.n)));
	dim3 grid(DIV_UP(d_X.n,block.x));
	outln("varianceL(&,&,const&) for " << util<T>::pdm(d_X) << " have exctx grid " << b_util::pd3(grid) << " or blk " <<  b_util::pd3(block));
	varianceKernel<<<grid,block,0>>>(d_Sigmas, d_Mus, d_X);
}

/*
 * TODO re-implement
 *
 */

template<typename T> __global__ void featureAvgKernel(DMatrix<T> means, const DMatrix<T> x) {
	uint tid = threadIdx.x; // index into smem for block
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	T* sdata = SharedMemory<T>();
	// have to zero sm first?
	if (i < x.n) {
		sdata[tid] = x.elements[i];
	}

	if (i < x.n) {
		for (int row = 1; row < x.m; row++) {
				sdata[tid] += x.elements[row * x.p + i];
		}
	}
	//__syncthreads();
	if (tid < x.n) {
		sdata[tid] /= static_cast<T>(x.m);
	}
	//__syncthreads();
	if (i < x.n) {
		means.elements[i] = sdata[tid];
	}
}

template<typename T> __global__ void featureAvgKernelLv(DMatrix<T> means, const DMatrix<T> x) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	// have to zero sm first?
	T t = (i < x.n) ? x.elements[i] : 0;

	for (int row = 1; row < x.m; row++) {
		if (i < x.n) {
			t += x.elements[row * x.p + i];
			//if(i == 0){
			//	printf("row %d t %f\n",row,t);
			//}
		}
	}
	if (i < x.n) {
		means.elements[i] = t / x.m;
	}
}

template<typename T> __global__ void featureAvgKernelTxd(DMatrix<T> means, const DMatrix<T> txdX) {
	// 	T res = reduce(d_A, plusBinaryOp<T>(), 0 );
	T* sdata = SharedMemory<T>();
	int row = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	// have to zero sm first?
	//T t = (row < txdX.m) ? txdX.elements[row] : 0;
	printf("featureAvgKernelTxd txdX m %u* n %u\n", txdX.m, txdX.n);
	DMatrix<T> drow( txdX.elements + row * txdX.p, 1, txdX.n, txdX.p);
	printf("featureAvgKernelTxd drow.m %u drow.n %u drow.p %u\n", drow.m, drow.n, drow.p);
	printf("featureAvgKernelTxd row %u drow.elements %p drow.elements[0] %f drow.elements[n-1] %f\n",row, drow.elements,drow.elements[0],drow.elements[drow.n-1]);
	if(row==0)util<T>::printDm( drow,"featureAvgKernelTxd row 0 drow ");
	if(row==1)util<T>::printDm( drow,"featureAvgKernelTxd row 1 drow ");
	long nP = drow.n;
	int threads;
	int blocks;
	getReductionExecContext(blocks, threads, nP,128,512);
	printf("featureAvgKernelTxd row %u blocks %u thread %u nP %lu\n",row, blocks, threads, nP);
	//CuMatrix<T> res(blocks, 1, true, true);
	DMatrix<T> d_Res;
	d_Res.elements = means.elements + row;
	d_Res.m = d_Res.n = d_Res.p = 1;
	T dummyResult;
	//d_Res =
	//res.asDmatrix(d_Res, false);
	plusBinaryOp<T> op;
	//if(blocks == 1) { //BinaryOp op, T start, cudaStream_t stream
		reduceLauncher<T, plusBinaryOp>(&dummyResult, d_Res, nP, drow, op, 0,0);
	//}
	__syncthreads();
	if(row == 0)util<T>::printDm( d_Res,"featureAvgKernelTxd post reduceLauncherDm d_Res row 0");
	if(row == 1)util<T>::printDm( d_Res,"featureAvgKernelTxd post reduceLauncherDm d_Res row 1");
	//__syncthreads();
	printf("featureAvgKernelTxd row %d el[0] %f el[1] %f\n", row,d_Res.elements[0],d_Res.elements[1]  );
	printf("featureAvgKernelTxd  row %d nP %u avg[0] %f avg[1] %f\n", row, nP,d_Res.elements[0]/nP,d_Res.elements[1]/nP  );

	d_Res.elements[row] = d_Res.elements[row]/nP;
	__syncthreads();
	printf("featureAvgKernelTxd row %d avg %f\n", row,d_Res.elements[row] );
	//printf("d_Res.elements 0x%lx d_Res.elements[0] %f\n",d_Res.elements , d_Res.elements[0] );
	/*
	CuMatrix<T> res(blocks, 1, true, true);
	DMatrix<T> d_Res;
	res.asDmatrix(d_Res, false);
	T total;
	if(d_M.p != d_M.n) {
		total =
	} else {
	 total = reduceLauncher(res.d_ elements, d_M.elements, nP, op, start, stream);
	}
	if(checkDebug(debugMem)) outln("done with res " << res.toShortString());

	for (int row = 1; row < x.m; row++) {
		if (i < x.n) {
			t += x.elements[row * x.p + i];
		}
	}
	if (i < x.n) {
		means.elements[i] = t / x.m;
	}
*/
}

template<typename T> __global__ void featureSumKernelLv(DMatrix<T> means, const DMatrix<T> x) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	// have to zero sm first?
	T t = (i < x.n) ? x.elements[i] : 0;

	for (int row = 1; row < x.m; row++) {
		if (i < x.n) {
			t += x.elements[row * x.p + i];
			//if(i == 0){
			//	printf("row %d t %f\n",row,t);
			//}
		}
	}
	if (i < x.n) {
		means.elements[i] = t;
	}
}
// one thread per column
template<typename T> __global__ void featureSumKernelDivLv(DMatrix<T> means, const DMatrix<T> x, int count) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	// have to zero sm first?
	T t = (i < x.n) ? x.elements[i] : 0;

	for (int row = 1; row < x.m; row++) {
		if (i < x.n) {
			t += x.elements[row * x.p + i];
			//if(i == 0){
			//	printf("row %d t %f\n",row,t);
			//}
		}
	}
	if (i < x.n) {
		means.elements[i] = t/count;
	}
}

/*
 * TODO re-implement with column major mats that launch reducers for each column vector
 * 	(1 launch of nCols cores vs  (sum  (i 1-x) (x : m*n/2^i > 0)  cores over i launches)
 * 	need a way of dynamically selecting not just exec context but also among different algorithm -- ie learn which algorithm
 * 	to apply
 */


template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::featureAvgKernelL(DMatrix<T>& d_means, const DMatrix<T>& d_x, bool localVar, cudaStream_t stream ) {
	int threads =  MAX(d_means.m, d_means.n);
	dim3 dBlocks, dThreads;
	uint smem = d_x.n * sizeof(T);
	if (smem > ExecCaps::currCaps()->memSharedPerBlock) {
		setLastError ( smemExceededEx);
	}
	b_util::vectorExecContext(threads, d_x.n, dBlocks, dThreads);
	if (localVar) {
		featureAvgKernelLv<<<dBlocks, dThreads, 0, stream>>>(d_means, d_x);
	} else {
		featureAvgKernel<<<dBlocks, dThreads, smem, stream>>>(d_means, d_x);
	}
}


template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::featureAvgMultiStreamL(
		DMatrix<T>& d_means, const DMatrix<T>& d_x, bool localVar,
		int nstreams) {
	uint maxDim = MAX(d_means.m, d_means.n);
	int threads = maxDim;
	dim3 dBlocks, dThreads;
	uint smem = d_x.n * sizeof(T);
	if (smem > ExecCaps::currCaps()->memSharedPerBlock) {
		setLastError(smemExceededEx);
	}
	b_util::vectorExecContext(threads, d_x.n, dBlocks, dThreads);

	//int meanSize = MAX(d_means.m, d_means.n) * sizeof(T);
	cudaStream_t *streams = (cudaStream_t *) malloc(
			nstreams * sizeof(cudaStream_t));
	for (int i = 0; i < nstreams; i++) {
		cherr(cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking));
	}
	// create CUDA event handles
	// use blocking sync
	cudaEvent_t start_event, stop_event;

	cherr(cudaEventCreateWithFlags(&start_event, cudaStreamNonBlocking));
	cherr(cudaEventCreateWithFlags(&stop_event, cudaStreamNonBlocking));
	cudaEventRecord(start_event, 0);
	T* cols = new T[d_x.n];
	plusBinaryOp<T> pbo  = Functory<T,plusBinaryOp>::pinch();
	for (int i = 0; i < d_x.n; i++) {
		CuMatrix<T>::reduceColumn(cols + i, d_x, pbo, 0, i,
				streams[i % nstreams]);
	}
	cudaEventRecord(stop_event, 0);
#ifndef __CUDA_ARCH__
	cudaEventSynchronize(stop_event);
#endif
	// release resources
	for (int i = 0; i < nstreams; i++) {
		cudaStreamDestroy(streams[i]); // implicit sync on stream
	}
#ifndef __CUDA_ARCH__
	if(checkDebug(debugRedux))outln("cols before div " << util<T>::parry(cols,d_x.n));
#endif

	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	scaleUnaryOp<T> multf = Functory<T,scaleUnaryOp>::pinch( static_cast<T>(1. / d_x.n));
#ifndef __CUDA_ARCH__
	T* d_cols;
	cherr(cudaMalloc(&d_cols,d_x.n*sizeof(T)));
	cherr(cudaMemcpy(d_cols,cols,d_x.n*sizeof(T), cudaMemcpyHostToDevice));
	DMatrix<T> d_meansF(d_cols, d_x.n, 1);
	unaryOpL(d_means, d_meansF, multf);
#else
	DMatrix<T> d_meansF(cols, d_x.n, 1);
	unaryOpL(d_means, d_meansF, multf);
#endif

#ifndef __CUDA_ARCH__
	cudaFree(d_cols);
#endif
	delete [] cols;

}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::featureAvgTxdKernelL(DMatrix<T>& d_means, const DMatrix<T>& d_x) {
	int threads = MIN(512, MAX(d_means.m, d_means.n));
	dim3 dBlocks, dThreads;
	b_util::vectorExecContext(threads, d_x.m, dBlocks, dThreads);
	if(checkDebug(debugRedux)) {
		printf("featureAvgTxdKernelL threads %u rows %u \n ",threads, d_x.m );
		b_util::prd3(dBlocks,"featureAvgTxdKernelL dBlocks");
		b_util::prd3(dThreads,"featureAvgTxdKernelL dThreads");
	}

	// get row sums of txd X (==column sums of x)
	rowSum(d_means, d_x);

	// divide by sample count (row count of x, col count of txd x)
	scaleUnaryOp<T> multf = Functory<T,scaleUnaryOp>::pinch( static_cast<T>(1. / d_x.n));
	unaryOp1dKernel<<<dBlocks,dThreads>>>(d_means.elements, d_means.elements, multf, d_x.m);
	//return unaryOp(multf);

	//featureAvgKernelTxd<<<dBlocks, dThreads, smem>>>(d_means, d_x);
#ifdef __CUDA_ARCH__
	__syncthreads();
#endif
}

// very bad
template<typename T> __global__ void meanSubKernel_slow(DMatrix<T> res, const DMatrix<T> x,
		const DMatrix<T> means) {
//	uint tid = threadIdx.x; // index into smem for block
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	T mu;
	// have to zero sm first?
	if (i < x.n) {
		mu = means.elements[i];
		uint idx;
		for (int row = 0; row < x.m; row++) {
			idx = row * x.n + i;
			res.elements[idx] = x.elements[idx] - mu;
		}
	}
}

template<typename T> __global__ void meanSubSqrKernel(DMatrix<T> res, const DMatrix<T> x,
		const DMatrix<T> means) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	T mu;
	T curr;
	// have to zero sm first?
	if (i < x.n) {
		mu = means.elements[i];
		uint idx;
		for (int row = 0; row < x.m; row++) {
			idx = row * x.n + i;
			curr = x.elements[idx] - mu;
			res.elements[idx] = curr * curr;
		}
	}
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::meanSubL(DMatrix<T>& d_res, const DMatrix<T>& d_x, const DMatrix<T>& d_means, cudaStream_t stream) {
	int threads = 512;
	dim3 dBlocks, dThreads;
	uint smem = d_x.n * sizeof(T);
	if (smem > ExecCaps::currCaps()->memSharedPerBlock) {
		setLastError(smemExceededEx);
	}
	b_util::vectorExecContext(threads, d_x.n, dBlocks, dThreads);
	if(checkDebug(debugExec))printf(
			"meanSubL for %d*%d, have %d bloks of %d threads, smem %d\n", d_x.m, d_x.n, dBlocks.x, dThreads.x,smem );
	//outln("rows " << d_x.m);
	meanSubKernel_slow<<<dBlocks, dThreads, smem, stream>>>(d_res, d_x, d_means);
}

template<typename T> void CuMatrix<T>::meanSubSqrL(DMatrix<T>& d_res, const DMatrix<T>& d_x, const DMatrix<T>& d_means) {
	int threads = 512;
	dim3 dBlocks, dThreads;
	uint smem = d_x.n * sizeof(T);
	if (smem > ExecCaps::currCaps()->memSharedPerBlock) {
		throw new smemExceeded;
	}
	b_util::vectorExecContext(threads, d_x.n, dBlocks, dThreads);
	outln("meanSubL for " << d_x.m << "*" << d_x.n << ", have " << dBlocks.x << " blocks of " << dThreads.x << " threads with " << smem << " smem");
	//outln("rows " << d_x.m);
	meanSubSqrKernel<<<dBlocks, dThreads, smem>>>( d_res, d_x, d_means);
}

template<typename T> __global__ void columnProdKernel(DMatrix<T> prod, const DMatrix<T> x) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	T product = 1;
	if(i < x.n) {
		for (long row = 0; row < x.m; row++) {
			product *= x.elements[i + row * x.p];
		}
		prod.elements[i] = product;
	}
}

template<typename T> void CuMatrix<T>::columnProduct(DMatrix<T>& d_prod, const DMatrix<T>& d_x) {
	dim3 block(32), grid(DIV_UP( d_x.n,block.x));
	//outln("rows " << d_x.m);
	columnProdKernel<<<grid, block,0>>>(d_prod, d_x);
}

// copies as many rows as will fit to smem,
// uses first column of thread block to sum rows
// todo shuffle reduction for x.n < warp_size
//
template<typename T> __global__ void rowSumKernel(DMatrix<T> d_rowSums, const DMatrix<T> d_x) {
	int col = threadIdx.x; // index into column
	int row = blockIdx.y * blockDim.y + threadIdx.y; //
	uint tileIdx = threadIdx.x + threadIdx.y * blockDim.x;
	T* tile = SharedMemory<T>();
	T currRowSum = 0;
	if(threadIdx.x < d_x.n && row < d_x.m) {
		// copy tile from source
		tile[tileIdx] = d_x.elements[row * d_x.p + col];
		__syncthreads();
		// use first column of threads to sum rows
		if(threadIdx.x == 0) {
			if(checkDebug(debugExec)){flprintf("threadIdx.y %d\n", threadIdx.y);}
			for(int icol = 0; icol< d_x.n;icol++) {
				currRowSum += tile[tileIdx + icol];
			}
			d_rowSums.elements[row] = currRowSum;
		}
	}
}

template<typename T> string CuMatrix<T>::dimsString() const {
	stringstream ssout;
	ssout << m << "x" << n << "x" << p << "-" << size;
	return ssout.str();
}

template<typename T> string CuMatrix<T>::toShortString() const {
	stringstream ssout;
	ssout << "[[";
	ssout << this;
	ssout << " ";
	ssout << m;
	ssout << "x";
	ssout << n;
	ssout << "x";
	ssout << p;
	if(ownsBuffers) ssout << " owns";
	ssout << " (sz " << size << ") [";
	ssout << b_util::modStr(lastMod);
	ssout << (colMajor ? "] ColMajor" : "]");
	ssout << " h: ";
	if(elements) {
		ssout << elements; // << "/" << getMgr().refCount(elements);
	}	else
		ssout << "null";
	ssout << ", tiler: ";
	ssout << tiler;
	pair<int,int> par = refCounts();
	//if(tiler.hasDmemQ() && elements) {
		ssout << ", refCounts  h " << par.first << ", d " << par.second;
	//}
	ssout << "]]";
	return ssout.str();
}

template<typename T> template<template <typename> class CostFunction> void CuMatrix<T>::gradientApprox(
		CostFunction<T> costFn, DMatrix<T> theta, DMatrix<T> perturb, DMatrix<T> gradApprox, T epsilon) {
	const uint l = theta.m * theta.n;
	ulong i = 0;
	constFiller<T> filler;
	filler.value = 0;
	fillFn(filler, perturb);
	fillFn(filler, gradApprox);
	T jMinus = 0, jPlus = 0;
	while (i < l) {
		//perturb.set(i, epsilon);
		CuMatrix<T>::set(perturb.elements, perturb.m, perturb.n, perturb.p, i, static_cast<T>(epsilon));
		jMinus = costFn(theta - perturb);
		jPlus = costFn(theta + perturb);
		//gradApprox.set(i, (jPlus - jMinus) / (2. * epsilon));
		CuMatrix<T>::set(gradApprox.elements, gradApprox.m, gradApprox.n, gradApprox.p, i, static_cast<T>((jPlus - jMinus) / (2. * epsilon)));
		//perturb.set(i, 0);
		CuMatrix<T>::set(perturb.elements, perturb.m, perturb.n, perturb.p, i, static_cast<T>(0));
		i += 1;
	}
}

extern template class CuMatrix<float>;
extern template class CuMatrix<double>;
extern template class CuMatrix<long>;
extern template class CuMatrix<ulong>;
extern template class CuMatrix<int>;
extern template class CuMatrix<uint>;

#include "CuMatrixInster.cu"

