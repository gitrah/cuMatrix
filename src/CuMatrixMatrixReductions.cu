/*
 * CuMatrixMatrixReductions.cu
 *
 *      Author: reid
 */


#include "CuMatrix.h"
#include <helper_cuda.h>
#include "caps.h"
#include "debug.h"
#include "Kernels.h"

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class CombineOp, template <typename> class ReduceOp> __host__ CUDART_DEVICE
T CuMatrix<T>::combineReduceL(const DMatrix<T>& d_M1, const DMatrix<T>& d_M2, CombineOp<T> cop, ReduceOp<T> rop, T start, cudaStream_t stream) const
#else
template<typename T> template<int CopDim, int RopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::combineReduceL(const DMatrix<T>& d_M1, const DMatrix<T>& d_M2, BinaryOpF<T,CopDim> cop, BinaryOpF<T,RopDim> rop, T start, cudaStream_t stream) const
#endif
{
	if(checkDebug(debugRedux))prlocf("combineReduceL enter");
	long nP = d_M1.m * d_M1.n;
	int threads;
	int blocks;
	getReductionExecContext(blocks,threads, nP);
	if(checkDebug(debugRedux)){
		flprintf("combineReduceL(const DMatrix<T>& d_M) blocks %u threads %u nP %u\n", blocks, threads, nP);
	}
	CuMatrix<T> res(blocks, 1,true,true);
	DMatrix<T> d_Res;
	res.tile0(d_Res, false);
	T total = combineReduceOpLauncher(d_Res.elements, d_M1.elements, d_M2.elements, nP, cop, rop, start, stream);
	return total;
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class CombineOp, template <typename> class ReduceOp> __host__ CUDART_DEVICE
T CuMatrix<	T>::combineReduceL(CuMatrix<T>& buffer, const DMatrix<T>& d_M1, const DMatrix<T>& d_M2,
		CombineOp<T> cop, ReduceOp<T> rop, T start, cudaStream_t stream ) const
#else
template<typename T> template<int CopDim, int RopDim> __host__ CUDART_DEVICE
T CuMatrix<	T>::combineReduceL(CuMatrix<T>& buffer, const DMatrix<T>& d_M1, const DMatrix<T>& d_M2,
		BinaryOpF<T,CopDim> cop, BinaryOpF<T,RopDim> rop, T start, cudaStream_t stream ) const
#endif
{
	long nP = d_M1.m * d_M1.n;
	T total = combineReduceOpLauncher(buffer.tiler.currBuffer(), d_M1.elements, d_M2.elements, nP, cop, rop, start, stream);
	return total;
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class CombineOp, template <typename> class ReduceOp> __host__ CUDART_DEVICE
T CuMatrix<T>::combineReduce(CombineOp<T> cop, ReduceOp<T> rop, const CuMatrix<T>& o, T start, cudaStream_t stream) const
#else
template<typename T> template<int CopDim, int RopDim> __host__ CUDART_DEVICE
T CuMatrix<T>::combineReduce(BinaryOpF<T,CopDim> cop, BinaryOpF<T,RopDim> rop, const CuMatrix<T>& o, T start, cudaStream_t stream) const
#endif
{
	DMatrix<T> d_A, d_B;
	tile0(d_A,lastMod == mod_host);
	o.tile0(d_B,o.lastMod == mod_host);
	T res = combineReduceL(d_A, d_B, cop, rop, start, stream);
	return res;
}

template<typename T> void cublasSgemm(DMatrix<T> d_C, const DMatrix<T> d_A, const DMatrix<T> d_B) {
	outln("d_C " << util<T>::pdm(d_C));
	outln("d_A " << util<T>::pdm(d_A));
	outln("d_B " << util<T>::pdm(d_B));
#ifdef CuMatrix_UseCublas
	cublasStatus_t ret;
	if(sizeof(T) < 8) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        chblerr( cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_B.n, d_A.m, d_A.n, &alpha, (const float*)d_B.elements, d_B.p, (const float*)d_A.elements, d_A.p, &beta, (float*)d_C.elements, d_C.p));
	} else {
        const double alpha = 1.0;
        const double beta  = 0.0;
        chblerr( cublasDgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_B.n, d_A.m, d_A.n, &alpha, (const double*)d_B.elements, d_B.p, (const double*)d_A.elements, d_A.p, &beta, (double*)d_C.elements, d_C.p));
	}
#endif
}

/*
 * ie non-tiled (not 'in place' )
 */
template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::matrixProductResident(const CuMatrix<T>& b, dim3* block, cudaStream_t stream) const {
	DMatrix<T> d_A, d_B, d_C;
	ExecCaps* caps = ExecCaps::currCaps();
	uint blockY = 16;
	if(checkDebug(debugMatProd))	prlocf("non-tiled matprod\n");
	CuMatrix<T> res(m, b.n,b.n,true, true);
/*
	if(checkDebug(debugMatProd)) {
#ifndef __CUDA_ARCH__
		outln("this ");
		outln(toString());
		outln("b ");
		outln(b.toShortString());
#endif
	}
*/
	if(block) blockY = block->y;
	uint gridY = DIV_UP(res.m, blockY);
	float ratioY = 1.0 * gridY / caps->maxGrid.y;
	if(ratioY > 1) {
		return mkMatrixProduct(b, block, stream);
	}
	if(checkDebug(debugMatProd))flprintf("before tile0 d_A.m %u d_A.n %u\n", d_A.m, d_A.n);
	tile0(d_A, lastMod == mod_host);
	if(checkDebug(debugMatProd))flprintf("after tile0 d_A.m %u d_A.n %u\n", d_A.m, d_A.n);

	if(checkDebug(debugMatProd)) prlocf("tile0(d_A,true) ok\n");
	if(checkDebug(debugMatProd)) flprintf("pre b.tile0(d_B,tiler != b.tiler [[==%d]])\n",tiler != b.tiler );
	b.tile0(d_B, tiler != b.tiler && b.lastMod == mod_host);
	if(checkDebug(debugMatProd)) flprintf("b.tile0(d_B,tiler != b.tiler [[==%d]]) ok\n",tiler != b.tiler );
	res.tile0(d_C, false);
	if(checkDebug(debugMatProd)) prlocf("res.tile0(d_C, false) ok\n");
#ifndef __CUDA_ARCH__

	#ifdef CuMatrix_UseCublas
		if(checkDebug(debugMatProd)) prlocf("CuMatrix_UseCublas\n");
		if( g_useCublas) {
			if(checkDebug(debugMatProd)) prlocf("g_useCublas\n");
			cublasSgemm(d_C, d_A, d_B);
		}else {
			if(checkDebug(debugMatProd)) prlocf("!g_useCublas\n");
	#endif

		if(g_matrix_product_kernel) {
			if(checkDebug(debugMatProd)) prlocf("g_matrix_product_kernel\n");
			matrixProductKPtrL(d_C,g_matrix_product_kernel, d_A, d_B,  block,stream);
		}else {
			if(checkDebug(debugMatProd)) prlocf("!g_matrix_product_kernel\n");
			matrixProductL(d_C, d_A, d_B,  block,stream);
		}

	#ifdef CuMatrix_UseCublas
		}
	#endif
#else
	matrixProductL(d_C, d_A, d_B,  block);
#endif

	res.invalidateHost();
	return res;

}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::matrixProduct( const CuMatrix<T>& b, dim3* block, cudaStream_t stream ) const {
	if(checkDebug(debugMatProd))

	if(n != b.m) {
#ifndef __CUDA_ARCH__
		outln("this " << toShortString());
		outln("b " << b.toShortString());
#endif

		flprintf("n %u != b.m %u\n",n, b.m);
		setLastError(matricesOfIncompatibleShapeEx);
	}
	if(checkDebug(debugMatProd))	prlocf("matProd entre\n");
	// todo convert mats into big warp div matrix and one < blocksize matrix
	if(b.scalarQ()) {
		return operator *(b.get(0));
	} else if(scalarQ()) {
		return b.operator *(get(0));
	} else if(rowVectorQ() && b.columnVectorQ()) {
		// better as a reduction
		if(checkDebug(debugMatProd))
			prlocf("rowv . colv => reducc\n");
		assert(rowVectorQ() && b.columnVectorQ());
#ifdef  CuMatrix_Enable_KTS
		T ret = combineReduce(multBinaryOp<T>(), plusBinaryOp<T>(), b, 0, stream);
#else
		T ret = combineReduce(Functory<T,multBinaryOp>::pinch(), Functory<T,plusBinaryOp>::pinch(), b, 0, stream);
#endif
		if(checkDebug(debugMatProd))
			flprintf("matProd expecting a vector %.20g\n",ret);
		return fromScalar(ret);
	}

	if(n != b.m) {
		if(checkDebug(debugMatProd))flprintf("this mXn %u X %u, b %u x %u\n", m, n, b.m,b.n);
	}
	assert(n == b.m);
	if(checkDebug(debugMatProd)) prlocf("mat * mat\n");

	// non-tiled case
	if(tiler.tileSize == tiler.m_size&& b.tiler.tileSize == b.tiler.m_size ) {
		return matrixProductResident(b,block,stream);
	}

	// tiles
	DMatrix<T> d_A, d_B, d_C;
	if(checkDebug(debugMatProd)) prlocf("with tiling\n");
	const Tiler<T>* btiler = nullptr;
	bool freeBtiler = false;
	if(tiler == b.tiler ) {
		//
		btiler = new Tiler<T>(b,true,tiler.gpuMask);
		freeBtiler = true;
	} else{
		btiler = &(b.tiler);
	}

	if(checkDebug(debugMatProd)) flprintf("creating res matrix tiler.getTileCount() %d, btiler->getTileCount() %d\n",tiler.getTileCount(), btiler->getTileCount());
	int tileCount = tiler.getTileCount();
	CuMatrix<T> res(m, b.n,b.n, m/tileCount,m/tileCount, Tiler<T>::gpuMaskFromCurrGpu( ),true, true);
	if(checkDebug(debugMatProd)) flprintf("mat.elements %p tile %p\n",res.elements, res.tiler.buff());

	// in case *this and b are the same matrix, can't simply re-use tilers because
	// they will be tiling in different directions and dev buffer contents will be different
	uint tileM, tileN;

	tiler.tileDims(tileM, tileN, tdRows);
	tileCount = MAX(tileCount,DIV_UP(m,tileM));
	assert(tileM * tileN * sizeof(T) <= btiler->tileSize);

	if (checkDebug(debugTiler)) {
		flprintf("this %p, &o %p, &res %p\n",this, &b, &res);
		b.printShortString("matrix b");
#ifndef __CUDA_ARCH__
		outln("b.tiler " << b.tiler);
		outln("btiler " << *btiler);
		outln("tiler " << tiler);
		outln("res.tiler " << res.tiler);
#endif
		flprintf("tileM %u tileN %u tileCount %d\n",tileM, tileN, tileCount);
		printShortString("this");
		res.printShortString("ret");

	}
	uint aroff = 0,acoff = 0;
	uint broff = 0,bcoff = 0;
	uint croff = 0,ccoff = 0;
	int lastGpu = 0;
	int gpuCount = tiler.countGpus();
	int orgDevice = ExecCaps::currDev();
	cudaStream_t* streams = nullptr;
	if(gpuCount > 1) {
		assert(!stream);
		cudaStream_t* streams = (cudaStream_t* ) malloc(gpuCount * sizeof(cudaStream_t));
		for(int i =0 ; i < gpuCount; i++) {
			lastGpu = tiler.nextGpu(lastGpu);
			ExecCaps_setDevice(lastGpu);
			cherr(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));
		}
	}

	lastGpu = tiler.nextGpu(0);
	for(int i =0; i < tileCount; i++ ) {
		if(checkDebug(debugMatProd)) flprintf("i %d  tileM %u, tileN %u,\n ",i, tileM, tileN);
		ExecCaps_setDevice(lastGpu); // do it here so this (i)tile, (j) b-tile and (i,j) res-tile all on same dev
		tiler.tileLike(d_A, aroff,acoff, tileM, tileN, i, tdRows, true, lastGpu, gpuCount > 1 ? streams[i] : stream);
		if(checkDebug(debugMatProd)) flprintf("d_A after tileLike d_A.m %u d_A.n %u aroff %u acoff %u\n", d_A.m, d_A.n, aroff, acoff);
		if( checkDebug(debugMatProd)) {
			flprintf("d_A with d_A.p %u\n\n",d_A.p);
			printArrayDiag(d_A.elements, d_A.p, 10);
			flprintf("d_A with res.tiler.m_p %u\n\n",res.tiler.m_p);
			printArrayDiag(d_A.elements, res.tiler.m_p, 10);
		}
		for(int j = 0; j < tileCount; j++) {
			if(checkDebug(debugMatProd)) flprintf("\n------->i,j (%d,%d)\n",i,j);
			btiler->tileLike(d_B, broff,bcoff, tileN,tileM,j, tdCols, true,lastGpu, gpuCount > 1 ? streams[i] : stream);
			if(checkDebug(debugMatProd)) flprintf("d_B after tileLike d_B.m %u d_B.n %u broff %u bcoff %u\n", d_B.m, d_B.n, broff,bcoff);
			if(i == j) {
				flprintf("d_B with d_B.p %u\n\n",d_B.p);
				printArrayDiag(d_B.elements, d_B.p, 10);
				flprintf("d_B with res.tiler.m_p %u\n\n",res.tiler.m_p);
				printArrayDiag(d_B.elements, res.tiler.m_p, 10);
				//printDevArrayDiagSpan(d_B.elements, res.tiler.m_p, d_C.m - 1, 10);
			}
			res.tiler.tile2D(d_C,  croff,ccoff, tileM, tileM, tileCount, tileCount, i, j, false,lastGpu, gpuCount > 1 ? streams[i] : stream);
			if(checkDebug(debugMatProd)) flprintf("d_C after tileLike d_C.m %u d_C.n %u d_C.p %u croff %u ccoff %u\n", d_C.m, d_C.n, d_C.p, croff,ccoff);
			// res has nTiles^2 tileM x tileM tiles
			#ifndef __CUDA_ARCH__
#ifdef CuMatrix_UseCublas
				if( g_useCublas) {
					if(checkDebug(debugMatProd)) prlocf("g_useCublas\n");
					cublasSgemm(d_C, d_A, d_B);
				}else {
#endif
					if(g_matrix_product_kernel) {
						if(checkDebug(debugMatProd)) prlocf("g_matrix_product_kernel\n");
						matrixProductKPtrL(d_C,g_matrix_product_kernel, d_A, d_B,  block,gpuCount > 1 ? streams[i] : stream);
					}else {
						if(checkDebug(debugMatProd)) prlocf("matrixProductL\n");
						matrixProductL(d_C, d_A, d_B, block, gpuCount > 1 ? streams[i] : stream);
					}
					if(i == j && checkDebug(debugMatProd)) {
						outln("diagsies of the result d-tile");
						if(i == 0 || true) {
							flprintf("d_C with res.tiler.m_p, %u\n\n",res.tiler.m_p);
							printArrayDiag(d_C.elements, res.tiler.m_p, 10);
							flprintf("d_C with d_C.p %u\n\n",d_C.p);
							printArrayDiag(d_C.elements, d_C.p, 40);
							//printArrayDiag(d_C.elements, res.tiler.m_p, d_C.m - 1);
							//printDevArrayDiagSpan(d_C.elements, d_C.p, d_C.m, 10);
						}else {
							printArrayDiag(d_C.elements, d_C.p, 10);
						}
					}
#ifdef CuMatrix_UseCublas
				}
#endif
				#else
				matrixProductL(d_C, d_A, d_B,  block,gpuCount > 1 ? streams[i] : stream);
			#endif
			//
			//printArrayDiag(d_C.elements, d_C.p, 10);
			res.tiler.syncTile(d_C, croff, ccoff,  gpuCount > 1 ? streams[i] : stream);
			if(i == j && checkDebug(debugMatProd)) {
				printColoArrayDiagNe(res.elements + croff* res.p + ccoff, res.p, d_C.m, res.elements[0]);
			}
		}
	}
	if(gpuCount > 1) {
		for(int i =0 ; i < gpuCount; i++) {
			cherr(cudaStreamDestroy(streams[i]));
		}
		free(streams);
	}
	res.invalidateDevice();
	if(orgDevice != ExecCaps::currDev()) {
		ExecCaps_setDevice(orgDevice);
	}
	if(freeBtiler) delete(btiler);

	if(checkDebug(debugMatProd)){
		res.printShortString("matrixProduct updated res to mod_device\n");
	}
	return res;
}

/*
 * Three strategies for  c = a . b
 * multi kernel:
 * 	split the a and c matrices into row slices (what this method does)
 * 	split the c matrix into column slices
 * looping kernel:
 * 	within the kernel(s) have each thread loop and move in the direction normal to adjacent memory locations
 * 	(columnwise for row-major matrices, the default)
 */
template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::mkMatrixProduct( const CuMatrix<T>& b, dim3* block, cudaStream_t stream ) const {
    assert(tiler.tileSize == tiler.m_size);
	uint blockY = block ? block->y : 16;
	ExecCaps* caps = ExecCaps::currCaps();
	CuMatrix<T> res(m, b.n,true, true);
	flprintf("entre, blockY %u\n",blockY);

	uint newM = (caps->maxGrid.y - 1) * blockY;
	if(checkDebug(debugMatProd)) flprintf("oldM %u, newM %u\n",res.m, newM);

	DMatrix<T> d_B;
	b.tile0(d_B, b.getLastMod() == mod_host );
	util<T>::prdm("d_B",d_B);

	int sections = DIV_UP(res.m,newM);
	CuMatrix<T> aRowSlice;
	//CuMatrix<T> bColSlice;
	CuMatrix<T> resSubmatrix;
	// todo omp version
	uint rowOffset;
	for(int i = 0; i  < sections; i++) {
		rowOffset = i * newM;
		flprintf("for slice %d, offset row %u, aRowSlice:  ", i, rowOffset);

		if(i == sections -1) {
			newM = res.m - rowOffset;
		}
		submatrix(aRowSlice, newM, n, rowOffset, 0);

		res.submatrix(resSubmatrix, newM, n, rowOffset, 0);

		DMatrix<T> d_A, d_Res;
		aRowSlice.tile0(d_A, lastMod == mod_host);

		resSubmatrix.tile0(d_Res, false);
#ifndef __CUDA_ARCH__
#ifdef CuMatrix_UseCublas
	if( g_useCublas) {
		if(checkDebug(debugMatProd)) prlocf("g_useCublas\n");
		cublasSgemm(d_Res, d_A, d_B);
	}else {
#endif
		if(g_matrix_product_kernel) {
			if(checkDebug(debugMatProd)) prlocf("g_matrix_product_kernel\n");
			matrixProductKPtrL(d_Res,g_matrix_product_kernel, d_A, d_B,  block);
		}else {
			if(checkDebug(debugMatProd)) prlocf("matrixProductL\n");
			matrixProductL(d_Res, d_A, d_B,  block);
		}
#ifdef CuMatrix_UseCublas
	}
#endif
#else
	matrixProductL(d_Res, d_A, d_B,  block);
#endif
	}

	res.invalidateHost();

	if(checkDebug(debugMatProd)){
		res.printShortString("matrixProduct updated res to mod_device\n");
	}
	return res;
}

template<typename T> __host__ CUDART_DEVICE bool CuMatrix<T>::operator==( const CuMatrix<T> o) const {
	if(checkDebug(debugRedux))prlocf("operator== enter\n");
	bool thisZero = CuMatrix<T>::size == 0;
	bool oZero = o.size == 0;
	if(this == &o || ( thisZero && oZero)) {
		if(checkDebug(debugRedux)) {
			prlocf("CuMatrix<T>::operator== comparing to zero mats\n");
			//b_util::dumpStack();
		}
		return true;
	}
	if( oZero || thisZero ) {
		return false;
	}

#ifdef  CuMatrix_Enable_KTS
	return combineReduce(equalsBinaryOp<T>(), andBinaryOp<T>(), o, true);
#else
	return combineReduce(Functory<T,equalsBinaryOp>::pinch(), Functory<T,andBinaryOp>::pinch(), o, true);
#endif
}

template<typename T> __host__ CUDART_DEVICE bool CuMatrix<T>::operator!=( const CuMatrix<T> o) const {
#ifdef  CuMatrix_Enable_KTS
	return !combineReduce(equalsBinaryOp<T>(), andBinaryOp<T>(), o, true);
#else
	return !combineReduce(Functory<T,equalsBinaryOp>::pinch(), Functory<T,andBinaryOp>::pinch(), o, true);
#endif
}

template<typename T> __host__ CUDART_DEVICE bool CuMatrix<T>::almostEq( const CuMatrix<T>& o, T epsilon) const {

	if(checkDebug(debugRedux)){
		prloc();
		printf(" epsilon %e\n", epsilon);
	}
	almostEqualsBinaryOp<T> op = Functory<T,almostEqualsBinaryOp>::pinch(epsilon);
	return combineReduce(op, Functory<T,andBinaryOp>::pinch(), o, true);
}
#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class CombineOp, template <typename> class ReduceOp>
__host__ CUDART_DEVICE T CuMatrix<T>::combineReduce(CuMatrix<T>& buffer, CombineOp<T> cop, ReduceOp<T> rop, const CuMatrix<T>& o,
		T start, cudaStream_t stream ) const
#else
template<typename T> template<int CopDim, int RopDim>
__host__ CUDART_DEVICE T CuMatrix<T>::combineReduce(CuMatrix<T>& buffer, BinaryOpF<T,CopDim> cop, BinaryOpF<T,RopDim> rop, const CuMatrix<T>& o,
		T start, cudaStream_t stream ) const
#endif
{
	assert(tiler.tileSize == tiler.m_size);
	DMatrix<T> d_A, d_B;
	tile0(d_A,true);
	o.tile0(d_B,true);
	T res = combineReduceL(buffer, d_A, d_B, cop, rop, start,stream);
	return res;
}

template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::sumSqrDiff( const CuMatrix<T>& o) const {
#ifdef  CuMatrix_Enable_KTS
	return combineReduce(diffSquaredBinaryOp<T>(), plusBinaryOp<T>(), o, 0);
#else
	return combineReduce(Functory<T,diffSquaredBinaryOp>::pinch(), Functory<T,plusBinaryOp>::pinch(), o, 0);
#endif
}

template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::sumSqrDiff( CuMatrix<T>& reductionBuffer, const CuMatrix<T>& o) const {
#ifdef  CuMatrix_Enable_KTS
	return combineReduce(reductionBuffer, diffSquaredBinaryOp<T>(), plusBinaryOp<T>(), o, 0);
#else
	return combineReduce(reductionBuffer, Functory<T,diffSquaredBinaryOp>::pinch(), Functory<T,plusBinaryOp>::pinch(), o, 0);
#endif
}

template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::accuracy( const CuMatrix<T>& o) const {
#ifdef  CuMatrix_Enable_KTS
	return combineReduce(equalsBinaryOp<T>(), plusBinaryOp<T>(), o, 0)/m;
#else
	return combineReduce(Functory<T,equalsBinaryOp>::pinch(), Functory<T,plusBinaryOp>::pinch(), o, 0)/m;
#endif
}

template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::accuracy( CuMatrix<T>& reductionBuffer, const CuMatrix<T>& o) const {
#ifdef  CuMatrix_Enable_KTS
	return combineReduce(reductionBuffer, equalsBinaryOp<T>(), plusBinaryOp<T>(), o, 0)/m;
#else
	return combineReduce(reductionBuffer, Functory<T,equalsBinaryOp>::pinch(), Functory<T,plusBinaryOp>::pinch(), o, 0)/m;
#endif
}



#include "CuMatrixInster.cu"
