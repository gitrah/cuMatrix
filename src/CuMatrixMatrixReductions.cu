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
	prlocf("combineReduceL enter");
	uint nP = d_M1.m * d_M1.n;
	uint threads;
	uint blocks;
	getReductionExecContext(blocks,threads, nP);
	if(checkDebug(debugRedux)){
		flprintf("combineReduceL(const DMatrix<T>& d_M) blocks %u threads %u nP %u\n", blocks, threads, nP);
	}
	CuMatrix<T> res(blocks, 1,true,true);
	DMatrix<T> d_Res;
	res.asDmatrix(d_Res, false);
	T total = combineReduceOpLauncher(res.d_elements, d_M1.elements, d_M2.elements, nP, cop, rop, start, stream);
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
	uint nP = d_M1.m * d_M1.n;
	T total = combineReduceOpLauncher(buffer.d_elements, d_M1.elements, d_M2.elements, nP, cop, rop, start, stream);
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
	asDmatrix(d_A);
	o.asDmatrix(d_B);
	T res = combineReduceL(d_A, d_B, cop, rop, start, stream);
	return res;
}

template<typename T> void cublasSgemm(DMatrix<T> d_C, const DMatrix<T> d_A, const DMatrix<T> d_B) {

	cublasStatus_t ret;
	if(sizeof(T) < 8) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
		ret = cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_B.n, d_A.m, d_A.n, &alpha, (const float*)d_B.elements, d_B.p, (const float*)d_A.elements, d_A.p, &beta, (float*)d_C.elements, d_C.p);
	} else {
        const double alpha = 1.0;
        const double beta  = 0.0;
		ret = cublasDgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_B.n, d_A.m, d_A.n, &alpha, (const double*)d_B.elements, d_B.p, (const double*)d_A.elements, d_A.p, &beta, (double*)d_C.elements, d_C.p);
	}
	if (ret != CUBLAS_STATUS_SUCCESS)
	{
	   printf("cublasDgemm returned error code %d, line(%d)\n", ret, __LINE__);
	   exit(EXIT_FAILURE);
	}
}


template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::matrixProduct( const CuMatrix<T>& b, dim3* block, cudaStream_t stream ) const {
	// todo convert mats into big warp div matrix and one < blocksize matrix
	if(b.scalarQ()) {
		return operator *(b.get(0));
	} else if(scalarQ()) {
		return b.operator *(get(0));
	} else if(vectorQ() && b.vectorQ()) {
		// better as a reduction
		if( !(rowVectorQ() && b.rowVectorQ()) && !(columnVectorQ() && b.columnVectorQ())) {
			if(n == 1) {
				if(checkDebug(debugMatProd)) printShortString(" posing as row");
			} else {
				if(checkDebug(debugMatProd)) printShortString(" posing as row");
			}
		}

		if(checkDebug(debugMatProd))
			prlocf("cuplavects\n");
#ifdef  CuMatrix_Enable_KTS
		T ret = combineReduce(multBinaryOp<T>(), plusBinaryOp<T>(), b, 0, stream);
#else
		T ret = combineReduce(Functory<T,multBinaryOp>::pinch(), Functory<T,plusBinaryOp>::pinch(), b, 0, stream);
#endif
		if(checkDebug(debugMatProd))
			flprintf("matProd expecting a vector %f\n",ret);
		return fromScalar(ret);
	}
	if(checkDebug(debugMatProd)) prlocf("mat * mat\n");
	CuMatrix<T> res(m, b.n,false, true);
	if(checkDebug(debugMatProd)) {
		prlocf("a(this) dims" );printShortString();
		prlocf("b dims "); b.printShortString();
		prlocf("res.dims "); res.printShortString( );
		flprintf("matrix product this.lastMod %b, o.lastMod %b\n", lastMod , b.lastMod );
	}
	DMatrix<T> d_A, d_B, d_C;
	asDmatrix(d_A);
	b.asDmatrix(d_B);
	res.asDmatrix(d_C, false);
#ifndef __CUDA_ARCH__
	if( g_useCublas) {
		if(checkDebug(debugMatProd)) prlocf("g_useCublas\n");
		cublasSgemm(d_C, d_A, d_B);
	}else {
		if(g_matrix_product_kernel) {
			if(checkDebug(debugMatProd)) prlocf("g_matrix_product_kernel\n");
			matrixProductKPtrL(d_C,g_matrix_product_kernel, d_A, d_B,  block);
		}else {
			if(checkDebug(debugMatProd)) prlocf("matrixProductL\n");
			matrixProductL(d_C, d_A, d_B,  block);
		}
	}
#else
	matrixProductL(d_C, d_A, d_B,  block);
#endif
	res.invalidateHost();

	if(checkDebug(debugMatProd)){
		res.printShortString("matrixProduct updated res to mod_device\n");
	}
	return res;
}

template<typename T> __host__ CUDART_DEVICE bool CuMatrix<T>::operator==( const CuMatrix<T> o) const {
	prlocf("operator== enter\n");
	bool thisZero = CuMatrix<T>::size == 0;
	bool oZero = o.size == 0;
	if(this == &o || ( thisZero && oZero)) {
		if(checkDebug(debugMatProd)) {
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

	prloc(); printf(" epsilon %e\n", epsilon);
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
	DMatrix<T> d_A, d_B;
	asDmatrix(d_A);
	o.asDmatrix(d_B);
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
