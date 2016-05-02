/*
 * CuMatrixQs.cu
 *
 *      Author: reid
 */
#include "CuMatrix.h"



template<typename T> bool CuMatrix<T>::zeroQ(T epsilon) {
	if( !zeroDimsQ() ) {
		outln("zeroQ !zeroDims");
		almostEqUnaryOp<T> op = Functory<T,almostEqUnaryOp>::pinch((T)0, epsilon);
		andBinaryOp<T> andOp =  Functory<T,andBinaryOp>::pinch();
		return gloloReduce(op, andOp, true);
	}
	return true;
}

template<typename T> __host__ __device__  bool CuMatrix<T>::biasedQ() const {
	//CuMatrix<T> col1;
	//submatrix(col1, m, 1, 0,0);
	if(checkDebug(debugCheckValid))flprintf("this sum() %d\n ", sum() );
	T* dptr = currBuffer();
	CuMatrix<T> col1 = columnMatrix(0);
#ifndef __CUDA_ARCH__
	if(checkDebug(debugCheckValid))outln("col1  " << col1.toShortString() );
#endif
	if(checkDebug(debugCheckValid))flprintf("col1 sum %d\n ",col1.sum() );
	almostEqUnaryOp<T> eqOp = Functory<T,almostEqUnaryOp>::pinch((T)1, util<T>::epsilon());
	const bool  ret = col1.all(eqOp);
#ifndef __CUDA_ARCH__
	if(checkDebug(debugCheckValid))outln(toShortString() << " is biased " << ret);
#else
	if(checkDebug(debugCheckValid))flprintf("mat %dx%dx%d with dbuff %p is biased %d\n", m,n,p,currBuffer(),ret);
#endif
	return ret;
}


template<typename T> bool CuMatrix<T>::isBinaryCategoryMatrix() const {
#ifdef  CuMatrix_Enable_KTS
	return gloloReduce( oneOrZeroUnaryOp<T>(), andBinaryOp<T>(), true);
#else
	return gloloReduce( Functory<T,oneOrZeroUnaryOp>::pinch(), Functory<T,andBinaryOp>::pinch(), true);
#endif
}

#include "CuMatrixInster.cu"
