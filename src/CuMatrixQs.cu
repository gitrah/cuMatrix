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

template<typename T> bool CuMatrix<T>::biasedQ() {
	CuMatrix<T> col1;
	submatrix(col1, m, 1, 0,0);
	almostEqUnaryOp<T> eqOp = Functory<T,almostEqUnaryOp>::pinch((T)1, util<T>::epsilon());
	return col1.all(eqOp);
}

template<typename T> bool CuMatrix<T>::isBinaryCategoryMatrix() const {
#ifdef  CuMatrix_Enable_KTS
	return gloloReduce( oneOrZeroUnaryOp<T>(), andBinaryOp<T>(), true);
#else
	return gloloReduce( Functory<T,oneOrZeroUnaryOp>::pinch(), Functory<T,andBinaryOp>::pinch(), true);
#endif
}

#include "CuMatrixInster.cu"
