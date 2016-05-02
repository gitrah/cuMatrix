/*
 * CuMatrixOps.cu
 *
 *      Author: reid
 */
#include "CuMatrix.h"
#include <typeinfo>

/////////////////////////////////////////////////////////////////////////
//
// operators
//
/////////////////////////////////////////////////////////////////////////

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator=(const CuMatrix<T> o) {
	if(checkDebug(debugMem)){
		printShortString("operator= from");
		o.printShortString("hrt") ;
	}
	if (this == &o) {
		return *this;
	}

#ifndef __CUDA_ARCH__
	if(checkDebug(debugCheckValid)) flprintf(" (%s) this %p; &o %p\n", b_util::caller().c_str(), this, &o);
#endif
	m = o.m;
	n = o.n;
	p = o.p;
	size = o.size;
	lastMod = o.lastMod;
	if(elements) {
		if(checkDebug(debugMem)) printf( "%p operator=(const CuMatrix<T> o) freeing h %p\n", this,elements );
#ifndef __CUDA_ARCH__
		CuMatrix<T>::getMgr().freeHost(*this);
#endif
	}
	if(tiler.hasDmemQ()) {
		if(checkDebug(debugMem)) printf( "%p operator=(const CuMatrix<T> o) freeing h %p\n", this,elements );
#ifndef __CUDA_ARCH__
		CuMatrix<T>::getMgr().freeTiles(*this);
#else
		tiler.free();
#endif
	}
	if(o.elements) {
		elements=o.elements;
#ifndef __CUDA_ARCH__
		if(checkDebug(debugMem))prlocf("adding host\n");

		getMgr().addHost(*this);
#endif
	}
	if(o.tiler.hasDmemQ()) {
		tiler.tileSize = o.tiler.tileSize;
		tiler.m_size = o.tiler.m_size;
		tiler.buffers = o.tiler.buffers;
		tiler.gpuMask = o.tiler.gpuMask;
		tiler.m_m = o.m;
		tiler.m_n = o.n;
		tiler.m_p = o.p;
		tiler.m_size = o.size;
		tiler.m_elems = o.elements;
#ifndef __CUDA_ARCH__
		if(checkDebug(debugMem))prlocf("adding host\n");
		CuMatrix<T>::getMgr().addTiles(&tiler);
#endif
	}
	CuMatrix<T>::freeTxp();
	return *this;
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator^(T o) const {
	return pow(o);
}

/*
template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator^(int o) const {
	return pow( static_cast<T>(  o));
}
*/

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator<(T o) const {
	ltUnaryOp<T> ltf = Functory<T,ltUnaryOp>::pinch(o);
	return unaryOp(ltf);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator<=(T o) const {
	lteUnaryOp<T> ltef = Functory<T,lteUnaryOp>::pinch(o);
	return unaryOp(ltef);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator>(T o) const {
	gtUnaryOp<T> gtf = Functory<T,gtUnaryOp>::pinch(o);
	return unaryOp(gtf);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator>=(T o) const {
	gteUnaryOp<T> gtef = Functory<T,gteUnaryOp>::pinch(o);
	return unaryOp(gtef);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator==(T o) const {
	eqUnaryOp<T> eqf = Functory<T,eqUnaryOp>::pinch(o);
	return unaryOp(eqf);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator+(T o) const {
	translationUnaryOp<T> txf = Functory<T,translationUnaryOp>::pinch(o);
	return unaryOp(txf);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator-(T o) const {
	translationUnaryOp<T> txf = Functory<T,translationUnaryOp>::pinch(-o);
	return unaryOp(txf);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator*(CuMatrix<T> o)  const {
	return matrixProduct(o);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator%(CuMatrix<T> o) const {
	return hadamardProduct(o);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator*(T o) const {
	scaleUnaryOp<T> sf = Functory<T,scaleUnaryOp>::pinch(o);
	return unaryOp(sf);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator/(T o) const {
	scaleUnaryOp<T> sf = Functory<T,scaleUnaryOp>::pinch(static_cast<T>( 1. / o));
	return unaryOp(sf);
}

template<typename T> __host__ CUDART_DEVICE  CuMatrix<T> CuMatrix<T>::operator|=( const CuMatrix<T> b) const {
	return rightConcatenate(b);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator/=( const CuMatrix<T> b) const {
	return bottomConcatenate(b);
}



#include "CuMatrixInster.cu"
