/*
 * CuMatrixUnaryOps.cu
 *
 *      Author: reid
 */
#include "CuMatrix.h"
#include "caps.h"
#include "Kernels.h"

template<typename T> CuMatrix<T> CuMatrix<T>::negate() const {
	return unaryOp(Functory<T,negateUnaryOp>::pinch());
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class UnaryOp> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::unaryOp(UnaryOp<T> op, cudaStream_t stream ) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::unaryOp(UnaryOpF<T,StateDim> op, cudaStream_t stream ) const
#endif
{
	CuMatrix<T> res(m, n, true, true);
	if(checkDebug(debugUnaryOp)) {
		prlocf("in unaryOp(UnaryOp,...)\n");
		printShortString("unary op, src");
		res.printShortString("unary op, targ");
	}
	unaryOp(res, op, stream);
	return res;
}
#ifdef  CuMatrix_Enable_KTS

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<absUnaryOp>(absUnaryOp<float>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp<absUnaryOp>(absUnaryOp<int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<uint> CuMatrix<uint>::unaryOp<absUnaryOp>(absUnaryOp<uint>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp<absUnaryOp>(absUnaryOp<ulong>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp<absUnaryOp>(absUnaryOp<long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<absUnaryOp>(absUnaryOp<double>, CUstream_st*) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<powUnaryOp>(powUnaryOp<float>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp<powUnaryOp>(powUnaryOp<int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<uint> CuMatrix<uint>::unaryOp<powUnaryOp>(powUnaryOp<uint>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp<powUnaryOp>(powUnaryOp<ulong>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp<powUnaryOp>(powUnaryOp<long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<powUnaryOp>(powUnaryOp<double>, CUstream_st*) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<expUnaryOp>(expUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<expUnaryOp>(expUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<translationUnaryOp>(translationUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<translationUnaryOp>(translationUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp<translationUnaryOp>(translationUnaryOp<ulong>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<scaleUnaryOp>(scaleUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<scaleUnaryOp>(scaleUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp<scaleUnaryOp>(scaleUnaryOp<ulong>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<subFromUnaryOp>(subFromUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<subFromUnaryOp>(subFromUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp<subFromUnaryOp>(subFromUnaryOp<ulong>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<negateUnaryOp>(negateUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<negateUnaryOp>(negateUnaryOp<double>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<sigmoidUnaryOp>(sigmoidUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<sigmoidUnaryOp>(sigmoidUnaryOp<double>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<sigmoidGradientUnaryOp>(sigmoidGradientUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<sigmoidGradientUnaryOp>(sigmoidGradientUnaryOp<double>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<logUnaryOp>(logUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<logUnaryOp>(logUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<oneOverUnaryOp>(oneOverUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<oneOverUnaryOp>(oneOverUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<sqrtUnaryOp>(sqrtUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<sqrtUnaryOp>(sqrtUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<sqrUnaryOp>(sqrUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<sqrUnaryOp>(sqrUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<divSqrtUnaryOp>(divSqrtUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<divSqrtUnaryOp>(divSqrtUnaryOp<double>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<ltUnaryOp>(ltUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<ltUnaryOp>(ltUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp<ltUnaryOp>(ltUnaryOp<ulong>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<lteUnaryOp>(lteUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<lteUnaryOp>(lteUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp<lteUnaryOp>(lteUnaryOp<ulong>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<gtUnaryOp>(gtUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<gtUnaryOp>(gtUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp<gtUnaryOp>(gtUnaryOp<ulong>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<gteUnaryOp>(gteUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<gteUnaryOp>(gteUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp<gteUnaryOp>(gteUnaryOp<ulong>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp<eqUnaryOp>(eqUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp<eqUnaryOp>(eqUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp<eqUnaryOp>(eqUnaryOp<ulong>, cudaStream_t) const;

template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp<subFromUnaryOp>(subFromUnaryOp<int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned int> CuMatrix<unsigned int>::unaryOp<subFromUnaryOp>(subFromUnaryOp<unsigned int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp<ltUnaryOp>(ltUnaryOp<int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp<lteUnaryOp>(lteUnaryOp<int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp<gtUnaryOp>(gtUnaryOp<int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp<gteUnaryOp>(gteUnaryOp<int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp<eqUnaryOp>(eqUnaryOp<int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp<translationUnaryOp>(translationUnaryOp<int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp<scaleUnaryOp>(scaleUnaryOp<int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned int> CuMatrix<unsigned int>::unaryOp<ltUnaryOp>(ltUnaryOp<unsigned int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned int> CuMatrix<unsigned int>::unaryOp<lteUnaryOp>(lteUnaryOp<unsigned int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned int> CuMatrix<unsigned int>::unaryOp<gtUnaryOp>(gtUnaryOp<unsigned int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned int> CuMatrix<unsigned int>::unaryOp<gteUnaryOp>(gteUnaryOp<unsigned int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned int> CuMatrix<unsigned int>::unaryOp<eqUnaryOp>(eqUnaryOp<unsigned int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned int> CuMatrix<unsigned int>::unaryOp<translationUnaryOp>(translationUnaryOp<unsigned int>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned int> CuMatrix<unsigned int>::unaryOp<scaleUnaryOp>(scaleUnaryOp<unsigned int>, CUstream_st*) const;

template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp<scaleUnaryOp>(scaleUnaryOp<long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp<gteUnaryOp>(gteUnaryOp<long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp<translationUnaryOp>(translationUnaryOp<long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp<lteUnaryOp>(lteUnaryOp<long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp<ltUnaryOp>(ltUnaryOp<long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp<eqUnaryOp>(eqUnaryOp<long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp<gtUnaryOp>(gtUnaryOp<long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp<subFromUnaryOp>(subFromUnaryOp<long>, CUstream_st*) const;

#else
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp(UnaryOpF<float,0>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp(UnaryOpF<double,0>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp(UnaryOpF<long,0>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp(UnaryOpF<ulong,0>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp(UnaryOpF<int,0>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<uint> CuMatrix<uint>::unaryOp(UnaryOpF<uint,0>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp(UnaryOpF<float,1>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp(UnaryOpF<double,1>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<long> CuMatrix<long>::unaryOp(UnaryOpF<long,1>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp(UnaryOpF<ulong,1>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<int> CuMatrix<int>::unaryOp(UnaryOpF<int,1>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<uint> CuMatrix<uint>::unaryOp(UnaryOpF<uint,1>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::unaryOp(UnaryOpF<float,2>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::unaryOp(UnaryOpF<double,2>, cudaStream_t) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::unaryOp(UnaryOpF<ulong,2>, cudaStream_t) const;
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class UnaryOp> __host__ CUDART_DEVICE void CuMatrix<T>::unaryOp(CuMatrix<T>& res, UnaryOp<T> op, cudaStream_t stream) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE void CuMatrix<T>::unaryOp(CuMatrix<T>& res, UnaryOpF<T,StateDim> op, cudaStream_t stream) const
#endif
{
/*
	if(checkDebug(debugUnaryOp)) {
		flprintf("unaryOp tileCount %d lastMod %s\n", tiler.getTileCount(), b_util::modStr(lastMod));
	}
*/

	uint tileM, tileN, roff, coff;
	tiler.tileDims(tileM, tileN, tdRows);
	int tileCount = DIV_UP(m,tileM);
	DMatrix<T> d_A, d_Res;
	int lastGpu = ExecCaps::currDev();
	for(int i = 0; i < tileCount; i++) {
		if(checkDebug(debugFill))flprintf("tileM %d tileN %d tile %d lastGpu %u\n", tileM, tileN, i, lastGpu);
		if(checkDebug(debugFill))flprintf("roff %u coff %u\n",roff, coff);
		tiler.tileLike(d_A, roff, coff, tileM, tileN, i, tdRows, lastMod == mod_host, lastGpu, stream);
		if(checkDebug(debugFill))flprintf("after tiler.tileLike for tile %d; roff %u coff %u\n", i, roff, coff);
		lastGpu = res.tiler.tileLike(d_Res, roff, coff, tileM, tileN, i, tdRows, false,lastGpu, stream);
		if(checkDebug(debugFill))flprintf("after res.tiler.tileLike for tile %d; roff %u coff %u lastGpu %d\n", i, roff, coff, lastGpu);
		if(p == n) {
			unaryOpL( d_Res, d_A, op,stream);
		} else {
			if(checkDebug(debugUnaryOp)) {
				printf("invoking DMatrix version of unaryOp\n");
			}
			unaryOpDmL(d_Res, d_A, op, DefaultWidth2Height , stream);
		}
		res.tiler.syncTile(d_Res, roff, coff, stream);
	}
	if(checkDebug(debugUnaryOp)) {
		printDevArray(d_Res.elements,"d_Res",-1, MIN(10, m*n));
		printColoArray(res.elements,MIN(10, m*n));
	}

	//res.invalidateHost();
	res.lastMod = (tileCount>1) ? mod_host : mod_synced;
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void CuMatrix<float>::unaryOp<approxInvSqrtUnaryOp>(CuMatrix<float>&, approxInvSqrtUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE void CuMatrix<double>::unaryOp<approxInvSqrtUnaryOp>(CuMatrix<double>&, approxInvSqrtUnaryOp<double>, cudaStream_t) const;
template __host__ CUDART_DEVICE void CuMatrix<float>::unaryOp<slowInvSqrtUnaryOp>(CuMatrix<float>&, slowInvSqrtUnaryOp<float>, cudaStream_t) const;
template __host__ CUDART_DEVICE void CuMatrix<double>::unaryOp<slowInvSqrtUnaryOp>(CuMatrix<double>&, slowInvSqrtUnaryOp<double>, cudaStream_t) const;
#else
template __host__ CUDART_DEVICE void CuMatrix<float>::unaryOp(CuMatrix<float>&, UnaryOpF<float,0>, cudaStream_t) const;
template __host__ CUDART_DEVICE void CuMatrix<double>::unaryOp(CuMatrix<double>&, UnaryOpF<double,0>, cudaStream_t) const;
#endif

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::sigmoid() const {
	return unaryOp(Functory<T,sigmoidUnaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::sigmoidGradient() const {
	return unaryOp(Functory<T,sigmoidGradientUnaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::log() const {
	return unaryOp(Functory<T,logUnaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::ceil() const {
	return unaryOp(Functory<T,ceilUnaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::floor() const {
	return unaryOp(Functory<T,floorUnaryOp>::pinch());
}


template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::oneOver() const {
	return unaryOp(Functory<T,oneOverUnaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::setAll(int val) {
	assert(tiler.tileSize == tiler.m_size);
#ifndef __CUDA_ARCH__
	checkCudaErrors(cudaMemset( tiler.currBuffer(), val, size));
#else
	memset(tiler.currBuffer(), val, size);
#endif
	lastMod = mod_device;
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::exp() const {
	return unaryOp(Functory<T,expUnaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::sqrt() const {
	return unaryOp(Functory<T,sqrtUnaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::sqr() const {
	return unaryOp(Functory<T,sqrUnaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::pow(T o) const {
	powUnaryOp<T> pf = Functory<T,powUnaryOp>::pinch(o);
	return unaryOp(pf);
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::divSqrt(T divisor) const {
	divSqrtUnaryOp<T> dsf = Functory<T,divSqrtUnaryOp>::pinch(divisor);
	return unaryOp(dsf);
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE bool CuMatrix<T>::all(BoolUnaryOp<T> op) const
{
	return gloloReduce(op, andBinaryOp<T>(), true);
}
#else
template<typename T> template <int StateDim> __host__ CUDART_DEVICE bool CuMatrix<T>::all(UnaryOpF<T,StateDim> op) const
{
	return gloloReduce(op, Functory<T, andBinaryOp>::pinch(), true);
}
#endif
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE bool CuMatrix<float>::all<almostEqUnaryOp>(almostEqUnaryOp<float>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::all<almostEqUnaryOp>(almostEqUnaryOp<double>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::all<almostEqUnaryOp>(almostEqUnaryOp<ulong>) const;
template __host__ CUDART_DEVICE bool CuMatrix<float>::all<ltUnaryOp>(ltUnaryOp<float>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::all<ltUnaryOp>(ltUnaryOp<double>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::all<ltUnaryOp>(ltUnaryOp<ulong>) const;
template __host__ CUDART_DEVICE bool CuMatrix<int>::all<almostEqUnaryOp>(almostEqUnaryOp<int>) const;
template __host__ CUDART_DEVICE bool CuMatrix<unsigned int>::all<almostEqUnaryOp>(almostEqUnaryOp<unsigned int>) const;
#else
template __host__ CUDART_DEVICE bool CuMatrix<float>::all<1>(UnaryOpF<float,1>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::all<1>(UnaryOpF<double,1>) const;
template __host__ CUDART_DEVICE bool CuMatrix<long>::all<1>(UnaryOpF<long,1>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::all<1>(UnaryOpF<ulong,1>) const;
template __host__ CUDART_DEVICE bool CuMatrix<float>::all<2>(UnaryOpF<float,2>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::all<2>(UnaryOpF<double,2>) const;
template __host__ CUDART_DEVICE bool CuMatrix<int>::all<2>(UnaryOpF<int,2>) const;
template __host__ CUDART_DEVICE bool CuMatrix<uint>::all<2>(UnaryOpF<uint,2>) const;
template __host__ CUDART_DEVICE bool CuMatrix<long>::all<2>(UnaryOpF<long,2>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::all<2>(UnaryOpF<ulong,2>) const;
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE bool CuMatrix<T>::any(BoolUnaryOp<T> op) const
{
	return gloloReduce(op, orBinaryOp<T>(), false);
}
#else
template<typename T> template <int StateDim> __host__ CUDART_DEVICE bool CuMatrix<T>::any(UnaryOpF<T,StateDim> op) const
{
	return gloloReduce(op, Functory<T,orBinaryOp>::pinch(), false);
}
#endif
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE bool CuMatrix<float>::any<almostEqUnaryOp>(almostEqUnaryOp<float>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::any<almostEqUnaryOp>(almostEqUnaryOp<double>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::any<almostEqUnaryOp>(almostEqUnaryOp<ulong>) const;
template __host__ CUDART_DEVICE bool CuMatrix<float>::any<ltUnaryOp>(ltUnaryOp<float>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::any<ltUnaryOp>(ltUnaryOp<double>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::any<ltUnaryOp>(ltUnaryOp<ulong>) const;
#else
template __host__ CUDART_DEVICE bool CuMatrix<float>::any<1>(UnaryOpF<float,1>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::any<1>(UnaryOpF<double,1>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::any<1>(UnaryOpF<ulong,1>) const;
template __host__ CUDART_DEVICE bool CuMatrix<float>::any<2>(UnaryOpF<float,2>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::any<2>(UnaryOpF<double,2>) const;
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE bool CuMatrix<T>::none(	BoolUnaryOp<T> fn) const
#else
template<typename T> template <int StateDim> __host__ CUDART_DEVICE bool CuMatrix<T>::none(	UnaryOpF<T,StateDim> fn) const
#endif
{
	return !any(fn);
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE bool CuMatrix<float>::none<almostEqUnaryOp>(almostEqUnaryOp<float>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::none<almostEqUnaryOp>(almostEqUnaryOp<double>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::none<almostEqUnaryOp>(almostEqUnaryOp<ulong>) const;
template __host__ CUDART_DEVICE bool CuMatrix<float>::none<ltUnaryOp>(ltUnaryOp<float>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::none<ltUnaryOp>(ltUnaryOp<double>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::none<ltUnaryOp>(ltUnaryOp<ulong>) const;
template __host__ CUDART_DEVICE bool CuMatrix<float>::none<gtUnaryOp>(gtUnaryOp<float>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::none<gtUnaryOp>(gtUnaryOp<double>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::none<gtUnaryOp>(gtUnaryOp<ulong>) const;
#else
template __host__ CUDART_DEVICE bool CuMatrix<float>::none<1>(UnaryOpF<float,1>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::none<1>(UnaryOpF<double,1>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::none<1>(UnaryOpF<ulong,1>) const;
template __host__ CUDART_DEVICE bool CuMatrix<float>::none<2>(UnaryOpF<float,2>) const;
template __host__ CUDART_DEVICE bool CuMatrix<double>::none<2>(UnaryOpF<double,2>) const;
template __host__ CUDART_DEVICE bool CuMatrix<ulong>::none<2>(UnaryOpF<ulong,2>) const;
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE long CuMatrix<T>::count(BoolUnaryOp<T> fn) const
{
	return gloloReduce(fn, plusBinaryOp<T>(), 0);
}
#else
template<typename T> template <int StateDim> __host__ CUDART_DEVICE long CuMatrix<T>::count( UnaryOpF<T,StateDim> fn) const
{
	return gloloReduce(fn, Functory<T, plusBinaryOp>::pinch(), 0);
}
#endif
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE long CuMatrix<float>::count<almostEqUnaryOp>(almostEqUnaryOp<float>) const;
template __host__ CUDART_DEVICE long CuMatrix<double>::count<almostEqUnaryOp>(almostEqUnaryOp<double>) const;
template __host__ CUDART_DEVICE long CuMatrix<ulong>::count<almostEqUnaryOp>(almostEqUnaryOp<ulong>) const;
template __host__ CUDART_DEVICE long CuMatrix<float>::count<notAlmostEqUnaryOp>(notAlmostEqUnaryOp<float>) const;
template __host__ CUDART_DEVICE long CuMatrix<double>::count<notAlmostEqUnaryOp>(notAlmostEqUnaryOp<double>) const;
template __host__ CUDART_DEVICE long CuMatrix<ulong>::count<notAlmostEqUnaryOp>(notAlmostEqUnaryOp<ulong>) const;
template __host__ CUDART_DEVICE long CuMatrix<float>::count<neqUnaryOp>(neqUnaryOp<float>) const;
template __host__ CUDART_DEVICE long CuMatrix<double>::count<neqUnaryOp>(neqUnaryOp<double>) const;
template __host__ CUDART_DEVICE long CuMatrix<ulong>::count<neqUnaryOp>(neqUnaryOp<ulong>) const;

template __host__ CUDART_DEVICE long CuMatrix<float>::count<ltUnaryOp>(ltUnaryOp<float>) const;
template __host__ CUDART_DEVICE long CuMatrix<double>::count<ltUnaryOp>(ltUnaryOp<double>) const;
template __host__ CUDART_DEVICE long CuMatrix<ulong>::count<ltUnaryOp>(ltUnaryOp<ulong>) const;
#else
template __host__ CUDART_DEVICE long CuMatrix<float>::count<1>(UnaryOpF<float,1>) const;
template __host__ CUDART_DEVICE long CuMatrix<double>::count<1>(UnaryOpF<double,1>) const;
template __host__ CUDART_DEVICE long CuMatrix<ulong>::count<1>(UnaryOpF<ulong,1>) const;
template __host__ CUDART_DEVICE long CuMatrix<float>::count<2>(UnaryOpF<float,2>) const;
template __host__ CUDART_DEVICE long CuMatrix<double>::count<2>(UnaryOpF<double,2>) const;
template __host__ CUDART_DEVICE long CuMatrix<ulong>::count<2>(UnaryOpF<ulong,2>) const;
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE
IndexArray CuMatrix<T>::find(BoolUnaryOp<T> fn) const
#else
template<typename T> template <int StateDim> __host__ CUDART_DEVICE IndexArray CuMatrix<T>::find( UnaryOpF<T,StateDim> fn) const
#endif
{
	CuMatrix<T> m = unaryOp(fn);

	uint len = m.size/sizeof(T);
	int arraySize = 10;
	if(checkDebug(debugUnaryOp)) prlocf("creating intial idx array\n");
	uint* arry, *temp;
#ifdef __CUDA_ARCH__
	cherr(cudaMalloc(&arry, arraySize * sizeof(uint)));
#else
	checkCudaError(cudaHostAlloc(&arry, arraySize * sizeof(uint),0));
#endif
	int currIdx = 0;
	for(int i =0; i < len; i++ ) {
		if(m.get(i)) {
			flprintf("adding idx %d\n", i);
			arry[currIdx++] = i;
			if(currIdx == arraySize) {
				arraySize *= 2;
#ifdef __CUDA_ARCH__
	cherr(cudaMalloc(&temp, arraySize * sizeof(uint)));
	cherr(cudaMemcpyAsync(temp, arry, (currIdx -1) * sizeof(uint), cudaMemcpyDeviceToDevice));
	cherr(cudaDeviceSynchronize());
	cherr(cudaFree(arry));
	arry = temp;
#else
	checkCudaError(cudaHostAlloc(&temp, arraySize * sizeof(uint),0));
	cherr(cudaMemcpy(temp, arry, (currIdx -1) * sizeof(uint), cudaMemcpyHostToHost));
	if(checkDebug(debugDestr))flprintf("freeing host arry %p\n", arry);
	cherr(cudaFreeHost(arry));
	arry = temp;
#endif
			}
		} else {
			if(checkDebug(debugUnaryOp)) flprintf("skipping idx %d\n", i);
		}
	}
	if(currIdx < arraySize) {
		if(checkDebug(debugUnaryOp)) flprintf("shrinking idx array from %d to %d\n", arraySize, currIdx);
	}
	//arry = (uint*) realloc(arry, arraySize);
#ifdef __CUDA_ARCH__
	cherr(cudaMalloc(&temp, arraySize * sizeof(uint)));
	cherr(cudaMemcpyAsync(temp, arry, (currIdx -1)* sizeof(uint), cudaMemcpyDeviceToDevice));
	cherr(cudaDeviceSynchronize());
	cherr(cudaFree(arry));
	arry = temp;
#else
	checkCudaError(cudaHostAlloc(&temp, arraySize * sizeof(uint),0));
	cherr(cudaMemcpy(temp, arry, (currIdx -1) * sizeof(uint), cudaMemcpyHostToHost));
	if(checkDebug(debugDestr))flprintf("freeing host arry %p\n", arry);
	cherr(cudaFreeHost(arry));
	arry = temp;
#endif
	return IndexArray(arry, currIdx);
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE IndexArray CuMatrix<float>::find<ltUnaryOp>(ltUnaryOp<float>) const;
template __host__ CUDART_DEVICE IndexArray CuMatrix<double>::find<ltUnaryOp>(ltUnaryOp<double>) const;
#else
template __host__ CUDART_DEVICE IndexArray CuMatrix<float>::find<1>(UnaryOpF<float,1>) const;
template __host__ CUDART_DEVICE IndexArray CuMatrix<double>::find<1>(UnaryOpF<double,1>) const;
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE
void CuMatrix<T>::findFirstN(IndexArray arry, BoolUnaryOp<T> op) const
#else
template<typename T> template <int StateDim> __host__ CUDART_DEVICE void CuMatrix<T>::findFirstN( IndexArray arry, UnaryOpF<T,StateDim> op) const
#endif
{
	CuMatrix<T> m = unaryOp(op);
	m.syncBuffers();

	uint len = m.size/sizeof(T);
	int currIdx = 0;
	for(int i =0; i < len; i++ ) {
		if(i == len -1 ){
			if(checkDebug(debugUnaryOp)) flprintf("lastIdx %d (+tiler.currBuffer() = %p)\n", i, i + tiler.currBuffer());
		}
		if(m.elements[i]) {
			if(checkDebug(debugUnaryOp)) flprintf("adding idx %d\n", i);
			if(currIdx < arry.count) {
				arry.indices[currIdx++] = i;
			} else {
				if(checkDebug(debugUnaryOp)) prlocf("exceeded capacity of indexarry; stopping\n");
				return;
			}
		} else {
		//	if(checkDebug(debugUnaryOp)) flprintf("skipping idx %d\n", i);
		}
	}
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void CuMatrix<float>::findFirstN<ltUnaryOp>(IndexArray , ltUnaryOp<float>) const;
template __host__ CUDART_DEVICE void CuMatrix<double>::findFirstN<ltUnaryOp>(IndexArray, ltUnaryOp<double>) const;
template __host__ CUDART_DEVICE void CuMatrix<float>::findFirstN<gtUnaryOp>(IndexArray , gtUnaryOp<float>) const;
template __host__ CUDART_DEVICE void CuMatrix<double>::findFirstN<gtUnaryOp>(IndexArray, gtUnaryOp<double>) const;
template __host__ CUDART_DEVICE void CuMatrix<float>::findFirstN<notAlmostEqUnaryOp>(IndexArray , notAlmostEqUnaryOp<float>) const;
template __host__ CUDART_DEVICE void CuMatrix<double>::findFirstN<notAlmostEqUnaryOp>(IndexArray, notAlmostEqUnaryOp<double>) const;
template __host__ CUDART_DEVICE void CuMatrix<float>::findFirstN<neqUnaryOp>(IndexArray , neqUnaryOp<float>) const;
template __host__ CUDART_DEVICE void CuMatrix<double>::findFirstN<neqUnaryOp>(IndexArray, neqUnaryOp<double>) const;
template __host__ CUDART_DEVICE void CuMatrix<unsigned long>::findFirstN<ltUnaryOp>(IndexArray, ltUnaryOp<unsigned long>) const;
#else
template __host__ CUDART_DEVICE void CuMatrix<float>::findFirstN<1>(IndexArray , UnaryOpF<float,1>) const;
template __host__ CUDART_DEVICE void CuMatrix<ulong>::findFirstN<1>(IndexArray , UnaryOpF<ulong,1>) const;
template __host__ CUDART_DEVICE void CuMatrix<double>::findFirstN<1>(IndexArray, UnaryOpF<double,1>) const;
template __host__ CUDART_DEVICE void CuMatrix<float>::findFirstN<2>(IndexArray , UnaryOpF<float,2>) const;
template __host__ CUDART_DEVICE void CuMatrix<double>::findFirstN<2>(IndexArray, UnaryOpF<double,2>) const;
template __host__ CUDART_DEVICE void CuMatrix<ulong>::findFirstN<2>(IndexArray, UnaryOpF<ulong,2>) const;
#endif

#include "CuMatrixInster.cu"
