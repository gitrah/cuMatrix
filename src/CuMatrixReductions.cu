/*
 * CuMatrixReductions.cu
 *
 *	basic reduction kernels, exec ctx
 *      Author: reid
 */
#include "CuMatrix.h"
#include "Kernels.h"
#include "caps.h"
#include "util.h"
#include "CuDefs.h"


#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE
T CuMatrix<T>::reduce(const DMatrix<T>& d_M, BinaryOp<T> op, T start, cudaStream_t stream  )
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE
T CuMatrix<T>::reduce(const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, cudaStream_t stream  )
#endif
{
	uint nP = d_M.m * d_M.n;
	uint threads;
	uint blocks;
	::getReductionExecContext(blocks, threads, nP);
	if(checkDebug(debugRedux | debugNoRedux))flprintf("CuMatrix<T>::reduce blocks %d threads %d np %d\n", blocks,threads, nP);
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaPeekAtLastError());
#endif
	if(checkDebug( debugNoRedux)) {
		prlocf("early exit\n");
		return start;
	}
	CuMatrix<T> res(blocks, 1, false, true);
	if(checkDebug(debugRedux)) {
		prlocf("res ");
		res.printShortString();
	}
	DMatrix<T> d_Res;
	res.asDmatrix(d_Res, false);

	if(checkDebug(debugRedux| debugNoRedux)) {
		prlocf("after res.asDmatrix(..)\n");
	}

#ifndef __CUDA_ARCH__
	checkCudaError(cudaDeviceSynchronize());
	if(checkDebug(debugRedux| debugNoRedux)){ prlocf("host ");}
#else
	if(checkDebug(debugRedux| debugNoRedux)){ prlocf("dev ");}
#endif
	T total = 0;
	if(checkDebug(debugRedux| debugNoRedux)){ flprintf("&total %p\n",&total); }
	if(!checkDebug(debugNoRedux)) {
		reduceLauncher(&total, d_Res, nP, d_M, op, start, 1, 0, stream);
	} else {
		prlocf("skipping reducelauncher\n");
	}
#ifndef __CUDA_ARCH__
	cherr(cudaPeekAtLastError());
	cherr(cudaStreamSynchronize(stream));
#else
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaPeekAtLastError());
#endif
	__syncthreads();
#endif
	if(checkDebug(debugRedux))flprintf("total now %f\n",total);
	return total;
}
#ifdef  CuMatrix_Enable_KTS

template __host__ CUDART_DEVICE float CuMatrix<float>::reduce<maxBinaryOp>(DMatrix<float> const&, maxBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE float CuMatrix<float>::reduce<minBinaryOp>(DMatrix<float> const&, minBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce<maxBinaryOp>(DMatrix<double> const&, maxBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce<minBinaryOp>(DMatrix<double> const&, minBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce<minBinaryOp>(DMatrix<ulong> const&, minBinaryOp<ulong>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce<maxBinaryOp>(DMatrix<ulong> const&, maxBinaryOp<ulong>, ulong, CUstream_st*);

template  __host__ CUDART_DEVICE float CuMatrix<float>::reduce<plusBinaryOp>(DMatrix<float> const&, plusBinaryOp<float>, float, CUstream_st*);
template  __host__ CUDART_DEVICE double CuMatrix<double>::reduce<plusBinaryOp>(DMatrix<double> const&, plusBinaryOp<double>, double, CUstream_st*);
template  __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce<plusBinaryOp>(DMatrix<ulong> const&, plusBinaryOp<ulong>, ulong, CUstream_st*);

template  __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce<sqrPlusBinaryOp>(DMatrix<ulong> const&, sqrPlusBinaryOp<ulong>, ulong, CUstream_st*);
template  __host__ CUDART_DEVICE int CuMatrix<int>::reduce<maxBinaryOp>(DMatrix<int> const&, maxBinaryOp<int>, int, CUstream_st*);
template  __host__ CUDART_DEVICE int CuMatrix<int>::reduce<minBinaryOp>(DMatrix<int> const&, minBinaryOp<int>, int, CUstream_st*);

template  __host__ CUDART_DEVICE unsigned int CuMatrix<unsigned int>::reduce<maxBinaryOp>(DMatrix<unsigned int> const&, maxBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template  __host__ CUDART_DEVICE unsigned int CuMatrix<unsigned int>::reduce<minBinaryOp>(DMatrix<unsigned int> const&, minBinaryOp<unsigned int>, unsigned int, CUstream_st*);


#else
template __host__ CUDART_DEVICE float CuMatrix<float>::reduce(DMatrix<float> const&, MonoidF<float,1>, float, CUstream_st*);
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce(DMatrix<double> const&, MonoidF<double,1>, double, CUstream_st*);
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce(DMatrix<ulong> const&, MonoidF<ulong,1>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce(DMatrix<int> const&, MonoidF<int,1>, int, CUstream_st*);
template __host__ CUDART_DEVICE uint CuMatrix<uint>::reduce(DMatrix<uint> const&, MonoidF<uint,1>, uint, CUstream_st*);
#endif


#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE
void CuMatrix<T>::reduceColumn(T* total, const DMatrix<T>& d_M, BinaryOp<T> op, T start, uint col, cudaStream_t stream  )
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE
void CuMatrix<T>::reduceColumn(T* total, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, uint col, cudaStream_t stream  )
#endif
{
	uint nP = d_M.m;
	uint threads;
	uint blocks;
	::getReductionExecContext(blocks, threads, nP);
	if(checkDebug(debugRedux | debugNoRedux))flprintf("CuMatrix<T>::reduceColumn blocks %d threads %d np %d\n", blocks,threads, nP);
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaPeekAtLastError());
#endif
	if(checkDebug( debugNoRedux)) {
		prlocf("early exit\n");
		*total = start;
	}
	CuMatrix<T> res(blocks, 1, false, true);
	if(checkDebug(debugRedux)) {
		prlocf("res ");
		res.printShortString();
	}
	DMatrix<T> d_Res;
	res.asDmatrix(d_Res, false);

	if(checkDebug(debugRedux| debugNoRedux)) {
		prlocf("after res.asDmatrix(..)\n");
	}

#ifndef __CUDA_ARCH__
	checkCudaError(cudaDeviceSynchronize());
	if(checkDebug(debugRedux| debugNoRedux)){ prlocf("host ");}
#else
	if(checkDebug(debugRedux| debugNoRedux)){ prlocf("dev ");}
#endif
	if(checkDebug(debugRedux| debugNoRedux)){ flprintf("&total %p\n",&total); }
	if(!checkDebug(debugNoRedux)) {
		reduceLauncher(total, d_Res, nP, d_M, op, start, d_M.p, col, stream);
	} else {
		prlocf("skipping reducelauncher\n");
	}
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceColumn<maxBinaryOp>(float*,DMatrix<float> const&, maxBinaryOp<float>, float, uint, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceColumn<minBinaryOp>(float*,DMatrix<float> const&, minBinaryOp<float>, float, uint, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceColumn<maxBinaryOp>(double*,DMatrix<double> const&, maxBinaryOp<double>, double, uint, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceColumn<minBinaryOp>(double*,DMatrix<double> const&, minBinaryOp<double>, double, uint, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceColumn<plusBinaryOp>(int*, DMatrix<int> const&, plusBinaryOp<int>, int, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned int>::reduceColumn<plusBinaryOp>(unsigned int*, DMatrix<unsigned int> const&, plusBinaryOp<unsigned int>, unsigned int, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned long>::reduceColumn<plusBinaryOp>(unsigned long*, DMatrix<unsigned long> const&, plusBinaryOp<unsigned long>, unsigned long, unsigned int, CUstream_st*);

#else
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceColumn(float*, DMatrix<float> const&, MonoidF<float, 1>, float, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceColumn(double*, DMatrix<double> const&, MonoidF<double, 1>, double, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceColumn(int*, DMatrix<int> const&, MonoidF<int, 1>, int, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<uint>::reduceColumn(uint*, DMatrix<uint> const&, MonoidF<uint, 1>, uint, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceColumn(ulong*, DMatrix<ulong> const&, MonoidF<ulong, 1>, ulong, unsigned int, CUstream_st*);
#endif


#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsync(
		T* result, const DMatrix<T>& d_M, BinaryOp<T> op, T start, cudaStream_t stream  )
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsync(
		T* result, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, cudaStream_t stream  )
#endif
{
	uint nP = d_M.m * d_M.n;
	uint threads;
	uint blocks;
	::getReductionExecContext(blocks, threads, nP);
	if(checkDebug(debugExec)) flprintf("reduceAsync blocks %d\n", blocks);
	CuMatrix<T> res(blocks, 1, true, true);
	DMatrix<T> d_Res;
	res.asDmatrix(d_Res, false);
	reduceLauncher(result, d_Res, nP, d_M, op, start, 1, 0, stream);
#ifndef __CUDA_ARCH__
	checkCudaError(cudaStreamSynchronize(stream));
#else
	__syncthreads();
#endif
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsync<maxBinaryOp>(float*,DMatrix<float> const&, maxBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsync<minBinaryOp>(float*,DMatrix<float> const&, minBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsync<maxBinaryOp>(double*,DMatrix<double> const&, maxBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsync<minBinaryOp>(double*,DMatrix<double> const&, minBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsync<maxBinaryOp>(ulong*,DMatrix<ulong> const&, maxBinaryOp<ulong>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsync<minBinaryOp>(ulong*,DMatrix<ulong> const&, minBinaryOp<ulong>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsync<minBinaryOp>(int*, DMatrix<int> const&, minBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsync<maxBinaryOp>(int*, DMatrix<int> const&, maxBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned int>::reduceAsync<minBinaryOp>(unsigned int*, DMatrix<unsigned int> const&, minBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned int>::reduceAsync<maxBinaryOp>(unsigned int*, DMatrix<unsigned int> const&, maxBinaryOp<unsigned int>, unsigned int, CUstream_st*);

#else
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsync<1>(float*,DMatrix<float> const&, MonoidF<float,1>, float, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsync<1>(double*,DMatrix<double> const&, MonoidF<double,1>, double, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsync<1>(int*,DMatrix<int> const&, MonoidF<int,1>, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<uint>::reduceAsync<1>(uint*,DMatrix<uint> const&, MonoidF<uint,1>, uint, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsync<1>(ulong*,DMatrix<ulong> const&, MonoidF<ulong,1>, ulong, CUstream_st*);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsyncBuffer(
		 T* result, DMatrix<T>& buffer, uint blocks, uint threads, ulong nP, const DMatrix<T>& d_M, BinaryOp<T> op, T start, cudaStream_t stream  )
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsyncBuffer(
		 T* result, DMatrix<T>& buffer, uint blocks, uint threads, ulong nP, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, cudaStream_t stream  )
#endif
{
	if(checkDebug(debugExec)) flprintf("reduceAsyncBuffer blocks %d\n", blocks);
	reduceLauncher(result, buffer, nP, d_M, op, start, 1, 0, stream);
#ifndef __CUDA_ARCH__
	checkCudaError(cudaStreamSynchronize(stream));
#else
	__syncthreads();
#endif
}

#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsyncBuffer<maxBinaryOp>( float*,DMatrix<float>&,uint, uint, ulong, DMatrix<float> const&, maxBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsyncBuffer<minBinaryOp>(float*,DMatrix<float>&,uint, uint, ulong, DMatrix<float> const&, minBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsyncBuffer<maxBinaryOp>(double*,DMatrix<double>&,uint, uint, ulong, DMatrix<double> const&, maxBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsyncBuffer<minBinaryOp>(double*,DMatrix<double>&,uint, uint, ulong, DMatrix<double> const&, minBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsyncBuffer<maxBinaryOp>(ulong*,DMatrix<ulong>&,uint, uint, ulong, DMatrix<ulong> const&, maxBinaryOp<ulong>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsyncBuffer<minBinaryOp>(ulong*,DMatrix<ulong>&,uint, uint, ulong, DMatrix<ulong> const&, minBinaryOp<ulong>, ulong, CUstream_st*);

template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsyncBuffer<minBinaryOp>(int*, DMatrix<int>&, unsigned int, unsigned int, unsigned long, DMatrix<int> const&, minBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsyncBuffer<maxBinaryOp>(int*, DMatrix<int>&, unsigned int, unsigned int, unsigned long, DMatrix<int> const&, maxBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned int>::reduceAsyncBuffer<minBinaryOp>(unsigned int*, DMatrix<unsigned int>&, unsigned int, unsigned int, unsigned long, DMatrix<unsigned int> const&, minBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned int>::reduceAsyncBuffer<maxBinaryOp>(unsigned int*, DMatrix<unsigned int>&, unsigned int, unsigned int, unsigned long, DMatrix<unsigned int> const&, maxBinaryOp<unsigned int>, unsigned int, CUstream_st*);

#else
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsyncBuffer(float*, DMatrix<float>&, unsigned int, unsigned int, unsigned long, DMatrix<float> const&, MonoidF<float, 1>, float, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsyncBuffer(double*, DMatrix<double>&, unsigned int, unsigned int, unsigned long, DMatrix<double> const&, MonoidF<double, 1>, double, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsyncBuffer(int*, DMatrix<int>&, unsigned int, unsigned int, unsigned long, DMatrix<int> const&, MonoidF<int, 1>, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<uint>::reduceAsyncBuffer(uint*, DMatrix<uint>&, unsigned int, unsigned int, unsigned long, DMatrix<uint> const&, MonoidF<uint, 1>, uint, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsyncBuffer(ulong*, DMatrix<ulong>&, unsigned int, unsigned int, unsigned long, DMatrix<ulong> const&, MonoidF<ulong, 1>, ulong, CUstream_st*);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE T CuMatrix<T>::reduce(BinaryOp<T> op, T start, cudaStream_t stream ) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE T CuMatrix<T>::reduce(MonoidF<T,StateDim> op, T start, cudaStream_t stream ) const
#endif
{
	DMatrix<T> d_A;
	asDmatrix(d_A);
	T res = reduce(d_A, op, start, stream);
	return res;
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE float CuMatrix<float>::reduce<sqrPlusBinaryOp>(sqrPlusBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce<sqrPlusBinaryOp>(sqrPlusBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce<sqrPlusBinaryOp>(sqrPlusBinaryOp<ulong>, ulong, CUstream_st*) const;
template  __host__ CUDART_DEVICE int CuMatrix<int>::reduce<sqrPlusBinaryOp>(sqrPlusBinaryOp<int>, int, CUstream_st*) const;
template  __host__ CUDART_DEVICE unsigned int CuMatrix<unsigned int>::reduce<sqrPlusBinaryOp>(sqrPlusBinaryOp<unsigned int>, unsigned int, CUstream_st*) const;
#else
template __host__ CUDART_DEVICE float CuMatrix<float>::reduce(MonoidF<float,1>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce(MonoidF<double,1>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce(MonoidF<int,1>, int, CUstream_st*) const;
template __host__ CUDART_DEVICE uint CuMatrix<uint>::reduce(MonoidF<uint,1>, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce(MonoidF<ulong,1>, ulong, CUstream_st*) const;
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE T CuMatrix<T>::reduceColumn(BinaryOp<T> op, T start, uint col, cudaStream_t stream ) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE T CuMatrix<T>::reduceColumn(MonoidF<T,StateDim> op, T start, uint col, cudaStream_t stream ) const
#endif
{
	DMatrix<T> d_A;
	asDmatrix(d_A);
	T res;
	reduceColumn(&res, d_A, op, start, col, stream);
	cherr(cudaDeviceSynchronize());
	return res;
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE float CuMatrix<float>::reduceColumn<sqrPlusBinaryOp>(sqrPlusBinaryOp<float>, float, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::reduceColumn<sqrPlusBinaryOp>(sqrPlusBinaryOp<double>, double, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE float CuMatrix<float>::reduceColumn<plusBinaryOp>(plusBinaryOp<float>, float, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::reduceColumn<plusBinaryOp>(plusBinaryOp<double>, double, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE unsigned long CuMatrix<unsigned long>::reduceColumn<plusBinaryOp>(plusBinaryOp<unsigned long>, unsigned long, unsigned int, CUstream_st*) const;
#else
template __host__ CUDART_DEVICE float CuMatrix<float>::reduceColumn<1>(MonoidF<float,1>, float, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::reduceColumn<1>(MonoidF<double,1>, double, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE int CuMatrix<int>::reduceColumn<1>(MonoidF<int,1>, int, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE uint CuMatrix<uint>::reduceColumn<1>(MonoidF<uint,1>, uint, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduceColumn<1>(MonoidF<ulong,1>, ulong, uint, CUstream_st*) const;
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsync(T* result, BinaryOp<T> op, T start, cudaStream_t stream ) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsync(T* result, MonoidF<T,StateDim> op, T start, cudaStream_t stream ) const
#endif
{
	DMatrix<T> d_A;
	asDmatrix(d_A);
	reduce(result, d_A, op, start, stream);
}

// reduction with addition (Σ)
template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::sum(cudaStream_t stream ) const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
#ifdef  CuMatrix_Enable_KTS
	T res = reduce(d_A, plusBinaryOp<T>(), 0 , stream);
#else
	T res = reduce(d_A, Functory<T,plusBinaryOp>::pinch(), 0 , stream);
#endif
	return res;
}

template<typename T> __host__  T CuMatrix<T>::kahanSum() const {
	if(lastMod == mod_device) {
		dthrow(notSyncedHost());
	}
	T sum = 0;
	T c = 0;
	for(int i = 0; i < m * p ;i++) {
		if(i == m * p -1 ){
			if(checkDebug(debugRedux)) outln("last idx " << i << ", (d_elements + idx) = " << (d_elements + i));
		}
		if(i % p < n) { // verify idx inside meat
			T y = elements[i] - c;
			T t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}else {
			if(checkDebug(debugRedux)) outln("skipping idx " << i << " ( i %p == ) " << (i %p));
		}
	}
	return sum;
}

// reduction with multiplication (Π)
template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::prod( cudaStream_t stream ) const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
#ifdef  CuMatrix_Enable_KTS
	T res = reduce(d_A, multBinaryOp<T>(), 1.0, stream);
#else
	T res = reduce(d_A, Functory<T,multBinaryOp>::pinch(), 1.0, stream);
#endif
	return res;
}


template<typename T> void CuMatrix<T>::featureMeans( CuMatrix<T>& means, bool lv) const {
	DMatrix<T> d_Means, d_X;
	asDmatrix(d_X);
	means.asDmatrix(d_Means);
	featureAvgKernelL(d_Means, d_X, lv);
	means.invalidateHost();
}

template<typename T> void CuMatrix<T>::featureMeansTx( CuMatrix<T>& means) const {
	DMatrix<T> d_Means, d_X;
	asDmatrix(d_X);
	means.asDmatrix(d_Means);
	//outln("d_Means " << util<T>::pdm(d_Means));
	featureAvgTxdKernelL(d_Means, d_X);
	means.invalidateHost();
}

template<typename T> void CuMatrix<T>::featureMeansStreams( CuMatrix<T>& means, bool lv,int nstreams) const {
	DMatrix<T> d_Means, d_X;
	asDmatrix(d_X);
	means.asDmatrix(d_Means);
	//outln("d_Means " << util<T>::pdm(d_Means));
	featureAvgMultiStreamL(d_Means, d_X, lv, nstreams);
	means.invalidateHost();
}

template<typename T> CuMatrix<T> CuMatrix<T>::featureMeans(bool lv) const {
	CuMatrix<T> means = zeros(n, 1);
	featureMeans(means, lv);
	return means;
}


// matrix is transposed and average calced by doing sum-reductions of each row
// one thread for each row
template<typename T> __global__ void rowSumKernel2(DMatrix<T> sums, const DMatrix<T> x) {
	uint rowIdx = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	if(checkDebug(debugRedux) && rowIdx == 0) {
		util<T>::printRow(x, rowIdx);
	}
	__syncthreads();
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaPeekAtLastError());
#endif
	// T* sdata = SharedMemory<T>();
	DMatrix<T> row(x.elements + x.p * rowIdx, 1, x.n);
	__syncthreads();
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaPeekAtLastError());
#endif
	if(checkDebug(debugRedux) && rowIdx == x.m - 1) {
		prlocf("last row as ");
		util<T>::printRow(row, 0);
	}
	if(rowIdx < x.m) {
		if(checkDebug(debugRedux)) {
			flprintf("reducing row %d\n",rowIdx);
		}
		T sum = 0;
#ifdef CuMatrix_Enable_Cdp
		sum = CuMatrix<T>::reduce(row, Functory<T,plusBinaryOp>::pinch(),0);
#else
		prlocf("not implemented for non-cdp\n");
		assert(false);
#endif
		if(checkDebug(debugRedux)) {
			flprintf("row %d sum %f\n",rowIdx, sum);
		}
		sums.elements[rowIdx]= sum;
	}
	__syncthreads();
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaPeekAtLastError());
#endif

}
// sum reduces each row

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::rowSum(DMatrix<T>& d_rowSums, const DMatrix<T>& d_x, cudaStream_t stream) {
	cherr(cudaPeekAtLastError());
	uint blockX = MIN(256, d_x.m);
	uint gridX = blockX >= d_x.m ? 1 : DIV_UP(d_x.m,blockX);
	if(checkDebug(debugRedux)){
		prlocf("rowSum on ");
		::prdm(d_x);
		flprintf(" with gridX %d and blockX %d\n",gridX,blockX);
	}

	//b_util::pFuncPtrAtts((T*)rowSumKernel2<T>);
	bool valid = b_util::validLaunchQ((void*)rowSumKernel2<T>,dim3(gridX), dim3(blockX));
	flprintf("valid %s\n", tOrF(valid));
	rowSumKernel2<<<gridX, blockX, 0, stream>>>(d_rowSums, d_x);
#ifdef __CUDA_ARCH__
	__syncthreads();
#else
	cherr(cudaDeviceSynchronize());
#endif
}

//template<typename T> __global__ smallestMutalFactor(uint* factor, )


/*
 * works but bad shuffln
template<typename T, int StateDim> __global__ void rowReductionKernelNlte64(DMatrix<T> resVec, MonoidF bop, const DMatrix<T> x, uint slice, uint slices) {
	assert(x.n < 65);
	// shared mem to relay partial sums of rows that span warps
	uint col = threadIdx.x;
	uint row = threadIdx.y + blockIdx.y * blockDim.y;
	if(row < x.m && col < x.n) {
		// test for elements processed by by cols with laneid > WARP_SIZE
		int2 laneid = b_util::laneId(threadIdx, blockIdx, blockDim);
		ulong soffset = slice * x.m * x.p/slices;
		uint toffset = slice * x.m * resVec.p/slices;
		int currLen = x.n - 1;
		T* xrow = x.elements + soffset + row * x.p;
		// first reduction fills warp
		T  s = col < x.n ? xrow[col] : bop.identity;

		if(	col + WARP_SIZE < x.n )
			s = bop(s,xrow[col + WARP_SIZE]);
		__syncthreads();

		while(currLen > 0) {
			int dLane =  col == 0 ? currLen : 0;
			if(WARP_SIZE - laneid.y < x.n - col) {
				flnfprintf("row %u laneid %u.%u y + x.n-col %d > ws\n", row, laneid.x,laneid.y, x.n-col);
				// on a spanrow, so explicitly load elems of spanrow seg in 2nd warp
				if(col + dLane < x.n) {
					flnfprintf("loading lane %u.%u from next warps for row %u col %u s was %f\n", laneid.x + 1 ,   WARP_SIZE - (laneid.y + dLane), row,col,s);
					s = bop(s, xrow[col+dLane]);
					flnfprintf("s is %f\n",s);
				} else {
					flnfprintf("ignoring lane %u from next warps for row %u col %u\n" , (laneid.y + dLane - WARP_SIZE), row,col);
				}
			} else  {
				T os = shflDown<T>(s, dLane );
				//  (laneid<x.n && laneid > col) skips elems of segment of spanrow in 2nd warp
				s = col == 0 ? bop(s, os)  : s;
				if(checkDebug(debugRedux)) flnfprintf("x.n %u row %u col %u lane id %u.%u dLane %d os %f s %f\n", x.n, row, col, laneid.x,laneid.y, dLane, os,s);
			}
			currLen--;
		}

		if(col == 0) {
			resVec.elements[row + toffset] = s;
		}
	}
}
*/


/*
 * todo version for well-formed matrices
 * needs smem T for each spanrow that holds partial reduction of 1st part of row
 */
#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOpF> __global__
void rowReductionKernelNlte64(DMatrix<T> resVec, BinaryOpF<T> bop, const DMatrix<T> x, uint slice, uint slices)
#else
template<typename T, int StateDim> __global__
void rowReductionKernelNlte64(DMatrix<T> resVec, MonoidF<T,StateDim> bop, const DMatrix<T> x, uint slice, uint slices)
#endif
{
	assert(x.n < 65);
	// shared mem to relay partial sums of rows that span warps
	uint col = threadIdx.x;
	uint row = threadIdx.y + blockIdx.y * blockDim.y;
	// thread in mat bounds
	if(row < x.m && col < x.n) {
		uint laneid;
		b_util::laneid(laneid);
		ulong soffset = slice * x.m * x.p;
		uint toffset = slice * x.m * resVec.p;
		if(checkDebug(debugRedux) && col == 0  && row == 0) {
			flprintf("slice %u soffset %lu toffset %u\n", slice, soffset, toffset);
		}
		int currLen = MIN(WARP_SIZE/2, b_util::nextPowerOf2(x.n)/2);
		T* xrow = x.elements + soffset + row * x.p;
		// first reduction fills warp local ses
		T  s = col < x.n ? xrow[col] : bop.identity_ro();

		if(	col + WARP_SIZE < x.n )
			s = bop(s,xrow[col + WARP_SIZE]);
		__syncthreads();

		if(col == 0 && x.n > WARP_SIZE - laneid && x.n < WARP_SIZE) {
			// row spans warps, so don't shuffle
			for(int i =1; i < x.n; i++) {
				s = bop(s, xrow[col + i]);
			}
		} else {
			while(currLen > 0) {
				int dLane =  currLen; // col == 0 ? currLen : 0;
				// check for beginning of a warp spanrow
				T os = shflDown<T>(s, dLane);
				//  (laneid<x.n && laneid > col) skips elems of segment of spanrow in 2nd warp
				if(col > 0 && col > laneid ) {
					// todo retrieve partial redux of 1st part of span row and reduce rest of spanrow with that
					//if(checkDebug(debugRedux)) flnfprintf("skipn x.n %u row %u col %u lane id  %u dLane %d os %f s %f\n", x.n, row, col, laneid, dLane, os,s);
				} else if(col + dLane < x.n){
					s = bop(s, os);
					//if(checkDebug(debugRedux)) flnfprintf("bopn x.n %u row %u col %u lane id  %u dLane %d os %f s %f\n", x.n, row, col, laneid, dLane, os,s);
				}
				currLen >>= 1;
			}
		}
		if(col == 0) {
			resVec.elements[row + toffset] = s;
		}
	}
}

//
// 'slices' are row-wise
// stripes of 64 cols are col-wise
#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOpF> __global__ void rowReductionKernel(DMatrix<T> resVec, BinaryOpF<T> bop, const DMatrix<T> x, uint slice, uint slices, uint stripes)
#else
template<typename T, int StateDim> __global__ void rowReductionKernel(DMatrix<T> resVec, MonoidF<T,StateDim> bop, const DMatrix<T> x, uint slice, uint slices, uint stripes)
#endif
{
	uint col = threadIdx.x;
	uint row = threadIdx.y + blockIdx.y * blockDim.y;
	ulong soffset = slice * x.m * x.p;
	uint toffset = slice * x.m * resVec.p;
	for(int stripe = 0; stripe < stripes; stripe++) {
		int currLen = MIN(WARP_SIZE/2, b_util::nextPowerOf2(x.n)/2);
		T* xrow = x.elements + soffset + row * x.p + stripe * 64;
		// first reduction fills warp
		T  s = col < x.n ? xrow[col] : bop.identity_ro();
		if(	col + WARP_SIZE < x.n )
			s = bop(s,xrow[col + WARP_SIZE]);
		__syncthreads();
		while(currLen > 0) {
			int lane = col + currLen;
			s = bop(s, shfl<T>(s, lane));
			currLen >>= 1;
		}

		if(threadIdx.x == 0) {
			resVec.elements[row * resVec.p + stripe + toffset] = s;
		}
	}
	__syncthreads();
	assert(stripes < 65);
	int currLen = MIN(WARP_SIZE/2, b_util::nextPowerOf2(stripes)/2);
	T* resVecRow = resVec.elements + toffset + row * resVec.p;
	// first reduction fills warp
	T  s = col < stripes ? resVecRow[col] : bop.identity_ro();
	if(	col + WARP_SIZE < x.n )
		s = bop(s,resVecRow[col + WARP_SIZE]);
	__syncthreads();
	while(currLen > 0) {
		int lane = col + currLen;
		s = bop(s, shfl<T>(s, lane));
		currLen >>= 1;
	}

	if(threadIdx.x == 0) {
		resVec.elements[row + toffset] = s;
	}

}

#ifdef  CuMatrix_Enable_KTS
template <typename T> template <template <typename> class BinaryOp> __host__ CUDART_DEVICE void CuMatrix<T>::reduceRowsNlte64(DMatrix<T>& resVec, const DMatrix<T>& d_x, BinaryOp<T> op, cudaStream_t stream  )
#else
template <typename T> template <int StateDim> __host__ CUDART_DEVICE void CuMatrix<T>::reduceRowsNlte64(DMatrix<T>& resVec, const DMatrix<T>& d_x, MonoidF<T,StateDim> op, cudaStream_t stream  )
#endif
{

	assert(d_x.n <= WARP_SIZE*2);
	ExecCaps * pcaps = ExecCaps::currCaps();
	uint blockW= MIN(WARP_SIZE, d_x.n);
	uint blockH = MIN(d_x.m, maxH<T>(*pcaps,blockW));
	if(checkDebug(debugRedux))flprintf("for d_x %ux%ux%u first %p last %p\n", d_x.m,d_x.n, d_x.p, d_x.elements, d_x.elements + (d_x.m -1)*d_x.p + d_x.n -1);

	if(checkDebug(debugRedux))flprintf("blockH %d  maxH<T>(*pcaps,blockW) %d\n",blockH,  maxH<T>(*pcaps,blockW));
	int gridY = DIV_UP(d_x.m, blockH);
	dim3 grid(1, gridY);
	// in case grid y is too big
	int slices = DIV_UP(grid.y, pcaps->maxGrid.y);
	if(checkDebug(debugRedux))flprintf("slices %d\n",slices);
	dim3 block(blockW,blockH);
	int sliceGridY = grid.y/ slices;
	DMatrix<T> d_slice(d_x);
	d_slice.m = sliceGridY * blockH;
	if(checkDebug(debugRedux))flprintf("init sliceGridY %d d_slice.m %d\n",sliceGridY, d_slice.m);

	int offset;
	for(int currSlice =0; currSlice < slices; currSlice++) {
		offset = currSlice * d_slice.m * d_x.p ;
		if(currSlice == slices - 1) {
			if(checkDebug(debugRedux))prlocf("last fill slice");
			d_slice.m = d_x.m - (slices - 1 ) * d_slice.m;
			sliceGridY =  DIV_UP(d_slice.m, blockH);
		}
		grid.y = sliceGridY;
		if(checkDebug(debugRedux)){
			flprintf("sliceGridY %d\n",sliceGridY);
			flprintf("slice %d on mat offset %d %dX%d(X%d) (d_slice.elements 1st %p last %p)\n",
					currSlice, offset, d_slice.m, d_slice.n, d_slice.p, d_slice.elements+ offset,d_slice.elements+ offset + (d_slice.m-1)* d_slice.p + d_slice.n -1);
			 b_util::prd3(grid, " grid of ");
			 b_util::prd3(block, "block of");
		}
		rowReductionKernelNlte64<<<grid, block, 0, stream>>>(resVec, op, d_slice, currSlice, slices);
	}
	cherr(cudaDeviceSynchronize());
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceRows<plusBinaryOp>(DMatrix<float>&, DMatrix<float> const&, plusBinaryOp<float>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceRows<plusBinaryOp>(DMatrix<double>&, DMatrix<double> const&, plusBinaryOp<double>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceRows<plusBinaryOp>(DMatrix<ulong>&, DMatrix<ulong> const&, plusBinaryOp<ulong>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceRows<plusBinaryOp>(DMatrix<int>&, DMatrix<int> const&, plusBinaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned int>::reduceRows<plusBinaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, plusBinaryOp<unsigned int>, CUstream_st*);

#else
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceRows(DMatrix<float>&, DMatrix<float> const&, MonoidF<float,1>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceRows(DMatrix<double>&, DMatrix<double> const&, MonoidF<double,1>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceRows(DMatrix<ulong>&, DMatrix<ulong> const&, MonoidF<ulong,1>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceRows(DMatrix<int>&, DMatrix<int> const&, MonoidF<int,1>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<uint>::reduceRows(DMatrix<uint>&, DMatrix<uint> const&, MonoidF<uint,1>, CUstream_st*);
#endif

#ifdef  CuMatrix_Enable_KTS
template <typename T> template <template <typename> class BinaryOp> __host__ CUDART_DEVICE void CuMatrix<T>::reduceRows(DMatrix<T>& resVec, const DMatrix<T>& d_x, BinaryOp<T> op, cudaStream_t stream  )
#else
template <typename T> template <int StateDim> __host__ CUDART_DEVICE void CuMatrix<T>::reduceRows(DMatrix<T>& resVec, const DMatrix<T>& d_x, MonoidF<T,StateDim> op, cudaStream_t stream  )
#endif
{
	if(d_x.n < 65) {
		reduceRowsNlte64(resVec,d_x,op, stream);
	}
}
#include "CuMatrixInster.cu"
