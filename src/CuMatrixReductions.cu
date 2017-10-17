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
	long nP = d_M.m * d_M.n;

	if(nP == 1) {
		T result;
		CuTimer timer;
		timer.start();
#ifndef __CUDA_ARCH__
		cherr(cudaMemcpy(&result, d_M.elements, sizeof(T), cudaMemcpyDeviceToHost));
		//CuMatrix<T>::incDhCopy("CuMatrix<T>::reduce(long l)", sizeof(T),timer.stop());
#else
		memcpy(&result, d_M.elements, sizeof(T));
#endif
		return result;
	}
	int threads;
	int blocks;

	::getReductionExecContext(blocks, threads, nP);
	if(checkDebug(debugRedux )) flprintf("CuMatrix<T>::reduce blocks %d threads %d np %d\n", blocks,threads, nP);
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaPeekAtLastError());
#endif
	//CuMatrix<T> res(blocks, 1, false, true);
	cherr(cudaPeekAtLastError());
	if(checkDebug(debugRedux)) {
		prlocf("res ");
		//res.printShortString();
	}

	DMatrix<T> d_Res(blocks,1);
	cherr(cudaMalloc( &(d_Res.elements), blocks*sizeof(T)));
	//res.tile0(d_Res, false);

	if(checkDebug(debugRedux)) {
		prlocf("after res.tile0(..)\n");
	}

#ifndef __CUDA_ARCH__
	checkCudaError(cudaDeviceSynchronize());

	if(checkDebug(debugRedux)) prlocf("after tile0");

	if(checkDebug(debugRedux)) prlocf("host \n");
#else
	if(checkDebug(debugRedux)) prlocf("dev \n");
#endif
	T total = 0;
	if(checkDebug(debugRedux)) flprintf("&total %p\n",&total);
	if(checkDebug(debugRedux)) flprintf("curr dev %d device of d_m.elems %d device of d_Res.elems %d\n",ExecCaps::currDev(),
			b_util::getDevice(d_M.elements), b_util::getDevice(d_Res.elements));
	if(checkDebug(debugRedux)) flprintf("d_M.m,d_M.n %d d_M.p %d stride %d\n",d_M.m, d_M.n, d_M.p ,d_M.n != d_M.p ? d_M.p : 1);
	reduceLauncher(&total, d_Res, nP, d_M, op, start, 0, stream);
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
	cherr(cudaFree( d_Res.elements));

	return total;
}
#ifdef  CuMatrix_Enable_KTS

template __host__ CUDART_DEVICE float CuMatrix<float>::reduce<maxBinaryOp>(DMatrix<float> const&, maxBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE float CuMatrix<float>::reduce<minBinaryOp>(DMatrix<float> const&, minBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce<maxBinaryOp>(DMatrix<double> const&, maxBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce<minBinaryOp>(DMatrix<double> const&, minBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce<minBinaryOp>(DMatrix<ulong> const&, minBinaryOp<ulong>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce<maxBinaryOp>(DMatrix<ulong> const&, maxBinaryOp<ulong>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce<maxBinaryOp>(DMatrix<int> const&, maxBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce<minBinaryOp>(DMatrix<int> const&, minBinaryOp<int>, int, CUstream_st*);

template __host__ CUDART_DEVICE unsigned int CuMatrix<unsigned int>::reduce<maxBinaryOp>(DMatrix<unsigned int> const&, maxBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE unsigned int CuMatrix<unsigned int>::reduce<minBinaryOp>(DMatrix<unsigned int> const&, minBinaryOp<unsigned int>, unsigned int, CUstream_st*);

template __host__ CUDART_DEVICE float CuMatrix<float>::reduce<andBinaryOp>(DMatrix<float> const&, andBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce<andBinaryOp>(DMatrix<double> const&, andBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce<andBinaryOp>(DMatrix<int> const&, andBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE unsigned int CuMatrix<unsigned int>::reduce<andBinaryOp>(DMatrix<unsigned int> const&, andBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE long CuMatrix<long>::reduce<andBinaryOp>(DMatrix<long> const&, andBinaryOp<long>, long, CUstream_st*);
template __host__ CUDART_DEVICE unsigned long CuMatrix<unsigned long>::reduce<andBinaryOp>(DMatrix<unsigned long> const&, andBinaryOp<unsigned long>, unsigned long, CUstream_st*);

template __host__ CUDART_DEVICE float CuMatrix<float>::reduce<orBinaryOp>(DMatrix<float> const&, orBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce<orBinaryOp>(DMatrix<double> const&, orBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce<orBinaryOp>(DMatrix<int> const&, orBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE unsigned int CuMatrix<unsigned int>::reduce<orBinaryOp>(DMatrix<unsigned int> const&, orBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE long CuMatrix<long>::reduce<orBinaryOp>(DMatrix<long> const&, orBinaryOp<long>, long, CUstream_st*);
template __host__ CUDART_DEVICE unsigned long CuMatrix<unsigned long>::reduce<orBinaryOp>(DMatrix<unsigned long> const&, orBinaryOp<unsigned long>, unsigned long, CUstream_st*);


template __host__ CUDART_DEVICE float CuMatrix<float>::reduce<plusBinaryOp>(DMatrix<float> const&, plusBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce<plusBinaryOp>(DMatrix<double> const&, plusBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce<plusBinaryOp>(DMatrix<ulong> const&, plusBinaryOp<ulong>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce<plusBinaryOp>(DMatrix<int> const&, plusBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE unsigned int CuMatrix<unsigned int>::reduce<plusBinaryOp>(DMatrix<unsigned int> const&, plusBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE long CuMatrix<long>::reduce<plusBinaryOp>(DMatrix<long> const&, plusBinaryOp<long>, long, CUstream_st*);


#else
template __host__ CUDART_DEVICE float CuMatrix<float>::reduce(DMatrix<float> const&, MonoidF<float,1>, float, CUstream_st*);
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce(DMatrix<double> const&, MonoidF<double,1>, double, CUstream_st*);
template __host__ CUDART_DEVICE long CuMatrix<long>::reduce(DMatrix<long> const&, MonoidF<long,1>, long, CUstream_st*);
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce(DMatrix<ulong> const&, MonoidF<ulong,1>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce(DMatrix<int> const&, MonoidF<int,1>, int, CUstream_st*);
template __host__ CUDART_DEVICE uint CuMatrix<uint>::reduce(DMatrix<uint> const&, MonoidF<uint,1>, uint, CUstream_st*);
#endif


#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE
void CuMatrix<T>::reduceColumn(T* total, const DMatrix<T>& d_M, BinaryOp<T> op, T start, int col, cudaStream_t stream  )
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE
void CuMatrix<T>::reduceColumn(T* total, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, int col, cudaStream_t stream  )
#endif
{
	long nP = d_M.m;
	int threads;
	int blocks;
	::getReductionExecContext(blocks, threads, nP);
	if(checkDebug(debugRedux))flprintf("CuMatrix<T>::reduceColumn blocks %d threads %d np %d\n", blocks,threads, nP);
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaPeekAtLastError());
#endif
	CuMatrix<T> res(blocks, 1, false, true);
	if(checkDebug(debugRedux)) {
		prlocf("res ");
		res.printShortString();
	}
	DMatrix<T> d_Res;
	res.tile0(d_Res, false);

	if(checkDebug(debugRedux)) {
		prlocf("after res.tile0(..)\n");
	}

#ifndef __CUDA_ARCH__
	checkCudaError(cudaDeviceSynchronize());
	if(checkDebug(debugRedux)){ prlocf("host ");}
#else
	if(checkDebug(debugRedux)){ prlocf("dev ");}
#endif
	if(checkDebug(debugRedux)){ flprintf("&total %p\n",&total); }
	reduceLauncher(total, d_Res, nP, d_M, op, start, col, stream);
}
#ifdef  CuMatrix_Enable_KTS
template  __host__ CUDART_DEVICE void CuMatrix<float>::reduceColumn<plusBinaryOp>(float*, DMatrix<float> const&, plusBinaryOp<float>, float, int, CUstream_st*);
template  __host__ CUDART_DEVICE  void CuMatrix<int>::reduceColumn<plusBinaryOp>(int*, DMatrix<int> const&, plusBinaryOp<int>, int, int, CUstream_st*);
template  __host__ CUDART_DEVICE  void CuMatrix<long>::reduceColumn<plusBinaryOp>(long*, DMatrix<long> const&, plusBinaryOp<long>, long, int, CUstream_st*);
template  __host__ CUDART_DEVICE  void CuMatrix<unsigned int>::reduceColumn<plusBinaryOp>(unsigned int*, DMatrix<unsigned int> const&, plusBinaryOp<unsigned int>, unsigned int, int, CUstream_st*);
template  __host__ CUDART_DEVICE  void CuMatrix<double>::reduceColumn<plusBinaryOp>(double*, DMatrix<double> const&, plusBinaryOp<double>, double, int, CUstream_st*);
template  __host__ CUDART_DEVICE  void CuMatrix<unsigned long>::reduceColumn<plusBinaryOp>(unsigned long*, DMatrix<unsigned long> const&, plusBinaryOp<unsigned long>, unsigned long, int, CUstream_st*);
#else
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceColumn(float*, DMatrix<float> const&, MonoidF<float, 1>, float, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceColumn(double*, DMatrix<double> const&, MonoidF<double, 1>, double, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceColumn(int*, DMatrix<int> const&, MonoidF<int, 1>, int, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<uint>::reduceColumn(uint*, DMatrix<uint> const&, MonoidF<uint, 1>, uint, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<long>::reduceColumn(long*, DMatrix<long> const&, MonoidF<long, 1>, long, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceColumn(ulong*, DMatrix<ulong> const&, MonoidF<ulong, 1>, ulong, int, CUstream_st*);
#endif


#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsync(
		T* result, const DMatrix<T>& d_M, BinaryOp<T> op, T start, cudaStream_t stream  )
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsync(
		T* result, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, cudaStream_t stream  )
#endif
{
	long nP = d_M.m * d_M.n;
	int threads;
	int blocks;
	::getReductionExecContext(blocks, threads, nP);
	if(checkDebug(debugExec)) flprintf("reduceAsync blocks %d\n", blocks);
	CuMatrix<T> res(blocks, 1, true, true);
	DMatrix<T> d_Res;
	res.tile0(d_Res, false);
	reduceLauncher(result, d_Res, nP, d_M, op, start, 0, stream);
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
template __host__ CUDART_DEVICE void CuMatrix<uint>::reduceAsync<minBinaryOp>(uint*, DMatrix<uint> const&, minBinaryOp<uint>, uint, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<uint>::reduceAsync<maxBinaryOp>(uint*, DMatrix<uint> const&, maxBinaryOp<uint>, uint, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<long>::reduceAsync<minBinaryOp>(long*, DMatrix<long> const&, minBinaryOp<long>, long, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<long>::reduceAsync<maxBinaryOp>(long*, DMatrix<long> const&, maxBinaryOp<long>, long, CUstream_st*);

#else
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsync<1>(float*,DMatrix<float> const&, MonoidF<float,1>, float, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsync<1>(double*,DMatrix<double> const&, MonoidF<double,1>, double, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsync<1>(int*,DMatrix<int> const&, MonoidF<int,1>, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<uint>::reduceAsync<1>(uint*,DMatrix<uint> const&, MonoidF<uint,1>, uint, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<long>::reduceAsync<1>(long*,DMatrix<long> const&, MonoidF<long,1>, long, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsync<1>(ulong*,DMatrix<ulong> const&, MonoidF<ulong,1>, ulong, CUstream_st*);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsyncBuffer(
		 T* result, DMatrix<T>& buffer, int blocks, int threads, long nP, const DMatrix<T>& d_M, BinaryOp<T> op, T start, cudaStream_t stream  )
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsyncBuffer(
		 T* result, DMatrix<T>& buffer, int blocks, int threads, long nP, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, cudaStream_t stream  )
#endif
{
	if(checkDebug(debugExec)) flprintf("reduceAsyncBuffer blocks %d\n", blocks);
	reduceLauncher(result, buffer, nP, d_M, op, start, 0, stream);
#ifndef __CUDA_ARCH__
	checkCudaError(cudaStreamSynchronize(stream));
#else
	__syncthreads();
#endif
}

#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsyncBuffer<maxBinaryOp>( float*,DMatrix<float>&,int, int, long, DMatrix<float> const&, maxBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsyncBuffer<minBinaryOp>(float*,DMatrix<float>&,int, int, long, DMatrix<float> const&, minBinaryOp<float>, float, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsyncBuffer<maxBinaryOp>(double*,DMatrix<double>&,int, int, long, DMatrix<double> const&, maxBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsyncBuffer<minBinaryOp>(double*,DMatrix<double>&,int, int, long, DMatrix<double> const&, minBinaryOp<double>, double, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsyncBuffer<maxBinaryOp>(ulong*,DMatrix<ulong>&,int, int, long, DMatrix<ulong> const&, maxBinaryOp<ulong>, ulong, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsyncBuffer<minBinaryOp>(ulong*,DMatrix<ulong>&,int, int, long, DMatrix<ulong> const&, minBinaryOp<ulong>, ulong, CUstream_st*);

template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsyncBuffer<minBinaryOp>(int*, DMatrix<int>&, int, int, long, DMatrix<int> const&, minBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsyncBuffer<maxBinaryOp>(int*, DMatrix<int>&, int, int, long, DMatrix<int> const&, maxBinaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned int>::reduceAsyncBuffer<minBinaryOp>(unsigned int*, DMatrix<unsigned int>&, int, int, long, DMatrix<unsigned int> const&, minBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned int>::reduceAsyncBuffer<maxBinaryOp>(unsigned int*, DMatrix<unsigned int>&, int, int, long, DMatrix<unsigned int> const&, maxBinaryOp<unsigned int>, unsigned int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<long>::reduceAsyncBuffer<minBinaryOp>(long*, DMatrix<long>&, int, int, long, DMatrix<long> const&, minBinaryOp<long>, long, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<long>::reduceAsyncBuffer<maxBinaryOp>(long*, DMatrix<long>&, int, int, long, DMatrix<long> const&, maxBinaryOp<long>, long, CUstream_st*);

#else
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsyncBuffer(float*, DMatrix<float>&, int, int, long, DMatrix<float> const&, MonoidF<float, 1>, float, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsyncBuffer(double*, DMatrix<double>&, int, int, long, DMatrix<double> const&, MonoidF<double, 1>, double, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsyncBuffer(int*, DMatrix<int>&, int, int, long, DMatrix<int> const&, MonoidF<int, 1>, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<uint>::reduceAsyncBuffer(uint*, DMatrix<uint>&, int, int, long, DMatrix<uint> const&, MonoidF<uint, 1>, uint, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<long>::reduceAsyncBuffer(long*, DMatrix<long>&, int, int, long, DMatrix<long> const&, MonoidF<long, 1>, long, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsyncBuffer(ulong*, DMatrix<ulong>&, int, int, long, DMatrix<ulong> const&, MonoidF<ulong, 1>, ulong, CUstream_st*);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE T CuMatrix<T>::reduce(BinaryOp<T> op, T start, cudaStream_t stream ) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE T CuMatrix<T>::reduce(MonoidF<T,StateDim> op, T start, cudaStream_t stream ) const
#endif
{
	//assert(lastMod != mod_host);

	DMatrix<T> d_A;
	if(checkDebug(debugRedux) ) flprintf("tiler.m_m %u, tiler.m_n %u, tiler.m_p %u m_size %lu tileSize %ld\n",
			tiler.m_m,tiler.m_n,tiler.m_p, tiler.m_size, tiler.tileSize);

	int roff=0, coff=0, tileM = 0, tileN = 0, tileP=0;
	int tileCount  = tiler.getTileCount();
	//tiler.tileDims(tileM,tileN,tdRows);
	//tileCount = MAX(tileCount, DIV_UP(m,tileM));
	T* resA;
	T res;
#ifndef __CUDA_ARCH__
	cherr(cudaHostAlloc(&resA,tileCount*sizeof(T),0));
#else
	resA = (T*) malloc(tileCount*sizeof(T));
#endif

	int lastGpu = -1;
	int orgDevice = ExecCaps::currDev();
	int gpuCount = tiler.countGpus();
	if(checkDebug(debugRedux) ) flprintf("orgDev %d gpuCount %d\n",orgDevice, gpuCount);
	cudaStream_t* streams = null;

	if(gpuCount > 1) {
		assert(!stream);
		cudaStream_t* streams = (cudaStream_t* ) malloc(gpuCount * sizeof(cudaStream_t));
		for(int i =0 ; i < gpuCount; i++) {
			lastGpu = tiler.nextGpu(lastGpu);
			if(gpuCount> 1)
				ExecCaps_setDevice(lastGpu);
			//cherr(cudaSetDevice(lastGpu));
			cherr(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));
		}
	}
	lastGpu = -1;
	if(checkDebug(debugRedux) ) flprintf("m %d, n %d, p %d, tileP %d, tiler.getTileCount() %d tileDir %s\n",m, n, p, tileP, tileCount, b_util::tileDir(tiler.tileD));
	for(int tile = 0; tile < tileCount; tile++) {
		if(gpuCount> 1)
			ExecCaps_setDevice(lastGpu);
		tiler.tile1D( d_A,roff,coff,tileM, tileN, tileP, tile, tdRows,lastMod == mod_host, lastGpu,gpuCount > 1 ? streams[tile] : stream);
		if(checkDebug(debugRedux) ) util<T>::prdm("d_A", d_A);
		resA[tile] = reduce(d_A, op, start, gpuCount > 1 ? streams[tile] : stream);
	}
	if(gpuCount > 1) {
		for(int i =0 ; i < gpuCount; i++) {
			cherr(cudaStreamDestroy(streams[i]));
		}
		free(streams);
	}
	if(tileCount > 1) {
		if(checkDebug(debugRedux) ) flprintf("reduce tileCount %d\n",tileCount);
		// reduce across tile reductions
		T* dres = null;
#ifndef __CUDA_ARCH__
		cherr(cudaMalloc(&dres, tileCount *sizeof(T)));
		cherr(cudaMemcpy(dres, resA, tileCount*sizeof(T), cudaMemcpyHostToDevice));
#else
		dres = resA;
#endif
		d_A.elements = dres;
		d_A.m = tileCount;
		d_A.n = 1; d_A.p = 1;
		res = reduce(d_A, op, start, stream);
#ifndef __CUDA_ARCH__
		cudaFree(dres);
#endif
	} else {
		if(checkDebug(debugRedux) ) flprintf("single tile reduction -> %f\n", (float) resA[0]);
		res =  resA[0];
	}
#ifndef __CUDA_ARCH__
	if(checkDebug(debugDestr))flprintf("freeing host resA %p\n", resA);
	cherr(cudaFreeHost(resA));
#else
	free(resA);
#endif

	return res;
}

#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE float CuMatrix<float>::reduce<maxBinaryOp>(maxBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE float CuMatrix<float>::reduce<minBinaryOp>(minBinaryOp<float>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce<maxBinaryOp>(maxBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce<minBinaryOp>(minBinaryOp<double>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce<maxBinaryOp>(maxBinaryOp<int>, int, CUstream_st*) const;
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce<minBinaryOp>(minBinaryOp<int>, int, CUstream_st*) const;
template __host__ CUDART_DEVICE long CuMatrix<long>::reduce<maxBinaryOp>(maxBinaryOp<long>, long, CUstream_st*) const;
template __host__ CUDART_DEVICE long CuMatrix<long>::reduce<minBinaryOp>(minBinaryOp<long>, long, CUstream_st*) const;
template __host__ CUDART_DEVICE unsigned int CuMatrix<unsigned int>::reduce<maxBinaryOp>(maxBinaryOp<unsigned int>, unsigned int, CUstream_st*) const;
template __host__ CUDART_DEVICE unsigned int CuMatrix<unsigned int>::reduce<minBinaryOp>(minBinaryOp<unsigned int>, unsigned int, CUstream_st*) const;
template __host__ CUDART_DEVICE unsigned long CuMatrix<unsigned long>::reduce<maxBinaryOp>(maxBinaryOp<unsigned long>, unsigned long, CUstream_st*) const;
template __host__ CUDART_DEVICE unsigned long CuMatrix<unsigned long>::reduce<minBinaryOp>(minBinaryOp<unsigned long>, unsigned long, CUstream_st*) const;

#else
template __host__ CUDART_DEVICE float CuMatrix<float>::reduce(MonoidF<float,1>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::reduce(MonoidF<double,1>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE int CuMatrix<int>::reduce(MonoidF<int,1>, int, CUstream_st*) const;
template __host__ CUDART_DEVICE uint CuMatrix<uint>::reduce(MonoidF<uint,1>, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduce(MonoidF<ulong,1>, ulong, CUstream_st*) const;
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE T CuMatrix<T>::reduceColumn(BinaryOp<T> op, T start, int col, cudaStream_t stream ) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE T CuMatrix<T>::reduceColumn(MonoidF<T,StateDim> op, T start, int col, cudaStream_t stream ) const
{
	DMatrix<T> d_A;
#endif

	T* resA;
	T res;
	int lastGpu = 0;
	int orgDevice = ExecCaps::currDev();
	int gpuCount = tiler.countGpus();
	int tileCount = tiler.getTileCount();
	cudaStream_t* streams = null;

	if(gpuCount > 1) {
		assert(!stream);
		streams = (cudaStream_t* ) malloc(gpuCount * sizeof(cudaStream_t));
		for(int i =0 ; i < gpuCount; i++) {
			lastGpu = tiler.nextGpu(lastGpu);
#ifndef __CUDA_ARCH__
			cherr(cudaSetDevice(lastGpu));
#endif
			cherr(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));
		}
	}
	lastGpu = tiler.nextGpu(0);
	int roff, coff, tileM = 0, tileN = 0, tileP = 0;
#ifndef __CUDA_ARCH__
	cherr(cudaHostAlloc(&resA,tileCount*sizeof(T),0));
#else
	resA = (T*) malloc(tileCount*sizeof(T));
#endif

	for(int tile = 0; tile < tileCount; tile++) {
		// tile1D(DMatrix<T>& dm,  int& roff, int& coff,int& tileM, int& tileN, int& tileP,   int t, TileDirection tileD = tdRows, bool copy = true, int lastGpu = -1, cudaStream_t stream = 0)
		tiler.tile1D(d_A, roff, coff, tileM, tileN, tileP, tile, tdRows, true);
		reduceColumn(resA + tile, d_A, op, start, col, stream);
	}
	if(tileCount > 1) {
		if(checkDebug(debugRedux) ) flprintf("reduce across %d tile reductions %d\n",tileCount);
		// reduce across tile reductions
		T* dres = null;
#ifndef __CUDA_ARCH__
		cherr(cudaMalloc(&dres, tileCount*sizeof(T)));
		cherr(cudaMemcpy(dres, resA, tileCount*sizeof(T), cudaMemcpyHostToDevice));
#else
		dres = resA;
#endif
		d_A.elements = dres;
		d_A.m = tileCount;
		d_A.n = 1; d_A.p = 1;
		res = reduce(d_A, op, start, stream);
#ifndef __CUDA_ARCH__
		cudaFree(dres);
#endif
	} else {
		if(checkDebug(debugRedux) ) flprintf("single tile reduction -> %f\n", (float) resA[0]);
		res =  resA[0];
	}
#ifndef __CUDA_ARCH__
	if(checkDebug(debugDestr))flprintf("freeing host resA %p\n", resA);
	cherr(cudaFreeHost(resA));
#else
	free(resA);
#endif

	return res;
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE float CuMatrix<float>::reduceColumn<plusBinaryOp>(plusBinaryOp<float>, float, int, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::reduceColumn<plusBinaryOp>(plusBinaryOp<double>, double, int, CUstream_st*) const;
template __host__ CUDART_DEVICE unsigned long CuMatrix<unsigned long>::reduceColumn<plusBinaryOp>(plusBinaryOp<unsigned long>, unsigned long, int, CUstream_st*) const;


#else
template __host__ CUDART_DEVICE float CuMatrix<float>::reduceColumn<1>(MonoidF<float,1>, float, int, CUstream_st*) const;
template __host__ CUDART_DEVICE double CuMatrix<double>::reduceColumn<1>(MonoidF<double,1>, double, int, CUstream_st*) const;
template __host__ CUDART_DEVICE int CuMatrix<int>::reduceColumn<1>(MonoidF<int,1>, int, int, CUstream_st*) const;
template __host__ CUDART_DEVICE uint CuMatrix<uint>::reduceColumn<1>(MonoidF<uint,1>, uint, int, CUstream_st*) const;
template __host__ CUDART_DEVICE ulong CuMatrix<ulong>::reduceColumn<1>(MonoidF<ulong,1>, ulong, int, CUstream_st*) const;
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsync(T* result, BinaryOp<T> op, T start, cudaStream_t stream ) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE void CuMatrix<T>::reduceAsync(T* result, MonoidF<T,StateDim> op, T start, cudaStream_t stream ) const
#endif
{
	reduce(op, start, stream);
}

#ifdef  CuMatrix_Enable_KTS


#else
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceAsync<1>(float*, MonoidF<float, 1>, float, CUstream_st*) const;
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceAsync<1>(double*, MonoidF<double, 1>, double, CUstream_st*) const;
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceAsync<1>(int*, MonoidF<int, 1>, int, CUstream_st*) const;
template __host__ CUDART_DEVICE void CuMatrix<uint>::reduceAsync<1>(uint*, MonoidF<uint, 1>, uint, CUstream_st*) const;
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceAsync<1>(ulong*, MonoidF<ulong, 1>, ulong, CUstream_st*) const;
#endif


// reduction with addition (Σ)
template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::sum(cudaStream_t stream ) const {
	//T res[factor];
#ifndef __CUDA_ARCH__
	if(checkDebug(debugRedux)){
		flprintf("this %p %dx%dx%d elems %p dev %p lastMod %s\n", this, m,n,p,elements, tiler.buff(), b_util::modStr(lastMod));
		printColoArray<T>(elements,20);
		printDevArray<T>(tiler.buff(),"EVE",-1,20);
	}

#endif
#ifdef  CuMatrix_Enable_KTS
	T res = reduce( plusBinaryOp<T>(), 0 , stream);
#else
	T res = reduce(Functory<T,plusBinaryOp>::pinch(), 0 , stream);
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
			if(checkDebug(debugRedux)) outln("last idx " << i << ", (elements + idx) = " << (elements + i));
		}
		if(i % p < n) { // verify idx inside meat
			T y = elements[i] - c;
			T t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}/*else {
			if(checkDebug(debugRedux)) outln("skipping idx " << i << " ( i %p == ) " << (i %p));
		}*/
	}
	return sum;
}

// reduction with multiplication (Π)
template<typename T> __host__ CUDART_DEVICE T CuMatrix<T>::prod( cudaStream_t stream ) const {
#ifdef  CuMatrix_Enable_KTS
	T res = reduce( multBinaryOp<T>(), 1.0, stream);
#else
	T res = reduce( Functory<T,multBinaryOp>::pinch(), 1.0, stream);
#endif
	return res;
}


template<typename T> void CuMatrix<T>::featureMeans( CuMatrix<T>& means, bool lv) const {
	DMatrix<T> d_Means, d_X;
	int  roff, coff;

	int tileM, tileN, tileP;
	tiler.tileDims(tileM, tileN, tileP, tdCols);
	int tileCount = DIV_UP(m,tileM);

	for(int i = 0; i < tileCount; i++) {
		tiler.tileLike(d_X, roff,coff, tileM, tileN, tileP, i, tdCols, true);
		if(checkDebug(debugTiler))prlocf("means tiling");
		means.tiler.tileLike(d_Means, roff,coff, tileM, tileN, tileP, i, tdCols, true);
		if(vectorQ()) {
			means.set(0, sum()/length());
		} else {
			featureAvgKernelL(d_Means, d_X, lv);
		}
		means.tiler.syncTile(d_Means, roff, coff);
	}
	means.invalidateHost();
}

template<typename T> void CuMatrix<T>::featureMeansTx( CuMatrix<T>& means) const {
	DMatrix<T> d_Means, d_X;
	int roff,coff;
	int tileM, tileN, tileP;
	tiler.tileDims(tileM, tileN, tileP, tdRows); // todo check this
	int tileCount = DIV_UP(m,_tileM);
	for(int i = 0; i < tileCount; i++) {
		tiler.tileLike(d_X, roff,coff, tileM, tileN, tileP, i, tdRows, true);
		means.tiler.tileLike(d_Means, roff,coff, tileM, tileN, tileP, i, tdRows, true);
		featureAvgTxdKernelL(d_Means, d_X);
		means.tiler.syncTile(d_Means, roff, coff);
	}
	means.invalidateHost();
}

template<typename T> void CuMatrix<T>::featureMeansStreams( CuMatrix<T>& means, bool lv,int nstreams) const {
	DMatrix<T> d_Means, d_X;
	int roff, coff;
	int tileCount = tiler.getTileCount();
	int tileM, tileN, tileP;

	tiler.tileDims(tileM, tileN, tileP, tdCols); // todo check this
	for(int i = 0; i < tileCount; i++) {
		tiler.tileLike(d_X, roff,coff, tileM, tileN, tileP, i, tdCols, true);
		means.tiler.tileLike(d_Means, roff,coff, tileM, tileN, tileP, i, tdCols, true);
		//outln("d_Means " << util<T>::pdm(d_Means));
		featureAvgMultiStreamL(d_Means, d_X, lv, nstreams);
		means.tiler.syncTile(d_Means, roff, coff);
	}
	means.invalidateHost();
}

template<typename T> CuMatrix<T> CuMatrix<T>::featureMeans(bool lv) const {
	CuMatrix<T> means = zeros(n, 1);
	featureMeans(means, lv);
	return means;
}


// matrix is transposed and average calced by doing sum-reductions of each row
// one thread (in x dir) for each row
template<typename T> __global__ void rowSumKernel2(DMatrix<T> sums, const DMatrix<T> xTxd) {
	uint rowIdx = blockIdx.x * blockDim.x + threadIdx.x; // index into column of un-txd source matrix
	if(checkDebug(debugRedux) && rowIdx == 0) {
		util<T>::printRow(xTxd, rowIdx);
	}
	__syncthreads();
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaPeekAtLastError());
#endif
	// T* sdata = SharedMemory<T>();
	DMatrix<T> row(xTxd.elements + xTxd.p * rowIdx, 1, xTxd.n);
	__syncthreads();
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaPeekAtLastError());
#endif
	if(checkDebug(debugRedux) && rowIdx == xTxd.m - 1) {
		prlocf("last row as ");
		util<T>::printRow(row, 0);
	}
	if(rowIdx < xTxd.m) {
		if(checkDebug(debugRedux)) {
			flprintf("reducing row %d\n",rowIdx);
		}
		T sum = 0;
#ifdef CuMatrix_Enable_Cdp
		sum = CuMatrix<T>::reduce(row, Functory<T,plusBinaryOp>::pinch(),0);
//		flprintf("sum %g\n", sum);
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
		util<T>::prdm(d_x);
		flprintf(" with gridX %d and blockX %d\n",gridX,blockX);
	}

	//b_util::pFuncPtrAtts((T*)rowSumKernel2<T>);
	bool valid = b_util::validLaunchQ((void*)rowSumKernel2<T>,dim3(gridX), dim3(blockX));
	if(checkDebug(debugRedux))flprintf("valid %s\n", tOrF(valid));
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
	int col = threadIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
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
	int col = threadIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
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
template<typename T, template <typename> class BinaryOpF> __global__ void rowReductionKernel(DMatrix<T> resVec, BinaryOpF<T> bop, const DMatrix<T> x, int slice, int slices, int stripes)
#else
template<typename T, int StateDim> __global__ void rowReductionKernel(DMatrix<T> resVec, MonoidF<T,StateDim> bop, const DMatrix<T> x, int slice, int slices, int stripes)
#endif
{
	int col = threadIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	ulong soffset = slice * x.m * x.p;
	int toffset = slice * x.m * resVec.p;
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
template __host__ CUDART_DEVICE void CuMatrix<long>::reduceRows<plusBinaryOp>(DMatrix<long>&, DMatrix<long> const&, plusBinaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<ulong>::reduceRows<plusBinaryOp>(DMatrix<ulong>&, DMatrix<ulong> const&, plusBinaryOp<ulong>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<int>::reduceRows<plusBinaryOp>(DMatrix<int>&, DMatrix<int> const&, plusBinaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned int>::reduceRows<plusBinaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, plusBinaryOp<unsigned int>, CUstream_st*);

#else
template __host__ CUDART_DEVICE void CuMatrix<float>::reduceRows(DMatrix<float>&, DMatrix<float> const&, MonoidF<float,1>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::reduceRows(DMatrix<double>&, DMatrix<double> const&, MonoidF<double,1>, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<long>::reduceRows(DMatrix<long>&, DMatrix<long> const&, MonoidF<long,1>, CUstream_st*);
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
