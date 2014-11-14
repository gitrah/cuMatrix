#include "CuMatrix.h"
#include "util.h"
#include "Kernels.h"
#include "debug.h"
#include "caps.h"
#include "MatrixExceptions.h"

#include <pthread.h>

template<typename T, template <typename> class FillOp> __global__ void fillOpKernel(
		FillOp<T> op,
		T* trg,
		uint height,
		uint width,
		uint pitch,
		bool colMajor)
{
	if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0){
		if(checkDebug(debugFill)) {
			flprintf("fillOpKernel trg %p height %d width %d pitch %d\n",trg, height, width,pitch);
		}
	}
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    ulong indexOut = colMajor ? xIndex * pitch + yIndex : yIndex * pitch + xIndex;
    if(indexOut < height * pitch)  {
    	trg[indexOut] = op(indexOut);
    	if(indexOut == height * pitch -1) {
    		if(checkDebug(debugFill)) {
    			flprintf("fillOpKernel last elem idx %d pos %p\n",indexOut, trg +indexOut);
    		}
    	}
    } else {
    	//if(checkDebug(debugFill))flprintf("trg %p skipping xIndex %d, yIndex %d\n", trg, xIndex, yIndex);
    }
}

template<typename T, int StateDim> __global__ void fillOpKernel(
		UnaryOpIndexF<T,StateDim> op,
		T* trg,
		uint height,
		uint width,
		uint pitch,
		bool colMajor)
{

	//(UnaryOpIndexF<T>::operator()*)
	FirstThread {
#ifdef CuMatrix_StatFunc
		flprintf("UnaryOpIndexF<T> op.fn %p\n",op.fn);
#else
		flprintf("UnaryOpIndexF<T> op.operation %p\n",op.operation);
#endif
		flprintf("(op->*oprtr)(5) %f\n",(float)op(5ul));
		if(checkDebug(debugFill)) {
			flprintf("fillOpKernel trg %p height %d width %d pitch %d\n",trg, height, width,pitch);
		}
	}
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    ulong indexOut = colMajor ? xIndex * pitch + yIndex : yIndex * pitch + xIndex;
    if(indexOut < height * pitch)  {
    	trg[indexOut] = op(indexOut); // boom
    	if(indexOut == height * pitch -1) {
    		if(checkDebug(debugFill)) {
    			flprintf("fillOpKernel last elem idx %d pos %p\n",indexOut, trg +indexOut);
    		}
    	}
    } else {
    	//if(checkDebug(debugFill))flprintf("trg %p skipping xIndex %d, yIndex %d\n", trg, xIndex, yIndex);
    }
}

#ifdef  CuMatrix_Enable_KTS
template <typename T> template <template <typename> class FillOp>  __host__ CUDART_DEVICE
void CuMatrix<T>::fillFn( FillOp<T> op, CuMatrix<T>& ret, cudaStream_t stream)
#else
template<typename T> template<int StateDim>  __host__ CUDART_DEVICE
void CuMatrix<T>::fillFn( const UnaryOpIndexF<T,StateDim>& op, CuMatrix<T>& ret, cudaStream_t stream)
#endif
{
	cherr(cudaPeekAtLastError());
	cherr(cudaDeviceSynchronize());
	prlocf("CuMatrix<T>::fillFn(UnaryOpIndexF<T,StateDim> op, CuMatrix<T>& ret, cudaStream_t stream) enter\n");
#ifdef CuMatrix_StatFunc
	flprintf("op.fn %p\n", op.fn);
#else
	#ifdef CuMatrix_Enable_KTS
	#else
		flprintf("op.operation %p\n", op.operation);
	#endif
#endif
	//outln("b_util::pPtrAtts(op.state) ");
	//b_util::pPtrAtts(op.state);

	if(ret.m ==0 || ret.n ==0 ){
		setLastError(badDimensionsEx);
	}
	uint blockW = MIN(ret.n, WARP_SIZE);
	ExecCaps * pcaps = ExecCaps::currCaps();
	if(checkDebug(debugFill)) {
		flprintf("ret %ux%ux%u\n", ret.m, ret.n, ret.p);
		int lastCellIdx = ret.m * ret.p -1 ;
		flprintf("last cell idx %d pos %p ", lastCellIdx, ret.d_elements + lastCellIdx);
		flprintf("maxBlock.y %d, memSharedPerBlock/(specW*sizeof(T) %d\n", pcaps->maxBlock.y,
				pcaps->memSharedPerBlock/(blockW*sizeof(T)));
		flprintf("(ecaps.thrdPerBlock  )/ blockW %d\n", (pcaps->thrdPerBlock   )/ blockW);
	}
	uint blockH = MIN(ret.m, maxH<T>(*pcaps,blockW));
	if(checkDebug(debugFill))flprintf("blockH %d  maxH<T>(*pcaps,blockW) %d\n",blockH,  maxH<T>(*pcaps,blockW));
	int gridY = DIV_UP(ret.m, blockH);
	dim3 grid(DIV_UP(ret.n,blockW), gridY);
	// in case grid y is too big
	int slices = DIV_UP(grid.y, pcaps->maxGrid.y);
	if(checkDebug(debugFill))flprintf("slices %d\n",slices);
	dim3 block(blockW,blockH);
	int sliceGridY = grid.y/ slices;
	uint sliceM = sliceGridY * blockH;
	//if(checkDebug(debugFill))flprintf("init sliceGridY %d sliceM &d\n",sliceGridY, sliceM);
	int offset;
	for(int currSlice =0; currSlice < slices; currSlice++) {
		offset = currSlice * sliceM * ret.p ;
		if(currSlice == slices - 1) {
			if(checkDebug(debugFill))prlocf("last fill slice");
			sliceM = ret.m - (slices - 1 ) * sliceM;
			sliceGridY =  DIV_UP(sliceM, blockH);
		}
		grid.y = sliceGridY;
		if(checkDebug(debugFill)){
			flprintf("sliceGridY %d\n",sliceGridY);
			flprintf("fillFn slice %d on mat offset %d %dX%d(X%d) (ret.d_elements+ offset %p)\n", currSlice, offset, sliceM, ret.n, ret.p, (ret.d_elements+ offset));
			 b_util::prd3(grid, " grid of ");
			 b_util::prd3(block, "block of");
		}
		fillOpKernel<<<grid, block, 0, stream>>>(op, ret.d_elements + offset, sliceM, ret.n, ret.p, ret.colMajor);

	}
	cherr(cudaDeviceSynchronize());
	//ret.invalidateHost();
}
#ifdef  CuMatrix_Enable_KTS

template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn<stepFiller>(stepFiller<float>, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn<constFiller>(constFiller<float>, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn<sinFiller>(sinFiller<float>, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn<cosFiller>(cosFiller<float>, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn<sequenceFiller>(sequenceFiller<float>, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn<seqModFiller>(seqModFiller<float>, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn<diagonalFiller>(diagonalFiller<float>, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn<increasingColumnsFiller>(increasingColumnsFiller<float>, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn<increasingRowsFiller>(increasingRowsFiller<float>, CuMatrix<float>&, CUstream_st*);

template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn<stepFiller>(stepFiller<double>, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn<constFiller>(constFiller<double>, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn<sinFiller>(sinFiller<double>, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn<cosFiller>(cosFiller<double>, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn<sequenceFiller>(sequenceFiller<double>, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn<seqModFiller>(seqModFiller<double>, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn<diagonalFiller>(diagonalFiller<double>, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn<increasingColumnsFiller>(increasingColumnsFiller<double>, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn<increasingRowsFiller>(increasingRowsFiller<double>, CuMatrix<double>&, CUstream_st*);

template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn<oneOverFiller>(oneOverFiller<float>, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn<oneOverFiller>(oneOverFiller<double>, CuMatrix<double>&, CUstream_st*);

#else
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn(const UnaryOpIndexF<float, 0>&, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn(const UnaryOpIndexF<double, 0>&, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn(const UnaryOpIndexF<float, 1>&, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn(const UnaryOpIndexF<double, 1>&, CuMatrix<double>&, CUstream_st*);
#endif

/*
template<typename T> template<template <typename> class FillOp> __host__ CUDART_DEVICE
void CuMatrix<T>::fillFn(FillOp<T> op, CuMatrix<T>& ret, cudaStream_t stream) {

	if(ret.m ==0 || ret.n ==0 ){
		setLastError(badDimensionsEx);
	}
	uint blockW = MIN(ret.n, WARP_SIZE);
	ExecCaps * pcaps = ExecCaps::currCaps();
	if(checkDebug(debugFill)) {
		flprintf("ret %ux%ux%u\n", ret.m, ret.n, ret.p);
		int lastCellIdx = ret.m * ret.p -1 ;
		flprintf("last cell idx %d pos %p ", lastCellIdx, ret.d_elements + lastCellIdx);
		flprintf("maxBlock.y %d, memSharedPerBlock/(specW*sizeof(T) %d\n", pcaps->maxBlock.y,
				pcaps->memSharedPerBlock/(blockW*sizeof(T)));
		flprintf("(ecaps.thrdPerBlock  )/ blockW %d\n", (pcaps->thrdPerBlock   )/ blockW);
	}
	uint blockH = MIN(ret.m, maxH<T>(*pcaps,blockW));
	if(checkDebug(debugFill))flprintf("blockH %d  maxH<T>(*pcaps,blockW) %d\n",blockH,  maxH<T>(*pcaps,blockW));
	int gridY = DIV_UP(ret.m, blockH);
	dim3 grid(DIV_UP(ret.n,blockW), gridY);
	// in case grid y is too big
	int slices = DIV_UP(grid.y, pcaps->maxGrid.y);
	if(checkDebug(debugFill))flprintf("slices %d\n",slices);
	dim3 block(blockW,blockH);
	int sliceGridY = grid.y/ slices;
	int sliceM = sliceGridY * blockH;
	//if(checkDebug(debugFill))flprintf("init sliceGridY %d sliceM &d\n",sliceGridY, sliceM);
	int offset;
	for(int currSlice =0; currSlice < slices; currSlice++) {
		offset = currSlice * sliceM * ret.p ;
		if(currSlice == slices - 1) {
			if(checkDebug(debugFill))prlocf("last fill slice");
			sliceM = ret.m - (slices - 1 ) * sliceM;
			sliceGridY =  DIV_UP(sliceM, blockH);
		}
		grid.y = sliceGridY;
		if(checkDebug(debugFill)){
			flprintf("sliceGridY %d\n",sliceGridY);
			flprintf("fillFn slice %d on mat offset %d %dX%d(X%d) (ret.d_elements+ offset %p)\n", currSlice, offset, sliceM, ret.n, ret.p, (ret.d_elements+ offset));
			 b_util::prd3(grid, " grid of ");
			 b_util::prd3(block, "block of");
		}
		fillOpKernel<<<grid, block, 0, stream>>>(op, ret.d_elements + offset, sliceM, ret.n, ret.p, ret.colMajor);
		flprintf("cudaPeekAtLastError() %u\n", cudaPeekAtLastError());
		cherr(cudaPeekAtLastError());
	}
	cherr(cudaDeviceSynchronize());
	//ret.invalidateHost();
}
*/

template<typename T, int StateDim> __global__ void fillOpNsbKernel(
		UnaryOpIndexF<T,StateDim> op,
		T* trg,
		uint height,
		uint width,
		uint pitch,
		bool colMajor)
{
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.x + threadIdx.y;
    uint indexOut = colMajor ? xIndex * pitch + yIndex : yIndex * pitch + xIndex;
    uint ip;
	if(xIndex < width )
		for(int i = 0; i < blockDim.x; i+= blockDim.y) {
			if( i + yIndex < height) {
				ip = i * pitch;
				trg[ip + indexOut] = op(indexOut + ip);
			}
		}
}

template<typename T, template <typename> class FillOp> __global__ void fillOpNsbKernel(
		FillOp<T> op,
		T* trg,
		uint height,
		uint width,
		uint pitch,
		bool colMajor)
{
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.x + threadIdx.y;
    uint indexOut = colMajor ? xIndex * pitch + yIndex : yIndex * pitch + xIndex;
    uint ip;
	if(xIndex < width )
		for(int i = 0; i < blockDim.x; i+= blockDim.y) {
			if( i + yIndex < height) {
				ip = i * pitch;
				trg[ip + indexOut] = op(indexOut + ip);
			}
		}
}

#ifdef  CuMatrix_Enable_KTS
template <typename T> template <template <typename> class FillOp> __host__ CUDART_DEVICE
void CuMatrix<T>::fillFnNsb(FillOp<T> op, CuMatrix<T>& ret, int w2h, cudaStream_t stream ){
#else
template<typename T> template<int StateDim>__host__ CUDART_DEVICE
	void CuMatrix<T>::fillFnNsb(UnaryOpIndexF<T,StateDim> op, CuMatrix<T>& ret, int w2h, cudaStream_t stream) {
#endif

	if(ret.m ==0 || ret.n ==0 ){
		setLastError(badDimensionsEx);
	}
	if(checkDebug(debugFill))printf("fillFnNsb m %d n %d colmaj %s\n", ret.m, ret.n , tOrF(ret.colMajor));

	uint blockW = MIN(b_util::nextPowerOf2(ret.n), WARP_SIZE);
	if(checkDebug(debugFill))printf("fillFn m %d n %d colmaj %s\n", ret.m, ret.n , tOrF(ret.colMajor));
	//uint blockH = MIN(ret.m, maxH<T>(*ExecCaps::currCaps(),blockW));
	uint blockH = blockW/w2h;
	if(checkDebug(debugFill))printf("blockH %d\n",blockH);
	dim3 grid(DIV_UP(ret.n, blockW), DIV_UP(ret.m,blockW));
	dim3 block(blockW,blockH);
	if(checkDebug(debugFill))printf("fillFnNsb on mat d_elements %p %dx%d\n",ret.d_elements, ret.m,ret.n);
	fillOpNsbKernel<<<grid,block,0,stream>>>(op, ret.d_elements, ret.m,ret.n,ret.p, ret.colMajor);
	ret.invalidateHost();
}

#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFnNsb<oneOverFiller>(oneOverFiller<float>, CuMatrix<float>&, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFnNsb<oneOverFiller>(oneOverFiller<double>, CuMatrix<double>&, int, CUstream_st*);
#else
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFnNsb<0>(UnaryOpIndexF<float, 0>, CuMatrix<float>&, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFnNsb<0>(UnaryOpIndexF<double, 0>, CuMatrix<double>&, int, CUstream_st*);
#endif
/*
template<typename T> template<template <typename> class FillOp> __host__ CUDART_DEVICE
void CuMatrix<T>::fillFnNsb(FillOp<T> op, CuMatrix<T>& ret, int w2h, cudaStream_t stream) {

	if(ret.m ==0 || ret.n ==0 ){
		setLastError(badDimensionsEx);
	}
	if(checkDebug(debugFill))printf("fillFnNsb m %d n %d colmaj %s\n", ret.m, ret.n , tOrF(ret.colMajor));

	uint blockW = MIN(b_util::nextPowerOf2(ret.n), WARP_SIZE);
	if(checkDebug(debugFill))printf("fillFn m %d n %d colmaj %s\n", ret.m, ret.n , tOrF(ret.colMajor));
	//uint blockH = MIN(ret.m, maxH<T>(*ExecCaps::currCaps(),blockW));
	uint blockH = blockW/w2h;
	if(checkDebug(debugFill))printf("blockH %d\n",blockH);
	dim3 grid(DIV_UP(ret.n, blockW), DIV_UP(ret.m,blockW));
	dim3 block(blockW,blockH);
	if(checkDebug(debugFill))printf("fillFnNsb on mat d_elements %p %dx%d\n",ret.d_elements, ret.m,ret.n);
	fillOpNsbKernel<<<grid,block,0,stream>>>(op, ret.d_elements, ret.m,ret.n,ret.p, ret.colMajor);
	ret.invalidateHost();
}


//#ifdef CUTMPLT
template void CuMatrix<float>::fillFn<oneOverIdxFiller>(oneOverIdxFiller<float>, CuMatrix<float>&, CUstream_st*);
template void CuMatrix<float>::fillFn<constFiller>(constFiller<float>, CuMatrix<float>&, CUstream_st*);
template void CuMatrix<float>::fillFn<sinFiller>(sinFiller<float>, CuMatrix<float>&, CUstream_st*);
template void CuMatrix<float>::fillFn<cosFiller>(cosFiller<float>, CuMatrix<float>&, CUstream_st*);
template void CuMatrix<float>::fillFn<sequenceFiller>(sequenceFiller<float>, CuMatrix<float>&, CUstream_st*);
template void CuMatrix<float>::fillFn<seqModFiller>(seqModFiller<float>, CuMatrix<float>&, CUstream_st*);
template void CuMatrix<float>::fillFn<diagonalFiller>(diagonalFiller<float>, CuMatrix<float>&, CUstream_st*);
template void CuMatrix<float>::fillFn<increasingColumnsFiller>(increasingColumnsFiller<float>, CuMatrix<float>&, CUstream_st*);
template void CuMatrix<float>::fillFn<increasingRowsFiller>(increasingRowsFiller<float>, CuMatrix<float>&, CUstream_st*);

template void CuMatrix<double>::fillFn<oneOverIdxFiller>(oneOverIdxFiller<double>, CuMatrix<double>&, CUstream_st*);
template void CuMatrix<double>::fillFn<constFiller>(constFiller<double>, CuMatrix<double>&, CUstream_st*);
template void CuMatrix<double>::fillFn<sinFiller>(sinFiller<double>, CuMatrix<double>&, CUstream_st*);
template void CuMatrix<double>::fillFn<cosFiller>(cosFiller<double>, CuMatrix<double>&, CUstream_st*);
template void CuMatrix<double>::fillFn<sequenceFiller>(sequenceFiller<double>, CuMatrix<double>&, CUstream_st*);
template void CuMatrix<double>::fillFn<seqModFiller>(seqModFiller<double>, CuMatrix<double>&, CUstream_st*);
template void CuMatrix<double>::fillFn<diagonalFiller>(diagonalFiller<double>, CuMatrix<double>&, CUstream_st*);
template void CuMatrix<double>::fillFn<increasingColumnsFiller>(increasingColumnsFiller<double>, CuMatrix<double>&, CUstream_st*);
template void CuMatrix<double>::fillFn<increasingRowsFiller>(increasingRowsFiller<double>, CuMatrix<double>&, CUstream_st*);
//#endif
template void CuMatrix<float>::fillFnNsb<oneOverIdxFiller>(oneOverIdxFiller<float>, CuMatrix<float>&,int, CUstream_st*);
template void CuMatrix<double>::fillFnNsb<oneOverIdxFiller>(oneOverIdxFiller<double>, CuMatrix<double>&,int, CUstream_st*);
*/

/*
template void CuMatrix<double>::fillFn<eXfiller>(eXfiller<double>, CuMatrix<double>&, CUstream_st*);
template void CuMatrix<float>::fillFn<eXfiller>(eXfiller<float>, CuMatrix<float>&, CUstream_st*);
*/

template <typename T> CuMatrix<T> CuMatrix<T>::freeform(int cols, const T* vals, ulong count ) {
	int rows =count/cols;
	CuMatrix<T> mat(rows,cols,false,true);
	cherr(cudaMemcpy(mat.d_elements,vals, count*sizeof(T),cudaMemcpyHostToDevice));
	mat.invalidateHost();
 	HDCopied++;
	MemHdCopied += count * sizeof(T);
	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::fromScalar(T t, bool colMajor) {
	CuMatrix<T> mat(1,1,false,true);
	mat.colMajor = colMajor;
	mat.set(0, t);
	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::fill(T t, uintPair dims, bool colMajor, cudaStream_t stream) {
	return fill(t, dims.first, dims.second,colMajor,stream);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::fill(T t, uint nRows, uint nCols, bool colMajor, cudaStream_t stream) {
	constFiller<T> filler = Functory<T,constFiller>::pinch(t);
#ifdef CuMatrix_StatFunc
	flprintf("filler: fn %p  state %f\n", filler.fn, (float)filler.state);
#else
	#ifdef  CuMatrix_Enable_KTS
		flprintf("filler: %f\n",  (float)filler.state);
	#else
		flprintf("filler: %p  %f\n", filler.operation, (float)filler.state);
	#endif

#endif
	CuMatrix<T> mat(nRows,nCols,false,true);
	mat.colMajor=colMajor;
	fillFn(filler,mat);
/*
 *
  	UnaryOpIndexF<T,1> uof(filler);
	flprintf("uof: %p  %f\n", uof.operation, (float)uof.state);
	fillFn<1>(uof,mat);

#ifdef __CUDA_ARCH__
	fillFn(&filler, mat);
#else
	typename UnaryOpIndexF<T>::uintFunction theOpr =  fillas<T>::ops[id_constFiller]; // (typename UnaryOpIndexF<T>::uintFunction) &constFiller<T>::operator();
	flprintf("CuMatrix<T>::fill typename UnaryOpIndexF<T>::uintFunction theOpr %p\n",theOpr);
	constFiller<T>* pfiller = null;
	T (constFiller<T>::*method)(uint) const; // adding (uint) const noncompyl
	checkCudaError(cudaMalloc(&pfiller, sizeof(constFiller<T>)));
	checkCudaError(cudaMalloc((void**)&method, sizeof(typename UnaryOpIndexF<T>::uintFunction)));
	flprintf("CuMatrix<T>::fill pfiller %p\n",pfiller);
	constFiller<T>::instance(pfiller, t, method);
	flprintf("CuMatrix<T>::fill after instance pfiller %p\n",pfiller);
	//copyFillFunctor(pfiller, &filler);
	fillFn(pfiller, mat, (typename UnaryOpIndexF<T>::uintFunction) method);
	//b_util::freeOnDevice(pfiller);
	cherr(cudaFree(pfiller));
	cherr(cudaFree((void*)method));
#endif
*/
	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::zeros(uint nRows, uint nCols, bool colMajor) {

	flprintf("zeros %uX%u\n", nRows,nCols);
	return fill(0,nRows,nCols, colMajor);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::zeros(uintPair dims, bool colMajor) {
	return fill(0,dims.first, dims.second);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::ones(uint nRows, uint nCols, bool colMajor) {
	if(checkDebug(debugFill)) flprintf("\n\nones(%u, %u)\n", nRows, nCols);
	return fill(1,nRows,nCols);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::ones(uintPair dims, bool colMajor) {
	return fill(1, dims.first, dims.second);
}

template <typename T>__host__ CUDART_DEVICE  CuMatrix<T> CuMatrix<T>::sin(uint m, uint n, T amplitude, T period, T phase, bool colMajor) {
	sinFiller<T> filler = Functory<T, sinFiller>::pinch(amplitude,period,phase);
	CuMatrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
	fillFn(filler, mat);
	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::cos(uint m, uint n, T amplitude, T period, T phase, bool colMajor) {
	cosFiller<T> filler = Functory<T, cosFiller>::pinch(amplitude,period,phase);
	CuMatrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
	fillFn(filler, mat);
	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::sin(uintPair dims, T amplitude, T period, T phase, bool colMajor) {
	return sin(dims.first, dims.second,amplitude,period,phase,colMajor);
}
template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::cos(uintPair dims, T amplitude, T period, T phase, bool colMajor) {
	return cos(dims.first, dims.second,amplitude,period,phase,colMajor);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::diagonal(uint dim, T val, bool colMajor) {
	if(dim > MaxDim) {
		setLastError(badDimensionsEx);
	}
	assert((dim <= MaxDim));

	diagonalFiller<T> filler = Functory<T, diagonalFiller>::pinch(val,dim);

	CuMatrix<T> mat(dim,dim,false,true);
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat);
#else
	fillFn<2>(filler, mat);
#endif
	return mat;
}

template <typename T> CuMatrix<T> CuMatrix<T>::diagonal(uint dim, const T* val, bool colMajor) {
	return diagonal(dim, *val,colMajor);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::identity(uint dim, bool colMajor) {
	return diagonal(dim, static_cast<T>( 1), colMajor);
}

template <typename T> CuMatrix<T> CuMatrix<T>::randn(uint rows, uint cols, T epsilon, bool colMajor) {
	if(colMajor) {
		dthrow(notImplemented());
	}
	CuMatrix<T> ret(rows,cols, false,true);
	DMatrix<T> d_ret;
	ret.asDmatrix(d_ret,false);
	randn(d_ret, epsilon);
	ret.lastMod = mod_device;
	return ret;
}

template <typename T> CuMatrix<T> CuMatrix<T>::randn( const uintPair& dims, float epsilon, bool colMajor) {
	return (randn(dims.first, dims.second, epsilon, colMajor));
}
template <typename T> CuMatrix<T> CuMatrix<T>::randn( const uintPair& dims, bool colMajor) {
	return (randn(dims.first, dims.second, colMajor));
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::span(T start, T end, uint m, uint n, bool colMajor) {
	spanFiller<T> filler= Functory<T, spanFiller>::pinch(start,end,m*n);
	CuMatrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat);
#else
	fillFn<3>(filler, mat);
#endif
	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::sequence(T start, uint m, uint n, bool colMajor) {
	sequenceFiller<T> filler=  Functory<T, sequenceFiller>::pinch(start);
	CuMatrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat);
#else
	fillFn<1>(filler, mat);
#endif
	return mat;
}
/*

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::sequenceScale(T start, T scale, uint m, uint n, bool colMajor) {
	sequenceScaleFiller<T> filler;
	filler.phase() = start;
	filler.scale() = scale;
	CuMatrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
	fillFn<2>(&filler, mat);
	return mat;
}
*/

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::seqMod(T start, T mod, uint m, uint n, bool colMajor) {
	seqModFiller<T> filler=  Functory<T, seqModFiller>::pinch(start,mod);
	CuMatrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat);
#else
	fillFn<2>(filler, mat);
#endif
	return mat;
}


template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::increasingColumns(T start, int rows, int cols, bool colMajor ) {
	increasingColumnsFiller<T> filler =  Functory<T, increasingColumnsFiller>::pinch(start,cols);
	CuMatrix<T> mat(rows,cols,false,true);
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	CuMatrix<T>::fillFn(filler, mat);
#else
	CuMatrix<T>::fillFn<2>(filler, mat);
#endif

	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::increasingRows(T start, int rows, int cols, bool colMajor ) {
	increasingRowsFiller<T> filler = 	Functory<T, increasingRowsFiller>::pinch(start,cols);
	CuMatrix<T> mat(rows,cols,false,true);
	mat.colMajor= colMajor;
	fillFn(filler, mat);
	return mat;
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::extrude(uint depth) const {
	if (m != 1) {
#ifndef __CUDA_ARCH__
		dthrow (notRowVector());
#else
		setLastError(notRowVectorEx);
#endif
	}
	CuMatrix<T> ret(*this);
	for (uint r = 0; r < depth; r++) {
		ret = ret /= *this;
	}
	return ret;
}

#include "CuMatrixInster.cu"

