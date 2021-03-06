#include "CuMatrix.h"
#include "util.h"
#include "Kernels.h"
#include "debug.h"
#include "caps.h"
#include "MemMgr.h"
#include "MatrixExceptions.h"

#include <pthread.h>


#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class FillOp> __global__ void fillOpKernel(
		FillOp<T> op,
		T* trg,
		int height,
		int width,
		int pitch,
		int rowStart,
		int colStart,
		bool colMajor)
#else
template<typename T, int StateDim> __global__ void fillOpKernel(
		UnaryOpIndexF<T,StateDim> op,
		T* trg,
		int height,
		int width,
		int pitch,
		int rowStart,
		int colStart,
		bool colMajor)
#endif
{
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    ulong indexOut = colMajor ? xIndex * pitch + yIndex : yIndex * pitch + xIndex;
    ulong indexIn = colMajor ? (xIndex + colStart) * height + yIndex + rowStart: (yIndex + rowStart) * width + xIndex;
	if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0){
		if(checkDebug(debugFill)) {
			//flprintf("fillOpKernel %f -> trg %p height %d width %d pitch %d\n", (float)op(indexOut), trg, height, width,pitch);
		}
	}
    if(indexOut < height * pitch)  {
    	trg[indexOut] = op(indexIn);
    	if(indexOut == height * pitch -1) {
    		//if(checkDebug(debugFill))flprintf("fillOpKernel %f -> last elem idx %d pos %p\n",(float)op(indexOut),indexOut, trg +indexOut);
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
	//c herr(cudaPeekAtLastError());
	//cherr(cudaDeviceSynchronize());
	if(checkDebug(debugFill))prlocf("CuMatrix<T>::fillFn(UnaryOpIndexF<T,StateDim> op, CuMatrix<T>& ret, cudaStream_t stream) enter\n");
#ifdef CuMatrix_StatFunc
	if(checkDebug(debugFill))flprintf("op.fn %p\n", op.fn);
#else
	#ifdef CuMatrix_Enable_KTS
	#else
	if(checkDebug(debugFill))flprintf("op.operation %p\n", op.operation);
	#endif
#endif

#ifndef __CUDA_ARCH__
	if(checkDebug(debugFill))outln("ret " << ret.toShortString());
#endif

	if(ret.m ==0 || ret.n ==0 ){
		setLastError(badDimensionsEx);
	}

	assert(ret.tiler.hasDmemQ());

	uint blockW = MIN(ret.n, WARP_SIZE);
	ExecCaps * pcaps = ExecCaps::currCaps();
	int tileCount = ret.tiler.getTileCount();
	if(checkDebug(debugFill)) {
		flprintf("ret %ux%ux%u  ret.tiler.getTileCount() %d\n", ret.m, ret.n, ret.p,  tileCount);
		flprintf("current device %d\n", ExecCaps::currDev());
		flprintf("maxBlock.y %d, memSharedPerBlock/(specW*sizeof(T) %d\n", pcaps->maxBlock.y,
				pcaps->memSharedPerBlock/(blockW*sizeof(T)));
		flprintf("(ecaps.thrdPerBlock  )/ blockW %d\n", (pcaps->thrdPerBlock   )/ blockW);
	}

	DMatrix<T> d_A;
	int tileM = 0, tileN=0, tileP = 0;
	int roff = 0, coff = 0 ;

	int lastGpu = -1;
	int gpuCount = ret.tiler.countGpus();
	int orgDevice = ExecCaps::currDev();
	DevStream** devStreams = nullptr;
	if(checkDebug(debugMem))usedDevMem();
	if(gpuCount > 1) {
		assert(!stream);
		devStreams = (DevStream** ) malloc(gpuCount * sizeof(DevStream*));
		for(int i =0 ; i < gpuCount; i++) {
			lastGpu = ret.tiler.nextGpu(lastGpu);

			devStreams[i] = new DevStream(lastGpu);
			if(checkDebug(debugFill))flprintf("created devStreams %d at %p\n",i,devStreams[i]);
		}
	}
	if(checkDebug(debugMem))usedDevMem();
	if (checkDebug(debugFill)) {
		char buff[33];
		buff[32]=0;
		b_util::printBinInt(buff, ret.tiler.gpuMask);
		flprintf("ret.tiler.gpuMask bits %s\n",buff);
	}
	if(checkDebug(debugMem))usedDevMem();
	lastGpu = -1;
	cudaStream_t currStream = nullptr;
	cherr(cudaDeviceSynchronize());
	for(int tile = 0; tile < tileCount; tile++) {
		if(checkDebug(debugMem))usedDevMem();

		currStream = gpuCount == 1 ? stream : devStreams[ tile % ret.tiler.gpuModulus ]->stream;
		if(checkDebug(debugFill)) flprintf("tile %d currGpu %d lastGpu %d currStream %p\n",tile, ExecCaps::currDev(), lastGpu, currStream);
		cherr(cudaDeviceSynchronize());

		lastGpu= ret.tiler.tile1D(d_A, roff, coff, tileM, tileN, tileP, tile, ret.tiler.tileD, false, lastGpu, currStream);
		if(ExecCaps::currDev() != lastGpu) {
			ExecCaps_setDevice(lastGpu);
		}
		cherr(cudaDeviceSynchronize());

		if(checkDebug(debugFill)) {
#ifndef __CUDA_ARCH__
			outln("dev " << ExecCaps::currDev() << " zeroing d_A " << util<T>::pdm(d_A));
			outln("b_util::getDevice(d_A.elements) " << b_util::getDevice(d_A.elements));
#endif
			// cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
			cherr(cudaGetLastError());
			MemMgr<T>::checkValid(d_A.elements + d_A.p * d_A.m -1 );
			cherr(cudaMemset2DAsync((void*)d_A.elements, d_A.p * sizeof(T), 0, d_A.n * sizeof(T), d_A.m, stream));
			//cherr( cudaMemset((void*)d_A.elements,0, (long)d_A.m * d_A.p * sizeof(T)));
		}
		cherr(cudaDeviceSynchronize());

		if(checkDebug(debugFill))flprintf("tileM %d tileN %d tile %d lastGpu %u\n", tileM, tileN, tile, lastGpu);
		if(checkDebug(debugFill))flprintf("roff %u coff %u\n",roff, coff);
		if(checkDebug(debugMem))usedDevMem();
		if(checkDebug(debugFill)) {
			flprintf("after ret.tiler.tile1D for tile %d; roff %u coff %u\n", tile, roff, coff);
			prlocf("front of buffer ");
			b_util::pPtrAtts(d_A.elements);

			ulong eob = d_A.m * d_A.n-1;
			flprintf("end of buffer %lu ", eob);
			if(checkDebug(debugMem))usedDevMem();
			prlocf("d_A.elements + eob ");
			b_util::pPtrAtts(d_A.elements + eob );
		}

		uint blockH = MIN(d_A.m, maxH<T>(*pcaps,blockW));
		if(checkDebug(debugFill))flprintf("blockH %d  maxH<T>(*pcaps,blockW) %d\n",blockH,  maxH<T>(*pcaps,blockW));
		int gridY = DIV_UP(d_A.m, blockH);
		dim3 grid(DIV_UP(d_A.n,blockW), gridY);

		// when grid y is too big
		int slices = DIV_UP(grid.y, pcaps->maxGrid.y);
		if(checkDebug(debugFill))flprintf("slices %d\n",slices);
		dim3 block(blockW,blockH);
		// todo make distinction between gram tiles and thread grid slices mobvios
		int sliceGridY = grid.y/ slices;
		int sliceM = sliceGridY * blockH;
		if( checkDebug(debugFill)){
			if(slices)flprintf("init sliceGridY %d sliceM %d\n",sliceGridY, sliceM);
		}
		ulong sliceOffset;
		if(checkDebug(debugMem))usedDevMem();
		for(int currSlice =0; currSlice < slices; currSlice++) {
			sliceOffset = currSlice * sliceM * d_A.p ;
			if(currSlice == slices - 1) {
				if(checkDebug(debugFill))prlocf("last fill slice\n");
				sliceM = d_A.m - (slices - 1 ) * sliceM;
				sliceGridY =  DIV_UP(sliceM, blockH);
			}
			grid.y = sliceGridY;
			if(checkDebug(debugMem))usedDevMem();

			if(checkDebug(debugFill)){
				flprintf("currStream %p, sliceGridY %d\n",currStream, sliceGridY);

				if(slices > 1 ) {
					flprintf(
							"sliceGridY %d currSlice %d sliceM %d sliceOffset %d\n",
							stream, sliceGridY, currSlice, sliceOffset);
				} else {
					flprintf(
							"gpu %d tile %d currSlice %d on mat offset %d sliceM %d X ret.n %d( X ret.p %d)\n(d_A.elements+ offset %p)\n",
							ExecCaps::currDev(), tile, currSlice, sliceOffset,
							sliceM, ret.n, ret.p, (d_A.elements + sliceOffset));
				}

				 b_util::prd3(grid, " grid of ");
				 b_util::prd3(block, "block of");
			}
#ifndef __CUDA_ARCH__
			outln("d_A.elements + sliceOffset " <<  d_A.elements + sliceOffset << ", sliceM " << sliceM << ", d_A.n " <<  d_A.n <<
					", tileP " << tileP << ", roff " << roff << ", coff " <<coff);
#endif
			fillOpKernel<<<grid, block, 0, currStream>>>(
					op, d_A.elements + sliceOffset, sliceM, d_A.n, /*ret.tiler.tileP*/d_A.p, roff, coff, ret.colMajor);

			cherr(cudaDeviceSynchronize());
			if(checkDebug(debugMem))usedDevMem();
			if(checkDebug(debugFill)){
				flprintf("post fill tile %d slice %d\n", tile, currSlice);
			//	printArray<<<1,1>>>(d_A.elements, 10);
			}
		}
		if(checkDebug(debugFill)) flprintf("done with slices, syncing tile %d, before call roff %u coff %u\n", tile, roff,coff);

		//cherr(cudaDeviceSynchronize());

		if(checkDebug(debugMem))usedDevMem();

		if(stream) {
			cherr(cudaStreamSynchronize(stream));
		}
		else if(!devStreams){
			cherr(cudaDeviceSynchronize());
		}

		if(checkDebug(debugFill))flprintf("after tile-fill, dtile %d...\n",tile);
		//printArrayDiagNeq(d_A.elements + sliceOffset,d_A.p,d_A.n,(T)1);

		// + roff term to start at (offset to) the column for the diagonal at that tile's row
		//printArrayNe(d_A.elements + sliceOffset,d_A.p * 2, (T) 0);
		//printArrayDiagNeq(d_A.elements + sliceOffset + roff,d_A.p, MIN(d_A.n,d_A.m), 1, (T) 1);

		ret.tiler.syncTile(d_A, roff, coff, currStream);
		cherr(cudaGetLastError());
		T* retHostTile= ret.elements + ret.tiler.offset(roff,coff);
		if(checkDebug(debugFill))flprintf("\n\nafter host-sync, hostTile %p...\n",retHostTile);
		//countColoArrayDiagNe( retHostTile + roff, ret.p, MIN(d_A.n,d_A.m), (T)1);
		//printColoArrayDiagNe( retHostTile + roff, ret.p, MIN(d_A.n,d_A.m), (T)1);
		//flprintf("printColorArray(retHostTile %p + roff %d - 5, 10)\n", retHostTile + roff - 5, roff);
		//printColoArray(retHostTile + roff - ( roff > 5 ? 5 : 0), 10);

		if(checkDebug(debugMem))usedCurrMem();
	}
	if(gpuCount > 1) {
		for(int i =0 ; i < gpuCount; i++) {
			if(checkDebug(debugFill))flprintf("syncing stream %d at %p\n", i , devStreams[i]);

			cherr(devStreams[i]->sync());
			//cherr(cudaStreamSynchronize(streams[i]));
			//cherr(cudaStreamDestroy(streams[i]));
			delete devStreams[i];
		}
		free(devStreams);
	}

	if(ret.tiler.tileSize >= ret.tiler.m_size) {
		ret.invalidateHost();
	} else {
		ret.invalidateDevice();
	}

	if(checkDebug(debugFill))prlocf("fillFn exit\n");

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
template __host__ CUDART_DEVICE void CuMatrix<unsigned long>::fillFn<oneOverFiller>(oneOverFiller<unsigned long>, CuMatrix<unsigned long>&, CUstream_st*);

#else
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn(const UnaryOpIndexF<float, 0>&, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn(const UnaryOpIndexF<double, 0>&, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFn(const UnaryOpIndexF<float, 1>&, CuMatrix<float>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFn(const UnaryOpIndexF<double, 1>&, CuMatrix<double>&, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned long>::fillFn<0>(UnaryOpIndexF<unsigned long, 0> const&, CuMatrix<unsigned long>&, CUstream_st*);
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
		flprintf("last cell idx %d pos %p ", lastCellIdx, ret.tiler.currBuffer() + lastCellIdx);
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
			flprintf("fillFn slice %d on mat offset %d %dX%d(X%d) (ret.tiler.currBuffer()+ offset %p)\n", currSlice, offset, sliceM, ret.n, ret.p, (ret.tiler.currBuffer()+ offset));
			 b_util::prd3(grid, " grid of ");
			 b_util::prd3(block, "block of");
		}
		fillOpKernel<<<grid, block, 0, stream>>>(op, ret.tiler.currBuffer() + offset, sliceM, ret.n, ret.p, ret.colMajor);
		flprintf("cudaPeekAtLastError() %u\n", cudaPeekAtLastError());
		cherr(cudaPeekAtLastError());
	}
	cherr(cudaDeviceSynchronize());
	//ret.invalidateHost();
}
*/

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class FillOp> __global__ void fillOpNsbKernel(
		FillOp<T> op,
		T* trg,
		int height,
		int width,
		int pitch,
		bool colMajor)
#else
template<typename T, int StateDim> __global__ void fillOpNsbKernel(
		UnaryOpIndexF<T,StateDim> op,
		T* trg,
		int height,
		int width,
		int pitch,
		bool colMajor)
#endif
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
	if(checkDebug(debugFill))printf("fillFnNsb on mat tiler.currBuffer() %p %dx%d\n",ret.tiler.currBuffer(), ret.m,ret.n);
	fillOpNsbKernel<<<grid,block,0,stream>>>(op, ret.tiler.currBuffer(), ret.m,ret.n,ret._tileP, ret.colMajor);
	ret.invalidateHost();
}

#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFnNsb<oneOverFiller>(oneOverFiller<float>, CuMatrix<float>&, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFnNsb<oneOverFiller>(oneOverFiller<double>, CuMatrix<double>&, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned long>::fillFnNsb<oneOverFiller>(oneOverFiller<unsigned long>, CuMatrix<unsigned long>&, int, CUstream_st*);
#else
template __host__ CUDART_DEVICE void CuMatrix<float>::fillFnNsb<0>(UnaryOpIndexF<float, 0>, CuMatrix<float>&, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<double>::fillFnNsb<0>(UnaryOpIndexF<double, 0>, CuMatrix<double>&, int, CUstream_st*);
template __host__ CUDART_DEVICE void CuMatrix<unsigned long>::fillFnNsb<0>(UnaryOpIndexF<unsigned long, 0>, CuMatrix<unsigned long>&, int, CUstream_st*);
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
	if(checkDebug(debugFill))printf("fillFnNsb on mat tiler.currBuffer() %p %dx%d\n",ret.tiler.currBuffer(), ret.m,ret.n);
	fillOpNsbKernel<<<grid,block,0,stream>>>(op, ret.tiler.currBuffer(), ret.m,ret.n,ret.p, ret.colMajor);
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
	assert(mat.tiler.tileSize == mat.tiler.m_size);
	cherr(cudaMemcpy(mat.tiler.currBuffer(),vals, count*sizeof(T),cudaMemcpyHostToDevice));
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

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::fill(intPair dims, T t, int gpuMask, TileDirection tileD, bool colMajor, cudaStream_t stream) {
	return fill(t, dims.first, dims.second,gpuMask,tileD,colMajor,stream);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::fill(int nRows, int nCols, T t, int gpuMask, TileDirection tileD, bool colMajor, cudaStream_t stream) {
	constFiller<T> filler = Functory<T,constFiller>::pinch(t);
#ifdef CuMatrix_StatFunc
	if(checkDebug(debugFill))flprintf("filler: fn %p  state %f colmajor %d\n", filler.fn, (float)filler.state, colMajor);
#else
	#ifdef  CuMatrix_Enable_KTS
		if(checkDebug(debugFill))flprintf("filler: %f\n",  (float)filler.state);
	#else
		if(checkDebug(debugFill))flprintf("filler: %p  %f\n", filler.operation, (float)filler.state);
	#endif

#endif

	if(checkDebug(debugFill)) flprintf("creating resmat %dX%dX%d mask %d tileD %s\n", nRows, nCols,nCols, gpuMask, b_util::tileDir(tileD));

	CuMatrix<T> mat(nRows,nCols,nCols, true,true,gpuMask,tileD);

	if(checkDebug(debugFill)) prlocf("created resmat\n");

	if(checkDebug(debugFill)) {
#ifndef __CUDA_ARCH__
		outln("created " << mat.toShortString());
		outln(mat.elements << " href " << mat.getMgr().refCount(mat.elements));
		outln(mat.currBuffer() << " dref " << mat.getMgr().refCount(mat.currBuffer()));
#endif
	}
	if(mat.tiler.tileSize >= mat.size) {
		util<T>::fillNDev(mat.tiler.buff(), t,(long)nRows*nCols);
		mat.invalidateHost();
		return mat;
	}

	mat.colMajor=colMajor;
	if(checkDebug(debugFill))flprintf(" CuMatrix<T>::fill mat.elements %p  tiler.currBuffer() %p\n", mat.elements, mat.tiler.currBuffer());
	fillFn(filler,mat);

	cherr(cudaDeviceSynchronize());
	if (checkDebug(debugFill)) {
		cherr(cudaDeviceSynchronize());
		printColoArrayInterval(mat.elements, mat.m * mat.p, 10, 40);

	}
	if(checkDebug(debugFill)) {
#ifndef __CUDA_ARCH__
		outln("after allocTiles mat " <<  mat.toShortString());
		outln(mat.elements << " href " << mat.getMgr().refCount(mat.elements));
		outln(mat.currBuffer() << " dref " << mat.getMgr().refCount(mat.currBuffer()));
#endif
	}

	return mat;
}


template <typename T> static void setNI2( T* trg, T val, long n) {
	flprintf("trg %p val %g n %d\n", trg, val, n);
	switch(sizeof(T)) {
		case 4: {
			T lval = val;
			unsigned char* puval = (unsigned char*)& lval;
			char cval1 = *puval++;
			char cval2 = *puval++;
			char cval3 = *puval++;
			char cval4 = *puval;
			puval = (unsigned char*) trg;
			checkCudaErrors(cudaMemset2D(puval++, 4, cval1, 1, n ));
			checkCudaErrors(cudaMemset2D(puval++, 4, cval2, 1, n ));
			checkCudaErrors(cudaMemset2D(puval++, 4, cval3, 1, n ));
			checkCudaErrors(cudaMemset2D(puval,   4, cval4, 1, n ));
			break;
		}
		case 8: {
			cudaStream_t streams[8];
			T lval = val;
			unsigned char* putrg = (unsigned char*)trg;
			unsigned char* puval = (unsigned char*)&lval;
			for(int i = 0; i < 8; i++) {
				cherr(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));
				checkCudaErrors(cudaMemset2DAsync(putrg + i, 	8,   puval[i],  1,  n, streams[i]));
			}

			for(int i = 0; i < 8; i++) {
				cherr(cudaStreamDestroy(streams[i]));
			}
		}
	}
}

template <typename T> __host__ CuMatrix<T> CuMatrix<T>::sfill(int nRows, int nCols, T t, bool colMajor, cudaStream_t stream) {
	if(checkDebug(debugFill)) prlocf("creating resmat\n");
	CuMatrix<T> mat(nRows,nCols,nCols, true,true,Tiler<T>::gpuMaskFromCurrGpu());
	if(checkDebug(debugFill)) prlocf("created resmat\n");
	if(checkDebug(debugFill)) {
#ifndef __CUDA_ARCH__
		outln(mat.elements << " href " << mat.getMgr().refCount(mat.elements));
		outln(mat.currBuffer() << " dref " << mat.getMgr().refCount(mat.currBuffer()));
		outln("res " << mat.toShortString());
#endif
	}
	if(!mat.tiler.hasDmemQ()) {
		mat.tiler.allocTiles(mat._tileM, mat._tileN, mat._tileP);
		mat.getMgr().addTiles(&(mat.tiler));
	} else {
		if(checkDebug(debugFill)) flprintf("already had buffer %p\n", mat.tiler.currBuffer());
	}

	assert(mat.tiler.tileSize == mat.tiler.m_size);

	setNI2(mat.tiler.buff(), t,nRows*nCols);
	mat.invalidateHost();

	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::zeros(int nRows, int nCols, int gpuMask, TileDirection tileD, bool colMajor) {

	if(checkDebug(debugFill))flprintf("zeros %uX%u\n", nRows,nCols);
	return fill(nRows,nCols,(T)0, gpuMask, tileD, colMajor);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::zeros(intPair dims, bool colMajor) {
	return fill(dims.first, dims.second,(T)0);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::ones(int nRows, int nCols, int gpuMask, TileDirection tileD, bool colMajor) {
	if(checkDebug(debugFill)) flprintf("\n\nones(%u, %u)\n", nRows, nCols);
	return fill(nRows,nCols,(T)1, gpuMask, tileD, colMajor);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::ones(intPair dims, bool colMajor) {
	return fill(dims.first, dims.second,(T)1);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::evens(int nRows, int nCols, T phase, int gpuMask, TileDirection tileD, bool colMajor) {
	if(checkDebug(debugFill)) flprintf("\n\nevens(%u, %u, %f)\n", nRows, nCols, (float) phase);
	evensFiller<T> filler = Functory<T,evensFiller>::pinch();
	filler.phase() = phase;
	CuMatrix<T> mat(nRows,nCols,nCols, true,true,gpuMask);
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat);
#else
	fillFn<1>(filler, mat);
#endif
	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::odds(int nRows, int nCols, T phase, int gpuMask, TileDirection tileD, bool colMajor) {
	if(checkDebug(debugFill)) flprintf("\nodds(%u, %u, %f)\n", nRows, nCols, (float) phase);
	oddsFiller<T> filler = Functory<T,oddsFiller>::pinch();
	filler.phase() = phase;
	CuMatrix<T> mat(nRows,nCols,nCols, true,true,gpuMask);
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat);
#else
	fillFn<1>(filler, mat);
#endif
	return mat;
}


template <typename T>__host__ CUDART_DEVICE  CuMatrix<T> CuMatrix<T>::sin(int m, int n, T amplitude, T period, T phase, bool colMajor) {

	flprintf("%uX%u amp %.2f per %.2f phaz %.2f\n", m, n, amplitude, period, phase);
	CuMatrix<T> mat = CuMatrix<T>::zeros(m,n).syncBuffers();

	int len =  m*n;
	T *arry= mat.elements;
	for(int i =0; i < len; i++){
		int colIdx = i % n * m + i / n;
		arry[i] = amplitude * ::sinf(colIdx+phase);
	}
	mat.invalidateDevice();
	//CuMatrix<T> mat(m,n,n, Tiler<T>::gpuMaskFromCurrGpu(), true,true);
	//mat.colMajor= colMajor;
	//fillFn(filler, mat);
	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::cos(int m, int n, T amplitude, T period, T phase, bool colMajor) {
	cosFiller<T> filler = Functory<T, cosFiller>::pinch(amplitude,period,phase);
	CuMatrix<T> mat(m,n,n, true,true, Tiler<T>::gpuMaskFromCurrGpu());
	mat.colMajor= colMajor;
	fillFn(filler, mat);
	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::sin(intPair dims, T amplitude, T period, T phase, bool colMajor) {
	return sin(dims.first, dims.second,amplitude,period,phase,colMajor);
}
template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::cos(intPair dims, T amplitude, T period, T phase, bool colMajor) {
	return cos(dims.first, dims.second,amplitude,period,phase,colMajor);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::diagonal(int dim, T val, bool colMajor) {
	if(dim > MaxDim) {
		setLastError(badDimensionsEx);
	}
	assert((dim <= MaxDim));
	if(checkDebug(debugFill)) flprintf("\n\ndiagonal(%u) = %f\n", dim, (float)val);
	diagonalFiller<T> filler = Functory<T, diagonalFiller>::pinch(val,dim);

	CuMatrix<T> mat(dim,dim,dim, true,true, Tiler<T>::gpuMaskFromCurrGpu());
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat);
#else
	fillFn<2>(filler, mat);
#endif
	return mat;
}

template <typename T> CuMatrix<T> CuMatrix<T>::diagonal(int dim, const T* val, bool colMajor) {
	return diagonal(dim, *val,colMajor);
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::identity(int dim, bool colMajor) {
	return diagonal(dim, static_cast<T>( 1), colMajor);
}

template <typename T> CuMatrix<T> CuMatrix<T>::randn(int rows, int cols, T epsilon, bool colMajor) {
	if(colMajor) {
		dthrow(notImplemented());
	}

	CuMatrix<T> ret(rows,cols,cols, true,true, Tiler<T>::gpuMaskFromCurrGpu());

	if(ret.tiler.tileSize < ret.tiler.m_size) {
		dthrow(notImplemented());
	}
	DMatrix<T> d_ret;
	ret.tile0(d_ret,false);
	randn(d_ret, epsilon);
	ret.lastMod = mod_device;
	return ret;
}

template <typename T> CuMatrix<T> CuMatrix<T>::randmod(int rows, int cols, int mod, T epsilon, bool colMajor) {
	if(colMajor) {
		dthrow(notImplemented());
	}

	CuMatrix<T> ret(rows,cols,cols, true,true, Tiler<T>::gpuMaskFromCurrGpu());

	if(ret.tiler.tileSize < ret.tiler.m_size) {
		dthrow(notImplemented());
	}
	DMatrix<T> d_ret;
	ret.tile0(d_ret,false);
	randmod(d_ret, mod, epsilon);
	ret.lastMod = mod_device;
	return ret;
}

template <typename T> CuMatrix<T> CuMatrix<T>::randn( const intPair& dims, float epsilon, bool colMajor) {
	return (randn(dims.first, dims.second, epsilon, colMajor));
}
template <typename T> CuMatrix<T> CuMatrix<T>::randn( const intPair& dims, bool colMajor) {
	return (randn(dims.first, dims.second, colMajor));
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::span(int m, int n, T start, T end, int gpuMask, TileDirection tileD, bool colMajor, cudaStream_t stream) {
	spanFiller<T> filler= Functory<T, spanFiller>::pinch(start,end,m*n);
	CuMatrix<T> mat(m,n,n, true,true,gpuMask, tileD );
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat, stream);
#else
	fillFn<3>(filler, mat, stream);
#endif
	return mat;
}

template <typename T> __host__ CUDART_DEVICE
CuMatrix<T> CuMatrix<T>::spanMod(int m, int n, T start, T end, int steps, int gpuMask, TileDirection tileD, bool colMajor, cudaStream_t stream) {
	spanModFiller<T> filler= Functory<T, spanModFiller>::pinch(start,end,steps);
	CuMatrix<T> mat(m,n,n, true,true,gpuMask, tileD );
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat, stream);
#else
	fillFn<3>(filler, mat, stream);
#endif
	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::sequence(int m, int n, T start, T step, int gpuMask, TileDirection tileD, bool colMajor, cudaStream_t stream) {
	sequenceFiller<T> filler=  Functory<T, sequenceFiller>::pinch(start, step);
	CuMatrix<T> mat(m,n,n, true,true,gpuMask, tileD );
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat, stream);
#else
	fillFn<2>(filler, mat, stream);
#endif
	return mat;
}
/*

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::sequenceScale(T start, T scale, int m, int n, bool colMajor) {
	sequenceScaleFiller<T> filler;
	filler.phase() = start;
	filler.scale() = scale;
	CuMatrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
	fillFn<2>(&filler, mat);
	return mat;
}
*/

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::seqMod(int m, int n, T start, T mod, int gpuMask, TileDirection tileD, bool colMajor, cudaStream_t stream ) {
	seqModFiller<T> filler=  Functory<T, seqModFiller>::pinch(start,mod);
	CuMatrix<T> mat(m,n,n, true,true,gpuMask, tileD );
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat, stream);
#else
	fillFn<2>(filler, mat, stream);
#endif
	return mat;
}


template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::increasingColumns(int rows, int cols, T start, int gpuMask, TileDirection tileD, bool colMajor ) {
	increasingColumnsFiller<T> filler =  Functory<T, increasingColumnsFiller>::pinch(start,cols);
	CuMatrix<T> mat(rows,cols,cols, true,true,gpuMask, tileD );
#ifndef __CUDA_ARCH__
	outln("mat " << mat.toShortString());
#endif
	mat.colMajor= colMajor;
#ifdef CuMatrix_Enable_KTS
	fillFn(filler, mat);
#else
	fillFn<2>(filler, mat);
#endif

	return mat;
}

template <typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::increasingRows(int rows, int cols, T start, int gpuMask, TileDirection tileD, bool colMajor ) {
	increasingRowsFiller<T> filler = 	Functory<T, increasingRowsFiller>::pinch(start,cols);
	CuMatrix<T> mat(rows,cols,cols, true,true, gpuMask, tileD);
	mat.colMajor= colMajor;
	fillFn(filler, mat);
	return mat;
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::extrude(int depth) const {
	if (m != 1) {
#ifndef __CUDA_ARCH__
		dthrow (notRowVector());
#else
		setLastError(notRowVectorEx);
#endif
	}
	CuMatrix<T> ret(*this);
	for (int r = 0; r < depth; r++) {
		ret = ret /= *this;
	}
	return ret;
}

#include "CuMatrixInster.cu"

