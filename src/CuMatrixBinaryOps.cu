/*
 * CuMatrixBinaryOps.cu
 *
 *      Author: reid
 */
#include "CuMatrix.h"
#include "Kernels.h"
#include "Maths.h"
#include <execinfo.h>
#include <typeinfo>

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __global__
void binaryOpKernel1dNeqP(T* trg, const T* src1, const T* src2, BinaryOp<T> op,
		ulong len)
#else
template<typename T, int StateDim> __global__
void binaryOpKernel1dNeqP(T* trg, const T* src1, const T* src2, BinaryOpF<T,StateDim> op,
		ulong len)
#endif
{
	ulong i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		trg[i] = op(src1[i], src2[i]);
	}
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __global__
void binaryOpKernel1dNeqPRow(T* trg, const T* src1, const T* src2, long n, BinaryOp<T> op,
		ulong len)
#else
template<typename T, int StateDim> __global__
void binaryOpKernel1dNeqPRow(T* trg, const T* src1, const T* src2, long n, BinaryOpF<T,StateDim> op,
		ulong len)
#endif
{
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		trg[i] = op(src1[i], src2[mod<long>(i,n)]);
	}
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __global__
void binaryOpKernel2(T* trg, const T* src1, const T* src2, BinaryOp<T> op,
		ulong len, int p, uint * misses)
#else
template<typename T, int StateDim> __global__
void binaryOpKernel2(T* trg, const T* src1, const T* src2, BinaryOpF<T,StateDim> op,
		ulong len, int p, uint * misses)
#endif
{
	// 2 A + 2 M
	int col = threadIdx.x  + blockIdx.x *blockDim.x;
	int row = threadIdx.y  + blockIdx.y *blockDim.x;

	ulong i =  col + row * p;
	ulong j = i;
	for(int k = 0; k < blockDim.x; k += blockDim.y) {
		j = i+ k * p;
/*
		if(threadIdx.x == 0 && k > 0) {
			printf("row %u col %u, j %lu threadIdx.y %d, k %d\n",row, col, j, threadIdx.y,k);
		}
*/
		if (j < len) {
			trg[j] = op(src1[j], src2[j]);
		} else {
			//printf("over row %u col %u\n",threadIdx.y  + blockIdx.y *blockDim.y, threadIdx.y  + blockIdx.y *blockDim.y );
			if(misses)atomicInc(misses, UINT_MAX);
		}
	}
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __global__
void binaryOpDmKernel(DMatrix<T> trg, const DMatrix<T> src1, const DMatrix<T> src2, BinaryOp<T> op )
#else
template<typename T, int StateDim> __global__
void binaryOpDmKernel(DMatrix<T> trg, const DMatrix<T> src1, const DMatrix<T> src2, BinaryOpF<T,StateDim> op )
#endif
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.x + threadIdx.y;
	uint src1Off = y * src1.p + x;
	uint src2Off = y * src2.p + x;
	uint trgOff = y * trg.p + x;
	for(int i = 0; i < blockDim.x; i+=blockDim.y) {
		if(x < src1.n && y + i < src1.m && x < src2.n && y + i < src2.m) {
			trg.elements[trgOff + i * trg.p] = op(src1.elements[src1Off + i * src1.p],src2.elements[src2Off + i * src2.p]);
		}
	}
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __global__
void binaryOpDmKernel2(DMatrix<T> trg, const DMatrix<T> src1, const DMatrix<T> src2, BinaryOp<T> op )
#else
template<typename T, int StateDim> __global__
void binaryOpDmKernel2(DMatrix<T> trg, const DMatrix<T> src1, const DMatrix<T> src2, BinaryOpF<T,StateDim> op )
#endif
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.x + threadIdx.y;
	int n = MIN(src1.n,src2.n);
	int m = MIN(src1.m,src2.m);

	uint src1Off = y * src1.p + x;
	uint src2Off = y * src2.p + x;
	uint trgOff = y * trg.p + x;
	for(int i = 0; i < blockDim.x; i+=blockDim.y) {
		if(x < n && y + i < m ) {
			trg.elements[trgOff + i * trg.p] = op(src1.elements[src1Off + i * src1.p],src2.elements[src2Off + i * src2.p]);
		}
	}
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __host__ CUDART_DEVICE
void binaryOpL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOp<T> op, cudaStream_t stream)
#else
template<typename T, int StateDim> __host__ CUDART_DEVICE
void binaryOpL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOpF<T,StateDim> op, cudaStream_t stream)
#endif
{
	int threads = 256;
	uint len = src1.m * src2.n;
	dim3 dBlocks, dThreads;
	b_util::vectorExecContext(threads, len, dBlocks, dThreads);

/*
#ifndef __CUDA_ARCH__
	b_util::dumpStack();
#endif
*/
	if(checkDebug(debugExec)){
		prlocf("binaryOpL grid " );
		b_util::prd3(dBlocks);
		prlocf("binaryOpL of block ");
		b_util::prd3(dThreads);
	}

	if(src1.m != src2.m && src2.m == 1) {
		binaryOpKernel1dNeqPRow<<<dBlocks,dThreads,0, stream>>>(trg.elements, src1.elements, src2.elements, src2.n, op, len);
	} else {
		binaryOpKernel1dNeqP<<<dBlocks,dThreads,0, stream>>>(trg.elements, src1.elements, src2.elements, op, len);
	}
#ifndef __CUDA_ARCH__
	if(!stream) {
		checkCudaError(cudaDeviceSynchronize());
	}
#endif
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __host__ CUDART_DEVICE
void binaryOpL2(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOp<T> op, int h2w, uint* misses, cudaStream_t stream)
#else
template<typename T, int StateDim> __host__ CUDART_DEVICE
void binaryOpL2(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOpF<T,StateDim> op, int h2w, uint* misses, cudaStream_t stream)
#endif
{
	uint len = src1.m * src2.n;
	dim3 grid, block(32,32);
	grid.x = DIV_UP(src1.n,block.x);
	grid.y = DIV_UP(src1.m,block.x);
	block.y = block.x / h2w;
	//if(checkDebug(debugExec)){
		printf("binaryOpL2 p %d\n", src1.p);
		printf("binaryOpL2 grid " );
		b_util::prd3(grid);
		printf("binaryOpL2 of block ");
		b_util::prd3(block);
	//}
	binaryOpKernel2<<<grid,block,0, stream>>>(trg.elements, src1.elements, src2.elements, op, len, src1.p, misses);
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void binaryOpL2<float, multBinaryOp>(DMatrix<float>&, DMatrix<float> const&, DMatrix<float> const&, multBinaryOp<float>, int, unsigned int*, CUstream_st*);
template __host__ CUDART_DEVICE void binaryOpL2<double, multBinaryOp>(DMatrix<double>&, DMatrix<double> const&, DMatrix<double> const&, multBinaryOp<double>, int, unsigned int*, CUstream_st*);
#else
template __host__ CUDART_DEVICE void binaryOpL2<float, 1>(DMatrix<float>&, DMatrix<float> const&, DMatrix<float> const&, BinaryOpF<float,1>, int, unsigned int*, CUstream_st*);
template __host__ CUDART_DEVICE void binaryOpL2<double, 1>(DMatrix<double>&, DMatrix<double> const&, DMatrix<double> const&, BinaryOpF<double,1>, int, unsigned int*, CUstream_st*);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __host__ CUDART_DEVICE
void binaryOpDmL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOp<T> op, int w2h, cudaStream_t stream)
#else
template<typename T, int StateDim> __host__ CUDART_DEVICE
void binaryOpDmL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOpF<T,StateDim> op, int w2h, cudaStream_t stream)
#endif
{
	assert( trg.m >= MIN( src1.m,src2.m) && trg.n >= MIN(src1.n,src2.n) );
	int blockW = UNOP_BLOCK_SIZE;
	dim3 block(blockW,blockW/w2h);
    dim3 grid(DIV_UP(MIN(src1.n,src2.n),blockW), DIV_UP(MIN(src1.m,src2.m), blockW));
    if(checkDebug(debugExec)) {
		printf("grid " );
		b_util::prd3(grid);
		printf(" of block ");
		b_util::prd3(block);
    }
	binaryOpDmKernel<<<grid,block,0,stream>>>(trg, src1, src2, op);
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE
void CuMatrix<T>::binaryOp(CuMatrix<T>& res, const CuMatrix<T>& o, BinaryOp<T> op, cudaStream_t stream   ) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE
void CuMatrix<T>::binaryOp(CuMatrix<T>& res, const CuMatrix<T>& o, BinaryOpF<T, StateDim> op , cudaStream_t stream  ) const
#endif
{
	DMatrix<T> d_A, d_B, d_res;

	assert( compatibleQ(o));
	if(tiler.tileSize == tiler.m_size) {

		tile0(d_A,lastMod==mod_host,stream);
		if (checkDebug(debugTiler)) prlocf("this tile0\n");
		o.tile0( d_B, tiler != o.tiler && o.lastMod==mod_host,stream);
		if (checkDebug(debugTiler)) prlocf("o.tile0\n");
		res.tile0( d_res, false,stream);
		if(n == p) {
			binaryOpL( d_res, d_A, d_B,op,stream);
			//if(0==p)binaryOpL2( d_res, d_A, d_B,op,1);
		} else {
			binaryOpDmL( d_res, d_A, d_B,op,DefaultWidth2Height,stream);
		}
		res.invalidateHost();
	} else {
		uint tileM, tileN;
		if (checkDebug(debugTiler)) {
			flprintf("this %p, &o %p, &res %p\n",this, &o, &res);
			o.printShortString("matrix o");
			printShortString("this");
			res.printShortString("ret");
		}
		tiler.tileDims(tileM, tileN, tdRows);
		int tileCount = DIV_UP(m,tileM);
		uint roff,coff;
		int gpuCount = tiler.countGpus();
		int orgDevice = ExecCaps::currDev();
		int lastGpu =gpuCount > 1 ? 0 : -1;
		cudaStream_t* streams = null;
		if(gpuCount > 1) {
			assert(!stream);
			cudaStream_t* streams = (cudaStream_t*)malloc(gpuCount * sizeof(cudaStream_t));
			for(int i =0 ; i < gpuCount; i++) {
				lastGpu = tiler.nextGpu(lastGpu);
				ExecCaps_setDevice(lastGpu);
				cherr(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));
			}
		}

		lastGpu = tiler.nextGpu(0);
		for(int i =0; i < tileCount; i++ ) {
			ExecCaps_setDevice(lastGpu); // do it here so this (i)tile, (j) b-tile and (i,j) res-tile all on same dev
			if (checkDebug(debugTiler))prlocf("binaryOp tileLike d_A");
			tiler.tileLike(d_A, roff,coff, tileM, tileN, i, tdRows, true,lastGpu,gpuCount > 1 ? streams[i] : stream);
			if (checkDebug(debugTiler))prlocf("binaryOp tileLike d_B");
			o.tiler.tileLike( d_B, roff,coff, tileM, tileN, i, tdRows, tiler != o.tiler, lastGpu,gpuCount > 1 ? streams[i] : stream);
			if (checkDebug(debugTiler))prlocf("binaryOp tileLike d_Res");
			lastGpu = res.tiler.tileLike( d_res, roff,coff, tileM, tileN, i, tdRows, false,lastGpu,gpuCount > 1 ? streams[i] : stream);
			if(checkDebug(debugExec)) {
				prlocf("void CuMatrix<T>::binaryOp(CuMatrix<T> &,CuMatrix<T> &, BinaryOp -> " );
			}
		#ifndef __CUDA_ARCH__
			if(checkDebug(debugExec)){
				flprintf(" %s  " , b_util::unmangl(typeid(op).name()).c_str());
			}
		#endif
			if(n == p) {
				binaryOpL( d_res, d_A, d_B,op,stream);
				//if(0==p)binaryOpL2( d_res, d_A, d_B,op,1);
			} else {
				binaryOpDmL( d_res, d_A, d_B,op,DefaultWidth2Height,stream);
			}
			// flush tile to hostmem
			res.tiler.syncTile(d_res, roff, coff, stream);
		}
		if(gpuCount > 1) {
			for(int i =0 ; i < gpuCount; i++) {
				cherr(cudaStreamDestroy(streams[i]));
			}
			free(streams);
		}
		if(orgDevice != ExecCaps::currDev()) {
			ExecCaps_setDevice(orgDevice);
		}
		res.lastMod = mod_synced;  // have to
	}
}

#ifdef  CuMatrix_Enable_KTS
template<typename T> template<template <typename> class BinaryOp> __host__ CUDART_DEVICE
CuMatrix<T> CuMatrix<T>::binaryOp(const CuMatrix<T>& o, BinaryOp<T> op, cudaStream_t stream  ) const
#else
template<typename T> template<int StateDim> __host__ CUDART_DEVICE
CuMatrix<T> CuMatrix<T>::binaryOp(const CuMatrix<T>& o, BinaryOpF<T,StateDim> op , cudaStream_t stream  ) const
#endif
{
	if(!equalDims(o)  && !( o.vectorQ() && o.n == n ) ) {
		printShortString(" can't be bin-opd with ");
		o.printShortString();
		setLastError(matricesOfIncompatibleShapeEx);
	}
	if(checkDebug(debugBinOp)) prlocf("binaryOp(const CuMatrix<T>&,...) creating res\n" );

	CuMatrix<T> res(m, n, true, true);
	binaryOp(res, o, op,stream);
	return res;
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::binaryOp<almostEqualsBinaryOp>(CuMatrix<float> const&, almostEqualsBinaryOp<float>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::binaryOp<almostEqualsBinaryOp>(CuMatrix<double> const&, almostEqualsBinaryOp<double>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned long> CuMatrix<unsigned long>::binaryOp<almostEqualsBinaryOp>(CuMatrix<unsigned long> const&, almostEqualsBinaryOp<unsigned long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::binaryOp<plusBinaryOp>(CuMatrix<float> const&, plusBinaryOp<float>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned long> CuMatrix<unsigned long>::binaryOp<plusBinaryOp>(CuMatrix<unsigned long> const&, plusBinaryOp<unsigned long>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::binaryOp<plusBinaryOp>(CuMatrix<double> const&, plusBinaryOp<double>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::binaryOp<equalsBinaryOp>(CuMatrix<float> const&, equalsBinaryOp<float>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::binaryOp<equalsBinaryOp>(CuMatrix<double> const&, equalsBinaryOp<double>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<unsigned long> CuMatrix<unsigned long>::binaryOp<equalsBinaryOp>(CuMatrix<unsigned long> const&, equalsBinaryOp<unsigned long>, CUstream_st*) const;
#else
template __host__ CUDART_DEVICE CuMatrix<float> CuMatrix<float>::binaryOp<1>(CuMatrix<float> const&, BinaryOpF<float, 1>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<double> CuMatrix<double>::binaryOp<1>(CuMatrix<double> const&, BinaryOpF<double, 1>, CUstream_st*) const;
template __host__ CUDART_DEVICE CuMatrix<ulong> CuMatrix<ulong>::binaryOp<1>(CuMatrix<ulong> const&, BinaryOpF<ulong, 1>, CUstream_st*) const;
#endif

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator+( const CuMatrix<T> o) const {
	return binaryOp(o, Functory<T,plusBinaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator-( const CuMatrix<T> o) const {
	return binaryOp(o, Functory<T,minusBinaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator&&(CuMatrix<T> o) const {
	return binaryOp(o, Functory<T,andBinaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::operator||(CuMatrix<T> o) const {
	return binaryOp(o, Functory<T,orBinaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::hadamardProduct(const CuMatrix<T> o) const {
	return binaryOp(o, Functory<T,multBinaryOp>::pinch());
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::hadamardQuotient(const CuMatrix<T> o) const {
	return binaryOp(o,  Functory<T,quotientBinaryOp>::pinch());
}

#include "CuMatrixInster.cu"
