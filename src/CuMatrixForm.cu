#include "CuMatrix.h"
#include "util.h"
#include "functors.h"
#include "debug.h"
#include "caps.h"
#include "MatrixExceptions.h"
#define TX_BLOCK_SIZE 	32
int blockH = TX_BLOCK_SIZE/4;
#define CAT_BLOCK_SIZE 	32


template<typename T, typename FillOp> __global__ void fillOpKernel(
		FillOp op,
		T* trg,
		uint height,
		uint width,
		uint pitch,
		bool colMajor)
{
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    uint indexOut = colMajor ? xIndex * pitch + yIndex : yIndex * pitch + xIndex;
    if(xIndex < width && yIndex < height)
    	trg[indexOut] = op(indexOut);
}

// non-square block version, to amortize index calcs
template<typename T, typename FillOp> __global__ void fillOpNsbKernel(
		FillOp op,
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

template<typename T> template<typename FillOp> void CuMatrix<T>::fillFn(
		FillOp op, CuMatrix<T>& ret) {

	if(ret.m ==0 || ret.n ==0 ){
		dthrow(badDimensions());
	}
	if(debugFill)outln("fillFn m " << ret.m << " n " << ret.n << ", colMajor " << tOrF(ret.colMajor));

	uint blockW = MIN(b_util::nextPowerOf2(ret.n), WARP_SIZE);
	if(debugFill)outln("blockW "<< blockW);
	uint blockH = MIN(ret.m, maxH<T>(caps,blockW));
	if(debugFill)outln("blockH "<< blockH);
	dim3 grid(b_util::enough(blockW, ret.n), b_util::enough(blockH, ret.m));
	dim3 block(blockW,blockH);
	if(debugFill)outln("fillFn on mat " << ret.m << "x" << ret.n << " grid of " << b_util::pd3(grid) << " of block " << b_util::pd3(block));
	fillOpKernel<<<grid,block>>>(op, ret.d_elements, ret.m,ret.n,ret.p, ret.colMajor);
}

template<typename T> template<typename FillOp> void CuMatrix<T>::fillFnNsb(
		FillOp op, CuMatrix<T>& ret, int w2h) {

	if(ret.m ==0 || ret.n ==0 ){
		dthrow(badDimensions());
	}
	if(debugFill)outln("fillFn m " << ret.m << " n " << ret.n << ", colMajor " << tOrF(ret.colMajor));

	uint blockW = MIN(b_util::nextPowerOf2(ret.n), WARP_SIZE);
	if(debugFill)outln("blockW "<< blockW);
	//uint blockH = MIN(ret.m, maxH<T>(caps,blockW));
	uint blockH = blockW/w2h;
	if(debugFill)outln("blockH "<< blockH);
	dim3 grid(b_util::enough(blockW, ret.n), b_util::enough(blockW, ret.m));
	dim3 block(blockW,blockH);
	if(debugFill)outln("fillFn on mat " << ret.m << "x" << ret.n << " grid of " << b_util::pd3(grid) << " of block " << b_util::pd3(block));
	fillOpNsbKernel<<<grid,block>>>(op, ret.d_elements, ret.m,ret.n,ret.p, ret.colMajor);
}


//#ifdef CUTMPLT
template void CuMatrix<float>::fillFn<stepFiller<float> >(stepFiller<float>, CuMatrix<float>&);
template void CuMatrix<float>::fillFn<constFiller<float> >(constFiller<float>, CuMatrix<float>&);
template void CuMatrix<float>::fillFn<sinFiller<float> >(sinFiller<float>, CuMatrix<float>&);
template void CuMatrix<float>::fillFn<cosFiller<float> >(cosFiller<float>, CuMatrix<float>&);
template void CuMatrix<float>::fillFn<randFiller<float> >(randFiller<float>, CuMatrix<float>&);
template void CuMatrix<float>::fillFn<sequenceFiller<float> >(sequenceFiller<float>, CuMatrix<float>&);
template void CuMatrix<float>::fillFn<seqModFiller<float> >(seqModFiller<float>, CuMatrix<float>&);
template void CuMatrix<float>::fillFn<diagonalFiller<float> >(diagonalFiller<float>, CuMatrix<float>&);
template void CuMatrix<float>::fillFn<increasingColumnsFiller<float> >(increasingColumnsFiller<float>, CuMatrix<float>&);
template void CuMatrix<float>::fillFn<increasingRowsFiller<float> >(increasingRowsFiller<float>, CuMatrix<float>&);

template void CuMatrix<double>::fillFn<stepFiller<double> >(stepFiller<double>, CuMatrix<double>&);
template void CuMatrix<double>::fillFn<constFiller<double> >(constFiller<double>, CuMatrix<double>&);
template void CuMatrix<double>::fillFn<sinFiller<double> >(sinFiller<double>, CuMatrix<double>&);
template void CuMatrix<double>::fillFn<cosFiller<double> >(cosFiller<double>, CuMatrix<double>&);
template void CuMatrix<double>::fillFn<randFiller<double> >(randFiller<double>, CuMatrix<double>&);
template void CuMatrix<double>::fillFn<sequenceFiller<double> >(sequenceFiller<double>, CuMatrix<double>&);
template void CuMatrix<double>::fillFn<seqModFiller<double> >(seqModFiller<double>, CuMatrix<double>&);
template void CuMatrix<double>::fillFn<diagonalFiller<double> >(diagonalFiller<double>, CuMatrix<double>&);
template void CuMatrix<double>::fillFn<increasingColumnsFiller<double> >(increasingColumnsFiller<double>, CuMatrix<double>&);
template void CuMatrix<double>::fillFn<increasingRowsFiller<double> >(increasingRowsFiller<double>, CuMatrix<double>&);

//#endif
template void CuMatrix<float>::fillFnNsb<stepFiller<float> >(stepFiller<float>, CuMatrix<float>&,int);
template void CuMatrix<double>::fillFnNsb<stepFiller<double> >(stepFiller<double>, CuMatrix<double>&,int);

template <typename T> __global__ void
copyKernel(T* tElements, const T* sElements, uint tWidth, uint tHeight, uint tPitch, uint sWidth, uint sHeight, uint sPitch, uint xOff, uint yOff)
{
    ulong x = blockIdx.x * blockDim.x + threadIdx.x;
    ulong y = blockIdx.y * blockDim.y + threadIdx.y;
    ulong tx = x + xOff;
    ulong ty = y + yOff;
    ulong sIdx = y * sPitch + x;
    ulong tIdx = ty * tPitch + tx;
    if(x < sWidth && y < sHeight && tx < tWidth && ty < tHeight) {
    	tElements[tIdx] = sElements[sIdx];
    }
}

template <typename T> __global__ void
copyDmKernelUlong(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) {
    ulong x = blockIdx.x * blockDim.x + threadIdx.x;
    ulong y = blockIdx.y * blockDim.x + threadIdx.y;
    ulong tx = x + tcoff;
    ulong ty = y + troff;
    ulong sIdx = y * src.p + x;
    ulong tIdx = ty * trg.p + tx;
    for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
		if(x < src.n && y + i < src.m && tx < trg.n && ty + i < trg.m) {
			trg.elements[tIdx + i * trg.p] = src.elements[sIdx + i * src.p];
		}
    }
}

template <typename T> __global__ void
copyDmKernelUint(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.x + threadIdx.y;
	uint tx = x + tcoff;
	uint ty = y + troff;
	uint sIdx = y * src.p + x;
	uint tIdx = ty * trg.p + tx;
    for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
		if(x < src.n && y + i < src.m && tx < trg.n && ty + i < trg.m) {
			trg.elements[tIdx + i * trg.p] = src.elements[sIdx + i * src.p];
		}
    }
}

template <typename T> __global__ void
copyDmKernelUlongDvrg(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) {
    ulong x = blockIdx.x * blockDim.x + threadIdx.x;
    ulong y = blockIdx.y * blockDim.x + threadIdx.y;
    ulong tx = x + tcoff;
    ulong ty = y + troff;
    ulong sIdx = y * src.p + x;
    ulong tIdx = ty * trg.p + tx;
    if(x<src.n && tx < trg.n) {
    	uint sdiff = src.m - y;
    	uint tdiff = trg.m - y;
		for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
			if ( i < sdiff && i < tdiff) {
				trg.elements[tIdx + i * trg.p] = src.elements[sIdx + i * src.p];
			}
		}
    }
}

template <typename T> __global__ void
copyDmKernelUintDvrg(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.x + threadIdx.y;
    uint tx = x + tcoff;
    uint ty = y + troff;
    uint sIdx = y * src.p + x;
    uint tIdx = ty * trg.p + tx;
    if(x<src.n && tx < trg.n) {
    	uint sdiff = src.m - y;
    	uint tdiff = trg.m - y;
		for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
			if ( i < sdiff && i < tdiff) {
				trg.elements[tIdx + i * trg.p] = src.elements[sIdx + i * src.p];
			}
		}
    }
}

template <typename T> __global__ void
copyDmKernelIntDvrg(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.x + threadIdx.y;
	int tx = x + tcoff;
	int ty = y + troff;
	int sIdx = y * src.p + x;
	int tIdx = ty * trg.p + tx;
    if(x<src.n && tx < trg.n) {
    	int sdiff = src.m - y;
    	int tdiff = trg.m - y;
		for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
			if ( i < sdiff && i < tdiff) {
				trg.elements[tIdx + i * trg.p] = src.elements[sIdx + i * src.p];
			}
		}
    }
}

// indices.length === trg.m
template <typename T> __global__ void
copyDmRowShuffleKernel(DMatrix<T> trg, const DMatrix<T> src, uint* indices) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.x + threadIdx.y;
	uint tIdx = y * trg.p + x;
    if(x < trg.n && y < trg.m) {
    	int tdiff = trg.m - y;
		for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
			if ( i < tdiff) {
				trg.elements[tIdx + i * trg.p] = src.elements[ indices[y + i] * src.p + x];
			} else {
		    	if(blockIdx.x == 0 && threadIdx.x == 0) {
		    		printf("i (%d) !< tdiff (%d), y = %d\n", i, tdiff, y);
		    	}
			}
		}
    } else {
    	if(blockIdx.x == 0 && threadIdx.x == 0) {
    		printf("%d !< trg.n or %d !< trg.m\n", x, y);
    	}
    }
}

template <typename T> __global__ void
copyKernel(T* tElements, const T* sElements, uint amountInTs, uint offsetInTs, ulong lengthInTs)
{
    ulong id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < amountInTs && id + offsetInTs < lengthInTs) {
    	tElements[id + offsetInTs] = sElements[id];
    }
}



template <typename T> void CuMatrix<T>::rightConcatenateL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2) {
	dim3 block(CAT_BLOCK_SIZE,CAT_BLOCK_SIZE,1);
	dim3 gridAc(b_util::enough(CAT_BLOCK_SIZE,src1.n), b_util::enough(CAT_BLOCK_SIZE,src1.m),1);
	dim3 gridBc(b_util::enough(CAT_BLOCK_SIZE,src2.n), b_util::enough(CAT_BLOCK_SIZE,src2.m),1);
	if(debugExec)outln("gridAc " << b_util::pd3(gridAc).c_str() << " for " << util<T>::pdm(src1));
	if(debugExec)outln("gridBc " << b_util::pd3(gridBc).c_str() << " for " << util<T>::pdm(src2));
	if(syncHappy)b_util::syncGpu();
	//copyKernel<<<gridAc,block>>>(d_C, d_A, 0, 0);
	copyKernel<<<gridAc,block>>>(trg.elements, src1.elements, trg.n, trg.m, trg.p, src1.n, src1.m, src1.p, 0, 0);
	copyKernel<<<gridBc,block>>>(trg.elements, src2.elements, trg.n, trg.m, trg.p, src2.n, src2.m, src2.p, src1.n, 0);
	if(syncHappy)b_util::syncGpu();
}

template <typename T> void CuMatrix<T>::bottomConcatenateL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2)  {
	dim3 block(CAT_BLOCK_SIZE,CAT_BLOCK_SIZE,1);
	dim3 dmBlock(CAT_BLOCK_SIZE,CAT_BLOCK_SIZE/8,1);
	dim3 gridAc(b_util::enough(CAT_BLOCK_SIZE,src1.n), b_util::enough(CAT_BLOCK_SIZE,src1.m),1);
	dim3 gridBc(b_util::enough(CAT_BLOCK_SIZE,src2.n), b_util::enough(CAT_BLOCK_SIZE,src2.m),1);
	if(debugExec)outln("gridAc " << b_util::pd3(gridAc).c_str() << " for " << util<T>::pdm(src1));
	if(debugExec)outln("gridBc " << b_util::pd3(gridBc).c_str() << " for " << util<T>::pdm(src2));
	if(syncHappy)b_util::syncGpu();
	if(src1.n == src1.p) {
		copyKernel<<<gridAc,block>>>(trg.elements, src1.elements,trg.n, trg.m, trg.p, src1.n, src1.m, src1.p, 0, 0);
	} else {
		dthrow(notImplemented());
	}
	copyKernel<<<gridBc,block>>>(trg.elements, src2.elements, trg.n, trg.m, trg.p, src2.n, src2.m, src2.p, 0, src1.m);
	if(syncHappy)b_util::syncGpu();
}

template <typename T> __global__ void setKernel(T* elements, uint p, uint row, uint col, T val) {
	elements[row * p + col] = val;
}

template <typename T> void CuMatrix<T>::set(T* elements, uint m, uint n, uint p, uint row, uint col, T val) {
	if(row > m) dthrow(rowOutOfBounds());
	if(col > n) dthrow(columnOutOfBounds());
	setKernel<<<1,1>>>(elements, p, row,col,val);
}

template <typename T> __global__ void setKernel(T* elements,  uint m, uint n, uint p,  ulong l, T val) {
	if(n == p) {
		elements[l] = val;
	} else {
		// todo simplify this
		uint div = l /n;
		uint idx = div * p;
		idx += l - div * n;
		//printf("offset l %u -> %u\n", l, idx);
		elements[idx ] = val;
	}
}

template <typename T> void CuMatrix<T>::set(T* elements, uint m, uint n, uint p, ulong l, T val) {
	if(l > m * p) dthrow(outOfBounds());
	setKernel<<<1,1>>>(elements, m, n, p, l,val);
}

// tiles must be square
template <typename T> __global__ void transposeNaive(const T* sElements, T* tElements, uint width, uint height) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.x + threadIdx.y; // not blockDim.y, which we assume a factor of blockDim.x

    int index_in = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;
    // threads must iterate normal to the cache line,
    // which doesn't happen in this write to t
	for (int i=0; i<blockDim.x; i+=blockDim.y)
		if(xIndex < width && yIndex + i < height)
			tElements[index_out+i] = sElements[index_in+i*width];
}
template void __global__ transposeNaive<float>(const float*,float*,uint,uint);
template void __global__ transposeNaive<double>(const double*,double*,uint,uint);

template <typename T> __global__ void transposeSubTile(const T* sElements, T* tElements, uint width, uint height)
{
	T* tile = SharedMemory<T>();
    uint xIndex = threadIdx.x;
    uint yIndex = threadIdx.y;
	uint vmemIdx = yIndex * width + xIndex;
	uint txIdx = xIndex * height + yIndex;

	if(xIndex < width && yIndex < height) {
		tile[txIdx] = sElements[vmemIdx];
	}
	__syncthreads();
	if(xIndex < width && yIndex < height) {
		tElements[vmemIdx] = tile[vmemIdx];
	}
}

template <typename T> __global__ void transposeCoalesced(const T* sElements, T* tElements, uint width, uint height)
{
	T* tile = SharedMemory<T>();

    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.x + threadIdx.y;
    uint index_in = xIndex + yIndex * width;
    uint tileIdxOut = threadIdx.x * blockDim.x + threadIdx.y;
    uint tileIdxIn = threadIdx.y * blockDim.x + threadIdx.x;

    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < width && yIndex + i < height )
    		tile[tileIdxIn + i * blockDim.x] = sElements[index_in + i * width];

    __syncthreads();

    xIndex = blockIdx.y * blockDim.x + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y; // not blockDim.y
    int index_out = xIndex + yIndex * height;
    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < height && yIndex + i < width)
    		tElements[index_out + i * height] = tile[tileIdxOut + i];
}
template void __global__ transposeCoalesced<float>(const float*,float*,uint,uint);
template void __global__ transposeCoalesced<double>(const double*,double*,uint,uint);

template <typename T>
__global__ void transposeNoBankConflicts(const T* sElements, T* tElements, uint width, uint height)
{
	T* tile = SharedMemory<T>();

    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.x + threadIdx.y;
    uint index_in = xIndex + yIndex*width;
    uint tileIdxIn = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    uint tileIdxOut = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < width && yIndex + i < height )
    		tile[tileIdxIn + i* ( blockDim.x +1)] = sElements[index_in + i * width];
    __syncthreads();
    xIndex = blockIdx.y * blockDim.x + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;
    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < height && yIndex + i < width)
    		tElements[index_out + i * height] = tile[tileIdxOut + i];

}

template <typename T> __global__ void transposeDiagonalKernel(const T* sElements, T* tElements, uint width, uint height)
{
	T* tile = SharedMemory<T>();

	int blockIdx_x, blockIdx_y;

	if (width == height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	} else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }
    uint xIndex = blockIdx_x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx_y * blockDim.x + threadIdx.y;
    uint index_in = xIndex + (yIndex)*width;
    uint tileIdx = threadIdx.y * (blockDim.x + 1)+ threadIdx.x;
    uint tileIdxOut = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < width && yIndex + i < height )
    		tile[tileIdx + i * (blockDim.x + 1)] = sElements[index_in+i*width];

    __syncthreads();
    xIndex = blockIdx_y * blockDim.x + threadIdx.x;
    yIndex = blockIdx_x * blockDim.x + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;
    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < height && yIndex + i < width)
    		tElements[index_out + i * height] = tile[tileIdxOut + i];

}

template <typename T> void CuMatrix<T>::transposeKernelPtrL(DMatrix<T>& t,void (*kernel)(const T*, T*, uint, uint), const DMatrix<T>& s )  {
	ulong len = s.m * s.n;
	dassert( ( len == t.m * t.n) );
	int blockW = TX_BLOCK_SIZE;
	dim3 block(blockW,blockH);
    dim3 grid(b_util::enough(blockW,s.n), b_util::enough(blockW, s.m));
	if(syncHappy)b_util::syncGpu("before knl " );

    void (*txNmbcPtr)(const T*,T*,uint,uint);
    txNmbcPtr=&transposeNoBankConflicts;
    void (*txDiagPtr)(const T*,T*,uint,uint);
    txDiagPtr=&transposeDiagonalKernel;
	int tileWidth = blockW;
	if(kernel == txNmbcPtr || kernel == txDiagPtr) {
		tileWidth++;
	}
	int smem = TX_BLOCK_SIZE * (tileWidth)* sizeof(T);
	kernel<<<grid, block, smem>>>(s.elements, t.elements, s.n, s.m);
	//outln("tx with grid " << b_util::pd3(grid).c_str() << " of block " << b_util::pd3(block).c_str() << " smem " << smem);
	if(syncHappy)b_util::syncGpu("after");
}

template <typename T> void CuMatrix<T>::transposeL( DMatrix<T>& t, const DMatrix<T>& s)  {
    void (*txNmbcPtr)(const T*,T*,uint,uint);
    txNmbcPtr=&transposeDiagonalKernel;
    //txNmbcPtr=&transposeNoBankConflicts;
	return transposeKernelPtrL(t, txNmbcPtr,s);
}

template <typename T> __global__ void binaryCategoryKernel(const T* sElements, T* tElements, uint width, uint height, bool oneBased)
{
	T* tile = SharedMemory<T>();

    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    uint idxOut = xIndex + yIndex * width;
    if(blockDim.x == threadIdx.x == 0) {
    	tile[threadIdx.y] = sElements[yIndex];
    }
    __syncthreads();
    if(xIndex < width && yIndex < height) {
    	tElements[idxOut] = tile[threadIdx.y] == xIndex + (oneBased ? 1 : 0);
    }
}

template <typename T> void CuMatrix<T>::binaryCategoryKernelL(DMatrix<T>& t, const DMatrix<T>& s, bool oneBased)  {
	uint blockW = b_util::nextPowerOf2(t.n);
	uint blockH = maxH<T>(caps,blockW);
	dim3 block(blockW, blockH);
	if(debugExec)outln("binCat blockW " << blockW << ", blockH " << blockH << " block " << b_util::pd3(block));
    dim3 grid(b_util::enough(block.x, t.n), b_util::enough(block.y, t.m));
	if(syncHappy)b_util::syncGpu("before knl " );
	int smem = block.x * block.y * sizeof(T);
	binaryCategoryKernel<<<grid, block, smem>>>(s.elements, t.elements, t.n, t.m, oneBased);
	if(debugExec)outln("binCatKernel with grid " << b_util::pd3(grid).c_str() << " of block " << b_util::pd3(block).c_str() << " smem " << smem);
	if(syncHappy)b_util::syncGpu("after");
}


template <typename T> __device__ inline T devMaxValue() {
	T ret = 0;
	T* ptr = &ret;
	if(sizeof(T) == 4) {
		*ptr = 0x7f7fffff;
	} else if( sizeof(T) == 8) {
		*ptr = 0x7fefffffffffffff;
	}
	return ret;
}

template<typename T> __global__ void maxColumnIndicesSubTileWidthKernel(
		const T* aElements,
		T* bElements,
		uint height,
		uint width,
		uint pitch
		) {
	T* tile = SharedMemory<T>();
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    uint indexIn = yIndex * pitch + xIndex;

    // fill block-sized smem tile with chunk of A
    if(xIndex < width && yIndex < height)
    	tile[threadIdx.y * blockDim.x + threadIdx.x] = aElements[indexIn];
    __syncthreads();

	T theMax = -devMaxValue<T>();
	T curr = theMax;
	for(int col = 0; col < width; col++) {
		if(threadIdx.x == 0) {
			curr = tile[threadIdx.y * blockDim.x + col];
		}
		if( curr > theMax) {
			theMax = curr;
			if(threadIdx.x == 0) {
				bElements[yIndex] = col;
			}
		}
	}
}

// b is a.m * gridDim.x
// each column holds row max for that block
template<typename T> __global__ void maxColumnGridValuesKernel(
		const T* aElements,
		T* bElements,
		uint height,
		uint width,
		uint pitch
		) {
	T* tile = SharedMemory<T>();
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    uint tileY = threadIdx.y * blockDim.x;
    uint indexIn = yIndex * pitch + xIndex;

    // fill block-sized smem tile with chunk of A
    if(xIndex < width && yIndex < height)
    	tile[tileY + threadIdx.x] = aElements[indexIn];

    __syncthreads();

    const T theMax = -devMaxValue<T>();
	T currMax = theMax;
	T curr;
	// first thread of each block row iterates across tile row
	if(threadIdx.x == 0) {
		for(int col = 0; col < blockDim.x; col++) {
			if(xIndex + col < width && yIndex < height) {
				curr = tile[tileY + col];
				if( curr > currMax)
					currMax = curr;
			}
		}
		if(currMax > theMax)
			bElements[yIndex * gridDim.x + blockIdx.x] = currMax;
	}
}

template<typename T> __global__ void indicesOfValuesKernel(
		T* trgElements,
		const T* source,
		const T* values,
		uint height,
		uint width,
		uint pitch)
{
	T* tile = SharedMemory<T>();
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    uint indexIn = yIndex * pitch + xIndex;
    if(threadIdx.x == 0 && yIndex < height) {
    	tile[threadIdx.y] = values[yIndex];
    }
    __syncthreads();
    if( xIndex < width && yIndex < height && tile[threadIdx.y] == source[indexIn]) {
    	trgElements[yIndex] = xIndex;
    }

}

/*
 * maps an m*n source matrix into a m*1 vector whose each row is the index of the max-valued column for that row in the source matrix
 */
template<typename T> void CuMatrix<T>::toMaxColumnIndexVectorL(DMatrix<T>& trg, const DMatrix<T>& src) {
	outln("toMaxColumnIndexVectorL on " << util<T>::pdm(src).c_str());
	uint blockW = b_util::nextPowerOf2(src.n);
	if(blockW <= 32) {
		uint blockH = maxH<T>(caps,blockW);
		dim3 block(blockW, blockH);
		outln("maxcv-st block " << b_util::pd3(block));
		dim3 grid(b_util::enough(blockW, src.n), b_util::enough(blockH, src.m));
		outln("maxcv-st grid " << b_util::pd3(grid));
		uint smem = blockW*blockH*sizeof(T);
		outln("maxcv-st smem " << smem);
		maxColumnIndicesSubTileWidthKernel<<< grid, block, smem>>> (src.elements, trg.elements, src.m, src.n, src.p);
		checkCudaError(cudaDeviceSynchronize());
	} else {
		blockW = 32;
		T* source = src.elements;
		vector<T*> buffers;
		DMatrix<T> buffMat;
		buffMat.n = src.n;
		buffMat.elements = null;
		uint sourceW = src.n;
		uint sourceH = src.m;
		uint sourceP = src.p;
		do {
			uint blockH = MIN(maxH<T>(caps,blockW), src.m);
			dim3 block(blockW, blockH);
			outln("maxcv block " << b_util::pd3(block));
			dim3 grid(b_util::enough(blockW, buffMat.n), b_util::enough(blockH, src.m));
			outln("maxcv grid " << b_util::pd3(grid));
			if(buffMat.elements != null) {
				buffers.insert(buffers.end(),buffMat.elements);
			}
			buffMat.m = sourceH;
			buffMat.p = buffMat.n = grid.x;
			uint size = src.m * src.p * sizeof(T);
			checkCudaError(cudaMalloc((void**)&buffMat.elements, size));
			uint smem = blockW*blockH*sizeof(T);
			outln("maxcv smem " << smem);
			maxColumnGridValuesKernel<<< grid, block, smem>>>(source, buffMat.elements, sourceH, sourceW, sourceP);
			checkCudaError(cudaDeviceSynchronize());
			outln("returned buffer " << util<T>::pdm(buffMat));
			// last buffer becomes next source
			blockW = grid.x;
			sourceW = sourceP = buffMat.n;
			source = buffMat.elements;
		} while(blockW > 1);
		blockW = b_util::nextPowerOf2(src.n);
		uint blockH = maxH<T>(caps,blockW);
		dim3 block(blockW, blockH);
		outln("indicesOfVals block " << b_util::pd3(block));
		dim3 grid(b_util::enough(blockW, src.n), b_util::enough(blockH, src.m));
		outln("indicesOfVals grid " << b_util::pd3(grid));
		uint smem = blockW*blockH*sizeof(T);
		outln("indicesOfVals smem " << smem);
		indicesOfValuesKernel<<< grid, block, smem>>> ( trg.elements, src.elements, source, src.m, src.n, src.p);
		checkCudaError(cudaDeviceSynchronize());
		outln("freeing " << buffers.size() << " temp buffers");
		util<T>::cudaFreeVector(buffers);
	}
	checkCudaError(cudaGetLastError());
}

template <typename T> void CuMatrix<T>::copy1D(T* trg, const T* src, uint amountInTs, uint offsetInTs, uint lengthInTs, cudaStream_t* stream) {

	dim3 block(32);
	dim3 grid(b_util::enough(block.x, lengthInTs));
	if(stream)
		copyKernel<<<grid,block,0,*stream>>>(trg,src, amountInTs, offsetInTs, lengthInTs);
	else
		copyKernel<<<grid,block>>>(trg,src, amountInTs, offsetInTs, lengthInTs);
}

template <typename T> void CuMatrix<T>::copy(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	copyUint(trg, src, troff, tcoff);
}

template <typename T> void CuMatrix<T>::copyUlong(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(b_util::enough(block.x, src.n), b_util::enough(block.x, src.m));
	copyDmKernelUlong<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::copyUint(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(b_util::enough(block.x, src.n), b_util::enough(block.x, src.m));
	copyDmKernelUint<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::copyUlongDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(b_util::enough(block.x, src.n), b_util::enough(block.x, src.m));
	copyDmKernelUlongDvrg<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::copyUintDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(b_util::enough(block.x, src.n), b_util::enough(block.x, src.m));
	copyDmKernelUintDvrg<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::copyIntDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(b_util::enough(block.x, src.n), b_util::enough(block.x, src.m));
	copyDmKernelIntDvrg<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::shuffleCopyRows(DMatrix<T>& trg, const DMatrix<T>& src, uint* rowIndices) {
	dim3 block(32,8);
	dim3 grid(b_util::enough(block.x, trg.n), b_util::enough(block.x, trg.m));
	outln("grid " << b_util::pd3(grid));
	copyDmRowShuffleKernel<<<grid,block>>>(trg,src, rowIndices);
}

#include "CuMatrixInster.cu"

