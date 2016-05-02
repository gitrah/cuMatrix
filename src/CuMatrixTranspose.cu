/*
 * CuMatrixTranspose.cu
 *
 */
#include "CuMatrix.h"
#include "util.h"
#include "debug.h"
#include "caps.h"
#include "MatrixExceptions.h"

int blockH = TX_BLOCK_SIZE/4;

// tiles must be square
template <typename T> __global__ void transposeNaive(const T* sElements, T* tElements, int width, int height) {
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
template void __global__ transposeNaive<float>(const float*,float*,int,int);
template void __global__ transposeNaive<double>(const double*,double*,int,int);
template void __global__ transposeNaive<ulong>(const ulong*,ulong*,int,int);

template <typename T> __global__ void transposeSubTile(const T* sElements, T* tElements, int width, int height)
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

template <typename T> __global__ void transposeCoalesced(const T* sElements, T* tElements, int width, int height)
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
template void __global__ transposeCoalesced<float>(const float*,float*,int,int);
template void __global__ transposeCoalesced<double>(const double*,double*,int,int);
template void __global__ transposeCoalesced<ulong>(const ulong*,ulong*,int,int);

template <typename T>
__global__ void transposeNoBankConflicts(const T* sElements, T* tElements, int width, int height)
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

template <typename T> __global__ void transposeDiagonalKernel(const T* sElements, T* tElements, int width, int height)
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

template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::transposeKernelPtrL(DMatrix<T>& t,void (*kernel)(const T*, T*, int, int), const DMatrix<T>& s )  {
	ulong len = s.m * s.n;
	assert( len == t.m * t.n );
	int blockW = TX_BLOCK_SIZE;
	dim3 block;
	defaultBlock(block);
    dim3 grid(DIV_UP(s.n, blockW), DIV_UP(s.m, blockW));

    void (*txNmbcPtr)(const T*,T*,int,int);
    txNmbcPtr=&transposeNoBankConflicts;
    void (*txDiagPtr)(const T*,T*,int,int);
    txDiagPtr=&transposeDiagonalKernel;
	int tileWidth = blockW;
	if(kernel == txNmbcPtr || kernel == txDiagPtr) {
		tileWidth++;
	}
	int smem = TX_BLOCK_SIZE * (tileWidth)* sizeof(T);
	kernel<<<grid, block, smem>>>(s.elements, t.elements, s.n, s.m);
	//outln("tx with grid " << b_util::pd3(grid).c_str() << " of block " << b_util::pd3(block).c_str() << " smem " << smem);
}

template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::transposeL( DMatrix<T>& t, const DMatrix<T>& s)  {\

    void (*txNmbcPtr)(const T*,T*,int,int);
    txNmbcPtr=&transposeDiagonalKernel;

    int tdev =  util<T>::getDevice(t.elements);
    int sdev =  util<T>::getDevice(t.elements);
    if(checkDebug(debugTxp))flprintf("trg elems %p (dev %d) src %p (dev %d)\n", t.elements,tdev, s.elements, sdev);
    assert( tdev == sdev );

    //txNmbcPtr=&transposeNoBankConflicts;
	transposeKernelPtrL(t, txNmbcPtr,s);
	cherr(cudaDeviceSynchronize());
}

template<typename T> CuMatrix<T> CuMatrix<T>::transposeKernelPtr(void (*kernel)(const T*  sElements,  T* tElements, int width, int height)) {

	if(vectorQ()) {
		if(checkDebug(debugTxp))prlocf("degenerate tx");
		CuMatrix<T> ret = copy(true);
		ret.m = n;
		ret.n = m;
		return ret;
	}
	assert(tiler.tileSize == tiler.m_size);
	CuMatrix<T> ret(n,m,true,true);
#ifndef __CUDA_ARCH__
	if(checkDebug(debugTxp))outln("tx on dev " << ExecCaps::currDev() << " from " << this->toShortString() << " to " << ret.toShortString() );
#endif
	DMatrix<T> retD, d_A;
	tile0(d_A, lastMod == mod_host);
	ret.tile0(retD, false);
	transposeKernelPtrL(retD, kernel, d_A);
	ret.invalidateHost();
	return ret;
}


template<typename T>  __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::transpose() const {
	if(scalarQ()) {
		return *this;
	}
	if(vectorQ() && n == p) {
		if(checkDebug(debugTxp)) prlocf("transpose() on nonaliased vector");
		CuMatrix<T> ret = copy(true);
		ret.m = n;
		ret.n = m;
		ret.p = m;
		ret.tiler.m_m = n;
		ret.tiler.m_n = m;
		ret.tiler.m_p = m;
/*
		ret.ownsBuffers = ownsBuffers;
		ret.elements = elements;
		ret.tiler = tiler;
		ret.tiler.m_m = n;
		ret.tiler.m_n = m;
		ret.lastMod = lastMod;
		ret.size = size;
		if(ret.tiler.hasDmemQ()) ret.getMgr().addTiles(&(ret.tiler));
		if(ret.elements) ret.getMgr().addHost(ret);
		if(checkDebug(debugTxp)) outln("spoofing transpose for column/row matrix " << toShortString());
*/
		return ret;
	}
	assert(tiler.tileSize == tiler.m_size);
	//ExecCaps_setDevice( tiler.nextGpu());
	CuMatrix<T> ret(n,m, true,true);
	//if(checkDebug(debugTxp)) outln("tx from " << toShortString() << " to " << ret.toShortString() );
	DMatrix<T> retD, d_A;
	tile0(d_A, lastMod == mod_host);
	ret.tile0(retD, false);
	transposeL(retD, d_A);
	ret.invalidateHost();

	return ret;
}

template<typename T> void CuMatrix<T>::transposeKernelPtr(DMatrix<T>& retD, void (*kernel)(const T*,T*,int,int)) {
	DMatrix<T>  d_A;
	tile0(d_A, lastMod == mod_host);
	transposeKernelPtrL(retD, kernel, d_A);
	invalidateHost();
}

template<typename T> void CuMatrix<T>::transpose(DMatrix<T>& retD) {
	DMatrix<T>  d_A;
	tile0(d_A, lastMod == mod_host);
	transposeL(retD, d_A);
	invalidateHost();
}



#include "CuMatrixInster.cu"

