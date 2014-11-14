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

template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::transposeKernelPtrL(DMatrix<T>& t,void (*kernel)(const T*, T*, uint, uint), const DMatrix<T>& s )  {
	ulong len = s.m * s.n;
	assert( len == t.m * t.n );
	int blockW = TX_BLOCK_SIZE;
	dim3 block;
	defaultBlock(block);
    dim3 grid(DIV_UP(s.n, blockW), DIV_UP(s.m, blockW));

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
}

template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::transposeL( DMatrix<T>& t, const DMatrix<T>& s)  {
    void (*txNmbcPtr)(const T*,T*,uint,uint);
    txNmbcPtr=&transposeDiagonalKernel;
    //txNmbcPtr=&transposeNoBankConflicts;
	return transposeKernelPtrL(t, txNmbcPtr,s);
}

template<typename T> CuMatrix<T> CuMatrix<T>::transposeKernelPtr(void (*kernel)(const T*  sElements,  T* tElements, uint width, uint height)) {
/*
	if(scalarQ()) {
		return *this;
	}
	if(!CuMatrix<T>::txp) {
		CuMatrix<T>::txp = new CuMatrix<T>(n, m);
		CuMatrix<T>::ownsTxp = true;
		DMatrix<T> retD;
		CuMatrix<T>::txp->asDmatrix(retD, false);
		transposeL(asDmatrix(), retD);
		CuMatrix<T>::txp->lastMod = device;
		CuMatrix<T>::txp->txp = this;
		CuMatrix<T>::txp->ownsTxp = false;
		if(checkDebug(debugTxp))outln("created txp for " << this->toShortString() << " ( " << CuMatrix<T>::txp->toShortString() << ")");
	} else {
		if(checkDebug(debugTxp))outln("reusing txp for " << this->toShortString() << " ( " << CuMatrix<T>::txp->toShortString() << ")");
	}
	return *CuMatrix<T>::txp;
*/
	if(vectorQ()) {
		outln("degenerate tx");
		CuMatrix<T> ret = copy(true);
		ret.m = n;
		ret.n = m;
		return ret;
	}
	CuMatrix<T> ret(n,m,false,true);
	outln("tx from " << this->toShortString() << " to " << ret.toShortString() );
	DMatrix<T> retD;
	ret.asDmatrix(retD, false);
	transposeKernelPtrL(retD, kernel, asDmatrix());
	ret.invalidateHost();
	return ret;
}


template<typename T>  CuMatrix<T> CuMatrix<T>::transpose() const {
/*
	if(scalarQ()) {
		return *this;
	}
	if(!CuMatrix<T>::txp) {
		CuMatrix<T>::txp = new CuMatrix<T>(n, m);
		CuMatrix<T>::ownsTxp = true;
		DMatrix<T> retD;
		CuMatrix<T>::txp->asDmatrix(retD, false);
		transposeL(asDmatrix(), retD);
		CuMatrix<T>::txp->lastMod = device;
		CuMatrix<T>::txp->txp = this;
		CuMatrix<T>::txp->ownsTxp = false;
		if(checkDebug(debugTxp))outln("created txp for " << toShortString() << " ( " << CuMatrix<T>::txp->toShortString() << ")");
	} else {
		if(checkDebug(debugTxp))outln("reusing txp for " << toShortString() << " ( " << CuMatrix<T>::txp->toShortString() << ")");
	}
	return *CuMatrix<T>::txp;
*/
	if(vectorQ() && n == p) {
		if(checkDebug(debugExec)) outln("transpose() on nonaliased vector");
		CuMatrix<T> ret;
		ret.m = n;
		ret.n = m;
		ret.p = m;
		ret.ownsBuffers = ownsBuffers;
		ret.elements = elements;
		ret.d_elements = d_elements;
		ret.lastMod = lastMod;
		ret.size = size;
		if(ret.d_elements) ret.getMgr().addDevice(ret);
		if(ret.elements) ret.getMgr().addHost(ret);
		if(checkDebug(debugExec))outln("spoofing transpose for column/row matrix " << toShortString());
		return ret;
	}
	CuMatrix<T> ret(n,m, false,true);
	//outln("tx from " << toShortString() << " to " << ret.toShortString() );
	DMatrix<T> retD;
	ret.asDmatrix(retD, false);
	transposeL(retD, asDmatrix());
	ret.invalidateHost();
	return ret;
}

template<typename T> void CuMatrix<T>::transposeKernelPtr(DMatrix<T>& retD, void (*kernel)(const T*,T*,uint,uint)) {
	transposeKernelPtrL(retD, kernel, CuMatrix<T>::asDmatrix());
	invalidateHost();
}

template<typename T> void CuMatrix<T>::transpose(DMatrix<T>& retD) {
	transposeL(retD, asDmatrix());
	invalidateHost();
}



#include "CuMatrixInster.cu"

