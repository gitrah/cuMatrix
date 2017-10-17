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
template<typename T> __global__ void transposeNaive(T* tElements,
		const T* sElements, int width, int height, int spitch, int tpitch) {
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.x + threadIdx.y; // not blockDim.y, which we assume a factor of blockDim.x

	int index_in = xIndex + spitch * yIndex;
	int index_out = yIndex + tpitch * xIndex;
	// threads must iterate normal to the cache line,
	// which doesn't happen in this write to t
	for (int i = 0; i < blockDim.x; i += blockDim.y)
		if (xIndex < width && yIndex + i < height)
			tElements[index_out + i] = sElements[index_in + i * spitch];
}
template void __global__ transposeNaive<float>(float*,const float*,int,int,int,int);
template void __global__ transposeNaive<double>(double*,const double*,int,int,int,int);
template void __global__ transposeNaive<ulong>(ulong*,const ulong*,int,int,int,int);

template <typename T> __global__ void transposeSubTile(T* tElements, const T* sElements, int width, int height, int spitch, int tpitch)
{
	T* tile = SharedMemory<T>();
    uint xIndex = threadIdx.x;
    uint yIndex = threadIdx.y;
	uint vmemIdx = yIndex * spitch + xIndex;
	uint txIdx = xIndex * tpitch + yIndex;

	if(xIndex < width && yIndex < height) {
		tile[txIdx] = sElements[vmemIdx];
	}
	__syncthreads();
	if(xIndex < width && yIndex < height) {
		tElements[vmemIdx] = tile[vmemIdx];
	}
}

template <typename T> __global__ void transposeCoalesced(T* tElements, const T* sElements, int width, int height, int spitch, int tpitch)
{
	T* tile = SharedMemory<T>();

    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.x + threadIdx.y;
    uint index_in = xIndex + yIndex * spitch;
    uint tileIdxOut = threadIdx.x * blockDim.x + threadIdx.y;
    uint tileIdxIn = threadIdx.y * blockDim.x + threadIdx.x;

    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < width && yIndex + i < height )
    		tile[tileIdxIn + i * blockDim.x] = sElements[index_in + i * spitch];

    __syncthreads();

    xIndex = blockIdx.y * blockDim.x + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y; // not blockDim.y
    int index_out = xIndex + yIndex * tpitch;
    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < height && yIndex + i < width)
    		tElements[index_out + i * tpitch] = tile[tileIdxOut + i];
}
template void __global__ transposeCoalesced<float>( float*,const float*,int,int,int,int);
template void __global__ transposeCoalesced<double>(double*,const double*,int,int,int,int);
template void __global__ transposeCoalesced<ulong>(ulong*,const ulong*,int,int,int,int);

template <typename T>
__global__ void transposeNoBankConflicts(T* tElements, const T* sElements, int width, int height)
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

template <typename T>
__global__ void transposeNoBankConflictsPitch(T* tElements, const T* sElements, int width, int height, int spitch, int tpitch)
{
	T* tile = SharedMemory<T>();

    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.x + threadIdx.y;
    uint index_in = xIndex + yIndex*spitch;
    uint tileIdxIn = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    uint tileIdxOut = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < width && yIndex + i < height )
    		tile[tileIdxIn + i* ( blockDim.x +1)] = sElements[index_in + i * spitch];
    __syncthreads();
    xIndex = blockIdx.y * blockDim.x + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y;
    int index_out = xIndex + (yIndex)*tpitch;
    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < height && yIndex + i < width)
    		tElements[index_out + i * tpitch] = tile[tileIdxOut + i];

}

template <typename T> __global__ void transposeDiagonalKernel( T* tElements, const T* sElements, int width, int height)
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
template <typename T> __global__ void transposeDiagonalPitchKernel( T* tElements, const T* sElements, int width, int height, int spitch, int tpitch)
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
    uint index_in = xIndex + (yIndex)*spitch;
    uint tileIdx = threadIdx.y * (blockDim.x + 1)+ threadIdx.x;
    uint tileIdxOut = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < width && yIndex + i < height )
    		tile[tileIdx + i * (blockDim.x + 1)] = sElements[index_in+i*spitch];

    __syncthreads();
    xIndex = blockIdx_y * blockDim.x + threadIdx.x;
    yIndex = blockIdx_x * blockDim.x + threadIdx.y;
    int index_out = xIndex + (yIndex)*tpitch;
    for(int i = 0; i < blockDim.x; i += blockDim.y)
    	if(xIndex < height && yIndex + i < width)
    		tElements[index_out + i * tpitch] = tile[tileIdxOut + i];

}
template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::transposeKernelPtrL(DMatrix<T>& t,void (*kernel)( T*, const T*, int,int), const DMatrix<T>& s , cudaStream_t stream )  {
	ulong len = s.m * s.n;
	assert( len == t.m * t.n );
	int blockW = TX_BLOCK_SIZE;
	dim3 block;
	defaultBlock(block);
    dim3 grid(DIV_UP(s.n, blockW), DIV_UP(s.m, blockW));

    void (*txNmbcPtr)( T*,const T*,int,int);
    txNmbcPtr=&transposeNoBankConflicts;
    void (*txDiagPtr)(T*,const T*,int,int);
    txDiagPtr=&transposeDiagonalKernel;
	int tileWidth = blockW;
	if(kernel == txNmbcPtr || kernel == txDiagPtr) {
		tileWidth++;
	}
	int smem = TX_BLOCK_SIZE * (tileWidth)* sizeof(T);
	kernel<<<grid, block, smem, stream>>>(t.elements, s.elements, s.n, s.m );
	//outln("tx with grid " << b_util::pd3(grid).c_str() << " of block " << b_util::pd3(block).c_str() << " smem " << smem);
}
template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::transposePitchKernelPtrL(DMatrix<T>& t,void (*kernel)( T*, const T*, int, int,int,int), const DMatrix<T>& s , cudaStream_t stream )  {
	ulong len = s.m * s.n;
	assert( len == t.m * t.n );
	int blockW = TX_BLOCK_SIZE;
	dim3 block;
	defaultBlock(block);
    dim3 grid(DIV_UP(s.n, blockW), DIV_UP(s.m, blockW));

    static void (*txNmbcPtr)( T*,const T*,int,int,int,int);
    txNmbcPtr=&transposeNoBankConflictsPitch;
    static void (*txDiagPtr)(T*,const T*,int,int,int,int);
    txDiagPtr=&transposeDiagonalPitchKernel;
	int tileWidth = blockW;
	if(kernel == txNmbcPtr || kernel == txDiagPtr) {
		tileWidth++;
	}
	int smem = TX_BLOCK_SIZE * (tileWidth)* sizeof(T);
	kernel<<<grid, block, smem, stream>>>(t.elements, s.elements, s.n, s.m, s.p, t.p);
	//outln("tx with grid " << b_util::pd3(grid).c_str() << " of block " << b_util::pd3(block).c_str() << " smem " << smem);
}

template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::transposeL( DMatrix<T>& t, const DMatrix<T>& s, cudaStream_t stream)  {\

    void (*txNmbcPtr)( T*,const T*,int,int,int,int);
    txNmbcPtr=&transposeDiagonalPitchKernel;

#ifndef __CUDA_ARCH__
    int tdev =  b_util::getDevice((void*)t.elements);
    int sdev =  b_util::getDevice((void*)s.elements);
    if(checkDebug(debugTxp))flprintf("trg elems %p (dev %d) src %p (dev %d)\n", t.elements,tdev, s.elements, sdev);
    assert( tdev == sdev );
#endif

    //txNmbcPtr=&transposeNoBankConflicts;
    transposePitchKernelPtrL(t, txNmbcPtr,s);
	cherr(cudaDeviceSynchronize());
}

template<typename T> CuMatrix<T> CuMatrix<T>::transposeKernelPtr(
		void (*kernel)(T* sElements, const T* tElements, int width,
				int height)) {

	if (vectorQ()) {
		if (checkDebug(debugTxp))
			prlocf("degenerate tx");
		CuMatrix<T> ret = copy(true);
		ret.m = n;
		ret.n = m;
		ret.p = m;
		ret._tileP = m;
		return ret;
	}
	assert(tiler.tileSize >= tiler.m_size);
	CuMatrix<T> ret(n, m, true, true);
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

template<typename T> CuMatrix<T> CuMatrix<T>::transposePitchKernelPtr(
		void (*kernel)(T*, const T*, int, int, int, int)) {

	if (vectorQ()) {
		if (checkDebug(debugTxp))
			prlocf("degenerate tx");
		CuMatrix<T> ret = copy(true);
		ret.m = n;
		ret.n = m;
		ret.p = m;
		return ret;
	}
	assert(tiler.tileSize >= tiler.m_size);
	CuMatrix<T> ret(n, m, true, true);
#ifndef __CUDA_ARCH__
	if(checkDebug(debugTxp))outln("tx on dev " << ExecCaps::currDev() << " from " << this->toShortString() << " to " << ret.toShortString() );
#endif
	DMatrix<T> retD, d_A;
	tile0(d_A, lastMod == mod_host);
	ret.tile0(retD, false);
	transposePitchKernelPtrL(retD, kernel, d_A);
	ret.invalidateHost();
	return ret;
}


template<typename T>  __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::transpose(cudaStream_t stream ) const {
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
		ret._tileP = m;
/*
		ret.ownsDBuffers = ownsDBuffers;
		ret.ownsHBuffers = ownsHBuffers;
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
	if(tiler.tileSize < tiler.m_size) {
		return transposeXr();
	}
	CuMatrix<T> ret(n,m, true,true);
	DMatrix<T> retD, d_A;
	tile0(d_A, lastMod == mod_host);
	ret.tile0(retD, false);
	transposeL(retD, d_A, stream );
	ret.invalidateHost();

	return ret;
}


template<typename T>  __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::transposeXr(cudaStream_t stream) const {
	CuMatrix<T> ret(n,m, true,true);
	/*
	 *  tile of source starts at 0,0 and moves right
	 *  tiler of target starts at 0,0  and moves down
	 *  each subs source starts at lr corner of last start tile and moves right
	 *
	 tile2D(DMatrix<T>& dm,
            uint& roff, uint& coff,
            uint& tileM, uint& tileN,
            int rowTileIdx, int colTileIdx,
            int rowTileCount, int colTileCount,
            bool copy = true, int lastGpu =-1, cudaStream_t stream = 0)

	 */


	DMatrix<T> d_A, d_B;

	int maxTileD = (int) (sqrtf( (float) tiler.tileSize) * MIN(m,n)/MAX(m,n));
	int tileD = b_util::prevPowerOf2(maxTileD);
	tileD = MIN(m, MIN(n, tileD));
	int colSteps = DIV_UP(n,tileD);
	int rowSteps = DIV_UP(m, tileD);

	const Tiler<T>* btiler =&(ret.tiler);

	int aroff = 0,acoff = 0;
	int lastGpu = -1;
	int gpuCount = tiler.countGpus();
	int orgDevice = ExecCaps::currDev();
    int rowTileIdx, colTileIdx;
    int rowTileCount, colTileCount;

	cudaStream_t* streams = nullptr;
	lastGpu = tiler.nextGpu(lastGpu);
	int tileM = _tileM, tileN = _tileN, tileP = _tileP;
	if(gpuCount > 1) {
		assert(!stream);
		cudaStream_t* streams = (cudaStream_t* ) malloc(gpuCount * sizeof(cudaStream_t));
		for(int i =0 ; i < gpuCount; i++) {
			lastGpu = tiler.nextGpu(lastGpu);
			ExecCaps_setDevice(lastGpu);
			cherr(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));
		}
	}

	int coliStart = 0;
    for(int rowi = 0; rowi < rowSteps; rowi++) {
    	for(int coli = coliStart; coli < colSteps; coli++) {
    		tiler.tile2D(d_A, aroff, acoff, tileM, tileN, tileP, rowSteps, colSteps, rowi, coli, true,  lastGpu, gpuCount > 1 ? streams[coli] : stream);
    		btiler->tile2D(d_B, acoff, aroff, tileM, tileN, tileP, rowSteps, colSteps, rowi, coli, false, lastGpu, gpuCount > 1 ? streams[coli] : stream);
    		transposeL(d_A,d_B, gpuCount > 1 ? streams[coli] : stream);
    	}
    	coliStart++;
	}
}


template<typename T> void CuMatrix<T>::transposeKernelPtr(DMatrix<T>& retD, void (*kernel)( T*, const T*,int,int)) {
	DMatrix<T>  d_A;
	tile0(d_A, lastMod == mod_host);
	transposeKernelPtrL(retD, kernel, d_A);
	invalidateHost();
}

template<typename T> void CuMatrix<T>::transposePitchKernelPtr(DMatrix<T>& retD, void (*kernel)( T*, const T*,int,int,int,int)) {
	DMatrix<T>  d_A;
	tile0(d_A, lastMod == mod_host);
	transposePitchKernelPtrL(retD, kernel, d_A);
	invalidateHost();
}

template<typename T> void CuMatrix<T>::transpose(DMatrix<T>& retD) {
	DMatrix<T>  d_A;
	tile0(d_A, lastMod == mod_host);
	transposeL(retD, d_A);
	invalidateHost();
}



#include "CuMatrixInster.cu"

