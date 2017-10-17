#include "CuMatrix.h"
#include "util.h"
//#include "CuFunctor.h"
#include "debug.h"
#include "caps.h"
#include "MatrixExceptions.h"

template <typename T> __device__ inline T devMaxValue() {
	T ret = 0;
	T* ptr = &ret;
	if(sizeof(T) == 4) {
		*ptr = 0x7f7fffff;
	} else if( sizeof(T) == 8) {
		*ptr = (T)0x7fefffffffffffff;
	}
	return ret;
}

template<typename T> __global__ void maxColumnIndicesSubTileWidthKernel(
		const T* aElements,
		T* bElements,
		int height,
		int width,
		int pitch
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
		int height,
		int width,
		int pitch
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
		int height,
		int width,
		int pitch)
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
		uint blockH = maxH<T>(*ExecCaps::currCaps(),blockW);
		dim3 block(blockW, blockH);
		outln("maxcv-st block " << b_util::pd3(block));
		dim3 grid(DIV_UP(src.n,blockW), DIV_UP(src.m,blockH));
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
			uint blockH = MIN(maxH<T>(*ExecCaps::currCaps(),blockW), src.m);
			dim3 block(blockW, blockH);
			outln("maxcv block " << b_util::pd3(block));
			dim3 grid(DIV_UP( buffMat.n,blockW), DIV_UP(src.m,blockH));
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
		uint blockH = maxH<T>(*ExecCaps::currCaps(),blockW);
		dim3 block(blockW, blockH);
		outln("indicesOfVals block " << b_util::pd3(block));
		dim3 grid(DIV_UP(src.n,blockW), DIV_UP(src.m,blockH));
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

template<typename T> inline IndexArray CuMatrix<T>::rowIndices(int row) const {
	dassert( validRowQ(row));
	if (colMajor) {
		uint* ary = new uint[n];
		for(uint i =0; i< n; i++) {
			ary[i] = i + row * p;
		}
		return IndexArray(ary, n);
	} else  {
		uint start = row * n;
		return IndexArray(start, start + n - 1);
	}
}

template<typename T> inline IndexArray CuMatrix<T>::columnIndices(int col) const {
	dassert( validColQ(col));
	if (colMajor) {
		uint start = col * m;
		return IndexArray(start, start + m - 1);
	} else {
		uint* c = new uint[m];
		for(uint i = 0; i < m; i++) {
			c[i] = i * m + col;
		}
		return IndexArray(c, m);
	}
}

template<typename T> CuMatrix<T> CuMatrix<T>::toMaxColumnIndexVector() const {
	dassert(tiler.tileSize >= tiler.m_size);
	CuMatrix<T> ret(m,1,false, true);
	DMatrix<T> d_A, d_res;
	tile0(d_A, lastMod == mod_host);
	ret.tile0(d_res, lastMod == mod_host);
	toMaxColumnIndexVectorL(d_res,d_A);
	ret.invalidateHost();
	return ret;
}

template<typename T> T CuMatrix<T>::min(  cudaStream_t stream ) const {
	return reduce(Functory<T,minBinaryOp>::pinch(), util<T>::maxValue());
}

template<typename T> T CuMatrix<T>::max( cudaStream_t stream ) const {
	return reduce(Functory<T,maxBinaryOp>::pinch(), util<T>::minValue());
}

/*
template<typename T> T CuMatrix<T>::vectorLength() const {
	return ::sqrt(autoDot());
}
*/



// fixme not stl on this side of the membrane! use float2 and double2
template<typename T> __host__ pair<T,T> CuMatrix<T>::bounds() const {

	DMatrix<T> d_A;
	tile0(d_A, lastMod == mod_host);

	cudaStream_t stream[2];
	for(int i = 0; i < 2; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
	CuTimer watch;

	cudaEvent_t start_event, stop_event;

	checkCudaErrors(cudaEventCreateWithFlags(&start_event, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&stop_event, cudaEventDisableTiming));

    checkCudaErrors(cudaEventRecord(start_event, 0));
    watch.start();
    T min,max;

    // toto find out why noworkdis
    reduceAsync(&min, d_A, Functory<T,minBinaryOp>::pinch(), util<T>::maxValue(), stream[0]);
    //if(checkDebug(debugExec))outln("min launch");
    reduceAsync(&max, d_A, Functory<T,maxBinaryOp>::pinch(), util<T>::minValue(), stream[1]);
    //if(checkDebug(debugExec))outln("max launch");

    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));   // block until the event is actually recorded
    //if(checkDebug(debugExec))outln("minmax took " << watch.stop());
	b_util::syncGpu();
	for(int i = 0; i < 2; i++) {
		checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
    checkCudaErrors(cudaEventDestroy(start_event));
	checkCudaErrors(cudaEventDestroy(stop_event));

	return pair<T,T>(min,max);
}

template<typename T> void CuMatrix<T>::bounds(T* min, T* max) const {
	dassert(tiler.tileSize >= tiler.m_size);
	DMatrix<T> d_A;
	tile0(d_A, lastMod == mod_host);
	cudaStream_t stream[2];
	for(int i = 0; i < 2; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
    CuTimer timer;
    float exeTime,exeTime2;

	cudaEvent_t start_event, stop_event;

	checkCudaErrors(cudaEventCreateWithFlags(&start_event, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&stop_event, cudaEventDisableTiming));

    checkCudaErrors(cudaEventRecord(start_event, 0));
    timer.start();
    reduceAsync(min, d_A, Functory<T,minBinaryOp>::pinch(), util<T>::maxValue(), stream[0]);
    exeTime = timer.stop();
    timer.start();
    if(checkDebug(debugExec))outln("min launch");
    reduceAsync(max, d_A, Functory<T,maxBinaryOp>::pinch(), util<T>::minValue(), stream[1]);
    if(checkDebug(debugExec)) outln("max launch");
    exeTime2 = timer.stop();

    //b_util::syncGpu();
    timer.start();

	for(int i = 0; i < 2; i++) {
	    checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
	if(checkDebug(debugExec))outln("min " << exeTime << ", max " << exeTime2 << ", sync/destroy " << timer.stop());
}

template<typename T> void CuMatrix<T>::bounds( T* min, T* max, DMatrix<T>& minBuff, DMatrix<T>& maxBuff, const DMatrix<T>& src, int blocks, int threads, long nP) {
	cudaStream_t stream[2];
	for(int i = 0; i < 2; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
    //CuTimer timer;
    //float exeTime,exeTime2;
    //float delTimes = 0,delTimes2 = 0;

    //timer.start();
    reduceAsyncBuffer(min, minBuff,  blocks,threads, nP, src, Functory<T,minBinaryOp>::pinch(), util<T>::maxValue(), stream[0]);
    ///exeTime = timer.stop();
    //timer.start();
    if(checkDebug(debugExec))outln("min launch");
    reduceAsyncBuffer(max, maxBuff,  blocks, threads,nP, src, Functory<T,maxBinaryOp>::pinch(), util<T>::minValue(), stream[1]);
    if(checkDebug(debugExec))outln("max launch");
   // exeTime2 = timer.stop();
  //  timer.start();
   // outln("blocked on stop event for " << timer.stop());
    //b_util::syncGpu();
    //timer.start();

	for(int i = 0; i < 2; i++) {
	    checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
	//outln("min took " << exeTime << ", max took " << exeTime2<< ", sync/destroy " << timer.stop());
}

#include "CuMatrixInster.cu"

