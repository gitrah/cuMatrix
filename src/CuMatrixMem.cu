#include "CuMatrix.h"
#include "util.h"
#include "Kernels.h"
#include "debug.h"
#include "caps.h"
#include "MatrixExceptions.h"
#include <typeinfo>
#include <thread>


/*
template<typename T> __host__ __device__ void CuMatrix<T>::calcTiles(float headroom) {
	assert(!tiler.nTiles);
	ExecCaps* pcaps = ExecCaps::currCaps();
	ulong tile = pcaps->maxReasonable(headroom);
	tiler.nTiles = DIV_UP(size,tile);
}
*/
template<typename T> __host__ __device__ int CuMatrix<T>::releaseBuffers() {
	int clearCount= 0;
	if(elements != null) {
		getMgr().freeHost(*this);
		clearCount++;
		elements=nullptr;
	}
	if(tiler.hasDmemQ()) {
		clearCount += getMgr().freeTiles(*this);
		tiler.clear();
	}
	return clearCount;
}
template<typename T> __host__ __device__ CuMatrix<T>& CuMatrix<T>::syncBuffers(bool copy ) {
	if(checkDebug(debugMem)){
		printf("syncBuffers(%s) on %dX%d currBuffer() %p\n", tOrF(copy), m,n,tiler.currBuffer());
#ifndef __CUDA_ARCH__
		outln("[caller " << b_util::caller() << "]");
		if(checkDebug(debugCopyDh)) {
			outln("[["<< b_util::unmangl(typeid(this).name())<< "]] from syncBuffers\n" << print_stacktrace());
		}
#else
		prlocf("syncBuffers called from device");
#endif
	}
	int currDev = ExecCaps::currDev();
	if(lastMod != mod_synced) {
		if (lastMod == mod_device) {
			assert(tiler.tileSize == tiler.m_size);
			//outln*
#ifndef __CUDA_ARCH__
			if(!elements) {
				if(checkDebug(debugMem)) outln("syncBuffers !elements");
				cherr(cudaPeekAtLastError());
				getMgr().allocHost(*this);
			}

			if(n != p) {
				if (checkDebug(debugMem))
					outln( "syncBuffers() n != p so doing line by line copy");
				cudaError_t res = cudaMemcpy2D(elements, p* sizeof(T), tiler.buffer(currDev), p* sizeof(T), n * sizeof(T), m, cudaMemcpyDeviceToHost);
				if(res != cudaSuccess) {
					outln("ERR m " << toShortString());
					flprintf("elements %p destpitch %u dev src %p srcP %u widthBytes %u height %u\n", elements, p* sizeof(T), tiler.buffer(currDev), p* sizeof(T), n * sizeof(T), m);
				}
				cherr(res);
			}else {
				if (checkDebug(debugCopy | debugMem | debugTiler)) {
					flprintf("currDev %d, elements %p, tiler.buffer(currDev) %p, size %d\n", currDev, elements, tiler.buffer(currDev) , size);
					MemMgr<T>::checkValid(tiler.buffer(currDev));
					MemMgr<T>::checkValid(tiler.buffer(currDev) + (size - 1)/sizeof(T));
					MemMgr<T>::checkValid(elements);
					MemMgr<T>::checkValid(elements +(size - 1)/sizeof(T));
				}

				cherr(cudaMemcpy(elements, tiler.buffer(currDev) , size, cudaMemcpyDeviceToHost));
			}
			//err = cudaMemcpy(elements, d_elements, size, cudaMemcpyDeviceToHost);
			lastMod = mod_synced;
			DHCopied++;
			MemDhCopied += size;
			if (checkDebug(debugCopy | debugMem))
				outln( "syncBuffers() mat " << this << " copied " << size << " from d " << tiler.currBuffer() << " to  h " << elements);
#else
			printShortString("WARN syncBuffers can't update host from device");
			setLastError(cantSyncHostFromDeviceEx);
#endif
		} else {
/*
			if(size  < ExecCaps::currCaps()->maxReasonable()) {
				flprintf("size %u maxReas %u\n",size, ExecCaps::currCaps()->maxReasonable());
			}
			assert(size  < ExecCaps::currCaps()->maxReasonable());
*/
			if(!tiler.hasDmemQ()) {
				tiler.allocTiles();
				getMgr().addTiles(&tiler);
			}
			if(checkDebug(debugCheckValid)) flprintf("%dx%dx%d - %p\n", m, n, p, tiler.buff());
			DMatrix<T> dm;
			tile0(dm,copy);
			lastMod = mod_synced;
		}
	}
	return *this;
}

template<typename T> __host__ __device__  CuMatrix<T> CuMatrix<T>::syncHost() {
	invalidateHost();
	return syncBuffers();
}

template<typename T> __host__ __device__  CuMatrix<T> CuMatrix<T>::syncDevice() {
	invalidateDevice();
	return syncBuffers();
}

template<typename T> __host__ __device__ void CuMatrix<T>::invalidateHost() {
/*
	if(!tiler.hasDmemQ()) {
		setLastError(noDeviceBufferEx);
	}
*/
	if(checkDebug(debugMem) && lastMod != mod_device) {
#ifndef __CUDA_ARCH__
		outln("matrix " << this << " invalHost clr " << b_util::callerN(3));
#else
		printf("matrix %p invalHost clr\n",this);
#endif
	}
	lastMod = mod_device;
	freeTxp();
}

template<typename T> __host__ __device__ void CuMatrix<T>::invalidateDevice() {
	if(!elements) {
		setLastError(noHostBufferEx);
	}
	if(checkDebug(debugMem) && lastMod != mod_host) {
#ifndef __CUDA_ARCH__
		outln("matrix " << this << " invalidateDevice caller " << b_util::callerN(3));
#else
		printf("matrix %p invalidateDevice clr\n",this);
#endif
	}
	lastMod = mod_host;
	freeTxp();
}

template <typename T> __global__ void
copyKernel(T* tElements, const T* sElements, uint tWidth, uint tHeight, uint tPitch, uint sWidth, uint sHeight, uint sPitch, uint xOff, uint yOff)
{
    ulong x = blockIdx.x * blockDim.x + threadIdx.x;
    ulong y = blockIdx.y * blockDim.y + threadIdx.y;
    ulong tx = x + xOff;
    ulong ty = y + yOff;
    ulong sIdx = y * sPitch + x;
    ulong tIdx = ty * tPitch + tx;
    if(threadIdx.x == 0 && threadIdx.y == 0) {
    	if(checkDebug(debugCopy))flprintf("block %u,%u tx %lu ty %lu sIdx %lu tIdx %lu\n", blockIdx.x, blockIdx.y, tx,ty,sIdx,tIdx);
    }
    if(x < sWidth && y < sHeight && tx < tWidth && ty < tHeight) {
    	tElements[tIdx] = sElements[sIdx];
    }
}

// each thread copies a row
template <typename T> __global__ void
rowCopyKernel(T* tElements, const T* sElements, uint tHeight, uint tPitchTs, uint sWidth, uint sHeight, uint sPitchTs, uint xOff, uint yOff)
{
    ulong y = blockIdx.y * blockDim.y + threadIdx.y;
    ulong ty = y + yOff;
    ulong sIdx = y * sPitchTs ;
    ulong tIdx = ty * tPitchTs + xOff;
    if(threadIdx.x == 0 && threadIdx.y == 0) {
    	if(checkDebug(debugCopy))flprintf("block %u,%u y %lu ty %lu sIdx %lu tIdx %lu\n", blockIdx.x, blockIdx.y, y,ty,sIdx,tIdx);
    }
    if( y < sHeight && ty < tHeight) {
    	memcpy(tElements + tIdx, sElements + sIdx, sWidth * sizeof(T));
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
copyDmRowShuffleKernel(DMatrix<T> trg, const DMatrix<T> src, int* indices) {
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
		    		if (checkDebug(debugMem))flprintf("i (%d) !< tdiff (%d), y = %d\n", i, tdiff, y);
		    	}
			}
		}
    } else {
    	if(blockIdx.x == 0 && threadIdx.x == 0) {
    		if (checkDebug(debugMem))flprintf("%d !< trg.n or %d !< trg.m\n", x, y);
    	}
    }
}

template <typename T> __global__ void prtDevArray(const T* array, const char* msg, int line, int n, int direction) {
	FirstThread {
		printf("%s:%d darray %p[0::%d] ", msg, line, array, n);
		for(int i =0; i < n; i++) {
			printf("%f",  (float)array[direction * i]);
			if(i < n -1) printf(", ");
		}
		printf("\n");
	}
}

template <typename T> __global__ void cntDevArray(const T* array, const char* msg, int line, int n, int direction, T test) {
	FirstThread {
		int neqCnt = 0;
		int idxFb= -1;
		const T* firstBad = nullptr;
		printf("%s:%d darray %p[0::%d] ", msg, line, array, n);
		for(int i =0; i < n; i++) {
			if((array[direction * i] != test && firstBad == null) ) {
				printf("%p + %d (%p) = %f != %f", array, direction*i, array + direction*i,(float) array[direction * i], test);
//				printf("(%p) = %f", array + i*(p+1), (float)array[direction*i*(p + 1)]);
				if(i < n -1) printf(", ");
				idxFb = i; firstBad = array  + i * direction; neqCnt++;
			}
		}
		if(!neqCnt)
			flprintf("\nfound none != %f\n",test);
		else {
			flprintf("\nfound %d unexpected values starting at %p idx %d\n", neqCnt, firstBad,idxFb);
		}
	}
}

template <typename T> __global__ void prtDevArrayDiag(
		const T* array, const char* msg, int line, int p, int n, int direction, T test) {
	FirstThread {
		int neqCnt = 0;
		int idxFb= -1;
		const T* firstBad = nullptr;
		printf("%s:%d darraydiag %p (p:%d)[0::%d] ", msg, line, array, p,n);
		for(int i =0; i < n; i++) {
			if((array[i * (p + 1)] != test) || !test ) {
				printf("%p + %d (%p) = %f != %f", array, direction*i, array + direction * i * (p + 1), (float) array[i * (p + 1)], test);
//				printf("(%p) = %f", array + i*(p+1), (float)array[direction*i*(p + 1)]);
				if(i < n -1) printf(", ");
				if(test) { idxFb = i; firstBad = array  + i * (p + 1); neqCnt++; }
			}
		}
		if(!neqCnt)
			flprintf("\nfound none != %f\n",test);
		else {
			flprintf("\nfound %d unexpected values starting at %p idx %d\n", neqCnt, firstBad,idxFb);
		}
	}
}

template <typename T> __global__ void cntDevArrayDiag(
		const T* array, const char* msg, int line, int p, int n, int direction, T test) {
	FirstThread {
		int neqCnt = 0;
		int eqCnt = 0;
		int idxFirstNeq= -1, idxFirstEq;
		const T* firstNeq = nullptr;
		const T* firstEq = nullptr;

		printf("%s:%d darraydiag %p (p:%d)[0::%d] ", msg, line, array, p,n);
		for(int i =0; i < n-1; i++) {
			if(array[i * (p + 1)] == test) {
				if(firstNeq == null) {firstEq= array  + i * (p + 1); idxFirstEq = i; }
				eqCnt++;
			}else {
				if(firstEq == null) { firstNeq  = array  + i * (p + 1);idxFirstNeq = i; }
				neqCnt++;
			}
		}
		flprintf("\nfound %d neq %f, %d eq out of %d; first neq @ %p idx %d first eq %p idx %d\n", neqCnt, test, eqCnt, n, firstNeq,idxFirstNeq,firstEq,idxFirstEq);
	}
}

template <typename T> __host__ __device__
void
printDevArray(const T* array, const char* msg, int line, int n,int direction, T test) {
	int len = strlen(msg);
	char* dmsg=nullptr;
	cherr(cudaMalloc(&dmsg,len+1));
	cherr(cudaMemcpy(dmsg,msg,len+1,cudaMemcpyHostToDevice));
	//cntDevArray<<<1,1>>>(array,dmsg,line, n,direction,test);
	prtDevArray<<<1,1>>>(array,dmsg,line, n,direction);
	cherr(cudaDeviceSynchronize());
	cherr(cudaFree(dmsg));
}

template __host__ __device__ void printDevArray<float>(const float*, const char*,int,int,int,float);
template __host__ __device__ void printDevArray<double>(const double*, const char*,int,int,int,double);
template __host__ __device__ void printDevArray<int>(const int*, const char*,int,int,int,int);
template __host__ __device__ void printDevArray<uint>(const uint*, const char*,int,int,int,uint);
template __host__ __device__ void printDevArray<long>(const long*, const char*,int, int,int,long);
template __host__ __device__ void printDevArray<ulong>(const ulong*, const char*,int, int,int,ulong);

template <typename T> __host__ __device__ void printDevArrayDiag(
		const T* array, const char* msg, int line, int pitch, int n, int direction, T test) {
	flprintf("array %p line %d pitch %d n %d direction %d test %f\n", array,line,pitch,n,direction, test);
	int len = strlen(msg);
	char* dmsg=nullptr;
	cherr(cudaMalloc(&dmsg,len+1));
	cherr(cudaMemcpy(dmsg,msg,len+1,cudaMemcpyHostToDevice));
	cntDevArrayDiag<<<1,1>>>(array,dmsg,line,pitch, n, direction, test);
	//prtDevArrayDiag<<<1,1>>>(array,dmsg,line,pitch, n, direction, notEq);
	cherr(cudaDeviceSynchronize());
	cherr(cudaFree(dmsg));
}

template __host__ __device__ void printDevArrayDiag<float>(const float*, const char*,int,int,int,int,float);
template __host__ __device__ void printDevArrayDiag<double>(const double*, const char*,int,int,int,int,double);
template __host__ __device__ void printDevArrayDiag<int>(const int*, const char*,int,int,int,int,int);
template __host__ __device__ void printDevArrayDiag<uint>(const uint*, const char*,int,int,int,int,uint);
template __host__ __device__ void printDevArrayDiag<long>(const long*, const char*,int,int, int,int,long);
template __host__ __device__ void printDevArrayDiag<ulong>(const ulong*, const char*,int,int, int,int,ulong);

template <typename T> __host__ __device__ void prtDevArrayDiagSpan(
		const T* array, const char* msg, int line, int pitch, int n, int spanCount, int direction) {
	int spanSize = n/spanCount;
	flprintf("array %p line %d pitch %d n %d span %d chunkSize %d direction %d\n", array,line,pitch,n,spanCount, spanSize, direction);
	int len = strlen(msg);
	char* dmsg=nullptr;
	cherr(cudaMalloc(&dmsg,len+1));
	cherr(cudaMemcpy(dmsg,msg,len+1,cudaMemcpyHostToDevice));

	uint startIdx = 0;
	for(int s = 0; s < spanCount; s++) {
		flprintf("span %d starting at %p\n", s, array + (pitch +1 )* s);
		startIdx = s * spanSize * (pitch + 1);
		if(s == spanCount -1) {
			spanSize = n - s * spanSize;
			flprintf("last span spanSize %d\n",spanSize);
		}
		flprintf("startIdx %u\n", startIdx);
		prtDevArrayDiag<<<1,1>>>(array + startIdx,dmsg,line,pitch, spanSize, direction, (T)0);
		cherr(cudaDeviceSynchronize());
	}
	cherr(cudaFree(dmsg));
}

template __host__ __device__ void prtDevArrayDiagSpan<float>(const float*, const char*,int,int,int,int,int);
template __host__ __device__ void prtDevArrayDiagSpan<double>(const double*, const char*,int,int,int,int,int);
template __host__ __device__ void prtDevArrayDiagSpan<int>(const int*, const char*,int,int,int,int,int);
template __host__ __device__ void prtDevArrayDiagSpan<uint>(const uint*, const char*,int,int,int,int,int);
template __host__ __device__ void prtDevArrayDiagSpan<long>(const long*, const char*,int,int, int,int,int);
template __host__ __device__ void prtDevArrayDiagSpan<ulong>(const ulong*, const char*,int,int, int,int,int);


template <typename T> __global__ void
copyKernel(T* tElements, const T* sElements, uint amountInTs, uint offsetInTs, ulong lengthInTs)
{
    ulong id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < amountInTs && id + offsetInTs < lengthInTs) {
    	tElements[id + offsetInTs] = sElements[id];
    }
}

template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::rightConcatenateL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, cudaStream_t stream ) {
	assert(src1.m == src2.m);
	ExecCaps* pcaps = ExecCaps::currCaps();

	dim3 block;
	if(src1.m > src1.n) {
		block.x = MIN(MAX(src1.n, src2.n), CAT_BLOCK_SIZE);
		block.y = MIN(MIN(src1.m, pcaps->thrdPerBlock/block.x), pcaps->maxBlock.y);
	} else {
		block.y = MIN(src1.m, CAT_BLOCK_SIZE);
		block.x = MIN(MIN(src1.n, pcaps->thrdPerBlock/block.y), pcaps->maxBlock.x);
	}
	dim3 grid1(DIV_UP(src1.n,block.x), DIV_UP(src1.m,block.y),1);
	dim3 grid2(DIV_UP(src2.n,block.x), grid1.y,1);

	dim3 launches1 ( grid1.x < pcaps->maxGrid.x ? 1 : DIV_UP(pcaps->maxGrid.x, grid1.x), grid1.y < pcaps->maxGrid.y ? 1 : DIV_UP(pcaps->maxGrid.y, grid1.y));
	dim3 launches2 (grid2.x < pcaps->maxGrid.x ? 1 : DIV_UP(pcaps->maxGrid.x, grid2.x),  grid2.y < pcaps->maxGrid.y ? 1 : DIV_UP(pcaps->maxGrid.y, grid2.y));

	if(checkDebug(debugCopy)) {
		flprintf("block %u,%u\n", block.y, block.x);
		flprintf("grid1 %u,%u\n", grid1.y, grid1.x);
		flprintf("grid2 %u,%u\n", grid2.y, grid2.x);
		flprintf("launches1 %d,%d, launches2 %d,%d\n", launches1.y, launches1.x, launches2.y, launches2.x);
	}

	//copyKernel<<<gridAc,block>>>(d_C, d_A, 0, 0);
	for(int y = 0; y < launches1.y || y < launches2.y; y++) {
		for(int x =0; x < launches1.x || x < launches2.x; x++) {
			if(y < launches1.y && x< launches1.x) {
				dim3 grid(grid1.x/launches1.x,grid1.y/launches1.y);
				uint xoff = src1.n / launches1.x * x;
				uint yoff = src1.m / launches1.y * y;
				copyKernel<<<grid1,block,0,stream>>>(trg.elements, src1.elements, trg.n, trg.m, trg.p, src1.n, src1.m, src1.p, xoff, yoff);
			}
			if(y < launches2.y && x< launches2.x) {
				dim3 grid(grid2.x/launches1.x,grid2.y/launches1.y);
				uint xoff = src2.n / launches2.x * x;
				uint yoff = src2.m / launches2.y * y;
				copyKernel<<<grid2,block,0,stream>>>(trg.elements, src2.elements, trg.n, trg.m, trg.p, src2.n, src2.m, src2.p, src1.n + xoff, yoff);
			}
		}
	}
	cherr(cudaStreamSynchronize(stream));
}

template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::bottomConcatenateL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, cudaStream_t stream )  {
	dim3 block(CAT_BLOCK_SIZE,CAT_BLOCK_SIZE,1);
	dim3 dmBlock(CAT_BLOCK_SIZE,CAT_BLOCK_SIZE/8,1);
	dim3 gridAc(DIV_UP(src1.n,CAT_BLOCK_SIZE), DIV_UP(src1.m,CAT_BLOCK_SIZE),1);
	dim3 gridBc(DIV_UP(src2.n,CAT_BLOCK_SIZE), DIV_UP(src2.m,CAT_BLOCK_SIZE),1);
	if(checkDebug(debugExec)){
#ifndef __CUDA_ARCH__
		outln("gridAc " << b_util::pd3(gridAc).c_str() << " for " << util<T>::pdm(src1));
		outln("gridBc " << b_util::pd3(gridBc).c_str() << " for " << util<T>::pdm(src2));
#else
		printf("gridAc " );
		b_util::prd3(gridAc);
		//c_str() << " for " << util<T>::pdm(src1));

	//	printf("gridBc " << b_util::pd3(gridBc).c_str() << " for " << util<T>::pdm(src2));
#endif
	}
	if(src1.n == src1.p) {
		copyKernel<<<gridAc,block,0,stream>>>(trg.elements, src1.elements,trg.n, trg.m, trg.p, src1.n, src1.m, src1.p, 0, 0);
	} else {
#ifndef __CUDA_ARCH__
		dthrow(notImplemented());
#else
		setLastError(notImplementedEx);
#endif
	}
	copyKernel<<<gridBc,block,0,stream>>>(trg.elements, src2.elements, trg.n, trg.m, trg.p, src2.n, src2.m, src2.p, 0, src1.m);
}

template <typename T> void CuMatrix<T>::copy1D(T* trg, const T* src, int amountInTs, int offsetInTs, int lengthInTs, cudaStream_t stream) {

	dim3 block(32);
	dim3 grid(DIV_UP(lengthInTs, block.x));
	if(stream)
		copyKernel<<<grid,block,0,stream>>>(trg,src, amountInTs, offsetInTs, lengthInTs);
	else
		copyKernel<<<grid,block>>>(trg,src, amountInTs, offsetInTs, lengthInTs);
}

template <typename T> void CuMatrix<T>::copyK(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	copyUint(trg, src, troff, tcoff);
}

template <typename T>  __host__ CUDART_DEVICE void CuMatrix<T>::copy(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
	T* dst = trg.elements + troff*trg.p + tcoff;
	const size_t tSize= sizeof(T);
	if(checkDebug(debugCopy)) flprintf("cudaMemcpy2D(dst %p, dpitch %d, src %p, spitch %d, width %d, height %d, cpyKind %d\n",
			dst, trg.p * tSize, src.elements, src.p* tSize, src.n* tSize, src.m,cudaMemcpyDeviceToDevice);
	//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
	cherr(cudaMemcpy2D(dst, trg.p * tSize, src.elements, src.p* tSize, src.n* tSize, src.m,cudaMemcpyDeviceToDevice));
}

template <typename T> void CuMatrix<T>::copyAsync(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
	T* dst = trg.elements + troff*trg.p + tcoff;
	const size_t tSize= sizeof(T);
	checkCudaError(cudaMemcpy2DAsync(dst, trg.p * tSize, src.elements, src.p* tSize, src.n* tSize, src.m,cudaMemcpyDeviceToDevice));
}

template <typename T> void CuMatrix<T>::copyUlong(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(DIV_UP(src.n,block.x), DIV_UP(src.m,block.x));
	copyDmKernelUlong<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::copyUint(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(DIV_UP(src.n,block.x), DIV_UP(src.m,block.x));
	copyDmKernelUint<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::copyUlongDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(DIV_UP(src.n,block.x), DIV_UP(src.m,block.x));
	copyDmKernelUlongDvrg<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::copyUintDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(DIV_UP(src.n,block.x), DIV_UP(src.m,block.x));
	copyDmKernelUintDvrg<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::copyIntDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(DIV_UP(src.n,block.x), DIV_UP(src.m,block.x));
	copyDmKernelIntDvrg<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::shuffleCopyRows(DMatrix<T>& trg, const DMatrix<T>& src, int* rowIndices) {
	dim3 block(32,8);
	dim3 grid(DIV_UP(trg.n,block.x), DIV_UP(trg.m,block.x ));
//	outln("grid " << b_util::pd3(grid));
	copyDmRowShuffleKernel<<<grid,block>>>(trg,src, rowIndices);
}

template<typename T> __host__ __device__ CuMatrix<T> CuMatrix<T>::copy(bool copyDeviceMem) const {
	assert(tiler.tileSize == tiler.m_size);
	CuMatrix<T> ret( m, n, elements != 0, tiler.hasDmemQ());
#ifndef __CUDA_ARCH__
	if (elements) {
		cherr(
				cudaMemcpy(ret.elements, elements, size, cudaMemcpyHostToHost));
		HHCopied++;
		MemHhCopied += size;
	}
#endif
	if ( tiler.currBuffer() && copyDeviceMem) {
#ifndef __CUDA_ARCH__
		T* retp = ret.tiler.currBuffer();
		T* thisp = tiler.currBuffer();
		MemMgr<T>::checkValid(retp, "retp");
		MemMgr<T>::checkValid(thisp, "thisp");
		MemMgr<T>::checkValid(retp + m * n -1, "retp+ m * n -1");
		MemMgr<T>::checkValid(thisp+ m * n -1, "thisp+ m * n -1");
		cherr(
				cudaMemcpy(ret.tiler.currBuffer(), tiler.currBuffer(), size, cudaMemcpyDeviceToDevice));
		DDCopied++;
		MemDdCopied += size;
#else
		memcpy(ret.tiler.currBuffer(), tiler.currBuffer(), size);

#endif
	}
	ret.lastMod  = lastMod;
	if(checkDebug(debugMem) && ret.lastMod == mod_host) {
		printf("CuMatrix (%p::copy(%s) -> %p set lastMod of host\n",this, tOrF(copyDeviceMem),&ret );
	}
	ret.posed = posed;
	ret.colMajor = colMajor;
	ret.oldM = oldM;
	ret.oldN = oldN;
	ret.p = p;
	ret.size = size;
	if(txp && ownsTxp) {
		if(checkDebug(debugMem))printf("copy() recreating txp\n");
		ret.txp = new CuMatrix<T>(n,m, true, true);
		ret.ownsTxp = true;
		if (txp->elements) {
			if(checkDebug(debugMem))printf("copy() copying txp->elements\n");
#ifndef __CUDA_ARCH__
			cherr(
					cudaMemcpy(ret.txp->elements, txp->elements, size, cudaMemcpyHostToHost));
			HHCopied++;
			MemHhCopied += size;
#endif
		}
		if (txp->tiler.currBuffer() && copyDeviceMem) {
			setLastError(notImplementedEx);
			return CuMatrix<T>(0,0,false,false);
		}
		ret.txp->lastMod  = txp->lastMod;
		ret.txp->posed = txp->posed;
		ret.txp->colMajor = txp->colMajor;
		ret.txp->oldM =txp->oldM;
		ret.txp->oldN = txp->oldN;
		ret.txp->p = txp->p;
		ret.txp->size = txp->size;
	}
	return ret;
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::columnSubset( const int* indices,
		int count) const {
	if(count > 0) {
		int i = 1;
		CuMatrix<T> res = columnMatrix(indices[0]) ;
		while (i < count) {
			CuMatrix<T> cVec = columnMatrix(indices[i]);
			res |= cVec;
			i++;
		}
		res.printShortString("columnSubset ");
		//res.lastMod = mod_device;
		return res;
	}else {
		setLastError(illegalArgumentEx);
		return CuMatrix<T>();
	}
}

template<typename T> CuMatrix<T> CuMatrix<T>::clippedRowSubset( const int *r, int count,
		intPair colRange) const {
	if(colMajor) {
		dthrow(notImplemented())
	}
	if(!elements) {
		dthrow(noHostBuffer())
	}
	if(lastMod == mod_device) {
		dthrow(notSyncedHost());
	}
	//outln("clippedRowSubset *this " << *this);
	printf("clippedRowSubset colRange %d-%d, n %d\n", colRange.first, colRange.second,n);
	assert((colRange.first < colRange.second && colRange.second < n));
	int width = colRange.second - colRange.first + 1;
	int newM = count;
	CuMatrix<T> res = zeros(newM,width).syncBuffers();
	//res.printShortString("clippedRowSubset res " );
	//printShortString("clippedRowSubset this ");
	int i = 0;
	while (i < newM) {
		//outln("i " << i << " r[i] * p " << (r[i]*p) << ",  i * res.p " << ( i * res.p) );
		memcpy(res.elements + i * res.p, elements + r[i] * p, width * sizeof(T));
		i++;
	}
	//outln("res after " << res);
	res.lastMod = mod_host;
	//res.syncBuffers();
	//outln("clippedRowSubset res " << res);
	return res;
}

template<typename T> CuMatrix<T> CuMatrix<T>::addBiasColumn() const {
	//CuMatrix<T> bias = CuMatrix<T>::ones(m, 1);
	//bias.syncBuffers();
	return CuMatrix<T>::ones(m, 1) |= *this;
}

template<typename T> CuMatrix<T> CuMatrix<T>::replicateTiled(int mcopies, int ncopies) const {
	if(!mcopies || ! ncopies) {
		dthrow(illegalArgument());
	}
	CuMatrix<T> tiled(mcopies*m, ncopies *n,true,true);
	DMatrix<T> dTiled;
	tiled.tile0(dTiled,false);
	DMatrix<T> dSrc;
	tile0(dSrc, lastMod == mod_host);
	for(int row = 0; row < mcopies; row++) {
		for(int col = 0; col < ncopies; col++) {
			CuMatrix<T>::copyAsync(dTiled, dSrc, row*m, col*n);
		}
	}
	tiled.invalidateHost();
	checkCudaError(cudaDeviceSynchronize());
	return tiled;
}

template<typename T> void CuMatrix<T>::copy(CuMatrix<T>& res, int roff, int coff, bool onlyDevice) const {
	if(roff + m  >  res.m || coff + n > res.n ) {
		outln("roff " << roff + " +  m " << m << " > res.m" << res.m << ", or coff " << coff << " + n " << n << " > res.n " << res.n);
		dthrow(outOfBounds());
	}

	if(contiguousQ() && res.contiguousQ() && roff == 0 && coff == 0) {
		if(!onlyDevice && (!res.elements && elements)) {
			dthrow(noHostBuffer());
		}
		if(!res.tiler.hasDmemQ() && tiler.hasDmemQ()) {
			dthrow(noDeviceBuffer());
		}

		//assert(lastMod == mod_host || lastMod == mod_synced);

		if(elements && !onlyDevice) {
			cherr( cudaMemcpy(res.elements, elements, size, cudaMemcpyHostToHost));
			HHCopied++;
			MemHhCopied +=size;
			if(checkDebug(debugCopy)) outln("host copied " << toShortString() << " to " << res.toShortString());
		}
	} else {
		DMatrix<T> d_res, d_M;
		tile0(d_M,lastMod == mod_host);
		res.tile0(d_res,false);
		copy(d_res, d_M, roff, coff);
		res.lastMod = mod_device;
	}
}


template<typename T> __host__ __device__ cudaError_t CuMatrix<T>::rowCopy(CuMatrix<T>& trg, int tRow, int sRow) const {
	IndexArray tary = trg.rowIndices(tRow);
	IndexArray sary = rowIndices(sRow);
/*
	outln("tRow " << tRow << " tary " << tary);
	outln("sRow " << sRow << " sary " << sary);
*/
	return copyIndexed(trg, tary, *this, sary);
}

/*
 * uses
 */
template<typename T> __host__ __device__ CuMatrix<T> CuMatrix<T>::derefIndices(const IndexArray& indices) const {
	CuMatrix<T> deref = CuMatrix<T>::zeros(indices.count, n);
	for(int i = 0; i < indices.count; i++) {
		rowCopy(deref, i, indices.indices[i]);
	}
	return deref;
}

template<typename T> cudaError_t CuMatrix<T>::copyIndexed(CuMatrix<T>& trg, const IndexArray& tary,  const CuMatrix<T>& src, const IndexArray& sary) {
	if(!src.tiler.hasDmemQ() || !trg.tiler.hasDmemQ()) {
		dthrow(noDeviceBuffer());
	}

	if(tary.count == sary.count && sary.count == 2) {
		// rowMajr to rowMaj
		//cudaMemcpy(targ.elements + tary.indices[0], src.elements + sary.indices[0], (tary.indices[1]-tary.indices[0])* sizeof(T), cudaMemcpyHostToHost);
		//flprintf("trg.d_elements  %p tary.indices[0] %d src.d_elements %p sary.indices[0] %d\n", trg.d_elements, tary.indices[0], src.elements , sary.indices[0] );
		cherr(cudaMemcpy(trg.tiler.currBuffer() + tary.indices[0], src.tiler.currBuffer() + sary.indices[0], (1 + tary.indices[1]-tary.indices[0])* sizeof(T),cudaMemcpyDeviceToDevice));
		DDCopied++;
		MemDdCopied += (tary.indices[1]-tary.indices[0])* sizeof(T);
		trg.invalidateHost();
		return cudaSuccess;
	} else if(tary.count == 2 && sary.count > 2) {
		uint start = tary.indices[0];
		uint tlen = tary.indices[1] - tary.indices[0];
		dassert(tlen == sary.count);
		for(uint i = 0; i < sary.count; i++) {
			trg.elements[start + i] = src.elements[ sary.indices[i]];
		}
	} else if(sary.count == 2 && tary.count > 2) {
		uint start = sary.indices[0];
		uint slen = sary.indices[1] - sary.indices[0];
		dassert(slen == tary.count);
		for(uint i = 0; i < tary.count; i++) {
			trg.elements[tary.indices[i]] = src.elements[start + i];
		}
	} else {
		outln("error, bad source indexarray " << sary.toString().c_str() << " or bad targ array " << tary.toString().c_str());
		return cudaErrorInvalidConfiguration;
	}
	trg.invalidateDevice();
	return cudaSuccess;
}

/*
 *  todo implement randSequence as a kernel on column or row matrix
 * 		whose re-arrangment when sorted (idx0 -> idx0sorted ...) is applied to the original sequence
 */
template<typename T> void CuMatrix<T>::shuffle(CuMatrix<T>* trg, CuMatrix<T>* leftovers, T fraction, vector<int>& vIndices ) const {

	outln("std::this_thread::get_id  " <<  std::this_thread::get_id());

	if( !(fraction >= 0. && fraction <= 1.)) {
		dthrow(outOfBounds());
	}
	if(!tiler.hasDmemQ()){
		dthrow(noDeviceBuffer());
	}
	if(lastMod == mod_host) {
		dthrow(notSyncedDev());
	}

	int rows;
	if(checkDebug(debugRefcount))
		if(trg)
			outln("refcounts trg: " << trg->elements << ": " << trg->hRefCount() << ", " << trg->dRefCount());

	if(integralTypeQ())  {
		rows = round(m *  fraction/100.);
	} else {
		rows = round(m * (double)fraction);
	}

	if(!trg->tiler.hasDmemQ()) {
		if(checkDebug(debugMem)) outln("!trg.tiler.hasDmemQ()");
		*trg = CuMatrix<T>::zeros(rows,n);
		if(checkDebug(debugRefcount))outln("refcounts post CuMatrix<T>:zero() trg: " << trg->elements << ": " << trg->hRefCount() << ", " << trg->dRefCount());
		if(checkDebug(debugMem))outln(trg->elements << " href " << trg->getMgr().refCount(trg->elements));
		if(checkDebug(debugMem))outln(trg->currBuffer() << " dref " << trg->getMgr().refCount(trg->currBuffer()));

		if(checkDebug(debugMem))outln("makazero " << trg->toShortString()) ;
		trg->syncBuffers();
		MemMgr<T>::checkValid(trg->tiler.currBuffer() ,"trg->tiler.currBuffer() ");
		//trg->getMgr().addHost(*trg);
		//trg->getMgr().addTiles(&(trg->tiler));
		if(checkDebug(debugMem))outln("trg.tiler.currBuffer() " << trg->tiler.currBuffer() );
	} else {
		if(checkDebug(debugMem))outln("trg has dbuff " << trg->currBuffer());
	}
	if(checkDebug(debugMem))
		outln("exoblock");

/*
	trg.m = rows;
	trg.n = trg.p = n;
	trg.size = trg.m * trg.p * sizeof(T);
	trg.tiler.allocTiles();
*/
	if(rows == m) {
		*leftovers = ZeroMatrix;
	} else {
		*leftovers =  CuMatrix<T>::zeros(m - rows,n);
		outln("leftovers " << leftovers->toShortString());
	}

	// re-use passed-in index buffer, to keep multple sample matrices in sync
	if(vIndices.size() == 0 ) {
		b_util::randSequence(vIndices, m, 0);
	} else if (vIndices.size() != m) {
		outln("shuffle passed a row index vector, but it was the wrong size (" << vIndices.size() << " <> " <<  m << ")");
		dthrow(badDimensions());
	}

	//if(checkDebug(debugFill))outln("vIndices\n" << b_util::pvec(vIndices));
	int* indices, *d_indices;
	int indexSize = m * sizeof(uint);

	cherr( cudaHostAlloc( (void**)&indices, indexSize, 0));
	b_util::toArray(indices, vIndices, 0, rows);
	cherr( cudaMalloc( (void**)&d_indices, indexSize));
	cherr(cudaMemcpy(d_indices, indices,indexSize, cudaMemcpyHostToDevice));
	HDCopied++;
	MemHdCopied += indexSize;
	DMatrix<T> s, t, l;
	assert(tiler.tileSize == tiler.m_size);

	if(checkDebug(debugMem))outln("shuffle pre  iler.tile0(s,false)");
	tile0(s,lastMod == mod_host);
	if(checkDebug(debugMem))outln("shuffle pre trg.tile0(t,false)");
	trg->tile0(t,false);
	outln("refcounts post trg.tile0(t,false) trg: "<< trg->elements << ": "  << trg->hRefCount() << ", " << trg->dRefCount());
	shuffleCopyRows(t,s, d_indices);

	trg->lastMod = mod_device;
	trg->syncBuffers();
	outln("refcounts post trg.syncBuffers() trg: "<< trg->elements << ": "  << trg->hRefCount() << ", " << trg->dRefCount());
	cherr(cudaDeviceSynchronize());

	if( !leftovers->zeroDimsQ()) {
		if(checkDebug(debugMem))outln("leftovers " << leftovers->toShortString());
		indexSize = leftovers->m * sizeof(uint);
		if(leftovers->m > rows) {
			// need a bigger index buffer
			outln("freeing indices " << indices);
			cherr( cudaFreeHost(indices));
			cherr( cudaHostAlloc( (void**)&indices, indexSize, 0));
			cherr( cudaFree( d_indices));
			cherr( cudaMalloc( (void**)&d_indices, indexSize));
		}
		b_util::toArray( indices, vIndices, rows, leftovers->m);
		cherr(cudaMemcpy(d_indices, indices, indexSize, cudaMemcpyHostToDevice));
		HDCopied++;
		MemHdCopied += indexSize;
		leftovers->tile0(l,false);
		shuffleCopyRows(l,s, d_indices);
		leftovers->lastMod = mod_device;
	}

	cherr(cudaDeviceSynchronize());
	outln("freeing indices " << indices);
	cherr(cudaFreeHost(indices));
	cherr( cudaFree( d_indices));

}

template<typename T> void CuMatrix<T>::toDevice(int dev) {
	getMgr().migrate(dev, *this);
}

template<typename T> __host__ __device__ void CuMatrix<T>::zero() {
	if(!tiler.hasDmemQ()) {
		setLastError(noDeviceBufferEx);
	}
	if(!lastMod == mod_host) {
		setLastError(notSyncedDevEx);
	}
#ifndef __CUDA_ARCH__
	assert(tiler.tileSize == tiler.m_size);
	//checkCudaError(cudaMemset(tiler.currBuffer(), 0, size));
#else
	memset(tiler.currBuffer(), 0, size);
#endif
	lastMod = mod_device;
}

/*
 * this and other must be col tiled
 * 	(row tiles converted to col tiles)
 * ret coltiles is sum,   with up to 2^
 *
 */
template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::rightConcatenate(
		 const CuMatrix<T>& other, cudaStream_t stream ) const {
	if(other.zeroDimsQ()) {
		return *this;
	}
	if(zeroDimsQ()) {
		return other;
	}
	if(other.m != m) {
		flprintf("other.m %u != m %u (other.n %u, n %u)\n", other.m, m,other.n, n);
		setLastError(matricesOfIncompatibleShapeEx);
	}
	//outln("this " << toShortString());
	//outln("other " << other.toShortString());

	uint newCols = n + other.n;
	if (colMajor){
		setLastError (  notImplementedEx);
		return CuMatrix();
	} else {
		DMatrix<T> d_A, d_B, d_Res;
		if(tiler.tileSize == tiler.m_size &&
				other.tiler.tileSize == other.tiler.m_size &&
				newCols * m * sizeof(T) <= ExecCaps::currCaps()->maxReasonable() ) {
			CuMatrix<T> ret(m, newCols,true, true);
			if(checkDebug(debugTiler)) flprintf("ret.tiler %p ret.tiler.getTileCount %d ret.tiler.m_m %u ret.tiler.m_n %u\n",
					&ret.tiler, ret.tiler.getTileCount(), ret.tiler.m_m, ret.tiler.m_n);
			if(! gpuReadyQ() ) {
				prlocf("this not synced\n");
				setLastError(notSyncedEx);
			}
			if(! other.gpuReadyQ() ) {
				prlocf("other not synced\n");
				setLastError(notSyncedEx);
			}
			tile0(d_A,lastMod == mod_host,stream);
			other.tile0(d_B,other.lastMod == mod_host,stream);
			ret.tile0(d_Res,false,stream);
			rightConcatenateL(d_Res, d_A,d_B,stream);
			ret.invalidateHost();
			cherr(cudaPeekAtLastError());
			return ret;
		} else {
			if(checkDebug(debugTiler)) {
				flprintf("m %u newCols %u\n" ,m, newCols);
			}
			CuMatrix<T> ret(m, newCols,newCols, Tiler<T>::gpuMaskFromCurrGpu(), true, true);
			if(checkDebug(debugTiler)) {
				flprintf("ret.size %lu\n" , ret.size);
				ret.printShortString( __func__ );
			}
			// all in hmem
			flprintf("rightConcat in hmem only;  ret.elements %p, ret.p %u, elements %p, p %u\n",ret.elements, ret.p, elements, p);
			cherr(cudaMemcpy2D(ret.elements, ret.p * sizeof(T), elements, p* sizeof(T), n*sizeof(T), m, cudaMemcpyHostToHost));
			if(checkDebug(debugTiler)) flprintf("after 1st memcpy2d ret.tiler %p ret.tiler.getTileCount %d ret.tiler.m_m %u ret.tiler.m_n %u\n",
					&ret.tiler, ret.tiler.getTileCount(), ret.tiler.m_m, ret.tiler.m_n);
			cherr(cudaMemcpy2D(ret.elements + n, ret.p* sizeof(T), other.elements, other.p* sizeof(T), other.n*sizeof(T), other.m, cudaMemcpyHostToHost));
			if(checkDebug(debugTiler)) flprintf("after 2nd memcpy2d ret.tiler %p ret.tiler.getTileCount %d ret.tiler.m_m %u ret.tiler.m_n %u\n", &ret.tiler, ret.tiler.getTileCount(), ret.tiler.m_m, ret.tiler.m_n);
			ret.invalidateDevice();
			return ret;
		}
	}
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::bottomConcatenate(
		 const CuMatrix<T>& other, cudaStream_t stream ) const {
	if(other.zeroDimsQ()) {
		return *this;
	}
	if(zeroDimsQ()) {
		return other;
	}
	if (other.n != n) {
#ifndef __CUDA_ARCH__
		dthrow (  matricesOfIncompatibleShape());
#else
		setLastError(matricesOfIncompatibleShapeEx);
#endif
	}
	uint newRows = m + other.m;
	if (colMajor) {
#ifndef __CUDA_ARCH__
		dthrow ( notImplemented() );
#else
		setLastError(notImplementedEx);
#endif
	} else {
		DMatrix<T> d_A, d_B, d_Res;
		if(tiler.tileSize == tiler.m_size &&
				other.tiler.tileSize == other.tiler.m_size &&
				newRows * n * sizeof(T) <= ExecCaps::currCaps()->maxReasonable() )  {
			CuMatrix<T> ret(newRows, n,true,true);
			tile0(d_A, lastMod == mod_host);
			other.tile0(d_B, other.lastMod == mod_host);
			ret.tile0(d_Res,false);
			bottomConcatenateL(d_Res, d_A,d_B,stream);
			ret.invalidateHost();
			return ret;
		}else {
			// all in hmem
			CuMatrix<T> ret(newRows, n,n, Tiler<T>::gpuMaskFromCurrGpu(), true, true);
			//if(checkDebug(debugTiler))usedDevMem();
			if(checkDebug(debugTiler))flprintf("HMEM ret.elements %p, ret.p %u, elements %p, p %u\n",ret.elements, ret.p, elements, p);
			cherr(cudaMemcpy2D(ret.elements, ret.p * sizeof(T), elements, p* sizeof(T), n*sizeof(T), m, cudaMemcpyHostToHost));
			if(checkDebug(debugTiler)) flprintf("after 1st memcpy2d ret.tiler %p ret.tiler.getTileCount %d ret.tiler.m_m %u ret.tiler.m_n %u\n", &ret.tiler, ret.tiler.getTileCount(), ret.tiler.m_m, ret.tiler.m_n);
			cherr(cudaMemcpy2D(ret.elements + m * p, ret.p* sizeof(T), other.elements, other.p* sizeof(T), other.n*sizeof(T), other.m, cudaMemcpyHostToHost));
			if(checkDebug(debugTiler)) flprintf("after 2nd memcpy2d ret.tiler %p ret.tiler.getTileCount() %d ret.tiler.m_m %u ret.tiler.m_n %u\n", &ret.tiler, ret.tiler.getTileCount(), ret.tiler.m_m, ret.tiler.m_n);
			ret.invalidateDevice();
			return ret;
		}
	}
	return CuMatrix();
}

template <typename T> __global__ void setKernel(T* elements, int p, int row, int col, T val) {
	elements[row * p + col] = val;
}


template<typename T> __host__ __device__ void CuMatrix<T>::set(int r, int c, T val) {
	if (r >= m || c >= n)
		setLastError(outOfBoundsEx);
	if(!tiler.hasDmemQ()) {
		setLastError(noDeviceBufferEx);
	}
	uint idx = colMajor ? c * p + r : r*p + c;
	set(idx, val);
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::set(int l, T val) {
	if (l >= size / sizeof(T))
		setLastError(outOfBoundsEx);
	if(!tiler.hasDmemQ()) {
		setLastError(noDeviceBufferEx);
	}
	assert(tiler.tileSize == tiler.m_size);
	::setL<T>(tiler.currBuffer(), m, n, p, l, val);
	invalidateHost();
}

template<typename T> __host__ __device__ T CuMatrix<T>::get(long l) const {
	if (( n == p && l >= size / sizeof(T)) ||
			( n != p && l >= m * p)) {
		setLastError(outOfBoundsEx);
	}
	if(!tiler.hasDmemQ()) {
		setLastError(noDeviceBufferEx);
	}
//	if(checkDebug(debugMem)) flprintf("lastMod %d\n", lastMod);
#ifndef __CUDA_ARCH__
	if(lastMod == mod_synced || lastMod == mod_host) {
		return elements[l];
	}
	T res;
	assert(tiler.tileSize == tiler.m_size);
	cherr(cudaMemcpy(&res, tiler.currBuffer()+ l, sizeof(T), cudaMemcpyDeviceToHost));
	if(checkDebug(debugCopyDh))outln("debugCopyDh " << "CuMatrix<T>::get");
	DHCopied++;
	MemDhCopied += sizeof(T);
	return res;
#else
	if(lastMod == mod_synced || lastMod == mod_device) {
		return tiler.currBuffer()[l];
	} else {
		setLastError(notSyncedDevEx);
		return -1;
	}
#endif
}

template<typename T> __host__ __device__ T CuMatrix<T>::get(int r, int c) const {
	if (r >= CuMatrix<T>::m || c >= CuMatrix<T>::n)
		setLastError(outOfBoundsEx);
	return get(r * p + c);
}

template<typename T> __host__ __device__ T CuMatrix<T>::getH(int r, int c) const {
	if (r >= CuMatrix<T>::m || c >= CuMatrix<T>::n)
		setLastError(outOfBoundsEx);
	return elements[r * p + c];
}

template <typename T> void CuMatrix<T>::linterp(CuMatrix<T>& result, const CuMatrix<T>& src, const CuMatrix<T>& dest, T factor) {
	result = src + factor * (dest-src);
}


#include "CuMatrixInster.cu"

