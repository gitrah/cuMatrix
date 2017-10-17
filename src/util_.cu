#include "util.h"
#include "caps.h"
#include <string>
#include <sstream>
#include "DMatrix.h"
#include "CuMatrix.h"
#include <float.h>
#include <limits>
#include "Maths.h"
#include "Kernels.h"

using std::numeric_limits;

template <typename T> __host__ __device__ void printColoArray(const T* array, int n, int direction) {
#ifndef __CUDA_ARCH__
	printf("caller %s\n", b_util::caller().c_str());
#endif
	flprintf("array %p[0::%d] ", array, n);
	for(int i =0; i < n; i++) {
		printf("%f", (float) array[i]);
		if(i < n -1) printf(", ");
	}
	printf("\n");
}
template __host__ __device__ void printColoArray<float>(const float*,int,int);
template __host__ __device__ void printColoArray<double>(const double*,int,int);
template __host__ __device__ void printColoArray<int>(const int*,int,int);
template __host__ __device__ void printColoArray<uint>(const uint*,int,int);
template __host__ __device__ void printColoArray<long>(const long*, int,int);
template __host__ __device__ void printColoArray<ulong>(const ulong*, int,int);



template <typename T> __host__ __device__ void prtColoArrayDiag(
		const T* array,const char*msg,int line,  int pitch,int n, int direction, T notEq) {
	flprintf("%s:%d h arraydiag %p p:%d[0::%d] notEq %f\n", msg, line, array, pitch, n, notEq);
	int neqCnt = 0;
	int idxFb= -1;
	const T* firstBad = nullptr;
	for(int i =0; i < n; i++) {
		if(!notEq || array[i * (pitch + 1)] != notEq) {
			printf("%p + %d (%p) = %f != %f", array, direction*i, array + direction * i * (pitch + 1), (float) array[i * (pitch + 1)], notEq);
			if(i < n -1) printf(", ");
			if(notEq) { idxFb = i; firstBad = array  + i * (pitch + 1); neqCnt++; }
		}
	}
	if(!neqCnt)
		flprintf("\nfound none != %f\n",notEq);
	else {
		flprintf("found %d unexpected values starting at %p idx %d\n", neqCnt, firstBad,idxFb);
	}
	assert(neqCnt == 0);

}
template __host__ __device__ void prtColoArrayDiag<float>(const float*,const char*msg,int line,int,int,int,float);
template __host__ __device__ void prtColoArrayDiag<double>(const double*,const char*msg,int line,int,int,int,double);
template __host__ __device__ void prtColoArrayDiag<int>(const int*,const char*msg,int line,int,int,int,int);
template __host__ __device__ void prtColoArrayDiag<uint>(const uint*,const char*msg,int line,int,int,int,uint);
template __host__ __device__ void prtColoArrayDiag<long>(const long*,const char*msg,int line,int,int,int,long);
template __host__ __device__ void prtColoArrayDiag<ulong>(const ulong*,const char*msg,int line,int,int,int,ulong);

template <typename T> __host__ __device__ void cntColoArrayDiag(
		const T* array,const char*msg,int line,  int pitch,int n, int direction, T test) {
	flprintf("%s:%d h arraydiag %p p:%d[0::%d] test %f\n", msg, line, array, pitch, n, test);
	int neqCnt = 0;
	int idxNeq= -1;
	int eqCnt = 0;
	int idxEq= -1;
	const T* firstNeq = nullptr,* firstEq = nullptr;
	for(int i =0; i < n; i++) {
		if(!test || array[i * (pitch + 1)] != test) {
			//printf("%p + %d (%p) = %f != %f", array, direction*i, array + direction * i * (pitch + 1), (float) array[i * (pitch + 1)], notEq);
			if(test && firstNeq == nullptr) { idxNeq = i; firstNeq= array  + i * (pitch + 1);  }
			neqCnt++;
		}else {
			if(test&& firstEq == nullptr) { idxEq = i; firstEq = array  + i * (pitch + 1);  }
			eqCnt++;
		}
	}

	flprintf("\nfound %d neq %f, %d eq out of %d; first neq @ %p idx %d first eq %p idx %d\n", neqCnt, test, eqCnt, n, firstNeq,idxNeq,firstEq,idxEq);
}

template __host__ __device__ void cntColoArrayDiag<float>(const float*,const char*msg,int line,int,int,int,float);
template __host__ __device__ void cntColoArrayDiag<double>(const double*,const char*msg,int line,int,int,int,double);
template __host__ __device__ void cntColoArrayDiag<int>(const int*,const char*msg,int line,int,int,int,int);
template __host__ __device__ void cntColoArrayDiag<uint>(const uint*,const char*msg,int line,int,int,int,uint);
template __host__ __device__ void cntColoArrayDiag<long>(const long*,const char*msg,int line,int,int,int,long);
template __host__ __device__ void cntColoArrayDiag<ulong>(const ulong*,const char*msg,int line,int,int,int,ulong);

template <typename T> __host__ __device__ void prtColoArrayInterval(const T* array,
		const char* msg, long n, int sampleElemCount, int sampleCount) {
	int step = n / sampleCount;
	printf("%s array %p n %ld selems %d samCnt %d step %d\n", msg, array, n, sampleElemCount, sampleCount, step);
	printf("array %p[0::%d]\n", array, n);
	for(int s = 0; s < n -sampleElemCount; s += step) {
		printf("  %d::%d --> ", s, s+ sampleElemCount);
		for(int i =0; i < sampleElemCount; i++) {
			printf("  %f", (float) array[s + i ]);
			if(i < sampleElemCount -1) printf(", ");
		}
		printf("\n");
	}
	printf("\n");
}
template __host__ __device__ void prtColoArrayInterval<float>(const float*, const char*,long,int,int);
template __host__ __device__ void prtColoArrayInterval<double>(const double*, const char*,long,int,int);
template __host__ __device__ void prtColoArrayInterval<int>(const int*, const char*,long,int,int);
template __host__ __device__ void prtColoArrayInterval<uint>(const uint*, const char*,long,int,int);
template __host__ __device__ void prtColoArrayInterval<long>(const long*, const char*, long,int,int);
template __host__ __device__ void prtColoArrayInterval<ulong>(const ulong*, const char*, long,int,int);
__host__ void b_util::warmupL() {
	outln("warminup");
	warmup<<<1,1>>>();
	outln("blokin");
	checkCudaError(cudaDeviceSynchronize());
}

__host__ __device__ const char* b_util::tileDir(TileDirection tileD) {
	switch(tileD) {
	case tdNeither:
		return "tdNeither";
	case tdRows:
		return "tdRows";
	case tdCols:
		return "tdCols";
	case tdBoth:
		return "tdBoth";
	default:
		return "???";
	}
}


__host__ __device__ bool b_util::isPow2(uint x) {
    return ((x&(x-1))==0);
}

__host__ __device__ int b_util::threadIdx1D() {
#ifdef __CUDA_ARCH__
	return blockIdx.x * blockDim.x * 2 + threadIdx.x;
#else
	return false;
#endif
}

__host__ __device__ int b_util::threadIdx1Dblock(uint blocksize) {
#ifdef __CUDA_ARCH__
	return blockIdx.x * blocksize * 2 + threadIdx.x;
#else
	return false;
#endif
}



__host__ __device__  const char* tdStr(TileDirection td) {
	switch (td ){
	case tdNeither:
		return "tdNeither";
	case tdRows:
		return "tdRows";
	case tdCols:
		return "tdCols";
	}
	return "unknown";
}


__host__ __device__ bool b_util::adjustExpectations(dim3& grid, dim3& block, const cudaFuncAttributes& atts) {
    uint curBlk = block.x * block.y;
    flprintf("curBlk %d\n",curBlk);
    float factor =  1.0 * curBlk / atts.maxThreadsPerBlock;
    flprintf("factor %f\n",factor);
    if(factor > 1) {
    	flprintf("was block(%d,%d)\n", block.x, block.y);
    	flprintf("factor %d\n", factor);
    	if(block.x  > block.y) {
    		block.x /= factor;
    		grid.x *= factor;
    	} else {
    		block.y /= factor;
    		grid.y *= factor;
    	}
    	flprintf("now block(%d,%d)\n", block.x, block.y);
    	return true;
    }
    return false;
}


__host__ __device__ bool b_util::onGpuQ() {
#ifdef __CUDA_ARCH__
	return true;
#else
	return false;
#endif
}

__global__ void devFree(void * mem) {
#ifdef CuMatrix_Enable_Cdp
	FirstThread {
		cherr(cudaFree(mem));
	}
#endif
}
__host__ void  b_util::freeOnDevice(void * mem) {
	devFree<<<1,1>>>(mem);
}

__device__ __host__ uint b_util::nextPowerOf2(uint x) {
	if (x < 2) {
		return 2;
	}
	x = maskShifts(--x);
	return ++x;
}

__device__ __host__ uint b_util::prevPowerOf2(uint x) {
    x = maskShifts(x);
    return x - (x >> 1);
}

template<> __host__ __device__ float util<float>::epsilon() {
	return 1e-6;
}
template<> __host__ __device__ double util<double>::epsilon() {
	return 1e-10;
}
template<> __host__ __device__ long util<long>::epsilon() {
	return 0;
}
template<> __host__ __device__ ulong util<ulong>::epsilon() {
	return 0;
}
template<> __host__ __device__ uint util<uint>::epsilon() {
	return 0;
}
template<> __host__ __device__ int util<int>::epsilon() {
	return 0;
}

template<typename T> __host__ __device__ T util<T>::minValue() {
#ifndef __CUDA_ARCH__
	return numeric_limits<T>::min();
#else
	setLastError(notImplementedEx);
	return 0;
#endif
}
template <> __host__  __device__ uint util<uint>::minValue() {
	return 0;
}

template<typename T>  __host__  __device__ bool util<T>::almostEquals(T t1, T t2, T epsilon) {
	if(checkDebug(debugVerbose))flprintf("t2 %f - t1 %f < epsilon %f ::abs(t2 - t1) %f (%d)\n",
			(double)t2, (double) t1, (double) epsilon,(double) ::fabs(t2 - t1),  ::fabs(t2 - t1) < epsilon);
	return ::fabs(t2 - t1) < epsilon;
}
template   __host__  __device__ bool util<long>::almostEquals(long, long, long);
template   __host__  __device__ bool util<unsigned long>::almostEquals(unsigned long, unsigned long, unsigned long);
template   __host__  __device__ bool util<double>::almostEquals(double, double, double);
template   __host__  __device__ bool util<int>::almostEquals(int, int, int);
template   __host__  __device__ bool util<float>::almostEquals(float, float, float);
template   __host__  __device__ bool util<unsigned int>::almostEquals(unsigned int, unsigned int, unsigned int);

template<typename T>  __host__  __device__ bool util<T>::almostEqualsNormalized(T t1, T t2, T epsilon) {
	if(checkDebug(debugVerbose))flprintf("t2 %f - t1 %f < epsilon %f ::abs(t2 - t1) %f (%d)\n",
			(double)t2, (double) t1, (double) epsilon,(double) ::fabs(t2 - t1),  ::fabs(t2 - t1) < epsilon);
	return (::fabs(t2 - t1))/t1 < epsilon;
}

template   __host__  __device__ bool util<long>::almostEqualsNormalized(long, long, long);
template   __host__  __device__ bool util<unsigned long>::almostEqualsNormalized(unsigned long, unsigned long, unsigned long);
template   __host__  __device__ bool util<double>::almostEqualsNormalized(double, double, double);
template   __host__  __device__ bool util<int>::almostEqualsNormalized(int, int, int);
template   __host__  __device__ bool util<float>::almostEqualsNormalized(float, float, float);
template   __host__  __device__ bool util<unsigned int>::almostEqualsNormalized(unsigned int, unsigned int, unsigned int);


template<typename T>__global__ void vectorAddPitch(T *c, const T *a, const T *b, int n, int p) {
	uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
    	int tidx = idx * p;
        tidx[c] = tidx[a] + tidx[b]; // fun with [] syntax
    }
}


template<typename T> __host__ float util<T>::vAddGflops(int device){
	int orgDev;
	cherr(cudaPeekAtLastError());
	cherr(cudaGetDevice(&orgDev));
	if(orgDev != device) {
		ExecCaps_visitDevice(device);
	}
	flprintf("util<T>::vAddGflops(device = %d, %s) set device\n", device, gpuNames[device].c_str());
	outln("checking fer dev " << device);
	//usedDevMem();
	int n = 1000000;
	//outln("before mc");
	//usedDevMem();
	CuMatrix<T> mc = CuMatrix<T>::ones(n,1);
	outln("after mc " << mc.toShortString());
	T mcsum = mc.sum();
	outln("mcsum " << mcsum);
	assert(mcsum == (T)n);

	//outln("checking fer dev " << device);
	//usedDevMem();
	//outln("made mc\n " << mc.syncBuffers());
	//outln("after mc syncbuffes \n ");
	//printColoArrayInterval(mc.elements, n, 10, 40);
	//usedDevMem();

	CuMatrix<T> m2 = CuMatrix<T>::fill( n,1,(T)2);
	//outln("made m2\n " << m2.syncBuffers());
	//usedDevMem();
	CuMatrix<T> m3 = CuMatrix<T>::zeros( n,1);
	outln("made mc " << mc.toShortString() << ", " << m2.toShortString() << ", m3 " << m3.toShortString());
	//m3.syncBuffers();

	DMatrix<T> dc = mc.asDmatrix();
	DMatrix<T> d2 = m2.asDmatrix();
	DMatrix<T> d3 = m3.asDmatrix();
	outln("after d1-d3");
	//usedDevMem();
	//flprintf("util<T>::vAddGflops dc.el %p d2.el %p d3.el %p\n", dc.elements, d2.elements, d3.elements);
	uint blockSize = 1024;
	CuTimer timer;
	timer.start();
	cherr(cudaPeekAtLastError());
	MemMgr<T>::checkValid(  dc.elements);
	MemMgr<T>::checkValid(  d2.elements);
	MemMgr<T>::checkValid(  d3.elements);
	outln("all valid d1-d3");

	vectorAddPitch<T><<<DIV_UP(n,blockSize), blockSize>>>(d3.elements, dc.elements, d2.elements, n, mc._tileP);
	cherr(cudaDeviceSynchronize());
	outln("cudaDeviceSynchronize after vectorAddPitch");

	m3.invalidateHost();
	float addTimeMs = timer.stop();
	timer.start();
	m3.syncBuffers();
	//outln("m3 " << m3 );
	T m3sum = m3.sum();
	outln("m3sum " << m3sum );
	assert(m3sum == 3 * n);
	float memTimeMs = timer.stop();

//	b_util::usedDmem(1);
	//printColoArrayInterval(m3.elements, n, 10, 40);
	//flprintf("n %u adds took exeTimeMs %f millis (%f s)\n", n, exeTimeMs, exeTimeMs/Kilo);
	addTimeMs /= Kilo;
	flprintf("n/addTimeMs %f\n", n/addTimeMs);
	float nExe = n/addTimeMs;
	flprintf("n/extTimeS %f\n", nExe);
	flprintf("nExe/Giga %f\n", nExe/Giga);

	// one add per result element, so n adds per invocation
	if(orgDev != device) {
		ExecCaps_restoreDevice(orgDev);
	}
	return (n/(addTimeMs/Kilo) )/ Giga;
}
template __host__  float util<float>::vAddGflops(int device);
template __host__  float util<double>::vAddGflops(int device);
template __host__  float util<ulong>::vAddGflops(int device);


template<> __host__ __device__ float util<float>::minValue() {
	return FLT_MIN;
}
template<> __host__ __device__ double util<double>::minValue() {
	return DBL_MIN;
}
template<> __host__ __device__ long util<long>::minValue() {
	return 0;
}
template<> __host__ __device__ ulong util<ulong>::minValue() {
	return 0;
}
template<> __host__ __device__ int util<int>::minValue() {
	return INT_MIN;
}

template<typename T> __host__ __device__ T util<T>::maxValue() {
#ifndef __CUDA_ARCH__
	return numeric_limits<T>::max();
#else
	setLastError(notImplementedEx);
	return 0;
#endif
}
template<> __host__ __device__ float util<float>::maxValue() {
	return FLT_MAX;
}
template<> __host__ __device__ double util<double>::maxValue() {

	return DBL_MAX;
}
template<> __host__ __device__ long util<long>::maxValue() {
	return LONG_MAX;
}
template<> __host__ __device__ ulong util<ulong>::maxValue() {
	return 0xffffffff;
}
template<> __host__ __device__ int util<int>::maxValue() {
	return INT_MAX;
}
template<> __host__ __device__ uint util<uint>::maxValue() {
	return 0xFFFF;
}


/*
namespace mods {
static const char * host = "host";
static const char * device = "device";
static const char * synced = "synced";
static const char * neither = "neither";
}

*/
#define mods_host "host"
#define mods_device "device"
#define mods_synced "synced"
#define mods_neither "neither"

__host__ __device__ const char * b_util::modStr(Modification lastMod) {
	switch (lastMod) {
	case mod_host:
		return "lstmd: " mods_host;
	case mod_device:
		return "lstmd: " mods_device;
	case mod_synced:
		return "lstmd: " mods_synced;
	case mod_neither:
		return "lstmd: " mods_neither;
	default:
		return "????";
	}
}

int b_util::kernelOccupancy( void* kernel, int* maxBlocks, int blockSize) {
	ExecCaps* curr = ExecCaps::currCaps();
	//cudaOccupancyMaxActiveBlocksPerMultiprocessor(maxBlocks, kernel, blockSize,0);
	int device = ExecCaps::currDev();
	cudaDeviceProp prop;
	int numBlocks;
	cherr(cudaGetDeviceProperties(&prop, device))
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel,
			blockSize, 0);
	int activeWarps = numBlocks * blockSize / prop.warpSize;
	int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
	//int activeWarps = *maxBlocks * blockSize / curr->deviceProp.warpSize;
	//int maxWarps = curr->deviceProp.maxThreadsPerMultiProcessor /curr->deviceProp.warpSize;;
	outln("Occupancy " << (double) activeWarps/maxWarps * 100 << "%");
	return 0;
}


template<typename T> __host__ __device__ void b_util::pPtrAtts(const T * ptr) {

#ifndef __CUDA_ARCH__
	int ptrDev = b_util::getDevice((void*)ptr);
	int orgDev = ExecCaps::currDev();
	outln("b_util::pPtrAtts( " << ptr << ")");
	cudaPointerAttributes ptrAtts;
	if(ptrDev != orgDev) {
		ExecCaps_visitDevice(ptrDev);
	}
	chsuckor(cudaPointerGetAttributes(&ptrAtts, ptr), cudaErrorInvalidValue);
	if(ptrDev != orgDev) {
		ExecCaps_restoreDevice(orgDev);
	}

	stringstream ss;
	flprintf("raw %p %s %p, %s %p gpu %d, managed %s, type %s\n",
			ptr,
			ptrAtts.devicePointer != nullptr ?  "dev: " : "",
			ptrAtts.devicePointer != nullptr ? ptrAtts.devicePointer : 0,
			ptrAtts.hostPointer != nullptr ? "host: " : "",
			ptrAtts.hostPointer != nullptr ? ptrAtts.hostPointer : 0,
			ptrAtts.device,
			tOrF(ptrAtts.isManaged),
			ptrAtts.memoryType == cudaMemoryTypeHost  ? "host"
					:  ptrAtts.memoryType == cudaMemoryTypeDevice ? "device" : "unknown");
#else
    flprintf("b_util::pPtrAtts(..) can't call cudaPointerGetAttributes() from device, ptr %p\n", ptr );
#endif
}
template __host__ __device__ void b_util::pPtrAtts<UnaryOpIndexF<float,0> const>(UnaryOpIndexF<float,0> const*);
template __host__ __device__ void b_util::pPtrAtts<UnaryOpIndexF<double,0> const>(UnaryOpIndexF<double,0> const*);
template __host__ __device__ void b_util::pPtrAtts<UnaryOpIndexF<unsigned long,0> const>(UnaryOpIndexF<unsigned long,0> const*);
template __host__ __device__ void b_util::pPtrAtts<float (*)(float) >(float (* const *)(float));
template __host__ __device__ void b_util::pPtrAtts<double (*)(double)>(double (* const *)(double));
template __host__ __device__ void b_util::pPtrAtts<unsigned long (*)(unsigned long)>(unsigned long (* const *)(unsigned long));
template __host__ __device__ void b_util::pPtrAtts<void>( const void*);
template __host__ __device__ void b_util::pPtrAtts<constFiller<float> >( const constFiller<float>*);
template __host__ __device__ void b_util::pPtrAtts<constFiller<double> >( const constFiller<double>*);
template __host__ __device__ void b_util::pPtrAtts<constFiller<unsigned long> >( const constFiller<unsigned long>*);
template __host__ __device__  void b_util::pPtrAtts<float const>(float const*);
template __host__ __device__  void b_util::pPtrAtts<ulong const>(ulong const*);
template __host__ __device__  void b_util::pPtrAtts<double const>(double const*);

template __host__ __device__  void b_util::pPtrAtts<float>( const float*);
template __host__ __device__  void b_util::pPtrAtts<double>( const double*);
template __host__ __device__  void b_util::pPtrAtts<int>( const int*);
template __host__ __device__  void b_util::pPtrAtts<uint>( const uint*);
template __host__ __device__  void b_util::pPtrAtts<long>( const long*);
template __host__ __device__  void b_util::pPtrAtts<ulong>( const ulong*);

__host__ __device__ void b_util::pFuncPtrAtts( const void * ptr) {
//#ifndef __CUDA_ARCH__
#ifdef CuMatrix_Enable_Cdp
	struct cudaFuncAttributes fa;
	cherr(cudaFuncGetAttributes(&fa, ptr));
    flprintf("\n\tfa.binaryVersion %d\n\tfa.cacheModeCA %d\n\tfa.constSizeBytes %d\n\tfa.localSizeBytes %d\n\t,fa.maxThreadsPerBlock %d\n\tfa.numRegs %d\n\tptxVersion %d\n\tsharedSizeBytes %d\n",
    		fa.binaryVersion,fa.cacheModeCA,fa.constSizeBytes,fa.localSizeBytes,fa.maxThreadsPerBlock,fa.numRegs,fa.ptxVersion,fa.sharedSizeBytes);
#endif
    /*
#else
    flprintf("can't call cudaPointerGetAttributes() from device, ptr %p\n", ptr );
#endif
*/
}

template<typename T> __host__ __device__ enum cudaMemcpyKind  b_util::copyKind( T * dst,  T const * const src ) {
#ifndef __CUDA_ARCH__
	int dstDevice = b_util::getDevice((void*)dst);
	int srcDevice = b_util::getDevice((void*)src);

	int orgDev = ExecCaps::currDev();
	outln("b_util::copyKind( dst " << dst << ", src " << src << ") orgDev " << orgDev << ", dstDev " <<  dstDevice << ",  srcDev " << srcDevice);
	cudaPointerAttributes padst, pasrc;
	if(dstDevice != orgDev) {
		outln("b_util::copyKind() visiting dstDevice " << dstDevice);
		ExecCaps_visitDevice(dstDevice);
	}
	chsuckor(cudaPointerGetAttributes(&padst, dst), cudaErrorInvalidValue);
	if(dstDevice != orgDev) {
		outln("b_util::copyKind() restoring orgDev " << orgDev);
		ExecCaps_restoreDevice(orgDev);
	}
	if(srcDevice != orgDev) {
		outln("b_util::copyKind() visiting srcDevice " << srcDevice);
		ExecCaps_visitDevice(srcDevice);
	}
	chsuckor(cudaPointerGetAttributes(&pasrc, src), cudaErrorInvalidValue);
	if(srcDevice != orgDev) {
		outln("b_util::copyKind() restoring orgDev " << orgDev);
		ExecCaps_restoreDevice(orgDev);
	}

	outln("b_util::copyKind() pasrc.memoryType " << pasrc.memoryType << "; padst.memoryType " << padst.memoryType);

	if(pasrc.memoryType == cudaMemoryTypeHost) {
		if(padst.memoryType == cudaMemoryTypeHost)
			return cudaMemcpyHostToHost;
		else
			return cudaMemcpyHostToDevice;
	}else {
		if(padst.memoryType == cudaMemoryTypeHost)
			return cudaMemcpyDeviceToHost;
		else
			return cudaMemcpyDeviceToDevice;
	}

#else
    flprintf("b_util<T>::copyType(..) can't call cudaPointerGetAttributes() from device, dst %p\n", dst );
    return cudaMemcpyDefault; // kind inferred from pointer; for managed buffers
#endif
}
template __host__ __device__ enum  cudaMemcpyKind b_util::copyKind<float>(float*, float const*);
template __host__ __device__ enum  cudaMemcpyKind b_util::copyKind<double>(double*, double const*);
template __host__ __device__ enum  cudaMemcpyKind b_util::copyKind<unsigned long>(unsigned long*, unsigned long const*);

__host__ __device__ int b_util::maxBlockThreads(const void * ptr) {
	struct cudaFuncAttributes fa;
	cherr(cudaFuncGetAttributes(&fa, ptr));
	return fa.maxThreadsPerBlock;
}

__host__ CUDART_DEVICE bool b_util::validLaunchQ( const void* pKernel, dim3 grid, dim3 block) {
	cudaFuncAttributes fatts;
	cherr(cudaFuncGetAttributes(&fatts, pKernel));
	uint blockThreads = block.x * block.y * block.z;
	flprintf("blockThreads %u\n", blockThreads);
	if(blockThreads > fatts.maxThreadsPerBlock) {
		flprintf("kernel @ %p unlaunchable: blockThreads %u > fatts.maxThreadsPerBlock %u\n", pKernel, blockThreads, fatts.maxThreadsPerBlock);
		return false;
	}
	ExecCaps* pCaps = ExecCaps::currCaps();
	uint gridVol = grid.x * grid.y * grid.z;
	flprintf("gridVol %u\n", gridVol);
	uint blockRegs = fatts.numRegs * blockThreads;
	flprintf("blockRegs %u\n", blockRegs);
	if(blockRegs > pCaps->regsPerBlock) {
		flprintf("kernel @ %p unlaunchable: blockRegs %u > pCaps->regsPerBlock %u, gridVol %u\n", pKernel, blockRegs, pCaps->regsPerBlock, gridVol);
		return false;
	}
	if(fatts.sharedSizeBytes > pCaps->memSharedPerBlock) {
		flprintf("kernel @ %p unlaunchable: fatts.sharedSizeBytes %u > pCaps->memSharedPerBlock %u\n", pKernel, fatts.sharedSizeBytes, pCaps->memSharedPerBlock);
		return false;
	}

	return true;
}


__host__ CUDART_DEVICE void b_util::validLaunch( dim3 &  block, const void* pKernel) {
	cudaFuncAttributes fatts;
	uint minF = MIN( block.x, MIN(block.y, block.z));
	uint maxF = MAX( block.x, MAX(block.y, block.z));
	uint blocks = block.x * block.y * block.z;
	uint otherF =  blocks / (minF * maxF);
	cudaFuncGetAttributes(&fatts, pKernel);

	if(fatts.maxThreadsPerBlock < blocks) {
		double factor = 1.0 * blocks/ fatts.maxThreadsPerBlock;
		if(factor < maxF) {
			maxF = (uint) maxF/ factor;
		}
		block.x = maxF; block.y = otherF; block.z = minF;
	}
}



__host__ __device__ void b_util::prd3(const dim3& d3,const char* msg) {
	if(msg)
		printf("%s (%u,%u,%u)\n", msg, d3.x, d3.y, d3.z);
	else
		printf("(%u,%u,%u)\n", d3.x, d3.y, d3.z);
}

#ifdef CuMatrix_NVML
__host__ void b_util::enableGpuMode() {
	//nvmlDeviceSetGpuOperationMode();
}
#endif
/* for rows of n<65
	each warpsized sublock (should) reduces multiple spans of 64 columns
	? for given matrix dimension, how many warp-spanning rows will there be?
		for x.n factorOf WARP_SIZE -> 0
		for x.n < ws
		for ws % x.n != 0,
			for primeQ(x.n) there are x.n-1 spanrows per ws*x.n threads
			for !primeQ(x.n) there are x.n/largestCommonFactor(x.n, ws) - 1 spanrows
				for every ws * x.n/largestCommonFactor(x.n, ws) warps

	for m*n threads, there can be at most totalWarps -1 spanrows ( totalWarps = DIV_UP(threads, warpSize))

*/

__host__ CUDART_DEVICE int b_util::countSpanrows( int m, int n, uint warpSize ) {
	uint num = MAX(n,warpSize), den = MIN(n,warpSize);
	int div = num/den;
	if( div* den == num) {
		//flprintf("div %d * num %d (== %d)\n", div, num, (div * num));
		return 0;
	}
	uint sf = smallestFactor(n);
	uint warps = DIV_UP(m * n,warpSize);

	//flprintf("sf %u n/sf %u warps %d\n", sf, n/sf, warps);
	uint factor = sf == n ? n : n/sf;
	return warps * (factor-1)/factor;
	//flprintf("factor (%u) s.t. (uint) (  m * (1. * (factor-1)/(factor))) == %u\n", factor, sr);
}

__host__ __device__ bool b_util::spanrowQ( int row, int n, uint warpSize) {
#ifdef __CUDA_ARCH__
	return ::spanrowQ(row, n);
#else
	uint warpS = row * n / warpSize;
	uint warpE = (row + 1 ) * n / warpSize;
	return warpS != warpE;
#endif
}

template<typename T> __host__ __device__ void util<T>::prdm(const char* msg, const DMatrix<T>& md) {
	printf("%s d: %p (%u*%u*%u)", msg, md.elements, md.m,md.n,md.p);
}
template  __host__ __device__ void util<float>::prdm(const char*,const DMatrix<float>& md);
template  __host__ __device__ void util<double>::prdm(const char*,const DMatrix<double>& md);
template  __host__ __device__ void util<long>::prdm(const char*,const DMatrix<long>& md);
template  __host__ __device__ void util<ulong>::prdm(const char*,const DMatrix<ulong>& md);
template  __host__ __device__ void util<uint>::prdm(const char*,const DMatrix<uint>& md);
template  __host__ __device__ void util<int>::prdm(const char*,const DMatrix<int>& md);

template<typename T> __host__ __device__ void util<T>::prdmln(const char* msg, const DMatrix<T>& md) {
	printf("%s d: %p (%u*%u*%u)\n", msg, md.elements, md.m,md.n,md.p);
}
template  __host__ __device__ void util<float>::prdmln(const char*,const DMatrix<float>& md);
template  __host__ __device__ void util<double>::prdmln(const char*,const DMatrix<double>& md);
template  __host__ __device__ void util<long>::prdmln(const char*,const DMatrix<long>& md);
template  __host__ __device__ void util<ulong>::prdmln(const char*,const DMatrix<ulong>& md);
template  __host__ __device__ void util<uint>::prdmln(const char*,const DMatrix<uint>& md);
template  __host__ __device__ void util<int>::prdmln(const char*,const DMatrix<int>& md);

template<typename T> __host__ __device__ void util<T>::printDm( const DMatrix<T>& dm , const char* msg) {
	uint size = dm.m*dm.p;
	printf("%s (%d*%d*%d) %d &dmatrix=%p elements %p\n",msg,dm.m,dm.n,dm.p,size, &dm,dm.elements);
	T * elems = NULL;
#ifndef __CUDA_ARCH__
	if(dm.elements) {
		checkCudaError(cudaHostAlloc(&elems, size,0));
		CuTimer timer;
		timer.start();
		checkCudaError(cudaMemcpy(elems,dm.elements, size, cudaMemcpyDeviceToHost));
		//CuMatrix<T>::incDhCopy("util<T>::printDm-" + b_util::caller(),size,timer.stop() );
	}
#else
	elems = dm.elements;
#endif
	if(!elems) {
		printf("printDm nothing to see here\n");
		return;
	}

	bool header = false;
	if (checkDebug(debugVerbose) || (dm.m < CuMatrix<T>::getMaxRowsDisplayed() && dm.n < CuMatrix<T>::getMaxColsDisplayed())) {
		for (uint i1 = 0; i1 < dm.m; i1++) {
			if(!header) {
				printf("-");
				for (uint j1 = 0; j1 < dm.n; j1++) {
					if(j1 % 10 == 0) {
						printf(" %d", j1/10);
					}else {
						printf("  ");
					}
					printf(" ");
				}
				printf("\n");
				header = true;
			}
			printf("[");
			for (uint j1 = 0; j1 < dm.n; j1++) {

				if(sizeof(T) == 4)
					printf("% 2.10g", elems[i1 * dm.p + j1]);  //get(i1,j1) );
				else
					printf("% 2.16g", elems[i1 * dm.p + j1]); // get(i1,j1) );
						//);
				if (j1 < dm.n - 1) {
					printf(" ");
				}
			}
			printf("] ");
			if(i1 % 10 == 0) {
				printf("%d", i1);
			}

			printf("\n");
		}
		if(header) {
			printf("+");
			for (uint j1 = 0; j1 < dm.n; j1++) {
				if(j1 % 10 == 0) {
					printf(" %d",j1/10);
				}else {
					printf("  ");
				}
				printf(" ");
			}
			printf("\n");
			header = false;
		}

	} else {
		for (uint i2 = 0; i2 < CuMatrix<T>::getMaxRowsDisplayed() + 1 && i2 < dm.m; i2++) {
			if (i2 == CuMatrix<T>::getMaxRowsDisplayed()) {
				printf(".\n.\n.\n");
				continue;
			}
			for (uint j2 = 0; j2 < CuMatrix<T>::getMaxColsDisplayed() + 1 && j2 < dm.n; j2++) {
				if (j2 == CuMatrix<T>::getMaxColsDisplayed()) {
					printf("...");
					continue;
				}
				if(sizeof(T) == 4)
					printf("% 2.10g", elems[i2 * dm.p + j2]); //get(i2,j2));
				else
					printf("% 2.16g", elems[i2 * dm.p + j2]); //get(i2,j2));
						//elements[i2 * p + j2]);
				if (j2 < dm.n - 1) {
					printf(" ");
				}
			}
			printf("\n");
		}
		if (dm.m > CuMatrix<T>::getMaxRowsDisplayed()) {
			for (uint i3 =dm.m - CuMatrix<T>::getMaxRowsDisplayed(); i3 < dm.m; i3++) {
				if (dm.n > CuMatrix<T>::getMaxColsDisplayed()) {
					for (uint j3 = dm.n - CuMatrix<T>::getMaxColsDisplayed(); j3 < dm.n; j3++) {
						if (j3 == dm.n - CuMatrix<T>::getMaxColsDisplayed()) {
							printf("...");
							continue;
						}
						if(sizeof(T) == 4)
							printf("% 2.10g", elems[i3 * dm.p + j3]);//get(i3, j3));
						else
							printf("% 2.16g", elems[i3 * dm.p + j3]); //get(i3,j3));
								//elements[i3 * p + j3]);
						if (j3 < dm.n - 1) {
							printf(" ");
						}
					}
				} else {
					for (uint j4 = 0; j4 < dm.n; j4++) {
						if(sizeof(T) == 4)
							printf("% 2.10g", elems[i3 * dm.p + j4]); // get(i3,j4));
						else
							printf("% 2.16g", elems[i3 * dm.p + j4]); //get(i3,j4));
								//elements[i3 * p + j4]);

						if (j4 < dm.n - 1) {
							printf(" ");
						}
					}

				}
				printf("\n");
			}
		} else { //if(dm.m > 10) -> dm.n > 10
			for (uint i5 = 0; i5 < CuMatrix<T>::getMaxRowsDisplayed() + 1 && i5 < dm.m; i5++) {

				if (dm.n > CuMatrix<T>::getMaxColsDisplayed()) {
					for (uint j5 = dm.n - CuMatrix<T>::getMaxColsDisplayed(); j5 < dm.n; j5++) {
						if (j5 == dm.n - CuMatrix<T>::getMaxColsDisplayed()) {
							printf("...");
							continue;
						}
						T t = elems[i5 * dm.p + j5];

						if(sizeof(T) == 4)
							printf("% 2.10g", t);
						else
							printf("% 2.16g", t);
						if (j5 < dm.n - 1) {
							printf(" ");
						}
					}
				} else {
					for (uint j4 = 0; j4 < dm.n; j4++) {
						if(sizeof(T) == 4)
							printf("% 2.10g", elems[i5 * dm.p + j4]); //get(i5,j4));
						else
							printf("% 2.16g", elems[i5 * dm.p + j4]); //get(i5,j4));

						if (j4 < dm.n - 1) {
							printf(" ");
						}
					}
				}

				printf("\n");
			}

		}
	}
#ifndef __CUDA_ARCH__
	if(elems) {
		flprintf("freeing host elems %p\n", elems);
		checkCudaErrors(cudaFreeHost(elems));
	}
#endif

}
template void util<float>::printDm(DMatrix<float> const&,const char*);
template void util<double>::printDm(DMatrix<double> const&,const char*);
template void util<ulong>::printDm(DMatrix<ulong> const&,const char*);
template void util<uint>::printDm(DMatrix<uint> const&,const char*);
template void util<int>::printDm(DMatrix<int> const&,const char*);

template<typename T> __host__ __device__ void util<T>::printRow(const DMatrix<T>& dm, int row) {
	prlocf("printRow\n");
	if(!dm.elements) {
		printf("row %d: null elements\n", row);
		return;
	}
	ulong idx = row * dm.p;
	T* elems;
#ifndef __CUDA_ARCH__
	uint rowSize = dm.n*sizeof(T);
	checkCudaError(cudaHostAlloc(&elems, rowSize,0));
	checkCudaError(cudaMemcpy(elems, dm.elements + idx, rowSize, cudaMemcpyHostToDevice));
#else
	elems = dm.elements;
#endif
	printf("row %d: ", row);
	for(int c = 0; c < dm.n; c++) {
		printf("%5.2f ", elems[idx + c]);
	}
	printf("\n");
#ifndef __CUDA_ARCH__
	flprintf("freeing host elems %p\n", elems);
	checkCudaError(cudaFreeHost(elems));
#endif
}
template void util<float>::printRow(DMatrix<float> const&,int);
template void util<double>::printRow(DMatrix<double> const&,int);
template void util<long>::printRow(DMatrix<long> const&,int);
template void util<ulong>::printRow(DMatrix<ulong> const&,int);
template void util<int>::printRow(DMatrix<int> const&,int);
template void util<uint>::printRow(DMatrix<uint> const&,int);

__host__ __device__ void b_util::vectorExecContext(int threads, int count, dim3& dBlocks,
		dim3& dThreads, const void* kernel) {
	if (threads % WARP_SIZE != 0) {
			printf("WARN: %d is not a multiple of the warp size (32)\n",threads);
	}
	int blocks = DIV_UP(count, threads);
	dThreads.y = dThreads.z = 1;
	dThreads.x = threads;

	if(kernel) {
		validLaunch(dThreads,kernel);
		if(dThreads.x != threads) {
			blocks = DIV_UP(count, dThreads.x);
		}
	}

	dBlocks.y = dBlocks.z = 1;
	dBlocks.x = blocks;

#ifndef __CUDA_ARCH__
	totalThreads += dBlocks.x * dThreads.x;
	totalElements += count;
#endif
	if (checkDebug(debugExec))
		printf(
				"contxt of %d  blks of %d threads for count of %d\n", dBlocks.x, dThreads.x,count);
}

template<typename T> __host__ __device__ void util<T>::pDarry(const T* arry, int cnt) {
	for(int i = 0; i < cnt; i++) {
		printf("%f ", arry[i]);
	}
	printf("\n");
}

template<> __host__ __device__ void util<uint>::pDarry(const uint* arry, int cnt) {
	uint* ptr = (uint*)arry;
#ifndef __CUDA_ARCH__
	checkCudaError(cudaMallocHost(&ptr, cnt * sizeof(uint)));
	CuTimer timer;
	timer.start();
	checkCudaError(cudaMemcpy(ptr, arry, cnt*sizeof(uint), cudaMemcpyDeviceToHost));
	//CuMatrix<uint>::incDhCopy("util<uint>::pDarry-" + b_util::caller() , cnt*sizeof(uint),timer.stop());
#endif
	for(int i = 0; i < cnt; i++) {
		printf("%.3f ", (double) ptr[i]);
	}
	printf("\n");
#ifndef __CUDA_ARCH__
	flprintf("freeing host ptr %p\n", ptr);
	checkCudaError(cudaFreeHost(ptr));
#endif
}

template __host__ __device__ void util<float>::pDarry(const float*, int);
template __host__ __device__ void util<double>::pDarry(const double*, int);
template __host__ __device__ void util<ulong>::pDarry(const ulong*, int);
template __host__ __device__ void util<int>::pDarry(const int*, int);

template <typename T>  __host__ __device__ void util<T>::fillNDev(T* trg, T val, long n) {
	int threads = 512;
	dim3 dBlocks, dThreads;
	b_util::vectorExecContext(threads, n, dBlocks, dThreads);
	if(checkDebug(debugCheckValid)) {
		flprintf("currdev %d -- trg %p val %.5f N = %d\n", ExecCaps::currDev(), trg,val,n);
		MemMgr<T>::checkValid(trg, "fillNDev start");
		MemMgr<T>::checkValid(trg + n/2, "fillNDev half/N");
		MemMgr<T>::checkValid(trg + 3* n/4, "fillNDev 3N/4");
		MemMgr<T>::checkValid(trg + n-1, "fillNDev N-1");

		//MemMgr<T>::checkRange(trg, n-1, "fillNDev range of N");
	}

	fillKernel<<<dBlocks,dThreads>>>(trg, val, n);
	cherr(cudaDeviceSynchronize());
}
template void util<float>::fillNDev(float*, float, long);
template void util<double>::fillNDev(double*, double, long);
template void util<ulong>::fillNDev(ulong*, ulong, long);
template void util<long>::fillNDev(long*, long, long);
template void util<uint>::fillNDev(uint*, uint, long);
template void util<int>::fillNDev(int*, int, long);

//////////////////////////
//
// IndexArray
//
//////////////////////////
//

__host__ __device__ IndexArray::IndexArray() :
		indices(null), count(0), owner(true) {
}

__host__ __device__ IndexArray::IndexArray(const IndexArray & o) :
		indices(o.indices), count(o.count), owner(false) {
}

__host__ __device__ IndexArray::IndexArray(uint* _indices, uint _count, bool _owner) :
		indices(_indices), count(_count), owner(_owner) {
}

__host__ __device__ IndexArray::IndexArray(uint idx1, uint idx2) :
		count(2), owner(true) {
	indices = new uint[2];
	indices[0] = idx1;
	indices[1] = idx2;
}

/*
intPair IndexArray::toPair() const {
	assert(count == 2);
	return intPair(indices[0], indices[1]);
}
*/

__host__ __device__ IndexArray::~IndexArray() {
	if (indices != null && owner) {
		delete[] indices;
	}
}

string IndexArray::toString(bool lf) const {
	stringstream ssout;
	ssout << "IdxArr" << (owner ? "+" : "-") << "(" << count << ")[ \n";

	for (uint i = 0; i < count; i++) {
		ssout << indices[i];
		if (i < count - 1) {
			ssout << (lf ? ( i % 50 == 0 ? "----\n" : "\n") : ", ");
		}
	}
	ssout << " ]";
	return ssout.str();

}

__host__ __device__ void b_util::expNotation(char* buff, long val) {
#ifndef __CUDA_ARCH__
	double factor = 1.;
	if (val >= Giga) {
		factor = 1. / Giga;
		sprintf(buff, "%2.3gGb", (double) val * factor);
	} else if (val >= Mega) {
		factor = 1. / Mega;
		sprintf(buff, "%2.3gMb", (double)val * factor);
	} else if (val >= Kilo) {
		factor = 1. / Kilo;
		sprintf(buff, "%2.3gKb", (double)val * factor);
	} else {
		sprintf(buff, "%2.3gb", (double)val * factor);
	}
#endif
}
__host__ void b_util::expNotationMem(char* buff, long val) {
	double factor = 1.;
	if (val >= Terab) {
			factor = 1. / Terab;
		sprintf(buff, "%2.3gTb", (double) val * factor);
	} else if (val >= Gigab) {
		factor = 1. / Gigab;
		sprintf(buff, "%2.3gGb", (double) val * factor);
	} else if (val >= Megab) {
		factor = 1. / Megab;
		sprintf(buff, "%2.3gMb", (double)val * factor);
	} else if (val >= Kilob) {
		factor = 1. / Kilob;
		sprintf(buff, "%2.3gKb", (double)val * factor);
	} else {
		sprintf(buff, "%2.3gb", (double)val * factor);
	}
}
__host__ string b_util::expNotationMemStr( long val) {
	char buff[256];
	expNotationMem(buff, val);
	stringstream ss;
	ss << buff;
	return ss.str();
}
__host__ CUDART_DEVICE double b_util::currMemRatio( ) {
	size_t freeMemory =1, totalMemory =1;
	cherr(		cudaMemGetInfo(&freeMemory, &totalMemory));
	return 100 * (1 - freeMemory * 1. / totalMemory);
}
__host__ CUDART_DEVICE double b_util::usedMemRatio(bool allDevices) {
	//outln("b_util::usedMemRatio("<< tOrF(allDevices) <<  ") ent");
	//b_util::dumpStack();
	size_t freeMemory =1, totalMemory =1;
	if(allDevices) {
#ifndef __CUDA_ARCH__
		ExecCaps::allGpuMem(&freeMemory, &totalMemory);
#endif
	}
	else {
		cudaMemGetInfo(&freeMemory, &totalMemory);
	}
	return 100 * (1 - freeMemory * 1. / totalMemory);
}



__host__ CUDART_DEVICE void b_util::usedDmem(bool allDevices) {
	//outln("b_util::usedDmem("<< tOrF(allDevices) <<  ") ent");
#ifndef __CUDA_ARCH__
	flprintf("Memory %.3f%% used\n", usedMemRatio(allDevices));
#else
	flprintf("Memory %.3f%% used\n", usedMemRatio(false));
#endif
}

__host__ __device__  void b_util::_checkCudaError(const char* file, int line, cudaError_t val) {
	if (val != cudaSuccess) {
		printf("CuMatrixException (%d) %s at %s : %d\n",val ,__cudaGetErrorEnum(val),file, line);
#ifndef __CUDA_ARCH__
//		cout << print_stacktrace() << endl;
#endif
		assert(false);
	}
}

template <> __host__ __device__ float sqrt_p(float val) {
	return sqrtf(val);
}
template <> __host__ __device__ double sqrt_p(double val) {
	return sqrt(val);
}
template <> __host__ __device__ long sqrt_p(long val) {
	return (long)sqrtf(val);
}
template <> __host__ __device__ ulong sqrt_p(ulong val) {
	return (ulong)sqrtf(val);
}
template <> __host__ __device__ int sqrt_p(int val) {
	return (int)sqrtf((float)val);
}
template <> __host__ __device__ uint sqrt_p(uint val) {
	return (uint)sqrtf((float)val);
}

template<typename T> __host__ __device__ void prry(T* ry) {

}

template <> template <> __host__ __device__ void Packeur<double4>::packNext<double>(
		double h) {
	switch (currIdx) {
	case 0:
		target.x = h;
		break;
	case 1:
		target.y = h;
		break;
	case 2:
		target.z = h;
		break;
	case 3:
		target.w = h;
		break;
	default:
		setLastError(outOfBoundsEx);
		return;
	}
	currIdx++;
}

template<> template <> __host__ __device__ void Packeur<double3>::packNext<double>(
		double h) {
	switch (currIdx) {
	case 0:
		target.x = h;
		break;
	case 1:
		target.y = h;
		break;
	case 2:
		target.z = h;
		break;
	default:
		setLastError(outOfBoundsEx);
		return;
	}
	currIdx++;
}

template<> template <> __host__ __device__ void Packeur<int3>::packNext<int>(
		int h) {
	switch (currIdx) {
	case 0:
		target.x = h;
		break;
	case 1:
		target.y = h;
		break;
	case 2:
		target.z = h;
		break;
	default:
		setLastError(outOfBoundsEx);
		return;
	}
	currIdx++;
}

template<> template <> __host__ __device__ void Packeur<uint3>::packNext<uint>(
		uint h) {
	switch (currIdx) {
	case 0:
		target.x = h;
		break;
	case 1:
		target.y = h;
		break;
	case 2:
		target.z = h;
		break;
	default:
		setLastError(outOfBoundsEx);
		return;
	}
	currIdx++;
}

template<> __host__ __device__ void Packeur2::render<double4>(double4& target) {
	switch (currIdx) {
	case 1:
		target.x = vals[0];
		break;
	case 2:
		target.x = vals[0];
		target.y = vals[1];
		break;
	case 3:
		target.x = vals[0];
		target.y = vals[1];
		target.z = vals[2];
		break;
	case 4:
		target.x = vals[0];
		target.y = vals[1];
		target.z = vals[2];
		target.w = vals[3];
		break;
	default:
		setLastError(outOfBoundsEx);
		return;
	}
}


template<typename T> template <typename PackType> __host__ __device__ PackType Packeur3<T>::render() {
	if(sizeof(T) == 4) {
		switch (currIdx) {
		case 1:
			return vals[0];
		case 2:
			return render<float2>();
		case 3:
			return render<float3>();
		case 4:
			return render<float4>();
		default:
			setLastError(outOfBoundsEx);
		}

	}else {
		switch (currIdx) {
		case 1:
			return vals[0];
		case 2:
			return render<double2>();
		case 3:
			return render<double3>();
		case 4:
			return render<double4>();
		default:
			setLastError(outOfBoundsEx);
		}
	}
}


template<> template<> __host__ __device__ double4 Packeur3<double>::render() {
	double4 target;
	switch (currIdx) {
	case 1:
		target.x = vals[0];
		break;
	case 2:
		target.x = vals[0];
		target.y = vals[1];
		break;
	case 3:
		target.x = vals[0];
		target.y = vals[1];
		target.z = vals[2];
		break;
	case 4:
		target.x = vals[0];
		target.y = vals[1];
		target.z = vals[2];
		target.w = vals[3];
		break;
	default:
		setLastError(outOfBoundsEx);
	}
	return target;
}

template<> template<> __host__ __device__ float4 Packeur3<float>::render() {
	float4 target;
	switch (currIdx) {
	case 1:
		target.x = vals[0];
		break;
	case 2:
		target.x = vals[0];
		target.y = vals[1];
		break;
	case 3:
		target.x = vals[0];
		target.y = vals[1];
		target.z = vals[2];
		break;
	case 4:
		target.x = vals[0];
		target.y = vals[1];
		target.z = vals[2];
		target.w = vals[3];
		break;
	default:
		setLastError(outOfBoundsEx);
	}
	return target;
}

template<> template<> __host__ __device__ ulong4 Packeur3<ulong>::render() {
	ulong4 target;
	switch (currIdx) {
	case 1:
		target.x = vals[0];
		break;
	case 2:
		target.x = vals[0];
		target.y = vals[1];
		break;
	case 3:
		target.x = vals[0];
		target.y = vals[1];
		target.z = vals[2];
		break;
	case 4:
		target.x = vals[0];
		target.y = vals[1];
		target.z = vals[2];
		target.w = vals[3];
		break;
	default:
		setLastError(outOfBoundsEx);
	}
	return target;
}

