#include "caps.h"
#include "Kernels.h"
#include <cuda.h>
#include "MemMgr.h"

__device__ int d_MaxThreads = 512;
__device__ int d_MaxBlocks = 128;

__device__ ExecCaps** gd_devCaps = nullptr;
__constant__ int gd_devCount[MAX_GPUS];

 __host__ __device__ void getReductionExecContext(int &blocks, int &threads, long nP,int maxBlocks, int maxThreads) {
	int x = (nP + 1) / 2;
	if(x < 2) {
		 x=2;
	} else {
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		++x;
	}

	threads =  nP == 2 ? 1 : (nP < (ulong) maxThreads * 2) ? x : maxThreads;
	blocks =  DIV_UP(nP, threads*2);

	blocks = MIN(maxBlocks, blocks);
	if(checkDebug(debugRedux))flprintf("np %d -> blocks %d of threads %d\n", nP, blocks, threads);
}

__host__ __device__ ExecCaps::~ExecCaps() {
	flprintf( "ExecCaps::~ExecCaps() this %p\n",this);
#ifndef __CUDA_ARCH__
	if(checkDebug(debugExec))b_util::dumpStack();
#endif
}

__host__ __device__ int ExecCaps::currDev() {
	int dev = 0;
#ifndef __CUDA_ARCH__
	cherr(cudaGetDevice(&dev));
#endif
	return dev;
}
__host__ __device__ int ExecCaps::setDevice(const char* filename,const char* func, int line, int dev) {
	int orgDev = currDev();
#ifndef __CUDA_ARCH__
//	if(checkDebug(debugCheckValid))
		if(orgDev != dev) {
			checkCudaErrors(cudaSetDevice(dev));
			printf( "%s:%d %s changing device to %d from %d\n", filename, line, func, dev, orgDev);
		}
#endif
	return orgDev;
}
__host__ __device__ int ExecCaps::visitDevice(const char* filename,const char* func, int line, int dev) {
	int orgDev = currDev();
#ifndef __CUDA_ARCH__
//	if(checkDebug(debugCheckValid))
		if(orgDev != dev) {
			checkCudaErrors(cudaSetDevice(dev));
			if(checkDebug(debugExec))printf( "%s:%d %s visiting (temporarily) device %d from %d\n", filename, line, func, dev, orgDev);
		}
#endif
	return orgDev;
}

__host__ __device__ int ExecCaps::restoreDevice(const char* filename,const char* func, int line, int dev) {
	int orgDev = currDev();
#ifndef __CUDA_ARCH__
//	if(checkDebug(debugCheckValid))
		if(orgDev != dev) {
			cherr(cudaSetDevice(dev));
			if(checkDebug(debugExec))printf( "%s:%d %s restoring device %d from %d\n", filename, line, func, dev, orgDev);
		}
#endif
	return orgDev;
}


__global__ void freeDevSideDevCaps() {
	//flprintf( "freeDevSideDevCaps freeing gd_devCaps %p gd_devCount %d \n", gd_devCaps, gd_devCount);
	prlocf("freeDevSideDevCaps\n");
	FirstThread {
		for(int i = 0; i < ExecCaps::countGpus(); i++) {
			flprintf("freeDevSideDevCaps deleting gd_devCaps[%d] %p\n", i, gd_devCaps[i]);
			delete gd_devCaps[i];
		}
		flprintf( "freeinbg gd_devCaps %p\n", gd_devCaps);
		free(gd_devCaps);
	}
}

void ExecCaps::freeDevCaps() {
	flprintf( "ExecCaps::freeDevCaps freeDevCaps enter this %d\n",0);
	for(int i = 0; i < ExecCaps::countGpus(); i++) {
		flprintf("ExecCaps::freeDevCaps deleting g_devCaps[device = %d] %p\n",i, g_devCaps[i]);
		delete g_devCaps[i];
	}
	flprintf( "ExecCaps::freeDevCaps freeing g_devCaps %p\n",g_devCaps);
	free(g_devCaps);
	//freeDevSideDevCaps<<<1,1>>>();
}


__host__ __device__ cudaError_t ExecCaps::currCaps(ExecCaps** caps, int dev) {

#ifndef __CUDA_ARCH__
	if(dev < ExecCaps::countGpus()) {
		*caps = g_devCaps[dev];
		return cudaSuccess;
	}
	return cudaErrorUnknown;
#else
	if(true) {
	//s	setLastError(notImplementedEx);
		return cudaErrorAssert;
	}
#endif

}

__host__ __device__ ExecCaps* ExecCaps::currCaps(int dev ) {
#ifndef __CUDA_ARCH__
	if(checkDebug(debugExec))flprintf( "in ExecCaps::currCaps ExecCaps::countGpus() %d\n",ExecCaps::countGpus());
	if(dev < ExecCaps::countGpus()) {
		return g_devCaps[dev];
	}
#else
	if(checkDebug(debugExec))flprintf( "in [D}ExecCaps::currCaps gd_devCount %d\n",gd_devCount);
//	setLastError(notImplementedEx);
/*	if(dev < ExecCaps::countGpus()) {*/
		ExecCaps* pCaps = gd_devCaps[dev];
		flprintf( "pCaps %p \n ",pCaps);
		flprintf( "gd_devCaps[%d] %p \n ",dev, pCaps);

		return pCaps;
/*	}*/
#endif
	return nullptr;
}


__host__ __device__ void ExecCaps::printMaxDims(const char* msg) {
	printf("%s for dev %d maxGrid(%u,%u,%u)", msg, devNumber, maxGrid.x,maxGrid.y,maxGrid.z);
	printf("maxBlock(%u,%u,%u)", maxBlock.x,maxBlock.y,maxBlock.z);
}


/*
__host__ __device__  cudaError_t ExecCaps::currStream(cudaStream_t* stream,int dev) {
	ExecCaps* currCaps = null;
	cudaError_t res = ExecCaps::currCaps(&currCaps,dev) ;
	if(res != cudaSuccess) {
		cherr(res);
		return res;
	}
	if(currCaps) {
		if(checkDebug(debugStream))flprintf( "currStream(dev = %d) -> %p\n", dev, currCaps->stream);
		*stream = currCaps->stream;
		return cudaSuccess;
	}
	*stream = null;
	return cudaErrorUnknown;
}
*/
/*

__host__ __device__ cudaStream_t ExecCaps::currStream(int dev) {
	cudaStream_t currStream;
	cudaError_t res = ExecCaps::currStream(&currStream, dev);
	if(res != cudaSuccess) {
		if(checkDebug(debugStream))flprintf( "currStream error for dev %d\n", dev);
	}
	cherr(res);
	return currStream;
}

*/
__host__ __device__ const char *__cudaGetErrorEnum(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";

        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration";

        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";

        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";

        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";

        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure";

        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";

        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";

        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";

        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";

        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";

        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";

        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue";

        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol";

        case cudaErrorMapBufferObjectFailed:
            return "cudaErrorMapBufferObjectFailed";

        case cudaErrorUnmapBufferObjectFailed:
            return "cudaErrorUnmapBufferObjectFailed";

        case cudaErrorInvalidHostPointer:
            return "cudaErrorInvalidHostPointer";

        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";

        case cudaErrorInvalidTexture:
            return "cudaErrorInvalidTexture";

        case cudaErrorInvalidTextureBinding:
            return "cudaErrorInvalidTextureBinding";

        case cudaErrorInvalidChannelDescriptor:
            return "cudaErrorInvalidChannelDescriptor";

        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";

        case cudaErrorAddressOfConstant:
            return "cudaErrorAddressOfConstant";

        case cudaErrorTextureFetchFailed:
            return "cudaErrorTextureFetchFailed";

        case cudaErrorTextureNotBound:
            return "cudaErrorTextureNotBound";

        case cudaErrorSynchronizationError:
            return "cudaErrorSynchronizationError";

        case cudaErrorInvalidFilterSetting:
            return "cudaErrorInvalidFilterSetting";

        case cudaErrorInvalidNormSetting:
            return "cudaErrorInvalidNormSetting";

        case cudaErrorMixedDeviceExecution:
            return "cudaErrorMixedDeviceExecution";

        case cudaErrorCudartUnloading:
            return "cudaErrorCudartUnloading";

        case cudaErrorUnknown:
            return "cudaErrorUnknown";

        case cudaErrorNotYetImplemented:
            return "cudaErrorNotYetImplemented";

        case cudaErrorMemoryValueTooLarge:
            return "cudaErrorMemoryValueTooLarge";

        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle";

        case cudaErrorNotReady:
            return "cudaErrorNotReady";

        case cudaErrorInsufficientDriver:
            return "cudaErrorInsufficientDriver";

        case cudaErrorSetOnActiveProcess:
            return "cudaErrorSetOnActiveProcess";

        case cudaErrorInvalidSurface:
            return "cudaErrorInvalidSurface";

        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";

        case cudaErrorECCUncorrectable:
            return "cudaErrorECCUncorrectable";

        case cudaErrorSharedObjectSymbolNotFound:
            return "cudaErrorSharedObjectSymbolNotFound";

        case cudaErrorSharedObjectInitFailed:
            return "cudaErrorSharedObjectInitFailed";

        case cudaErrorUnsupportedLimit:
            return "cudaErrorUnsupportedLimit";

        case cudaErrorDuplicateVariableName:
            return "cudaErrorDuplicateVariableName";

        case cudaErrorDuplicateTextureName:
            return "cudaErrorDuplicateTextureName";

        case cudaErrorDuplicateSurfaceName:
            return "cudaErrorDuplicateSurfaceName";

        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";

        case cudaErrorInvalidKernelImage:
            return "cudaErrorInvalidKernelImage";

        case cudaErrorNoKernelImageForDevice:
            return "cudaErrorNoKernelImageForDevice";

        case cudaErrorIncompatibleDriverContext:
            return "cudaErrorIncompatibleDriverContext";

        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";

        case cudaErrorPeerAccessNotEnabled:
            return "cudaErrorPeerAccessNotEnabled";

        case cudaErrorDeviceAlreadyInUse:
            return "cudaErrorDeviceAlreadyInUse";

        case cudaErrorProfilerDisabled:
            return "cudaErrorProfilerDisabled";

        case cudaErrorProfilerNotInitialized:
            return "cudaErrorProfilerNotInitialized";

        case cudaErrorProfilerAlreadyStarted:
            return "cudaErrorProfilerAlreadyStarted";

        case cudaErrorProfilerAlreadyStopped:
            return "cudaErrorProfilerAlreadyStopped";

#if __CUDA_API_VERSION >= 0x4000

        case cudaErrorAssert:
            return "cudaErrorAssert";

        case cudaErrorTooManyPeers:
            return "cudaErrorTooManyPeers";

        case cudaErrorHostMemoryAlreadyRegistered:
            return "cudaErrorHostMemoryAlreadyRegistered";

        case cudaErrorHostMemoryNotRegistered:
            return "cudaErrorHostMemoryNotRegistered";
#endif

//#if __CUDA_API_VERSION >= 6000
        case cudaErrorOperatingSystem:
            return "cudaErrorOperatingSystem";
        case cudaErrorPeerAccessUnsupported:
            return "cudaErrorPeerAccessUnsupported";
        case cudaErrorLaunchMaxDepthExceeded:
            return "cudaErrorLaunchMaxDepthExceeded";
        case cudaErrorLaunchFileScopedTex:
            return "cudaErrorLaunchFileScopedTex";
        case cudaErrorLaunchFileScopedSurf:
            return "cudaErrorLaunchFileScopedSurf";
        case cudaErrorSyncDepthExceeded:
            return "cudaErrorSyncDepthExceeded";
        case cudaErrorLaunchPendingCountExceeded:
            return "cudaErrorLaunchPendingCountExceeded";
        case cudaErrorNotPermitted:
            return "cudaErrorNotPermitted";
        case cudaErrorNotSupported:
            return "cudaErrorNotSupported";
        case cudaErrorHardwareStackError:
            return "cudaErrorHardwareStackError";
        case cudaErrorIllegalInstruction:
            return "cudaErrorIllegalInstruction";
        case cudaErrorMisalignedAddress:
            return "cudaErrorMisalignedAddress";
        case cudaErrorInvalidAddressSpace:
            return "cudaErrorInvalidAddressSpace";
        case cudaErrorInvalidPc:
            return "cudaErrorInvalidPc";
        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";
//#endif
        case cudaErrorStartupFailure:
            return "cudaErrorStartupFailure";
        case cudaErrorApiFailureBase:
            return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
}

__host__ __device__ const char *__cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
    	return "CUBLAS_STATUS_SUCCESS =0";
    case   CUBLAS_STATUS_NOT_INITIALIZED:
    	return "CUBLAS_STATUS_NOT_INITIALIZED =1";
    case CUBLAS_STATUS_ALLOC_FAILED:
    	return "CUBLAS_STATUS_ALLOC_FAILED    =3";
    case CUBLAS_STATUS_INVALID_VALUE:
    	return "CUBLAS_STATUS_INVALID_VALUE   =7";
    case CUBLAS_STATUS_ARCH_MISMATCH:
    	return "CUBLAS_STATUS_ARCH_MISMATCH   =8";
    case CUBLAS_STATUS_MAPPING_ERROR:
    	return "CUBLAS_STATUS_MAPPING_ERROR   =11";
    case CUBLAS_STATUS_EXECUTION_FAILED:
    	return "CUBLAS_STATUS_EXECUTION_FAILED=13";
    case CUBLAS_STATUS_INTERNAL_ERROR:
    	return "CUBLAS_STATUS_INTERNAL_ERROR  =14";
    case CUBLAS_STATUS_NOT_SUPPORTED:
    	return "CUBLAS_STATUS_NOT_SUPPORTED   =15";
    case CUBLAS_STATUS_LICENSE_ERROR:
    	return "CUBLAS_STATUS_LICENSE_ERROR   =16";
    default:
    	return "UNKNOWN";
    }


}

__host__ __device__ int ExecCaps::maxThreads() {
#ifndef __CUDA_ARCH__
		return MaxThreads;
#else
		return d_MaxThreads;
#endif
}
__host__ __device__ int ExecCaps::maxBlocks() {
#ifndef __CUDA_ARCH__
		return MaxBlocks;
#else
		return d_MaxBlocks;
#endif
}

/*
 * survey all gpus and find the smallest 'reasonable' max buffer (as headroom fraction of total)
 */
__host__ __device__ size_t ExecCaps::minMaxReasonable(int gpuMask, float headroom) {
	int devCnt;
	cherr(cudaPeekAtLastError());
	checkCudaError(cudaGetDeviceCount(&devCnt));
	size_t minMax = 0, currMax = 0;

	for(int i = 0; i < devCnt; i++) {
		if(gpuMask & (1 << i)) {
			currMax = ExecCaps::currCaps(i)->maxReasonable(headroom);
			if(checkDebug(debugMem))flprintf("device %d has total %lu maxreas %lu (at %2.2f head)\n", i,  ExecCaps::currCaps(i)->deviceProp.totalGlobalMem, currMax, headroom);
			minMax = minMax == 0 ? currMax : MIN(minMax, currMax);
		}
	}
	return minMax;
}

__global__ void createCapsPPtr(int devCount) {
	FirstThread {
		gd_devCaps = (ExecCaps**) malloc(devCount* sizeof(ExecCaps*));
		flprintf("createCapsPPtr created %p (%d bytes)\n",gd_devCaps, sizeof(ExecCaps*));
	}
}

__global__ void addCaps(int dev, ExecCaps caps) {
	FirstThread {
		ExecCaps* pcaps = new ExecCaps(caps);
		gd_devCaps[dev] = pcaps;
		memcpy(pcaps, &caps, sizeof(ExecCaps));
		flprintf("addCaps created gd_devCaps[%d] %p (%d bytes)\n",dev, pcaps, sizeof(ExecCaps));
	}
}

cudaError_t ExecCaps::addDevice(int dev) {
	ExecCaps_setDevice(dev);
//	checkCudaError(cudaMemcpyToSymbol(gd_devCount, (void*) &ExecCaps::countGpus(), sizeof(int)));
 //	cuDeviceGet(&device,dev);

	createCapsPPtr<<<1,1>>>(ExecCaps::countGpus());
	addCaps<<<1,1>>>(dev, *g_devCaps[dev]);
	return cudaSuccess;
}

void ExecCaps::initDevCaps() {
	//flprintf("%s enter\n","ExecCaps::initDevCaps");
	if(checkDebug(debugVerbose))prlocf("enter\n");
	//ExecCaps* g_devCaps[];
	g_devCaps = (ExecCaps**) malloc(ExecCaps::countGpus()* sizeof(ExecCaps*));
	if(checkDebug(debugVerbose))flprintf("gpuCount %d, malloced-> %d\n",ExecCaps::countGpus(),ExecCaps::countGpus()* sizeof(ExecCaps*) );
	//outln("gpuCount " << ExecCaps::countGpus() << " mallocd " <<(ExecCaps::countGpus()* sizeof(ExecCaps*)));
	for(int i = 0; i < ExecCaps::countGpus(); i++) {
		ExecCaps* cap = new ExecCaps();
		if(checkDebug(debugVerbose))flprintf("created cap %p\n", cap);
		g_devCaps[i] = cap;
		ExecCaps::getExecCaps(*cap, i);
		if(checkDebug(debugVerbose))outln("adding " << i << "\n" << cap->toString() << "\n");
		ExecCaps::addDevice(i);
	}
}

__host__ void ExecCaps::allGpuMem(size_t* free, size_t* total) {
	if(checkDebug(debugVerbose))outln("allGpuMem free " << free <<  ", total " << total);
    int orgDev = ExecCaps::currDev();
    if(checkDebug(debugVerbose))outln("orgDev " << orgDev);
    *free=0;
    *total=0;
    size_t lFree = 0, lTotal = 0;
	int devCnt;
	char buff[20];
	cherr(cudaGetDeviceCount(&devCnt));
	if(checkDebug(debugVerbose))outln("devCnt " << devCnt);
//cout << "  ";
	float* dtest = nullptr;

	b_util::dumpStack();
	for(int i = devCnt -1; i >-1;i--) {
		ExecCaps_visitDevice(i);
		if(checkDebug(debugVerbose))outln("ExecCaps_setDevice  " << i );
		cherr(cudaMalloc(&dtest, 10 * sizeof(float)));
		if(checkDebug(debugVerbose))outln("cudaMalloc  dtest " << dtest << "...checking valid") ;
		MemMgr<float>::checkValid(dtest, "ExecCaps::allGpuMem dtest ");
		if(checkDebug(debugVerbose))outln("ExecCaps_setDevice  " << i );
		cherr(cudaMemGetInfo(&lFree, &lTotal));
		*free += lFree;
		*total += lTotal;
		sprintf(buff, " (%.2f%% used)", 100 * (1 - lFree * 1. / lTotal));
		if(checkDebug(debugMem)) cout << "[" << i <<":  " <<  b_util::expNotation(lFree) << " free /" << b_util::expNotation(lTotal) << buff<< "] ";
	}
	//cout << endl;
	ExecCaps_setDevice(orgDev);
}
__host__ __device__ DevStream::DevStream(int device) :  device(device) {
	int orgDev = ExecCaps::currDev();
	if(orgDev != device)
		ExecCaps_visitDevice(device);

#ifndef __CUDA__ARCH__
	cherr(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	flprintf("on gpieux %d, created DevStream(%d,%p)\n",  ExecCaps::currDev(), device, stream);
	if(orgDev != device)
		ExecCaps_setDevice(orgDev);
#else
	stream = 0;
#endif
}

__host__ __device__ cudaError_t  DevStream::sync() {
	int orgDev = ExecCaps::currDev();
	if(orgDev != device)
		ExecCaps_visitDevice(device);
	flprintf("on gpieux  %d syncing DevStream(%d,%p)\n",  ExecCaps::currDev(), device, stream);
#ifndef __CUDA_ARCH__
	cudaError_t res =  cudaStreamSynchronize(stream);
#else
	cudaError_t res =  cudaDeviceSynchronize();
#endif
	flprintf("sanched  -> %s\n", __cudaGetErrorEnum(res) );
	if(orgDev != device)
		ExecCaps_setDevice(orgDev);

	return res;
}
__host__ __device__ cudaError_t DevStream::destroy() {
	return cudaStreamDestroy(stream);
}

__host__ __device__ DevStream::~DevStream() {
	cherr(destroy());
}
