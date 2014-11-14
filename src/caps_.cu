#include "caps.h"
#include "Kernels.h"
#include <cuda.h>


//extern __host__ __device__ void setLastError(CuMatrixException lastEx);

__device__ int d_MaxThreads = 512;
__device__ int d_MaxBlocks = 128;

__device__ ExecCaps** gd_devCaps = null;
__constant__ int gd_devCount[MAX_GPUS];

 __host__ __device__ void getReductionExecContext(uint &blocks, uint &threads, ulong nP,int maxBlocks, int maxThreads) {
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
	if(checkDebug(debugUnaryOp))flprintf("np %d -> blocks %d of threads %d\n", nP, blocks, threads);
/*
	if (debugExec) {
			char buff[5];
			T blockOcc = threads* 1. / ExecCaps::currCaps()->thrdPerBlock;
			sprintf(buff,"%1.3g",blockOcc);
			ot("nP " << nP << ", threads " << threads << "(" << buff << ")");
			T globOcc = threads*blocks *1./ (ExecCaps::currCaps()->totalThreads());
			sprintf(buff,"%1.3g",globOcc);
			outln(", blocks " << blocks  << "(" << buff << ")" );
	}
*/
}

__host__ __device__ ExecCaps::~ExecCaps() {
	flprintf( "ExecCaps::~ExecCaps() this %p\n",this);
#ifndef __CUDA_ARCH__
	if(checkDebug(debugExec))b_util::dumpStack();
#endif
}

__global__ void freeDevSideDevCaps() {
	//flprintf( "freeDevSideDevCaps freeing gd_devCaps %p gd_devCount %d \n", gd_devCaps, gd_devCount);
	prlocf("freeDevSideDevCaps\n");
/*
	FirstThread {
		for(int i = 0; i < g_devCount; i++) {
			flprintf("freeDevSideDevCaps deleting gd_devCaps[%d] %p\n", i, gd_devCaps[i]);
			delete gd_devCaps[i];
		}
		flprintf( "freeinbg gd_devCaps %p\n", gd_devCaps);
		free(gd_devCaps);
	}
	*/
}

void ExecCaps::freeDevCaps() {
	flprintf( "ExecCaps::freeDevCaps freeDevCaps enter this %d\n",0);
	checkCudaError(cudaGetDeviceCount(&g_devCount));
	outln("gpuCount " << g_devCount);
	for(int i = 0; i < g_devCount; i++) {
		flprintf("ExecCaps::freeDevCaps deleting g_devCaps[device = %d] %p\n",i, g_devCaps[i]);
		delete g_devCaps[i];
	}
	flprintf( "ExecCaps::freeDevCaps freeing g_devCaps %p\n",g_devCaps);
	free(g_devCaps);
	//freeDevSideDevCaps<<<1,1>>>();
}


__host__ __device__ cudaError_t ExecCaps::currCaps(ExecCaps** caps, int dev) {

#ifndef __CUDA_ARCH__
	if(dev < g_devCount) {
		*caps = g_devCaps[dev];
		return cudaSuccess;
	}
#else
	if(true) {
	//s	setLastError(notImplementedEx);
		return cudaErrorAssert;
	}
#endif
}

__host__ __device__ ExecCaps* ExecCaps::currCaps(int dev ) {
#ifndef __CUDA_ARCH__
	if(checkDebug(debugExec))flprintf( "in ExecCaps::currCaps g_devCount %d\n",g_devCount);
	if(dev < g_devCount) {
		return g_devCaps[dev];
	}
//#else
//	if(checkDebug(debugExec))flprintf( "in [D}ExecCaps::currCaps gd_devCount %d\n",gd_devCount);
//	setLastError(notImplementedEx);
//	if(dev < gd_devCount) {
//		flprintf( "pCaps %p \n ",pCaps);
//		flprintf( "gd_devCaps[%d] %p \n ",dev, gd_devCaps[dev]);

//		return gd_devCaps[dev];
//	}
#endif
	return null;
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

cudaError_t ExecCaps::setDevice(int dev) {
	cudaError_t res = cudaSetDevice(dev);
	checkCudaError(res);
//	checkCudaError(cudaMemcpyToSymbol(gd_devCount, (void*) &g_devCount, sizeof(int)));
//	CUdevice device;
//	cuDeviceGet(&device,dev);

	createCapsPPtr<<<1,1>>>(g_devCount);
	addCaps<<<1,1>>>(dev, *g_devCaps[dev]);
	return res;
}

void ExecCaps::findDevCaps() {
	//flprintf("%s enter\n","ExecCaps::findDevCaps");
	prlocf("ExecCaps::findDevCaps enter\n");
	checkCudaError(cudaGetDeviceCount(&g_devCount));

	//ExecCaps* g_devCaps[];
	g_devCaps = (ExecCaps**) malloc(g_devCount* sizeof(ExecCaps*));
	flprintf("ExecCaps::findDevCaps gpuCount %d, malloced-> %d\n",g_devCount,g_devCount* sizeof(ExecCaps*) );
	//outln("gpuCount " << g_devCount << " mallocd " <<(g_devCount* sizeof(ExecCaps*)));
	for(int i = 0; i < g_devCount; i++) {
		ExecCaps* cap = new ExecCaps();
		flprintf("created cap %p\n", cap);
		g_devCaps[i] = cap;
		ExecCaps::getExecCaps(*cap, i);
		outln("adding " << i << "\n" << cap->toString() << "\n");
	}
}

