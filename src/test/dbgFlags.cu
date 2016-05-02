#include <assert.h>
#include <iostream>
#include <helper_cuda.h>
#include <sstream>
#include "../caps.h"

#define _flprintf( format, ...) printf ( "[d]%s:%d %s " format, __FILE__, __LINE__, __func__,  __VA_ARGS__)
#define _prlocf(exp) 	printf( "[d]" __FILE__ "(%d): " exp, __LINE__)
__constant__ uint _debugFlags;
uint h_debugFlags;

//#define _debugFlags 		(1 << 2)
#define Giga (1024*1024*1024)
#define tOrF(exp) ( (exp)? "true" : "false")
#define Mega (1024*1024)
#define Kilo (1024)

using std::string;
using std::stringstream;
using std::cout;

inline __host__ __device__ bool _checkDebug(uint flags) {
#ifndef __CUDA_ARCH__
	return h_debugFlags & flags;
#else
	//#ifdef CuMatrix_DebugBuild
		return _debugFlags & flags;
	//#else
	//	return false;
	//#endif
#endif
}

void _setCurrGpuDebugFlags(uint flags, bool orThem, bool andThem,  cudaStream_t stream ) {

	uint curr = flags;
	if(orThem) {
		_prlocf("copying DebugFlag fr device for or'n...\n");
		checkCudaErrors(cudaMemcpyFromSymbol(&curr, _debugFlags,sizeof(uint)));
		curr |= flags;
	} else if(andThem) {
		_prlocf("copying DebugFlag fr device fur and'n...\n");
		checkCudaErrors(cudaMemcpyFromSymbol(&curr, _debugFlags,sizeof(uint)));
		curr &= flags;
	}
	_prlocf("copying DebugFlag to device...\n");
	checkCudaErrors(cudaMemcpyToSymbolAsync(_debugFlags,&curr,sizeof(uint),0,  cudaMemcpyHostToDevice, stream));
	_prlocf("copied to device\n");
	h_debugFlags = curr;
}


void _setAllGpuDebugFlags(uint flags, bool orThem, bool andThem ) {
	_prlocf("_setAllGpuDebugFlags entre...\n");
	int devCount, currDev;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	checkCudaErrors(cudaGetDevice(&currDev));
	_flprintf("device count %d\n",devCount);
	_flprintf("curr device %d\n",currDev);

	cudaStream_t *streams = (cudaStream_t *) malloc(
			devCount * sizeof(cudaStream_t));

	for(int i = 0; i < devCount;i++) {

		if(strstr("gtx980m", "750 Ti")) {
			_prlocf("not skipping sluggish 750 ti\n");
			//continue;
		}
		_flprintf("setting DbugFlags for device %s %d\n","gtx980m",i);

		ExecCaps_visitDevice(i);
		_flprintf("set device %d\n",i);
		checkCudaErrors(cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking));
		_prlocf("create stream\n");
		_setCurrGpuDebugFlags(flags,orThem,andThem, streams[i]);
		_prlocf("set gpu dbg flags\n");
	}

	for(int i = 0; i < devCount; i++) {
		_flprintf("synching stream for dev %d\n",i);
		checkCudaErrors(cudaStreamSynchronize(streams[i]));
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	}

	ExecCaps_setDevice(currDev);
}
__host__ __device__ void _expNotation(char* buff, long val) {
	double factor = 1.;
	if (val >= Giga) {
		factor = 1. / Giga;
		sprintf(buff, "%2.3gGb", val * factor);
	} else if (val >= Mega) {
		factor = 1. / Mega;
		sprintf(buff, "%2.3gMb", val * factor);
	} else if (val >= Kilo) {
		factor = 1. / Kilo;
		sprintf(buff, "%2.3gKb", val * factor);
	} else {
		sprintf(buff, "%2.3gb", val * factor);
	}
}

string _expNotation(long val) {
	char buff[256];
	_expNotation(buff, val);
	stringstream ss;
	ss << buff;
	return ss.str();
}

__host__ void _allGpuMem(size_t* free, size_t* total) {
    int orgDev;
    checkCudaErrors(cudaGetDevice(&orgDev));
    *free=0;
    *total=0;
    size_t lFree = 0, lTotal = 0;
	int devCnt;
	char buff[20];
	checkCudaErrors(cudaGetDeviceCount(&devCnt));
	cout << "  ";
	assert(false);
	for(int i = 0; i < devCnt;i++) {
		ExecCaps_visitDevice(i);
		checkCudaErrors(cudaMemGetInfo(&lFree, &lTotal));
		*free += lFree;
		*total += lTotal;
		sprintf(buff, " (%.2f%% used)", 100 * (1 - lFree * 1. / lTotal));
		if(_checkDebug(_debugFlags)) cout << "[" << i <<":  " <<  _expNotation(lFree) << " free /" << _expNotation(lTotal) << buff<< "] ";
	}
	cout << endl;
	ExecCaps_setDevice(orgDev);
}

double _usedMemRatio(bool allDevices) {
	size_t freeMemory, totalMemory;
	if(allDevices)
		_allGpuMem(&freeMemory, &totalMemory);
	else {
		assert(false);
		cout << "calling cudaMemGetInfo\n";
		cudaMemGetInfo(&freeMemory, &totalMemory);
		cout << "callied cudaMemGetInfo\n";
	}
	int currDev;
	checkCudaErrors(cudaGetDevice(&currDev));
	if (_debugFlags) {
		if(allDevices )
			cout << "\tallDev freeMemory " << freeMemory << ", total " << totalMemory << "\n";
		else
			cout << "\tdev " << currDev<< " freeMemory " << freeMemory << ", total " << totalMemory << "\n";
	}
	return 100 * (1 - freeMemory * 1. / totalMemory);
}

void _usedDmem(bool allDevices) {
	cout << "Memory " << _usedMemRatio(allDevices) << "% used\n";
}


int dmain(int argc, const char **argv) {
	uint localDbgFlags = 1 << 2 | 1 << 5;
	_flprintf("localDbgFlags %d\n",localDbgFlags);
	_usedDmem(true);
	_setAllGpuDebugFlags(localDbgFlags,false,false);
	_prlocf("set debug flags\n");
	_usedDmem(true);
}
