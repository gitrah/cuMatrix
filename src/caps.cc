/*
 * caps.cc
 *
 *  Created on: Aug 3, 2012
 *      Author: reid
 */


#include <stdio.h>
#include "caps.h"
#include "debug.h"
#include "util.h"
#include <iostream>
#include <sstream>
#include "MatrixExceptions.h"

ExecCaps** g_devCaps;

unsigned long totalThreads;
unsigned long totalElements;
int MaxThreads = 512;
int MaxBlocks = 128;
int ExecCaps::deviceCounter = 0;

template <> int getTypeEnum<float>() {
	return dtFloat;
}
template <> int getTypeEnum<double>() {
	return dtDouble;
}
template <> int getTypeEnum<long>() {
	return dtLong;
}
template <> int getTypeEnum<ulong>() {
	return dtUlong;
}
template <> int getTypeEnum<int>() {
	return dtInt;
}
template <> int getTypeEnum<uint>() {
	return dtUint;
}


template <> const char* getTypeName<float>() {
	return "dtFloat";
}
template <> const char* getTypeName<double>() {
	return "dtDouble";
}
template <> const char* getTypeName<ulong>() {
	return "dtUlong";
}
template <> const char* getTypeName<long>() {
	return "dtLong";
}
template <> const char* getTypeName<int>() {
	return "dtInt";
}
template <> const char* getTypeName<uint>() {
	return "dtUint";
}

//ulong ExecCaps::streams[MAX_GPUS];
//int ExecCaps::refCount[MAX_GPUS];

uint ExecCaps::totalThreads() { return (smCount * thrdPerSM); }

uint ExecCaps::minBlocks(int threadCount) { return (threadCount / thrdPerBlock); };

string ExecCaps::toString()
{
	stringstream ss (stringstream::in | stringstream::out);
	ss << "ExecCaps instance:  "<< this << " for device # " <<  devNumber << "( " << deviceProp.name << " ) "<< endl;
	ss << "dev caps: " << ( deviceProp.major + .1 * deviceProp.minor) <<  "; dynPism " << tOrF(dynamicPism) << endl;
	ss << "sm count: " << smCount << "  (" << (cores/smCount) << " * " << smCount << " = " << cores << " cores)\n";
	ss << "thrdPerSM: " << thrdPerSM << endl;
	ss << "thrdPerBlock: " << thrdPerBlock << endl;
	ss << "regsPerBlock: " << regsPerBlock << endl;
	ss << "memSharedPerBlock: " << memSharedPerBlock << endl;
	ss << "memConst: " << memConst << endl;
	ss << "maxGrid: (" << maxGrid.x << ", " << maxGrid.y << ", " << maxGrid.z << ")" << endl;
	ss << "maxBlock: (" << maxBlock.x << ", " << maxBlock.y << ", " << maxBlock.z << ")" << endl;
	ss << "regsPerBlock: " << regsPerBlock  << endl;
	ss << "regsPerSM: " << regsPerSM << endl;
	ss << "clockRate: " << deviceProp.clockRate  << endl;
	ss << "totalConstMem: " << deviceProp.totalConstMem  << endl;
	ss << "deviceOverlap: " << deviceProp.deviceOverlap  << endl;
	ss << "concurrentKernels: " << deviceProp.concurrentKernels  << endl;
	ss << "asyncEngineCount: " << deviceProp.asyncEngineCount  << endl;
	ss << "unifiedAddressing: " << deviceProp.unifiedAddressing  << endl;
	ss << "maxThreadsPerMultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor  << endl;
	ss << "streamPrioritiesSupported: " << deviceProp.streamPrioritiesSupported  << endl;
	ss << "sharedMemPerMultiprocessor: " << deviceProp.sharedMemPerMultiprocessor  << endl;
	ss << "regsPerMultiprocessor: " << deviceProp.regsPerMultiprocessor  << endl;
	ss << "managedMemory: " << deviceProp.managedMemory  << endl;

	//ss << "stream: (" << stream << ")" << endl;
	return (ss.str());
}




void ExecCaps::getExecCaps(ExecCaps& execCaps, int dev)
{
	int orgDev = currDev();
	if(orgDev != dev) {
		ExecCaps_visitDevice(dev);
	}
    execCaps.devNumber = dev;
    checkCudaError(cudaGetDeviceProperties(&execCaps.deviceProp, dev));
    if(dev < MAX_GPUS) {
    	gpuNames[dev] = execCaps.deviceProp.name;
    }
   // checkCudaError(cudaStreamCreate(&execCaps.stream));
    //outln("getExecCaps for dev " << dev << " created stream " << execCaps.stream);
    execCaps.smCount = execCaps.deviceProp.multiProcessorCount;
    execCaps.maxBlock.x = execCaps.deviceProp.maxThreadsDim[0];
    execCaps.maxBlock.y = execCaps.deviceProp.maxThreadsDim[1];
    execCaps.maxBlock.z = execCaps.deviceProp.maxThreadsDim[2];

    execCaps.maxGrid.x = execCaps.deviceProp.maxGridSize[0];
    execCaps.maxGrid.y = execCaps.deviceProp.maxGridSize[1];
    execCaps.maxGrid.z = execCaps.deviceProp.maxGridSize[2];
    execCaps.alignment = execCaps.deviceProp.textureAlignment;
    execCaps.thrdPerSM = execCaps.deviceProp.maxThreadsPerMultiProcessor;
    execCaps.thrdPerBlock = execCaps.deviceProp.maxThreadsPerBlock;
    execCaps.regsPerBlock = execCaps.deviceProp.regsPerBlock;
    execCaps.regsPerSM = execCaps.deviceProp.regsPerMultiprocessor;
    execCaps.memSharedPerBlock= execCaps.deviceProp.sharedMemPerBlock;
    execCaps.memConst = execCaps.deviceProp.totalConstMem;
    execCaps.warpSize = execCaps.deviceProp.warpSize;
    execCaps.dynamicPism = execCaps.deviceProp.major + .1 * execCaps.deviceProp.minor >= 3.5;
    execCaps.cores = execCaps.smCount * _ConvertSMVer2Cores(execCaps.deviceProp.major, execCaps.deviceProp.minor);

    if(orgDev != dev) {
    	ExecCaps_restoreDevice(dev);
    }

}

KernelCaps KernelCaps::forArray(ExecCaps& caps, dim3& arrayDim) {
	KernelCaps kcaps;
	kcaps.dynSharedMemPerBlock = 0;
	kcaps.stream = 0;
	outln("here");

	if(arrayDim.z == 0) {
		outln("no z");
		kcaps.block.z = 0;
		if(arrayDim.y == 0) {
			outln("no y");
			kcaps.block.y = 0;
			const uint len = arrayDim.x;
			outln("have 1d array of len " << len);
			if(len < caps.totalThreads()) {
				printf("exceeds max thread count %d by %d\n", caps.totalThreads(), (len-caps.totalThreads()));
			}
			outln("before minBlocks " << caps.thrdPerBlock);
			const uint blocks = caps.minBlocks(len);
			outln("after minBlocks");
			outln("-> blocks " << blocks);
			if(blocks > caps.maxBlock.x) {
				printf("at %d threads/block, blocks %d exceeds max block.x dim %d by %d\n", caps.thrdPerBlock,
						blocks, caps.maxBlock.x, (blocks-caps.maxBlock.x));
			} else {
				kcaps.block.x = blocks;
			}
		}
	}

	return (kcaps);
}


/*
void ExecCaps::addStream(const ExecCaps& caps) {
	const cudaStream_t& str = caps.stream;

	if(!str) {
		outln(&caps << " doesn't have a stream");
		return;
	}
	outln(&caps << " adding stream " << str);
	ulong thisSt = (ulong) str;
	for(int i = 0; i < MAX_GPUS; i++) {
		ulong& st = streams[i];
		if(st == thisSt || st == NULL) {
			st = thisSt;
			refCount[i] += 1;
			if(refCount[i] == 1) {
				outln(&caps << " adding stream " << str << " at index " << i );
			} else {
				outln(&caps << " reffing stream " << str << " refCount now at " << refCount[i] );
			}
			return;
		}
	}
	dthrow(CapsException());
}

void ExecCaps::releaseStream(ExecCaps& caps) {
	cudaStream_t& str = caps.stream;
	ulong thisSt = (ulong) str;
	outln("ExecCaps::releaseStream str " << str);
	for(int i = 0; i < MAX_GPUS; i++) {
		ulong& st = streams[i];
		if(st == thisSt ) {
			refCount[i] -= 1;
			if(refCount[i] < 1) {
				outln("ExecCaps::releaseStream caps " << &caps << " destroying stream " << str << " refCount " << refCount[i] );
				st = 0;
				caps.stream = NULL;
				outln("it dead");

			} else {
				outln("ExecCaps::releaseStream caps " << &caps << " dereffing stream " << str << " refCount " << refCount[i] );
			}
			return;
		}
	}
	outln("couldn't find " << str);
	dthrow(CapsException());
}
*/

