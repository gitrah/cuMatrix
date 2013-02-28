/*
 * caps.cc
 *
 *  Created on: Aug 3, 2012
 *      Author: reid
 */


#include <stdio.h>
#include "caps.h"
#include "debug.h"
#include <iostream>
#include <sstream>

cudaDeviceProp deviceProp;
ExecCaps caps;

unsigned long totalThreads;
unsigned long totalElements;

uint ExecCaps::totalThreads() { return (smCount * thrdPerSM); }

uint ExecCaps::minBlocks(int threadCount) { return (threadCount / thrdPerBlock); };

string ExecCaps::toString()
{
	stringstream ss (stringstream::in | stringstream::out);
	ss << "sm count: " << smCount << endl;
	ss << "thrdPerSM: " << thrdPerSM << endl;
	ss << "thrdPerBlock: " << thrdPerBlock << endl;
	ss << "memSharedPerBlock: " << memSharedPerBlock << endl;
	ss << "memConst: " << memConst << endl;
	ss << "maxGrid: (" << maxGrid.x << ", " << maxGrid.y << ", " << maxGrid.z << ")" << std::endl;
	ss << "maxBlock: (" << maxBlock.x << ", " << maxBlock.y << ", " << maxBlock.z << ")" << std::endl;

	return (ss.str());
}


void ExecCaps::getExecCaps(ExecCaps& caps, int dev)
{
    cudaSetDevice(dev);
    cudaGetDeviceProperties(&deviceProp, dev);

    caps.smCount = deviceProp.multiProcessorCount;
    caps.maxBlock.x = deviceProp.maxThreadsDim[0];
    caps.maxBlock.y = deviceProp.maxThreadsDim[1];
    caps.maxBlock.z = deviceProp.maxThreadsDim[2];

    caps.maxGrid.x = deviceProp.maxGridSize[0];
    caps.maxGrid.y = deviceProp.maxGridSize[1];
    caps.maxGrid.z = deviceProp.maxGridSize[2];
    caps.alignment = deviceProp.textureAlignment;
    caps.thrdPerSM = deviceProp.maxThreadsPerMultiProcessor;
    caps.thrdPerBlock = deviceProp.maxThreadsPerBlock;
    caps.memSharedPerBlock= deviceProp.sharedMemPerBlock;
    caps.memConst = deviceProp.totalConstMem;
    caps.warpSize = deviceProp.warpSize;
    caps.dynamicPism = deviceProp.major + .1 * deviceProp.minor >= 3.5;
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


