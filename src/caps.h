/*
 * caps.h
 *
 *  Created on: Aug 3, 2012
 *      Author: reid
 */

#ifndef CAPS_H_
#define CAPS_H_
#include <cuda_runtime_api.h>
#include <string>
#include "util.h"

extern cudaDeviceProp deviceProp;
extern unsigned long totalThreads;
extern unsigned long totalElements;

std::string toString(dim3& dim);
class capsError {};
class smemExceeded : public capsError {};

struct ExecCaps {
		uint smCount;
		uint thrdPerSM;
		uint thrdPerBlock;
		uint memSharedPerBlock;
		uint memConst;
		dim3 maxGrid;
		dim3 maxBlock;
		uint warpSize;
		size_t alignment;
		bool dynamicPism;
		uint totalThreads();
		uint minBlocks(int);
		string toString();

		template <typename T> uint maxTsPerBlock() {
			return memSharedPerBlock/sizeof(T);
		}
		static void getExecCaps(ExecCaps& ecaps, int dev = 0);
};

template <typename T> inline uint maxH(ExecCaps& ecaps, uint specW) {
		return MIN(
			MIN (ecaps.maxBlock.y,
					ecaps.memSharedPerBlock/(specW*sizeof(T))),
			(ecaps.thrdPerBlock -1 )/ specW);
}

extern ExecCaps caps;

struct KernelCaps {
		dim3 grid;
		dim3 block;
		size_t dynSharedMemPerBlock;
		cudaStream_t stream;

		static KernelCaps forArray(ExecCaps & caps, dim3& array);
};


//////////////////////////////////



#endif /* CAPS_H_ */
