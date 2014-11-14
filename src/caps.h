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

#define MAX_GPUS 	10
enum DataTypes {
	dtFloat, dtDouble, dtUlong, dtUint, dtInt, dtLast
};

template <typename T> int getTypeEnum();
template <typename T> const char* getTypeName();
#define MAX_TYPES 	10

#define FirstThread  if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0 && threadIdx.z == 0 && blockIdx.z == 0)
extern int MaxThreads;
extern __device__ int d_MaxThreads;
extern int MaxBlocks;
extern __device__ int d_MaxBlocks;

extern unsigned long totalThreads;
extern unsigned long totalElements;

std::string toString(dim3& dim);
class capsError {};
class smemExceeded : public capsError {};
__global__ void freeDevSideDevCaps();
//
// FIXME something is deleting cap[0] twice...
//
struct ExecCaps {
		bool copy;
		int devNumber;
		uint smCount;
		uint cores;
		uint thrdPerSM;
		uint thrdPerBlock;
		uint regsPerBlock;
		uint regsPerSM;
		uint memSharedPerBlock;
		uint memConst;
		dim3 maxGrid;
		dim3 maxBlock;
		uint warpSize;
		size_t alignment;
		bool dynamicPism;
		//cudaStream_t stream;
		cudaDeviceProp deviceProp;

		uint totalThreads();
		uint minBlocks(int);
		std::string toString();
		__host__ __device__ ExecCaps() : copy(false), devNumber(-1), smCount(0), cores(0), thrdPerSM(0) , thrdPerBlock(0),
				regsPerBlock(0),regsPerSM(0),memSharedPerBlock(0), memConst(0),maxGrid( dim3(0,0,0)),maxBlock( dim3(0,0,0)),
						warpSize(0), alignment(0), dynamicPism(false)  {
			printf("ExecCaps::ExecCaps() this %p\n",this);

		}
		__host__ __device__ ExecCaps(const ExecCaps& o) : copy(true), devNumber(o.devNumber), smCount(o.smCount), cores(o.cores), thrdPerSM(o.thrdPerSM) , thrdPerBlock(o.thrdPerBlock),
				regsPerBlock(o.regsPerBlock),regsPerSM(o.regsPerSM),memSharedPerBlock(o.memSharedPerBlock), memConst(o.memConst),maxGrid(o.maxGrid),maxBlock(o.maxBlock),
				warpSize(o.warpSize), alignment(o.alignment), dynamicPism(o.dynamicPism), deviceProp(o.deviceProp) {
#ifndef __CUDA_ARCH__
			printf("copied ExecCap (%p) to this %p\n",&o,this);
#else
			printf("copied ExecCap (%p) to device this %p\n",&o,this);
#endif
		}
		__host__ __device__ ~ExecCaps();

		template <typename T> __host__ __device__ uint maxTsPerBlock() {
			return memSharedPerBlock/sizeof(T);
		}
		static void getExecCaps(ExecCaps& execCaps, int dev = 0);

		static int currDev();
		static cudaError_t setDevice(int dev);
		static __host__ void findDevCaps();
		static __host__ void freeDevCaps();
		static __host__ __device__ cudaError_t currCaps(ExecCaps** caps, int dev = 0);
		static __host__ __device__ ExecCaps* currCaps(int dev = 0);
		static __host__ __device__ int maxThreads();
		static __host__ __device__ int maxBlocks();

	//	static __host__ void initStreamRefs();
	//	static __host__ void addStream(const ExecCaps& execCaps);
	//	static __host__ void releaseStream(ExecCaps& execCaps);
		//static ulong streams[MAX_GPUS];
		//static int refCount[MAX_GPUS];

};

template <typename T> inline __host__ __device__ uint maxH(const ExecCaps& ecaps, uint specW, bool usingSmem = false) {
		if(usingSmem) {
		return MIN(
			MIN (ecaps.maxBlock.y,
					ecaps.memSharedPerBlock/(specW*sizeof(T))),
			(ecaps.thrdPerBlock  )/ specW);
		}else {
			return MIN(ecaps.maxBlock.y, ecaps.thrdPerBlock  / specW);
		}
}

 __host__ __device__ void getReductionExecContext(uint &blocks, uint &threads, ulong nP,
		int maxBlocks = ExecCaps::maxBlocks(), int maxThreads = ExecCaps::maxThreads());

__global__ void createCapsPPtr(int devCount);
__global__ void addCaps(int dev, ExecCaps caps);

//extern map<int, ExecCaps*> g_devCaps;
extern int g_devCount;
//extern __constant__ int gd_devCount;

extern ExecCaps** g_devCaps;
extern __device__ ExecCaps** gd_devCaps;
extern __constant__ char d_devCapBytes[10*sizeof(ExecCaps)];

struct KernelCaps {
		dim3 grid;
		dim3 block;
		size_t dynSharedMemPerBlock;
		cudaStream_t stream;

		static KernelCaps forArray(ExecCaps & caps, dim3& array);
};


//////////////////////////////////



#endif /* CAPS_H_ */
