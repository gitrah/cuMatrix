#include "MemMgr.h"

template<typename T> __host__ __device__ cudaError_t MemMgr<T>::allocDevice(
		T** pD_elements, uint size) {

	if(!pD_elements || !size) {
		return cudaErrorMemoryAllocation;
	}
	if(*pD_elements != null) {
		if(checkDebug(debugMem))printf("pointer was already pointing! %u\n" , *pD_elements);
		if(!(checkValid(*pD_elements ) && checkValid(*pD_elements  + size / sizeof(T) - 1))) {
			return cudaErrorMemoryAllocation;
		}
	} else {
		checkCudaError(cudaMalloc( (void**)pD_elements,size));
		dBytesAllocated += size;
		currDevice += size;
		if(checkDebug(debugMem)) {
			printf("allocDevice: success for %u, dBytesAllocated now at %d\n", *pD_elements, dBytesAllocated);
			//b_util::usedDmem();
		}
		dBuffers.insert(dBuffers.end(), pair<T*, int>(*pD_elements, 1));
	}
	return cudaSuccess;
}
