#include "MemMgr.h"
#include "Tiler.h"

template<typename T> __host__ __device__ cudaError_t MemMgr<T>::allocDevice(
		T** pD_elements, CuMatrix<T>& mat, uint size) {

	if(!pD_elements || !size) {
		return cudaErrorMemoryAllocation;
	}
	if(*pD_elements != null) {
		if(checkDebug(debugMem))printf("pointer was already pointing! %u\n" , *pD_elements);
		// what? todo who uses dis?
		if(mat.tiler.rowTiles == 0 && !(checkValid(*pD_elements ) && checkValid(*pD_elements  + size / sizeof(T) - 1))) {
			return cudaErrorMemoryAllocation;
		}
	} else {
		size_t max = ExecCaps::currCaps()->maxReasonable();
		if(size > max) {
			uint tileM = mat.m;
			uint tileP = mat.p;
			if(tileM > tileP) {
				tileM = DIV_UP(max,tileP);
			} else {
				tileP = DIV_UP(max,tileM);
			}
			mat.tiler.rowTiles = DIV_UP(size,(tileM * tileP));
			flprintf("size %lu maxComfy %lu --> tileCount %d\n", size,max,mat.tiler.nTiles);
		} else {
			mat.tiler.rowTiles = 1;

			checkCudaError(cudaMalloc( (void**)pD_elements,size));
			dBytesAllocated += size;
			currDeviceB += size;
			if(checkDebug(debugMem)) {
				flprintf("allocDevice: success for %p, dBytesAllocated now at %d\n", *pD_elements, dBytesAllocated);
				//b_util::usedDmem();
			}
			dBuffers.insert(dBuffers.end(), pair<T*, int>(*pD_elements, 1));
		}
		mat.tiler.colTiles = 1;
	}
	return cudaSuccess;
}
