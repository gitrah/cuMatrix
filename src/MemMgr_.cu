#include "MemMgr.h"
#include "Tiler.h"

template __host__ __device__ bool MemMgr<float>::checkValid(const float* addr, const char* msg);
template __host__ __device__ bool MemMgr<double>::checkValid(const double* addr, const char* msg);
template __host__ __device__ bool MemMgr<ulong>::checkValid(const ulong* addr, const char* msg);
template __host__ __device__ bool MemMgr<long>::checkValid(const long* addr, const char* msg);
template __host__ __device__ bool MemMgr<uint>::checkValid(const uint* addr, const char* msg);
template __host__ __device__ bool MemMgr<int>::checkValid(const int* addr, const char* msg);

template<typename T> __host__ __device__ bool MemMgr<T>::checkValid(const T* addr, const char* msg) {

		if(checkDebug(debugMem | debugCons | debugCheckValid)) {

			ExecCaps* currCaps;
			cherr(ExecCaps::currCaps(&currCaps));
			if (currCaps->deviceProp.major > 1) {
				struct cudaPointerAttributes ptrAtts;
				//ptrAtts.memoryType = dev ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
				cudaError_t ret = cudaPointerGetAttributes(&ptrAtts, addr);

				if (ret == cudaSuccess) {
/*
					if(checkDebug(debugMem | debugRefcount | debugCheckValid | debugCons )) {
						outln((msg != null ? msg : "") << " cudaSuccess " << addr << " points to " << ( ptrAtts.memoryType == cudaMemoryTypeDevice ? " device " : " host") << " mem" ) ;
					}
*/
					return true;
				} else {
					flprintf(" Imwavlid %p\n", addr);
					b_util::dumpStack();
					checkCudaError(ret);
				}
				//outln((msg != null ? msg : "") << " Imwalid " << addr);
				if(msg != null) {
					flprintf("(%s) Imwavlid %p\n", msg, addr);
				} else {
					flprintf("(null) Imwavlid %p\n", addr);
				}
				b_util::dumpStack();
				checkCudaError(ret);
			}
			return false;
		} else {
			return true;
		}
	}

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
