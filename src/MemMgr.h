/*
 * MemMgr.h
 *
 *  Created on: Oct 5, 2012
 *      Author: reid
 */

#ifndef MEMMGR_H_
#define MEMMGR_H_

#include <cuda_runtime_api.h>
#include "util.h"
#include "CMap.h"
//#include "Matrix.h"

template<typename T> class CuMatrix;
//template<typename T> struct MatrixD;
template<typename T> class MemMgr {
private:
	long hBytesAllocated;
	long hBytesFreed;
	long dBytesAllocated;
	long dBytesFreed;
	long mmHHCopied;
	long memMmHhCopied;
	long currHost;
	long currDevice;

	CMap<std::string, int> hDimCounts;
	CMap<std::string, int> dDimCounts;
	CMap<std::string, int> hDimFreeCounts;
	CMap<std::string, int> dDimFreeCounts;
	CMap<T*, std::string> ptrDims;
	CMap<T*, int> hBuffers;
	CMap<T*, int> dBuffers;
	CMap<T*, long> hSizes;
	CMap<T*, long> dSizes;
	CMap<CuMatrix<T>*, T*> mhBuffers;
	CMap<CuMatrix<T>*, T*> mdBuffers;

	CMap<const T*,const T*> parents;

public:
	__host__ MemMgr();
	__host__ ~MemMgr();
	__host__ string stats();
	__host__ cudaError_t allocHost(CuMatrix<T>& mat, T* source = null);
	__host__ void dumpUsage();
	__host__ void addHostDims( const CuMatrix<T>& mat) ;
	__host__ void addHost(  CuMatrix<T>& mat);
	__host__ void addSubmatrix( const CuMatrix<T>& mat, const CuMatrix<T>& container);
	__host__ bool submatrixQ( const CuMatrix<T>& mat);
	__host__ cudaError_t getHost( CuMatrix<T>& mat);
	__host__ int freeHost(	CuMatrix<T>& mat);
	__host__ void addDeviceDims( const CuMatrix<T>& mat);
	__host__ cudaError_t allocDevice(CuMatrix<T>& mat);
	__host__ __device__ cudaError_t allocDevice(T** pElements, uint size );
	__host__ void addDevice(  CuMatrix<T>& mat);
	__host__ cudaError_t getDevice(	 CuMatrix<T>& mat);
	__host__ int freeDevice(CuMatrix<T>& mat);
	__host__ void dumpLeftovers();
	__host__ void migrate(int dev, CuMatrix<T>& m);
	__host__ void locate(int dev, CuMatrix<T>& m);
	__host__ static bool checkValid(const T* addr);

};

extern template class MemMgr<float>;
extern template class MemMgr<double>;
extern template class MemMgr<ulong>;

#endif /* MEMMGR_H_ */
