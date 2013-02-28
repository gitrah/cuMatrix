/*
 * MemMgr.h
 *
 *  Created on: Oct 5, 2012
 *      Author: reid
 */

#ifndef MEMMGR_H_
#define MEMMGR_H_

#include <map>
#include <cuda_runtime_api.h>
#include "util.h"
//#include "Matrix.h"

template<typename T> class CuMatrix;
//template<typename T> struct MatrixD;
template<typename T> class MemMgr {
private:
	bool enabled;
	long hBytesAllocated;
	long hBytesFreed;
	long dBytesAllocated;
	long dBytesFreed;
	long mmHHCopied;
	long memMmHhCopied;
	long currHost;
	long currDevice;
	std::map<std::string, int> hDimCounts;
	std::map<std::string, int> dDimCounts;
	std::map<std::string, int> hDimFreeCounts;
	std::map<std::string, int> dDimFreeCounts;
	std::map<T*, std::string> ptrDims;
	std::map<T*, int> hBuffers;
	std::map<T*, int> dBuffers;
	std::map<T*, long> hSizes;
	std::map<T*, long> dSizes;
	std::map<CuMatrix<T>*, T*> mhBuffers;
	std::map<CuMatrix<T>*, T*> mdBuffers;

	std::map<const T*,const T*> parents;

public:
	__host__ MemMgr();
	__host__ ~MemMgr();
	__host__ string stats();
	__host__ void enable();
	__host__ cudaError_t allocHost(CuMatrix<T>& mat, T* source = 0);
	__host__ void dumpUsage();
	__host__ void addHostDims( const CuMatrix<T>& mat) ;
	__host__ void addHost(  CuMatrix<T>& mat);
	__host__ void addSubmatrix( const CuMatrix<T>& mat, const CuMatrix<T>& container);
	__host__ bool submatrixQ( const CuMatrix<T>& mat);
	__host__ cudaError_t getHost( CuMatrix<T>& mat);
	__host__ int freeHost(	CuMatrix<T>& mat);
	__host__ void addDeviceDims( const CuMatrix<T>& mat);
	__host__ cudaError_t allocDevice(CuMatrix<T>& mat);
	__host__ void addDevice(  CuMatrix<T>& mat);
	__host__ cudaError_t getDevice(	 CuMatrix<T>& mat);
	__host__ int freeDevice(CuMatrix<T>& mat);
	__host__ void dumpLeftovers();

	__host__ static bool checkValid(const T* addr);

};

extern template class MemMgr<float>;
extern template class MemMgr<double>;

#endif /* MEMMGR_H_ */
