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

using std::reference_wrapper;

template<typename T> class CuMatrix;
//template<typename T> struct MatrixD;
template<typename T> class Tiler;
extern bool pinnedQ;

#define mCheckValid(ptr) 	MemMgr<T>::checkValid(ptr, __func__ )

template<typename T> class MemMgr {
private:
	long hBytesAllocated;
	long hBytesFreed;
	long dBytesAllocated;
	long dBytesFreed;
	long mmHHCopied;
	long memMmHhCopied;
	long currHostB;
	long currDeviceB;

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
	//CMap<CuMatrix<T>*, T*> mdBuffers;

	CMap<const T*,const T*> parents;
	size_t page_size;
public:
	__host__ MemMgr();
	__host__ ~MemMgr();
	__host__ string stats();
	__host__ int refCount( T* els);
	__host__ std::pair<int,int> refCounts(const CuMatrix<T>& mat);
	__host__ cudaError_t allocHost(CuMatrix<T>& mat, T* source = null);
	__host__ void dumpUsage();
	__host__ void addHostDims( const CuMatrix<T>& mat) ;
	__host__ void addHost(  CuMatrix<T>& mat);
	__host__ void addSubmatrix( const CuMatrix<T>& mat, const CuMatrix<T>& container);
	__host__ bool submatrixQ( const CuMatrix<T>& mat);
	__host__ cudaError_t getHost( CuMatrix<T>& mat);
	__host__ int freeHost(	CuMatrix<T>& mat);
	__host__ void addDeviceDims( const CuMatrix<T>& mat,ulong tileSize = 0);
	__host__ cudaError_t allocDevice(CuMatrix<T>& mat,int device = ExecCaps::currDev(), ulong tileSize = 0);
	__host__ __device__ cudaError_t allocDevice(T** pElements, CuMatrix<T>& mat, uint size );
	__host__ void addTiles( const Tiler<T>* tiler);
	__host__ int freeTiles(CuMatrix<T>& mat);
	__host__ void dumpLeftovers();

	__host__ int migrate(int dev, CuMatrix<T>& m);
	__host__ int meet(const vector<CuMatrix<T> >& mats);
	__host__ void copy(int dev, CuMatrix<T>& m);
	__host__ pair<size_t,size_t> migrate(int dev, vector< reference_wrapper<CuMatrix<T>>>  ms);

	__host__ static bool checkValid(const T* addr, const char* msg = null);

	__host__ static void checkRange(const T* addr, int endIdx, const char* msg = null);


};

extern template class MemMgr<float>;
extern template class MemMgr<double>;
extern template class MemMgr<ulong>;

#endif /* MEMMGR_H_ */
