/*
 * MemMgr.cc
 *
 *  Created on: Sep 20, 2012
 *      Author: reid
 */

#include "MemMgr.h"
#include "CuMatrix.h"
#include "debug.h"
#include "MatrixExceptions.h"
#include <helper_cuda.h>
extern bool debugMem;
extern cudaDeviceProp deviceProp;
//#define MMGRTMPLT

template class MemMgr<float>;
template class MemMgr<double>;

template <typename K, typename V> void dump(string name, map<K, V>& map) {
	typedef typename std::map<K, V>::iterator iterator;
	iterator it = map.begin();
	cout << name.c_str() << endl;
	while(it != map.end()) {
		cout << "\t" << (*it).first << " -> " << (*it).second << endl;
		it++;
	}
}

template<typename T> __host__ MemMgr<T>::MemMgr() :
		enabled(false),
		hBytesAllocated(0l),
		hBytesFreed(0l),
		dBytesAllocated(0l),
		dBytesFreed(0l),
		mmHHCopied(0l),
		memMmHhCopied(0l),
		currHost(0l),
		currDevice(0l)
{
	outln("MemMgr<T>::MemMgr() this " << this << ", sizeof(T) " << sizeof(T)  );
}

template<typename T> __host__ MemMgr<T>::~MemMgr() {
	outln("~MemMgr (" << this << ", sizeof<T> " << sizeof(T) << " ) with " << hBuffers.size() << " in hBuffers");
	outln("dDimCounts " << dDimCounts.size());
    if(!enabled) {
		outln("exiting; this MemMgr is not enabled");
		return;
	}
    if(ptrDims.size() > 0) {
    	outln("leftovers!");
    	dump<T*,std::string>("ptrDims", ptrDims);
    }
	typedef typename std::map<T*, int>::iterator iterator;
	typedef typename std::map<T*, long>::iterator sizeit;
	typedef typename std::map<const T*,const  T*>::iterator parentsIt;
	iterator it = hBuffers.begin();
	size_t proced = 0;
	T* mem;
	while (it != hBuffers.end() && proced++ < hBuffers.size()) {

		if((*it).first) {
			if(debugMem)outln("~MemMgr freeing h " << (*it).first << " procing #" << proced);
			sizeit si = hSizes.find((*it).first);
			if(si != hSizes.end()) {
				long amt = (*si).second;
				hBytesFreed += amt;
				currHost -= amt;
				hSizes.erase(si);
			}
			checkCudaError(cudaFreeHost( (*it).first));
		}
		it++;
	}
	hBuffers.clear();
	it = dBuffers.begin();
	while (it != dBuffers.end()) {
		if(((mem = (*it).first) != 0 )&& checkValid(mem)) {
			outln("~MemMgr freeing d " << mem);
			sizeit si = dSizes.find((*it).first);
			if(si != dSizes.end()) {
				long amt = (*si).second;
				dBytesFreed += amt;
				currDevice -= amt;
				dSizes.erase(si);
			}
			checkCudaError(cudaFree( mem));
		}
		it++;
	}
	dBuffers.clear();

	dumpUsage();
	outln("~MemMgr currDevice " << b_util::expNotation(currDevice));
	outln("~MemMgr currHost " << b_util::expNotation(currHost));

	parentsIt pit = parents.begin();
	while (pit != parents.end()) {
		if(((mem = (*it).first) != 0 )&& checkValid(mem)) {
			outln("~MemMgr child " << (*pit).first  << "of container  " << (*pit).second );
		}
		pit++;
	}
	parents.clear();

}

template<typename T> void MemMgr<T>::dumpLeftovers() {
	if(ptrDims.size() > 0) {
		outln("leftovers!");
		dump<T*,std::string>("ptrDims", ptrDims);
	}
}

template<typename T> bool MemMgr<T>::checkValid(const T* addr) {
	if (deviceProp.major > 1) {
		struct cudaPointerAttributes ptrAtts;
		//ptrAtts.memoryType = dev ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
		cudaError_t ret = cudaPointerGetAttributes(&ptrAtts, addr);
		if (ret == cudaSuccess) {
			//outln("checkValid " << addr << " to " << ( ptrAtts.memoryType == cudaMemoryTypeDevice ? " device " : " not device"));
			return true;

		}
		outln("INValid " << addr);
		b_util::dumpStack();
		checkCudaError(ret);
	}
	return false;
}

template<typename T> __host__ void MemMgr<T>::dumpUsage() {
	typedef typename std::map<std::string, int>::iterator siterator;
	outln("\n\nMemMgr::dumpUsage()");
	b_util::announceTime();
	outln( hDimCounts.size() << " distinct dimensions of host allocations; counts:");
	siterator sit = hDimCounts.begin();
	while (sit != hDimCounts.end()) {
		cout << "\t" << (*sit).first.c_str()  << " allocated " << (*sit).second << " times" << endl;
		sit++;
	}

	outln("\n" << dDimCounts.size() << " distinct dimensions of device allocations; counts:");
	sit = dDimCounts.begin();
	while (sit != dDimCounts.end()) {
		cout << "\t" << (*sit).first.c_str() << " d-allocated " << (*sit).second << " times" << endl;
		sit++;
	}

	cout << "Host-- allocated:  " << b_util::expNotation(hBytesAllocated) << "; freed:  " << b_util::expNotation(hBytesFreed) << endl;
	cout << "Device-- allocated:  " << b_util::expNotation(dBytesAllocated) << "; freed:  " << b_util::expNotation(dBytesFreed) << endl;
	cout << "MemMgr hh copies " << mmHHCopied << ", mem " << memMmHhCopied << endl;
}

template<typename T> __host__ void MemMgr<T>::addHostDims( const CuMatrix<T>& mat) {
	typedef typename std::map<std::string, int>::iterator iterator;
	std::string dims = mat.dimsString();
	iterator it = hDimCounts.find(dims);
	int allocCount = 0;
	if (it != hDimCounts.end()) {
		allocCount = ((*it).second + 1) ;
		hDimCounts.erase(it);
		hDimCounts.insert( hDimCounts.end(),
				std::pair<std::string, int>(dims, allocCount));
		if(debugMem)outln( "host dim " << dims << " now has " << allocCount << " allocations");
	} else {
		hDimCounts.insert( hDimCounts.end(),
				std::pair<std::string, int>(dims, 1));
		if(debugMem)outln("host dim " << dims << " gets first allocation");
	}
	if(mat.ownsBuffers)ptrDims.insert(ptrDims.end(), std::pair<T*, std::string>(mat.elements, dims));
	hSizes.insert(hSizes.end(), std::pair<T*, long>(mat.elements, mat.size));
	if(debugMem)b_util::dumpStack();
}

template<typename T> __host__ cudaError_t MemMgr<T>::allocHost(CuMatrix<T>& mat, T* source) {
	if(mat.size > 0) {
		if(mat.elements && checkValid(mat.elements)) {
			outln("already allocated " << mat.elements << " to " << &mat.elements);
			dthrow(hostReallocation());
		}
		cudaError_t res = cudaHostAlloc( (void**)&mat.elements,mat.size,0);
		if(res != cudaSuccess) {
			outln("allocHost: FAILURE: " << mat.toShortString());
			checkCudaError(res);
			return res;
		}
		hBytesAllocated += mat.size;
		currHost += mat.size;
		if(debugMem)outln("allocHost: success for " << mat.toShortString() << " hBytesAllocated now at " << b_util::expNotation(hBytesAllocated));

		hBuffers.insert(hBuffers.end(),std::pair<T*, int>(mat.elements, 1));
		mhBuffers.insert(mhBuffers.end(),
				std::pair<CuMatrix<T>*, T*>(&mat, mat.elements));
		if (source != null) {
			checkCudaError(
					cudaMemcpy(source, mat.elements, mat.size, cudaMemcpyHostToHost));
			mmHHCopied++;
			memMmHhCopied += mat.size;
		}
		if(debugMem)b_util::dumpStack();
		addHostDims(mat);
	}
	return cudaSuccess;
}

template<typename T> __host__ void MemMgr<T>::enable() {
	enabled=true;
}

template<typename T> __host__ string MemMgr<T>::stats() {
	stringstream ss;
	// if verbose ss << "dev " << this->dBytesAllocated << "A/" << this->dBytesFreed << "F; host " << this->hBytesAllocated << "A/" << this->hBytesFreed << "F ";
	ss << "devdelta " << b_util::expNotation(this->dBytesAllocated - this->dBytesFreed) << "; hostdelta " << b_util::expNotation(this->hBytesAllocated - this->hBytesFreed) << " ";
	return ss.str();
}

template<typename T> __host__ void MemMgr<T>::addHost(  CuMatrix<T>& mat) {
	if (mat.elements) {

		typedef typename std::map<T*, int>::iterator iterator;
		iterator it = hBuffers.find(mat.elements);
		int refCount = -1;
		if (it != hBuffers.end()) {
			refCount = ((*it).second + 1) ;
			if(debugMem) {
				outln("addHost: matrix " << mat.toShortString() << " adding " << refCount << " ref for " << mat.elements);
				//dumpStack();
			}
			hBuffers.erase(it);
			hBuffers.insert( hBuffers.end(),
					std::pair<T*, int>(mat.elements, refCount));
		} else {
			if(debugMem) {
				outln("addHost: matrix " << &mat <<
					" found no host ref for " << mat.elements);
				b_util::dumpStack();
			}
		}
		typedef typename std::map<CuMatrix<T>*, T*>::iterator miterator;
		miterator mit = mhBuffers.find(&mat);
		if (mit == mhBuffers.end()) {
			mhBuffers.insert(mhBuffers.end(),
					std::pair<CuMatrix<T>*, T*>(&mat, mat.elements));
		}
	}
}

template<typename T> __host__ void MemMgr<T>::addSubmatrix( const CuMatrix<T>& mat, const CuMatrix<T>& container) {
	parents.insert(parents.end(), std::pair<const T*,const T* >(mat.elements,container.elements));
}

template<typename T> __host__ bool MemMgr<T>::submatrixQ( const CuMatrix<T>& mat) {
	typedef typename std::map<const T*,const T* >::iterator miterator;
	miterator it = parents.find(mat.elements);
	return(it != parents.end());
}

template<typename T> __host__ cudaError_t MemMgr<T>::getHost(  CuMatrix<T>& mat) {
	typedef typename std::map<CuMatrix<T>*, T*>::iterator miterator;
	miterator mit = mhBuffers.find(&mat);
	if (mit != mhBuffers.end()) {
		mat.elements = (*mit).second;
		return cudaSuccess;
	}
	return cudaErrorMemoryAllocation;
}

template<typename T> __host__ int MemMgr<T>::freeHost(CuMatrix<T>& mat) {
	int count = -1;
	if(mat.elements) {
		typedef typename std::map<T*, int>::iterator iterator;
		iterator it = hBuffers.find(mat.elements);
		if (it != hBuffers.end()) {
			count = (*it).second;
			hBuffers.erase(it);
			if (count == 1) {
				if(debugMem)outln("freeHost:  " << mat.toShortString() << " had only 1 ref; FREEING host " << mat.elements);
				typedef typename std::map<T*, long>::iterator sizeit;
				sizeit si = hSizes.find(mat.elements);
				if(si != hSizes.end()) {
					hSizes.erase(si);
					long amt = (*si).second;
					hBytesFreed += amt;
					currHost -= amt;
				}
				ptrDims.erase(mat.elements);
				checkCudaError(cudaFreeHost( mat.elements));
				mat.elements = null;
				count = 0;
			} else {
				if(debugMem)outln("freeHost:  " << mat.toShortString() << " host ( " << mat.elements << ") refcount now at " << (count - 1) );
				hBuffers.insert(hBuffers.end(),
						std::pair<T*, int>(mat.elements, --count));
			}
		} else {
			outln("freeHost: " << mat.toShortString() << " had no hBuffer ref for "<< mat.elements << " [caller " << b_util::caller().c_str() << "] disposition elsewhere" );
			if(debugMem)b_util::dumpStack();
		}
		typedef typename std::map<CuMatrix<T>*, T*>::iterator miterator;
		miterator mit = mhBuffers.find(&mat);
		if (mit != mhBuffers.end()) {
			mhBuffers.erase(mit);
		}
	}
	return count;
}

template<typename T> __host__ void MemMgr<T>::addDeviceDims( const CuMatrix<T>& mat) {
	typedef typename std::map<std::string, int>::iterator iterator;
	std::string dims = mat.dimsString();
	iterator it = dDimCounts.find(dims);
	int allocCount = 0;
	if (it != dDimCounts.end()) {
		allocCount = ((*it).second + 1) ;
		if(debugMem)outln( "dev dim " << dims << " now has " << allocCount << " allocations");
		dDimCounts.erase(it);
		dDimCounts.insert( dDimCounts.end(),
				std::pair<std::string, int>(dims, allocCount));
	} else {
		dDimCounts.insert( dDimCounts.end(),
				std::pair<std::string, int>(dims, 1));
		if(debugMem)outln( "dev dim " << dims << " gets first allocation");
	}
	if(mat.ownsBuffers)ptrDims.insert(ptrDims.end(), std::pair<T*, std::string>(mat.d_elements, dims));
	dSizes.insert(dSizes.end(), std::pair<T*, long>(mat.d_elements, mat.size));
}

template<typename T> std::string dimsString( const DMatrix<T>& mat) {
	std::stringstream sm, sn,ssize,ssout;
	sm << mat.m;
	sn << mat.n;
	ssize << mat.size;
	ssout << sm.str() << "x" << sn.str() << "-" << ssize.str();
	return ssout.str();
}

template<typename T> __host__ cudaError_t MemMgr<T>::allocDevice(
		CuMatrix<T>& mat) {
	dassert(mat.size && mat.ownsBuffers);
	if(mat.d_elements != null) {
		if(debugMem)outln("pointer was already pointing! " << mat.d_elements);
		if(!(checkValid(mat.d_elements ) && checkValid(mat.d_elements  + mat.size / sizeof(T) - 1))) {
			throw alreadyPointingDevice();
		}
	} else {
		checkCudaError(cudaMalloc( (void**)&mat.d_elements,mat.size));
		dBytesAllocated += mat.size;
		currDevice += mat.size;
		if(debugMem) {
			outln("allocDevice: success for " << mat.toShortString() << " dBytesAllocated now at " << b_util::expNotation(dBytesAllocated));
			b_util::usedDmem();
		}
		dBuffers.insert(dBuffers.end(), std::pair<T*, int>(mat.d_elements, 1));
		mdBuffers.insert(mdBuffers.end(),
				std::pair<CuMatrix<T>*, T*>(&mat, mat.d_elements));
		addDeviceDims(mat);
	}
	return cudaSuccess;
}

template<typename T> __host__ void MemMgr<T>::addDevice( CuMatrix<T>& mat) {
	if (mat.d_elements && mat.ownsBuffers) {
		typedef typename std::map<T*, int>::iterator iterator;
		int refs = -1;
		iterator it = dBuffers.find(mat.d_elements);
		if (it != dBuffers.end()) {
			refs = (*it).second + 1;
			if(debugMem)outln("addDevice: matrix " << &mat <<
					" adding " << refs << " ref for " << mat.d_elements);
			dBuffers.erase(it);
			dBuffers.insert(dBuffers.end(),
					std::pair<T*, int>(mat.d_elements, refs));
		} else {
			if(debugMem) outln("addDevice: matrix " << &mat <<
					" found no ref for " << mat.d_elements);
			dBuffers.insert(dBuffers.end(),
					std::pair<T*, int>(mat.d_elements, 1));
		}
		it = dBuffers.find(mat.d_elements);
		if(debugMem)outln(
				"after adding " << mat.d_elements << ", ref now at " << (*it).second);
		typedef typename std::map<CuMatrix<T>*, T*>::iterator miterator;
		miterator mit = mdBuffers.find(&mat);
		if (mit == mdBuffers.end()) {
			mdBuffers.insert(mdBuffers.end(),
					std::pair<CuMatrix<T>*, T*>(&mat, mat.d_elements));
		}
	}
}

template<typename T> __host__ cudaError_t MemMgr<T>::getDevice(  CuMatrix<T>& mat) {
	typedef typename std::map<CuMatrix<T>*, T*>::iterator miterator;
	miterator mit = mdBuffers.find(&mat);
	if (mit != mdBuffers.end()) {
		mat.d_elements = (*mit).second;
		return cudaSuccess;
	}
	return cudaErrorMemoryAllocation;
}

template<typename T> __host__ int MemMgr<T>::freeDevice(CuMatrix<T>& mat) {
	if(debugMem)outln("freeDevice mat " << &mat << " d_elements " << mat.d_elements);
	if(submatrixQ(mat)) {
		outln(mat.toShortString() << " is submatrix");
		return 0;
	}
	typedef typename std::map<T*, int>::iterator iterator;
	iterator it = dBuffers.find(mat.d_elements);
	int count = -1;
	if (it != dBuffers.end()) {
		count = (*it).second;
		dBuffers.erase(it);
		if(debugMem)outln("dBuffers erased");
		if (count == 1) {
			typedef typename std::map<T*, long>::iterator sizeit;
			sizeit si = dSizes.find(mat.d_elements);
			if(si != dSizes.end()) {
				dSizes.erase(si);
				long amt = (*si).second;
				dBytesFreed += amt;
				currDevice -= amt;
			}
			ptrDims.erase(mat.d_elements);
			if(checkValid(mat.d_elements)) {
				if(debugMem)outln("freeDevice:  " << mat.toShortString() << " had only 1 ref; FREEING device " << mat.d_elements);
				checkCudaError(cudaFree( mat.d_elements));
			}
			mat.d_elements = null;
		} else {
			dBuffers.insert(dBuffers.end(),
					std::pair<T*, int>(mat.d_elements, count - 1));
			if(debugMem)outln("freeDevice:  " << mat.toShortString() << " device ( " << mat.d_elements << ") refcount now at " << (count - 1) );
		}
		count--;
	} else {
		outln("freeDevice: " << mat.toShortString() << " had no dBuffer ref for "<< mat.d_elements << " [caller " << b_util::caller().c_str() << "] disposition elsewhere" );
	}
	typedef typename std::map<CuMatrix<T>*, T*>::iterator miterator;
	miterator mit = mdBuffers.find(&mat);
	if (mit != mdBuffers.end()) {
		mdBuffers.erase(mit);
	}
	return count;
}

