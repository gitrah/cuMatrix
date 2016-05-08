/*
 * MemMgr.cc
 *
 *  Created on: Sep 20, 2012
 *      Author: reid
 */

#include "MemMgr.h"
#include "CuMatrix.h"
#include "debug.h"
#include "caps.h"
#include "MatrixExceptions.h"
#include <helper_cuda.h>
//#define MMGRTMPLT

template class MemMgr<float>;
template class MemMgr<double>;
template class MemMgr<long>;
template class MemMgr<ulong>;
template class MemMgr<int>;
template class MemMgr<uint>;
bool pinnedQ = true;
/*
template <typename K, typename V> void printMap(string name, CMap<K,V>& theMap) {
	typedef typename map<K, V>::iterator iterator;
	iterator it = theMap.begin();
	cout << name.c_str() << endl;
	while(it != theMap.end()) {
		cout << "\t" << (*it).first << " -> " << (*it).second << endl;
		it++;
	}
}
*/

template<typename T> __host__ MemMgr<T>::MemMgr() :
		hBytesAllocated(0l),
		hBytesFreed(0l),
		dBytesAllocated(0l),
		dBytesFreed(0l),
		mmHHCopied(0l),
		memMmHhCopied(0l),
		currHostB(0l),
		currDeviceB(0l),
		page_size((size_t) sysconf (_SC_PAGESIZE))
{
	outln("MemMgr<T>::MemMgr() this " << this << ", sizeof(T) " << sizeof(T)  );
}

template<typename T> __host__ MemMgr<T>::~MemMgr() {
	outln("~MemMgr (" << this << ", sizeof<T> " << sizeof(T) << " ) with " << hBuffers.size() << " in hBuffers");
	outln("dDimCounts " << dDimCounts.size());
    if(ptrDims.size() > 0) {
    	outln("leftovers!");
    	printMap<T*,string>("ptrDims", ptrDims);
    }
	typedef typename map<T*, int>::iterator iterator;
	typedef typename map<T*, long>::iterator sizeit;
	typedef typename map<const T*,const  T*>::iterator parentsIt;
	iterator it = hBuffers.begin();
	size_t proced = 0;
	T* mem;
	while (it != hBuffers.end() && proced++ < hBuffers.size()) {

		if((*it).first) {
			if(checkDebug(debugMem))outln("~MemMgr freeing h " << (*it).first << " procing #" << proced);
			sizeit si = hSizes.find((*it).first);
			if(si != hSizes.end()) {
				long amt = (*si).second;
				hBytesFreed += amt;
				currHostB -= amt;
				hSizes.erase(si);
			}
			T* hostm = (*it).first;
			if(checkDebug(debugDestr)){
				checkValid(hostm,"~MemMgr");
				flprintf("freeing host hostm %p\n", hostm);
			}
			checkCudaError(cudaFreeHost( hostm));
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
				currDeviceB -= amt;
				dSizes.erase(si);
			}
			checkCudaError(cudaFree( mem));
		}
		it++;
	}
	dBuffers.clear();

	if(checkDebug(debugMem))dumpUsage();

	outln("~MemMgr currDeviceB " << b_util::expNotation(currDeviceB));
	outln("~MemMgr currHostB " << b_util::expNotation(currHostB));

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
		printMap<T*,string>("ptrDims", ptrDims);
	}
}

template<typename T> __host__ int MemMgr<T>::migrate(int dev, CuMatrix<T>& m) {
	assert(m.tiler.countGpus() == 1);

	if(checkDebug(debugMultGPU))outln( b_util::caller() << " migrating " << m.toShortString() << " to gpu " << dev);
	if(m.lastMod == mod_host) {
		dthrow(notSyncedDev());
	}
	int currentDev;
	checkCudaErrors(cudaGetDevice(&currentDev));

	if(currentDev != dev) {
		if(checkDebug(debugMultGPU))outln(b_util::caller() << " changing curr device from " << currentDev << " to " << dev);
		ExecCaps_setDevice(dev);
	}
    int mDevice = util<T>::getDevice(m.tiler.buffer(m.tiler.nextGpu(currentDev)));
    if(mDevice == dev) {
    	if(checkDebug(debugMultGPU))outln(m.toShortString() << " already on device " << dev);
    	return 0;
    } else {
		//if(checkDebug(debugMem)) {
		if(checkDebug(debugMultGPU))outln("org tiler " << m.tiler);
		if(checkDebug(debugMultGPU))outln("currBuff " << m.tiler.currBuffer() << " not on " << dev << "; moving from " << mDevice);
		//}
		T* devMem = null;
		if(checkDebug(debugMem)) {
			flprintf("migrate %p, dBytesAllocated now at %ld\n", devMem, m.size);
			//b_util::usedDmem();
		}
		checkCudaErrors( cudaMalloc((void**) &devMem, m.size));
		checkCudaErrors( cudaMemcpyPeer((void*)devMem, dev, (void*) m.tiler.currBuffer(), mDevice, m.size) );
		CuMatrix<T>::DDCopied++;
		CuMatrix<T>::MemDdCopied += m.size;

		freeTiles(m);
		m.tiler.setGpuMask( Tiler<T>::gpuMaskFromGpu(dev));
		m.tiler.setBuffer(dev,devMem);
		if(checkDebug(debugMultGPU))outln("after migatr " << m.tiler);
		addTiles(&(m.tiler));
		return 1;

	}
	if(currentDev != dev) {
		ExecCaps_restoreDevice(currentDev);
	}
}

template<typename T> __host__ pair<size_t,size_t> MemMgr<T>::migrate(int dev, vector<reference_wrapper<CuMatrix<T>>>  ms) {
	//typedef
	pair<size_t,size_t> load(0,0);

	for( CuMatrix<T> & curr : ms) {

		if(curr.elements)
			load.first += curr.size;
		if(curr.currBuffer())
			load.second += curr.tiler.tileSize;
		migrate(dev, curr);
	}
	return load;
}

template<typename T> __host__ int MemMgr<T>::meet(const vector<CuMatrix<T> >& mats) {
	vector<long> volumes;
	T* buff = nullptr;
	for(  auto& m : mats) {
		buff = m.currBuffer();
		if(buff)
			volumes.push_back(m.size);
	}

	long maxVol = -1;
	int targGpu = -1;
	int idx = 0;
	for( auto v : volumes) {
		if(v > maxVol) {
			maxVol = v;
			targGpu = idx;
		}
		idx++;
	}

	if(checkDebug(debugMultGPU))flprintf("chose device %d because it had morbytz %ld\n", targGpu, maxVol);

	return targGpu;
}

template<typename T> bool MemMgr<T>::checkValid(const T* addr, const char* msg) {

		if(checkDebug(debugMem | debugCons | debugCheckValid)) {

			ExecCaps* currCaps;
			checkCudaErrors(ExecCaps::currCaps(&currCaps));
			if (currCaps->deviceProp.major > 1) {
				struct cudaPointerAttributes ptrAtts;
				//ptrAtts.memoryType = dev ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
				cudaError_t ret = cudaPointerGetAttributes(&ptrAtts, addr);
				if (ret == cudaSuccess) {
					if(checkDebug(debugMem | debugRefcount | debugCheckValid | debugCons )) {
						outln((msg != null ? msg : "") << " cudaSuccess " << addr << " points to " << ( ptrAtts.memoryType == cudaMemoryTypeDevice ? " device " : " host") << " mem" ) ;
					}
					return true;
				}
				outln((msg != null ? msg : "") << " Imwalid " << addr);
				b_util::dumpStack();
				checkCudaError(ret);
			}
			return false;
		} else {
			return true;
		}
	}

template<typename T> void MemMgr<T>::checkRange(const T* addr, int endIdx, const char* msg) {
	ExecCaps* currCaps;
	checkCudaErrors(ExecCaps::currCaps(&currCaps));
	for(int i =0; i < endIdx; i++) {
		if (currCaps->deviceProp.major > 1) {
			struct cudaPointerAttributes ptrAtts;
			//ptrAtts.memoryType = dev ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
			cudaError_t ret = cudaPointerGetAttributes(&ptrAtts, addr + i);
			if (ret != cudaSuccess) {
				outln((msg != null ? msg : "") << " Imwalid " << addr << " + " << i );
				b_util::dumpStack();
				checkCudaError(ret);
			}
		}
	}
}

template<typename T> __host__ void MemMgr<T>::dumpUsage() {
	typedef typename map<string, int>::iterator siterator;
	outln("\n\nMemMgr::dumpUsage()");
	b_util::announceTime();
	if(checkDebug(debugMem))outln( hDimCounts.size() << " distinct dimensions of host allocations; counts:");
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
	typedef typename map<string, int>::iterator iterator;
	string dims = mat.dimsString();
	iterator it = hDimCounts.find(dims);
	int allocCount = 0;
	if (it != hDimCounts.end()) {
		allocCount = ((*it).second + 1) ;
		hDimCounts.erase(it);
		hDimCounts.insert( hDimCounts.end(),
				pair<string, int>(dims, allocCount));
		if(checkDebug(debugMem | debugCons ))outln( "host dim " << dims << " has now been allocated " << allocCount << " time" << (allocCount > 1 ? "s":""));
	} else {
		hDimCounts.insert( hDimCounts.end(),
				pair<string, int>(dims, 1));
		if(checkDebug(debugMem| debugCons ))outln("host dim " << dims << " gets first allocation");
	}
	if(mat.ownsBuffers)ptrDims.insert(ptrDims.end(), pair<T*, string>(mat.elements, dims));
	hSizes.insert(hSizes.end(), pair<T*, long>(mat.elements, mat.size));
	//if(checkDebug(debugMem))b_util::dumpStack();
}



template<typename T> __host__ cudaError_t MemMgr<T>::allocHost(CuMatrix<T>& mat, T* source) {
	if(mat.size > 0) {
		if(checkDebug(debugMem| debugCons ))usedDevMem();
		if(mat.elements && (pinnedQ && checkValid(mat.elements))) {
			outln("already allocated " << mat.elements << " to " << &mat.elements);
			dthrow(hostReallocation());
		}
		cudaError_t res = cudaSuccess;
		if(pinnedQ)
			res = cudaHostAlloc( (void**)&mat.elements,mat.size,0);
		else
			mat.elements = (T*) malloc(mat.size);

		if(res != cudaSuccess) {
			outln("allocHost: FAILURE: " << mat.toShortString());
			checkCudaError(res);
			return res;
		}
		mat.tiler.m_elems = mat.elements;
		hBytesAllocated += mat.size;
		currHostB += mat.size;
		if(checkDebug(debugMem| debugCons )) {
			outln("allocHost pinneQ " << pinnedQ <<": success for " << mat.toShortString() << " hBytesAllocated now at " << b_util::expNotation(hBytesAllocated));
			mCheckValid(mat.elements);
		}
		hBuffers.insert(hBuffers.end(),pair<T*, int>(mat.elements, 1));
		if(checkDebug(debugRefcount))  {
			outln("allocHost pinned " << pinnedQ << " success for " << mat.toShortString() << ", " << mat.elements << " refcount now at 1");
		}
		mhBuffers.insert(mhBuffers.end(),
				pair<CuMatrix<T>*, T*>(&mat, mat.elements));
		if (source != null) {
			outln("allocHost copying " << mat.size << " from " << mat.elements << " to " << source);
			checkCudaError(
					cudaMemcpy(source, mat.elements, mat.size, cudaMemcpyHostToHost));
			mmHHCopied++;
			memMmHhCopied += mat.size;
		}
		//if(checkDebug(debugRefcount))b_util::dumpStack();
		addHostDims(mat);
		if(checkDebug(debugMem))usedDevMem();
	}
	return cudaSuccess;
}

template<typename T> __host__ string MemMgr<T>::stats() {
	stringstream ss;
	// if verbose ss << "dev " << this->dBytesAllocated << "A/" << this->dBytesFreed << "F; host " << this->hBytesAllocated << "A/" << this->hBytesFreed << "F ";
	ss << "devdelta " << b_util::expNotation(this->dBytesAllocated - this->dBytesFreed) << "; hostdelta " << b_util::expNotation(this->hBytesAllocated - this->hBytesFreed) << " ";
	return ss.str();
}

template<typename T> __host__ pair<int,int> MemMgr<T>::refCounts(const CuMatrix<T>& m) {
	return pair<int,int>( m.elements ? refCount(m.elements) : 0,  m.tiler.buff() ? refCount(m.tiler.buff()) : 0);
}

template<typename T> __host__ void MemMgr<T>::addHost(  CuMatrix<T>& mat) {
	if (mat.elements) {
/*
		if(checkDebug(debugRefcount)){
			prlocf("dumping stack\n");
			b_util::dumpStack();
		}
*/

		typedef typename map<T*, int>::iterator iterator;
		iterator it = hBuffers.find(mat.elements);
		int refCount = -1;
		if (it != hBuffers.end()) {
			refCount = ((*it).second + 1) ;
			if(checkDebug(debugRefcount)) {
				outln("addHost: " << b_util::caller() << ": matrix " << mat.toShortString() << " setting " << refCount << " ref for " << mat.elements);
				//dumpStack();
			}
			hBuffers.erase(it);
			hBuffers.insert( hBuffers.end(),
					pair<T*, int>(mat.elements, refCount));
		} else {
			if(checkDebug(debugRefcount)) {
				outln("addHost: matrix " << &mat <<
					" found no host ref for " << mat.elements);
				b_util::dumpStack();
			}
		}
		typedef typename map<CuMatrix<T>*, T*>::iterator miterator;
		miterator mit = mhBuffers.find(&mat);
		if (mit == mhBuffers.end()) {
			mhBuffers.insert(mhBuffers.end(),
					pair<CuMatrix<T>*, T*>(&mat, mat.elements));
		}
	}
}

template<typename T> __host__ void MemMgr<T>::addSubmatrix( const CuMatrix<T>& mat, const CuMatrix<T>& container) {
	parents.insert(parents.end(), pair<const T*,const T* >(mat.elements,container.elements));
}

template<typename T> __host__ bool MemMgr<T>::submatrixQ( const CuMatrix<T>& mat) {
	typedef typename map<const T*,const T* >::iterator miterator;
	miterator it = parents.find(mat.elements);
	return(it != parents.end());
}

template<typename T> __host__ cudaError_t MemMgr<T>::getHost(  CuMatrix<T>& mat) {
	typedef typename map<CuMatrix<T>*, T*>::iterator miterator;
	miterator mit = mhBuffers.find(&mat);
	if (mit != mhBuffers.end()) {
		mat.elements = (*mit).second;
		return cudaSuccess;
	}
	return cudaErrorMemoryAllocation;
}

template<typename T> __host__ int MemMgr<T>::freeHost(CuMatrix<T>& mat) {
/*
	if(checkDebug(debugRefcount)){
		prlocf("dumping stack\n");
		b_util::dumpStack();
	}
*/
	//outln("freeHost enter " << b_util::caller());
	int count = -1;
	if(mat.elements) {
		typedef typename map<T*, int>::iterator iterator;
		iterator it = hBuffers.find(mat.elements);
		if (it != hBuffers.end()) {
			//outln("hBuffers.size " << hBuffers.size());
			count = (*it).second;
			hBuffers.erase(it);
			//outln("after erase, hBuffers.size " << hBuffers.size());
			if (count == 1) {
				typedef typename map<T*, long>::iterator sizeit;
				sizeit si = hSizes.find(mat.elements);
				if(si != hSizes.end()) {
					hSizes.erase(si);
					long amt = (*si).second;
					hBytesFreed += amt;
					currHostB -= amt;
				}
				ptrDims.erase(mat.elements);
				if(pinnedQ) {
					checkCudaError(cudaGetLastError());
					checkValid(mat.elements);
					if(checkDebug(debugRefcount | debugDestr ))
						outln("freeHost:  " << mat.toShortString() << " had only 1 ref; FREEING pinned host " << mat.elements);
					checkCudaError(cudaGetLastError());
					checkCudaError(cudaFreeHost( mat.elements));
				}
				else {
					if(checkDebug(debugRefcount | debugDestr ))
						outln("freeHost:  " << mat.toShortString() << " had only 1 ref; FREEING malloc host " << mat.elements);
					free(mat.elements);
				}
				mat.elements = null;
				count = 0;
			} else {
				if (checkDebug(debugRefcount | debugDestr))
					outln(
							"freeHost:  " << mat.toShortString() <<
							" host ( " << mat.elements << ") refcount now at " << (count - 1));
				hBuffers.insert(hBuffers.end(),
						pair<T*, int>(mat.elements, --count));
			}
		} else {
			outln("freeHost: " << mat.toShortString() << " had no hBuffer ref for "<< mat.elements << " [caller " << b_util::caller().c_str() << "] disposition elsewhere" );
		}
		typedef typename map<CuMatrix<T>*, T*>::iterator miterator;
		miterator mit = mhBuffers.find(&mat);
		if (mit != mhBuffers.end()) {
			mhBuffers.erase(mit);
		}
	}
	return count;
}

template<typename T> __host__ void MemMgr<T>::addDeviceDims( const CuMatrix<T>& mat, ulong tileSize) {
	typedef typename map<string, int>::iterator iterator;
	string dims = mat.dimsString();
	iterator it = dDimCounts.find(dims);
	int allocCount = 0;
	if (it != dDimCounts.end()) {
		allocCount = ((*it).second + 1) ;
		if(checkDebug(debugMem))outln( "dev dim " << dims << " has now been allocated " << allocCount << " time" << (allocCount > 1 ? "s":""));
		dDimCounts.erase(it);
		dDimCounts.insert( dDimCounts.end(),
				pair<string, int>(dims, allocCount));
	} else {
		dDimCounts.insert( dDimCounts.end(),
				pair<string, int>(dims, 1));
		if(checkDebug(debugMem))outln( "dev dim " << dims << " gets first allocation");
	}
	if(mat.ownsBuffers) {
		for(int i =0; i < mat.tiler.countGpus(); i++) {
			ptrDims.insert(ptrDims.end(), pair<T*, string>(mat.tiler.buffer(i), tileSize == 0 ? dims : string("tile of " + tileSize)));
			dSizes.insert(dSizes.end(), pair<T*, long>(mat.tiler.buffer(i), tileSize == 0 ? mat.size : tileSize));
		}
	}
}

template<typename T> string dimsString( const DMatrix<T>& mat) {
	stringstream sm, sn,ssize,ssout;
	sm << mat.m;
	sn << mat.n;
	ssize << mat.size;
	ssout << sm.str() << "x" << sn.str() << "-" << ssize.str();
	return ssout.str();
}

//multi gpu case?

template<typename T> __host__ cudaError_t MemMgr<T>::allocDevice(
		CuMatrix<T>& mat,int device, ulong tileSize) {
	assert(0);
	return cudaSuccess;
}

template<typename T> __host__ void MemMgr<T>::addTiles( const Tiler<T>* tiler) {
	if (tiler->hasDmemQ() ) {
/*
		if(checkDebug(debugRefcount)) {
			prlocf("dumpingstack");
			b_util::dumpStack();
		}
*/
		if(checkDebug(debugCheckValid)) {
			flprintf("curr dev %d %s  tiler& %p, %dx%dx%d - matsize %ld -:> %p\n", ExecCaps::currDev(), b_util::caller().c_str(), tiler, tiler->m_m, tiler->m_n, tiler->m_p, tiler->m_size, tiler->buff());
		}
		memblo;
		typedef typename map<T*, int>::iterator iterator;
		int refs = -1;
		if(checkDebug(debugCheckValid))outln("tiler->countGpus() " << tiler->countGpus()<< "\n");
		for(int i =0; i < tiler->countGpus(); i++) {
			int nextGpu = tiler->nextGpu(i);
			T* buffer = tiler->buffer(nextGpu);
			if(!buffer) continue;
			if( !util<T>::onDevice(buffer, nextGpu)) {
				if(checkDebug(debugRefcount))outln("addTiles: tiler " << tiler <<
						" buffer " << buffer << " not on device " << nextGpu);
				continue;
			}
			iterator it = dBuffers.find(buffer);
			if (it != dBuffers.end()) {
				refs = (*it).second + 1;
				if(checkDebug(debugRefcount))outln("addTiles: tiler " << tiler <<
						" incing refs to " << refs << " for " << buffer);
				dBuffers.erase(it);
				dBuffers.insert(dBuffers.end(),
						pair<T*, int>(buffer, refs));
			} else {
				if(checkDebug(debugRefcount)) outln("addTiles: tiler " << *tiler <<
						" found no ref for " << buffer);
				dBuffers.insert(dBuffers.end(),
						pair<T*, int>(buffer, 1));
			}
			it = dBuffers.find(buffer);
			if(checkDebug(debugRefcount))outln(
					"after adding " << buffer <<  " to &dBuffers (" << &dBuffers << "), ref now at " << (*it).second);
			dBytesAllocated += tiler->tileSize;
		}
		if(checkDebug(debugMem))usedDevMem();
	} else {
		outln("addTiles called on " << ExecCaps::currDev() << " with non-dmemq tiler " << *tiler );
		b_util::dumpStack();
	}
}

template<typename T> __host__ int MemMgr<T>::refCount( T* els) {
	assert(els);
	ExecCaps* currCaps;
	checkCudaErrors(ExecCaps::currCaps(&currCaps));
	int refCount = 0;
	if (currCaps->deviceProp.major > 1) {
		struct cudaPointerAttributes ptrAtts;
		//ptrAtts.memoryType = dev ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
		cudaError_t ret = cudaPointerGetAttributes(&ptrAtts, els);
		if (ret == cudaSuccess) {
			typedef typename map<T*, int>::iterator iterator;
			if (ptrAtts.memoryType == ::cudaMemoryTypeHost) {
				iterator it = hBuffers.find(els);
				if (it != hBuffers.end()) {
					refCount = (*it).second;
				}
			} else if (ptrAtts.memoryType == ::cudaMemoryTypeDevice) {
				iterator it = dBuffers.find(els);
				if (it != dBuffers.end()) {
					refCount = (*it).second;
				}
			}
		}
	}
	return refCount;
}

template<typename T> __host__ int MemMgr<T>::freeTiles(CuMatrix<T>& mat) {
/*
	if(checkDebug(debugRefcount)){
		flprintf("dumping stack %s\n","");
		b_util::dumpStack();
	}
*/
	if(checkDebug(debugCheckValid | debugDestr)) {
		flprintf("%s  tiler& %p, %dx%dx%d - matsize %ld -:> %p\n", b_util::caller().c_str(), &(mat.tiler), mat.tiler.m_m, mat.tiler.m_n, mat.tiler.m_p, mat.tiler.m_size, mat.tiler.buff());
	}
	int refCount = -1;
	int freedCount = 0;
	if (mat.tiler.hasDmemQ() && mat.ownsBuffers) {
		T* currBuff;
		int currGpu;
		for(int i = 0; i < mat.tiler.countGpus(); i++) {
			refCount = -1;
			currGpu = mat.tiler.nextGpu(i);
			currBuff = mat.tiler.buffer(currGpu);
			if(checkDebug(debugMem))outln("mat " << &mat << " buffers[nextGpu(" << i << ") == " << currGpu << "] " << currBuff);
			if(submatrixQ(mat)) {
				outln(mat.toShortString() << " is submatrix");
				return freedCount;
			}
			typedef typename map<T*, int>::iterator iterator;
			iterator it = dBuffers.find(currBuff);
			if (it != dBuffers.end()) {
				refCount = (*it).second;
				dBuffers.erase(it);
				if(checkDebug(debugMem))outln("dBuffers erased");
				if (refCount == 1) {
					typedef typename map<T*, long>::iterator sizeit;
					sizeit si = dSizes.find(currBuff);
					if(si != dSizes.end()) {
						if(checkDebug(debugRefcount))outln("si != dSizes.end()");
						dSizes.erase(si);
						long amt = (*si).second;
						currDeviceB -= amt;
					}
					ptrDims.erase(currBuff);
					if(checkDebug(debugRefcount))outln("refCount of " << currBuff << " was 1");
					size_t freeMemoryB, totalMemoryB;
					if(checkValid(currBuff)) {
						if(checkDebug(debugMem)) {
							ExecCaps::allGpuMem(&freeMemoryB, &totalMemoryB);
							b_util::usedDmem(true);
						}
						if(checkDebug(debugRefcount | debugDestr))outln(":  " << mat.toShortString() << " had only 1 ref; FREEING device " << currBuff);
						checkCudaError(cudaFree( currBuff));
						dBytesFreed += mat.tiler.tileSize;
						mat.tiler.setBuffer(currGpu,0);
						if(checkDebug(debugRefcount)) {
							outln("after freeing " << currBuff << ",  buff now at " << mat.tiler.currBuffer());
							b_util::dumpStack();
						}
						freedCount++;
						if(checkDebug(debugMem)) {
							size_t freeMemoryA, totalMemoryA, deltaGpu;
							ExecCaps::allGpuMem(&freeMemoryA, &totalMemoryA);
							deltaGpu = freeMemoryA - freeMemoryB;
							outln("deltaGpu " << deltaGpu << " mat.tiler.tileSize " <<  mat.tiler.tileSize <<", freeMemoryB " << freeMemoryB << " freeMemoryA " << freeMemoryA);
							outln("delta free " << (freeMemoryA-freeMemoryB) << ", checking validity of pointer post cudaFree\n");
							//if(deltaGpu < mat.tiler.tileSize && deltaGpu > ExecCaps::currCaps()->alignment ) checkValid(currBuff);
							//assert(freeMemoryA - freeMemoryB >= mat.tiler.tileSize || deltaGpu < ExecCaps::currCaps()->alignment);
							b_util::usedDmem(true);
						}
					}
				} else {
					dBuffers.insert(dBuffers.end(),
							pair<T*, int>(currBuff, refCount - 1));
					if(checkDebug(debugRefcount))outln(":  " << mat.toShortString() << " device ( " << currBuff << ") refrefCount now at " << (refCount - 1) );
				}
				refCount--;
			} else {
				if(checkDebug(debugMem)){
					outln(": " << mat.toShortString() << " had no dBuffer ref for "<< currBuff << " [caller " << b_util::caller().c_str() << "] disposition elsewhere" );
					b_util::dumpStack();
				}

			}
		}
	}
	return refCount;
}
