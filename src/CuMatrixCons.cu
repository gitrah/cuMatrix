/*
 * CuMatrixCons.cu
 *
 *      Author: reid
 */
#include <typeinfo>
#include <stdarg.h>

#include "CuMatrix.h"
#include "caps.h"
#include "Kernels.h"

__constant__ uint dDefaultMatProdBlockX;
__constant__ uint dDefaultMatProdBlockY;
__constant__ uint3 dgDefaultMatProdBlock;


void setAllGpuConstants() {
	int devCount, currDev;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	checkCudaErrors(cudaGetDevice(&currDev));

	for(int i = 0; i < devCount;i++) {
		if(strstr(gpuNames[i].c_str(), "750 Ti")) {
			prlocf("skipping sluggish 750 ti\n");
			continue;
		}
		flprintf("setting device %d\n",i);
		checkCudaErrors(cudaSetDevice(i));
		flprintf("setting constants for device %s %d\n",gpuNames[i].c_str(),i);
		setCurrGpuConstants();
	}
	checkCudaErrors(cudaSetDevice(currDev));
}

void setCurrGpuConstants() {
	prlocf("setting gpu constants...\n");
	checkCudaErrors(cudaMemcpyToSymbol(dDefaultMatProdBlockX,&gDefaultMatProdBlock.x, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(dDefaultMatProdBlockY,&gDefaultMatProdBlock.y, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(dgDefaultMatProdBlock,&gDefaultMatProdBlock, sizeof(uint3)));
	prlocf("set gpu constants\n");
}

template<typename T> __host__ __device__ void CuMatrix<T>::initMembers() {
	// ah, templates; do they not make the code beautiful?
	// yes, they do not
	elements = null;
	d_elements = null;
	m = 0;
	n = 0;
	p = 0;
	size = 0;
	oldM = 0;
	oldN = 0;
	posed = false;
	colMajor = false;
	lastMod = mod_neither;
	ownsBuffers = true;
	txp = null;
#ifndef __CUDA_ARCH__
	Constructed++;
#endif
}

template<typename T> __host__ MemMgr<T>& CuMatrix<T>::getMgr() {
	return *mgr;
}

template <typename T> string CuMatrix<T>::typeStr() {
	if(theTypeStr.length() == 0) {
		theTypeStr = string( typeid(T).name());
	}
	return theTypeStr;
}

__global__ void setCaps() {
	if(threadIdx.x == blockIdx.x == 0) {
/*
		gd_devCaps = (ExecCaps**) malloc(g_devCount* sizeof(ExecCaps*));
		for(int i = 0; i < g_devCount; i++) {
			ExecCaps* cap = new ExecCaps();
			gd_devCaps[i] = cap;
			ExecCaps* d_ptr;
			ExecCaps::getExecCaps(*cap, i);
			d_ptr =
			// copy caps
			checkCudaError(cudaMemcpy(d_ptr, cap, sizeof(ExecCaps), cudaMemcpyHostToDevice));
			// add to device-side caps array
			checkCudaError(cudaMemcpy(gd_devCaps + i, d_ptr, sizeof(ExecCaps*), cudaMemcpyDeviceToDevice));
			outln("adding " << i << "\n" << cap->toString() << "\n");
		}
*/
	}
}

/*
 * must be called for each parameter instance, because memory managers are type-specific
 * also initialized cublas
 */
template <typename T> void CuMatrix<T>::init(int maxThreads, int maxBlocks) {
	b_util::dumpStack();

#ifdef CuMatrix_DebugBuild
	buildType = debug;
#else
	buildType = release;
#endif
	outln((buildType == debug ? "Debug" : "Release" ) << " CuMatrix<" << typeStr() << ">::init creating MemMgr arch " );
	mgr = new MemMgr<T>();
	outln(" CuMatrix<" << typeStr() << ">::init created MemMgr " << mgr);
	MaxThreads = maxThreads;
	MaxBlocks = maxBlocks;

	setCurrGpuConstants();
	warmup<<<1,1>>>();
	setMaxColsDisplayed(25);
	setMaxRowsDisplayed(25);

	cublasStatus_t ret;

	ret = cublasCreate(&g_handle);

	if (ret != CUBLAS_STATUS_SUCCESS)
	{
	   printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
	   exit(EXIT_FAILURE);
	}

}


template <typename T> void CuMatrix<T>::cleanup() {
	if(mgr) {
		outln(" CuMatrix<T>::cleanup mgr " << mgr);
		delete mgr;
		mgr = null;
	}
    //ExecCaps::freeDevCaps();

	util<T>::deletePtrArray(g_devCaps,g_devCount);
	free(g_devCaps);
	//freeDevSideDevCaps<<<1,1>>>();

    cublasStatus_t error = cublasDestroy(g_handle);

	if (error != CUBLAS_STATUS_SUCCESS) {
		printf("cublasDestroy h_CUBLAS d_C returned error code %d, line(%d)\n",
				error, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaDeviceReset(); // cleans up the dev
	outln("\nConstructed " << Constructed << "\nDestructed " << Destructed);

	outln("HDCopied " << HDCopied << ", mem " << b_util::expNotation(MemHdCopied));
	outln("DDCopied " << DDCopied << ", mem " << b_util::expNotation(MemDdCopied));
	outln("DHCopied " << DHCopied << ", mem " << b_util::expNotation(MemDhCopied));
	outln("HHCopied " << HHCopied << ", mem " << b_util::expNotation(MemHhCopied));
}

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix() {
	elements = null;
	d_elements = null;
	m = 0;
	n = 0;
	p = 0;
	size = 0;
	oldM = 0;
	oldN = 0;
	posed = false;
	colMajor = false;
	lastMod = mod_neither;
	ownsBuffers = true;
	txp = null;
	currSum = 0;
/*
	if (checkDebug(debugCons | debugLife)) {
		outln( "default constructor CuMatrix() -> " << toShortString());
		b_util::dumpStack();
	}
*/
}

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix(const CuMatrix<T>& o) {
	elements = o.elements;
	d_elements = o.d_elements;
	m =  o.m;
	n =  o.n;
	p =  o.p;
	size = o.size;
	posed = o.posed;
	colMajor = o.colMajor;
	lastMod = o.lastMod;
	ownsBuffers = o.ownsBuffers;
	txp = o.txp;
	ownsTxp = o.ownsTxp;
	currSum = 0;
#ifndef __CUDA_ARCH__
	if (elements)
		getMgr().addHost(*this);
	if (d_elements)
		getMgr().addDevice(*this);
	Constructed++;
#endif

	if (checkDebug(debugCons | debugLife)) {
		printf("CuMatrix::CuMatrix() this %p\n",this);
#ifndef __CUDA_ARCH__
		b_util::dumpStack();
#endif
	}

}

/*
template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix(const CuMat<T>& o) {
	elements = o.elements;
	d_elements = o.d_elements;
	m =  o.m;
	n =  o.n;
	p =  o.p;
	size = o.size;
	posed = o.posed;
	colMajor = o.colMajor;
	lastMod = o.lastMod;
	ownsTxp = ownsBuffers = o.ownsBuffers;
	txp = null;
	oldN = oldM = 0;
#ifndef __CUDA_ARCH__
	if (!d_elements)
		getMgr().allocDevice(*this);
	else
		getMgr().addDevice(*this);
	Constructed++;
#endif

}
*/


template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix( T* h_data, uint m, uint n, bool allocateD )  {
	initMembers();
	elements = h_data;
	this->m = m;
	this->n = n;
	p = n;
	size = m * n * sizeof(T);
#ifndef __CUDA_ARCH__
	getMgr().addHost(*this);
#endif
	if(allocateD) {
		lastMod = mod_host;
#ifndef __CUDA_ARCH__
		getMgr().allocDevice(*this);
		syncBuffers();
#else
		d_elements = (T*) malloc(size);
#endif
	}
	if (checkDebug(debugCons | debugLife)) {
		printf("cons CuMatrix(%p, %u, %u, %s)\n", h_data,  m ,n ,tOrF(allocateD) );
#ifndef __CUDA_ARCH__
		b_util::dumpStack();
#endif
	}
}

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix( T* h_data, uint m, uint n, uint p, bool allocateD) {
	initMembers();
	elements = h_data;
	this->m = m;
	this->n = n;
	this->p = p;
	size = m * p * sizeof(T);
#ifndef __CUDA_ARCH__
	getMgr().addHost(*this);
#endif
	if(allocateD) {
		lastMod = mod_host;
#ifndef __CUDA_ARCH__
		getMgr().allocDevice(*this);
		syncBuffers();
#else
		d_elements = (T*) malloc(size);
#endif
	}
	if (checkDebug(debugCons | debugLife)) {
		printf("cons CuMatrix(%p, %u, %u, %u, %s)\n", h_data,  m ,n,p, tOrF(allocateD) );
#ifndef __CUDA_ARCH__
		b_util::dumpStack();
#endif
	}
}

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix(uint m, uint n, bool allocateH, bool allocateD) {
	initMembers();
	this->m = m;
	this->n = n;
	p = n;
	size = m * n * sizeof(T);
	if (size) {
		if(allocateH) {
#ifndef __CUDA_ARCH__
			getMgr().allocHost(*this);
#else
			setLastError(hostAllocationFromDeviceEx);
#endif
		}
		if(allocateD) {
#ifndef __CUDA_ARCH__
			checkCudaError(getMgr().allocDevice(*this));
#else
			d_elements = (T*) malloc(size);
#endif
		}
	}
	if (checkDebug(debugCons | debugLife)) {
		printf("cons CuMatrix(%u, %u, %s, %s)\n",  m ,n,tOrF(allocateD),tOrF(allocateH) );
#ifndef __CUDA_ARCH__
		b_util::dumpStack();
#endif
	}
}

template<typename T> __host__ __device__  void CuMatrix<T>::freeTxp() {
	if( txp ) {
		if(ownsTxp) {
			//if(checkDebug(debugTxp))outln(toShortString() <<" deleting txp " << txp->toShortString());
			delete txp;
		} else {
			//if(checkDebug(debugTxp))outln(toShortString() <<" reseting txp " << txp->toShortString());
		}
		txp = null;
	}
}


template<typename T> __host__ __device__ CuMatrix<T>::~CuMatrix() {
#ifndef __CUDA_ARCH__
	stringstream disposition;
	int refcount;
	disposition << "~CuMatrix() " << toShortString();
#endif
	T* e_ptr = elements;
	if (e_ptr != null )  {
#ifndef __CUDA_ARCH__
		if(ownsBuffers) {
			refcount = getMgr().freeHost(*this);
			if(refcount) {
				disposition << "\n\thost " << elements << " still has " << refcount << " refs";
			} else {
				disposition << "\n\thost " << e_ptr << " freed";
			}

		} else {
			disposition << "\nwas submatrix, no disposition of these host/dev buffers";
		}
#endif
		elements = null;
	}
	e_ptr = d_elements;
	if (e_ptr != null) {
		if(ownsBuffers) {
#ifndef __CUDA_ARCH__
			refcount = getMgr().freeDevice(*this);
			if(refcount) {
				disposition << "\n\tdevice " << d_elements << " still has " << refcount << " refs";
			} else {
				disposition << "\n\tdevice " << e_ptr << " freed";
			}
#else
			if (checkDebug(debugCons | debugLife))printf("~CuMatrix() d_elements %p\n", d_elements);
			free(d_elements);
#endif
		}
		d_elements = null;
	}
	freeTxp();
#ifndef __CUDA_ARCH__
	Destructed++;
#endif
	if (checkDebug(debugCons | debugLife)) {
#ifndef __CUDA_ARCH__
		outln(disposition.str());
		if(checkDebug(debugLife)) {
			b_util::dumpStack();
		}
#endif
	}
}

template <typename T> CuMatrix<T> CuMatrix<T>::fromBuffer(void* buffer, uint elemBytes, T (*converter)(void*), uint m, uint n, uint p) {
	CuMatrix<T> ret = CuMatrix<T>::zeros(m,p);
	ret.n = n;
	ret.syncBuffers();
	T* elemOut;
	char * elemIn;
	for(uint row = 0; row < m; row++) {
		elemOut = ret.elements + row * p;
		elemIn = (char*)buffer + row * n * elemBytes;
		for(uint col =0; col < n; col++) {
			*(elemOut + col) = (*converter)(elemIn + col * elemBytes);
		}
	}
	ret.invalidateDevice();
	ret.syncBuffers();
	if(checkDebug(debugCons)) {
		flprintf("from buffer of elemSize %d, created:\n", elemBytes);
		outln(ret);
	}
	return ret;
}


template <> __host__ __device__  bool  CuMatrix<float>::integralTypeQ()  { return false;}
template <> __host__ __device__  bool  CuMatrix<double>::integralTypeQ()  { return false;}
template <typename T> __host__ __device__  bool  CuMatrix<T>::integralTypeQ()  { return true;}



#include "CuMatrixInster.cu"
