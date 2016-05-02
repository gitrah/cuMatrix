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

uint CuDestructs=0;
__device__ uint CuDestructsD=0;

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix() :
	elements{0},
	m{0},n{0},p{0},
	size{0},
	oldM{0}, oldN{0},
	posed{false}, colMajor{false},
	lastMod{mod_neither},
	ownsBuffers{true},
	txp{0},
	ownsTxp{true},
	tiler{*this}
{}

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix(const CuMatrix<T>& o) :
	lastMod{ o.lastMod},
	elements{ o.elements},
	m{ o.m},
	n{ o.n},
	p{ o.p},
	size{ o.size},
	posed{ o.posed},
	colMajor{ o.colMajor},
	ownsBuffers{ o.ownsBuffers},
	txp{ o.txp},
	ownsTxp{o.ownsTxp},
	tiler{ o.tiler}
{

#ifndef __CUDA_ARCH__
	if (elements)
		getMgr().addHost(*this);

	getMgr().addTiles(&tiler);

	Constructed++;
#endif

	if (checkDebug(debugCons )) {
		flprintf("CuMatrix::CuMatrix(const CuMatrix<T>& o %p) copy cons this %p\n",&o, this);
#ifndef __CUDA_ARCH__
		outln("a copy am i " << toShortString());
		b_util::dumpStack();
#endif
	}

}

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix( T* h_data, int m, int n, bool allocateD ) :
	lastMod{ mod_host},
	elements{ h_data},
	m{ m},
	n{ n},
	p{ n},
	size{ m * p * sizeof(T)},
	posed{ false },
	colMajor{ false },
	ownsBuffers{ true },
	txp{ 0 },
	tiler{*this, allocateD} {
#ifndef __CUDA_ARCH__
	getMgr().addHost(*this);
#endif
	if(allocateD) {
		lastMod = mod_host;
		getMgr().addTiles(&tiler);
	}
	if (checkDebug(debugCons )) {
		printf("cons CuMatrix(%p, %u, %u, %s)\n", h_data,  m ,n ,tOrF(allocateD) );
#ifndef __CUDA_ARCH__
		b_util::dumpStack();
#endif
	}
}

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix( T* h_data, int m, int n, int p, bool allocateD) :
	lastMod{ mod_host},
	elements{ h_data},
	m{ m},
	n{ n},
	p{ p},
	size{ m * p * sizeof(T)},
	posed{ false },
	colMajor{ false },
	ownsBuffers{ true },
	txp{ 0 },
	tiler{*this, allocateD} {
#ifndef __CUDA_ARCH__
	getMgr().addHost(*this);
	if(allocateD)getMgr().addTiles(&tiler);
#endif
	if (checkDebug(debugCons )) {
		flprintf("cons CuMatrix(%p, %u, %u, %u, %s)\n", h_data,  m ,n,p, tOrF(allocateD) );
#ifndef __CUDA_ARCH__
		b_util::dumpStack();
#endif
	}
}

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix(int m, int n, bool allocateH, bool allocateD):
		CuMatrix(m,n,n,Tiler<T>::gpuMaskFromGpu(ExecCaps::currDev()),allocateH,allocateD) {
}

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix(int m, int n, int p, bool allocateH, bool allocateD) :
		CuMatrix(m,n,p,Tiler<T>::gpuMaskFromGpu(ExecCaps::currDev()),allocateH,allocateD) {
}

template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix(int m, int n, int p, int gpuMaskm,bool allocateH, bool allocateD):
			elements{0},
			m{m},n{n},p{p},
			size{m * p * sizeof(T)},
			oldM{0}, oldN{0},
			posed{false}, colMajor{false},
			lastMod{allocateH ? ( allocateD ? mod_synced : mod_host) : ( allocateD ? mod_device : mod_neither ) },
			ownsBuffers{true},
			txp{0},
			ownsTxp{true},
		tiler(*this, allocateD) {

	//if(checkDebug(debugMem))usedDevMem();

	if (size) {
		if(allocateH) {
#ifndef __CUDA_ARCH__
			getMgr().allocHost(*this);
#else
			setLastError(hostAllocationFromDeviceEx);
#endif
		}
		if(allocateD){
#ifndef __CUDA_ARCH__

			if (checkDebug(debugCons )) outln("calling add tiles on tiler " << tiler);
			getMgr().addTiles(&tiler);
			if(checkDebug(debugMem))usedDevMem();
#endif
		}
	} else {
		if (checkDebug(debugCons ))prlocf("size == 0!");
	}
	if (checkDebug(debugCons | debugRefcount )) {
		flprintf("cons CuMatrix(%u, %u, %u, %s, %s) ->%p on device %d\n",  m ,n,p, tOrF(allocateH), tOrF(allocateD),this,ExecCaps::currDev() );
#ifndef __CUDA_ARCH__
		b_util::dumpStack();
#endif
	}
}


template<typename T> __host__ __device__ CuMatrix<T>::CuMatrix(int m, int n,
		int p, int tileM, int tileN, int gpuMask, bool allocateH,
		bool allocateD) :
			elements{0},
			m{m},n{n},p{p},
			size{m * p * (int)sizeof(T)},
			oldM{0}, oldN{0},
			posed{false}, colMajor{false},
			lastMod{mod_neither},
			ownsBuffers{true},
			txp{0},
			ownsTxp{true},
		tiler(*this) {
		tiler.gpuMask = gpuMask;
		if(allocateD) tiler.allocTiles();

	if(checkDebug(debugMem))usedDevMem();

	if (size) {
		if(allocateH) {
#ifndef __CUDA_ARCH__
			getMgr().allocHost(*this);
#else
			setLastError(hostAllocationFromDeviceEx);
#endif
#ifndef __CUDA_ARCH__
		if(checkDebug(debugMem))usedDevMem();
#endif
		}
#ifndef __CUDA_ARCH__
		if(allocateD)getMgr().addTiles(&tiler);
#endif
	}
	if (checkDebug(debugCons )) {
		flprintf("cons CuMatrix(%u, %u, %u, %s, %s) ->%p on device %d\n",  m ,n,p, tOrF(allocateD),tOrF(allocateH), this,ExecCaps::currDev() );
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

	if(checkDebug( debugRefcount )) flprintf("die: %dX%dX%d h %p %s d{%p,%p,%p,%p}\n", m,n,p, elements, ownsBuffers ? "owns" : "", tiler.buffers.x, tiler.buffers.y, tiler.buffers.z, tiler.buffers.w);
//
#ifndef __CUDA_ARCH__
	CuDestructs++;
	if(checkDebug(debugRefcount)) {
		//flprintf("(%uX%u) this %p h %p d %p clr: %s\n",m,n,this,elements,tiler.currBuffer(), b_util::caller().c_str());
		outln("~" <<toShortString());
	}
	stringstream disposition;
	int refcount;
	disposition << "~CuMatrix() " << toShortString();
#else
	if(checkDebug(debugCons))flprintf("~CuMatrix (%u X %u) on th %p h %p d %p\n",m,n,this,elements,tiler.currBuffer());
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


			if(ownsBuffers) {
	#ifndef __CUDA_ARCH__
				T* currBuff = tiler.currBuffer();
				refcount = getMgr().freeTiles(*this);
				if(refcount) {
					disposition << "\n\tdevice " << currBuff << " still has " << refcount << " refs";
				} else {
					disposition << "\n\tdevice " << currBuff << " freed";
				}
	#else
				for(int i =0; i < tiler.countGpus();i++) {
					e_ptr = tiler.buffer(tiler.nextGpu(i));
					if (e_ptr != null) {
						if (checkDebug(debugCons ))printf("~CuMatrix() free(buffer(%d) %p)\n", i, e_ptr);
						free(e_ptr);
					}
				}
	#endif
			} else {
				if(checkDebug(debugCons))flprintf("buffer not owned %p %uX%uX%u\n", tiler.buff(), m,n,p);
			}
//		}
//	}
	freeTxp();

	if(checkDebug(debugMem)) b_util::usedDmem();
#ifndef __CUDA_ARCH__
	Destructed++;
#endif
	if (checkDebug(debugCons ) || cudaGetLastError() != cudaSuccess) {
#ifndef __CUDA_ARCH__
		outln(disposition.str());
		b_util::dumpStack();
#endif
	}
}

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
		ExecCaps_setDevice(i);
		flprintf("setting constants for device %s %d\n",gpuNames[i].c_str(),i);
		setCurrGpuConstants();
	}
	ExecCaps_setDevice(currDev);
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
//	mgr->addTiles(&tiler);

#ifndef __CUDA_ARCH__
	Constructed++;
#endif
}

template<typename T> __host__ MemMgr<T>& CuMatrix<T>::getMgr() const {
	return *mgr;
}

template <typename T> string CuMatrix<T>::typeStr() {
	if(theTypeStr.length() == 0) {
		theTypeStr = string( typeid(T).name());
	}
	return theTypeStr;
}

/*
 * must be called for each parameter instance, because memory managers are type-specific
 * also initialized cublas
 */
template <typename T> void CuMatrix<T>::initMemMgrForType(int maxThreads, int maxBlocks) {
	b_util::dumpStack();

#ifdef CuMatrix_DebugBuild
	buildType = debug;
#else
	buildType = release;
#endif
	outln( "\n\n\tExecutable version:  " << (buildType == debug ? "Debug" : "Release" ) << " CuMatrix<" << typeStr() << ">::init creating MemMgr arch " );


	mgr = new MemMgr<T>();
	outln(" CuMatrix<" << typeStr() << ">::init created MemMgr " << mgr);
	MaxThreads = maxThreads;
	MaxBlocks = maxBlocks;

	setCurrGpuConstants();
	setMaxColsDisplayed(25);
	setMaxRowsDisplayed(25);


}


template <typename T> void CuMatrix<T>::cleanup() {
	if(mgr) {
		outln(" CuMatrix<T>::cleanup mgr " << mgr);
		delete mgr;
		mgr = null;
	}
    //ExecCaps::freeDevCaps();

	util<T>::deletePtrArray(g_devCaps,ExecCaps::deviceCount);
	free(g_devCaps);
	//freeDevSideDevCaps<<<1,1>>>();

#ifdef CuMatrix_UseCublas
    cublasStatus_t error = cublasDestroy(g_handle);

	if (error != CUBLAS_STATUS_SUCCESS) {
		printf("cublasDestroy h_CUBLAS d_C returned error code %d, line(%d)\n",
				error, __LINE__);
		exit(EXIT_FAILURE);
	}
#endif
	//void (*ftor) =  }
	b_util::allDevices( [] (){cherr(cudaDeviceReset());});

	outln("\nConstructed " << Constructed << "\nDestructed " << Destructed);

	outln("HDCopied " << HDCopied << ", mem " << b_util::expNotation(MemHdCopied));
	outln("DDCopied " << DDCopied << ", mem " << b_util::expNotation(MemDdCopied));
	outln("DHCopied " << DHCopied << ", mem " << b_util::expNotation(MemDhCopied));
	outln("HHCopied " << HHCopied << ", mem " << b_util::expNotation(MemHhCopied));
}


template <typename T> CuMatrix<T> CuMatrix<T>::fromBuffer(void* buffer, int elemBytes, T (*converter)(void*), int m, int n, int p) {
	outln("m " << m << ", n " << p);
	CuMatrix<T> ret = CuMatrix<T>::zeros(m,p);
	ret.n = n;
	ret.syncBuffers();
	T* elemOut;
	char * elemIn;
	for(int row = 0; row < m; row++) {
		elemOut = ret.elements + row * p;
		elemIn = (char*)buffer + row * n * elemBytes;
		for(int col =0; col < n; col++) {
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
