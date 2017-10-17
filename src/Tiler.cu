/* *
 */
#include "Tiler.h"
#include "CuMatrix.h"
#include "caps.h"
#include <pthread.h>

template<typename T> __host__ __device__ bool Tiler<T>::hasDmemQ() const {
	T* curr = currBuffer();
	if (curr == null)
		return false;
#ifndef __CUDA_ARCH__
	return MemMgr<T>::checkValid(curr);
#else
	return false;
#endif

}
template __host__ __device__ bool Tiler<float>::hasDmemQ() const;
template __host__ __device__ bool Tiler<double>::hasDmemQ() const;
template __host__ __device__ bool Tiler<int>::hasDmemQ() const;
template __host__ __device__ bool Tiler<uint>::hasDmemQ() const;
template __host__ __device__ bool Tiler<long>::hasDmemQ() const;
template __host__ __device__ bool Tiler<ulong>::hasDmemQ() const;

//todo set gpuMask from m's device via getPtrAtts
template<typename T> __host__ __device__ Tiler<T>::Tiler(const CuMatrix<T>& m, void (*kernelle)()) :
		m_m(m.m), m_n(m.n), m_p(m.p), m_elems(m.elements), m_size(m.size), tileSize(
				0), /*tileP(0), */gpuMask(gpuMaskFromGpu(ExecCaps::currDev())), gpuModulus(
				modulusFromMask()), buffers { 0, 0, 0, 0 }, tileD(tdCols), kernelle(nullptr) {
	if (checkDebug(debugTiler)) {
		flprintf("Tiler<T>::Tiler(CuMatrix<T>& m) gpuModulus %d-> this %p, &m (m : ", gpuModulus, this,
				&m);
		m.print("for Tiler:Tiler(CuMatrix<T>&)");
	}
#ifndef __CUDA_ARCH__
	if(checkDebug(debugTiler))b_util::dumpStack();
#endif
}

template __host__ __device__ Tiler<float>::Tiler(const CuMatrix<float>&,  void(*)());
template __host__ __device__ Tiler<double>::Tiler(const CuMatrix<double>&,  void(*)());
template __host__ __device__ Tiler<int>::Tiler(const CuMatrix<int>&,  void(*)());
template __host__ __device__ Tiler<uint>::Tiler(const CuMatrix<uint>&,  void(*)());
template __host__ __device__ Tiler<long>::Tiler(const CuMatrix<long>&,  void(*)());
template __host__ __device__ Tiler<ulong>::Tiler(const CuMatrix<ulong>&,  void(*)());

template<typename T> __host__ __device__ Tiler<T>::Tiler( CuMatrix<T>& m,
		bool allocate, int gpuMask, TileDirection tileD, void (*kernelle)()) :
		m_m(m.m), m_n(m.n), m_p(m.p), m_elems(m.elements), m_size(m.size),
		gpuMask(gpuMask), gpuModulus(modulusFromMask()), buffers {0, 0, 0, 0 },
		tileD(tileD), kernelle(nullptr)  {
#ifndef __CUDA_ARCH__
	if (!gpuMask) {
		b_util::dumpStack();
	}
#endif
	if (checkDebug(debugTiler)) {
		flprintf(
				"(m(h:%p,%uX%uX%u sz: %lu), alloc %s, mask %d modulus %d) on device %d -> this %p \n",
				m.elements, m.m, m.n, m.p, m.size, tOrF(allocate), gpuMask, gpuModulus,
				ExecCaps::currDev(), this);
		prlocf("\tm : \n");
		m.print("Tiler:Tiler");
	}
	if (allocate) {
		allocTiles(m._tileM, m._tileN, m._tileP);
	}
}

template __host__ __device__ Tiler<float>::Tiler( CuMatrix<float>&, bool,
		int, TileDirection,  void(*)());
template __host__ __device__ Tiler<double>::Tiler( CuMatrix<double>&, bool,
		int, TileDirection,  void(*)());
template __host__ __device__ Tiler<int>::Tiler( CuMatrix<int>&, bool, int,
		TileDirection,  void(*)());
template __host__ __device__ Tiler<uint>::Tiler( CuMatrix<uint>&, bool,
		int, TileDirection,  void(*)());
template __host__ __device__ Tiler<ulong>::Tiler( CuMatrix<ulong>&, bool,
		int, TileDirection,  void(*)());
template __host__ __device__ Tiler<long>::Tiler( CuMatrix<long>&, bool,
		int, TileDirection,  void(*)());

/*
template<typename T> __host__ __device__ Tiler<T>::Tiler(int m, int n, int p,
		long size, T* elements, bool allocate,	int gpuMask , TileDirection tileD):
				m_m(m), m_n(n), m_p(p), m_elems(elements), m_size(size), gpuMask(
						gpuMask), gpuModulus(modulusFromMask()), tileP(0), tileD(tileD), buffers {
						0, 0, 0, 0 } {
		if (!gpuMask) {
			b_util::dumpStack();
		}
		if (checkDebug(debugTiler)) {
			flprintf(
					"(m(h:%p,%uX%uX%u sz: %ld), alloc %s, mask %d modulus %d) on device %d -> this %p \n",
					m_elems, m, n, p, size, tOrF(allocate), gpuMask, gpuModulus,
					ExecCaps::currDev(), this);
			prlocf("\tm : \n");

		}
		if (allocate) {
			allocTiles();
		}
}

template __host__ __device__ Tiler<double>::Tiler(int, int, int, long, double*, bool, int, TileDirection );
template __host__ __device__ Tiler<float>::Tiler(int, int, int, long, float*, bool, int, TileDirection );
template __host__ __device__ Tiler<ulong>::Tiler(int, int, int, long, ulong*, bool, int, TileDirection );
template __host__ __device__ Tiler<long>::Tiler(int, int, int, long, long*, bool, int, TileDirection );
template __host__ __device__ Tiler<uint>::Tiler(int, int, int, long, uint*, bool, int, TileDirection );
template __host__ __device__ Tiler<int>::Tiler(int, int, int, long, int*, bool, int, TileDirection );
*/

template<typename T> __host__ __device__ int Tiler<T>::deviceOfResidence() const {
	if (!(buffers.x || buffers.y || buffers.z || buffers.w))
		return -1;

	if ((buffers.x != 0) + (buffers.y != 0) + (buffers.z != 0)
			+ (buffers.w != 0) > 1)
		setLastError(multipleGpusEx);

	if (buffers.x != 0) {
		return 0;
	} else if (buffers.y != 0) {
		return 1;
	}else if (buffers.z != 0) {
		return 2;
	}
	return 3;
}

template __host__ __device__ int Tiler<float>::deviceOfResidence() const;
template __host__ __device__ int Tiler<double>::deviceOfResidence() const;
template __host__ __device__ int Tiler<int>::deviceOfResidence() const;
template __host__ __device__ int Tiler<uint>::deviceOfResidence() const;
template __host__ __device__ int Tiler<long>::deviceOfResidence() const;
template __host__ __device__ int Tiler<ulong>::deviceOfResidence() const;

template<typename T> __host__ __device__ int Tiler<T>::indexInMask(int gpu) const {
	if(!( gpuMask >> gpu &  1))
		return -1;
	int idx = -1;
	int position = 0;
	int mask=gpuMask;
	while( mask != 0 && position <= gpu) {
		if ((mask >> position) & 1) {
			idx ++;
		}
		position++;
	}

	return idx;

}

template __host__ __device__ int Tiler<float>::indexInMask(int) const;
template __host__ __device__ int Tiler<double>::indexInMask(int) const;
template __host__ __device__ int Tiler<int>::indexInMask(int) const;
template __host__ __device__ int Tiler<uint>::indexInMask(int) const;
template __host__ __device__ int Tiler<long>::indexInMask(int) const;
template __host__ __device__ int Tiler<ulong>::indexInMask(int) const;

template<typename T> __host__ __device__ int Tiler<T>::nextGpu(
		int lastGpu) const {
	assert(gpuModulus);
	//if(checkDebug(debugTiler))prlocf("\n\nnextGPU ent\n\n");
	//if (checkDebug(debugTiler)) dumpMask(gpuMask);
	int idx = lastGpu + 1;
	if (false && checkDebug(debugTiler)) {
		flprintf("gpuModulus %d lastGpu %d (idx %% gpuModulus) %d\n",
				gpuModulus, lastGpu, (idx % gpuModulus));
		flprintf("gpuMask >>  (idx %% gpuModulus) %d\n",
				gpuMask >> (idx % gpuModulus));
		flprintf("( gpuMask >>  (idx %% gpuModulus) & 1) %d\n",
				(gpuMask >> (idx % gpuModulus) & 1));
	}
	while (!(gpuMask >> (idx % gpuModulus) & 1)) {
		idx++;
		if (false && checkDebug(debugTiler))
			flprintf(
					"idx %d  idx mod gpuModulus %d, gpuMask >>  (idx moid gpuModulus) & 1 %d\n",
					idx, idx % gpuModulus, gpuMask >> (idx % gpuModulus) & 1);

	}
	// if(checkDebug(debugTiler))flprintf("found bit at idx %d (started at %d)\n", idx % gpuModulus, lastGpu);
	return idx % gpuModulus;
}
template __host__ __device__ int Tiler<long>::nextGpu(int) const;
template __host__ __device__ int Tiler<unsigned long>::nextGpu(int) const;
template __host__ __device__ int Tiler<double>::nextGpu(int) const;
template __host__ __device__ int Tiler<int>::nextGpu(int) const;
template __host__ __device__ int Tiler<unsigned int>::nextGpu(int) const;
template __host__ __device__ int Tiler<float>::nextGpu(int) const;

/*
 __host__ __device__ Tiler(int forceTileCount, CuMatrix<T>& m, int gpuMask = DEFAULT_GPUMASK) : m(m), gpuMask(gpuMask){
 alloc(forceTileCount);
 }
 */
template<typename T> __host__ __device__ Tiler<T>::Tiler(const Tiler<T>& o) :
		m_m(o.m_m), m_n(o.m_n), m_p(o.m_p), m_elems(o.m_elems), m_size(
				o.m_size), tileSize(o.tileSize), /*tileP(o.tileP), */tileD(tdRows), gpuMask(o.gpuMask), gpuModulus(
				o.gpuModulus), buffers { o.buffers.x, o.buffers.y, o.buffers.z,
				o.buffers.w } {
/*
#ifndef __CUDA_ARCH__
	if(checkDebug(debugTiler))outln("Tiler<T>::Tiler(const Tiler<T>& o) -> o: " << o);
	if(checkDebug(debugTiler))outln("this " << toString());
	if(checkDebug(debugTiler))b_util::dumpStack();
#endif
*/
}
template __host__ __device__ Tiler<float>::Tiler(const Tiler<float>&);
template __host__ __device__ Tiler<double>::Tiler(const Tiler<double>&);
template __host__ __device__ Tiler<int>::Tiler(const Tiler<int>&);
template __host__ __device__ Tiler<uint>::Tiler(const Tiler<uint>&);
template __host__ __device__ Tiler<long>::Tiler(const Tiler<long>&);
template __host__ __device__ Tiler<ulong>::Tiler(const Tiler<ulong>&);

template<typename T> __host__ __device__ void Tiler<T>::allocSingleTile(bool padded) {
	int orgDevice = ExecCaps::currDev();
	//outln("osrgDevice " << orgDevice);
	int device = orgDevice;     //nxtGpu (orgDevice);
	//outln("nextGpu(orgDevice " << nxtGpu (orgDevice));
#ifndef __CUDA_ARCH__
	if(device != orgDevice) {
		ExecCaps_setDevice(device);
	}
#endif
	T* buff;
	if (checkDebug(debugMem)) {
		flprintf("before cmalloc of tileSize %lu\n", tileSize);
#ifndef __CUDA_ARCH__
		b_util::usedDmem(1);
#endif
	}

	// size_t st_m = m_m, st_n = m_n, st_p = m_p;
	// cherr(cudaMallocPitch( &buff,  &st_p, st_n, st_m));
#ifndef __CUDA_ARCH__
	size_t stFree, stTotal;
	cherr(cudaMemGetInfo(&stFree, &stTotal));
	if (checkDebug(debugTiler | debugFill | debugRefcount))flprintf("before alloc free %s \n", b_util::expNotationMem(stFree).c_str());
#endif
	int tileP;
	if(padded) {
		size_t stp;
#ifndef __CUDA_ARCH__
		cherr(cudaMallocPitch(&buff, &stp, m_n* sizeof(T), m_m));
#else
		stp = m_n * sizeof(T);
		buff = (T*) malloc(m_n* stp);
#endif
		if(checkDebug(debugTiler | debugRefcount))flprintf("created %u\n byte buff\n", m_m * stp);
	 	tileP = (int) stp / sizeof(T);
		tileSize = stp * m_m;
	} else {
		cherr( cudaMalloc( &buff, tileSize));
		tileP = m_p;
		if(checkDebug(debugTiler | debugRefcount))flprintf("created %lu byte buff with pitch(els) %d at %p\n",tileSize, tileP, buff);
	}
#ifndef __CUDA_ARCH__
	cherr(cudaMemGetInfo(&stFree, &stTotal));
	if (checkDebug(debugTiler | debugFill | debugRefcount))flprintf("dev %d allocated %ld at pitch %lu to %p %s\n", orgDevice,
			m_n* sizeof(T)* m_m, tileP, buff, b_util::expNotationMem(stFree).c_str() );
#endif
	//memblo;

	char maskBuff[34];
	if (checkDebug(debugCheckValid)) {
		flprintf("from mask %s created %dx%dx%d - %p\n",
				Tiler<T>::maskStr(maskBuff, gpuMask), m_m, m_n, m_p, buff);
#ifndef __CUDA_ARCH__
		b_util::dumpStack();
#endif
	}
#ifndef __CUDA_ARCH__
	if(checkDebug(debugMem)) {
		flprintf("after allocating %p (%lu bytes)\n", buff, tileSize);
		b_util::usedDmem(1);
		MemMgr<T>::checkValid(buff);
		MemMgr<T>::checkValid(buff + tileSize/sizeof(T) - 1);
	}
#endif
	setBuffer(device, buff);
	if (checkDebug(debugTiler))
		flprintf("created %ld-byte buffer device %d ==> %p\n", tileSize, device,
				buffer(device));
#ifndef __CUDA_ARCH__
	if(orgDevice != device) {
		ExecCaps_setDevice(orgDevice);
	}
#endif
}
/*
 * if you specify a multi gpu mask
 * 	and you specify tileMnN
 *
 */

template<typename T> __host__ __device__ bool Tiler<T>::validQ() const{
	if(buffers.x != 0 && !b_util::onDevice((void*)buffers.x, 0)) {
		return false;
	}
	if(buffers.y != 0 && !b_util::onDevice((void*)buffers.y, 1)) {
		return false;
	}
	if(buffers.z != 0 && !b_util::onDevice((void*)buffers.z, 2)) {
		return false;
	}
	if(buffers.w != 0 && !b_util::onDevice((void*)buffers.w, 3)) {
		return false;
	}
}

template<typename T> __host__ __device__ void Tiler<T>::allocTiles(int& tileM,
		int& tileN, int& tileP, bool padded, float headroom) {
	assert(m_size);
	assert(gpuMask);
	assert(!hasDmemQ());
	if (checkDebug(debugTiler))
		flprintf("tiler tileM %u tileN %u tileDir %s m_size %ld\n", tileM, tileN,
				b_util::tileDir(tileD), m_size);
	int gpuCount = countGpus();
	long maxHeadroom = (long) ExecCaps::minMaxReasonable(gpuMask, headroom);
	cherr(cudaPeekAtLastError());
#ifndef __CUDA_ARCH__
	if (checkDebug(debugTiler))
		flprintf("for mask %s and headeroom %.2f: maxHeadroom %ld\n",
				maskStr(gpuMask).c_str(), headroom, maxHeadroom);
#endif
	if (gpuCount > 1) {
		assert(DIV_UP(m_size, gpuCount) <= maxHeadroom);
		tileSize = DIV_UP(m_size, gpuCount);
		if (checkDebug(debugTiler))
			flprintf("DIV_UP(m_size %ld, gpuCount %d) * sizeof(T) %d\n", m_size,
					gpuCount, sizeof(T));
	} else {
		tileSize = tileM * tileN * sizeof(T);
	}

	if (checkDebug(debugTiler)) {
		flprintf("tiler %p->tileSize set to %d\n", this, tileSize);
	}
	if (tileSize == 0 && gpuCount== 1) {
		tileSize = m_size;
		if (checkDebug(debugTiler))
			flprintf("no spec'd tileM X tileN ==> tileSize == m_size == %ld\n",
					tileSize);
	} else {
		tileSize = m_size / gpuCount;
		if (checkDebug(debugTiler))
			flprintf("no spec'd tileM X tileN ==> tileSize == m_size/gpuCount == %ld\n",
					tileSize);
	}

	if (tileSize > maxHeadroom) {
		if (checkDebug(debugTiler))
			flprintf(
					"dev buffer %ld exceeds headroom %ld, clipping by %ld bytes\n",
					tileSize, maxHeadroom, (tileSize - maxHeadroom ));
		tileSize = maxHeadroom;
	} else if (gpuCount == 1 && tileSize == m_size) {
		if (checkDebug(debugTiler))
			prlocf("in allocTiles single\n");
		tileM=m_m;
		tileN=m_n;
		tileP=m_p;
		allocSingleTile(padded);
		return;
	}

	int orgDevice = ExecCaps::currDev();
	int device = -1;

	size_t stp;
	tileP = 0;
	if(tileD == tdCols) {
		tileM = m_m;
		tileN = DIV_UP( DIV_UP(tileSize/sizeof(T),gpuCount), m_m);
	} else {
		tileN = m_n;
		tileM = DIV_UP( DIV_UP(tileSize/sizeof(T),gpuCount), m_n);
	}
	for (int i = 0; i < gpuCount; i++) {
		device = gpuCount == 1 ? orgDevice : nextGpu(device);
		if (checkDebug(debugFill))
			flprintf("creating tile %d of %d for gpu %d\n", i + 1, gpuCount,
					device);
#ifndef __CUDA_ARCH__
		if(device != ExecCaps::currDev()) {
			ExecCaps_setDevice(device);
		}
#endif
		T* buff; //
		//if(checkDebug(debugFill))
		size_t stFree = 0, stTotal = 0;
		if(padded /*|| m_n > 1*/) {
#ifndef __CUDA_ARCH__
			cherr(cudaMemGetInfo(&stFree, &stTotal));
			flprintf("before cudaMallocPitch %s \n", b_util::expNotationMem(stFree).c_str());
			cherr(cudaMallocPitch(&buff, &stp, tileN* sizeof(T), tileM));
#else
/*
			char msgBuff[256];
			b_util::expNotationMem(msgBuff,stFree);
			flprintf("before cudaMallocPitch %s \n", msgBuff);
*/
			buff = (T*)malloc(tileN* sizeof(T) * tileM);
			stp = tileN;
#endif
			if(tileP != 0) {
				assert(tileP == (int) stp / sizeof(T));
			} else {
				tileP = (int) stp / sizeof(T);
				tileSize = stp * tileM;
			}
		} else {
#ifndef __CUDA_ARCH__
			flprintf("before cudaMalloc %s \n", b_util::expNotationMem(stFree).c_str());
#endif
			cherr(cudaMalloc(&buff, tileSize));
			tileP = m_p;
		}
#ifndef __CUDA_ARCH__
		cherr(cudaMemGetInfo(&stFree, &stTotal));
#endif

		if (checkDebug(debugFill))
			flprintf("mat: %dx%dx%d - dev : %d tile: %dX%dX%d alloced %p\n",
					m_m, m_n, m_p, device, tileM, tileN, tileP, buff);
		if (checkDebug(debugMem)) {
			flprintf("after allocating %p ( %ld bytes)\n", buff, tileSize);
#ifndef __CUDA_ARCH__
			memblo;
			flprintf("dev %d gpu %d allocated %ld to %p free now %s\n", device, ExecCaps::currDev(),
					tileSize, buff, b_util::expNotationMem(stFree).c_str());
			b_util::usedDmem();
			MemMgr<T>::checkValid(buff);
			MemMgr<T>::checkValid(buff + tileSize/sizeof(T) - 1);
#endif
		}
		if (checkDebug(debugTiler)) {
			prlocf("before setBuffer\n");
			dumpBuff(__func__);
		}
		setBuffer(device, buff);
		assert(buffer(device) == buff);
		if (checkDebug(debugTiler)) {
			prlocf("after setBuffer\n");
			dumpBuff(__func__);
		}
		if (checkDebug(debugTiler))
			flprintf(
					"allocTiles created %lu-byte buffer %d of %d on device %d ==> %p\n",
					tileSize, i + 1, gpuCount, device, buffer(device));
	}
#ifndef __CUDA_ARCH__
	if(orgDevice != device) {
		ExecCaps_setDevice(orgDevice);
	}
	if(checkDebug(debugMem))usedDevMem();
#endif
}

template __host__ __device__ void Tiler<float>::allocTiles(int&, int&, int&, bool, float);
template __host__ __device__ void Tiler<double>::allocTiles(int&, int&, int&, bool, float);
template __host__ __device__ void Tiler<int>::allocTiles(int&, int&, int&, bool, float);
template __host__ __device__ void Tiler<uint>::allocTiles(int&, int&, int&, bool, float);
template __host__ __device__ void Tiler<long>::allocTiles(int&, int&, int&, bool, float);
template __host__ __device__ void Tiler<ulong>::allocTiles(int&, int&, int&, bool, float);

template<typename T> __host__ __device__ void Tiler<T>::tileDims(int& tileM,
		int& tileN, int& tileP, TileDirection tileD) const {
	if (checkDebug(debugTiler))
		flprintf(
				"tileDims dir %s m_m %d, m_n %d,  m_p %d, tileP %d, tileSize %lu  tileSize/sizeof(T) %lu tileCount %u\n",
				b_util::tileDir(tileD), m_m, m_n, m_p, tileP, tileSize, tileSize / sizeof(T), getTileCount());

	if (tileSize >= m_size) {
		tileN = m_n;
		tileM = m_m;
		tileP = m_p;
	} else {
		if (tileD == tdRows) {
			tileN = m_n;
			if (checkDebug(debugTiler))
				prlocf("tileM bef\n");
			tileM = tileSize / (sizeof(T) * tileN);
			if (checkDebug(debugTiler))
				prlocf("tileM aft\n");
		} else {
			tileM = m_m;
			if (checkDebug(debugTiler))
				prlocf("tileN bef\n");
			tileN = tileSize / (sizeof(T) * tileM);
			if (checkDebug(debugTiler))
				prlocf("tileN aft\n");
		}
		tileP = tileN;

		if (checkDebug(debugTiler))
			flprintf("tileDims  m_m %u,  m_n %u\n", m_m, m_n);
	}
	if (checkDebug(debugTiler))
		flprintf("tileDims tileM %u, tileN %u\n", tileM, tileN);
	if (checkDebug(debugTiler))
		prlocf("exiting\n");
}
template __host__ __device__ void Tiler<int>::tileDims(int&, int&, int&,
		TileDirection) const;
template __host__ __device__ void Tiler<float>::tileDims(int&, int&, int&,
		TileDirection) const;
template __host__ __device__ void Tiler<double>::tileDims(int&, int&, int&,
		TileDirection) const;
template __host__ __device__ void Tiler<long>::tileDims(int&, int&, int&,
		TileDirection) const;
template __host__ __device__ void Tiler<ulong>::tileDims(int&, int&, int&,
		TileDirection) const;
template __host__ __device__ void Tiler<uint>::tileDims(int&, int&, int&,
		TileDirection) const;

template<typename T> __host__ __device__ void Tiler<T>::tile0(DMatrix<T>& dm,
		bool copy, cudaStream_t stream) const {
//	cherr(cudaPeekAtLastError());
	int gpuCount = countGpus();

	if (gpuCount > 1) {
#ifndef __CUDA_ARCH__
		outln("!!! gpuCount > 1:  " << gpuCount);
#endif
		setLastError(badDimensionsEx);
		return;
	}

	if(tileSize < m_size) {
#ifndef __CUDA_ARCH__
		outln("!!! matrix needs tiling; tileSize " << tileSize << " <  m_size " << m_size );
#endif
		setLastError(needsTilingEx);
		return;
	}
	if (!buffer(ExecCaps::currDev())) {
		flprintf("currDev %d %dx%dx%d h %p buff (%p,%p,%p,%p)\n",
				ExecCaps::currDev(), m_m, m_n, m_p, m_elems, buffers.x,
				buffers.y, buffers.z, buffers.w);
	}
	dm.elements = buffer(ExecCaps::currDev());

	assert(dm.elements);
	dm.m = m_m;
	dm.n = m_n;
	dm.p = m_p;
// should be constant (512 is texture alignment)
#ifndef __CUDA_ARCH__
	if(checkDebug(debugMem|debugTiler) && m_size > 512) {
		if (checkDebug(debugTiler)) flprintf("tile0 cudaMemcpy dm.elements %p, m_elements %p, m_size %lu \n", dm.elements, m_elems, m_size );
		if (checkDebug(debugTiler)) flprintf("tile0 dm.m * dm.n - 1 %d\n",dm.m * dm.n - 1);
		MemMgr<T>::checkValid(dm.elements );
		MemMgr<T>::checkValid(dm.elements + dm.m * dm.p - 1);
	}
#endif
	if (checkDebug(debugTiler))
		flprintf("dm.mXdm.n %uX%u at %p\n", dm.m, dm.n, dm.elements);

	if (copy && m_elems) {
#ifndef __CUDA_ARCH__
		if (checkDebug(debugTiler)) flprintf(
				"tile0 cudaMemcpy2DAsync dm.elements %p, spitch %lu,  m_elems %p, dpitch %lu width %lu height %d\n",
				dm.elements, dm.p*sizeof(T), m_elems, m_p*sizeof(T), m_n*sizeof(T), m_m);
//			cherr( cudaMemcpyAsync(dm.elements, m_elems, m_size, cudaMemcpyHostToDevice,stream));
		//cherr( cudaMemcpyAsync(dm.elements, m_elems, m_size, cudaMemcpyHostToDevice,stream));
		// cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
		CuTimer timer;
		timer.start();
		cherr( cudaMemcpy2DAsync(dm.elements, dm.p*sizeof(T), m_elems, m_p*sizeof(T), m_n*sizeof(T), m_m,  cudaMemcpyHostToDevice,stream));
		//CuMatrix<T>::incDhCopy("Tiler<T>::tile0", m_n*sizeof(T)* m_m,timer.stop());

#else
		if (checkDebug(debugTiler))
			prlocf("tile0 memcpy tile\n");
		cherr( cudaMemcpy2DAsync(dm.elements, dm.p*sizeof(T), m_elems, m_p*sizeof(T), m_n*sizeof(T), m_m,  cudaMemcpyHostToDevice,stream));
#endif
	}
}

template __host__ __device__ void Tiler<float>::tile0(DMatrix<float>&, bool,
		cudaStream_t) const;
template __host__ __device__ void Tiler<double>::tile0(DMatrix<double>&, bool,
		cudaStream_t) const;
template __host__ __device__ void Tiler<int>::tile0(DMatrix<int>&, bool,
		cudaStream_t) const;
template __host__ __device__ void Tiler<uint>::tile0(DMatrix<uint>&, bool,
		cudaStream_t) const;
template __host__ __device__ void Tiler<ulong>::tile0(DMatrix<ulong>&, bool,
		cudaStream_t) const;
template __host__ __device__ void Tiler<long>::tile0(DMatrix<long>&, bool,
		cudaStream_t) const;

template<typename T> __host__ __device__ void Tiler<T>::clipTile(DMatrix<T>& dm,
		int tileM, int tileN, int rowTiles, int colTiles, int rowTileIdx,
		int colTileIdx) const {
	if (checkDebug(debugTiler))
		flprintf(
				"tileM %u tileN %u rowTiles %d colTiles %d rowTileIdx %d colTileIdx %d\n",
				tileM, tileN, rowTiles, colTiles, rowTileIdx, colTileIdx);
	if (rowTiles > 1 && rowTileIdx == rowTiles - 1) {
		if (checkDebug(debugTiler))
			prlocf("rowTiles > 1 && rowTileIdx == rowTiles - 1\n");
		dm.m = m_m - rowTileIdx * tileM; // in above ex., m_m == 5, dm.m == 2, rowTileIdx == 2 --> rowPatchHeight = 1;
	} else {
		if (checkDebug(debugTiler))
			prlocf("! (rowTiles > 1 && rowTileIdx == rowTiles - 1)\n");
		if (tileM > m_m) {
			if (checkDebug(debugTiler))
				flprintf(
						"warning, tileM %u is bigger than hostbuff's m_m %u!\n",
						tileM, m_m);
		}
		dm.m = MIN(tileM, m_m);
	}
	if (colTiles > 1 && colTileIdx == colTiles - 1) {
		if (checkDebug(debugTiler))
			prlocf("colTiles > 1 && colTileIdx == colTiles - 1\n");
		dm.n = m_n - colTileIdx * tileN; // in above ex., m_n == 5, dm.n == 2, colTileIdx == 2 --> colPatchWidth = 1
	} else {
		if (checkDebug(debugTiler))
			prlocf("!(colTiles > 1 && colTileIdx == colTiles - 1)\n");
		dm.n = tileN;
	}
	if (dm.n > tileN) {
		if (checkDebug(debugTiler))
			flprintf("warning, tileN %u is bigger than hostbuff's m_n %u!\n",
					tileN, m_n);
	}
	dm.n = MIN(dm.n, m_n);
}

template __host__ __device__ void Tiler<float>::clipTile(DMatrix<float>&, int,
		int, int, int, int, int) const;
template __host__ __device__ void Tiler<double>::clipTile(DMatrix<double>&,
		int, int, int, int, int, int) const;
template __host__ __device__ void Tiler<int>::clipTile(DMatrix<int>&, int,
		int, int, int, int, int) const;
template __host__ __device__ void Tiler<uint>::clipTile(DMatrix<uint>&, int,
		int, int, int, int, int) const;
template __host__ __device__ void Tiler<ulong>::clipTile(DMatrix<ulong>&, int,
		int, int, int, int, int) const;
template __host__ __device__ void Tiler<long>::clipTile(DMatrix<long>&, int,
		int, int, int, int, int) const;

template<typename T> __host__ __device__ int Tiler<T>::tile1D(DMatrix<T>& dm,
		int& roff, int& coff, int& tileM, int& tileN,  int& tileP, int t,
		TileDirection tileD, bool copy, int lastGpu,
		cudaStream_t stream) const {
	int tileCount = getTileCount();
	if (tileM == 0 || tileN == 0) {
		if (checkDebug(debugTiler))
			prlocf("tileM or tileN zero, calling tileDims\n");
		tileDims(tileM, tileN, tileP, tileD);
	} else {
		if (tileD == tdRows)
			tileCount = MAX(tileCount, DIV_UP(m_m, tileM));
		else
			tileCount = MAX(tileCount, DIV_UP(m_n, tileN));
	}
	if (t < 0 || t >= tileCount) {
		setLastError(illegalArgumentEx);
		return -1;
	}

	if (checkDebug(debugTiler))
		flprintf(
				"tile1D enter roff %u coff %u tileM %u tileN %u tileCount %d\n",
				roff, coff, tileM, tileN, tileCount);
	roff = (tileD == tdRows) ? t * tileM : 0;
	coff = (tileD == tdCols) ? t * tileN : 0;

	int orgDevice = ExecCaps::currDev();
	lastGpu = nextGpu(lastGpu);
	if (checkDebug(debugTiler)) {
		flprintf("before buffer(lastGpu %d)\n", lastGpu);
		dumpBuff(__func__);
	}
	dm.elements = buffer(lastGpu);

	if (checkDebug(debugTiler))
		flprintf("before clipTile tileM %u tileN %u\n", tileM, tileN);
	clipTile(dm, tileM, tileN, tileD == tdRows ? tileCount : 1,
			tileD == tdCols ? tileCount : 1, tileD == tdRows ? t : 0,
			tileD == tdCols ? t : 0);
	if (checkDebug(debugTiler))
		flprintf("after clipTile dm.m %u dm.n %u\n", dm.m, dm.n);
	if (tileCount != 1)
		dm.p = dm.n;
	else
		dm.p = tileP;
	int gpuCount = countGpus();
	long thisTileSize = (long)dm.m * tileP * sizeof(T);

	if (checkDebug(debugTiler))
		flprintf("thisTileSize tile %ld\n", thisTileSize);
	if (checkDebug(debugTiler))
		flprintf("t %d gpuCnt %d lastGpu %d at roff %u coff%u\n", t, gpuCount,
				lastGpu, roff, coff);
	long hoffset = offset(roff, coff);
	if (copy) {
#ifndef __CUDA_ARCH__
		CuTimer timer;
		timer.start();
		if (checkDebug(debugTiler)) flprintf(
				"cudaMemcpy tile %d m_elems %p dm.elements %p thisTileSize %lu loffset %lu\n",
				t , m_elems, dm.elements, thisTileSize,hoffset);
		if(lastGpu > -1 && lastGpu != orgDevice) {
			ExecCaps_setDevice(lastGpu);
		}
		MemMgr<T>::checkValid(dm.elements);
		MemMgr<T>::checkValid(dm.elements + thisTileSize/sizeof(T)-1);
		if(pinnedQ) MemMgr<T>::checkValid(m_elems +hoffset );
		if(pinnedQ) MemMgr<T>::checkValid(m_elems + hoffset + thisTileSize/sizeof(T)-1);
#endif

		if (tileD == tdRows) {
#ifndef __CUDA_ARCH__
//			cherr( cudaMemcpyAsync(dm.elements, m_elems + loffset, thisTileSize, cudaMemcpyHostToDevice,stream));
			if (checkDebug(debugCopy))flprintf("dev %d m_elems %p + offset(roff, coff) %d, m_p * sizeof(T) %d \
							dm.elements %p, tileP*sizeof(T) %d, dm.n * sizeof(T) %d, dm.m %d\n",
							ExecCaps::currDev(), m_elems, offset(roff, coff), m_p * sizeof(T),
														dm.elements, tileP*sizeof(T), dm.n * sizeof(T), dm.m);
			cherr(
					cudaMemcpy2DAsync(dm.elements, tileP*sizeof(T),
							m_elems + offset(roff, coff), m_p * sizeof(T),dm.n * sizeof(T), dm.m,
							cudaMemcpyHostToDevice, stream));
			//CuMatrix<T>::incDhCopy("Tiler<T>::tile1D",dm.n * sizeof(T)* dm.m,timer.stop());
#else
			if (checkDebug(debugTiler))
				flprintf("tile1D memcpy tile %d\n", t);
			memcpy(dm.elements, m_elems + offset(roff, coff), thisTileSize);
#endif
		} else {
			cherr(
					cudaMemcpy2DAsync(dm.elements, tileP*sizeof(T),
							m_elems + offset(roff, coff), m_p * sizeof(T),dm.n * sizeof(T), dm.m,
							cudaMemcpyHostToDevice, stream));

		}
#ifndef __CUDA_ARCH__
		if(lastGpu > -1 && lastGpu != orgDevice) {
			ExecCaps_setDevice(orgDevice);
		}
#endif
	}
	return lastGpu;
}

template __host__ __device__ int Tiler<float>::tile1D(DMatrix<float>&, int&,
		int&, int&, int&, int&, int, TileDirection, bool, int, cudaStream_t) const;
template __host__ __device__ int Tiler<double>::tile1D(DMatrix<double>&, int&,
		int&, int&, int&, int&, int, TileDirection, bool, int, cudaStream_t) const;
template __host__ __device__ int Tiler<int>::tile1D(DMatrix<int>&, int&, int&,
		int&, int&, int&, int, TileDirection, bool, int, cudaStream_t) const;
template __host__ __device__ int Tiler<uint>::tile1D(DMatrix<uint>&, int&,
		int&, int&, int&, int&, int, TileDirection, bool, int, cudaStream_t) const;
template __host__ __device__ int Tiler<ulong>::tile1D(DMatrix<ulong>&, int&,
		int&, int&, int&, int&, int, TileDirection, bool, int, cudaStream_t) const;
template __host__ __device__ int Tiler<long>::tile1D(DMatrix<long>&, int&,
		int&, int&, int&, int&, int, TileDirection, bool, int, cudaStream_t) const;

template<typename T> __host__ __device__ int Tiler<T>::tile2D(DMatrix<T>& dm,
		int& roff, int& coff, int& tileM, int& tileN, int& tileP, int rowTileCount,
		int colTileCount, int rowTileIdx, int colTileIdx, bool copy,
		int lastGpu, cudaStream_t stream) const {
	if (checkDebug(debugTiler))
		flprintf(
				"dm.elements %p, roff %u  coff %u tileM %u tileN %u rowTileIdx %d colTileIdx %d\n",
				dm.elements, roff, coff, tileM, tileN, rowTileIdx, colTileIdx);
	if (checkDebug(debugTiler))
		flprintf("rowTileCount %u colTileCount %u\n", rowTileCount,
				colTileCount);
	if (rowTileIdx < 0 || rowTileIdx >= rowTileCount || colTileIdx < 0
			|| colTileIdx >= colTileCount) {
		setLastError(illegalArgumentEx);
		return -1;
	}

	if (tileM == 0 || tileN == 0)
		tileDims(tileM, tileN, tileP, tdBoth);

	roff = (rowTileCount > 1) ? rowTileIdx * tileM : 0;
	coff = (colTileCount > 1) ? colTileIdx * tileN : 0;

	int orgDevice = ExecCaps::currDev();
	int gpuCount = countGpus();
	if (gpuCount > 1 && lastGpu != -1) {
		lastGpu = nextGpu(lastGpu);
	}
	dm.elements = buffer(lastGpu);
	ulong thisTileSize = dm.m * dm.p * sizeof(T);
	if (checkDebug(debugTiler))
		flprintf(
				"tile2D bef (%d,%d) gpuCnt %d gpuCurr %d timeMN %uX%u at rcoff %uX%u\n",
				rowTileIdx, colTileIdx, gpuCount, lastGpu, tileM, tileN, roff,
				coff);
	if (checkDebug(debugTiler))
		flprintf("tile2D bef dm.mn  %uX%u\n", dm.m, dm.n);
	if (checkDebug(debugTiler))
		flprintf("tile2D bef thisTileSize tile %lu\n", thisTileSize);
	clipTile(dm, tileM, tileN, rowTileCount, colTileCount, rowTileIdx,
			colTileIdx);
	dm.p = dm.n;
	thisTileSize = dm.m * dm.p * sizeof(T);

	if (checkDebug(debugTiler))
		flprintf("tile2D aft thisTileSize tile %lu\n", thisTileSize);
	if (checkDebug(debugTiler))
		flprintf("tile2D aft dm.mn  %uX%u\n", dm.m, dm.n);
	if (checkDebug(debugTiler))
		flprintf(
				"tile2D aft (%d,%d) gpuCnt %d gpuCurr %d timeMN %uX%u at rcoff %uX%u\n",
				rowTileIdx, colTileIdx, gpuCount, lastGpu, tileM, tileN, roff,
				coff);

	if (copy) {
#ifndef __CUDA_ARCH__
		if (checkDebug(debugTiler)) flprintf("tile2D cudaMemcpy tile (%d,%d) m_elems %p dm.elements %p thisTileSize %lu offset(roff, coff) %d\n", rowTileIdx, colTileIdx, m_elems, dm.elements, thisTileSize,offset(roff, coff));
		MemMgr<T>::checkValid(dm.elements);
		MemMgr<T>::checkValid(dm.elements + thisTileSize/sizeof(T)-1);
		if(pinnedQ) MemMgr<T>::checkValid(m_elems);
		if(pinnedQ) MemMgr<T>::checkValid(m_elems + thisTileSize/sizeof(T)-1);
		if(lastGpu > -1 && lastGpu != orgDevice) {
			ExecCaps_setDevice(lastGpu);
		}
//		cherr( cudaMemcpyAsync(dm.elements, m_elems +offset(roff, coff), thisTileSize, cudaMemcpyHostToDevice,stream));


		CuTimer timer;
		timer.start();
		cherr(
				cudaMemcpy2DAsync(m_elems + offset(roff, coff), m_p * sizeof(T),
						dm.elements, tileP *sizeof(T), dm.n * sizeof(T), dm.m,
						cudaMemcpyDeviceToHost, stream));
		//CuMatrix<T>::incDhCopy("Tiler<T>::tile2D",dm.n * sizeof(T)* dm.m, timer.stop());
		if (checkDebug(debugTiler))
			flprintf("tile2D memcpy rowTileIdx %d colTileIdx %d\n", rowTileIdx,
					colTileIdx);
#else
		memcpy(dm.elements, m_elems + offset(roff, coff), thisTileSize);
#endif
#ifndef __CUDA_ARCH__
		if(lastGpu > -1 && orgDevice != lastGpu) {
			ExecCaps_setDevice(orgDevice);
		}
#endif
	}
	return lastGpu;
}
template __host__ __device__ int Tiler<float>::tile2D(DMatrix<float>&, int&,
		int&, int&, int&, int&, int, int, int, int, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<double>::tile2D(DMatrix<double>&, int&,
		int&, int&, int&, int&, int, int, int, int, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<int>::tile2D(DMatrix<int>&, int&, int&,
		int&, int&, int&, int, int, int, int, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<uint>::tile2D(DMatrix<uint>&, int&,
		int&, int&, int&, int&, int, int, int, int, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<ulong>::tile2D(DMatrix<ulong>&, int&,
		int&, int&, int&, int&, int, int, int, int, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<long>::tile2D(DMatrix<long>&, int&,
		int&, int&, int&, int&, int, int, int, int, bool, int, CUstream_st*) const;

template<typename T> __host__ __device__ int Tiler<T>::tileLike(DMatrix<T>& dm,
		int& roff, int& coff, int tileM, int tileN, int tileP, int t,
		TileDirection tileD, bool copy, int lastGpu,
		cudaStream_t* streams) const {
	//int nTiles = getTileCount();
	if (checkDebug(debugTiler))
		flprintf("tileLike entre lastGpu %d, currDev %d\n",lastGpu, ExecCaps::currDev());

	if (t < 0 || t >= MAX(DIV_UP(m_m,tileM), DIV_UP(m_n,tileN))) {
		flprintf("t %d MAX(DIV_UP(m_m,tileM), DIV_UP(m_n,tileN))  %d\n", t,
				MAX(DIV_UP(m_m,tileM), DIV_UP(m_n,tileN)));
		flprintf("m_m %u tileM %u, m_n %u  tileN %u\n", m_m, tileM, m_n, tileN);
		setLastError(illegalArgumentEx);
		return -1;
	}
	//int orgDevice = ExecCaps::currDev();
	int gpuCount = countGpus();
	if (gpuCount > 1 && lastGpu != -1) {
		lastGpu = nextGpu(lastGpu);
	} else if (lastGpu == -1) {
		lastGpu = nextGpu();
	}
	int idx =  indexInMask(lastGpu);
	cudaStream_t stream = streams == nullptr ? nullptr : streams[idx];
	if (checkDebug(debugTiler)) {
		flprintf("tiler %p\n",this);
		dumpMask(gpuMask);
		flprintf("streams %p lastGpu %d -> indexInMask %d -> stream %p\n", streams, lastGpu, idx, stream);
	}
	dm.elements = buffer(lastGpu);
	if (checkDebug(debugMem))
		dumpBuff(__func__);
	if (checkDebug(debugTiler))
		flprintf("m_m %u m_n %u m_p %u lastGpu %u m_elems %p dm.elements %p\n",
				m_m, m_n, m_p, lastGpu, m_elems, dm.elements);

	if (checkDebug(debugTiler))
		flprintf(
				"tileLike bef dm.m %u dm.n %u clip  DIV_UP(m_m,tileM) %d DIV_UP(m_n,tileN) %d tileM %u tileN %u\n",
				dm.m, dm.n, DIV_UP(m_m,tileM), DIV_UP(m_n,tileN), tileM, tileN);
	clipTile(dm, tileM, tileN, DIV_UP(m_m, tileM), DIV_UP(m_n, tileN),
			tileD == tdRows ? t : 0, tileD == tdCols ? t : 0);
	if (checkDebug(debugTiler))
		flprintf("tileLike aft  dm.m %u dm.n %u clip tileM %u tileN %u\n", dm.m,
				dm.n, tileM, tileN);
	dm.p = tileP;
	ulong thisTileSize = dm.m * dm.p * sizeof(T);

	roff = (tileD == tdRows) ? t * tileM : 0;
	coff = (tileD == tdCols) ? t * tileN : 0;
	ulong hoffset = offset(roff, coff);
	ulong doffset = offsetd(roff, coff, tileP);
	if (checkDebug(debugTiler))
		flprintf(
				"tileLike thisTileSize dm %uX%uX%u, roff %u coff %u thisTileSize %lu hoffset %lu doffset %lu stream %p\n",
				dm.m, dm.n, dm.p, roff, coff, thisTileSize, hoffset, doffset, stream);
	if (copy) {
		if (lastGpu > -1 && lastGpu != ExecCaps::currDev()) {
			ExecCaps_setDevice(lastGpu);
		}
#ifndef __CUDA_ARCH__

		if (checkDebug(debugCopy))flprintf("tileLike chekin valid, m_elems + m_size(==%d)/sizeof(T) - 1\n",m_size);
		MemMgr<T>::checkValid(m_elems + (m_size/sizeof(T) -1));

		if (checkDebug(debugCopy))
		flprintf("tileLike chekin valid, dm.elements %p\n", dm.elements);

		MemMgr<T>::checkValid(dm.elements);
		if (checkDebug(debugMeans)) {

			pthread_t currThread = pthread_self ();
			outln("currThread " << currThread);
			flprintf("tileLike chekin valid, dm.elements (== %p) + thisTileSize/sizeof(T) -1\n",dm.elements);
			MemMgr<T>::checkRange(dm.elements, (thisTileSize/sizeof(T) -1));
		}
		if (checkDebug(debugCopy))flprintf("doffset %lu thisTileSize/sizeof(T) -1 %lu\n",doffset, thisTileSize/sizeof(T) -1);
		if (checkDebug(debugCopy))flprintf("tileLike chekd dm.elements + thisTileSize/sizeof(T) %p\n",dm.elements + thisTileSize/sizeof(T));
		MemMgr<T>::checkValid(m_elems + hoffset);
		if (checkDebug(debugCopy))flprintf("tileLike chekd m_elems + hoffset %p\n",m_elems + hoffset);
		MemMgr<T>::checkValid(m_elems + hoffset + thisTileSize/sizeof(T) -1);
		if (checkDebug(debugCopy))flprintf("tileLike chekd m_elems + loffset + thisTileSize/sizeof(T) %p\n",m_elems + hoffset + thisTileSize/sizeof(T));
		cherr(cudaPeekAtLastError());

		if(stream) {
			if (checkDebug(debugTiler))flprintf("cudaMemcpy2DAsync copying from %p (hoffset %u = %p) to %p with stream %p\n", m_elems, hoffset, m_elems + hoffset, dm.elements, stream );
			cherr( cudaMemcpy2DAsync(dm.elements, dm.p * sizeof(T), m_elems +hoffset, m_p* sizeof(T), dm.n* sizeof(T), dm.m, cudaMemcpyHostToDevice,stream));
		} else {
			if (checkDebug(debugTiler)) {
				outln(*this);
			}
			if (checkDebug(debugCopy | debugTiler))flprintf(
					"cudaMemcpy2D copying from %p (hoffset %u = %p) to %p\n",
					m_elems, hoffset, m_elems + hoffset,
					dm.elements);
			//(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
			MemMgr<T>::checkValid(dm.elements);
			MemMgr<T>::checkValid(m_elems);
			if (checkDebug(debugCopy | debugTiler))outln("tileP * sizeof(T) * dm.m  "<< (tileP * sizeof(T) * dm.m ));
			MemMgr<T>::checkValid(dm.elements + tileP * dm.m - 1 );
			if (checkDebug(debugCopy | debugTiler))outln("dm.p * sizeof(T) * dm.m  "<< (m_p * sizeof(T) * dm.m ));

			/*
			 * cudaMemcpy2D(void *dst, size_t dpitch,
			 * 				const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
			 *
			 */
			MemMgr<T>::checkValid(m_elems + dm.n * sizeof(T) * dm.m - 1 );
				cudaError_t res=
					cudaMemcpy2D(dm.elements, tileP * sizeof(T),
							m_elems+ hoffset, m_p* sizeof(T), dm.n * sizeof(T), dm.m,
							cudaMemcpyHostToDevice);
			if(res != cudaSuccess) {
				flprintf(
				"cudaMemcpy2D failed with %d on dev %d while copying trg %p tpitch %d src %p spitch %d wBytes %d height %d\n",
				res, ExecCaps::currDev(),
				dm.elements, tileP * sizeof(T), m_elems+ hoffset, m_p* sizeof(T), dm.n * sizeof(T), dm.m);
				cherr(res);
			}
		}
#else
		if (checkDebug(debugCopy))
			prlocf("tileLike memcpy");
		memcpy(dm.elements, m_elems + hoffset, thisTileSize);
#endif
/*
		if (lastGpu > -1 && orgDevice != lastGpu) {
			ExecCaps_setDevice(orgDevice);
		}
*/
	}
	return lastGpu;
}

template __host__ __device__ int Tiler<float>::tileLike(DMatrix<float>&, int&,
		int&, int, int, int, int, TileDirection, bool, int, CUstream_st**) const;
template __host__ __device__ int Tiler<double>::tileLike(DMatrix<double>&,
		int&, int&, int, int, int, int, TileDirection, bool, int,
		CUstream_st**) const;
template __host__ __device__ int Tiler<int>::tileLike(DMatrix<int>&, int&,
		int&, int, int, int, int, TileDirection, bool, int, CUstream_st**) const;
template __host__ __device__ int Tiler<uint>::tileLike(DMatrix<uint>&, int&,
		int&, int, int, int, int, TileDirection, bool, int, CUstream_st**) const;
template __host__ __device__ int Tiler<ulong>::tileLike(DMatrix<ulong>&, int&,
		int&, int, int, int, int, TileDirection, bool, int, CUstream_st**) const;
template __host__ __device__ int Tiler<long>::tileLike(DMatrix<long>&, int&,
		int&, int, int, int, int, TileDirection, bool, int, CUstream_st**) const;


template<typename T> __host__ __device__ void Tiler<T>::syncTile(const DMatrix<T>&dm, int roff, int coff, cudaStream_t stream) {
    if(checkDebug(debugTiler)) {
        flprintf("syncTile dm %uX%uX%u dm.elements %p roff %d coff %d\n", dm.m, dm.n, dm.p, dm.elements, roff, coff);
    }
#ifndef __CUDA_ARCH__
    if (checkDebug(debugTiler))flprintf("dpitchB(m_p) %u spitchB(dm.p) %u widthB(dm.n) %u height(dm.m) %u\n",  m_p* sizeof(T), dm.p* sizeof(T), dm.n* sizeof(T), dm.m );
	CuTimer timer;
	timer.start();
    if(stream) {
    	cherr(cudaDeviceSynchronize());
    	MemMgr<T>::checkValid(m_elems);
    	MemMgr<T>::checkValid(m_elems + (dm.m -1) * m_p + dm.n);
    	MemMgr<T>::checkValid(m_elems + dm.m * m_p - 1);

      	MemMgr<T>::checkValid(dm.elements);
       	MemMgr<T>::checkValid(dm.elements  + (dm.m -1) * dm.p + dm.n);
    	MemMgr<T>::checkValid(dm.elements  + dm.m * dm.p - 1);

        flprintf("dev %d cudaMemcpy2DAsync roff %u coff %u copying from %p to %p with stream %p\n", ExecCaps::currDev(), roff, coff, dm.elements, m_elems +offset(roff, coff), stream );
        cherr( cudaMemcpy2DAsync(m_elems +offset(roff, coff), m_p * sizeof(T), dm.elements, dm.p* sizeof(T), dm.n* sizeof(T), dm.m, cudaMemcpyDeviceToHost,stream));
		//CuMatrix<T>::incDhCopy("Tiler<T>::syncTile-" + b_util::caller(), dm.n* sizeof(T)* dm.m, timer.stop());
   }else {
        if (checkDebug(debugTiler))flprintf("cudaMemcpy2D roff %u coff %u copying from %p to %p with stream %p\n", roff, coff, dm.elements, m_elems +offset(roff, coff), stream );
        if (checkDebug(debugTiler))flprintf(
                "cudaMemcpy2D m_elems +offset(roff, coff) %p, m_p* sizeof(T) %u, dm.elements %p, dm.p* sizeof(T) %u, dm.n* sizeof(T) %u, dm.m %u\n",
                m_elems +offset(roff, coff), m_p* sizeof(T), dm.elements, dm.p* sizeof(T), dm.n* sizeof(T), dm.m);
        if (checkDebug(debugTiler|debugFill)) {
            flprintf(
                "void *dst %p, size_t dpitch %u, targ end %p\n",
                m_elems +offset(roff, coff), m_p* sizeof(T), m_elems +offset(roff, coff) + dm.m * m_p + dm.n);
            flprintf(
                "const void *src %p, size_t spitch %u, size_t width %u, size_t height %u\n",
                dm.elements, dm.p* sizeof(T), dm.n* sizeof(T), dm.m);
            MemMgr<T>::checkValid( dm.elements);
            MemMgr<T>::checkValid( dm.elements + dm.p * dm.m - 1);
        }
        cherr( cudaMemcpy2D(m_elems +offset(roff, coff), m_p* sizeof(T), dm.elements, dm.p* sizeof(T), dm.n* sizeof(T), dm.m, cudaMemcpyDeviceToHost));
		//CuMatrix<T>::incDhCopy("Tiler<T>::syncTile-" + b_util::caller(), dm.n* sizeof(T)* dm.m,timer.stop() );
    }
#else
    if (checkDebug(debugTiler))printf("syncTile memcpy, synching stream %p\n",stream);
    memcpy2D(m_elems +offset(roff, coff), m_p, dm.elements, dm.p, dm.n, dm.m);
#endif

}


 template __host__ __device__ void Tiler<float>::syncTile(const DMatrix<float>&, int, int, cudaStream_t) ;
 template __host__ __device__ void Tiler<double>::syncTile(const DMatrix<double>&, int, int, cudaStream_t) ;
 template __host__ __device__ void Tiler<int>::syncTile(const DMatrix<int>&, int, int, cudaStream_t) ;
 template __host__ __device__ void Tiler<uint>::syncTile(const DMatrix<uint>&, int, int, cudaStream_t) ;
 template __host__ __device__ void Tiler<ulong>::syncTile(const DMatrix<ulong>&, int, int, cudaStream_t) ;
 template __host__ __device__ void Tiler<long>::syncTile(const DMatrix<long>&, int, int, cudaStream_t) ;
