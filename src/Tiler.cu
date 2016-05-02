/* *
 */
#include "Tiler.h"
#include "CuMatrix.h"
#include "caps.h"
#include <pthread.h>


template<typename T> __host__ __device__ bool Tiler<T>::hasDmemQ() const {
	T* curr = currBuffer();
	if(checkDebug(debugTiler)) flprintf("curr %p\n", curr);
	if(curr == null)
		return false;
#ifndef __CUDA_ARCH__
	return MemMgr<T>::checkValid(curr);
#else
	return false;
#endif

}
template __host__ __device__ bool Tiler<float>::hasDmemQ( ) const;
template __host__ __device__ bool Tiler<double>::hasDmemQ()const;
template __host__ __device__ bool Tiler<int>::hasDmemQ()const;
template __host__ __device__ bool Tiler<uint>::hasDmemQ()const;
template __host__ __device__ bool Tiler<long>::hasDmemQ()const;
template __host__ __device__ bool Tiler<ulong>::hasDmemQ()const;


//todo set gpuMask from m's device via getPtrAtts
template<typename T> __host__ __device__ Tiler<T>::Tiler(const CuMatrix<T>& m) :
		m_m(m.m), m_n(m.n), m_p(m.p), m_elems(m.elements), m_size(m.size), tileSize(0),
		gpuMask(gpuMaskFromGpu(ExecCaps::currDev())), gpuModulus(modulusFromMask()),
		buffers { 0, 0, 0, 0 } {
	if (checkDebug(debugTiler)) {
		flprintf("Tiler<T>::Tiler(CuMatrix<T>& m) -> this %p, &m (m : ",this,&m);
		m.print("for Tiler:Tiler(CuMatrix<T>&)");
	}
#ifndef __CUDA_ARCH__
		if(checkDebug(debugTiler))b_util::dumpStack();
#endif
}

template __host__ __device__ Tiler<float>::Tiler( const CuMatrix<float>&);
template __host__ __device__ Tiler<double>::Tiler(const  CuMatrix<double>&);
template __host__ __device__ Tiler<int>::Tiler(const  CuMatrix<int>&);
template __host__ __device__ Tiler<uint>::Tiler(const  CuMatrix<uint>&);
template __host__ __device__ Tiler<long>::Tiler(const  CuMatrix<long>&);
template __host__ __device__ Tiler<ulong>::Tiler(const  CuMatrix<ulong>&);

template<typename T> __host__ __device__ Tiler<T>::Tiler(const CuMatrix<T>& m, bool allocate, int gpuMask) :
		m_m(m.m), m_n(m.n), m_p(m.p), m_elems(m.elements), m_size(m.size),
		gpuMask(gpuMask), gpuModulus(modulusFromMask()),
		buffers { 0, 0, 0, 0 } {
	if(checkDebug(debugTiler)) {
		flprintf("(m(h:%p,%uX%uX%u sz: %lu), alloc %s, mask %d) on device %d -> this %p \n", m.elements, m.m, m.n,m.p, m.size, tOrF(allocate), gpuMask, ExecCaps::currDev(),this);
		prlocf("\tm : \n");
		m.print("Tiler:Tiler");
	}
	if(allocate) {
		allocTiles();
	}
}

template __host__ __device__ Tiler<float>::Tiler( const CuMatrix<float>&,bool,int);
template __host__ __device__ Tiler<double>::Tiler( const CuMatrix<double>&,bool,int);
template __host__ __device__ Tiler<int>::Tiler( const CuMatrix<int>&,bool,int);
template __host__ __device__ Tiler<uint>::Tiler(const  CuMatrix<uint>&,bool,int);
template __host__ __device__ Tiler<ulong>::Tiler(const  CuMatrix<ulong>&,bool,int);
template __host__ __device__ Tiler<long>::Tiler(const  CuMatrix<long>&,bool,int);

template<typename T> __host__ __device__ int Tiler<T>::deviceOfResidence() const {
  	if( !( buffers.x || buffers.y || buffers.z || buffers.w))
  		return -1;
  	if( (buffers.x != 0)+ (buffers.y != 0)+ (buffers.z != 0 ) + (buffers.w != 0) >1 )
  		setLastError(multipleGpusEx);

  	return nextGpu();
}

template __host__ __device__ int Tiler<float>::deviceOfResidence()const;
template __host__ __device__ int Tiler<double>::deviceOfResidence()const;
template __host__ __device__ int Tiler<int>::deviceOfResidence( )const;
template __host__ __device__ int Tiler<uint>::deviceOfResidence()const;
template __host__ __device__ int Tiler<long>::deviceOfResidence()const;
template __host__ __device__ int Tiler<ulong>::deviceOfResidence()const;


template<typename T> __host__ __device__ void Tiler<T>::set(const CuMatrix<T>& m) {
	m_m = m.m;
	m_n = m.n;
	m_p = m.p;
	m_elems = m.elements;
	m_size = m.size;
	gpuMask = gpuMaskFromGpu(ExecCaps::currDev());
	gpuModulus = modulusFromMask();
	buffers = { 0, 0, 0, 0 };
}
template __host__ __device__ void Tiler<float>::set( const CuMatrix<float>&);
template __host__ __device__ void Tiler<double>::set( const CuMatrix<double>&);
template __host__ __device__ void Tiler<int>::set( const CuMatrix<int>&);
template __host__ __device__ void Tiler<uint>::set(const  CuMatrix<uint>&);
template __host__ __device__ void Tiler<long>::set(const  CuMatrix<long>&);
template __host__ __device__ void Tiler<ulong>::set(const  CuMatrix<ulong>&);

template<typename T> __host__ __device__ int Tiler<T>::nextGpu(int lastGpu ) const {
	assert(gpuModulus);
	//if(checkDebug(debugTiler))prlocf("\n\nnextGPU ent\n\n");
     if (checkDebug(debugTiler))
         dumpMask(gpuMask);
     uint idx = lastGpu;
     if(checkDebug(debugTiler)) flprintf("gpuModulus %d currdev %d\n",gpuModulus, ExecCaps::currDev());
     while( !( gpuMask >>  (idx % gpuModulus) & 1)) {
     	idx++;
     	if(checkDebug(debugTiler))flprintf("idx %d  idx mod gpuModulus %d, gpuMask >>  (idx moid gpuModulus) & 1 %d\n",
     			idx,  idx % gpuModulus, gpuMask >>  (idx % gpuModulus) & 1);

     }
     if(checkDebug(debugTiler))flprintf("found bit at idx %d (started at %d)\n", idx % gpuModulus, lastGpu);
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
				o.m_size),	tileSize(o.tileSize), gpuMask(o.gpuMask), gpuModulus(o.gpuModulus),
				buffers { o.buffers.x, o.buffers.y, o.buffers.z,
				o.buffers.w } {
#ifndef __CUDA_ARCH__
		if(checkDebug(debugTiler))b_util::dumpStack();
#endif
}
template __host__ __device__ Tiler<float>::Tiler( const Tiler<float>&);
template __host__ __device__ Tiler<double>::Tiler( const Tiler<double>&);
template __host__ __device__ Tiler<int>::Tiler( const Tiler<int>&);
template __host__ __device__ Tiler<uint>::Tiler( const Tiler<uint>&);
template __host__ __device__ Tiler<long>::Tiler( const Tiler<long>&);
template __host__ __device__ Tiler<ulong>::Tiler( const Tiler<ulong>&);


template<typename T> __host__ __device__ void Tiler<T>::allocOneTile(){
    tileSize = m_size;
    int orgDevice = ExecCaps::currDev();
    //outln("osrgDevice " << orgDevice);
    int device = orgDevice;//nxtGpu (orgDevice);
    //outln("nextGpu(orgDevice " << nxtGpu (orgDevice));
#ifndef __CUDA_ARCH__
    if(device != orgDevice) {
        ExecCaps_setDevice(device);
    }
#endif
    T* buff;
    if(checkDebug(debugMem)){
        flprintf("before cmalloc of tileSize %lu\n",tileSize);
#ifndef __CUDA_ARCH__
        b_util::usedDmem(1);
#endif
    }
    cherr(cudaMalloc( &buff, tileSize));
    char maskBuff[34];
    if(checkDebug(debugCheckValid)) {
    	flprintf("from mask %s created %dx%dx%d - %p\n",  Tiler<T>::maskStr(maskBuff,gpuMask), m_m, m_n, m_p, buff);
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
    if(checkDebug(debugTiler))
            flprintf("created %ld-byte buffer device %d ==> %p\n",tileSize, device, buffer(device) );
#ifndef __CUDA_ARCH__
    if(orgDevice != device) {
        ExecCaps_setDevice(orgDevice);
    }
#endif
}
template __host__ __device__ void Tiler<long>::allocOneTile();
template __host__ __device__ void Tiler<unsigned long>::allocOneTile();
template __host__ __device__ void Tiler<double>::allocOneTile();
template __host__ __device__ void Tiler<int>::allocOneTile();
template __host__ __device__ void Tiler<unsigned int>::allocOneTile();
template __host__ __device__ void Tiler<float>::allocOneTile();

/*
 * if you specify a multi gpu mask
 * 	and you specify tileMnN
 *
 */

/* Inst{float, double, long, int, u} */
template<typename T> __host__ __device__ void Tiler<T>::allocTiles( uint tileM,  uint tileN, float headroom) {
	assert(m_size);
	assert(gpuMask);
	assert(!hasDmemQ());

	int gpuCount = countGpus();
	ulong maxHeadroom = ExecCaps::minMaxReasonable(gpuMask,headroom);

	tileSize = tileM * tileN * sizeof(T);

	if(tileSize == 0) {
		tileSize = m_size;
		if(checkDebug(debugTiler))flprintf("no spec'd tileM X tileN ==> tileSize == m_size == %lu\n", tileSize);
	}

	if(tileSize > maxHeadroom) {
		if(checkDebug(debugTiler))flprintf("dev buffer exceeds headroom, clipping by %lu bytes\n", (maxHeadroom - tileSize));
		tileSize = maxHeadroom;
	} else if(gpuCount == 1 && tileSize == m_size) {
		if(checkDebug(debugTiler))prlocf("in allocTiles single\n");
		allocOneTile();
		return;
	}


	// size tile, for now, use smallest buffer size when multi-gpu
	/* need to
	 * 		calc # of tiles
	 * 				gpuCount, card memory, requested tile structure
	 */

/*
	if(m_m == 1 || m_n == 1) {
		if(checkDebug(debugTiler))prlocf("source is vector\n");
		// check mandated tile counts agree with vector dims
		assert( (m_m == 1 && rowTileCount < 2) || m_m > 1 );
		assert( (m_n == 1 && colTileCount < 2) || m_n > 1 );

		// shrink totalSize if counts are specified
		if(m_m == 1) {
			if(checkDebug(debugTiler))prlocf("source is col vector\n");
			colTileCount = DIV_UP(m_n,tileN);
			rowTileCount = 1;
		}else {
			if(checkDebug(debugTiler))prlocf("source is row vector\n");
			tileN = m_n;
			tileM = tileSize/(sizeof(T)*tileN);
			rowTileCount = DIV_UP(m_m,tileM);
			colTileCount = 1;
		}
	} else {
		if(rowTileCount ==  0) {
			if(checkDebug(debugTiler))prlocf("rowTileCount == 0\n");
			// requested tiling by row chunks
			//assert(colTileCount != 0 || (tileM > 0 && tileN > 0));
			if(colTileCount == 1) {
				// tiles span all columns
				tileN = m_n;
				tileM = tileSize/(sizeof(T)*tileN);
				rowTileCount = DIV_UP(m_m,tileM);
				if(checkDebug(debugTiler))flprintf("row tiling %d row TileCount X %d colTileCount\n",rowTileCount,colTileCount);
			} else {
				if(tileM > 0 && tileN > 0) {
					if(checkDebug(debugTiler))flprintf("specd NZ tileM %u & tileN %u, calcing tile counts from these\n",tileM,tileN);
					colTileCount = DIV_UP(m_m,tileM);
					rowTileCount = DIV_UP(m_n,tileN);
				} else if(colTileCount != 0){
				// source matrix tiled by rows and columns
					rowTileCount = DIV_UP(m_size,colTileCount * tileSize);
				}
				if(checkDebug(debugTiler))flprintf("row tiling with specd col tile count %d row tiles X colTileCount\n",rowTileCount,colTileCount);
			}
		} else {
			if(checkDebug(debugTiler))flprintf("rowTileCount %d\n",rowTileCount);
			// have specified >1 row tiles
			if(colTileCount == 0) {
				// requested tiling by col chunks
				colTileCount = DIV_UP(m_size,rowTileCount * tileSize);
				if(checkDebug(debugTiler))flprintf("specd row tiling %d calced col tiles %d\n",rowTileCount,colTileCount);
			} else {
				// both col and row tiles are specified
				ulong specdTileSize = DIV_UP( m_size, colTileCount * rowTileCount );
				if(specdTileSize > tileSize){
					flprintf("specd row+col tiling %dX%d buffer %lu exceeds maxReasonable %lu\n",rowTileCount,colTileCount, specdTileSize, tileSize);
					b_util::dumpStack();
				}
				tileSize = specdTileSize;
			}
		}
	//}
*/

	int orgDevice = ExecCaps::currDev();
	int device = 0;

	for(int i =0; i < gpuCount ; i++ ) {
		device = gpuCount == 1 ? orgDevice : nextGpu();
		if(checkDebug(debugTiler))
			flprintf("allocTiles creating tile %d of %d for gpu %d\n",i+1,gpuCount,device );
#ifndef __CUDA_ARCH__
		if(device != orgDevice) {
			ExecCaps_setDevice(device);
		}
#endif
		T* buff; //
		if(checkDebug(debugMem))
			flprintf("after allocating %ld\n", tileSize);
		cherr(cudaMalloc( &buff, tileSize));
		if(checkDebug(debugCheckValid)) flprintf("%dx%dx%d - %p\n", m_m, m_n, m_p, buff);
		if(checkDebug(debugMem)) {
			flprintf("after allocating %p ( %ld bytes)\n", buff, tileSize);
#ifndef __CUDA_ARCH__
			b_util::usedDmem();
			MemMgr<T>::checkValid(buff);
			MemMgr<T>::checkValid(buff + tileSize/sizeof(T) - 1);
#endif
		}
		if(checkDebug(debugTiler)){
			prlocf("before setBuffer\n");
			dumpBuff(__func__);
		}
		setBuffer(device, buff);
		if(checkDebug(debugTiler)){
			prlocf("after setBuffer\n");
			dumpBuff(__func__);
		}
		if(checkDebug(debugTiler))
			flprintf("allocTiles created %lu-byte buffer %d of %d on device %d ==> %p\n",tileSize, i+1,gpuCount, device, buffer(device) );
	}
#ifndef __CUDA_ARCH__
	if(orgDevice != device) {
		ExecCaps_setDevice(orgDevice);
	}
	if(checkDebug(debugMem))usedDevMem();
#endif
}

template __host__ __device__ void Tiler<float>::allocTiles( uint, uint, float);
template __host__ __device__ void Tiler<double>::allocTiles(uint, uint, float);
template __host__ __device__ void Tiler<int>::allocTiles(  uint, uint, float);
template __host__ __device__ void Tiler<uint>::allocTiles( uint, uint, float);
template __host__ __device__ void Tiler<long>::allocTiles(  uint, uint, float);
template __host__ __device__ void Tiler<ulong>::allocTiles(  uint, uint, float);




template<typename T>  __host__ __device__ void Tiler<T>::tileDims(uint& tileM, uint& tileN, TileDirection tileD) const {
	if(checkDebug(debugTiler)) flprintf("tileDims m_m %u, m_n %u,  m_p %u, tileSize %lu  tileSize/sizeof(T) %lu tileCount %u\n",
			m_m, m_n, m_p, tileSize, tileSize/sizeof(T),getTileCount());

	if(tileSize == m_size) {
		tileN = m_n;
		tileM = m_m;
	} else {
		if(tileD == tdRows) {
			tileN = m_n;
			if(checkDebug(debugTiler))prlocf("tileM bef\n");
			tileM = tileSize/( sizeof(T)* tileN);
			if(checkDebug(debugTiler))prlocf("tileM aft\n");
		} else {
			tileM = m_m;
			if(checkDebug(debugTiler))prlocf("tileN bef\n");
			tileN = tileSize/(sizeof(T)* tileM);
			if(checkDebug(debugTiler))prlocf("tileN aft\n");
		}
		if(checkDebug(debugTiler)) flprintf("tileDims  m_m %u,  m_n %u\n", m_m, m_n);
	}
	if(checkDebug(debugTiler)) flprintf("tileDims tileM %u, tileN %u\n", tileM, tileN);
	if(checkDebug(debugTiler))prlocf("exiting\n");
}
template __host__ __device__ void Tiler<int>::tileDims(uint&, uint&, TileDirection) const;
template __host__ __device__ void Tiler<float>::tileDims(uint&, uint&, TileDirection) const;
template __host__ __device__ void Tiler<double>::tileDims(uint&, uint&, TileDirection) const;
template __host__ __device__ void Tiler<long>::tileDims(uint&, uint&, TileDirection) const;
template __host__ __device__ void Tiler<ulong>::tileDims(uint&, uint&, TileDirection) const;
template __host__ __device__ void Tiler<uint>::tileDims(uint&, uint&, TileDirection) const;

template<typename T> __host__ __device__ void Tiler<T>::tile0(DMatrix<T>& dm, bool copy, cudaStream_t stream) const {
	cherr(cudaPeekAtLastError());
	if(tileSize != m_size) {
#ifndef __CUDA_ARCH__
		outln("!!! tileStze " << tileSize << " != m_size " << m_size);
#endif
		setLastError(badDimensionsEx);
		return;
	}

	if(! buffer(ExecCaps::currDev())) {
		flprintf("currDev %d %dx%dx%d h %p buff (%p,%p,%p,%p)\n", ExecCaps::currDev(), m_m, m_n, m_p, m_elems, buffers.x,buffers.y,buffers.z,buffers.w );
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
			MemMgr<T>::checkValid(dm.elements + dm.m * dm.n - 1);
		}
#endif
		if(checkDebug(debugTiler)) flprintf("dm.mXdm.n %uX%u at %p\n", dm.m, dm.n, dm.elements);

		if(copy &&  m_elems) {
#ifndef __CUDA_ARCH__
			if (checkDebug(debugTiler)) flprintf("tile0 cudaMemcpy dm.elements %p, m_elements %p, m_size %lu \n", dm.elements, m_elems, m_size );
			cherr( cudaMemcpyAsync(dm.elements, m_elems, m_size, cudaMemcpyHostToDevice,stream));
#else
			if (checkDebug(debugTiler)) prlocf("tile0 memcpy tile\n" );
			memcpy(dm.elements, m_elems, m_size);
#endif
	}
}

template __host__ __device__ void Tiler<float>::tile0(DMatrix<float>&, bool, cudaStream_t) const;
template __host__ __device__ void Tiler<double>::tile0(DMatrix<double>&, bool, cudaStream_t) const;
template __host__ __device__ void Tiler<int>::tile0(DMatrix<int>&, bool, cudaStream_t) const;
template __host__ __device__ void Tiler<uint>::tile0(DMatrix<uint>&, bool, cudaStream_t) const;
template __host__ __device__ void Tiler<ulong>::tile0(DMatrix<ulong>&, bool, cudaStream_t) const;
template __host__ __device__ void Tiler<long>::tile0(DMatrix<long>&, bool, cudaStream_t) const;

template<typename T> __host__ __device__ void Tiler<T>::clipTile(
		DMatrix<T>& dm,
		uint tileM, uint tileN,
		int rowTiles, int colTiles,
		int rowTileIdx, int colTileIdx) const {
	if(checkDebug(debugTiler))flprintf("tileM %u tileN %u rowTiles %d colTiles %d rowTileIdx %d colTileIdx %d\n",
			tileM, tileN,rowTiles, colTiles,rowTileIdx, colTileIdx);
	if(rowTiles > 1 && rowTileIdx == rowTiles - 1) {
		if(checkDebug(debugTiler))prlocf("rowTiles > 1 && rowTileIdx == rowTiles - 1\n");
		dm.m =  m_m - rowTileIdx * tileM;  // in above ex., m_m == 5, dm.m == 2, rowTileIdx == 2 --> rowPatchHeight = 1;
	} else {
		if(checkDebug(debugTiler))prlocf("! (rowTiles > 1 && rowTileIdx == rowTiles - 1)\n");
		if(tileM > m_m) {
			if(checkDebug(debugTiler))flprintf("warning, tileM %u is bigger than hostbuff's m_m %u!\n",tileM, m_m);
		}
		dm.m = MIN(tileM,m_m);
	}
	if(colTiles > 1 && colTileIdx == colTiles - 1) {
		if(checkDebug(debugTiler))prlocf("colTiles > 1 && colTileIdx == colTiles - 1\n");
		dm.n = m_n - colTileIdx * tileN;  // in above ex., m_n == 5, dm.n == 2, colTileIdx == 2 --> colPatchWidth = 1
	}else {
		if(checkDebug(debugTiler))prlocf("!(colTiles > 1 && colTileIdx == colTiles - 1)\n");
		dm.n = tileN;
	}
	if(dm.n > tileN) {
		if(checkDebug(debugTiler))flprintf("warning, tileN %u is bigger than hostbuff's m_n %u!\n",tileN, m_n);
	}
	dm.n = MIN(dm.n, m_n);
}

template __host__ __device__  void  Tiler<float>::clipTile(DMatrix<float>&, uint,uint, int,int,int,int) const;
template __host__ __device__  void Tiler<double>::clipTile(DMatrix<double>&, uint,uint, int,int,int,int) const;
template __host__ __device__  void Tiler<int>::clipTile(DMatrix<int>&, uint,uint, int,int,int,int) const;
template __host__ __device__  void Tiler<uint>::clipTile(DMatrix<uint>&, uint,uint, int,int,int,int) const;
template __host__ __device__  void Tiler<ulong>::clipTile(DMatrix<ulong>&, uint,uint, int,int,int,int) const;
template __host__ __device__  void Tiler<long>::clipTile(DMatrix<long>&, uint,uint, int,int,int,int) const;

template <typename T> __host__ __device__ int Tiler<T>::tile1D(DMatrix<T>& dm,  uint& roff, uint& coff,uint& tileM, uint& tileN,
		int t, TileDirection tileD, bool copy, int lastGpu, cudaStream_t stream) const {
	int tileCount = getTileCount();
	if(tileM == 0 || tileN == 0) {
		if(checkDebug(debugTiler)) prlocf("tileM or tileN zero, calling tileDims\n");
		tileDims(tileM, tileN, tileD);
	} else {
		if( tileD == tdRows) tileCount = MAX(tileCount, DIV_UP(m_m, tileM));
		else tileCount = MAX(tileCount, DIV_UP(m_n, tileN));
	}
	if(t < 0 || t >= tileCount ) {
		setLastError(illegalArgumentEx);
		return -1;
	}

	if(checkDebug(debugTiler)) flprintf("tile1D enter roff %u coff %u tileM %u tileN %u tileCount %d\n", roff, coff, tileM, tileN,tileCount);
	roff = (tileD == tdRows) ? t * tileM : 0;
	coff = (tileD == tdCols) ? t * tileN : 0;

	int orgDevice = ExecCaps::currDev();
	if(lastGpu != -1) {
		lastGpu = nextGpu(lastGpu);
	}
	if(checkDebug(debugTiler)){
		flprintf("before buffer(lastGpu %d)\n",lastGpu);
		dumpBuff(__func__);
	}
	dm.elements = buffer(lastGpu);

	if (checkDebug(debugTiler)) flprintf("before clipTile tileM %u tileN %u\n",tileM, tileN);
	clipTile(dm, tileM, tileN, tileD == tdRows ? tileCount : 1,tileD == tdCols ? tileCount : 1, tileD == tdRows ? t : 0, tileD == tdCols ? t : 0 );
	if (checkDebug(debugTiler)) flprintf("after clipTile dm.m %u dm.n %u\n",dm.m, dm.n);
	if( tileCount != 1 )
		dm.p = dm.n;
	else
		dm.p = m_p;
	int gpuCount = countGpus();
	ulong thisTileSize = dm.m * dm.p * sizeof(T);

	if(checkDebug(debugTiler))flprintf("thisTileSize tile %lu\n", thisTileSize );
	if(checkDebug(debugTiler)) flprintf("t %d gpuCnt %d lastGpu %d at roff %u coff%u\n", t, gpuCount, lastGpu, roff, coff);
	ulong loffset = offset(roff, coff);
	if(copy) {
#ifndef __CUDA_ARCH__
		if (checkDebug(debugTiler)) flprintf(
				"cudaMemcpy tile %d m_elems %p dm.elements %p thisTileSize %lu loffset %lu\n",
				t , m_elems, dm.elements, thisTileSize,loffset);
		if(lastGpu > -1 && lastGpu != orgDevice) {
			ExecCaps_setDevice(lastGpu);
		}
		MemMgr<T>::checkValid(dm.elements);
		MemMgr<T>::checkValid(dm.elements + thisTileSize/sizeof(T)-1);
		if(pinnedQ) MemMgr<T>::checkValid(m_elems +loffset );
		if(pinnedQ) MemMgr<T>::checkValid(m_elems + loffset + thisTileSize/sizeof(T)-1);
#endif

		if(tileD == tdRows) {
#ifndef __CUDA_ARCH__
			cherr( cudaMemcpyAsync(dm.elements, m_elems + loffset, thisTileSize, cudaMemcpyHostToDevice,stream));
#else
			if (checkDebug(debugTiler)) flprintf("tile1D memcpy tile %d\n", t );
			memcpy(dm.elements, m_elems + offset(roff, coff), thisTileSize);
#endif
		} else {
			cherr( cudaMemcpy2DAsync(m_elems +offset(roff, coff), m_p, dm.elements, dm.p, dm.n, dm.m, cudaMemcpyDeviceToHost,stream));

		}
#ifndef __CUDA_ARCH__
		if(lastGpu > -1 && lastGpu != orgDevice) {
			ExecCaps_setDevice(orgDevice);
		}
#endif
	}
	return lastGpu;
}

template __host__ __device__  int  Tiler<float>::tile1D(DMatrix<float>&,uint&, uint&, uint&, uint&,int,TileDirection, bool,int,cudaStream_t) const;
template __host__ __device__  int  Tiler<double>::tile1D(DMatrix<double>&, uint&, uint&, uint&, uint&,int,TileDirection, bool,int,cudaStream_t) const;
template __host__ __device__  int  Tiler<int>::tile1D(DMatrix<int>&, uint&, uint&, uint&, uint&,int,TileDirection, bool,int,cudaStream_t) const;
template __host__ __device__  int  Tiler<uint>::tile1D(DMatrix<uint>&, uint&, uint&, uint&, uint&,int,TileDirection, bool,int,cudaStream_t) const;
template __host__ __device__  int  Tiler<ulong>::tile1D(DMatrix<ulong>&, uint&, uint&, uint&, uint&,int,TileDirection, bool,int,cudaStream_t) const;
template __host__ __device__  int  Tiler<long>::tile1D(DMatrix<long>&, uint&, uint&, uint&, uint&,int,TileDirection, bool,int,cudaStream_t) const;

template<typename T> __host__ __device__ int Tiler<T>::tile2D(
		DMatrix<T>& dm, uint& roff, uint& coff,
		uint& tileM, uint& tileN,
		int rowTileCount, int colTileCount,
		int rowTileIdx, int colTileIdx,
		bool copy, int lastGpu, cudaStream_t stream ) const {
	if(checkDebug(debugTiler))flprintf("dm.elements %p, roff %u  coff %u tileM %u tileN %u rowTileIdx %d colTileIdx %d\n",
														dm.elements, roff, coff, tileM,  tileN, rowTileIdx, colTileIdx);
	if(checkDebug(debugTiler))flprintf("rowTileCount %u colTileCount %u\n", rowTileCount, colTileCount);
	if(rowTileIdx < 0 || rowTileIdx >= rowTileCount || colTileIdx < 0 || colTileIdx >= colTileCount) {
		setLastError(illegalArgumentEx);
		return -1;
	}

	if(tileM == 0 || tileN == 0)
		tileDims(tileM, tileN, tdBoth);

	roff = (rowTileCount > 1) ? rowTileIdx * tileM : 0;
	coff = (colTileCount > 1) ? colTileIdx * tileN : 0;

	int orgDevice = ExecCaps::currDev();
	int gpuCount = countGpus();
	if(gpuCount > 1 && lastGpu != -1) {
		lastGpu = nextGpu(lastGpu);
	}
	dm.elements = buffer(lastGpu);
	ulong thisTileSize = dm.m * dm.p * sizeof(T);
	if(checkDebug(debugTiler)) flprintf("tile2D bef (%d,%d) gpuCnt %d gpuCurr %d timeMN %uX%u at rcoff %uX%u\n", rowTileIdx, colTileIdx, gpuCount, lastGpu, tileM, tileN, roff, coff);
	if(checkDebug(debugTiler)) flprintf("tile2D bef dm.mn  %uX%u\n", dm.m,dm.n );
	if(checkDebug(debugTiler)) flprintf("tile2D bef thisTileSize tile %lu\n", thisTileSize );
    clipTile(dm, tileM, tileN, rowTileCount,colTileCount,rowTileIdx, colTileIdx);
	dm.p = dm.n;
	thisTileSize = dm.m * dm.p * sizeof(T);

	if(checkDebug(debugTiler)) flprintf("tile2D aft thisTileSize tile %lu\n", thisTileSize );
	if(checkDebug(debugTiler)) flprintf("tile2D aft dm.mn  %uX%u\n", dm.m,dm.n );
	if(checkDebug(debugTiler)) flprintf("tile2D aft (%d,%d) gpuCnt %d gpuCurr %d timeMN %uX%u at rcoff %uX%u\n", rowTileIdx, colTileIdx, gpuCount, lastGpu, tileM, tileN, roff, coff);

	if(copy) {
#ifndef __CUDA_ARCH__
		if (checkDebug(debugTiler)) flprintf("tile2D cudaMemcpy tile (%d,%d) m_elems %p dm.elements %p thisTileSize %lu offset(roff, coff) %d\n", rowTileIdx, colTileIdx, m_elems, dm.elements, thisTileSize,offset(roff, coff));
		MemMgr<T>::checkValid(dm.elements);
		MemMgr<T>::checkValid(dm.elements + thisTileSize/sizeof(T)-1);
		if(pinnedQ) MemMgr<T>::checkValid(m_elems);
		if(pinnedQ) MemMgr<T>::checkValid(m_elems + thisTileSize/sizeof(T)-1);
		if(lastGpu > -1 && lastGpu != orgDevice) {
			ExecCaps_setDevice(lastGpu);
		}
		cherr( cudaMemcpyAsync(dm.elements, m_elems +offset(roff, coff), thisTileSize, cudaMemcpyHostToDevice,stream));
#else
		if (checkDebug(debugTiler)) flprintf("tile2D memcpy rowTileIdx %d colTileIdx %d\n", rowTileIdx,colTileIdx );
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
template __host__ __device__ int Tiler<float>::tile2D(DMatrix<float>&, uint&, uint&, uint&, uint&, int, int, int, int, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<double>::tile2D(DMatrix<double>&, uint&, uint&, uint&, uint&, int, int, int, int, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<int>::tile2D(DMatrix<int>&, uint&, uint&, uint&, uint&, int, int, int, int, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<uint>::tile2D(DMatrix<uint>&, uint&, uint&, uint&, uint&, int, int, int, int, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<ulong>::tile2D(DMatrix<ulong>&, uint&, uint&, uint&, uint&, int, int, int, int, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<long>::tile2D(DMatrix<long>&, uint&, uint&, uint&, uint&, int, int, int, int, bool, int, CUstream_st*) const;

template<typename T> __host__ __device__ int Tiler<T>::tileLike(DMatrix<T>& dm, uint& roff, uint& coff, uint tileM, uint tileN,
		int t, TileDirection tileD, bool copy, int lastGpu, cudaStream_t stream) const {
	//int nTiles = getTileCount();
	if(checkDebug(debugTiler)) prlocf("tileLike entre\n");

	if(t < 0 || t >= MAX(DIV_UP(m_m,tileM),  DIV_UP(m_n,tileN)) ) {
		flprintf("t %d MAX(DIV_UP(m_m,tileM), DIV_UP(m_n,tileN))  %d\n", t,  MAX(DIV_UP(m_m,tileM),  DIV_UP(m_n,tileN)));
		flprintf("m_m %u tileM %u, m_n %u  tileN %u\n", m_m, tileM, m_n, tileN);
		setLastError(illegalArgumentEx);
		return -1;
	}
	int orgDevice = ExecCaps::currDev();
	int gpuCount = countGpus();
	if(gpuCount > 1 && lastGpu != -1) {
		lastGpu = nextGpu(lastGpu);
	} else if(lastGpu == -1) {
		lastGpu = nextGpu();
	}
	dm.elements = buffer(lastGpu);
	if(checkDebug(debugMem))dumpBuff(__func__);
	if (checkDebug(debugTiler))flprintf("m_m %u m_n %u m_p %u lastGpu %u m_elems %p dm.elements %p\n",m_m, m_n, m_p,lastGpu, m_elems, dm.elements);

	if(checkDebug(debugTiler))flprintf("tileLike bef dm.m %u dm.n %u clip  DIV_UP(m_m,tileM) %d DIV_UP(m_n,tileN) %d tileM %u tileN %u\n",dm.m, dm.n, DIV_UP(m_m,tileM),DIV_UP(m_n,tileN),tileM,tileN);
	clipTile(dm, tileM, tileN, DIV_UP(m_m,tileM), DIV_UP(m_n,tileN), tileD == tdRows ? t : 0, tileD == tdCols ? t : 0);
	if(checkDebug(debugTiler))flprintf("tileLike aft  dm.m %u dm.n %u clip tileM %u tileN %u\n",dm.m, dm.n, tileM,tileN);
	dm.p = dm.n;
	ulong thisTileSize = dm.m * dm.p * sizeof(T);

	roff = (tileD == tdRows) ? t * tileM : 0;
	coff = (tileD == tdCols) ? t * tileN : 0;
	ulong loffset = offset(roff, coff);
	if (checkDebug(debugTiler))flprintf(
			"tileLike thisTileSize dm %uX%uX%u, roff %u coff %u thisTileSize %lu loffset %lu stream %p\n",
			dm.m, dm.n, dm.p, roff, coff, thisTileSize, loffset, stream );
	if(copy) {
		if(lastGpu > -1 && lastGpu != orgDevice) {
			ExecCaps_setDevice(lastGpu);
		}
#ifndef __CUDA_ARCH__

		if (checkDebug(debugCopy))flprintf("tileLike chekin valid, m_elems + m_size(==%d)/sizeof(T) - 1\n",m_size);
		MemMgr<T>::checkValid(m_elems + (m_size/sizeof(T) -1));

		if (checkDebug(debugCopy))
			flprintf("tileLike chekin valid, dm.elements %p\n", dm.elements);

		MemMgr<T>::checkValid(dm.elements);
		if (checkDebug(debugMeans)){

			pthread_t currThread = pthread_self ();
			outln("currThread " << currThread);
			flprintf("tileLike chekin valid, dm.elements (== %p) + thisTileSize/sizeof(T) -1\n",dm.elements);
			MemMgr<T>::checkRange(dm.elements, (thisTileSize/sizeof(T) -1));
		}
		if (checkDebug(debugCopy))flprintf("tileLike chekd dm.elements + thisTileSize/sizeof(T) %p\n",dm.elements + thisTileSize/sizeof(T));
		MemMgr<T>::checkValid(m_elems + loffset);
		if (checkDebug(debugCopy))flprintf("tileLike chekd m_elems + loffset %p\n",m_elems + loffset);
		MemMgr<T>::checkValid(m_elems + loffset + thisTileSize/sizeof(T) -1);
		if (checkDebug(debugCopy))flprintf("tileLike chekd m_elems + loffset + thisTileSize/sizeof(T) %p\n",m_elems + loffset + thisTileSize/sizeof(T));
		cherr(cudaPeekAtLastError());
		//cherr( cudaMemcpyAsync(dm.elements, m_elems +loffset, thisTileSize, cudaMemcpyHostToDevice,stream));

		//cherr( cudaMemcpy(dm.elements, m_elems +loffset, thisTileSize, cudaMemcpyHostToDevice));

		if(stream) {
			if (checkDebug(debugTiler))flprintf("cudaMemcpy2DAsync copying from %p (loffset %u = %p) to %p with stream %p\n", m_elems, loffset, m_elems + loffset, dm.elements, stream );
			cherr( cudaMemcpy2DAsync(dm.elements, dm.p * sizeof(T), m_elems +loffset, m_p* sizeof(T), dm.n* sizeof(T), dm.m, cudaMemcpyHostToDevice,stream));
		}else {
			if (checkDebug(debugTiler)) {
				outln(*this);
				flprintf(
						"cudaMemcpy2D copying from %p (loffset %u = %p) to %p with stream %p\n",
						m_elems, loffset, m_elems + loffset, dm.elements,
						stream);

				flprintf("*dst %p, size_t dpitch %u, const void *src %p, size_t spitch %d, size_t width %d\n",
						dm.elements, dm.p * sizeof(T), m_elems +loffset, m_p* sizeof(T), dm.n* sizeof(T), dm.m);
			}
			//(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);

			cherr( cudaMemcpy2D(dm.elements, dm.p * sizeof(T), m_elems +loffset, m_p* sizeof(T), dm.n* sizeof(T), dm.m, cudaMemcpyHostToDevice));
		}
#else
		if (checkDebug(debugCopy))prlocf("tileLike memcpy");
		memcpy(dm.elements, m_elems + loffset, thisTileSize);
#endif
		if(lastGpu > -1 && orgDevice != lastGpu ) {
			ExecCaps_setDevice(orgDevice);
		}
	}
	return lastGpu;
}

template __host__ __device__ int Tiler<float>::tileLike(DMatrix<float>&, uint&, uint&, uint, uint, int, TileDirection, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<double>::tileLike(DMatrix<double>&, uint&, uint&, uint, uint, int, TileDirection, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<int>::tileLike(DMatrix<int>&, uint&, uint&, uint, uint, int, TileDirection, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<uint>::tileLike(DMatrix<uint>&, uint&, uint&, uint, uint, int, TileDirection, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<ulong>::tileLike(DMatrix<ulong>&, uint&, uint&, uint, uint, int, TileDirection, bool, int, CUstream_st*) const;
template __host__ __device__ int Tiler<long>::tileLike(DMatrix<long>&, uint&, uint&, uint, uint, int, TileDirection, bool, int, CUstream_st*) const;


/*
template <typename T> __host__ __device__ void Tiler<T>::syncTile(
		const DMatrix<T>&dm, uint roff, uint coff, cudaStream_t stream) {
	if(checkDebug(debugTiler)) {
		flprintf("syncTile dm %uX%uX%u dm.elements %p roff %d coff %d\n", dm.m, dm.n, dm.p, dm.elements, roff, coff);
	}
#ifndef __CUDA_ARCH__
	if (checkDebug(debugTiler))flprintf("dpitchB(m_p) %u spitchB(dm.p) %u widthB(dm.n) %u height(dm.m) %u\n",  m_p* sizeof(T), dm.p* sizeof(T), dm.n* sizeof(T), dm.m );
	if(stream) {
		if (checkDebug(debugTiler))flprintf("cudaMemcpy2DAsync roff %u coff %u copying from %p to %p with stream %p\n", roff, coff, dm.elements, m_elems +offset(roff, coff), stream );
		cherr( cudaMemcpy2DAsync(m_elems +offset(roff, coff), m_p * sizeof(T), dm.elements, dm.p* sizeof(T), dm.n* sizeof(T), dm.m, cudaMemcpyDeviceToHost,stream));
	}else {
		if (checkDebug(debugTiler))flprintf("cudaMemcpy2D roff %u coff %u copying from %p to %p with stream %p\n", roff, coff, dm.elements, m_elems +offset(roff, coff), stream );
		if (checkDebug(debugTiler))flprintf(
				"cudaMemcpy2D m_elems +offset(roff, coff) %p, m_p* sizeof(T) %u, dm.elements %p, dm.p* sizeof(T) %u, dm.n* sizeof(T) %u, dm.m %u\n",
				m_elems +offset(roff, coff), m_p* sizeof(T), dm.elements, dm.p* sizeof(T), dm.n* sizeof(T), dm.m);
		if (checkDebug(debugTiler)) {
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
	}
#else
	if (checkDebug(debugTiler))printf("syncTile memcpy, synching stream %p\n",stream);
	cherr(cudaStreamSynchronize(stream));
	memcpy2D(m_elems +offset(roff, coff), m_p, dm.elements, dm.p, dm.n, dm.m);
#endif
}


template __host__ __device__ void Tiler<float>::syncTile(const DMatrix<float>&, uint, uint, cudaStream_t) ;
template __host__ __device__ void Tiler<double>::syncTile(const DMatrix<double>&, uint, uint, cudaStream_t) ;
template __host__ __device__ void Tiler<int>::syncTile(const DMatrix<int>&, uint, uint, cudaStream_t) ;
template __host__ __device__ void Tiler<uint>::syncTile(const DMatrix<uint>&, uint, uint, cudaStream_t) ;
template __host__ __device__ void Tiler<ulong>::syncTile(const DMatrix<ulong>&, uint, uint, cudaStream_t) ;
template __host__ __device__ void Tiler<long>::syncTile(const DMatrix<long>&, uint, uint, cudaStream_t) ;
*/
