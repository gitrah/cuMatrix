/*
 * Tiler.h
 *
 *  During matrix construction, don't need to specify tile direction
 *  cons: specify device count and create dbuffs
 *  tile: specify direction
 *
 */
#pragma once

#include <string>

#include "caps.h"
#include "MemMgr.h"
#include "Kernels.h"

const int GpuMaskAll = -1;
#define GpuMask(n) (1 << n)

template<typename T> class CuMatrix;

#define MAX_BUFFERS 4

/*
 * Tiles host matrices
 * 		A.  when the matrices can't fit (or some number matrices collaborating in a pipeline or of them can't all fit) into device memory
 * 		B.  so multiple GPUs can operate on parts of them independently & simultaneously
 *
 * Tiler maps host mem into device ram tiles that partition the source matrix
 * by width, height, or both, as required by the math
 *
 * todo: allow varying GPU tile sizes instead of minimum
 *
 * todo unempl
 * Tile size set to the smaller (host/device(s) buffer
 *     tiling for
 *         buffer limits
 *         multi-gpu data ||ism
 *         IO:  tile file into host (..into device)
 *    multiple tiles (one can be loading while another participates in a calcul
 *
 * todo multi gpu tests
 */
inline __host__ __device__ bool equiBuff(ulong4 a, ulong4 b) {
    return a.x == b.x && a.y ==b.y && a.z  == b.z &&  a.w == b.w;
}
template<typename T> struct Tiler {

    friend class CuMatrix<T>;

    int m_m, m_n, m_p;
    long m_size; // in BYTES
    T* m_elems;
    long tileSize;
    int gpuMask;
    int gpuModulus;

    ulong4 buffers;
    TileDirection tileD;

    void (*kernelle)();

    __host__ __device__ Tiler( const CuMatrix<T>& m,  void (*kernelle)() = nullptr);
    __host__ __device__ Tiler(
			CuMatrix<T>& m, bool allocate,
			int gpuMask = gpuMaskFromGpu(ExecCaps::currDev()),
			TileDirection tileD = tdCols, void (*kernelle)() = nullptr);
    __host__ __device__ Tiler(const Tiler<T>& o);

    __host__ __device__ bool validQ() const;

    __host__ __device__ void setKernelle(  void (*kernelle)()) { this->kernelle =  kernelle; }

    __host__ __device__ inline T* buff() const {
    	return buffer(nextGpu(ExecCaps::currDev()));
    }

    __host__ __device__ inline T* buffer(int i) const {
        switch (i) {
        case 0:
            return (T*) buffers.x;
        case 1:
            return (T*) buffers.y;
        case 2:
            return (T*) buffers.z;
        case 3:
            return (T*) buffers.w;
        default:
            flprintf("called with %d\n",i);
            setLastError(illegalArgumentEx);
            return 0;
        }
    }

   // __host__ __device__ inline T* buff() const;

    __host__ __device__ inline void set(ulong4 b) {
    	buffers = b;
    }

    __host__ __device__ inline bool onDeviceQ(int dev) {
    	return buffer(dev) != 0;
    }

    __host__ __device__ inline bool onAnyDeviceQ() {
    	return b_util::anyDevice([this]() { return currBuffer() != nullptr;});
    }

    __host__ __device__ int deviceOfResidence() const;

    __host__  __device__ int indexInMask(int gpu) const;

    __host__ __device__ inline void dumpBuff(const char* msg) const {
        printf("%s: {%p,%p,%p,%p}\n",msg, (T*)buffers.x,(T*)buffers.y,(T*)buffers.z,(T*)buffers.w);
    }

    __host__ __device__ inline T* currBuffer() const {
        return countGpus() > 1 ? buffer(ExecCaps::currDev()) :  buff() ;
    }

    __host__ __device__ inline void setCurrBuffer(T* buff) {
        setBuffer(ExecCaps::currDev(), buff);
    }

    __host__ __device__ inline void setBuffer(int i, T* elems) {
    	if(checkDebug(debugMem))
    		flprintf("i %d elems %p\n", i, elems);
        switch (i) {
        case 0:
        //    assert(buffers.x == 0);
            buffers.x = (ulong) elems;
            break;
        case 1:
        //    assert(buffers.y == 0);
            buffers.y = (ulong) elems;
            break;
        case 2:
        //    assert(buffers.z == 0);
            buffers.z = (ulong) elems;
            break;
        case 3:
            buffers.w = (ulong) elems;
            break;
        default:
            setLastError(illegalArgumentEx);
        }
    }

    __host__ __device__ void clear() {
        gpuMask =gpuMaskFromCurrGpu();
        buffers = {0,0,0,0};
        tileSize=0;
    }

    static __host__ __device__ void dumpMask(int mask) {
         char buff[33];
         buff[32]=0;
         b_util::printBinInt(buff, mask);
         flprintf("gpuMask [%s]\n",buff);
     }

    static __host__ string maskStr(int mask) {
         char buff[33];
         buff[32]=0;
         b_util::printBinInt(buff, mask);
         flprintf("gpuMask [%s]\n",buff);
         return string(buff);
     }

    static __host__ __device__ const char * maskStr(char *buff, int mask) {
         b_util::printBinInt(buff, mask);
         return buff;
    }

    static __host__ __device__ int gpuMaskFromCount(int count) {
        int mask = 0;
        for(int gpu =0; gpu < count;gpu++){
            mask |= 1 <<  gpu;
        }
        char buff[33];
        buff[32]=0;
        b_util::printBinInt(buff, mask);
        if(checkDebug(debugTiler))flprintf("gpuMaskFromCount %d mask %d bin %s\n",count,mask,buff);
        return mask;
    }

    __host__ __device__ void setGpuMask( int gpuMask) {
    	this->gpuMask = gpuMask;
    	gpuModulus = modulusFromMask();
    }
    static __host__ __device__ int gpuMaskFromGpu(int gpu) {
    	int maks = 1 <<  gpu;
        if (checkDebug(debugTiler)) {
            flprintf("gpu %d mask %d\n", gpu, maks);
            dumpMask(maks);
        }
        return maks;
    }
    static __host__ __device__ int gpuMaskFromCurrGpu() { return gpuMaskFromGpu(ExecCaps::currDev());}

    __host__ __device__ int countGpus() const {
        int count = 0;
        for(int idx =0; idx < (int) sizeof(int) * 8;idx++){
            if(gpuMask >>  idx & 1)
                count++;
        }
        return count;
    }
    __host__ __device__ int modulusFromMask() const {
        int mod = 0;
        for(int idx =0; idx < (int)sizeof(int) * 8;idx++){
            if(gpuMask >>  idx & 1)
            	mod = idx+1;
        }
        return mod;
    }

    __host__ __device__ int nextGpu(int lastGpu = -1) const;
    __host__ __device__ long offset(int roff, int coff, bool colMajor = false) const {
            return colMajor ? ( (long)coff * m_m + roff ) : ((long)roff * m_p + coff);
    }
    __host__ __device__ long offsetd(int roff, int coff, int tileP,  bool colMajor = false) const {
            return colMajor ? ((long)coff * m_m + roff ) : ((long)roff * tileP + coff);
    }


    __host__ __device__ bool hasDmemQ() const;
    inline __host__ __device__ int getTileCount() const {
    	assert(tileSize >0);
    	return m_size == 0? 0 : DIV_UP(m_size,tileSize);
    }

    __device__ void free() {
        // buffer mgt handled by owning CuMatrix
        T* buff;
        for(int i = 0; i < MAX_BUFFERS; i++) {
            buff = buffer(i);
            if(buff) {
                if(checkDebug(debugMem)) {
                    flprintf("freeing buffer %d at %p\n", i, buff);
                    cherr(cudaFree(buff));
                }
            }
        }
        buffers = {0,0,0,0};
    }

    __host__ __device__ void operator=(const Tiler<T>& o) {
        m_m = o.m_m;
        m_n = o.m_n;
        m_p = o.m_p;
        m_size= o.m_size; // in BYTES
        m_elems= o.m_elems;
        gpuMask = o.gpuMask;
        gpuModulus = o.gpuModulus;
        if(gpuMask != gpuMaskFromCurrGpu()) {
            flprintf("gpuMask %d unexpected %d\n", gpuMask, gpuMaskFromCurrGpu());
        }
        buffers = o.buffers;
        tileSize = o.tileSize;
        kernelle = o.kernelle;
        if(checkDebug(debugTiler)) flprintf("tiler %p->tileSize set to %d\n", this, tileSize);
    }

    __host__ __device__ void reset( CuMatrix<T>& m) {
#ifndef __CUDA_ARCH__
    	if(buffers.x)
    		m.getMgr().freeDbuff((T*) buffers.x, tileSize);
    	if(buffers.y)
    		m.getMgr().freeDbuff((T*) buffers.y, tileSize);
    	if(buffers.z)
    		m.getMgr().freeDbuff((T*) buffers.z, tileSize);
    	if(buffers.w)
    		m.getMgr().freeDbuff((T*) buffers.w, tileSize);
    	buffers.x = buffers.y = buffers.z = buffers.w = 0;
#else
    	free();
#endif
        m_m = m.m;
        m_n = m.n;
        m_p = m.p;
        m_size= m.size; // in BYTES
        m_elems= m.elements;

        allocTiles(m._tileM, m._tileN, m._tileP);
        m.getMgr().addTiles(this);
    }
    /*
     * source matrix maps to one tile (per gpu if so masked)
     */

    __host__ __device__ int nxtGpu(int lastGpu ) const  {
        if (checkDebug(debugTiler))
            dumpMask(gpuMask);

        for(int idx = lastGpu; idx < sizeof(int) * 8;idx++){
            if (checkDebug(debugTiler))flprintf("found bit at idx %d \n", idx);
            if(gpuMask >>  idx & 1) {
                if (checkDebug(debugTiler))flprintf("found bit at idx %d (started at %d)\n", idx, lastGpu);
                return idx;
            }
        }
        return -1;
    }

private:
    __host__ __device__ void allocSingleTile(bool padded);

public:
    /*
     *
     * allocates one tile per gpu (as specd in gpuMask)
     *
     *
     * in  Mc = Ma <<op>> Mb
     *         Ma, Mb are source matrices
     *         Mc is result matrix
     *         Ma and Mb will be both row- or colum-tiled unless mat product, when one will be row and the other col
     *         Mc will be row or col (if Ma and Mb are both row or both col) or Mc will both if mat prod (also tx?)
     *
     * tileCounts (.x is row, .y = col)
     *         A) if  unspec, assumes single tile
     *         C) if .x ==0 or .y == 0, assume tiling in == 0 direction.
     *         D) if .x > 1 &| .y > 1, use mandated tiling in one or both directions
     *
     */

    __host__ __device__ void allocTiles(  int& tileM, int& tileN, int& tileP, bool padded = false, float headroom = DEFAULT_GMEM_HEADROOM_FACTOR);

    __host__ __device__ void tileDims(int& tileM, int& tileN, int& tileP, TileDirection tileD = tdRows) const;

    static inline __device__ void memcpy2D(T *dst, size_t dpitchTs, const T *src, size_t spitchTs, size_t width, size_t height) {
        for(int i =0; i < height; i++) {
            memcpy(dst + i * dpitchTs, src + i * spitchTs, width);
        }
    }

    /*
     *
     * cases:
     *     tdRows only (simplest) : zero end of buffer
     *     tdCols only : zero rt side buffer
     *         calc area of
     *
     *         say rowTiles == colTiles == 3
     *
     *              |       |        |
     *         11 12 13 14 15 . 16
     *      __    21 22 23 24 25 . 26
     *           31 32 33 34 35 . 36
     *      __    41 42 43 44 45 . 46
     *         51 52 53 54 55 . 56
     *          .  .  .  .  .
     *      __    61 62 63 64 65   66
     *
     *      61-66 are rowPatch
     *      16-66 are colPatch
     *
     *      returns the
     */

    __host__ __device__ void clipTile(
            DMatrix<T>& dm,
            int tileM, int tileN,
            int rowTiles, int colTiles,
            int rowTileIdx, int colTileIdx) const;

protected:

    // todo:  remove til0 in favor of single-tile executions of tile1D or tile2D
    // @deprecated
    __host__ __device__ void tile0(DMatrix<T>& dm, bool copy = true, cudaStream_t stream = 0) const;
    /*
     * streams are per gpu
     * if a stream is specified and lastGpu/gpuMask cause a device change for this tile, kaboom
     */
    __host__ __device__ int tile1D(DMatrix<T>& dm,  int& roff, int& coff,int& tileM, int& tileN, int& tileP,
            int t, TileDirection tileD = tdRows, bool copy = true, int lastGpu = -1, cudaStream_t stream = 0) const;

    /*
     * calculate properties of next tile (indexed by rowTileIdx and or colTileIdx)
     *      offset in  source (roff, coff)
     *      width of tile (dm.m, dm.n; same for all but, possibly, edge tiles)
     */
    __host__ __device__ int tile2D(DMatrix<T>& dm,
            int& roff, int& coff,
            int& tileM, int& tileN, int& tileP,
            int rowTileIdx, int colTileIdx,
            int rowTileCount, int colTileCount,
            bool copy = true, int lastGpu =-1, cudaStream_t stream = 0) const;
    /*
     *
     * tileLike used by se
     * todo ensure compatible gpuMasks so 'collaborating' tiles Ta,Tb, Tc  (eg in Mc = Ma <<op>> Mb) are on same GPU...
     *
     */
public:
    __host__ __device__ int tileLike(DMatrix<T>& dm, int& roff, int& coff, int tileM, int tileN, int tileP,
            int t, TileDirection tileD = tdRows, bool copy = true, int lastGpu = -1, cudaStream_t* streams = nullptr) const ;

    __host__ __device__ void syncTile(const DMatrix<T>&dm, int roff = 0, int coff=0, cudaStream_t stream = nullptr);

    string toString() const {
        stringstream ss;
        const int gpus = countGpus();
        if(checkDebug(debugTiler))flprintf("gpus %d\n",gpus);
        ss << "[tiler " << this << " h_elems "<< m_elems << ", sz/msz " << tileSize  <<
        		"/" << m_size << ", " << m_m << " X " << m_n << " X " << m_p << " {";
        for(int i =0; i < gpus; i++) {
            ss << "gpu" <<  i <<": " << buffer(i);
            if(i < gpus - 1) {
                ss << ",";
            }
        }
        char buff[33];
        buff[32]=0;
        b_util::printBinInt(buff, gpuMask);
        ss << ", mask " << buff;
        ss << " tdir " << b_util::tileDir(tileD);
        ss << "}] ";
        return ss.str();
    }

    inline friend ostream& operator<<(ostream& os, const Tiler<T>& t)  {
            return os << t.toString();
    }

    __host__ __device__ bool operator==(const Tiler<T>& o) const {
        return gpuMask == o.gpuMask && m_elems == o.m_elems && equiBuff(buffers,o.buffers);
    }
    __host__ __device__ bool operator!=(const Tiler<T>& o) const {
        return !(*this==o);
    }
};
