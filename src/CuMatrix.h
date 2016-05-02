/*
 * CuMatrix.h
 *
 *  Created on: Dec 3, 2012
 *      Author: reid
 */

#pragma once

#include "util.h"
#include "CuFunctor.h"
#include "UnaryOpF_Gen.h"
#include "UnaryOpIndexF_Gen.h"
#include "BinaryOpF_Gen.h"
#include <curand_kernel.h>

#include "MemMgr.h"
#include "Migrator.h"
#include "CuDefs.h"
#include "DMatrix.h"
#include "Tiler.h"
#include "MatrixExceptions.h"

#include <cublas_v2.h>
#include <set>

template<typename T> class MemMgr;
/*
 * m-m-m-my m-matrix
 *
 * TODO
 * 	proper pitch/cuda-style array support
 *
 */

extern __constant__ uint D_MaxRowsDisplayed;
extern __constant__ uint D_MaxColsDisplayed;

extern uint CuDestructs;
extern __device__ uint CuDestructsD;


#ifdef CuMatrix_UseCublas
extern   bool g_useCublas;
extern   cublasHandle_t g_handle;
#endif

template<typename T> class CuMatrix {
//	friend class Tiler<T>;

public:
	static CuMatrix<T> ZeroMatrix;
	static MemMgr<T>* mgr;
	static BuildType buildType;
protected:
	bool ownsTxp;
	__host__ __device__ void freeTxp();
	CuMatrix<T>* txp;  // c++ limitation is that this won't compile as a value, only a pointer, because size of Matrix<T> is 'unknown' at this point
	__host__ __device__ void initMembers();
public:
	Modification lastMod;
	T* elements;

	int m, n, p;
	int oldM, oldN;
	long size; // in BYTES
	bool posed, colMajor, ownsBuffers;

	Tiler<T> tiler; // todo d and h tilers

	/*
	 * in the extra resident case, a suitable row or column tile is calculated
	 * tiler allocs one or more (if >1 gpus) tile-sized that will be shared by all the tiles
	 * operations are then decomposed into bunches that operate sequentially over (pairs of) tiles
	 */
	//TileDirection hTileD;
	//int hTileCount;

	__host__ MemMgr<T>& getMgr() const;

public:
	__host__ __device__ CuMatrix();
	__host__ __device__ CuMatrix(const CuMatrix<T>& o);
/*	__host__ __device__ CuMatrix(const CuMat<T>& o);*/
	__host__ __device__ CuMatrix( T* h_data, int m, int n, bool allocateD);
	__host__ __device__ CuMatrix( T* h_data, int m, int n, int p, bool allocateD);
	__host__ __device__ CuMatrix(int m, int n, bool allocateH, bool allocateD );
	__host__ __device__ CuMatrix(int m, int n, int p, bool allocateH, bool allocateD );
	__host__ __device__ CuMatrix(int m, int n, int p, int gpuMask, bool allocateH, bool allocateD );
	__host__ __device__ CuMatrix(int m, int n, int p, int tileM, int tileN, int gpuMask, bool allocateH, bool allocateD );
	virtual __host__ __device__ ~CuMatrix();

	__host__ __device__ CuMatrix copy(bool copyDeviceMem = true) const;
	static __host__ __device__ bool integralTypeQ();

//mem
	inline __host__ __device__ void tile0(DMatrix<T>& dm, cudaStream_t stream = 0) const {
		tile0(dm, lastMod == mod_host, stream);
	}
	inline __host__ __device__ void tile0(DMatrix<T>& dm, bool copy, cudaStream_t stream = 0) const {
		if(copy && !(lastMod == mod_synced || lastMod == mod_host)) {
			flprintf("bad mod %d\n", lastMod);
		}
		assert(!copy || ( lastMod == mod_synced || lastMod == mod_host));
		tiler.tile0(dm,copy,stream);
	}
	/*
	 * streams are per gpu
	 * if a stream is specified and lastGpu/gpuMask cause a device change for this tile, kaboom
	 */
	__host__ __device__ int tile1D(DMatrix<T>& dm,  int& roff, int& coff,int& tileM, int& tileN,
			int t, TileDirection tileD = tdRows, bool copy = true, int lastGpu = -1, cudaStream_t stream = 0) const;

	/*
	 * calculate properties of next tile (indexed by rowTileIdx and or colTileIdx)
	 *	  offset in source (roff, coff)
	 *	  width of tile (dm.m, dm.n; same for all but, possibly, edge tiles)
	 */
	__host__ __device__ int tile2D(DMatrix<T>& dm,
			int& roff, int& coff,
			int& tileM, int& tileN,
			int rowTileIdx, int colTileIdx,
			int rowTileCount, int colTileCount,
			bool copy = true, int lastGpu =-1, cudaStream_t stream = 0) const;

	__host__ __device__ int releaseBuffers();
	__host__ __device__ CuMatrix<T>& syncBuffers(bool copy = true);
	//__host__ __device__ void calcTiles(float headroom = DEFAULT_GMEM_HEADROOM_FACTOR);
	__host__ __device__ void invalidateHost();
	__host__ __device__ void invalidateDevice();
	__host__ __device__ T* currBuffer() const { return tiler.currBuffer(); }
	__host__ __device__ CuMatrix<T> syncHost();
	__host__ __device__ CuMatrix<T> syncDevice();
	inline __host__ __device__ ulong offset(int roff, int coff) const {return colMajor ? (coff * m + roff ) : (roff * p + coff);}

	// toRtiles, toLtiles
	__host__ __device__  DMatrix<T> asDmatrix( bool copy = true) const {
		DMatrix<T> dt;
		tile0(dt,lastMod == mod_host && copy);
		return dt;
	}
	__host__ __device__  void updateSize() {
		size = p * m * sizeof(T);
	}
	inline void allocHost() { getMgr().allocHost(*this); }
	__host__ CUDART_DEVICE CuMatrix<T> columnSubset( const int* indices, int count) const;
	__host__ CuMatrix<T> clippedRowSubset( const int *r, int count, intPair colRange) const;
	CuMatrix<T> addBiasColumn() const;
	void copy(CuMatrix<T>& res, int roff = 0, int coff = 0, bool onlyDevice = false) const;
	CuMatrix<T> replicateTiled(int mcopies, int ncopies) const;
	static void concat(CuMatrix<T>& canvas, int components, const CuMatrix<T>** parts, bool colMajor = false);
	static void concat(CuMatrix<T>& canvas, vector< CuMatrix<T>> parts, bool colMajor = false);
	__host__ __device__ cudaError_t rowCopy(CuMatrix<T>& targ, int tRow, int sRow) const;

	__host__ __device__ CuMatrix<T> derefIndices(const IndexArray& tary) const;

	static cudaError_t copyIndexed(CuMatrix<T>& targ, const IndexArray& tary,  const CuMatrix<T>& src, const IndexArray& sary);

	// copy 'fraction' (normalized) of rows into 'trg' and remainder into 'leftovers' shuffling row order by using
	// vIndices to hold a random sequence that become the new indices
	void shuffle(CuMatrix<T>* trg, CuMatrix<T>* leftovers, T fraction, vector<int>& vIndices ) const;

	std::pair<int,int> refCounts() const { return getMgr().refCounts(*this); }

	int hRefCount() { return elements ? getMgr().refCount(elements) : 0;}
	int dRefCount() { return tiler.buff() ? getMgr().refCount(tiler.buff()) : 0;}

	void toDevice(int dev);

	inline friend Migrator<T>& operator<<(Migrator<T>& mgrtr, CuMatrix<T>& m)  {
		return mgrtr.migrate(m);
	}

	__host__ __device__ void zero();

	__host__ void toFile(const char* fileName) const;

	__host__ CUDART_DEVICE CuMatrix<T> rightConcatenate( const CuMatrix<T>& other, cudaStream_t stream = 0) const;
	__host__ CUDART_DEVICE CuMatrix<T> bottomConcatenate(const CuMatrix<T>& other, cudaStream_t stream = 0) const;

	__host__ CUDART_DEVICE void set(int r, int c, T val);
	__host__ CUDART_DEVICE void set(int l, T val);
	__host__ __device__ T get(int r, int c) const;
	__host__ __device__ T getH(int r, int c) const;
	__host__ __device__ T get(long l) const;

// Qs
	__host__ __device__ inline bool zeroDimsQ() const {
		return m == 0 && n == 0;
	}
	__host__ __device__ inline bool containableQ(const CuMatrix<T>& o) const {
		return o.m <= m && o.n <= n;
	}
	__host__ __device__ inline bool compatibleQ(const CuMatrix<T>& o) const {
		return o.m == m && o.n == n && o.p == p;
	}
	bool zeroQ(T eps  = util<T>::epsilon());
	__host__ __device__ inline bool contiguousQ() const { return n == p; };
	__host__ __device__ inline bool paddedQ() { return !contiguousQ(); }
	__host__ __device__ inline bool biLocatedQ() const {
		return (elements && tiler.hasDmemQ());
	}
	__host__ __device__ inline bool gpuReadyQ() const {
		return tiler.hasDmemQ() &&
				(lastMod == mod_synced || lastMod == mod_device || lastMod == mod_neither);
	}
	__host__ __device__ inline bool synchedQ() const {
		return elements != null && lastMod == mod_synced;
	}
	__host__ __device__ inline bool hostReadyQ() const {
		return elements && (lastMod == mod_synced || lastMod == mod_host);
	}
	__host__ __device__ inline bool vectorQ() const {
		return (m == 1 || n == 1);
	}
	__host__ __device__ inline ulong length() const { return m * n;}
	__host__ __device__ inline bool squareQ() const { return n == m; }
	__host__ __device__ inline bool rowVectorQ() const { return m == 1;}
	__host__ __device__ inline bool columnVectorQ() const { return n == 1;}
	__host__ __device__ inline bool scalarQ() const { return m == 1 && n == 1;}
	__host__ __device__ inline bool validDimsQ() const { return m != 0 && n != 0; }
	__host__ __device__ inline bool validColQ(int col) const { return col < n; }
	__host__ __device__ inline bool validRowQ(int row) const { return row < m; }
	__host__ __device__ inline bool validIndicesQ(int row, int col) const { return col < n && row < m; };
	__host__ __device__ bool biasedQ() const;

#ifdef  CuMatrix_Enable_KTS
	template <template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE bool all(BoolUnaryOp<T> op) const;
	template <template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE long count(BoolUnaryOp<T> op) const;
	template <template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE long countRows(BoolUnaryOp<T> op) const;
	template <template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE IndexArray find(BoolUnaryOp<T> op) const;
	template <template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE void findFirstN(IndexArray arry, BoolUnaryOp<T> op) const;
	template <template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE bool any(BoolUnaryOp<T> op) const;
	template <template <typename> class BoolUnaryOp> __host__ CUDART_DEVICE bool none(BoolUnaryOp<T> op) const;
#else
	template <int StateDim> __host__ CUDART_DEVICE bool all(UnaryOpF<T,StateDim> op) const;
	template <int StateDim> __host__ CUDART_DEVICE long count(UnaryOpF<T,StateDim> op) const;
	template <int StateDim> __host__ CUDART_DEVICE long countRows(UnaryOpF<T,StateDim> op) const;
	template <int StateDim> __host__ CUDART_DEVICE IndexArray find(UnaryOpF<T,StateDim> op) const;
	template <int StateDim> __host__ CUDART_DEVICE void findFirstN(IndexArray arry, UnaryOpF<T,StateDim> op) const;
	template <int StateDim> __host__ CUDART_DEVICE bool any(UnaryOpF<T,StateDim> op) const;
	template <int StateDim> __host__ CUDART_DEVICE bool none(UnaryOpF<T,StateDim> op) const;
#endif

	template <typename IndexUnaryOp, template <typename> class BoolUnaryOp> bool indexedAll(IndexUnaryOp idxOp, BoolUnaryOp<T> op) const;
	template <typename IndexUnaryOp, template <typename> class BoolUnaryOp> bool indexedAny(IndexUnaryOp idxOp, BoolUnaryOp<T> op) const;
	template <typename IndexUnaryOp, template <typename> class BoolUnaryOp> bool indexedNone(IndexUnaryOp idxOp, BoolUnaryOp<T> op) const;

	bool isBinaryCategoryMatrix() const;

// extent

	inline intPair dims() const { return intPair( m,n );}
	inline int longAxis() const {	return MAX(m,n); }
	inline __host__ CUDART_DEVICE T vectorLength() const {
		return sqrt_p(autoDot());
	}
	T max( cudaStream_t stream = 0) const;
	T min( cudaStream_t stream = 0) const;
	IndexArray rowIndices(int row) const;
	IndexArray columnIndices(int col) const;
	inline float aspect() { return 1.0f * m / n; }
	template <int N> static int sizeOfArray( T (& arry)[N]) {
		return N;
	}

	CuMatrix<T> toBinaryCategoryMatrix() const;
	CuMatrix<T> toMaxColumnIndexVector()const;
	ulong distinctCount() const;
	std::set<T> distinct() const;

// form
	__host__ CUDART_DEVICE CuMatrix<T> transpose() const;
	__host__ CUDART_DEVICE CuMatrix<T> transposeKernelPtr(void (*kernel)(const T*  sElements,  T* tElements, int width, int height));
	__host__ CUDART_DEVICE void transpose(DMatrix<T>& res);
	__host__ CUDART_DEVICE void transposeKernelPtr(DMatrix<T>& res, void (*kernel)(const T*  sElements,  T* tElements, int width, int height));

	// create new matrix from array of row indices into this matrix
	CuMatrix<T> reconstruct(const IndexArray& arry);
	CuMatrix<T> poseAsRow();
	CuMatrix<T> poseAsCol();
	CuMatrix<T>& unPose();

	__host__ CUDART_DEVICE CuMatrix<T> reshape(int rows, int cols, long offsetInTs);
	__host__ CUDART_DEVICE void reshape(CuMatrix<T>& target, int rows, int cols, ulong offsetInTs);
	__host__ CUDART_DEVICE void recreate(int rows, int cols, int pitch, bool allocH, bool allocD );
	void unconcat(CuMatrix<T>& v, int rows, int cols, int pitch, int offsetInTs, bool colMajor = false) const;
	__host__ CUDART_DEVICE void submatrix(CuMatrix<T>& v, int rows, int cols, int roff, int coff) const;
	CuMatrix<T> redimension(pair<int, int>& dims, int offset = 0);
	__host__ CUDART_DEVICE CuMatrix<T> columnMatrix(int col) const;
	CuMatrix<T> rowMatrix(int row) const;
	CuMatrix<T> dropFirst(bool copy=true) const;
	CuMatrix<T> dropLast(bool copy=false) const;
	CuMatrix<T> vectorToDiagonal() const;
	__host__ CUDART_DEVICE CuMatrix<T> columnVector(int col) const;
	CuMatrix<T> rowVector(int row) const;
	CuMatrix<T> toRowVector() const;
	CuMatrix<T> toColumnVector() const;
	CuMatrix<T> toDiagonalsVector() const;
	T toScalar() const;
	__host__ CUDART_DEVICE CuMatrix<T> extrude(int depth) const;

	__host__ CUDART_DEVICE inline Modification getLastMod() const { return lastMod; }

/*
 * math
 */

	CuMatrix<T> featureMeans(bool lv)const;
	void featureMeans( CuMatrix<T>& means, bool lv)const;
	void featureMeansTx( CuMatrix<T>& means)const;
	void featureMeansStreams( CuMatrix<T>& means, bool lv,int nstreams)const;
	CuMatrix<T> subMeans( const CuMatrix<T>& means)const;
	__host__ CUDART_DEVICE void subMeans( CuMatrix<T>& res, const CuMatrix<T>& means)const;
	CuMatrix<T> sqrSubMeans(const CuMatrix<T>& mus)const; // sub means but sqaur
	cudaError_t sqrSubMeans( CuMatrix<T>& res, const CuMatrix<T>& mus)const;
	CuMatrix<T> normalize()const;
	__host__ CUDART_DEVICE void rowSum(CuMatrix<T>& rowSumM)const;
	__host__ CUDART_DEVICE CuMatrix<T> rowSum()const;
	int sgn(int row, int col)const;
	void matrixMinorM(CuMatrix<T>& trg, int row, int col) const;
	CuMatrix<T> matrixMinorM(int row, int col) const;
	T matrixMinor(int row, int col) const;
	T cofactor(int row, int col)const;
	CuMatrix<T> cofactorM()const;
	T determinant()const;
	CuMatrix<T> inverse()const;
	CuMatrix<T> inverse(T determinant)const;

	__host__ CUDART_DEVICE T norm(int l);

	// todo move this to anom det
	void fitGaussians( CuMatrix<T>& sqrdSigmas, CuMatrix<T>& mus) const;
	void variance( CuMatrix<T>& sqrdSigmas, const CuMatrix<T>& mus) const;
	CuMatrix<T> toCovariance() const;
	void toCovariance(CuMatrix<T>& covariance) const;
	void multivariateGaussianFeatures( CuMatrix<T>& pden, const CuMatrix<T>& sigmaSquared, const CuMatrix<T>& mu);
	void mvGaussianVectorFromFeatures( CuMatrix<T>& pvec);
	void multivariateGaussianVector( CuMatrix<T>& pvec, const CuMatrix<T>& sigmaSquared, const CuMatrix<T>& mu);
	CuMatrix<T> multivariateGaussianVectorM( const CuMatrix<T>& sigmaSquared, const CuMatrix<T>& mu);

	__host__ CUDART_DEVICE CuMatrix<T> matrixProduct( const CuMatrix<T>& b, dim3* block = null, cudaStream_t stream = 0)const;
	__host__ CUDART_DEVICE CuMatrix<T> matrixProductResident(const CuMatrix<T>& b, dim3* block=null, cudaStream_t stream = 0)const;
	// mk => multi kernel, for matrix prods where either one or both mat are large enough to exhaust compute resources
	// todo multi-gpu mult
	// looping threads
	__host__ CUDART_DEVICE CuMatrix<T> mkMatrixProduct( const CuMatrix<T>& b, dim3* block = null, cudaStream_t stream = 0)const;

	__host__ CUDART_DEVICE CuMatrix<T> subFrom(T o)const;

	__host__ CUDART_DEVICE CuMatrix<T> hadamardProduct(const CuMatrix<T> o)const;
	__host__ CUDART_DEVICE CuMatrix<T> hadamardQuotient(const CuMatrix<T> o)const;

// printing
	string dimsString() const;
	string toShortString() const;
	__host__ __device__ void printShortString(const char* msg = null) const;
	string toString() const;
	string pAsRow();
	__host__ __device__ void print(const char* msg) const;
	__host__ __device__ void printRow(int row) const;
	__host__ string rowStr(int row) const;

// stats
	__host__ __device__ float flow(int iterations, int iterationMemoryFactor, float exeTime);


// unary impls
#ifdef  CuMatrix_Enable_KTS
	template<template <typename> class UnaryOp> __host__ CUDART_DEVICE void unaryOp(CuMatrix<T>& res, UnaryOp<T> op, cudaStream_t stream = 0)const;
	template<template <typename> class UnaryOp> __host__ CUDART_DEVICE CuMatrix<T> unaryOp(UnaryOp<T> op, cudaStream_t stream = 0)const;
#else
	template<int StateDim> __host__ CUDART_DEVICE void unaryOp(CuMatrix<T>& res, UnaryOpF<T,StateDim> op, cudaStream_t stream = 0)const;
	template<int StateDim> __host__ CUDART_DEVICE CuMatrix<T> unaryOp(UnaryOpF<T,StateDim> op, cudaStream_t stream = 0)const;
#endif
	//template<typename UnaryOp> static void unaryOp(DMatrix<T>& dst, DMatrix<T>& src, UnaryOp op);

	__host__ CUDART_DEVICE CuMatrix<T> negate()const;
	__host__ CUDART_DEVICE CuMatrix<T> sigmoid()const;
	static __host__ CUDART_DEVICE void sigmoid(DMatrix<T>& trg, DMatrix<T> src, cudaStream_t stream = 0 );
	__host__ CUDART_DEVICE CuMatrix<T> sigmoidGradient()const;
	__host__ CUDART_DEVICE CuMatrix<T> log()const;
	__host__ CUDART_DEVICE CuMatrix<T> ceil()const;
	__host__ CUDART_DEVICE CuMatrix<T> floor()const;
	__host__ CUDART_DEVICE CuMatrix<T> oneOver()const;
	__host__ CUDART_DEVICE CuMatrix<T> exp()const;
	__host__ CUDART_DEVICE CuMatrix<T> sqrt()const;
	__host__ CUDART_DEVICE CuMatrix<T> sqr()const;
	__host__ CUDART_DEVICE CuMatrix<T> pow(T o)const;
	__host__ CUDART_DEVICE CuMatrix<T> divSqrt(T divisor)const;

	__host__ CUDART_DEVICE void setAll(int val);


// reduzziones
#ifdef  CuMatrix_Enable_KTS
	template<template <typename> class BinaryOp> __host__ CUDART_DEVICE void binaryOp(CuMatrix<T>& res, const CuMatrix<T>& o, BinaryOp<T> op, cudaStream_t stream = 0) const;
	template<template <typename> class BinaryOp> __host__ CUDART_DEVICE CuMatrix<T> binaryOp(const CuMatrix<T>& o, BinaryOp<T> op, cudaStream_t stream = 0) const;
#else
	template<int StateDim> __host__ CUDART_DEVICE void binaryOp(CuMatrix<T>& res, const CuMatrix<T>& o, BinaryOpF<T,StateDim> op, cudaStream_t stream = 0) const;
	template<int StateDim>__host__ CUDART_DEVICE CuMatrix<T> binaryOp(const CuMatrix<T>& o, BinaryOpF<T,StateDim> op, cudaStream_t stream = 0) const;
#endif

#ifdef  CuMatrix_Enable_KTS
	template <template <typename> class BinaryOp> __host__ CUDART_DEVICE T reduce(BinaryOp<T> op, T start, cudaStream_t stream = 0)const;
	template <template <typename> class BinaryOp> static __host__ CUDART_DEVICE T reduce(const DMatrix<T>& d_M, BinaryOp<T> op, T start, cudaStream_t stream = 0 );
#else
	template<int StateDim> __host__ CUDART_DEVICE T reduce(MonoidF<T,StateDim> op, T start, cudaStream_t stream = 0)const;
	template<int StateDim> static __host__ CUDART_DEVICE T reduce(const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, cudaStream_t stream = 0 );
#endif

#ifdef  CuMatrix_Enable_KTS
	template <template <typename> class BinaryOp> __host__ CUDART_DEVICE void reduceAsync(T* result, BinaryOp<T> op, T start, cudaStream_t stream = 0)const;
	template <template <typename> class BinaryOp> static __host__ CUDART_DEVICE void reduceAsync(T* result, const DMatrix<T>& d_M, BinaryOp<T> op, T start, cudaStream_t stream = 0 );
	template <template <typename> class BinaryOp> static __host__ CUDART_DEVICE void reduceAsyncBuffer(T* result, DMatrix<T>& buffer,  int blocks, int threads, long nP, const DMatrix<T>& d_M, BinaryOp<T> op, T start, cudaStream_t stream = 0 );
#else
	template<int StateDim> __host__ CUDART_DEVICE void reduceAsync(T* result, MonoidF<T,StateDim> op, T start, cudaStream_t stream = 0)const;
	template<int StateDim> static __host__ CUDART_DEVICE void reduceAsync(T* result, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, cudaStream_t stream = 0 );
	template<int StateDim> static __host__ CUDART_DEVICE void reduceAsyncBuffer(T* result, DMatrix<T>& buffer,  int blocks, int threads, long nP, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, cudaStream_t stream = 0 );
#endif

#ifdef  CuMatrix_Enable_KTS
	template <typename IndexBoolUnaryOp, template <typename> class BinaryOp> __host__ CUDART_DEVICE T indexedReduce(IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start, cudaStream_t stream = 0 )const;
	template <typename IndexBoolUnaryOp, template <typename> class BinaryOp> __host__ CUDART_DEVICE T indexedReduceL(const DMatrix<T>& d_M, IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start, cudaStream_t stream = 0 )const;
#else
	template <int IopDim, int BopDim> __host__ CUDART_DEVICE T indexedReduce(UnaryOpIndexF<T,IopDim> idxOp, MonoidF<T,BopDim> op, T start, cudaStream_t stream = 0 )const;
	template <int IopDim, int BopDim> __host__ CUDART_DEVICE T indexedReduceL(const DMatrix<T>& d_M, UnaryOpIndexF<T,IopDim>  idxOp, MonoidF<T,BopDim> op, T start, cudaStream_t stream = 0 )const;
#endif

#ifdef  CuMatrix_Enable_KTS
	template <template <typename> class BinaryOp> __host__ CUDART_DEVICE T reduceColumn(BinaryOp<T> op, T start, int col, cudaStream_t stream = 0)const;
	template <template <typename> class BinaryOp> static __host__ CUDART_DEVICE void reduceColumn(T* res, const DMatrix<T>& d_M, BinaryOp<T> op, T start, int col, cudaStream_t stream = 0 );
#else
	template <int StateDim> __host__ CUDART_DEVICE T reduceColumn(MonoidF<T,StateDim> op, T start, int col, cudaStream_t stream = 0)const;
	template <int StateDim> static __host__ CUDART_DEVICE void reduceColumn(T* res, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, T start, int col, cudaStream_t stream = 0 );
#endif

#ifdef  CuMatrix_Enable_KTS
	template <template <typename> class BinaryOp> static __host__ CUDART_DEVICE void reduceRowsNlte64(DMatrix<T>& resVec, const DMatrix<T>& d_M, BinaryOp<T> op, cudaStream_t stream = 0 );
	template <template <typename> class BinaryOp> static __host__ CUDART_DEVICE void reduceRows(DMatrix<T>& resVec, const DMatrix<T>& d_M, BinaryOp<T> op, cudaStream_t stream = 0 );
#else
	template<int StateDim> static __host__ CUDART_DEVICE void reduceRowsNlte64(DMatrix<T>& resVec, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, cudaStream_t stream = 0 );
	template<int StateDim> static __host__ CUDART_DEVICE void reduceRows(DMatrix<T>& resVec, const DMatrix<T>& d_M, MonoidF<T,StateDim> op, cudaStream_t stream = 0 );
#endif

#ifdef  CuMatrix_Enable_KTS
	template <template <typename> class UnaryOp, template <typename> class BinaryOp> __host__ CUDART_DEVICE T gloloReduce(UnaryOp<T> gop, BinaryOp<T> lop, T start, cudaStream_t stream = 0 ) const;
	template <template <typename> class UnaryOp, template <typename> class BinaryOp> __host__ CUDART_DEVICE T gloloReduceL(const DMatrix<T>& d_M, UnaryOp<T>  gop, BinaryOp<T> lop, T start, cudaStream_t stream = 0 )const;
#else
	template <int UopDim, int BopDim> __host__ CUDART_DEVICE T gloloReduce(UnaryOpF<T,UopDim> gop, MonoidF<T,BopDim> lop, T start, cudaStream_t stream = 0 ) const;
	template <int UopDim, int BopDim> __host__ CUDART_DEVICE T gloloReduceL(const DMatrix<T>& d_M, UnaryOpF<T,UopDim>  gop, MonoidF<T,BopDim> lop, T start, cudaStream_t stream = 0 )const;
#endif

	/*
	 * combine reduce first combines two matrices using the 'cop' binary op, which is applied element-wise
	 *  and the intermediate result is effectively a new matrix of same dim
	 * 	next the reduction binary op 'bop' is applied per usual to the intermediate matrix
	 */
#ifdef  CuMatrix_Enable_KTS
	// matbinaryop to superpose (combine) matrices, binary op to reduce
	template <template <typename> class CombineOp, template <typename> class ReduceOp> __host__ CUDART_DEVICE
	T combineReduce(CombineOp<T> cop, ReduceOp<T> rop, const CuMatrix<T>& o, T start, cudaStream_t stream = 0 )const;
	template <template <typename> class CombineOp, template <typename> class ReduceOp> __host__ CUDART_DEVICE
	T combineReduce(CuMatrix<T>& buffer, CombineOp<T> cop, ReduceOp<T> rop, const CuMatrix<T>& o, T start, cudaStream_t stream = 0 )const;
	template <template <typename> class CombineOp, template <typename> class ReduceOp> __host__ CUDART_DEVICE T
		combineReduceL(const DMatrix<T>& d_M1, const DMatrix<T>& d_M2, CombineOp<T> cop, ReduceOp<T> rop, T start, cudaStream_t stream = 0)const;
	template <template <typename> class CombineOp, template <typename> class ReduceOp> __host__ CUDART_DEVICE T
		combineReduceL(CuMatrix<T>& buffer, const DMatrix<T>& d_M1, const DMatrix<T>& d_M2, CombineOp<T> cop, ReduceOp<T> rop, T start, cudaStream_t stream = 0)const;
#else
	template <int CopDim, int RopDim> __host__ CUDART_DEVICE
	T combineReduce( BinaryOpF<T,CopDim> cop, BinaryOpF<T,RopDim> rop, const CuMatrix<T>& o, T start, cudaStream_t stream = 0 )const;
	template <int CopDim, int RopDim> __host__ CUDART_DEVICE
	T combineReduce(CuMatrix<T>& buffer, BinaryOpF<T,CopDim> cop, BinaryOpF<T,RopDim> rop, const CuMatrix<T>& o, T start, cudaStream_t stream = 0 )const;
	template <int CopDim, int RopDim> __host__ CUDART_DEVICE T
		combineReduceL(const DMatrix<T>& d_M1, const DMatrix<T>& d_M2, BinaryOpF<T,CopDim> cop, BinaryOpF<T,RopDim> rop, T start, cudaStream_t stream = 0)const;
	template <int CopDim, int RopDim>  __host__ CUDART_DEVICE T
		combineReduceL(CuMatrix<T>& buffer, const DMatrix<T>& d_M1, const DMatrix<T>& d_M2, BinaryOpF<T,CopDim> cop, BinaryOpF<T,RopDim> rop, T start, cudaStream_t stream = 0)const;
#endif

#ifdef  CuMatrix_Enable_KTS
	template <template <typename> class BinaryOp> __host__ CUDART_DEVICE
	T columnReduceIndexed(BinaryOp<T> op, int column, T start, cudaStream_t stream = 0 )const;
	__host__ CUDART_DEVICE T columnSum(int column, cudaStream_t stream = 0) const;
	template <template <typename> class BinaryOp> __host__ CUDART_DEVICE T rowReduce(BinaryOp<T> op, int row, T start, cudaStream_t stream = 0 )const;

#else
	template<int StateDim> __host__ CUDART_DEVICE
	T columnReduceIndexed(MonoidF<T,StateDim> op, int column, T start, cudaStream_t stream = 0 )const;
	__host__ CUDART_DEVICE T columnSum(int column, cudaStream_t stream = 0) const;
	template<int StateDim> __host__ CUDART_DEVICE T rowReduce(MonoidF<T,StateDim> op, int row, T start, cudaStream_t stream = 0 )const;
#endif

	__host__ CUDART_DEVICE T sum(cudaStream_t stream = 0 ) const;
	__host__ T kahanSum() const;

	inline __host__ CUDART_DEVICE T autoDot( cudaStream_t stream = 0) const {
#ifdef  CuMatrix_Enable_KTS
		return gloloReduce(sqrUnaryOp<T>(), plusBinaryOp<T>(), 0, stream);
#else
		MonoidF<T,1> pl = Functory<T,plusBinaryOp>::pinch();
		return gloloReduce(Functory<T,sqrUnaryOp>::pinch(), pl, 0, stream);
#endif
	}
	__host__ CUDART_DEVICE T rowSum(int column, cudaStream_t stream = 0) const;
	__host__ CUDART_DEVICE T prod( cudaStream_t stream = 0) const;
	pair<T,T> bounds() const;
	void bounds(T* min, T* max) const;

	__host__ CUDART_DEVICE T sumSqrDiff( const CuMatrix<T>& o)const;
	__host__ CUDART_DEVICE T sumSqrDiff(CuMatrix<T>& reductionBuffer, const CuMatrix<T>& o )const;
	__host__ CUDART_DEVICE T accuracy( const CuMatrix<T>& o)const;
	__host__ CUDART_DEVICE T accuracy( CuMatrix<T>& reductionBuffer, const CuMatrix<T>& o)const;

	//
	// operators
	//
	// matrix-matrix
	__host__ CUDART_DEVICE CuMatrix<T> operator|=(  const CuMatrix<T> b) const; // right concat
	__host__ CUDART_DEVICE CuMatrix<T> operator/=( const CuMatrix<T> b) const; // bottom concat
	__host__ CUDART_DEVICE CuMatrix<T> operator-( const CuMatrix<T> o) const;
	__host__ CUDART_DEVICE CuMatrix<T> operator+( const CuMatrix<T> o) const;

	__host__ CUDART_DEVICE bool operator==( const CuMatrix<T> o) const;
	__host__ CUDART_DEVICE bool operator!=( const CuMatrix<T> o) const;

	__host__ CUDART_DEVICE CuMatrix<T> operator*( const CuMatrix<T> o) const; // matrix product
	__host__ CUDART_DEVICE CuMatrix<T> operator%( const CuMatrix<T> o) const; // hadamard product
	__host__ CUDART_DEVICE CuMatrix<T> operator&&( const CuMatrix<T> o) const; // elmtwise and
	__host__ CUDART_DEVICE CuMatrix<T> operator||( const CuMatrix<T> o) const; // elmtwise or

	__host__ CUDART_DEVICE CuMatrix<T> operator=(const CuMatrix<T> o); // assignment

	// matrix-scalar -> returns new matrix of element-wise application of op
	__host__ CUDART_DEVICE CuMatrix<T> operator^(T o) const;
	__host__ CUDART_DEVICE CuMatrix<T> operator<(T o) const;
	__host__ CUDART_DEVICE CuMatrix<T> operator<=(T o) const;
	__host__ CUDART_DEVICE CuMatrix<T> operator>(T o) const;
	__host__ CUDART_DEVICE CuMatrix<T> operator>=(T o) const;
	__host__ CUDART_DEVICE CuMatrix<T> operator==(T o) const;

	__host__ CUDART_DEVICE CuMatrix<T> operator+(T o) const;
	friend __host__ CUDART_DEVICE CuMatrix<T> operator+(T o, const CuMatrix<T> m){
		return m + o;
	}

	__host__ CUDART_DEVICE CuMatrix<T> operator-(T o) const;
	inline friend __host__ CUDART_DEVICE CuMatrix<T> operator-(T o, const CuMatrix<T> m) {
		return m.unaryOp( Functory<T,subFromUnaryOp>::pinch(o));
	}

	__host__ CUDART_DEVICE CuMatrix<T> operator*(T o) const; // elemwise scale

	inline friend __host__ CUDART_DEVICE CuMatrix<T> operator*(T o, const CuMatrix<T> m) {
		return m * o;
	}

	inline friend ostream& operator<<(ostream& os, const CuMatrix<T>& m)  {
		return os << m.toString();
	}

	__host__ CUDART_DEVICE CuMatrix<T> operator/(T o) const; // elemwise inv scale

	friend CuMatrix<T> operator/(T o, const CuMatrix<T> m) {
		return m.oneOver() * o;
	}


	__host__ CUDART_DEVICE bool almostEq( const CuMatrix<T>& o, T eps = util<T>::epsilon()) const;
	inline __host__ __device__ bool equalDims(const CuMatrix<T>& other) const { return m == other.m && n == other.n;}

	// statics
	static void initMemMgrForType(int maxThreads, int maxBlocks);
	static void cleanup();
	static __host__ void cleanupHost();
	static __device__ void cleanupDev();
	static string typeStr();

	// serialization (deserialization is static 'fromFile')
	static void toFile(const char* fileName, const CuMatrix<T>& o);
	static void toOctaveFile(const char* fileName, const CuMatrix<T>& o);
	static CuMatrix<T> fromFile(const char* fileName);
	static vector< CuMatrix<T> > fromFileN(const char* fileName);
	static int releaseFromMap(std::map<std::string, CuMatrix<T>*>& map);
	//static cudaError_t migrate(int dev, CuMatrix<T>& m, ...);

	static void parseDataLine(string line, T* elementData,
			int currRow, int rows, int cols,
			bool colMajor);
	static void parseCsvDataLine(const CuMatrix<T>* x, int currLine, string line, const char* sepChars);
	static map<string,dim3> parseOctaveMatrixSizes(const char * path );
	static map<string, CuMatrix<T>*> parseOctaveDataFile(
			const char * path, bool colMajor, bool matrixOwnsBuffer = true);
	static map<string, CuMatrix<T>*> parseCsvDataFile(
			const char * path, const char * sepChars, bool colMajor, bool matrixOwnsBuffer = true, bool hasXandY = false);
	static CuMatrix<T>& getMatrix(std::map<std::string, CuMatrix<T>*> map, const char* key);

	/*
	 * constructs a matrix from a buffer of (potentially) different data type
	 */
	static CuMatrix<T> fromBuffer(void* buffer, int elemBytes, T (*converter)(void*), int m, int n, int p);

	static void bounds( T* min, T* max, DMatrix<T>& minBuff, DMatrix<T>& maxBuff, const DMatrix<T>& src, int blocks, int threads, long nP ) ;

	//cudaError_t set( DMatrix<T> m, int row, int col, T val);
	static int maxSlices(int n);
	static __host__ CUDART_DEVICE void transposeL(DMatrix<T>& t, const DMatrix<T>& s);
	static __host__ CUDART_DEVICE void transposeKernelPtrL( DMatrix<T>& t, void (*kernel)(const T*, T*, int, int), const DMatrix<T>& s);

	static __host__ CUDART_DEVICE void rightConcatenateL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, cudaStream_t stream = 0);
	static __host__ CUDART_DEVICE void bottomConcatenateL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, cudaStream_t stream = 0);

	static dim3 DefaultMatProdBlock;

/*
	static cudaError_t matrixProductL(DMatrix<T>& d_res, const DMatrix<T>& d_A, const DMatrix<T>& d_B,
			 dim3* block);
*/
	static cudaError_t matrixProductL2(DMatrix<T>& d_res, const DMatrix<T>& d_A, const DMatrix<T>& d_B,
			 dim3* block);
	static cudaError_t matrixProductL3(DMatrix<T>& d_res, const DMatrix<T>& d_A, const DMatrix<T>& d_B,
			 dim3* block);
/*
	static cudaError_t matrixProductKPtrL(DMatrix<T>& d_res,
			void (*matProdKptr) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int),
			const DMatrix<T>& d_A, const DMatrix<T>& d_B,
			 dim3* block);
*/
	static cudaError_t matrixProductTxdbL(DMatrix<T>& d_res, const DMatrix<T>& d_A, const DMatrix<T>& d_B,
			 dim3* block);
	static cudaError_t matrixProductReduxL(DMatrix<T>& d_res, const DMatrix<T>& d_A, const DMatrix<T>& d_B,
			 dim3* block);

	static void toMaxColumnIndexVectorL(DMatrix<T>& trg, const DMatrix<T>& src);


	static __host__ CUDART_DEVICE void columnMatrixL(DMatrix<T>& d_column, const DMatrix<T>& d_x, int col);
	static void rowMatrixCmL(DMatrix<T>& d_row, const DMatrix<T>& d_x, int row);
	static void varianceL(DMatrix<T>& d_sqrdSigmas, const DMatrix<T>& d_X, const DMatrix<T>& d_Mus);
	static void varianceAndMeanL(DMatrix<T>& d_sqrdSigmas, DMatrix<T>& d_Mus, const DMatrix<T>& d_X);

	static __host__ CUDART_DEVICE void featureAvgKernelL(DMatrix<T>& d_means, const DMatrix<T>& d_x, bool localVar, cudaStream_t stream = 0);
	static __host__ CUDART_DEVICE void featureAvgMultiStreamL(DMatrix<T>& d_means, const DMatrix<T>& d_x, bool localVar, int nstreams);
	static __host__ CUDART_DEVICE void featureAvgTxdKernelL(DMatrix<T>& d_means, const DMatrix<T>& d_x);

	static void multivariateGaussianFeatures( DMatrix<T>& d_pden, const DMatrix<T>& d_x, const DMatrix<T>& d_sqrdSigmas, const DMatrix<T>& d_mu);
	static void mvGaussianVectorFromFeatures( DMatrix<T>& d_pvec, const DMatrix<T>& d_pdens);
	static void multivariateGaussianVector( DMatrix<T>& d_pvec, const DMatrix<T>& d_x, const DMatrix<T>& d_sqrdSigmas, const DMatrix<T>& d_mu);

	static __host__ CUDART_DEVICE void meanSubL(DMatrix<T>& d_res, const DMatrix<T>& d_x, const DMatrix<T>& d_means, cudaStream_t stream = 0);
	static void meanSubSqrL(DMatrix<T>& d_res, const DMatrix<T>& d_x, const DMatrix<T>& d_means);
	static void columnProduct(DMatrix<T>& d_prod, const DMatrix<T>& d_x);
	static __host__ CUDART_DEVICE void rowSum(DMatrix<T>& d_prod, const DMatrix<T>& d_x, cudaStream_t stream = 0);


	/*
	template<typename BinaryOp> static T reduceLauncher(T* d_odata, const T* d_idata,
			long n, BinaryOp op, T start, cudaStream_t stream = 0);
*/

	// initial reduction is of values generated from the block/thread index alone
#ifdef  CuMatrix_Enable_KTS
	template<template <typename> class IndexUnaryOp,template <typename> class BinaryOp> static __host__ CUDART_DEVICE T indexReduceLauncher(
			T* d_odata, long n, IndexUnaryOp<T> idxOp, BinaryOp<T> op, T start, cudaStream_t stream = 0);
	template<typename IndexBoolUnaryOp,template <typename> class BinaryOp> static __host__ CUDART_DEVICE
	T indexedReduceLauncher(DMatrix<T> res, const T* d_idata, long n, IndexBoolUnaryOp idxOp, BinaryOp<T> op, T start, cudaStream_t stream = 0);
#else
	template<int IopDim, int BopDim> static __host__ CUDART_DEVICE T indexReduceLauncher(
			T* d_odata, long n, UnaryOpIndexF<T,IopDim> idxOp, MonoidF<T,BopDim> op, T start, cudaStream_t stream = 0);
	template<int IopDim, int BopDim> static __host__ CUDART_DEVICE
	T indexedReduceLauncher(DMatrix<T> res, const T* d_idata, long n, UnaryOpIndexF<T,IopDim> idxOp, MonoidF<T,BopDim> op, T start, cudaStream_t stream = 0);
#endif
	static __host__ CUDART_DEVICE T factorial(int val);

	// initial reduction is of values predicated by block/thread index

	static void binaryCategoryKernelL(DMatrix<T>& t, const DMatrix<T>& s, bool oneBased);

	static bool colocatedQ(vector<CuMatrix<T>>  ms) {
		int currDev = -1;
		for( auto& m : ms) {
			if(currDev == -1) {
				currDev = m.tiler.deviceOfResidence();
			} else {
				if(currDev != m.tiler.deviceOfResidence()){
					outln("not on " << currDev << ": " << m.tiler);
					return false;
				}
			}
		}
		return true;
	}

	static int devByOccupancy(vector<CuMatrix<T>>  ms) {
		map<int,long> devOx;
		for( int i = 0; i < ExecCaps::deviceCount;i++) {
			devOx[i] = 0;
		}
		for( auto& m : ms)
			devOx[m.tiler.deviceOfResidence()] += m.size;
		int maxGpu;
		long maxOx = 0;
		for( const auto &p : devOx) {
			//printf("CUmat devByOccupancy:  dev %d has ox %ld\n",p.first, p.second);
			if(p.second > maxOx) {
				maxOx = p.second;
				maxGpu = p.first;
			}
		}
		//outln("maxOx " << maxOx << " on g'piux " << maxGpu);
		return maxGpu;
	}

	template<template <typename> class CostFunction> static void gradientApprox(	CostFunction<T> costFn,  DMatrix<T> theta, DMatrix<T> perturb, DMatrix<T> grad, T epsilon);
	static void copy1D(T* trg, const T* src, int amountInTs, int offsetInTs, int lengthInTs, cudaStream_t stream = 0);
	static void copyK(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static __host__ CUDART_DEVICE void copy(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void copyAsync(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void copyUlong(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void copyUint(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void copyUlongDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static __host__ CUDART_DEVICE void copyUintDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void copyIntDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void shuffleCopyRows(DMatrix<T>& trg, const DMatrix<T>& src, int* rowIndices); // indices must be dev mem
	static void initRand(int height, int width);
	static void freeRand();

	// les methodes du fillois
	static CuMatrix<T> freeform(int cols, const T* vals, ulong count );
	static inline __host__ __device__ void defaultBlock(dim3& dim) { dim.x = TX_BLOCK_SIZE; dim.y = TX_BLOCK_SIZE/4; }
	static __host__ CUDART_DEVICE CuMatrix<T> fromScalar(T t, bool colMajor = false);

#ifdef  CuMatrix_Enable_KTS
	template <template <typename> class FillOp> static __host__ CUDART_DEVICE void fillFn(FillOp<T> op, CuMatrix<T>& res, cudaStream_t stream = 0);
	template <template <typename> class FillOp> static __host__ CUDART_DEVICE void fillFnNsb(FillOp<T> op, CuMatrix<T>& res, int w2h = 8, cudaStream_t stream = 0);
#else
	template<int StateDim> static __host__ CUDART_DEVICE void fillFn(const UnaryOpIndexF<T,StateDim>& op, CuMatrix<T>& res, cudaStream_t stream = 0);
	template<int StateDim> static __host__ CUDART_DEVICE void fillFnNsb(UnaryOpIndexF<T,StateDim> op, CuMatrix<T>& res, int w2h = 8, cudaStream_t stream = 0);
#endif
	static __host__ CUDART_DEVICE CuMatrix<T> increasingColumns(T start, int rows, int cols, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> increasingRows(T start, int rows, int cols, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> fill(T t, int nRows, int nCols, bool colMajor = false, cudaStream_t stream = 0);
	static __host__ CUDART_DEVICE CuMatrix<T> sfill(T t, int nRows, int nCols, bool colMajor = false, cudaStream_t stream = 0);
	static __host__ CUDART_DEVICE CuMatrix<T> fill(T t, intPair dims, bool colMajor = false, cudaStream_t stream = 0);
	static __host__ CUDART_DEVICE CuMatrix<T> zeros(int nRows, int nCols, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> zeros(intPair dims, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> ones(int nRows, int nCols, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> ones(intPair dims, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> sin(int m, int n, T amplitude = 1./10, T period = 2*Pi,T phase =0, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> sin(intPair dims, T amplitude = 1, T period = 2*Pi, T phase =0, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> cos(int m, int n, T amplitude = 1, T period = 2*Pi, T phase=0, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> cos(intPair dims, T amplitude = 1, T period = 2*Pi, T phase=0, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> diagonal(int dim, T val, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> diagonal(int dim, const T* val, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> identity(int dim, bool colMajor = false); // should cache

	static cudaError_t randn(DMatrix<T>& d_ret, T epsilon = 1);
	static CuMatrix<T> randn(int rows, int cols, T epsilon = static_cast<T>(  1), bool colMajor = false);
	static CuMatrix<T> randn( const intPair& dims, float epsilon, bool colMajor = false);
	static CuMatrix<T> randn( const intPair& dims, bool colMajor = false);

	static __host__ CUDART_DEVICE CuMatrix<T> span(T start, T end, int m, int n, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> sequence(T start, int m, int n, bool colMajor = false);
	//static __host__ CUDART_DEVICE CuMatrix<T> sequenceScale(T start, T scale, int m, int n, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> seqMod(T start, T mod, int m, int n, bool colMajor = false);
	static __host__ CUDART_DEVICE CuMatrix<T> mapFeature(CuMatrix<T> m1, CuMatrix<T> m2, int degree = 6);

	static inline __host__ CUDART_DEVICE CuMatrix<T> reductionBuffer(int rows) { return zeros(rows,1);	}

	static typename MatProd<T>::MatProdKptr g_matrix_product_kernel;

private:
	static int MaxRowsDisplayed; // to force accessor calls which also sets device copy
	static int MaxColsDisplayed;
public:
	static string theTypeStr;
	static long Constructed;
	static long Destructed;
	static long HDCopied;
	static long DDCopied;
	static long DHCopied;
	static long HHCopied;
	static long MemHdCopied;
	static long MemDdCopied;
	static long MemDhCopied;
	static long MemHhCopied;
	static const T MaxValue;
	static const T MinValue;
	static curandState * devStates;

	// fryeends
	template <typename U> friend class Migrator;
	template <typename U> friend class Locator;

	static CuMatrix<T>* Identities[1024];

	// applies srcPix + factor(destPix - srcPix)
	static void linterp(CuMatrix<T>& result, const CuMatrix<T>& src, const CuMatrix<T>& dest, T factor);

	static __host__ void setMaxRowsDisplayed(int rows);
	static __host__ void setMaxColsDisplayed(int cols);
	static inline __host__ __device__ int getMaxRowsDisplayed() {
#ifndef __CUDA_ARCH__
		return MaxRowsDisplayed;
#else
		//return D_MaxRowsDisplayed;
		return 5;
#endif
	}
	static inline __host__ __device__ int getMaxColsDisplayed() {
#ifndef __CUDA_ARCH__
		return MaxColsDisplayed;
#else
		//return D_MaxColsDisplayed;
		return 5;
#endif
	}

};

template<typename T> string pdm(const DMatrix<T>& md);

template<typename T> inline __device__ void SetElement(DMatrix<T>& A, int row, int col,
		float value) {
	A.elements[row * A.p + col] = value;
}
template<typename T> inline __device__ DMatrix<T> GetSubMatrix( const DMatrix<T>& A,
		int row, int col, const dim3& dimBlock) {
	DMatrix<T> Asub;
	Asub.n = MIN(dimBlock.x, A.n - col * dimBlock.x);
	Asub.m = MIN(dimBlock.y, A.m - row * dimBlock.y);
	Asub.p = A.p;
	Asub.elements = &A.elements[A.p * dimBlock.y * row + dimBlock.x * col];
	return Asub;
}

template <typename T> int numel(const CuMatrix<T>& x) {
	return x.m * x.n;
}

template <typename T> struct PackedMat {
	CuMatrix<T>* nn_params;
	bool owner;
	int layers;
	const uint2* dims;

	PackedMat( CuMatrix<T>* nn_params,const  int layers, const uint2* dims) : nn_params{nn_params}, owner{false}, layers{layers}, dims(dims) {
		outln("PackedMat created " << this);
	}
	PackedMat( CuMatrix<T>* nn_params,bool ownier, const  int layers, const uint2* dims) : nn_params{nn_params}, owner{ownier}, layers{layers}, dims(dims) {
		outln("PackedMat created " << this);
	}
	PackedMat(const PackedMat<T>& o);
	~PackedMat() {
		outln("PackedMat destroying this " << this  );

		delete[] dims;
		if(owner) {
			outln("PackedMat freeing nn_params " << nn_params->toShortString());
			delete nn_params;
		}else {
			outln("PackedMat sparing nn_params " << nn_params->toShortString());
		}
	}
	string dumpDims() {
		stringstream ss;
		for(int i =0; i < layers; i++) {
			ss << "i " << i << " x: " << dims[i].x << ", y: " << dims[i].y  << "\n";
		}
		return ss.str();
	}

	static PackedMat<T> pack( const vector<CuMatrix<T> >& v);
	static void pack( CuMatrix<T>& nn_params, uint2*& dims, int& layers, const vector<CuMatrix<T> >& v);
};

extern template class PackedMat<float>;
extern template class PackedMat<double>;
extern template class PackedMat<long>;
extern template class PackedMat<ulong>;
extern template class PackedMat<int>;
extern template class PackedMat<uint>;


