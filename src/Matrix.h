#ifndef MATRIX_H_
#define MATRIX_H_
#include <cstddef>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include "CuMatrix.h"
#include "debug.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include "util.h"
#include "functors.h"
#include "MemMgr.h"
#include <typeinfo>

/*
 * TODO cocktail party algorithm in octave
 * [W,s,v] = svd((repmat(sum(x .* x, 1), size(x,1),1) .* x) * x')
 */

#define MaxDim 65536u
#undef minor

template<typename T> class MemMgr;
/* rows and cols are 0-based! */

template<typename T> class Matrix : public CuMatrix<T> {

public:
	static Matrix<T> ZeroMatrix;
public:
	static bool verbose;

	static MemMgr<T>* mgr;
#define zerom Matrix<T>::ZeroMatrix
	MemMgr<T>& getMgr();

private:
	Matrix<T>* txp;  // c++ limitation is that this won't compile as a value, only a pointer, because size of Matrix<T> is 'unknown' at this point
	bool ownsTxp;
	void initMembers();
	void freeTxp();
public:
	Matrix();
	Matrix(const Matrix<T>& o);
	Matrix(const Matrix<T>& o, bool alloc);
	Matrix(const Matrix<T>& o, uint rows, uint cols, uint pitch, uint offset);
	Matrix( T* h_data, uint m, uint n, bool allocateD);
	Matrix( T* h_data, uint m, uint n, uint p, bool allocateD);
	Matrix(uint m, uint n, uint p, bool allocate, bool allocateD  = false);
	Matrix(uint m, uint n, bool allocate, bool allocateD  = false);
	Matrix(const DMatrix<T>& o, bool allocate_h,	bool copy);

	virtual ~Matrix();

	Matrix copy(bool copyDeviceMem = true) const;

	void invalidateHost();
	void invalidateDevice();

	// memory
	__host__ cudaError_t asDmatrix(DMatrix<T>& mf, bool copy = true, bool force = false) const;
	__host__ DMatrix<T> asDmatrix( bool copy = true) const;
	__host__ cudaError_t fromDevice(const DMatrix<T>& mf, bool copy = true);
	__host__ Matrix<T>& syncBuffers(bool copy = true);
	__host__ Matrix<T> syncHost();
	__host__ Matrix<T> syncDevice();

	// serialization (deserialization is static 'fromFile')
	__host__ cudaError_t toStream(std::ofstream& ofs) const;

	// Qs
	bool zeroDimsQ() const;
	bool zeroQ(T eps  = util<T>::epsilon());
	inline bool contiguousQ() const { return this->n == this->p; }
	inline bool paddedQ() { return !contiguousQ(); }
	inline bool biLocatedQ() const;
	inline bool gpuReadyQ() const;
	inline bool vectorQ() const;
	inline bool squareQ() const;
	inline bool rowVectorQ() const;
	inline bool columnVectorQ() const;
	inline bool scalarQ() const;
	inline bool validDimsQ() const;
	inline bool validColQ(uint col) const;
	inline bool validRowQ(uint row) const;
	inline bool validIndicesQ(uint row, uint col) const;
	bool hasBiasColumn();
	inline T flow(int iterations, int iterationMemoryFactor, float exeTime) {
		// iterationMemoryFactor eg how many reads and or writes of mat.size
		return (T) iterationMemoryFactor * 1000. * this->size / Giga / (exeTime/ iterations);
	}

	template <typename BoolUnaryOp> bool all(BoolUnaryOp op) const;
	template <typename BoolUnaryOp> bool any(BoolUnaryOp op) const;
	template <typename BoolUnaryOp> bool none(BoolUnaryOp op) const;

	// todo? for (all|none|any) in row or column (or diag etc)
	template <typename IndexUnaryOp, typename BoolUnaryOp> bool indexedAll(IndexUnaryOp idxOp, BoolUnaryOp op) const;
	template <typename IndexUnaryOp, typename BoolUnaryOp> bool indexedAny(IndexUnaryOp idxOp, BoolUnaryOp op) const;
	template <typename IndexUnaryOp, typename BoolUnaryOp> bool indexedNone(IndexUnaryOp idxOp, BoolUnaryOp op) const;

	bool isBinaryCategoryMatrix() const;
	uintPair dims() const;
	uint longAxis() const;
	T vectorLength() const;

	// form
	inline void updateSize() { this->size = this->p * this->m * sizeof(T); }
	inline void allocDevice() { getMgr().allocDevice(*this); }
	inline void allocHost() { getMgr().allocHost(*this); }

	Matrix<T> toBinaryCategoryMatrixCPU() const;
	Matrix<T> toBinaryCategoryMatrix() const;
	Matrix<T> transposeCpu() const;
	Matrix<T> transpose() const;
	Matrix<T> transposeKernelPtr(void (*kernel)(const T*  sElements,  T* tElements, uint width, uint height));
	void transpose(DMatrix<T>& res);
	void transposeKernelPtr(DMatrix<T>& res, void (*kernel)(const T*  sElements,  T* tElements, uint width, uint height));

	void shuffle(Matrix<T>& trg, Matrix<T>& leftovers, T fraction, vector<uint>& vIndices = null) const;
	Matrix<T> poseAsRow();
	Matrix<T> poseAsCol();
	Matrix<T> unPose();

	Matrix<T> reshape(uint rows, uint cols, ulong offsetInTs);
	void reshape(Matrix<T>& target, uint rows, uint cols, ulong offsetInTs);
	void unconcat(Matrix<T>& v, uint rows, uint cols, uint pitch, uint offsetInTs) const;
	void submatrix(Matrix<T>& v, uint rows, uint cols, uint roff, uint coff) const;
	Matrix<T> redimension(std::pair<uint, uint>& dims, uint offset = 0);
	Matrix<T> columnMatrix(int col) const;
	Matrix<T> dropFirst(bool copy=false) const;
	Matrix<T> vectorToDiagonal() const;

	Matrix<T> columnVector(int col) const;
	Matrix<T> rowVector(int row) const;
	Matrix<T> toRowVector() const;
	Matrix<T> toColumnVector() const;
	T toScalar() const;
	Matrix<T> toDiagonalsVector() const;
	inline T* getElements() { return CuMatrix<T>::elements;}


	Matrix<T> columnSubset( const uint* indices, uint count) const;
	Matrix<T> clippedRowSubset( const int *r, uint count, uintPair colRange) const;
	Matrix<T> addBiasColumn() const;
	IndexArray rowIndices(uint row) const;
	IndexArray columnIndices(uint col) const;
	void copy(Matrix<T>& ref, int roff, int coff) const;
	cudaError_t rowCopy(Matrix<T>& targ, uint tRow, uint sRow) const;
	Matrix<T> rightConcatenate( const Matrix<T>& other) const;
	Matrix<T> bottomConcatenate(const Matrix<T>& other) const;
    Matrix<T> prependColumnNew( T* col, uint count) const;
	Matrix<T> appendColumnNew( T* col, uint count) const;
	Matrix<T> prependRowNew( T* row, uint count) const;
	Matrix<T> appendRowNew( T* row, uint count) const;
	void flipud();
	// turns row matrix into matrix of depth copies of the row
	Matrix<T> extrude(uint depth) const;

	// printing
	string toString() const;
	string pAsRow();

	void set(uint r, uint c, T val);
	void set(uint l, T val);
	T get(uint r, uint c) const; // not const due to possible syncBuffers()
	T get(uint l) const;

	/*
	 * math
	 */


	template<typename UnaryOp> void unaryOp(Matrix<T>& res, UnaryOp op)const;
	template<typename UnaryOp> Matrix<T> unaryOp(UnaryOp op)const;

	template<typename BinaryOp> void binaryOp(const Matrix<T>& o, Matrix<T>& res, BinaryOp op)const;
	template<typename BinaryOp> Matrix<T> binaryOp(const Matrix<T>& o, BinaryOp op)const;

	template <typename BinaryOp> T reduce(BinaryOp op, T start, cudaStream_t stream = 0)const;
	template <typename BinaryOp> static __host__ T reduce(const DMatrix<T>& d_M, BinaryOp op, T start, cudaStream_t stream = 0 );

	template <typename IndexBoolUnaryOp, typename BinaryOp> T indexedReduce(IndexBoolUnaryOp idxOp, BinaryOp op, T start)const;
	template <typename IndexBoolUnaryOp, typename BinaryOp> __host__ T indexedReduceL(const DMatrix<T>& d_M, IndexBoolUnaryOp idxOp, BinaryOp op, T start)const;

	template <typename UnaryOp, typename BinaryOp> T gloloReduce(UnaryOp gop, BinaryOp lop, T start) const;
	template <typename UnaryOp, typename BinaryOp> __host__ T gloloReduceL(const DMatrix<T>& d_M, UnaryOp gop, BinaryOp lop, T start)const;

	template <typename IndexBoolUnaryOp, typename UnaryOp, typename BinaryOp> T indexedGloloReduce(IndexBoolUnaryOp idxOp, UnaryOp gop, BinaryOp lop, T start)const;

	template <typename IndexBoolUnaryOp, typename UnaryOp, typename BinaryOp> __host__ T
		indexedGloloReduceL(const DMatrix<T>& d_M, IndexBoolUnaryOp idxOp, UnaryOp gop, BinaryOp lop, T start)const;

	template <typename MatBinaryOp, typename BinaryOp> T matrixReduce(MatBinaryOp mop, BinaryOp op, const Matrix<T>& o, T start)const;
	template <typename MatBinaryOp, typename BinaryOp> T matrixReduce(MatBinaryOp mop, BinaryOp op, const Matrix<T>& o, Matrix<T>& temp, T start)const;
	template <typename MatBinaryOp, typename BinaryOp> __host__ T
		matrixReduceL(const DMatrix<T>& d_M1, const DMatrix<T>& d_M2, MatBinaryOp mop, BinaryOp op, T start)const;
	template <typename MatBinaryOp, typename BinaryOp> __host__ T
		matrixReduceL(const DMatrix<T>& d_M1, const DMatrix<T>& d_M2, const Matrix<T>& temp, MatBinaryOp mop, BinaryOp op, T start)const;

	T sum(  cudaStream_t stream = 0) const;
	T autoDot() const;
	template <typename BinaryOp> T columnReduce(BinaryOp op, uint column, T start )const;
	T columnSum(uint column) const;
	template <typename BinaryOp> T rowReduce(BinaryOp op, uint row, T start )const;
	T rowSum(uint column) const;
	T prod( cudaStream_t stream = 0) const;
	T max( cudaStream_t stream = 0) const;
	T min( cudaStream_t stream = 0) const;
	pair<T,T> bounds() const;

	T sumSqrDiff( const Matrix<T>& o)const;
	T sumSqrDiff(Matrix<T>& reductionBuffer, const Matrix<T>& o )const;
	T accuracy( const Matrix<T>& o)const;
	T accuracy( Matrix<T>& reductionBuffer, const Matrix<T>& o)const;

	Matrix<T> featureMeans(bool lv)const;
	void featureMeans( Matrix<T>& means, bool lv)const;
	Matrix<T> subMeans( const Matrix<T>& means)const;
	void subMeans( Matrix<T>& res, const Matrix<T>& means)const;
	Matrix<T> sqrSubMeans(const Matrix<T>& mus)const; // sub means but sqaur
	cudaError_t sqrSubMeans( Matrix<T>& res, const Matrix<T>& mus)const;
	void rowProductTx(Matrix<T>& res)const;
	void rowSum(Matrix<T>& rowSumM)const;
	Matrix<T> rowSum()const;
	int sgn(uint row, uint col)const;
	Matrix<T> minorM(uint row, uint col)const;
	T minor(uint row, uint col)const;
	T cofactor(uint row, uint col)const;
	Matrix<T> cofactorM()const;
	T determinant()const;
	Matrix<T> inverse()const;

	void fitGaussians( Matrix<T>& sqrdSigmas, Matrix<T>& mus) const;
	void variance( Matrix<T>& sqrdSigmas, const Matrix<T>& mus) const;
	Matrix<T> toCovariance() const;
	void toCovariance(Matrix<T>& covariance) const;
	void multivariateGaussianFeatures( Matrix<T>& pden, const Matrix<T>& sigmaSquared, const Matrix<T>& mu);
	void mvGaussianVectorFromFeatures( Matrix<T>& pvec);
	void multivariateGaussianVector( Matrix<T>& pvec, const Matrix<T>& sigmaSquared, const Matrix<T>& mu);
	Matrix<T> multivariateGaussianVectorM( const Matrix<T>& sigmaSquared, const Matrix<T>& mu);
	  //  Xi - μi
	  //  -------
	  //     σ
	Matrix<T> normalize()const;

	Matrix<T> matrixProduct( const Matrix<T>& b, dim3* block = null)const;
	Matrix<T> toMaxColumnIndexVector()const;
	Matrix<T> subFrom(T o)const;

	Matrix<T> hadamardProduct(const Matrix<T> o)const;
	Matrix<T> hadamardQuotient(const Matrix<T> o)const;

	// unary impls

	Matrix<T> negate()const;
	Matrix<T> sigmoid()const;
	Matrix<T> sigmoidGradient()const;
	Matrix<T> log()const;
	Matrix<T> oneOver()const;
	Matrix<T> exp()const;
	Matrix<T> sqrt()const;
	Matrix<T> sqr()const;
	Matrix<T> pow(T o)const;
	Matrix<T> divSqrt(T divisor)const;

	//
	// operators
	//
	Matrix<T> operator=(const Matrix<T> o);

	Matrix<T> operator^(T o) const;
	Matrix<T> operator^(int o) const;

	Matrix<T> operator<(T o) const;
	Matrix<T> operator<=(T o) const;
	Matrix<T> operator>(T o) const;
	Matrix<T> operator>=(T o) const;
	Matrix<T> operator==(T o) const;

	Matrix<T> operator+( const Matrix<T> o) const;
	Matrix<T> operator+(T o) const;
	friend Matrix<T> operator+(T o, const Matrix<T> m){
		return m + o;
	}

	Matrix<T> operator-( const Matrix<T> o) const;
	Matrix<T> operator-(T o) const;
	inline friend Matrix<T> operator-(T o, const Matrix<T> m) {
		subFromUnaryOp<T> subf;
		subf.source = o;
		return m.unaryOp( subf);
	}

	Matrix<T> operator*( const Matrix<T> o) const;
	Matrix<T> operator%( const Matrix<T> o) const;
	Matrix<T> operator&&( const Matrix<T> o) const;
	Matrix<T> operator*(T o) const;
	inline friend Matrix<T> operator*(T o, const Matrix<T> m) {
		return m * o;
	}

	inline friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& m)  {
		return os << m.toString();
	}

	Matrix<T> operator/(T o) const;
	friend Matrix<T> operator/(T o, const Matrix<T> m) {
		return m.oneOver() * o;
	}

	Matrix<T> operator|=(  const Matrix<T> b) const; // right concat
	Matrix<T> operator/=( const Matrix<T> b) const; // bottom concat

	bool almostEq( const Matrix<T>& o, T eps = util<T>::epsilon()) const;
	bool operator==( const Matrix<T> o) const;
	bool operator!=( const Matrix<T> o) const;

	/*
	 * statics
	 */
	static Matrix<T> fromFile(const char* fileName);
	static std::vector< Matrix<T> > fromFileN(const char* fileName);
	static string typeStr();
	static void init(int maxThreads, int maxBlocks);
	static void cleanup();

	template <int N> static int sizeOfArray( T (& arry)[N]) {
		return N;
	}

	// les methodes du filluer
	static void concat(Matrix<T>& canvas, int components, const Matrix<T>** parts);
	static Matrix<T> freeform(int cols, const T* vals, ulong count );
	static Matrix<T> fromScalar(T t, bool colMajor = false);
	template <typename FillOp> static void fillFn(FillOp op, Matrix<T>& ret );
	template <typename FillOp> static void fillFnCPU(FillOp op, Matrix<T>& ret );
	static Matrix<T> increasingColumns(T start, int rows, int cols, bool colMajor = false);
	static Matrix<T> increasingRows(T start, int rows, int cols, bool colMajor = false);
	static Matrix<T> fill(T t, uint nRows, uint nCols, bool colMajor = false);
	static Matrix<T> fill(T t, uintPair dims, bool colMajor = false);
	static Matrix<T> zeros(uint nRows, uint nCols, bool colMajor = false);
	static Matrix<T> zeros(uintPair dims, bool colMajor = false);
	static Matrix<T> ones(uint nRows, uint nCols, bool colMajor = false);
	static Matrix<T> ones(uintPair dims, bool colMajor = false);
	static Matrix<T> sin(uint m, uint n, T amplitude = 1./10, T period = 2*Pi,T phase =1, bool colMajor = false);
	static Matrix<T> sin(uintPair dims, T amplitude = 1, T period =1, T phase =0, bool colMajor = false);
	static Matrix<T> cos(uint m, uint n, T amplitude = 1, T period =1, T phase=0, bool colMajor = false);
	static Matrix<T> cos(uintPair dims, T amplitude = 1, T period =1, T phase=0, bool colMajor = false);
	static Matrix<T> diagonal(uint dim, T val, bool colMajor = false);
	static Matrix<T> diagonal(uint dim, const T* val, bool colMajor = false);
	static Matrix<T> identity(uint dim, bool colMajor = false);
	static Matrix<T> randn(uint rows, uint cols, T epsilon = static_cast<T>(  1), bool colMajor = false);
	static Matrix<T> randn( const uintPair& dims, float epsilon, bool colMajor = false);
	static Matrix<T> randn( const uintPair& dims, bool colMajor = false);
	static Matrix<T> sequence(T start, uint m, uint n, bool colMajor = false);
	static Matrix<T> seqMod(T start, T mod, uint m, uint n, bool colMajor = false);
	static Matrix<T> fromDmatrix(const DMatrix<T>& mf, bool allocate = true,bool copy = true);
	static Matrix<T> mapFeature(Matrix<T> m1, Matrix<T> m2, int degree = 6);
	static Matrix<T> reductionBuffer(uint rows);

	static uint MaxRowsDisplayed;
	static uint MaxColsDisplayed;
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
	// friends
	template <typename U> friend class MemMgr;
	friend cudaError_t copyIndexed(Matrix<T>& targ, const IndexArray& tary,  const Matrix<T>& src, const IndexArray& sary) {
		targ.syncHost();
		if(tary.count == sary.count && sary.count == 2) {
			// rowMajr to rowMaj
			//cudaMemcpy(targ.elements + tary.indices[0], src.elements + sary.indices[0], (tary.indices[1]-tary.indices[0])* sizeof(T), cudaMemcpyHostToHost);
			memcpy(targ.elements + tary.indices[0], src.elements + sary.indices[0], (tary.indices[1]-tary.indices[0])* sizeof(T));
			return cudaSuccess;
		} else if(tary.count == 2 && sary.count > 2) {
			uint start = tary.indices[0];
			uint tlen = tary.indices[1] - tary.indices[0];
			dassert(tlen == sary.count);
			for(uint i = 0; i < sary.count; i++) {
				targ.elements[start + i] = src.elements[ sary.indices[i]];
			}
		} else if(sary.count == 2 && tary.count > 2) {
			uint start = sary.indices[0];
			uint slen = sary.indices[1] - sary.indices[0];
			dassert(slen == tary.count);
			for(uint i = 0; i < tary.count; i++) {
				targ.elements[tary.indices[i]] = src.elements[start + i];
			}
		} else {
			outln("error, bad source indexarray " << sary.toString().c_str() << " or bad targ array " << tary.toString().c_str());
			return cudaErrorInvalidConfiguration;
		}
		targ.invalidateDevice();
		return cudaSuccess;
	}


};

template<typename T> Matrix<T> operator+(T lhs, Matrix<T> rhs) {
	return rhs + lhs;
}

template<typename T> Matrix<T> operator*(T lhs, Matrix<T> rhs) {
	return rhs * lhs;
}
template<typename T> Matrix<T> operator-(T lhs, Matrix<T> rhs) {
	return rhs.subFrom(lhs);
}

extern template class Matrix<float>;
extern template class Matrix<double>;

#endif
