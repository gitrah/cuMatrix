/*
 * CuMatrix.h
 *
 *  Created on: Dec 3, 2012
 *      Author: reid
 */

#ifndef CUMATRIX_H_
#define CUMATRIX_H_
//#define CUTMPLT
#include "util.h"
#include "functors.h"
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

using namespace std;

// Device Matrix
template<typename T> struct DMatrix {
public:
	T* elements;
	uint m, n, p;

	__host__ __device__ DMatrix() :
			elements(null), m(0), n(0), p(0) {
	}
	__host__ __device__ DMatrix(uint _m, uint _n) :
			elements(null), m(_m), n(_n), p(_n) {
	}
};

template<class T>
struct SharedMemory
{
	__device__ inline operator T *() {
		extern __shared__ int __smem[];
		return (T*) __smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T*) __smem;
	}
};

/*
 * base (device side) matrix class
 * methods follow the memcpy arg-sig standard of 'target' or output reference appearing first
 */
template<typename T> class CuMatrix {
public:
	static int MaxThreads;
	static int MaxBlocks;
protected:
	Modification lastMod;
public:
	T* elements;
	T* d_elements;
	//uint offset;

	uint m, n, p;
	uint oldM, oldN;
	uint size;
	bool posed, colMajor, ownsBuffers;

public:
	inline Modification getLastMod() const { return lastMod; }
	inline T aspect() { return 1.0 * m / n; }
	// printing
	string dimsString() const;
	string toShortString() const;
	inline bool contiguousQ() { return n == p; };

	//cudaError_t set( DMatrix<T> m, uint row, uint col, T val);
	static void getReductionExecContext(int &blocks, int &threads, ulong nP,
			int maxBlocks = MaxBlocks, int maxThreads = MaxThreads);

	static void transposeL(DMatrix<T>& t, const DMatrix<T>& s);
	static void transposeKernelPtrL( DMatrix<T>& t, void (*kernel)(const T*, T*, uint, uint), const DMatrix<T>& s);

	static void rightConcatenateL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2);
	static void bottomConcatenateL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2);

	static dim3 DefaultMatProdBlock;
	static cudaError_t matrixProductL(DMatrix<T>& d_res, const DMatrix<T>& d_A, const DMatrix<T>& d_B,
			 dim3* block);
	static cudaError_t matrixProductReduxL(DMatrix<T>& d_res, const DMatrix<T>& d_A, const DMatrix<T>& d_B,
			 dim3* block);

	static void toMaxColumnIndexVectorL(DMatrix<T>& trg, const DMatrix<T>& src);

	template<typename UnaryOp> static void unaryOpL(DMatrix<T>& trg,const DMatrix<T>& src, UnaryOp op);
	template<typename UnaryOp> static void unaryOpDmL(DMatrix<T>& trg,const DMatrix<T>& src, UnaryOp op, int w2h = DefaultWidth2Height);

	template<typename BinaryOp> static void binaryOpL( DMatrix<T>& trg, const DMatrix<T>& src1,
			const DMatrix<T>& src2, BinaryOp op);
	template<typename BinaryOp> static void binaryOpDmL( DMatrix<T>& trg, const DMatrix<T>& src1,
			const DMatrix<T>& src2, BinaryOp op, int w2h = DefaultWidth2Height);

	template<typename MatBinaryOp, typename BinaryOp>
	static T matrixReduceOpLauncher(T* d_odata, const T* d_idata1, const T* d_idata2, ulong n, MatBinaryOp mop, BinaryOp op, T start);

	static void columnMatrixL(DMatrix<T>& d_column, const DMatrix<T>& d_x, int col);
	static void varianceL(DMatrix<T>& d_sqrdSigmas, const DMatrix<T>& d_X, const DMatrix<T>& d_Mus);
	static void varianceAndMeanL(DMatrix<T>& d_sqrdSigmas, DMatrix<T>& d_Mus, const DMatrix<T>& d_X);

	static void featureAvgKernelL(DMatrix<T>& d_means, const DMatrix<T>& d_x, bool localVar);

	static void multivariateGaussianFeatures( DMatrix<T>& d_pden, const DMatrix<T>& d_x, const DMatrix<T>& d_sqrdSigmas, const DMatrix<T>& d_mu);
	static void mvGaussianVectorFromFeatures( DMatrix<T>& d_pvec, const DMatrix<T>& d_pdens);
	static void multivariateGaussianVector( DMatrix<T>& d_pvec, const DMatrix<T>& d_x, const DMatrix<T>& d_sqrdSigmas, const DMatrix<T>& d_mu);

	bool equalDims(const CuMatrix<T>& other) const;

	static void meanSubL(DMatrix<T>& d_res, const DMatrix<T>& d_x, const DMatrix<T>& d_means);
	static void meanSubSqrL(DMatrix<T>& d_res, const DMatrix<T>& d_x, const DMatrix<T>& d_means);
	static void columnProduct(DMatrix<T>& d_prod, const DMatrix<T>& d_x);
	static void rowSum(DMatrix<T>& d_prod, const DMatrix<T>& d_x);

	template<typename BinaryOp> static cudaError_t partialReduceLauncher( T* d_odata, uint n, BinaryOp op, T start, cudaStream_t stream = 0);

	template<typename BinaryOp> static T reduceLauncher(T* d_odata, const T* d_idata,
			ulong n, BinaryOp op, T start, cudaStream_t stream = 0);

	template<typename BinaryOp> static T reduceLauncherDm(DMatrix<T>& buff, const DMatrix<T>& src,
			ulong n, BinaryOp op, T start, cudaStream_t stream = 0);

	template<typename IndexBoolUnaryOp,typename BinaryOp> static T indexedReduceLauncher(
			T* d_odata, const T* d_idata, ulong n, IndexBoolUnaryOp idxOp, BinaryOp op, T start);

	template<typename UnaryOp, typename BinaryOp> static T gloloReduceOpLauncher(
			T* d_odata, const T* d_idata, ulong n, UnaryOp gop, BinaryOp lop,
			T start);

	template<typename IndexBoolUnaryOp, typename UnaryOp, typename BinaryOp> T indexedGloloReduceOpLauncher(
			T* d_odata, const T* d_idata, ulong n, IndexBoolUnaryOp idxOp, UnaryOp gop, BinaryOp lop,
			T start);

	static void binaryCategoryKernelL(DMatrix<T>& t, const DMatrix<T>& s, bool oneBased);

	template <typename FillOp> static void fillFn(FillOp op, CuMatrix<T>& res);
	template <typename FillOp> static void fillFnNsb(FillOp op, CuMatrix<T>& res, int w2h = 8);
	template<typename CostFunction> static void gradientApprox(	CostFunction costFn,  DMatrix<T> theta, DMatrix<T> perturb, DMatrix<T> grad, T epsilon);
	static cudaError_t randn(DMatrix<T>& d_ret, T epsilon = 1);
	static void copy1D(T* trg, const T* src, uint amountInTs, uint offsetInTs, uint lengthInTs, cudaStream_t* stream);
	static void copy(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void copyUlong(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void copyUint(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void copyUlongDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void copyUintDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void copyIntDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff);
	static void shuffleCopyRows(DMatrix<T>& trg, const DMatrix<T>& src, uint* rowIndices); // indices must be dev mem
	static void initRand(int height, int width);
	static void freeRand();

	static void set(T* elements, uint m, uint n, uint p, uint row, uint col, T val);
	static void set(T* elements, uint m, uint n, uint p, ulong idx, T val);

	static const T MaxValue;
	static const T MinValue;
	static const T DefaultWidth2Height;  // for non-square blocks that iterate across cache-lines
	static curandState * devStates;

};

template <typename T> const T CuMatrix<T>::MinValue = util<T>::minValue();
template <typename T> const T CuMatrix<T>::MaxValue = util<T>::maxValue();
template <typename T> dim3 CuMatrix<T>::DefaultMatProdBlock = dim3(16,16);

template <typename T> const T CuMatrix<T>::DefaultWidth2Height = 8;

template <typename T> curandState * CuMatrix<T>::devStates= null;

template<typename T> string pdm(const DMatrix<T>& md);

extern template class CuMatrix<float>;
extern template class CuMatrix<double>;

#endif /* CUMATRIX_H_ */
