/*
 * Kernels.h
 *
 *  Created on: Oct 10, 2013
 *      Author: reid
 */

#pragma once

#include "CuDefs.h"
#include "DMatrix.h"
#include "UnaryOpF_Gen.h"
#include "UnaryOpIndexF_Gen.h"
#include "BinaryOpF_Gen.h"


__global__ void warmup();
__global__ void slep(long slepMs);

template<typename T>__global__ void vectorAdd(T *c, const T *a, const T *b, int n) {
#ifdef __CUDA_ARCH__
	uint idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        idx[c] = idx[a] + idx[b];
/*
        if(idx > n - 5) {
        	printf("idx %u -> c[idx] %f &(c[idx]) %p a[idx] %f b[idx] %f \n", idx, (float)idx[c], c + idx, (float)a[idx], (float)b[idx]);
        }
*/
    }
#endif
}

template <typename T> __global__ void
copyKernel(T* tElements, const T* sElements, uint tWidth, uint tHeight, uint tPitch, uint sWidth, uint sHeight, uint sPitch, uint xOff, uint yOff);

template <typename T> __global__ void copyDmKernelUlong(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) ;
template <typename T> __global__ void copyDmKernelUint(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) ;
template <typename T> __global__ void copyDmKernelUintDvrg(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) ;
template <typename T> __global__ void copyDmKernelIntDvrg(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) ;
template <typename T> __global__ void copyDmRowShuffleKernel(DMatrix<T> trg, const DMatrix<T> src, uint* indices) ;
template <typename T> __global__ void copyDmColShuffleKernel(DMatrix<T> trg, const DMatrix<T> src, uint* indices) ;
//template<typename T> __global__ void fillOpKernel(


/*
template<typename T> __global__ void fill_Kernel(
		T* trg,
		int height,
		int width,
		int pitch,
		bool colMajor);
*/

// set
template <typename T> __global__ void setKernel(T* elements,  int m, int n, int p,  long l, T val);
template <typename T> __host__ CUDART_DEVICE void setL(T* elements, int m, int n, int p, int row, int col, T val);
template <typename T> __host__ CUDART_DEVICE void setL(T* elements, int m, int n, int p, long idx, T val);

// fill
template<typename T> __global__ void fillKernel(T* elements, T val, long n);

// unaryops
#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class UnaryOp> __host__ CUDART_DEVICE void unaryOpL(DMatrix<T>& trg, const DMatrix<T>& src, UnaryOp<T> op, cudaStream_t stream = 0 );
template<typename T, template <typename> class UnaryOp> __host__ CUDART_DEVICE void unaryOpDmL(DMatrix<T>& trg, const DMatrix<T>& src, UnaryOp<T> op, int w2h = DefaultWidth2Height, cudaStream_t stream = 0 ) ;
#else
template<typename T, int StateDim> __host__ CUDART_DEVICE void unaryOpL(DMatrix<T>& trg, const DMatrix<T>& src, UnaryOpF<T,StateDim> op, cudaStream_t stream = 0 );
template<typename T, int StateDim> __host__ CUDART_DEVICE void unaryOpDmL(DMatrix<T>& trg, const DMatrix<T>& src, UnaryOpF<T,StateDim> op, int w2h = DefaultWidth2Height, cudaStream_t stream = 0 ) ;
#endif
//  reductions

#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class BinaryOp>
__global__ void reduceOpKernel( T* g_odata, const T* g_idata, long n,
		BinaryOp<T> op, T start, uint stride = 1, uint offset = 0);
#else
template<typename T, uint blockSize, bool nIsPow2, int StateDim>
__global__ void reduceOpKernel( T* g_odata, const T* g_idata, long n,
		BinaryOpF<T,StateDim> op, T start, uint stride = 1, uint offset = 0);
#endif


// glolo apply unaryop while copying from global to local mem, then reduce with binaryop
#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, typename UnaryOp,
		typename BinaryOp>
__global__ void gloloReduceOpKernel(DMatrix<T> out, const DMatrix<T> src,
		long n, UnaryOp gop, BinaryOp lop, T start);
template<typename T, template <typename> class UnaryOp, template <typename> class BinaryOp>
__host__ CUDART_DEVICE void gloloReduceOpLauncher(T* result,
		DMatrix<T> buff, long n,	const DMatrix<T> src, UnaryOp<T> gop, BinaryOp<T> lop,
		T start, cudaStream_t stream = 0) ;
#else
template<typename T, uint blockSize, bool nIsPow2, int UopDim,
		int BopDim>
__global__ void gloloReduceOpKernel(DMatrix<T> out, const DMatrix<T> src,
		long n, UnaryOpF<T,UopDim> gop, MonoidF<T,BopDim> lop, T start);
template<typename T, int UopDim, int BopDim>
__host__ CUDART_DEVICE void gloloReduceOpLauncher(T* result,
		DMatrix<T> buff, long n,	const DMatrix<T> src, UnaryOpF<T,UopDim> gop, MonoidF<T,BopDim> lop,
		T start, cudaStream_t stream = 0) ;
#endif





#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __global__
void binaryOpKernel1dNeqP(T* trg, const T* src1, const T* src2, BinaryOp<T> op, ulong len);
#else
template<typename T, int StateDim> __global__
void binaryOpKernel1dNeqP(T* trg, const T* src1, const T* src2, BinaryOpF<T,StateDim> op, ulong len);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __global__
void binaryOpKernel1dNeqPRow(T* trg, const T* src1, const T* src2, long n, BinaryOp<T> op,	ulong len);
#else
template<typename T, int StateDim> __global__
void binaryOpKernel1dNeqPRow(T* trg, const T* src1, const T* src2, long n, BinaryOpF<T,StateDim> op,	ulong len);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __global__
void binaryOpKernel2d(T* trg, const T* src1, const T* src2, BinaryOp<T> op,
		ulong len, int p, uint * misses = 0);
#else
template<typename T, int StateDim> __global__
void binaryOpKernel2d(T* trg, const T* src1, const T* src2, BinaryOpF<T,StateDim> op,
		ulong len, int p, uint * misses = 0);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __host__ CUDART_DEVICE
void binaryOpL(	DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOp<T> op, cudaStream_t stream = 0);
#else
template<typename T, int StateDim> __host__ CUDART_DEVICE
void binaryOpL(	DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOpF<T,StateDim> op, cudaStream_t stream = 0);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __host__ CUDART_DEVICE
void binaryOpL2(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOp<T> op, int h2w, uint* misses = 0, cudaStream_t stream = 0);
#else
template<typename T, int StateDim> __host__ CUDART_DEVICE
void binaryOpL2(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOpF<T,StateDim> op, int h2w, uint* misses = 0, cudaStream_t stream = 0);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __global__
void binaryOpDmKernel(DMatrix<T> trg, const DMatrix<T> src1, const DMatrix<T> src2, BinaryOp<T> op );
#else
template<typename T, int StateDim> __global__
void binaryOpDmKernel(DMatrix<T> trg, const DMatrix<T> src1, const DMatrix<T> src2, BinaryOpF<T,StateDim> op );
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp> __host__ CUDART_DEVICE
void binaryOpDmL( DMatrix<T>& trg, const DMatrix<T>& src1,	const DMatrix<T>& src2, BinaryOp<T> op, int w2h = DefaultWidth2Height, cudaStream_t stream = 0);
#else
template<typename T, int StateDim> __host__ CUDART_DEVICE
void binaryOpDmL( DMatrix<T>& trg, const DMatrix<T>& src1,	const DMatrix<T>& src2, BinaryOpF<T,StateDim> op, int w2h = DefaultWidth2Height, cudaStream_t stream = 0);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp>  __host__ CUDART_DEVICE void  reduceLauncher(T* result, DMatrix<T> buff, long nP, const DMatrix<T> src,
		BinaryOp<T> op, T start, int offset = 0, cudaStream_t stream = 0);
#else
template<typename T, int StateDim>  __host__ CUDART_DEVICE void  reduceLauncher(T* result, DMatrix<T> buff, long nP, const DMatrix<T> src,
		MonoidF<T,StateDim> op, T start, int offset = 0, cudaStream_t stream = 0);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp>  __global__ void  reduceLauncherG(T* result, DMatrix<T> buff, long nP, const DMatrix<T> src,
		BinaryOp<T> op, T start, int stride= 1, cudaStream_t stream = 0);
#else
template<typename T, int StateDim>  __global__ void  reduceLauncherG(T* result, DMatrix<T> buff, long nP, const DMatrix<T> src,
		BinaryOpF<T,StateDim> op, T start, int stride= 1, cudaStream_t stream = 0);
#endif

#ifdef CuMatrix_Enable_Cdp
	#ifdef  CuMatrix_Enable_KTS
	template<typename T, template <typename> class BinaryOp>  __global__ void  reduceLauncherCount(T* result, DMatrix<T> buff, long nP, const DMatrix<T> src,
			BinaryOp<T> op, T start, int count);
	#else
	template<typename T, int StateDim>  __global__ void  reduceLauncherCount(T* result, DMatrix<T> buff, long nP, const DMatrix<T> src,
			MonoidF<T,StateDim> op, T start, int count);
	#endif
#endif

// combine combines matrices elementwise to a single matrix applying MatBinaryOp, then reduce that with BinaryOp
#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class BinaryOp1, template <typename> class  BinaryOp2>
__global__ void combineReduceOpKernel(DMatrix<T> d_odata, const DMatrix<T> d_idata1, const DMatrix<T> d_idata2,
		BinaryOp1<T> mop, BinaryOp2<T> bop, T start);
#else
template<typename T, uint blockSize, bool nIsPow2, int MopDim, int BopDim>
__global__ void combineReduceOpKernel(DMatrix<T> d_odata, const DMatrix<T> d_idata1, const DMatrix<T> d_idata2,
		BinaryOpF<T,MopDim> mop, BinaryOpF<T,BopDim> bop, T start);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class MatBinaryOp,
template <typename> class BinaryOp>
__global__ void combineReduceOpKernel(T* g_odata, const T* g_idata1, const T* g_idata2,
		long n, MatBinaryOp<T> mop, BinaryOp<T> op, T start);
#else
template<typename T, uint blockSize, bool nIsPow2, int MopDim, int BopDim>
__global__ void combineReduceOpKernel(T* g_odata, const T* g_idata1, const T* g_idata2,
		long n, BinaryOpF<T,MopDim> mop, BinaryOpF<T,BopDim> op, T start);
#endif

#ifdef  CuMatrix_Enable_KTS
template <typename T, template <typename> class  BinaryOp>
__inline__ __device__ void shreduceShared(T* g_odata, T* sdata, T& myReduction,  uint blockSize, uint tid, const uint blockIdx_x, BinaryOp<T> bop);
#else
template <typename T, int StateDim>
__inline__ __device__ void shreduceShared(T* g_odata, T* sdata, T& myReduction,  uint blockSize, uint tid, const uint blockIdx_x, BinaryOpF<T,StateDim> bop);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class MatBinaryOp,
template <typename> class BinaryOp>
__global__ void combineReduceOpPitchKernel(T* g_odata, const T* g_idata1, const T* g_idata2,
		int spitch1, int spitch2, int rows, int cols, MatBinaryOp<T> mop, BinaryOp<T> op, T start);
#else
template<typename T, uint blockSize, bool nIsPow2, int MopDim, int BopDim>
__global__ void combineReduceOpPitchKernel(T* g_odata, const T* g_idata1, const T* g_idata2,
		int spitch1, int spitch2, int rows, int cols, BinaryOpF<T,MopDim> mop, BinaryOpF<T,BopDim> op, T start);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOp1, template <typename> class BinaryOp2> __host__ CUDART_DEVICE T
	combineReduceOpLauncher(T* g_odata, const T* g_idata1, const T* g_idata2, int spitch1, int spitch2, int rows, int cols,  BinaryOp1<T> mop, BinaryOp2<T> bop, T start, cudaStream_t stream = 0);
#else
template<typename T, int MopDim, int BopDim> __host__ CUDART_DEVICE T
	combineReduceOpLauncher(T* g_odata, const T* g_idata1, const T* g_idata2, int spitch1, int spitch2, int rows, int cols,  BinaryOpF<T,MopDim> mop, BinaryOpF<T,BopDim> bop, T start, cudaStream_t stream = 0);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class MatBinaryOp,
template <typename> class BinaryOp>
__global__ void combineReduceOpKernel2(const T* g_idata1, const T* g_idata2,
		T* g_odata, long n, MatBinaryOp<T> mop, BinaryOp<T> op, T start);
#else
template<typename T, uint blockSize, bool nIsPow2, int MopDim, int BopDim>
__global__ void combineReduceOpKernel2(const T* g_idata1, const T* g_idata2,
		T* g_odata, long n, BinaryOpF<T,MopDim> mop, BinaryOpF<T,BopDim> op, T start);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, uint blockSize, bool nIsPow2, template <typename> class BinaryOp>
__global__ void reduceOpKernel2( T* g_odata, const T* g_idata, long n,
		BinaryOp<T> op, T start);
#else
template<typename T, uint blockSize, bool nIsPow2, int StateDim>
__global__ void reduceOpKernel2( T* g_odata, const T* g_idata, long n,
		BinaryOpF<T,StateDim> op, T start);
#endif

#ifdef CuMatrix_Enable_Cdp
template<typename T> __global__ void matrixProductReductionTxdBKernel(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int stepsDummy);
#endif

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class BinaryOpF> __global__ void rowReductionKernelNlte64(DMatrix<T> resVec, BinaryOpF<T> bop, const DMatrix<T> x, uint slice, uint slices);
template<typename T, template <typename> class BinaryOpF> __global__ void rowReductionKernel(DMatrix<T> resVec, BinaryOpF<T> bop, const DMatrix<T> x, uint slice, uint slices, uint stripes);
#else
template<typename T, int StateDim> __global__ void rowReductionKernelNlte64(DMatrix<T> resVec, MonoidF<T,StateDim> bop, const DMatrix<T> x, uint slice, uint slices);
template<typename T, int StateDim> __global__ void rowReductionKernel(DMatrix<T> resVec, MonoidF<T,StateDim> bop, const DMatrix<T> x, uint slice, uint slices, uint stripes);
#endif

// matrix product
extern dim3 gDefaultMatProdBlock;
extern __constant__ uint dDefaultMatProdBlockX;
extern __constant__ uint dDefaultMatProdBlockY;
extern __constant__ uint3 dgDefaultMatProdBlock;

void setAllGpuConstants();
void setCurrGpuConstants();

/*
 * matrix product kernels
 */

template<typename T> __global__ void matrixProductBandwidthKernel(DMatrix<T> C,const DMatrix<T> A, const DMatrix<T> B, int steps);
template<typename T> __global__ void matrixProductKernel(DMatrix<T> C,const DMatrix<T> A, const DMatrix<T> B, int steps);
template<typename T> __global__ void matrixProductKernel2(DMatrix<T> C,const DMatrix<T> A, const DMatrix<T> B, int steps);
template<typename T> __global__ void matrixProductKernel3(DMatrix<T> C,const DMatrix<T> A, const DMatrix<T> B, int steps);
template<typename T> __global__ void matrixProductKernel4(DMatrix<T> C,const DMatrix<T> A, const DMatrix<T> B, int steps);

template<typename T> __global__ void matrixProductKernelTxdB(DMatrix<T> C,const DMatrix<T> A, const DMatrix<T> B, int steps);
template<typename T> __global__ void matrixProductKernelTxdB2(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int steps);
template<typename T> __global__ void matrixProductKernelTxdB2(DMatrix<T> C,const DMatrix<T> A, const DMatrix<T> B, int steps);
template<typename T> __global__ void matrixProductReductionTxdBKernel(DMatrix<T> C,
		const DMatrix<T> A, const DMatrix<T> B, int stepsDummy);

template<typename T> __host__ CUDART_DEVICE
 void matrixProductL(DMatrix<T>& d_res,
		const DMatrix<T>& d_A, const DMatrix<T>& d_B, dim3* block,cudaStream_t stream = 0);

template <typename T> __host__ __device__ const char* matProdKernelName(void (*matProdKptr) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int));

template<typename T> __host__ CUDART_DEVICE void matrixProductKPtrL(DMatrix<T>& d_res,
		void (*matProdKptr) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int),
		const DMatrix<T>& d_A, const DMatrix<T>& d_B,
		 dim3* block,cudaStream_t stream = 0);



// filluers

template<typename T> void kayrnlL();

template<typename T, int StateDim> __global__ void fillOpKernel( UnaryOpIndexF<T,StateDim> op,
		T* trg,
		int height,
		int width,
		int pitch,
		uint rowStart,
		uint colStart,
		bool colMajor);


template<typename T, int StateDim> __global__
void fillOpNsbKernel( UnaryOpIndexF<T,StateDim> op,
		T* trg,
		int height,
		int width,
		int pitch,
		bool colMajor);

// rand
__global__ void setup_kernel ( curandState * state, int width, int pitch );

template <typename T> __global__ void generate_kernel ( curandState * state,
                                      T*  result, int height, int width );
template <typename T>  __global__ void generate_uniform_kernel ( curandState * state,
		T*  result, T epsilon, int height, int width, int pitch );
template <typename T>  __global__ void generate_uniform_kernel_mod( curandState * state,
		T*  result, T epsilon, int height, int width, int pitch,int mod );

template <typename T> T eXL(T x, ulong places);

template<typename T> __global__ void attributeCountsKernelShared(DMatrix<T> d_atts,
		DMatrix<int> d_counts, DMatrix<int> d_depths, DMatrix<int> d_distances,
		const DMatrix<T> d_x);
template<typename T> void shufftestLauncher();
template<typename T> void  shuffleSortTestLauncher();
