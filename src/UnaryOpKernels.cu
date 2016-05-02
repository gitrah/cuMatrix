/*
 * UnaryOpKernels.cu
 *
 *  Created on: Oct 19, 2013
 *      Author: reid
 */



#include "util.h"
#include "caps.h"
#include "Kernels.h"
#include "CuFunctor.h"
#include "UnaryOpF_Gen.h"
#include <assert.h>

template<typename T, typename UnaryOp> __global__ void unaryOp1dKernel(
		T* trg, const T* src, UnaryOp op, ulong len) {
	ulong i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		trg[i] = op(src[i]);
	}
}

template<typename T, typename UnaryOp> __global__ void unaryOpDmKernel(
		DMatrix<T> trg, const DMatrix<T> src, UnaryOp op ) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.x + threadIdx.y;
	uint srcOff = y * src.p + x;
	uint trgOff = y * trg.p + x;
	for(int i = 0; i < blockDim.x; i+=blockDim.y) {
		if(x < src.n && y + i < src.m) {
			trg.elements[trgOff + i * trg.p] = op(src.elements[srcOff + i * src.p]);
		}
	}
}

#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class UnaryOp> __host__ CUDART_DEVICE void unaryOpL(DMatrix<T>& trg, const DMatrix<T>& src, UnaryOp<T> op, cudaStream_t stream )
#else
template<typename T, int StateDim> __host__ CUDART_DEVICE void unaryOpL(DMatrix<T>& trg, const DMatrix<T>& src, UnaryOpF<T,StateDim> op, cudaStream_t stream )
#endif
{
	int threads = 512;
	uint len = src.m * src.n;
	dim3 dBlocks, dThreads;
	b_util::vectorExecContext(threads, len, dBlocks, dThreads);
	unaryOp1dKernel<<<dBlocks,dThreads,0,stream>>>(trg.elements, src.elements, op, len);
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void unaryOpL<float, approxInvSqrtUnaryOp>(DMatrix<float>&, DMatrix<float> const&, approxInvSqrtUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double, approxInvSqrtUnaryOp>(DMatrix<double>&, DMatrix<double> const&, approxInvSqrtUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<float, slowInvSqrtUnaryOp>(DMatrix<float>&, DMatrix<float> const&, slowInvSqrtUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double, slowInvSqrtUnaryOp>(DMatrix<double>&, DMatrix<double> const&, slowInvSqrtUnaryOp<double>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float, floorUnaryOp>(DMatrix<float>&, DMatrix<float> const&, floorUnaryOp<float>, CUstream_st*);

template __host__ CUDART_DEVICE void unaryOpL<unsigned int, absUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, absUnaryOp<unsigned int>, CUstream_st*);

template __host__ CUDART_DEVICE void unaryOpL<double, ceilUnaryOp>(DMatrix<double>&, DMatrix<double> const&, ceilUnaryOp<double>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, powUnaryOp>(DMatrix<long>&, DMatrix<long> const&, powUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, sigmoidGradientUnaryOp>(DMatrix<long>&, DMatrix<long> const&, sigmoidGradientUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, ceilUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, ceilUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, oneOverUnaryOp>(DMatrix<long>&, DMatrix<long> const&, oneOverUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, ltUnaryOp>(DMatrix<long>&, DMatrix<long> const&, ltUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, ceilUnaryOp>(DMatrix<long>&, DMatrix<long> const&, ceilUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, sqrtUnaryOp>(DMatrix<long>&, DMatrix<long> const&, sqrtUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned long, floorUnaryOp>(DMatrix<unsigned long>&, DMatrix<unsigned long> const&, floorUnaryOp<unsigned long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, gtUnaryOp>(DMatrix<long>&, DMatrix<long> const&, gtUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, ceilUnaryOp>(DMatrix<int>&, DMatrix<int> const&, ceilUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, negateUnaryOp>(DMatrix<long>&, DMatrix<long> const&, negateUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<double, floorUnaryOp>(DMatrix<double>&, DMatrix<double> const&, floorUnaryOp<double>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, divSqrtUnaryOp>(DMatrix<long>&, DMatrix<long> const&, divSqrtUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, logUnaryOp>(DMatrix<long>&, DMatrix<long> const&, logUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, gteUnaryOp>(DMatrix<long>&, DMatrix<long> const&, gteUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, floorUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, floorUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned long, ceilUnaryOp>(DMatrix<unsigned long>&, DMatrix<unsigned long> const&, ceilUnaryOp<unsigned long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, eqUnaryOp>(DMatrix<long>&, DMatrix<long> const&, eqUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, floorUnaryOp>(DMatrix<long>&, DMatrix<long> const&, floorUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, lteUnaryOp>(DMatrix<long>&, DMatrix<long> const&, lteUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, sqrUnaryOp>(DMatrix<long>&, DMatrix<long> const&, sqrUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<float, ceilUnaryOp>(DMatrix<float>&, DMatrix<float> const&, ceilUnaryOp<float>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, floorUnaryOp>(DMatrix<int>&, DMatrix<int> const&, floorUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, sigmoidUnaryOp>(DMatrix<long>&, DMatrix<long> const&, sigmoidUnaryOp<long>, CUstream_st*);

template __host__ CUDART_DEVICE void unaryOpL<unsigned long, absUnaryOp>(DMatrix<unsigned long>&, DMatrix<unsigned long> const&, absUnaryOp<unsigned long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, absUnaryOp>(DMatrix<int>&, DMatrix<int> const&, absUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<double, absUnaryOp>(DMatrix<double>&, DMatrix<double> const&, absUnaryOp<double>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<float, absUnaryOp>(DMatrix<float>&, DMatrix<float> const&, absUnaryOp<float>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<long, absUnaryOp>(DMatrix<long>&, DMatrix<long> const&, absUnaryOp<long>, CUstream_st*);

template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, absUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, absUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned long, absUnaryOp>(DMatrix<unsigned long>&, DMatrix<unsigned long> const&, absUnaryOp<unsigned long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, absUnaryOp>(DMatrix<int>&, DMatrix<int> const&, absUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<double, absUnaryOp>(DMatrix<double>&, DMatrix<double> const&, absUnaryOp<double>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<float, absUnaryOp>(DMatrix<float>&, DMatrix<float> const&, absUnaryOp<float>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, absUnaryOp>(DMatrix<long>&, DMatrix<long> const&, absUnaryOp<long>, int, CUstream_st*);

template __host__ CUDART_DEVICE void unaryOpDmL<float, floorUnaryOp>(DMatrix<float>&, DMatrix<float> const&, floorUnaryOp<float>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<double, ceilUnaryOp>(DMatrix<double>&, DMatrix<double> const&, ceilUnaryOp<double>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, powUnaryOp>(DMatrix<long>&, DMatrix<long> const&, powUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, sigmoidGradientUnaryOp>(DMatrix<long>&, DMatrix<long> const&, sigmoidGradientUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, scaleUnaryOp>(DMatrix<long>&, DMatrix<long> const&, scaleUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, ceilUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, ceilUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, oneOverUnaryOp>(DMatrix<long>&, DMatrix<long> const&, oneOverUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, ltUnaryOp>(DMatrix<long>&, DMatrix<long> const&, ltUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, ceilUnaryOp>(DMatrix<long>&, DMatrix<long> const&, ceilUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, translationUnaryOp>(DMatrix<long>&, DMatrix<long> const&, translationUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, sqrtUnaryOp>(DMatrix<long>&, DMatrix<long> const&, sqrtUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned long, floorUnaryOp>(DMatrix<unsigned long>&, DMatrix<unsigned long> const&, floorUnaryOp<unsigned long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, gtUnaryOp>(DMatrix<long>&, DMatrix<long> const&, gtUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, ceilUnaryOp>(DMatrix<int>&, DMatrix<int> const&, ceilUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, negateUnaryOp>(DMatrix<long>&, DMatrix<long> const&, negateUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<double, floorUnaryOp>(DMatrix<double>&, DMatrix<double> const&, floorUnaryOp<double>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, divSqrtUnaryOp>(DMatrix<long>&, DMatrix<long> const&, divSqrtUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, logUnaryOp>(DMatrix<long>&, DMatrix<long> const&, logUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, gteUnaryOp>(DMatrix<long>&, DMatrix<long> const&, gteUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, floorUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, floorUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, expUnaryOp>(DMatrix<long>&, DMatrix<long> const&, expUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned long, ceilUnaryOp>(DMatrix<unsigned long>&, DMatrix<unsigned long> const&, ceilUnaryOp<unsigned long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, eqUnaryOp>(DMatrix<long>&, DMatrix<long> const&, eqUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, floorUnaryOp>(DMatrix<long>&, DMatrix<long> const&, floorUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, lteUnaryOp>(DMatrix<long>&, DMatrix<long> const&, lteUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, sqrUnaryOp>(DMatrix<long>&, DMatrix<long> const&, sqrUnaryOp<long>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<float, ceilUnaryOp>(DMatrix<float>&, DMatrix<float> const&, ceilUnaryOp<float>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, floorUnaryOp>(DMatrix<int>&, DMatrix<int> const&, floorUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, sigmoidUnaryOp>(DMatrix<long>&, DMatrix<long> const&, sigmoidUnaryOp<long>, int, CUstream_st*);

template __host__ CUDART_DEVICE void unaryOpL<float,expUnaryOp>(DMatrix<float>&, const DMatrix<float>&, expUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,expUnaryOp>(DMatrix<double>&, const DMatrix<double>&, expUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<long,expUnaryOp>(DMatrix<long>&, const DMatrix<long>&, expUnaryOp<long>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,expUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, expUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,translationUnaryOp>(DMatrix<float>&, const DMatrix<float>&, translationUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,translationUnaryOp>(DMatrix<double>&, const DMatrix<double>&, translationUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<long,translationUnaryOp>(DMatrix<long>&, const DMatrix<long>&, translationUnaryOp<long>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,translationUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, translationUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,scaleUnaryOp>(DMatrix<float>&, const DMatrix<float>&, scaleUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,scaleUnaryOp>(DMatrix<double>&, const DMatrix<double>&, scaleUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<long,scaleUnaryOp>(DMatrix<long>&, const DMatrix<long>&, scaleUnaryOp<long>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,scaleUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, scaleUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,subFromUnaryOp>(DMatrix<float>&, const DMatrix<float>&, subFromUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,subFromUnaryOp>(DMatrix<double>&, const DMatrix<double>&, subFromUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,subFromUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, subFromUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,negateUnaryOp>(DMatrix<float>&, const DMatrix<float>&, negateUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,negateUnaryOp>(DMatrix<double>&, const DMatrix<double>&, negateUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,negateUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, negateUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,sigmoidUnaryOp>(DMatrix<float>&, const DMatrix<float>&, sigmoidUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,sigmoidUnaryOp>(DMatrix<double>&, const DMatrix<double>&, sigmoidUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,sigmoidUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, sigmoidUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,sigmoidGradientUnaryOp>(DMatrix<float>&, const DMatrix<float>&, sigmoidGradientUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,sigmoidGradientUnaryOp>(DMatrix<double>&, const DMatrix<double>&, sigmoidGradientUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,sigmoidGradientUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, sigmoidGradientUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,logUnaryOp>(DMatrix<float>&, const DMatrix<float>&, logUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,logUnaryOp>(DMatrix<double>&, const DMatrix<double>&, logUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,logUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, logUnaryOp<ulong>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<float,oneOverUnaryOp>(DMatrix<float>&, const DMatrix<float>&, oneOverUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,oneOverUnaryOp>(DMatrix<double>&, const DMatrix<double>&, oneOverUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,oneOverUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, oneOverUnaryOp<ulong>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<float,sqrtUnaryOp>(DMatrix<float>&, const DMatrix<float>&, sqrtUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,sqrtUnaryOp>(DMatrix<double>&, const DMatrix<double>&, sqrtUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,sqrtUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, sqrtUnaryOp<ulong>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<float,sqrUnaryOp>(DMatrix<float>&, const DMatrix<float>&, sqrUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,sqrUnaryOp>(DMatrix<double>&, const DMatrix<double>&, sqrUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,sqrUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, sqrUnaryOp<ulong>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<float,powUnaryOp>(DMatrix<float>&, const DMatrix<float>&, powUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,powUnaryOp>(DMatrix<double>&, const DMatrix<double>&, powUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,powUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, powUnaryOp<ulong>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<float,divSqrtUnaryOp>(DMatrix<float>&, const DMatrix<float>&, divSqrtUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,divSqrtUnaryOp>(DMatrix<double>&, const DMatrix<double>&, divSqrtUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,divSqrtUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, divSqrtUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,ltUnaryOp>(DMatrix<float>&, const DMatrix<float>&, ltUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,ltUnaryOp>(DMatrix<double>&, const DMatrix<double>&, ltUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,ltUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, ltUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,lteUnaryOp>(DMatrix<float>&, const DMatrix<float>&, lteUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,lteUnaryOp>(DMatrix<double>&, const DMatrix<double>&, lteUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,lteUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, lteUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,gtUnaryOp>(DMatrix<float>&, const DMatrix<float>&, gtUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,gtUnaryOp>(DMatrix<double>&, const DMatrix<double>&, gtUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,gtUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, gtUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,gteUnaryOp>(DMatrix<float>&, const DMatrix<float>&, gteUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,gteUnaryOp>(DMatrix<double>&, const DMatrix<double>&, gteUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,gteUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, gteUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,eqUnaryOp>(DMatrix<float>&, const DMatrix<float>&, eqUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,eqUnaryOp>(DMatrix<double>&, const DMatrix<double>&, eqUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,eqUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, eqUnaryOp<ulong>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,notAlmostEqUnaryOp>(DMatrix<float>&, const DMatrix<float>&, notAlmostEqUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,notAlmostEqUnaryOp>(DMatrix<double>&, const DMatrix<double>&, notAlmostEqUnaryOp<double>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<float,neqUnaryOp>(DMatrix<float>&, const DMatrix<float>&, neqUnaryOp<float>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,neqUnaryOp>(DMatrix<double>&, const DMatrix<double>&, neqUnaryOp<double>, CUstream_st *);


template __host__ CUDART_DEVICE void unaryOpL<int, negateUnaryOp>(DMatrix<int>&, DMatrix<int> const&, negateUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, sigmoidUnaryOp>(DMatrix<int>&, DMatrix<int> const&, sigmoidUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, sigmoidGradientUnaryOp>(DMatrix<int>&, DMatrix<int> const&, sigmoidGradientUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, logUnaryOp>(DMatrix<int>&, DMatrix<int> const&, logUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, oneOverUnaryOp>(DMatrix<int>&, DMatrix<int> const&, oneOverUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, expUnaryOp>(DMatrix<int>&, DMatrix<int> const&, expUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, sqrtUnaryOp>(DMatrix<int>&, DMatrix<int> const&, sqrtUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, sqrUnaryOp>(DMatrix<int>&, DMatrix<int> const&, sqrUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, powUnaryOp>(DMatrix<int>&, DMatrix<int> const&, powUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, divSqrtUnaryOp>(DMatrix<int>&, DMatrix<int> const&, divSqrtUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, negateUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, negateUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, sigmoidUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, sigmoidUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, sigmoidGradientUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, sigmoidGradientUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, logUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, logUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, oneOverUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, oneOverUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, expUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, expUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, sqrtUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, sqrtUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, sqrUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, sqrUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, powUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, powUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, divSqrtUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, divSqrtUnaryOp<unsigned int>, CUstream_st*);

template __host__ CUDART_DEVICE void unaryOpL<int, scaleUnaryOp>(DMatrix<int>&, DMatrix<int> const&, scaleUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, scaleUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, scaleUnaryOp<unsigned int>, CUstream_st*);

template __host__ CUDART_DEVICE void unaryOpL<int, subFromUnaryOp>(DMatrix<int>&, DMatrix<int> const&, subFromUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, ltUnaryOp>(DMatrix<int>&, DMatrix<int> const&, ltUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, lteUnaryOp>(DMatrix<int>&, DMatrix<int> const&, lteUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, gtUnaryOp>(DMatrix<int>&, DMatrix<int> const&, gtUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, gteUnaryOp>(DMatrix<int>&, DMatrix<int> const&, gteUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, eqUnaryOp>(DMatrix<int>&, DMatrix<int> const&, eqUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<int, translationUnaryOp>(DMatrix<int>&, DMatrix<int> const&, translationUnaryOp<int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, subFromUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, subFromUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, ltUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, ltUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, lteUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, lteUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, gtUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, gtUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, gteUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, gteUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, eqUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, eqUnaryOp<unsigned int>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpL<unsigned int, translationUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, translationUnaryOp<unsigned int>, CUstream_st*);

template __host__ CUDART_DEVICE void unaryOpL<long, subFromUnaryOp>(DMatrix<long>&, DMatrix<long> const&, subFromUnaryOp<long>, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<long, subFromUnaryOp>(DMatrix<long>&, DMatrix<long> const&, subFromUnaryOp<long>, int, CUstream_st*);


#else
template __host__ CUDART_DEVICE void unaryOpL<float,0>(DMatrix<float>&, const DMatrix<float>&, UnaryOpF<float,0>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,0>(DMatrix<double>&, const DMatrix<double>&, UnaryOpF<double,0>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<long,0>(DMatrix<long>&, const DMatrix<long>&, UnaryOpF<long,0>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,0>(DMatrix<ulong>&, const DMatrix<ulong>&, UnaryOpF<ulong,0>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<int,0>(DMatrix<int>&, const DMatrix<int>&, UnaryOpF<int,0>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<uint,0>(DMatrix<uint>&, const DMatrix<uint>&, UnaryOpF<uint,0>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,1>(DMatrix<float>&, const DMatrix<float>&, UnaryOpF<float,1>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,1>(DMatrix<double>&, const DMatrix<double>&, UnaryOpF<double,1>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<long,1>(DMatrix<long>&, const DMatrix<long>&, UnaryOpF<long,1>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,1>(DMatrix<ulong>&, const DMatrix<ulong>&, UnaryOpF<ulong,1>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<int,1>(DMatrix<int>&, const DMatrix<int>&, UnaryOpF<int,1>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<uint,1>(DMatrix<uint>&, const DMatrix<uint>&, UnaryOpF<uint,1>, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpL<float,2>(DMatrix<float>&, const DMatrix<float>&, UnaryOpF<float,2>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<double,2>(DMatrix<double>&, const DMatrix<double>&, UnaryOpF<double,2>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<long,2>(DMatrix<long>&, const DMatrix<long>&, UnaryOpF<long,2>, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpL<ulong,2>(DMatrix<ulong>&, const DMatrix<ulong>&, UnaryOpF<ulong,2>, CUstream_st *);
#endif


#ifdef  CuMatrix_Enable_KTS
template<typename T, template <typename> class UnaryOp> __host__ CUDART_DEVICE void unaryOpDmL(DMatrix<T>& trg, const DMatrix<T>& src, UnaryOp<T> op, int w2h, cudaStream_t stream )
#else
template<typename T, int StateDim> __host__ CUDART_DEVICE void unaryOpDmL(DMatrix<T>& trg, const DMatrix<T>& src, UnaryOpF<T,StateDim> op, int w2h, cudaStream_t stream )
#endif
{
	assert( trg.m >= src.m && trg.n >= src.n );
	int blockW = UNOP_BLOCK_SIZE;
	dim3 block(blockW,blockW/w2h);
    dim3 grid(DIV_UP(src.n,blockW), DIV_UP(src.m,blockW));
    if(checkDebug(debugExec)) { printf("unaryOpDmL grid "); b_util::prd3(grid); printf(" of block " );  b_util::prd3(block);}
    unaryOpDmKernel<<<grid,block,0,stream>>>(trg, src, op);
}
#ifdef  CuMatrix_Enable_KTS
template __host__ CUDART_DEVICE void unaryOpDmL<float, approxInvSqrtUnaryOp>(DMatrix<float>&, DMatrix<float> const&, approxInvSqrtUnaryOp<float>, int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double, approxInvSqrtUnaryOp>(DMatrix<double>&, DMatrix<double> const&, approxInvSqrtUnaryOp<double>, int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<float, slowInvSqrtUnaryOp>(DMatrix<float>&, DMatrix<float> const&, slowInvSqrtUnaryOp<float>, int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double, slowInvSqrtUnaryOp>(DMatrix<double>&, DMatrix<double> const&, slowInvSqrtUnaryOp<double>, int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,expUnaryOp>(DMatrix<float>&, const DMatrix<float>&, expUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,expUnaryOp>(DMatrix<double>&, const DMatrix<double>&, expUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,expUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, expUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,translationUnaryOp>(DMatrix<float>&, const DMatrix<float>&, translationUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,translationUnaryOp>(DMatrix<double>&, const DMatrix<double>&, translationUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,translationUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, translationUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,scaleUnaryOp>(DMatrix<float>&, const DMatrix<float>&, scaleUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,scaleUnaryOp>(DMatrix<double>&, const DMatrix<double>&, scaleUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,scaleUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, scaleUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,subFromUnaryOp>(DMatrix<float>&, const DMatrix<float>&, subFromUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,subFromUnaryOp>(DMatrix<double>&, const DMatrix<double>&, subFromUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,subFromUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, subFromUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,negateUnaryOp>(DMatrix<float>&, const DMatrix<float>&, negateUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,negateUnaryOp>(DMatrix<double>&, const DMatrix<double>&, negateUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,negateUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, negateUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,sigmoidUnaryOp>(DMatrix<float>&, const DMatrix<float>&, sigmoidUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,sigmoidUnaryOp>(DMatrix<double>&, const DMatrix<double>&, sigmoidUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,sigmoidUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, sigmoidUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,sigmoidGradientUnaryOp>(DMatrix<float>&, const DMatrix<float>&, sigmoidGradientUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,sigmoidGradientUnaryOp>(DMatrix<double>&, const DMatrix<double>&, sigmoidGradientUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,sigmoidGradientUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, sigmoidGradientUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,logUnaryOp>(DMatrix<float>&, const DMatrix<float>&, logUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,logUnaryOp>(DMatrix<double>&, const DMatrix<double>&, logUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,logUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, logUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,oneOverUnaryOp>(DMatrix<float>&, const DMatrix<float>&, oneOverUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,oneOverUnaryOp>(DMatrix<double>&, const DMatrix<double>&, oneOverUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,oneOverUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, oneOverUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,sqrtUnaryOp>(DMatrix<float>&, const DMatrix<float>&, sqrtUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,sqrtUnaryOp>(DMatrix<double>&, const DMatrix<double>&, sqrtUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,sqrtUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, sqrtUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,sqrUnaryOp>(DMatrix<float>&, const DMatrix<float>&, sqrUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,sqrUnaryOp>(DMatrix<double>&, const DMatrix<double>&, sqrUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,sqrUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, sqrUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,powUnaryOp>(DMatrix<float>&, const DMatrix<float>&, powUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,powUnaryOp>(DMatrix<double>&, const DMatrix<double>&, powUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,powUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, powUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,divSqrtUnaryOp>(DMatrix<float>&, const DMatrix<float>&, divSqrtUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,divSqrtUnaryOp>(DMatrix<double>&, const DMatrix<double>&, divSqrtUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,divSqrtUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, divSqrtUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,ltUnaryOp>(DMatrix<float>&, const DMatrix<float>&, ltUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,ltUnaryOp>(DMatrix<double>&, const DMatrix<double>&, ltUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,ltUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, ltUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,lteUnaryOp>(DMatrix<float>&, const DMatrix<float>&, lteUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,lteUnaryOp>(DMatrix<double>&, const DMatrix<double>&, lteUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,lteUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, lteUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,gtUnaryOp>(DMatrix<float>&, const DMatrix<float>&, gtUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,gtUnaryOp>(DMatrix<double>&, const DMatrix<double>&, gtUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,gtUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, gtUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,gteUnaryOp>(DMatrix<float>&, const DMatrix<float>&, gteUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,gteUnaryOp>(DMatrix<double>&, const DMatrix<double>&, gteUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,gteUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, gteUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,eqUnaryOp>(DMatrix<float>&, const DMatrix<float>&, eqUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,eqUnaryOp>(DMatrix<double>&, const DMatrix<double>&, eqUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,eqUnaryOp>(DMatrix<ulong>&, const DMatrix<ulong>&, eqUnaryOp<ulong>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,notAlmostEqUnaryOp>(DMatrix<float>&, const DMatrix<float>&, notAlmostEqUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,notAlmostEqUnaryOp>(DMatrix<double>&, const DMatrix<double>&, notAlmostEqUnaryOp<double>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<float,neqUnaryOp>(DMatrix<float>&, const DMatrix<float>&, neqUnaryOp<float>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,neqUnaryOp>(DMatrix<double>&, const DMatrix<double>&, neqUnaryOp<double>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<int, negateUnaryOp>(DMatrix<int>&, DMatrix<int> const&, negateUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, sigmoidUnaryOp>(DMatrix<int>&, DMatrix<int> const&, sigmoidUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, sigmoidGradientUnaryOp>(DMatrix<int>&, DMatrix<int> const&, sigmoidGradientUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, logUnaryOp>(DMatrix<int>&, DMatrix<int> const&, logUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, oneOverUnaryOp>(DMatrix<int>&, DMatrix<int> const&, oneOverUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, expUnaryOp>(DMatrix<int>&, DMatrix<int> const&, expUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, sqrtUnaryOp>(DMatrix<int>&, DMatrix<int> const&, sqrtUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, sqrUnaryOp>(DMatrix<int>&, DMatrix<int> const&, sqrUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, powUnaryOp>(DMatrix<int>&, DMatrix<int> const&, powUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, divSqrtUnaryOp>(DMatrix<int>&, DMatrix<int> const&, divSqrtUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, negateUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, negateUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, sigmoidUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, sigmoidUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, sigmoidGradientUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, sigmoidGradientUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, logUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, logUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, oneOverUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, oneOverUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, expUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, expUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, sqrtUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, sqrtUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, sqrUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, sqrUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, powUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, powUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, divSqrtUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, divSqrtUnaryOp<unsigned int>, int, CUstream_st*);


template __host__ CUDART_DEVICE void unaryOpDmL<int, subFromUnaryOp>(DMatrix<int>&, DMatrix<int> const&, subFromUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, ltUnaryOp>(DMatrix<int>&, DMatrix<int> const&, ltUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, lteUnaryOp>(DMatrix<int>&, DMatrix<int> const&, lteUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, gtUnaryOp>(DMatrix<int>&, DMatrix<int> const&, gtUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, gteUnaryOp>(DMatrix<int>&, DMatrix<int> const&, gteUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, eqUnaryOp>(DMatrix<int>&, DMatrix<int> const&, eqUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, translationUnaryOp>(DMatrix<int>&, DMatrix<int> const&, translationUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<int, scaleUnaryOp>(DMatrix<int>&, DMatrix<int> const&, scaleUnaryOp<int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, subFromUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, subFromUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, ltUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, ltUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, lteUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, lteUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, gtUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, gtUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, gteUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, gteUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, eqUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, eqUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, translationUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, translationUnaryOp<unsigned int>, int, CUstream_st*);
template __host__ CUDART_DEVICE void unaryOpDmL<unsigned int, scaleUnaryOp>(DMatrix<unsigned int>&, DMatrix<unsigned int> const&, scaleUnaryOp<unsigned int>, int, CUstream_st*);

#else
template __host__ CUDART_DEVICE void unaryOpDmL<float,0>(DMatrix<float>&, const DMatrix<float>&, UnaryOpF<float,0>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,0>(DMatrix<double>&, const DMatrix<double>&, UnaryOpF<double,0>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<long,0>(DMatrix<long>&, const DMatrix<long>&, UnaryOpF<long,0>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,0>(DMatrix<ulong>&, const DMatrix<ulong>&, UnaryOpF<ulong,0>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<int,0>(DMatrix<int>&, const DMatrix<int>&, UnaryOpF<int,0>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<uint,0>(DMatrix<uint>&, const DMatrix<uint>&, UnaryOpF<uint,0>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,1>(DMatrix<float>&, const DMatrix<float>&, UnaryOpF<float,1>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,1>(DMatrix<double>&, const DMatrix<double>&, UnaryOpF<double,1>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<long,1>(DMatrix<long>&, const DMatrix<long>&, UnaryOpF<long,1>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,1>(DMatrix<ulong>&, const DMatrix<ulong>&, UnaryOpF<ulong,1>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<int,1>(DMatrix<int>&, const DMatrix<int>&, UnaryOpF<int,1>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<uint,1>(DMatrix<uint>&, const DMatrix<uint>&, UnaryOpF<uint,1>,int, CUstream_st *);

template __host__ CUDART_DEVICE void unaryOpDmL<float,2>(DMatrix<float>&, const DMatrix<float>&, UnaryOpF<float,2>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<double,2>(DMatrix<double>&, const DMatrix<double>&, UnaryOpF<double,2>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<long,2>(DMatrix<long>&, const DMatrix<long>&, UnaryOpF<long,2>,int, CUstream_st *);
template __host__ CUDART_DEVICE void unaryOpDmL<ulong,2>(DMatrix<ulong>&, const DMatrix<ulong>&, UnaryOpF<ulong,2>,int, CUstream_st *);
#endif
