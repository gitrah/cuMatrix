/*
 * CuDefs.cu
 *
 *  Created on: Oct 19, 2013
 *      Author: reid
 */

#include "CuDefs.h"
#include "DMatrix.h"
#include "util.h"
#include "BinaryOpF_Gen.h"

// using __shflX in functions templated by functor types is tedious because, as of cuda5.5,
// function templates can't be partially specialized (precludes partially specializing just on the data type)
// member functions can't be __global__ (precludes using functors (param'd by data type with operator()s templated by functors) for kernels)
// this is the only other way I could think of; it would be nice instead if there were templated _shfls
template <typename T> __device__ T shfl(T& v, int lane, int width) {
	return __shfl(v,lane, width);
}
template __device__ float shfl<float>(float&, int, int);
template __device__ int shfl<int>(int&, int, int);
template __device__ uint shfl<uint>(uint&, int, int);
template <> __device__ double shfl(double& v, int lane, int width) {
	float half1 = *((float*)&v);
	float half2 = *(((float*)&v) + 1);
	half1 = __shfl(half1,lane, width);
	half2 = __shfl(half2,lane, width);
	double lv;
	*((float*)&lv) = half1;
	*(((float*)&lv) + 1) = half2;
	return lv;
/*
	int lo, hi;
	asm volatile( "mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(v));
	lo = __shfl(lo, lane, width);
	hi = __shfl(hi, lane, width);
	asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(v) : "r"(lo), "r"(hi));
	return v;
*/
}
template <> __device__ long shfl(long& v, int lane, int width) {
/*
	int lo, hi;
	asm volatile( "mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(v));
	lo = __shfl(lo, lane, width);
	hi = __shfl(hi, lane, width);
	asm volatile( "mov.b64 %0, {%1,%2};" : "=l"(v) : "r"(lo), "r"(hi));
*/
	int half1 = *((int*)&v);
	int half2 = *(((int*)&v) + 1);
	half1 = __shfl(half1,lane, width);
	half2 = __shfl(half2,lane, width);
	long lv;
	*((int*)&lv) = half1;
	*(((int*)&lv) + 1) = half2;
	return lv;
}
template <> __device__ ulong shfl(ulong& v, int lane, int width) {
/*
	int lo, hi;
	asm volatile( "mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(v));
	lo = __shfl(lo, lane, width);
	hi = __shfl(hi, lane, width);
	asm volatile( "mov.b64 %0, {%1,%2};" : "=l"(v) : "r"(lo), "r"(hi));
*/
	uint half1 = *((uint*)&v);
	uint half2 = *(((uint*)&v) + 1);
	half1 = __shfl((int)half1,lane, width);
	half2 = __shfl((int)half2,lane, width);
	ulong lv;
	*((uint*)&lv) = half1;
	*(((uint*)&lv) + 1) = half2;
	return lv;
}
/*
template __device__ double shfl<double>(double&, int, int);
*/

template <> __device__ float shflUp(float& v, int lane, int width) {
	return __shfl_up(v,lane, width);
}
template <> __device__ double shflUp(double& v, int lane, int width) {
	float half1 = *((float*)&v);
	float half2 = *(((float*)&v) + 1);
	half1 = __shfl_up(half1,lane, width);
	half2 = __shfl_up(half2,lane, width);
	double lv;
	*((float*)&lv) = half1;
	*(((float*)&lv) + 1) = half2;
	return lv;
/*
	int lo, hi;
	asm volatile( "mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(v));
	lo = __shfl_up(lo, lane, width);
	hi = __shfl_up(hi, lane, width);
	asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(v) : "r"(lo), "r"(hi));
	return v;
*/
}
/*
template __device__ float shfl<float>(float&, int, int);
template __device__ double shfl<double>(double&, int, int);
*/

template <> __device__ float shflDown(float& v, int lane, int width) {
	return __shfl_down(v,lane, width);
}
template <> __device__ int shflDown(int& v, int lane, int width) {
	return __shfl_down(v,lane, width);
}
template <> __device__ uint shflDown(uint& v, int lane, int width) {
	return __shfl_down(v,lane, width);
}
template <> __device__ double shflDown(double& v, int lane, int width) {
	float half1 = *((float*)&v);
	float half2 = *(((float*)&v) + 1);
	half1 = __shfl_down(half1,lane, width);
	half2 = __shfl_down(half2,lane, width);
	double lv;
	*((float*)&lv) = half1;
	*(((float*)&lv) + 1) = half2;
	return lv;
/*
	int lo, hi;
	asm volatile( "mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(v));
	lo = __shfl_down(lo, lane, width);
	hi = __shfl_down(hi, lane, width);
	asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(v) : "r"(lo), "r"(hi));
	return v;
*/
}
/*
template __device__ float shfl<float>(float&, int, int);
template __device__ double shfl<double>(double&, int, int);
*/

template <> __device__ ulong shflDown(ulong& v, int lane, int width) {
/*
	int lo, hi;
	asm volatile( "mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(v));
	lo = __shfl(lo, lane, width);
	hi = __shfl(hi, lane, width);
	asm volatile( "mov.b64 %0, {%1,%2};" : "=l"(v) : "r"(lo), "r"(hi));
*/
	uint half1 = *((uint*)&v);
	uint half2 = *(((uint*)&v) + 1);
	half1 = __shfl_down((int)half1,lane, width);
	half2 = __shfl_down((int)half2,lane, width);
	ulong lv;
	*((uint*)&lv) = half1;
	*(((uint*)&lv) + 1) = half2;
	return lv;
}


template<typename T> __global__ void dot(T* res, const T* a, const T* b, int len) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < len) {
		T s = a[idx] * b[idx];
	}

}

#ifdef  CuMatrix_Enable_KTS
template<typename T, typename BinaryOpF> __global__ void shuffle(T* res, const T* a, ulong len, BinaryOpF bop)
#else
template<typename T, int StateDim> __global__ void shuffle(T* res, const T* a, ulong len, MonoidF<T,StateDim> bop)
#endif
{
	int idx = threadIdx.x;
	int len2 = b_util::nextPowerOf2(len);
	int currLen = len2/2;
	T s = idx < len ? a[idx]: bop.identity_ro();
	T s0 = s;
	while(currLen > 0 ) {
		if(checkDebug(debugRedux)&&threadIdx.x == 0 && blockIdx.x == 0)flprintf("currLen %d\n", currLen);
		int lane = idx + currLen;
		//s = lane < len ? bop(s, __shfl(s, lane)) : s;
		s = bop(s, shfl<T>(s, lane));
		//s = bop(s, lane < len ? __shfl(s, lane) : bop.identity);
		if(checkDebug(debugRedux)&&threadIdx.x == 0 && blockIdx.x == 0) flprintf("idx: %d, lane %d, s0 %f, s %f\n", idx, lane,  s0, s);
		s0 += s;
		currLen >>= 1;
	}
	if(idx == 0) {
		*res = s;
	}
}

#ifdef  CuMatrix_Enable_KTS
#else
template void __global__ shuffle<float, 1>(float*, float const*, unsigned long, MonoidF<float, 1>);
template void __global__ shuffle<double, 1>(double*, double const*, unsigned long, MonoidF<double, 1>);
template void __global__ shuffle<int, 1>(int*, int const*, unsigned long, MonoidF<int, 1>);
template void __global__ shuffle<uint, 1>(uint*, uint const*, unsigned long, MonoidF<uint, 1>);

#endif

__global__ void warpReduce() {
	float value = threadIdx.x;
	for(int i = 16; i >= 1; i/=2) {
		printf("i %d xor(i,32) %d\n" , i, ((i ^ 32) & 0x1f));
		value += __shfl_xor(value,i,32);
	}

	printf("Thread %d final value %f\n", threadIdx.x, value);

}

template<typename T, typename BinaryOp>
__device__ T warpShuffleReduce(BinaryOp op) {
	uint tid = threadIdx.x;
	T s = op.identity;
	for(int i = 16; i >= 1; i/=2) {
		//printf("i %d xor(i,32) %d\n" , i, ((i ^ 32) & 0x1f));
		s = op(s, shfl<T>(s, tid));
	}
	return s;
}

__global__ void bcast(int arg) {
	int laneId = threadIdx.x & 0x1f;
	int value;
	if(laneId == 0) {
		value = arg;
	}
	value = __shfl(value,0);

	if(value != arg) {
		printf("Thread %d failed\n", threadIdx.x);
	}

}


#ifdef  CuMatrix_Enable_KTS
template <typename T, typename BinaryOp > __host__ CUDART_DEVICE T shuffle(  const T* a, ulong len, BinaryOp bop)
#else
template <typename T, int StateDim > __host__ CUDART_DEVICE T shuffle(  const T* a, ulong len, MonoidF<T,StateDim> bop)
#endif
{
	T* d_ary;
	size_t size = len * sizeof(T);
#ifndef __CUDA_ARCH__
	checkCudaErrors(cudaMalloc((void**)&d_ary, size));
	checkCudaErrors(cudaMemcpy(d_ary, a, size, cudaMemcpyHostToDevice));
#else
	d_ary = (T*)malloc(size);
	memcpy(d_ary,a,size);
#endif
	T* d_res;
#ifndef __CUDA_ARCH__
	checkCudaErrors(cudaMalloc((void**)&d_res, sizeof(T)));
#else
	d_res = (T*)malloc(sizeof(T));
#endif

#ifdef  CuMatrix_Enable_KTS
	shuffle<T><<<1,len>>>(d_res,d_ary ,len,bop);
#else
	shuffle<T,StateDim><<<1,len>>>(d_res,d_ary ,len,bop);
#endif
	cherr(cudaDeviceSynchronize());
#ifndef __CUDA_ARCH__
	T res;
	checkCudaErrors(cudaMemcpy(&res, d_res, sizeof(T), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_ary));
	return res;
#else
	return *d_res;
#endif
}

#ifdef  CuMatrix_Enable_KTS
template float shuffle<float, plusBinaryOp<float> >(const float*, ulong, plusBinaryOp<float>);
template double shuffle<double, plusBinaryOp<double> >(const double*, ulong, plusBinaryOp<double>);
template ulong shuffle<ulong, plusBinaryOp<ulong> >(const ulong*, ulong, plusBinaryOp<ulong>);
template int shuffle<int, plusBinaryOp<int> >(int const*, unsigned long, plusBinaryOp<int>);
template unsigned int shuffle<unsigned int, plusBinaryOp<unsigned int> >(unsigned int const*, unsigned long, plusBinaryOp<unsigned int>);

template float shuffle<float, multBinaryOp<float> >(const float*, ulong, multBinaryOp<float>);
template double shuffle<double, multBinaryOp<double> >(const double*, ulong, multBinaryOp<double>);
template ulong shuffle<ulong, multBinaryOp<ulong> >(const ulong*, ulong, multBinaryOp<ulong>);
template int shuffle<int, multBinaryOp<int> >(int const*, unsigned long, multBinaryOp<int>);
template unsigned int shuffle<unsigned int, multBinaryOp<unsigned int> >(unsigned int const*, unsigned long, multBinaryOp<unsigned int>);

template float shuffle<float, sqrPlusBinaryOp<float> >(const float*, ulong, sqrPlusBinaryOp<float>);
template double shuffle<double, sqrPlusBinaryOp<double> >(const double*, ulong, sqrPlusBinaryOp<double>);
template ulong shuffle<ulong, sqrPlusBinaryOp<ulong> >(const ulong*, ulong, sqrPlusBinaryOp<ulong>);

template float shuffle<float, maxBinaryOp<float> >(const float*, ulong, maxBinaryOp<float>);
template double shuffle<double, maxBinaryOp<double> >(const double*, ulong, maxBinaryOp<double>);
template ulong shuffle<ulong, maxBinaryOp<ulong> >(const ulong*, ulong, maxBinaryOp<ulong>);

template float shuffle<float, minBinaryOp<float> >(const float*, ulong, minBinaryOp<float>);
template double shuffle<double, minBinaryOp<double> >(const double*, ulong, minBinaryOp<double>);
template ulong shuffle<ulong, minBinaryOp<ulong> >(const ulong*, ulong, minBinaryOp<ulong>);

template float shuffle<float, andBinaryOp<float> >(float const*, unsigned long, andBinaryOp<float>);
template double shuffle<double, andBinaryOp<double> >(double const*, unsigned long, andBinaryOp<double>);
template ulong shuffle<ulong, andBinaryOp<ulong> >(ulong const*, unsigned long, andBinaryOp<ulong>);

template float shuffle<float, orBinaryOp<float> >(float const*, unsigned long, orBinaryOp<float>);
template double shuffle<double, orBinaryOp<double> >(double const*, unsigned long, orBinaryOp<double>);
template ulong shuffle<ulong, orBinaryOp<ulong> >(ulong const*, unsigned long, orBinaryOp<ulong>);

template int shuffle<int, maxBinaryOp<int> >(int const*, unsigned long, maxBinaryOp<int>);
template int shuffle<int, minNotZeroBinaryOp<int> >(int const*, unsigned long, minNotZeroBinaryOp<int>);
template int shuffle<int, andBinaryOp<int> >(int const*, unsigned long, andBinaryOp<int>);
template unsigned int shuffle<unsigned int, andBinaryOp<unsigned int> >(unsigned int const*, unsigned long, andBinaryOp<unsigned int>);
template int shuffle<int, sqrPlusBinaryOp<int> >(int const*, unsigned long, sqrPlusBinaryOp<int>);
template unsigned int shuffle<unsigned int, sqrPlusBinaryOp<unsigned int> >(unsigned int const*, unsigned long, sqrPlusBinaryOp<unsigned int>);
template int shuffle<int, minBinaryOp<int> >(int const*, unsigned long, minBinaryOp<int>);
template unsigned int shuffle<unsigned int, maxBinaryOp<unsigned int> >(unsigned int const*, unsigned long, maxBinaryOp<unsigned int>);
template unsigned int shuffle<unsigned int, minBinaryOp<unsigned int> >(unsigned int const*, unsigned long, minBinaryOp<unsigned int>);

#else
template float shuffle<float,1>(float const*, ulong, MonoidF<float, 1>);
template double shuffle<double,1>(double const*, ulong, MonoidF<double, 1>);
template int shuffle<int,1>(int const*, ulong, MonoidF<int, 1>);
template uint shuffle<uint,1>(uint const*, ulong, MonoidF<uint, 1>);
template ulong shuffle<ulong, 1>(ulong const*, ulong, MonoidF<ulong, 1>);
#endif

__host__ __device__ void bprintBinary(char buff_33char[33], uint val) {
	for (int i = 31; i >= 0; i--) {
		buff_33char[i] = '0' + ((val >> i) & 1);
	}
	buff_33char[32]='\0';
}
