/*
 * Maths.h
 *
 *  Created on: May 15, 2014
 *      Author: reid
 */

#ifndef MATHS_H_
#define MATHS_H_


#include "FuncPtr.h"

#define TEN_THOUSANDTH_PRIME 104729
extern __device__ int First10KPrimes[];
extern __device__ int First50KPrimes[];


__global__ void scanSum(int* d_res, int fin);
int scanSumL(int fin);

template <typename T>__device__  T cube (T x);
template <typename T> __host__ __device__ T sign(T x);
/*
 * spanrowQ returns true if the row spans multiple warps
 */
__device__ __forceinline__ bool spanrowQ( int row, int n ) {
	uint ret;

	asm("{\n\t"
			" .reg .u32 warpS, warpE;\n\t"    // warpS == start warp, warpE == end warp
			" .reg .pred p;\n\t"
			" mul.lo.u32 warpS, %1, %2;\n\t"	// start warp = [ row * n] / warpSize
			" mad.lo.u32 warpE, %1, %2, %2;\n\t"  // end warp = [ (row + 1 ) * n] / warpSize
			" div.u32 warpS, warpS, WARP_SZ;\n\t" // start warp = row * n [ / warpSize]
			" div.u32 warpE, warpE, WARP_SZ;\n\t"
			" setp.ne.u32 p, warpS, warpE;\n\t"		// return true if warpS != warpE
			" selp.u32 %0, 1, 0, p;\n\t"
			"}"
			: "=r"(ret) : "r" (row), "r" (n));
	return ret;
}
template <typename T> __device__ __forceinline__  T mod (T x, T divisor);

template<> __device__ __forceinline__ int mod<int>( int x, int divisor ) {
	int ret;

/*
	asm("{\n\t"
			" .reg .u32 temp;\n\t"
			" div.u32 temp, %1, %2;\n\t" // t = (floor) x/divisor
			" mul.lo.u32 temp, %2, temp;\n\t"	// t = t * divisor
			" sub.u32 %0, %1, temp;\n\t" 		// ret = x - t
			"}"
			: "=r"(ret) : "r" (x), "r" (divisor));
*/
	asm("{\n\t"
			" rem.s32 %0, %1, %2;\n\t" 		// ret = x - t
			"}"
			: "=r"(ret) : "r" (x), "r" (divisor));
	return ret;
}
template<> __device__ __forceinline__ uint mod<uint>( uint x, uint divisor ) {
	uint ret;

/*
	asm("{\n\t"
			" .reg .u32 temp;\n\t"
			" div.u32 temp, %1, %2;\n\t" // t = (floor) x/divisor
			" mul.lo.u32 temp, %2, temp;\n\t"	// t = t * divisor
			" sub.u32 %0, %1, temp;\n\t" 		// ret = x - t
			"}"
			: "=r"(ret) : "r" (x), "r" (divisor));
*/
	asm("{\n\t"
			" rem.u32 %0, %1, %2;\n\t" 		// ret = x - t
			"}"
			: "=r"(ret) : "r" (x), "r" (divisor));
	return ret;
}
template<> __device__ __forceinline__ long mod<long>( long x, long divisor ) {
	long ret;

	asm("{\n\t"
			" rem.s64 %0, %1, %2;\n\t" 		// ret = x - t
			"}"
			: "=l"(ret) : "l" (x), "l" (divisor));
	return ret;
}

__host__ CUDART_DEVICE  int largestFactor(uint v,bool smallest=false);
__host__ CUDART_DEVICE  int largestCommonFactor(uint v, uint w, bool smallest=false);
__host__ CUDART_DEVICE  int smallestFactor(uint v);
__host__ CUDART_DEVICE  bool primeQ(uint v);
__host__ CUDART_DEVICE  int smallestCommonFactor(uint v, uint w);
__host__ CUDART_DEVICE  void biggestBin( int* whichBin, const int* binCounts, int nbins);

template<typename T, template <typename> class Function> __host__ __device__ inline T gradient(Function<T> function, T x, T epsilon);

template<typename T, template <typename> class Function> __host__ CUDART_DEVICE
T bisection( T* roots, uint* count, uint maxRoots, Function<T> function, T a, T b, T epsilon, uint slices);

template<typename T> __global__ void getFunctionPtr(T* fptr, int funcIndex);


template<typename T> __host__ CUDART_DEVICE T bisection( T* roots, uint* count, uint maxRoots, typename func1<T>::inst fn, T a, T b, T epsilon, uint slices);
template<typename T> __host__ CUDART_DEVICE T bisection( T* roots, uint* count, uint maxRoots, FuncNums idx, T a, T b, T epsilon, uint slices);


#endif /* MATHS_H_ */
