/*
 * CuDefs.h
 *
 *  Created on: Oct 10, 2013
 *      Author: reid
 */
#pragma once

#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define MABS(exp) ((exp) < 0 ? -(exp) : (exp))
#define MAX_FUNCS 10

#ifdef __GNUC__
    #define MAYBE_UNUSED __attribute__((used))
#elif defined _MSC_VER
    #pragma warning(disable: Cxxxxx)
    #define MAYBE_UNUSED
#else
    #define MAYBE_UNUSED
#endif

#ifdef __CUDA_ARCH__
	#ifndef nullptr
		#define nullptr NULL
	#endif
#endif

#define null NULL
#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif
#ifndef Pi
#define Pi 3.141592653589793
#endif
#ifndef ONE_OVER_SQR_2PI
#define ONE_OVER_SQR_2PI 	0.3989422804014
#endif
#ifndef ONE_OVER_2PI
#define ONE_OVER_2PI 	0.159154943
#endif

#define Giga (1024*1024*1024)
#define tOrF(exp) ( (exp)? "true" : "false")
#define Mega (1024*1024)
#define Kilo (1024)

#ifndef WARP_SIZE
#define WARP_SIZE 32
#define MAX_BLOCK_DIM   64
#define MAX_THREAD_DIM   256
#endif

#define b_u b_util


#define DEFAULT_BLOCK_X	32
#define DEFAULT_BLOCK_Y	8
#define UNOP_BLOCK_SIZE 	DEFAULT_BLOCK_X
#define UNOP_X2Y			4
#define MaxDim 65536u
#define TX_BLOCK_SIZE 	DEFAULT_BLOCK_X
#define CAT_BLOCK_SIZE 	DEFAULT_BLOCK_X
#define DefaultWidth2Height  8
#define __h_ __host__
#define __d_ __device__
#define __g_ __global__


template <typename T> extern __device__ T shfl(T& v, int lane, int width= WARP_SIZE);
template <typename T> extern __device__ T shflUp(T& v, int lane, int width= WARP_SIZE);
template <typename T> extern __device__ T shflDown(T& v, int lane, int width= WARP_SIZE);
#ifdef  CuMatrix_Enable_KTS
template<typename T, typename BinaryOpF> extern __host__ CUDART_DEVICE T shuffle(  const T* a, ulong len, BinaryOpF bop);
template<typename T, typename BinaryOpF> __global__ void shuffle(T* res, const T* a, ulong len, BinaryOpF bop);
#else
template<typename T, int StateDim> struct MonoidF;
template<typename T, int StateDim> extern __host__ CUDART_DEVICE T shuffle(  const T* a, ulong len, MonoidF<T,StateDim> bop);
template<typename T, int StateDim> __global__ void shuffle(T* res, const T* a, ulong len, MonoidF<T,StateDim> bop);
#endif
#define Pi 3.141592653589793
#define OneOverSqrt2Pi 2.5066282746310002
#ifndef PRINT_BINARY_BUFF_LEN
	#define PRINT_BINARY_BUFF_LEN  32
#endif

__host__ __device__ void bprintBinary(char buff_33char[33], uint v);


