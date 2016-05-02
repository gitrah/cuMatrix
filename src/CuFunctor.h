/*
 * CuFunctor.h
 *
 *  Created on: Sep 28, 2014
 *      Author: reid
 *  header for CuFunctor base class and Functories
 */

#pragma once
#include <helper_cuda.h>
#include <helper_math.h>
#include "CuDefs.h"
#include "debug.h"
#include "FunctorEnums_Gen.h"

// true when all method tables initialized for this GPU
extern bool SetupMbrFuncs[dtLast][MAX_GPUS];
int cuFunctorMain();
void clearSetupMbrFlags();
/*
 * base functor parameterized by numeric type and state dimension
 * with [] operators to allow state to be accessed as an array
 */
template<typename T, int StateDim> struct CuFunctor {
	__host__ int toDevice(void** devInstance) {
		checkCudaErrors(cudaMalloc(devInstance, sizeof(this)));
		cudaError_t lerr =  cudaMemcpy(*devInstance, this, sizeof(this), cudaMemcpyHostToDevice);
		checkCudaErrors(lerr);
		return lerr;
	}
	__host__ cudaError_t free(void* devInstance) {
		return cudaFree(devInstance);
	}
	__host__ __device__ T& operator[](ptrdiff_t ofs);
	const T& operator[](ptrdiff_t ofs) const;
};


template<typename T> struct CuFunctor<T,1> {
	T state;
	inline __host__ __device__ T& operator[](ptrdiff_t ofs) {	assert(ofs == 0 );	return state;	}
	inline __host__ __device__ const T& operator[](ptrdiff_t ofs) const {assert(ofs == 0);return state;}
};

template<typename T> struct CuFunctor<T,2> {
	inline __host__ __device__ T& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const T& operator[](ptrdiff_t ofs) const;
};

template <> struct CuFunctor<float, 2> {
	float2 state;
	inline __host__ __device__ float& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const float& operator[](ptrdiff_t ofs)const;
};

template <> struct CuFunctor<double, 2> {
	double2 state;
	inline __host__ __device__ double& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const double& operator[](ptrdiff_t ofs)const ;
};

template <> struct CuFunctor<int, 2> {
	int2 state;
	inline __host__ __device__ int& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const int& operator[](ptrdiff_t ofs)const ;
};

template <> struct CuFunctor<uint, 2> {
	uint2 state;
	inline __host__ __device__ uint& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const uint& operator[](ptrdiff_t ofs)const ;
};

template <> struct CuFunctor<long, 2> {
	long2 state;
	inline __host__ __device__ long& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const long& operator[](ptrdiff_t ofs)const ;
};

template <> struct CuFunctor<ulong, 2> {
	ulong2 state;
	inline __host__ __device__ ulong& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const ulong& operator[](ptrdiff_t ofs)const ;
};

template <> struct CuFunctor<float, 3> {
	float3 state;
	inline __host__ __device__ float& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const float& operator[](ptrdiff_t ofs)const ;
};

template <> struct CuFunctor<double, 3> {
	double3 state;
	inline __host__ __device__ double& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const double& operator[](ptrdiff_t ofs)const ;
};

template <> struct CuFunctor<int, 3> {
	int3 state;
	inline __host__ __device__ int& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const int& operator[](ptrdiff_t ofs)const ;
};

template <> struct CuFunctor<uint, 3> {
	uint3 state;
	inline __host__ __device__ uint& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const uint& operator[](ptrdiff_t ofs)const ;
};

template <> struct CuFunctor<long, 3> {
	long3 state;
	inline __host__ __device__ long& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const long& operator[](ptrdiff_t ofs)const ;
};

template <> struct CuFunctor<ulong, 3> {
	ulong3 state;
	inline __host__ __device__ ulong& operator[](ptrdiff_t ofs);
	inline __host__ __device__ const ulong& operator[](ptrdiff_t ofs)const ;
};

/*
 * forward declaration of basic functor types
 * 	UnaryOpF:  T => T
 * 	UnaryOpIndexF: ulong => T
 * 	BinaryOpF :  (T,T) => T
 */
template<typename T, int StateDim> struct UnaryOpF;
template<typename T, int StateDim> struct AllUnaryOpF;
template<typename T, int StateDim> struct UnaryOpIndexF;
template<typename T, int StateDim> struct AllUnaryOpIndexF;
template<typename T, int StateDim> struct BinaryOpF;
template<typename T, int StateDim> struct AllBinaryOpF;
template<typename T> __host__ __device__ T Epsilon();

template<typename T> __host__ __device__ T epsilon() { return 0;}
template<typename T> __host__ __device__ T maxValue();
template<typename T> __host__ __device__ T minValue();

/*
 * Class that manages the arrays of device-side method pointers
 * that allows host-side construction of functors that can be passed to device code
 */

#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		template <typename T> struct FunctionTableMgr {
	#else
			template <typename T> struct MethodTableMgr {
	#endif

	#ifdef CuMatrix_StatFunc
		static void setup0sFunctionTables(int device);
		static void setup1sFunctionTables(int device);
		static void setup2sFunctionTables(int device);
		static void setup3sFunctionTables(int device);
	#else
		static void setup0sMethodTables(int device);
		static void setup1sMethodTables(int device);
		static void setup2sMethodTables(int device);
		static void setup3sMethodTables(int device);
	#endif

	#ifdef CuMatrix_StatFunc
		typedef T (*iop0sFunction)(const UnaryOpIndexF<T,0>&, ulong);
		typedef T (*iop1sFunction)(const UnaryOpIndexF<T,1>&, ulong);
		typedef T (*iop2sFunction)(const UnaryOpIndexF<T,2>&, ulong);
		typedef T (*iop3sFunction)(const UnaryOpIndexF<T,3>&, ulong);

		static iop0sFunction h_iop0sFunctions[MAX_GPUS][Iop0sLast];
		static iop1sFunction h_iop1sFunctions[MAX_GPUS][Iop1sLast];
		static iop2sFunction h_iop2sFunctions[MAX_GPUS][Iop2sLast];
		static iop3sFunction h_iop3sFunctions[MAX_GPUS][Iop3sLast];

		typedef T (*uop0sFunction)(const UnaryOpF<T,0>&,T);
		typedef T (*uop1sFunction)(const UnaryOpF<T,1>&,T);
		typedef T (*uop2sFunction)(const UnaryOpF<T,2>&,T);

		static uop0sFunction h_uop0sFunctions[MAX_GPUS][Uop0sLast];
		static uop1sFunction h_uop1sFunctions[MAX_GPUS][Uop1sLast];
		static uop2sFunction h_uop2sFunctions[MAX_GPUS][Uop2sLast];


		typedef T (*bop0sFunction)(const BinaryOpF<T,0>&, T,T);
		typedef T (*bop1sFunction)(const BinaryOpF<T,1>&, T,T);

		static bop0sFunction h_bop0sFunctions[MAX_GPUS][Bop0sLast];
		static bop1sFunction h_bop1sFunctions[MAX_GPUS][Bop1sLast];
	#else //CuMatrix_StatFunc
		typedef T (UnaryOpIndexF<T,0>::*iop0sMethod)(ulong)const;
		typedef T (UnaryOpIndexF<T,1>::*iop1sMethod)(ulong)const;
		typedef T (UnaryOpIndexF<T,2>::*iop2sMethod)(ulong)const;
		typedef T (UnaryOpIndexF<T,3>::*iop3sMethod)(ulong)const;

		static iop0sMethod h_iop0sMethods[MAX_GPUS][Iop0sLast];
		static iop1sMethod h_iop1sMethods[MAX_GPUS][Iop1sLast];
		static iop2sMethod h_iop2sMethods[MAX_GPUS][Iop2sLast];
		static iop3sMethod h_iop3sMethods[MAX_GPUS][Iop3sLast];

		typedef T (UnaryOpF<T,0>::*uop0sMethod)(T)const;
		typedef T (UnaryOpF<T,1>::*uop1sMethod)(T)const;
		typedef T (UnaryOpF<T,2>::*uop2sMethod)(T)const;

		static uop0sMethod h_uop0sMethods[MAX_GPUS][Uop0sLast];
		static uop1sMethod h_uop1sMethods[MAX_GPUS][Uop1sLast];
		static uop2sMethod h_uop2sMethods[MAX_GPUS][Uop2sLast];

		typedef T (BinaryOpF<T,0>::*bop0sMethod)(T,T)const;
		typedef T (BinaryOpF<T,1>::*bop1sMethod)(T,T)const;

		static bop0sMethod h_bop0sMethods[MAX_GPUS][Bop0sLast];
		static bop1sMethod h_bop1sMethods[MAX_GPUS][Bop1sLast];

	#endif // CuMatrix_StatFunc

	};

	// populates all host tables with pointers to device-side functor methods for the given data type T
	template<typename T> struct unaryOpIndexMbrs {
	#ifdef CuMatrix_StatFunc
		static void setupAllFunctionTables( int device);
	#else
		static void setupAllMethodTables( int device);
	#endif
	};
#endif //CuMatrix_Enable_KTS
/*
 * Factory classes that manage the construction and initialization of the functors
 * Functors can be created without these classes but you must remember to call the init method
 * or otherwise set the method pointer
 * (CUDART_DEVICE instead of __device__ below because calling getDevice from device code requires cdp)
 */
template<typename T, template <typename> class  Functor> struct Functory {
	static __host__ CUDART_DEVICE Functor<T> pinch() {
		int currentDevice;
		cherr(cudaGetDevice(&currentDevice));
		Functor<T> f;
#ifndef __CUDA_ARCH__
		//assert(SetupMbrFuncs[getTypeEnum<T>()][currentDevice]);
		if(!SetupMbrFuncs[getTypeEnum<T>()][currentDevice]) {
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
				unaryOpIndexMbrs<T>::setupAllFunctionTables(currentDevice);
	#else
				unaryOpIndexMbrs<T>::setupAllMethodTables(currentDevice);
	#endif
#endif
		}
#endif
		f.init(currentDevice);
		return f;
	}
	static __host__ CUDART_DEVICE Functor<T> pinch(T state) {
		Functor<T> f = pinch();
		f.state = state;
		return f;
	}

	static __host__ CUDART_DEVICE Functor<T> pinch(T state, T state2) {
		Functor<T> f = pinch();
		f.state.x = state;
		f.state.y = state2;
		return f;
	}

	static __host__ CUDART_DEVICE Functor<T> pinch(T state, T state2, T state3) {
		Functor<T> f = pinch();
		f.state.x = state;
		f.state.y = state2;
		f.state.z = state3;
		return f;
	}

};

