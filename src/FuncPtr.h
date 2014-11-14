/*
 * FuncPtr.h
 *
 *  Created on: Jun 22, 2014
 *      Author: reid
 */

#ifndef FUNCPTR_H_
#define FUNCPTR_H_

#include "CuDefs.h"


template <typename T> __device__  T poly2_1(T x) {
	T res =  x * x - 2 * x - 11.5;
	//flprintf("poly2_1(%f) = %f\n", (float) x, (float) res);
	return res;
}

template <typename T> __device__ T negateFn(T x) {
	return -x;
}
template <typename T> __device__ T sigmoidFn(T x);
template <typename T> __device__ T stepFillerFn(uint idx) {
	return 1. / static_cast<T>(1 + idx);
}

// max width of device function ptr array

//template <typename T> typedef	 T (*idxInst)( uint);

template <typename T> struct idxfunc1 {

	typedef	 T (*inst)( uint x);
	template<class C>    // or any type of null
	operator T C::*() const       // member pointer...
	{ return 0; }
};

template <typename T> struct idxOperatorfunc1 {
	typedef	 T (*inst)( uint x);
};

template <typename T> struct constFillerFn : public idxfunc1<T> {
	T value;
	T operator()(uint idx) {
		return value;
	}
};

template <typename T> struct sequenceFillerFn : public idxfunc1<T> {
	T phase;
	T operator()(uint idx) {
		return (T) phase + idx;
	}
};

template <typename T> struct func1 {
	typedef	 T (*inst)( T x);
};
template <typename T> struct func2 {
	typedef	 T (*inst)( T x1, T x2);
};


template <typename T> struct Funcs {
	 static typename func1<T>::inst** func1s;
	 int func1Count;
	 void buildFunc1Array();
};

extern __managed__ void * funcPtres[MAX_FUNCS];
template <typename T> __global__ void setDFarray() ;

template <typename T> __global__ void buildIdxFuncArrays(typename idxfunc1<T>::inst * idxfunc1Arry);
template <typename T> __global__ void buildDFuncArrays(typename func1<T>::inst * func1Arry);
template <typename T> __host__ void launchDFuncArrayBuilder();

enum FillerNums {
	eStepFiller
};

enum FuncNums {
	ePoly1_2, eNegateFn, eSigmoid
};


// read as index (of type I) function with 1 state var
template <typename T, typename I> struct idxfn1s {
	T s1;
	typedef T (*method)( const T& s1, I x);
	method inst;
	__h_ __d_ T operator() (I idx) const {
		return (*inst)(s1,idx);
	}
};

template <typename T, typename I> __h_ CUDART_DEVICE void fillFn(idxfn1s<T,I> filler);
template <typename T, typename I> __h_ CUDART_DEVICE void idxfn1sFillL();

template <typename T, typename I> struct d_idxfn1s {
	T s1;
	typedef T (*method)( const T& s1, I x);
	method inst;
	__d_ T operator() (I idx) const {
		return (*inst)(s1,idx);
	}
};

#endif /* FUNCPTR_H_ */
