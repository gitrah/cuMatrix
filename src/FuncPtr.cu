/*
 * FuncPtr.cu
 *
 *  Created on: Jun 22, 2014
 *      Author: reid
 */

#include "FuncPtr.h"
#include "debug.h"
#include "UnaryOpIndexF_Gen.h"
//template <typename T> __device__ void * DevFuncs<T>[];
template <typename T> typename func1<T>::inst**  Funcs<T>::func1s = null;
template <>  __device__ float sigmoidFn(float f) {
	return 1.0f / (1.0f + (float) exp(-f));
}

template <> __device__ double sigmoidFn(double d) {
	return 1.0 / (1.0 + exp(-d));
}

template <> __device__ ulong sigmoidFn(ulong d) {
	return ( ulong) 1.0 / (1.0 + exp((double)-d));
}

__global__ void doit() {

}

template <typename T> __global__ void buildIdxFuncArrays(typename idxfunc1<T>::inst * idxfunc1Arry) {
	prlocf("buildIdxFuncArrays\n");
	b_util::pPtrAtts(idxfunc1Arry);
	idxfunc1Arry[eStepFiller] = stepFillerFn<T>;
}

// funcs has been cudaMalloced by host
template <typename T> __global__ void buildDFuncArrays(typename func1<T>::inst* func1Arry) {
/*
	prlocf("buildDFuncArrays\n");
	b_util::pPtrAtts(func1Arry);
	//typename func1<T>::inst fn = ;
	func1Arry[ePoly1_2] = poly2_1<T>;
	func1Arry[eNegateFn] = negateFn<T>;
	func1Arry[eSigmoid] = sigmoidFn<T>;
	typedef T (UnaryOpIndexF<T,1>::*MethodTypeToCall)(uint) const;

	//MethodTypeToCall constOp = &UnaryOpIndexF<T>::func;
	//MethodTypeToCall op2 = &UnaryOpIndexF<T>::operator();
	MethodTypeToCall op3 = (MethodTypeToCall) &constFiller<T>::operator();

	constFiller<T> filleur;
	//filleur.value() = 1;

	constFiller<T>* pfill = (constFiller<T>*)&filleur;
	flprintf("buildDFuncArrays 	(pfill->*op3)(5) %f\n", 	(float) (pfill->*op3)(5));

	flprintf("func1Arry[eSigmoid](3.5) %f\n", func1Arry[eSigmoid](3.5));
*/
	/*
	func1<T>* poly = new func1<T>();
	poly->inst = null;
	func1Arry[ ePoly1_2 ] = poly;
	func1Arry[ ePoly1_2 ]inst = poly2_1<T>;
	func1Arry[ eNegateFn ].inst = negateFn<T>;
	func1Arry[ eSigmoid].inst = sigmoid<T>;
*/
}

template <typename T> __host__ void launchDFuncArrayBuilder() {
	typename func1<T>::inst * fptrs;
	checkCudaErrors(cudaMalloc(&fptrs,MAX_FUNCS * sizeof(typename func1<T>::inst)));
	buildDFuncArrays<T><<<1,1>>>(fptrs);
}
template __host__ void launchDFuncArrayBuilder<float>();
template __host__ void launchDFuncArrayBuilder<double>();
template __host__ void launchDFuncArrayBuilder<ulong>();

template <typename T> struct dfuncs {
	static typename func1<T>::inst funcs[MAX_FUNCS];
};
//template <typename T> __device__ dfuncs<T> theDefuncs;

template <typename T> typename func1<T>::inst dfuncs<T>::funcs[];

__device__ void * funcPtres[MAX_FUNCS];
//funcPtres[0] = 0;
//func[poly1_2] =
template <typename T> struct D_Funcs {
	static
	typename func1<T>::inst f_poly2_1;
};

template <typename T> typename func1<T>::inst D_Funcs<T>::f_poly2_1 = poly2_1<T>;

template <typename T> __global__ void setDFarray() {
	funcPtres[ePoly1_2] = (void*) poly2_1<T>; //func1<T>::inst D_Funcs<T>::f_poly2_1;
	flprintf("funcPtres[ePoly1_2] %p\n", funcPtres[ePoly1_2]);
}
template __global__ void setDFarray<float>();
template __global__ void setDFarray<double>();
template __global__ void setDFarray<ulong>();

template <typename T>  void Funcs<T>::buildFunc1Array() {
}
template <typename T,typename I> __h_ __d_ inline T constFill( const T& s1, I x) {
	return s1;
}
template <typename T,typename I>  __d_ inline T dConstFill( const T& s1, I x) {
	return s1;
}

//template <typename T, typename I> fillk(i)
template <typename T, typename I> __h_ CUDART_DEVICE void idxfn1sFillFn(idxfn1s<T,I> filler) {
	flprintf("idxfn1sFillFn filler(5) %f\n", filler(5) );
}

template <typename T, typename I> __g_ void d_idxfn1s_factory_constFill_k(d_idxfn1s<T,I>* filler, T s1) {
	new (filler) d_idxfn1s<T,I>(); // construct a d_idxfn1s using passed devmem buffer
	filler->s1 = s1;
	filler->inst = dConstFill<T,I>;
}

template <typename T, typename I> __h_ CUDART_DEVICE void d_idxfn1s_factory_constFill_l(d_idxfn1s<T,I>** filler, T s1) {
	cherr(cudaMalloc(filler, sizeof(d_idxfn1s<T,I>)));
	d_idxfn1s_factory_constFill_k<T,I><<<1,1>>>(*filler, s1);
	cherr(cudaDeviceSynchronize());
}



template <typename T, typename I> __g_ void idxfn1sFillFnK(d_idxfn1s<T,I> filler) {
	FirstThread {
		flprintf("idxfn1sFillFnK called with filler.s1 %f\n", (float)filler.s1);
		flprintf("idxfn1sFillFnK called with filler.inst %p\n", filler.inst);
		flprintf("idxfn1sFillFnK called with dConstFill %p\n", dConstFill<T,I>);
		flprintf("idxfn1sFillFnK called with &dConstFill %p\n", &dConstFill<T,I>);
		flprintf("idxfn1sFillFnK called with dConstFill(filler.s1, 5) %f\n", (float)dConstFill<T,I>(filler.s1, 5));
		filler.inst = dConstFill<T,I>;
		flprintf("idxfn1sFillFnK called with (*filler.inst)(filler.s1,5) %f\n", (float)(*filler.inst)(filler.s1,5));

		flprintf("idxfn1sFillFnK filler(5) %f\n", filler(5) );
	}
}
template <typename T, typename I> __g_ void idxfn1sFillFnPtrK(d_idxfn1s<T,I>* filler) {
	FirstThread {
		flprintf("idxfn1sFillFnK called with filler->s1 %f\n", (float)filler->s1);
		flprintf("idxfn1sFillFnK called with filler->inst %p\n", filler->inst);
		flprintf("idxfn1sFillFnK called with dConstFill %p\n", dConstFill<T,I>);
		flprintf("idxfn1sFillFnK called with &dConstFill %p\n", &dConstFill<T,I>);
		flprintf("idxfn1sFillFnK called with (*filler->inst)(filler.s1,5) %f\n", (float)(*filler->inst)(filler->s1,5));

		flprintf("idxfn1sFillFnK filler(5) %f\n", (*filler)(5) );
	}
}

template <typename T, typename I> __h_ CUDART_DEVICE void idxfn1sFillL() {
	prlocf("idxfn1sFillL enter\n" );
	idxfn1s<T,I> filler;
	filler.inst = constFill;
	filler.s1 = 1;
	idxfn1sFillFn(filler);

	d_idxfn1s<T,I> d_filler;
	d_filler.inst = dConstFill;
	d_filler.s1 = 27;
	flprintf("idxfn1sFillL d_filler.inst %p d_filler.s1 %f\n", d_filler.inst, (float)d_filler.s1);
	idxfn1sFillFnK<<<1,1>>>(d_filler);
	cherr(cudaDeviceSynchronize());


	d_idxfn1s<T,I>* pfiller;
	d_idxfn1s_factory_constFill_l(&pfiller, (T)3);
	idxfn1sFillFnPtrK<<<1,1>>>(pfiller);
	cherr(cudaDeviceSynchronize());

}

template __h_ CUDART_DEVICE void idxfn1sFillL<float, uint>();
template __h_ CUDART_DEVICE void idxfn1sFillL<double, uint>();
template __h_ CUDART_DEVICE void idxfn1sFillL<ulong, uint>();
