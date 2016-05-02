/*
 * testfunctors.cu
 *
 *  Created on: Sep 17, 2013
 *      Author: reid
 */




#include "tests.h"
#include "../CuMatrix.h"
#include "../util.h"
#include "../caps.h"


template <typename T,int StateDim> __global__ void compFunctors(UnaryOpIndexF<T,StateDim>* inManaged, UnaryOpIndexF<T,StateDim>* inDev, typename UnaryOpIndexF<T,StateDim>::uintFunction inmethod) {
	constFiller<T>* filbert = new constFiller<T>();
	filbert->value() = 5;
	typename UnaryOpIndexF<T,1>::uintFunction filberator = &constFiller<T>::operator();
	flprintf("filbert %p, inManaged %p, inDev %p", filbert, inManaged, inDev);
	flprintf("filbert.op %p, inManaged.op %p, inDev.op %p", &(filbert->operator()), &(inManaged->operator()), &(inDev->operator()));
	flprintf(" (*filbert)(1) %f\n", (float)  (*filbert)(1));
	delete filbert;
}

template <typename T> __global__ void cloneConstFiller( constFiller<T>* fil) {
	new (fil) constFiller<T>();
	fil->state = 27;
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		typename UnaryOpIndexF<T,1>::iopFunction filberator = &UnaryOpIndexF<T,1>::operatorConst;
	#else
		typename UnaryOpIndexF<T,1>::iopMethod filberator = (typename UnaryOpIndexF<T,1>::iopMethod)&constFiller<T>::operator();
	#endif
#endif
	flprintf("(*fil)(1) %f\n", (float) (*fil)(1));
	//cherr(cudaMemcpy(fil, filbert, sizeof(constFiller<T>), cudaMemcpyDeviceToDevice ));
	UnaryOpIndexF<T,1>* pif = fil;
//	pif->operation = null; // no error checking for now
//	flprintf("operation is null: (*pif)(2) %f\n", (float) (*pif)(2));
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		pif->fn = filberator;
		flprintf("pif %p, pif->fn %p\n",pif,pif->fn);
		flprintf("operation is filberator: (*pif)(2) %f\n", (float) (*pif)(2));
	#else
		pif->operation = filberator;
		flprintf("pif %p, pif->operation %p\n",pif,pif->operation);
		flprintf("operation is filberator: (*pif)(2) %f\n", (float) (*pif)(2));
	#endif
#endif
//#ifdef NO_REF
	d_useConstFiller((UnaryOpIndexF<T,1>*)fil);
//#else
	d_useConstFillerR((UnaryOpIndexF<T,1>*&)fil);

	UnaryOpIndexF<T,1> uif(*fil);
	// weird syntax
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		flprintf("uif.value %f uif.fn  %p, uif(6) %f (*uif.fn)(6) %f\n", (float)uif.state, uif.fn, (float) uif(6), (*uif.fn)(uif,6));
	#else
		flprintf("uif.value %f uif.operation  %p, uif(6) %f (*uif.operation)(6) %f\n", (float)uif.state, uif.operation, (float) uif(6), uif(6));
	#endif
#endif
	d_useConstFillerV(*((UnaryOpIndexF<T,1>*)fil));
//#endif
}

template <typename T> __device__ void d_useConstFiller(UnaryOpIndexF<T,1>* fil) {
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		flprintf("fil %p, fil->fn %p\n",fil,fil->fn);
	#else
		flprintf("fil %p, fil->operation %p\n",fil,fil->operation);
	#endif
#endif
	flprintf("(*fil)(3) %f\n", (float) (*fil)(3));
}

template <typename T> __device__ void d_useConstFillerR(UnaryOpIndexF<T,1>*& fil) {
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		flprintf("fil %p, fil->fn %p\n",fil,fil->fn);
	#else
		flprintf("fil %p, fil->operation %p\n",fil,fil->operation);
	#endif
#endif
	flprintf("(*fil)(3) %f\n", (float) (*fil)(3));

}

template <typename T> __device__ void d_useConstFillerV(UnaryOpIndexF<T,1> fil) {
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		flprintf("fil %p, fil->fn %p\n",&fil,fil.fn);
	#else
		flprintf("fil %p, fil->operation %p\n",&fil,fil.operation);
	#endif
#endif
	flprintf(" fil(4) %f\n", (float) fil(4));

}


template <typename T> __global__ void cloneConstFiller2( constFiller<T>* fil) {
	new (fil) constFiller<T>();
	fil->state = 5;

#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		typename UnaryOpIndexF<T,1>::iopFunction filberator = &UnaryOpIndexF<T,1>::operatorConst;
		fil->fn = filberator;
		flprintf("fil %p, fil->fn %p\n",fil,fil->fn);
	#else
		typename UnaryOpIndexF<T,1>::iopMethod filberator = (typename UnaryOpIndexF<T,1>::iopMethod)&constFiller<T>::operator();
		fil->operation = filberator;
		flprintf("(*fil)(1) %f\n", (float) (*fil)(1));
		flprintf("fil %p, fil->operation %p\n",fil,fil->operation);
	#endif
#endif
	flprintf("(*fil)(2) %f\n", (float) (*fil)(2));
	d_useConstFiller(fil);
}

template <typename T> __global__ void useConstFiller( UnaryOpIndexF<T,1>* fil) {
	flprintf("useConstFiller enter fil %p\n",fil);
	flprintf("(*fil)(2) %f\n", (float) (*fil)(2));
}

template <typename T> __global__ void useConstFillerTyped( constFiller<T>* fil) {
	flprintf("useConstFillerTyped enter fil %p\n",fil);
	flprintf("(*fil)(2) %f\n", (float) (*fil)(2));
}




template int testCopyFtor<float>::operator()(int argc, const char **argv) const;
template int testCopyFtor<double>::operator()(int argc, const char **argv) const;
template int testCopyFtor<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testCopyFtor<T>::operator()(int argc, const char **argv) const {
	outln("testCopyFtor() enter");
	constFiller<T>* pf;
	cherr(cudaMalloc(&pf, sizeof(constFiller<T>)));
	constFiller<T>* pf2;
	cherr(cudaMalloc(&pf2, sizeof(constFiller<T>)));
	cloneConstFiller<T><<<1,1>>>(pf);
	cherr(cudaDeviceSynchronize());
	cloneConstFiller2<T><<<1,1>>>(pf2);
	cherr(cudaDeviceSynchronize());
	useConstFillerTyped<T><<<1,1>>>(pf);
	cherr(cudaDeviceSynchronize());
	useConstFiller<T><<<1,1>>>(pf);
	cherr(cudaFree(pf));
	cherr(cudaFree(pf2));
	return 0;
}

template int testFastInvSqrt<float>::operator()(int argc, const char **argv) const;
template int testFastInvSqrt<double>::operator()(int argc, const char **argv) const;
template <typename T> int testFastInvSqrt<T>::operator()(int argc, const char **argv) const {
	int count = b_util::getCount(argc,argv,100);
    CuTimer timer;
	setCurrGpuDebugFlags( debugVerbose,true,false);
	slowInvSqrtUnaryOp<T> slow = Functory<T, slowInvSqrtUnaryOp>::pinch();
	approxInvSqrtUnaryOp<T> fast = Functory<T, approxInvSqrtUnaryOp>::pinch();
	CuMatrix<T> m0 = CuMatrix<T>::ones(40, 20);
	outln("m0"  << m0.syncBuffers());
	CuMatrix<T> m1 = CuMatrix<T>::ones(999, 999);
	CuMatrix<T> m1slow = CuMatrix<T>::zeros(999,999);
	CuMatrix<T> m1fast = CuMatrix<T>::zeros(999,999);

	timer.start();
	for(int i=0;i < count; i++) {
		m1.unaryOp(m1fast, fast);
	}
	float fastT = timer.stop();
	outln("fast " << count << " took " << fastT);
	T sumFast = m1fast.sum();
	outln("fast sanity " << sumFast);

	timer.start();
	for(int i=0;i < count; i++) {
		m1.unaryOp(m1slow, slow);
	}
	float slowT = timer.stop();
	outln("slow " << count << " took " << slowT );
	T sumSlow = m1slow.sum();
	outln("slow sanity " << sumSlow);
	outln("%diffT " << (100*(slowT-fastT)/slowT));
	assert(fabs((slowT-fastT)/slowT) < .05);
	outln("%accuracy " <<  ( 100 - 100*fabs(sumFast-sumSlow)/sumSlow));
	setCurrGpuDebugFlags( ~debugVerbose,false,true);
	return 0;
}

#include "tests.cc"
