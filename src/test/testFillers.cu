/*
 * testFillers.cc
 *
 *  Created on: Jul 17, 2014
 *      Author: reid
 */


#include "testKernels.h"
#include "tests.h"
#include "../CuDefs.h"
#include "../CuMatrix.h"
#include "../Kernels.h"
#include "../Maths.h"

template <typename T>  void __global__  fillFtr(T* res, UnaryOpIndexF<T,1> ftor,uint idx) {
	FirstThread {
		*res = ftor(idx);
		flprintf("fillFtr pfiller opertr res %f\n", (float) *res);
	}
}

template int testFillFPtr<float>::operator()(int argc, char const ** args) const;
template int testFillFPtr<double>::operator()(int argc, char const ** args) const;
template int testFillFPtr<ulong>::operator()(int argc, char const ** args) const;
template <typename T>  int testFillFPtr<T>::operator()(int argc, const char** args) const {
	outln("testFillFPtr<T>() enter ");
	assert(false);
	kayrnlL<T>();

	launchDFuncArrayBuilder<T>();

	//typename UnaryOpIndexF<T>::uintMethod * arry =  fillas<T>::ops;
/*
	flprintf("testFillFPtr creating %d bytes managed mem for functor operator pointers\n",  id_endOfFillers * sizeof(typename UnaryOpIndexF<T>::uintMethod));
	flprintf("testFillFPtr fillas<T>::ops %p\n", fillas<T>::ops);
	flprintf("testFillFPtr &fillas<T>::ops %p\n", &fillas<T>::ops);

	checkCudaErrors(cudaMallocManaged(&fillas<T>::ops, id_endOfFillers * sizeof(typename UnaryOpIndexF<T,1>::uintMethod)));
	flprintf("testFillFPtr after cudaMallocManaged fillas<T>::ops %p\n", fillas<T>::ops);
	buildFillerFtorArray<T><<<1,1>>>(fillas<T>::ops);
*/

	constFiller<T> ftor = Functory<T, constFiller>::pinch(17);
	//constFiller<T>::instance(ftor, 176);
	flprintf("testFillFPtr &constFiller<T>::operator() %p\n" , &constFiller<T>::operator());
	T* rest = null;
	checkCudaErrors(cudaMallocManaged(&rest, sizeof(T)));
	fillFtr<<<1,1>>>(rest, ftor, 3);
	checkCudaErrors(cudaDeviceSynchronize());
	flprintf("rest %f\n", (float) *rest);
	return 0;
}

template int testFillNsb<float>::operator()(int argc, char const ** args) const;
template int testFillNsb<double>::operator()(int argc, char const ** args) const;
template <typename T>  int testFillNsb<T>::operator()(int argc, const char** args) const {

	const int count = b_util::getCount(argc,args,1000);
	float exeTime;
	CuMatrix<T> m00 = CuMatrix<T>::ones(10,10);
	outln("m00.sum " << m00.sum());
	outln("m00 " << m00.syncBuffers());

	CuMatrix<T> trg = CuMatrix<T>::zeros(1000,1000);
	const float sizeG= 1. * trg.size / Giga;
	const uint xfer = count * sizeG;
	//const uint lengthInTs = src.size/sizeof(T);
	float memFlowIter = 0;
    CuTimer timer;
#ifdef  CuMatrix_Enable_KTS
	CuMatrix<T>::fillFnNsb(Functory<T, oneOverFiller>::pinch(), trg);
#else
	CuMatrix<T>::fillFnNsb(Functory<T, oneOverFiller>::pinch(), trg);
#endif
	outln("trg.sum " << trg.sum());
	outln("trg " << trg.syncBuffers());

	timer.start();
	oneOverFiller<T> step = Functory<T, oneOverFiller>::pinch();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::fillFn(step, trg);
	}
    exeTime = timer.stop();

    memFlowIter = xfer * 1000/exeTime;
	outln("CuMatrix<T>::fillFn N " << count << " took exeTime " << (exeTime /1000) << "s or flow (w) of " << memFlowIter << "GB/s");
	outln("trg " << trg.sum());

	timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::fillFnNsb(step, trg);
	}
    exeTime = timer.stop();
    memFlowIter = xfer * 1000/exeTime;
	outln("CuMatrix<T>::fillFnNsb N " << count << " took exeTime " << (exeTime /1000) << "s or flow (w) of " << memFlowIter << "GB/s");
	outln("trg " << trg.sum());

	return 0;
}


template int testFillers<float>::operator()(int argc, char const ** args) const;
template int testFillers<double>::operator()(int argc, char const ** args) const;
template int testFillers<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testFillers<T>::operator()(int argc, const char** args) const {
	outln("testFillers start");
	CuMatrix<T> pnes = CuMatrix<T>::ones(2 * Mega, 64);
	CuMatrix<T> msin = CuMatrix<T>::sin(10,1);
	CuMatrix<T> mcos = CuMatrix<T>::cos(10,1);
	outln("pnes\n" << pnes.syncBuffers());
	outln("mcos\n" << mcos.syncBuffers());
	outln("msin\n" << msin.syncBuffers());

	return 0;
}

