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
#include "../debug.h"
#include "../caps.h"

template int testFillNsb<float>::operator()(int argc, const char **argv) const;
template int testFillNsb<double>::operator()(int argc, const char **argv) const;
template <typename T>  int testFillNsb<T>::operator()(int argc, const char **argv) const {

	const int count = b_util::getCount(argc,argv,1000);
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


template int testFillers<float>::operator()(int argc, const char **argv) const;
template int testFillers<double>::operator()(int argc, const char **argv) const;
template int testFillers<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testFillers<T>::operator()(int argc, const char **argv) const {
	outln("testFillers start");
	CuMatrix<T> pnes = CuMatrix<T>::ones(2 * Mega, 64);
	CuMatrix<T> msin = CuMatrix<T>::sin(10,1);
	CuMatrix<T> mcos = CuMatrix<T>::cos(10,1);
	outln("pnes\n" << pnes.syncBuffers());
	outln("mcos\n" << mcos.syncBuffers());
	outln("msin\n" << msin.syncBuffers());

	return 0;
}

template<typename T> void testMemcpy2d() {
	CuMatrix<T> tiny = CuMatrix<T>::ones(10,10);
	CuMatrix<T> tiny2 = 2 * CuMatrix<T>::ones(10,10);

	tiny.syncBuffers();
	tiny2.syncBuffers();
	T* combo;
	cherr(cudaMallocHost(&combo, 2 * tiny.size,0 ));

	cherr(cudaMemcpy2D(combo, 20*sizeof(T), tiny.elements, tiny.p * sizeof(T), tiny.n*sizeof(T), tiny.m, cudaMemcpyHostToHost));
	cherr(cudaMemcpy2D(combo+10, 20*sizeof(T), tiny2.elements, tiny2.p * sizeof(T), tiny2.n*sizeof(T), tiny2.m, cudaMemcpyHostToHost));
	outln("testXRFill combo ");
	for(int rowi = 0; rowi < 10; rowi++) {
		printf("row %d: ",rowi);
		printColoArray(combo + rowi * 20, 20);
	}
}
template int testXRFill<float>::operator()(int argc, const char **argv) const;
template int testXRFill<double>::operator()(int argc, const char **argv) const;
template int testXRFill<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testXRFill<T>::operator()(int argc, const char **argv) const {
	outln("testXRFill start currDev " << ExecCaps::currDev());

	// the mats are too big for the sums to be accurate enough to pass the asserts below
	assert(sizeof(T) > 4);
	// todo todo
	// add a multi-gpu per cumat test for simpler ops

	ExecCaps* pcaps = ExecCaps::currCaps();
	//ExecCaps_setDevice(1);
	ulong hugeMrows = pcaps->maxReasonable( 8.5 * DEFAULT_GMEM_HEADROOM_FACTOR)/sizeof(T); // this will force tiling
	ulong squareDim = sqrt(hugeMrows);
	outln("testXRFill hugeMrows " << hugeMrows << " squareDim " << squareDim);
	CuMatrix<T> hugeM = 1.0 * CuMatrix<T>::ones(squareDim, squareDim);
	CuMatrix<T> huge2M = CuMatrix<T>::fill(2, squareDim, squareDim);

	prlocf("hugeM first 20 elemtns ");
	printColoArray(hugeM.elements, 20);
/*
	for(int i = 1; i < 100; i++) {
		flprintf("hugeM %d/100 = %lu of %lu\n", i, (i * hugeM/100), hugeM);
		printColoArray(hugeM.elements + hugeM * i / 100, 20);
	}
*/
//	setCurrGpuDebugFlags( debugVerbose,true,false);
	outln("hugeM " << hugeM);
//	setCurrGpuDebugFlags( ~debugVerbose,false,true);
	T hugeMsum = hugeM.sum();
	T huge2Msum = huge2M.sum();
	outln("testXRFill hugeMsum " << hugeMsum);
	outln("testXRFill huge2Msum " << huge2Msum);
	assert(hugeMsum == squareDim * squareDim);
	assert(huge2Msum == 2*squareDim * squareDim);

	outln("\n\nattempting sevenHalves");
	CuMatrix<T> sevenHalvesV = hugeM + 2.5 * hugeM;
	T sevenHalvesVsum = sevenHalvesV.sum();
	outln("testXRFill sevenHalvesVsum " << sevenHalvesVsum);
	outln("testXRFill sevenHalvesVsum " << (3.5 * hugeMsum - sevenHalvesVsum));
	outln("testXRFill relDiff(3.5 * hugeMsum,sevenHalvesVsum) " << util<T>::relDiff(3.5 * hugeMsum,sevenHalvesVsum));
	assert( 3.5 * hugeMsum == sevenHalvesVsum);
	outln("freed " << sevenHalvesV.releaseBuffers( ) << " buffers for sevenHalvesV");


	outln("\n\n|= --> hugeM2\n\n");
	CuMatrix<T> hugeM2 = hugeM |= hugeM;
	outln("after hugeM2");
	outln("hugeM2 ss: " << hugeM2.toShortString());
	outln("hugeM2.tiler addr: " << &(hugeM2.tiler));
	outln("hugeM2.tiler.m_m " << hugeM2.tiler.m_m  << ", hugeM2.tiler.m_n " << hugeM2.tiler.m_n);
	outln("hugeM2.elements@20");
	printColoArray(hugeM2.elements, 20);

	CuMatrix<T> hugeMbc2 = hugeM /= hugeM;
	outln("hugeMbc2: " << hugeMbc2.toShortString());
	//hugeMbc2.tiler.allocTiles(0);
	//hugeM2.tiler.allocTiles(0);
/*
	outln("hugeM2.elements@ last 20");
	printColoArray(hugeM2.elements + hugeM2.size/sizeof(T) - 21, 20);
*/
	T hugeM2sum = hugeM2.sum();
	outln("hugeM2sum " << hugeM2sum);
	assert(hugeM2sum == 2 * hugeMsum);
	T hugeMbc2sum = hugeMbc2.sum();
	outln("hugeMbc2sum " << hugeMbc2sum);
	assert(hugeM2sum == hugeMbc2sum);
	outln("freed " << hugeM2.releaseBuffers( ) << " buffers for hugeM2");
	outln("freed " << hugeM.releaseBuffers( ) << " buffers for hugeM");

	CuMatrix<T> bigIV = CuMatrix<T>::identity(10000);
	outln("bigIV\n" << bigIV);
	outln("bigIV.sum " << bigIV.sum());

	return 0;
}

template int testTinyXRFill<float>::operator()(int argc, const char **argv) const;
template int testTinyXRFill<double>::operator()(int argc, const char **argv) const;
template int testTinyXRFill<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testTinyXRFill<T>::operator()(int argc, const char **argv) const {
	outln("testTinyXRFill start currDev " << ExecCaps::currDev());

	// the mats are too big for the sums to be accurate enough to pass the asserts below
	assert(sizeof(T) > 4);

	ExecCaps* pcaps = ExecCaps::currCaps();
	ulong thugeMrows = 200; // pcaps->maxReasonable(0.000075)/sizeof(T); // this will force tiling
	ulong tsquareDim = sqrt(thugeMrows);
	outln("testTinyXRFill thugeMrows " << thugeMrows << " tsquareDim " << tsquareDim);
	CuMatrix<T> thugeM = 1.0 * CuMatrix<T>::identity(2048);
	diagonalFiller<T> filler = Functory<T, diagonalFiller>::pinch((T)6,2048);

	outln("testTinyXRFill freeing the old dbuff");
	thugeM.getMgr().freeTiles(thugeM);
	thugeM.tiler.gpuMask = Tiler<T>::gpuMaskFromCurrGpu();
	outln("testTinyXRFill allocing the new dbuff");
	thugeM.tiler.allocTiles(512,2048);
	CuMatrix<T>::fillFn(filler, thugeM);

	prlocf("thugeM first 20 elemtns ");
	printColoArray(thugeM.elements + 2048 * 2048-401, 400);
//	setCurrGpuDebugFlags( debugVerbose,true,false);
	outln("thugeM " << thugeM);
	outln("thugeM.sum() " << thugeM.sum());

	return 0;
}

template int testMemsetFill<float>::operator()(int argc, const char **argv) const;
template int testMemsetFill<double>::operator()(int argc, const char **argv) const;
template int testMemsetFill<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testMemsetFill<T>::operator()(int argc, const char **argv) const {
	outln("testMemsetFill start currDev " << ExecCaps::currDev());

	// the mats are too big for the sums to be accurate enough to pass the asserts below
	assert(sizeof(T) > 4);

	CuTimer timer;
	const int count = b_util::getCount(argc,argv,1000);

	timer.start();
	for(int i = 0; i  < count; i++ ) {
		CuMatrix<T> wunz = CuMatrix<T>::ones(1024,1024);
	}
	outln("testMemsetFill " << count << " ones took " << timer.stop());

	CuMatrix<float> sffm = CuMatrix<float>::sfill(6.5, 1024,1024);
	T sffmsum = sffm.sum();
	outln("sffm.sum " << sffmsum << " delta " << 1024*1024*6.5 - sffmsum  );
	outln("sffm " << sffm.syncBuffers());


	CuMatrix<T> sfm = CuMatrix<T>::sfill(6.5, 1024,1024);
	T sfmsum = sfm.sum();
	outln("sfm.sum " << sfmsum << " delta " << 1024*1024*6.5 - sfmsum  );
	outln("sfm " << sfm.syncBuffers());


	timer.start();
	double t = 6.5;
	int i1, i2;
	int* pint = (int*)&t;
	i1 =  *pint++;
	i2 =  *pint;
	outln("i1 " << i1 << " i2 " << i2);
	for(int i = 0; i  < count; i++ ) {
		CuMatrix<T> wunz = CuMatrix<T>::sfill(t, 1024,1024);
	}
	outln("testMemsetFill " << count << " sfill took " << timer.stop());


	return 0;
}
