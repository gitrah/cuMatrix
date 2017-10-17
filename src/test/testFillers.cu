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


template int testFillVsLoadRnd<float>::operator()(int argc, const char **argv) const;
template int testFillVsLoadRnd<double>::operator()(int argc, const char **argv) const;
template int testFillVsLoadRnd<ulong>::operator()(int argc, const char **argv) const;
template <typename T>  int testFillVsLoadRnd<T>::operator()(int argc, const char **argv) const {
	char* path = "./data_rnd_1000x1000_eps50.txt";
	const int count = b_util::getCount(argc,argv,1000);
	CuTimer timer;
	timer.start();
	float ftime;

	CuMatrix<T> m0 = CuMatrix<T>::randn(1000,1000,50, false);
	ftime = timer.stop();
	outln("randn(1000,1000) took " << ftime/1000 << "s");

	timer.start();
	m0.syncBuffers();
	ftime = timer.stop();
	outln("m0.syncBuffers() took " << ftime/1000 << "s");

	timer.start();
	CuMatrix<T>::toOctaveFile(path, m0);
	ftime = timer.stop();
	outln("toOctaveFile(\""<< path << "\") took " << ftime/1000 << "s");

	timer.start();
/*
	map<string, CuMatrix<T>*>  octfiles = CuMatrix<T>::parseOctaveDataFile(path,true);
	ftime = timer.stop();
	outln("parseOctaveDataFile(\""<< path << "\") took " << ftime/1000 << "s");

	timer.start();
	CuMatrix<T>* pm1 = octfiles[path];
*/
	ftime = timer.stop();
	outln("octfiles[\""<< path << "\"] took " << ftime/1000 << "s");
	return 0;
}


template int testFillNsb<float>::operator()(int argc, const char **argv) const;
template int testFillNsb<double>::operator()(int argc, const char **argv) const;
template int testFillNsb<ulong>::operator()(int argc, const char **argv) const;
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

	outln("pnes sum\n" << pnes.sum());
	outln("mcos sum\n" << mcos.sum());
	outln("msin sum\n" << msin.sum());

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
	ulong hugeMelems = pcaps->maxReasonable( 1.5 * DEFAULT_GMEM_HEADROOM_FACTOR)/sizeof(T); // this will force tiling
	ulong squareDim = sqrt(hugeMelems);
	ulong sqrdDim = squareDim*squareDim;
	if(sqrdDim < hugeMelems) hugeMelems = sqrdDim;
	outln("testXRFill hugeMelems " << hugeMelems << " squareDim " << squareDim);

	TilerOn(); FillOn();
	CuMatrix<T> huge2M = CuMatrix<T>::fill(squareDim, squareDim,(T)2);
	CuMatrix<T> hugeM = 1.0 * CuMatrix<T>::ones(squareDim, squareDim);
	TilerOff();FillOff();

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
	outln( "count ne 1 hugeM " << util<T>::hCountNe2D(hugeM.elements, (T) 1, hugeM.p,  hugeM.n, hugeM.m));

	T huge2Msum = huge2M.sum();
	outln( "count ne 2 huge2M " << util<T>::hCountNe2D(huge2M.elements, (T) 2, huge2M.p,  huge2M.n, huge2M.m));
	outln("testXRFill hugeMsum " << hugeMsum);
	outln("testXRFill huge2Msum " << huge2Msum);
	assert(hugeMsum == squareDim * squareDim);
	assert(huge2Msum == 2*squareDim * squareDim);

	huge2M = CuMatrix<T>::ones(2,2);
	outln("\n\nattempting sevenHalves");
	CuMatrix<T> sevenHalvesV = hugeM + 2.5 * hugeM;
	T sevenHalvesVsum = sevenHalvesV.sum();
	outln("testXRFill sevenHalvesVsum " << sevenHalvesVsum);
	outln("testXRFill sevenHalvesVsum " << (3.5 * hugeMsum - sevenHalvesVsum));
	outln("testXRFill relDiff(3.5 * hugeMsum,sevenHalvesVsum) " << util<T>::relDiff(3.5 * hugeMsum,sevenHalvesVsum));
	assert( 3.5 * hugeMsum == sevenHalvesVsum);
	T hugeM2_0sum=0;
	T hugeM2_1sum=1;
	T hugeM2_0col0Sum;
	T hugeM2_0colNSum;
	T hugeM2_0row0Sum ;
	T hugeM2_0rowMSum;
	{
		CuMatrix<T> hugeM2_0 = hugeM |= sevenHalvesV;
		hugeM2_0sum = hugeM2_0.sum();
		outln("hugeM2_0sum " << hugeM2_0sum);
		assert(hugeM2_0sum == hugeMsum + sevenHalvesVsum);
		hugeM2_0col0Sum = hugeM2_0.columnSum(0);
		hugeM2_0colNSum = hugeM2_0.columnSum(hugeM2_0.n-1);
		outln("hugeM2_0col0Sum " << hugeM2_0col0Sum);
		outln("hugeM2_0colNSum " << hugeM2_0colNSum);
		assert(hugeM2_0col0Sum * 3.5 == hugeM2_0colNSum);
		hugeM2_0row0Sum= hugeM2_0.rowSum(0);
		hugeM2_0rowMSum = hugeM2_0.rowSum(hugeM2_0.m-1);
		outln("hugeM2_0row0Sum " << hugeM2_0row0Sum);
		outln("hugeM2_0rowMSum " << hugeM2_0rowMSum);
	}

	{
		CuMatrix<T> hugeM2_1 = hugeM /= sevenHalvesV;
		hugeM2_1sum= hugeM2_1.sum();
		outln("hugeM2_1sum " << hugeM2_1sum);
		assert(hugeM2_1sum == hugeMsum + sevenHalvesVsum);
		T hugeM2_1col0Sum = hugeM2_1.columnSum(0);
		T hugeM2_1colNSum = hugeM2_1.columnSum(hugeM2_1.n-1);
		outln("hugeM2_1col0Sum " << hugeM2_1col0Sum);
		outln("hugeM2_1colNSum " << hugeM2_1colNSum);
		assert(hugeM2_1col0Sum == hugeM2_1colNSum);

		assert(hugeM2_1col0Sum == hugeM2_0row0Sum);

		T hugeM2_1row0Sum = hugeM2_1.rowSum(0);
		T hugeM2_1rowMSum = hugeM2_1.rowSum(hugeM2_1.m-1);

		assert(hugeM2_1row0Sum == hugeM2_0col0Sum);
		assert(hugeM2_1rowMSum == hugeM2_0colNSum);


		outln("hugeM2_1row0Sum " << hugeM2_1row0Sum);
		outln("hugeM2_1rowMSum " << hugeM2_1rowMSum);
	}
	outln("freed " << sevenHalvesV.releaseBuffers( ) << " buffers for sevenHalvesV");

	outln("\n\n|= --> hugeM2\n\n");
	CuMatrix<T> hugeM2 = hugeM |= hugeM;
	outln("after hugeM2 " << hugeM2.toss());
	outln( "count ne 1 hugeM2 " << util<T>::hCountNe2D(hugeM2.elements, (T) 1, hugeM2.p,  hugeM2.n, hugeM2.m));
	T hugeM2sum = hugeM2.sum();
	outln("hugeM2sum " << hugeM2sum);
	CuMatrix<T> hugeMbc2 = hugeM /= hugeM;
	outln("hugeMbc2: " << hugeMbc2.toShortString());
	outln( "count ne 1 hugeMbc2 " << util<T>::hCountNe2D(hugeMbc2.elements, (T) 1, hugeMbc2.p,  hugeMbc2.n, hugeMbc2.m));
	T hugeMbc2sum = hugeMbc2.sum();
	outln("hugeMbc2sum " << hugeMbc2sum);

	TilerOn();
	T hugeM2col0Sum = hugeM2.columnSum(0);
	T hugeM2colNSum = hugeM2.columnSum(hugeM2.n-1);
	T hugeMcol0Sum = hugeM.columnSum(0);
	outln("hugeMcol0Sum " << hugeMcol0Sum);
	outln("hugeM2col0Sum " << hugeM2col0Sum);
	outln("hugeM2colNSum " << hugeM2colNSum);

	assert( hugeM2col0Sum == hugeM2.m);
	assert( hugeM2colNSum == hugeM2.m);
	assert( hugeMcol0Sum == hugeM2.m);

	T hugeM2row0Sum = hugeM2.rowSum(0);
	T hugeMrow0Sum = hugeM.rowSum(0);
	outln("hugeMrow0Sum " << hugeMrow0Sum);
	outln("hugeM2row0Sum " << hugeM2row0Sum);
	assert(hugeM2row0Sum == 2 * hugeMrow0Sum );


	outln("hugeM2 ss: " << hugeM2.toShortString());
	outln("hugeM2.tiler addr: " << &(hugeM2.tiler));
	outln("hugeM2.tiler.m_m " << hugeM2.tiler.m_m  << ", hugeM2.tiler.m_n " << hugeM2.tiler.m_n);
	outln("hugeM2.elements@20");
	printColoArray(hugeM2.elements, 20);

	//hugeMbc2.tiler.allocTiles(0);
	//hugeM2.tiler.allocTiles(0);
/*
	outln("hugeM2.elements@ last 20");
	printColoArray(hugeM2.elements + hugeM2.size/sizeof(T) - 21, 20);
*/
	assert(hugeM2sum == 2 * hugeMsum);
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
	int tileM= 512;
	int tileN= 2048;
	int tileP= 2048;
	thugeM.tiler.allocTiles(tileM,tileN,tileP);
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

	CuMatrix<float> sffm = CuMatrix<float>::sfill(1024,1024,6.5);
	T sffmsum = sffm.sum();
	outln("sffm.sum " << sffmsum << " delta " << 1024*1024*6.5 - sffmsum  );
	outln("sffm " << sffm.syncBuffers());

	CuMatrix<T> sfm = CuMatrix<T>::sfill(1024,1024,6.5);
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
		CuMatrix<T> wunz = CuMatrix<T>::sfill(1024,1024,t);
	}
	outln("testMemsetFill " << count << " sfill took " << timer.stop());

	return 0;
}
