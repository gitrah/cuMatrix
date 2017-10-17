#include "tests.h"
#include "testKernels.h"
#include "../CuMatrix.h"
#include "../util.h"
#include "../caps.h"
#include "../Kernels.h"
#include "../CuSet.h"

template<typename T> __global__ void attributeCountsKernelShared(DMatrix<T> d_atts,
		DMatrix<int> d_counts, DMatrix<int> d_depths, DMatrix<int> d_distances,
		const DMatrix<T> d_x );

extern bool testAttFreqsDev1;

template int testAttFreqsLarge<float>::operator()(int argc, const char **argv) const;
template int testAttFreqsLarge<double>::operator()(int argc, const char **argv) const;
template int testAttFreqsLarge<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testAttFreqsLarge<T>::operator()(int argc, const char **argv) const {
	if(testAttFreqsDev1)
		ExecCaps_setDevice(1);

	int maks = Tiler<T>::gpuMaskFromCount(ExecCaps::countGpus());
	outln("mask " << Tiler<T>::maskStr(maks));

	//b_util::allDevices([]() { setCurrGpuDebugFlags(debugTiler | debugFill,false,false,0 );});

	TilerOn(); FillOn();
	dbgStr();

	CuMatrix<T> tEv= CuMatrix<T>::evens(100, 100, 0, maks, tdCols).syncBuffers();
	CuMatrix<T> tOd = CuMatrix<T>::odds(100, 100, 0, maks, tdCols).syncBuffers();
	outln("tEv  " << tEv);
	outln("tOd  " << tOd);
	CuMatrix<T> merj = CuMatrix<T>::zeros(200, 100, maks, tdCols).syncBuffers();
	int merjInt = CuSet<T>::mergeSorted(merj.elements, tOd.elements, tEv.elements, 100*100, 100*100);

	CuMatrix<T> ev= CuMatrix<T>::evens(4000, 1000, 0, maks, tdCols);
	CuMatrix<T> evens = ev |= ev;
	evens = evens |= evens;
	outln("evens  " << evens.syncBuffers());
	outln("evens.sum  " << evens.sum());
	outln("ev.sum  " << ev.sum());
	CuMatrix<T> odds = CuMatrix<T>::odds(4000, 4000, 0, maks, tdCols);
	outln("odds  " << odds.syncBuffers());
	CuMatrix<T> icols = CuMatrix<T>::seqMod(0, 15,10000, 10000,maks, tdCols);
	outln("icols.toShortString()  " << icols.toShortString());
	//icols.syncBuffers();
	outln("genning irows... ");
	CuMatrix<T> irows = CuMatrix<T>::seqMod(0, 50, 10000,10000,maks, tdCols);
	TilerOff(); FillOff();
	outln("irows.toShortString()  " << irows.toShortString());
	FlagsOn(debugFill | debugTiler | debugBinOp);
	CuMatrix<T> sumb = icols + irows;
	FlagsOff(debugFill | debugTiler | debugBinOp);
	sumb.syncBuffers();
	outln("sumb  " << sumb);
	CuMatrix<T> col0 = sumb.columnMatrix(0);
	outln("col0  " << col0.syncBuffers());
		//sumb.syncBuffers();
	CuTimer timer;
	outln("timer.start()");
	timer.start();
	cherr(cudaPeekAtLastError());

	col0.tiler.kernelle = (void(*)()) (&attributeCountsKernelShared<T>);
	col0.syncBuffers();

	CuMatrix<T> col0freqs = col0.attributeFrequencies();

/*

	decltype(&attributeCountsKernelShared<T>) p1 =  &attributeCountsKernelShared<T>;
	sumb.tiler.kernelle = (void(*)()) p1;
	// or this
*/
	sumb.tiler.kernelle = (void(*)()) (&attributeCountsKernelShared<T>);
	sumb.syncBuffers();

	CuMatrix<T> sumbfreqs = sumb.attributeFrequencies();
	float attFreqsTimeS = timer.stop()/1000.0;

	outln("freqs " << sumbfreqs.syncBuffers());
	T allFreqs = sumbfreqs.sum();
	outln("freqsum "<< allFreqs);

}

#include "tests.cc"
