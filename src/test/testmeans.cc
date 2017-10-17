/*
 * testMeans.cc
 *
 *  Created on: May 31, 2014
 *      Author: reid
 */
#include "../CuMatrix.h"
#include "../util.h"
#include "../Kmeans.h"

#include "tests.h"


template int testFeatureMeans<float>::operator()(int argc, const char **argv) const;
template int testFeatureMeans<double>::operator()(int argc, const char **argv) const;
template int testFeatureMeans<ulong>::operator()(int argc, const char **argv) const;
template<typename T> int testFeatureMeans<T>::operator()(int argc, const char **argv) const {
	outln("testFeatureMeans start");
	int count = b_util::getCount(argc,argv,100);
	CuMatrix<T> x = CuMatrix<T>::increasingColumns(5000, 1000,100) + 500;
	//outln("x " << x.syncBuffers());
	CuMatrix<T> tx= x.transpose();
	CuMatrix<T> means = CuMatrix<T>::zeros(x.n,1);
	DMatrix<T> d_Means, d_X, d_tX;
	x.tile0(d_X,true);
	tx.tile0(d_tX,true);
	means.tile0(d_Means,true);

	CuTimer timer;
	flprintf("%d feature means the old fashioned way\n", count);
	timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::featureAvgKernelL(d_Means, d_X, true);
	}
	flprintf("%d took %fms\n", count, timer.stop());
	outln("means " << means.syncBuffers());

	means.zero();
	flprintf("%d feature means the nu way\n", count);
	timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::featureAvgTxdKernelL(d_Means, d_tX);
	}
	flprintf("%d took %fms\n", count, timer.stop());
	outln("means " << means.syncBuffers());
	return 0;
}

template int testMeansLoop<float>::operator()(int argc, const char **argv) const;
template int testMeansLoop<double>::operator()(int argc, const char **argv) const;
template<typename T> int testMeansLoop<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> bign = CuMatrix<T>::increasingColumns(1000,1000,0);
	CuMatrix<T> txBign = bign.transpose();
	CuMatrix<T> bignMeans(1, bign.n,false,true);
	CuMatrix<T> bignMeansSum = CuMatrix<T>::zeros(1,bign.n);
	CuMatrix<T> txBignMeans(1, bign.n,false,true);
	CuMatrix<T> txBignMeansSum = CuMatrix<T>::zeros(1,bign.n);
	int count = b_util::getCount(argc,argv,1);
	CuTimer timer;
	float exeTime;

	timer.start();
	for (int i = 0; i < count; i++) {
		bign.featureMeansStreams(bignMeans,true,10);
	}
	exeTime = timer.stop();
	outln("featureMeansStreams count " << count << " took " << exeTime << " ms; bignMeans.sum() " << bignMeans.sum());

	timer.start();
	outln("bign\n"<<bign.syncBuffers());
	for (int i = 0; i < count; i++) {
		txBign.featureMeansTx(txBignMeans);
	}

	exeTime = timer.stop();
	outln("count " << count << " took " << exeTime << " ms; txBignMeans.sum() " << txBignMeans.sum());
	timer.start();
	for (int i = 0; i < count; i++) {
		bign.featureMeans(bignMeans,true);
	}
	exeTime = timer.stop();
	outln("count " << count << " took " << exeTime << " ms; bignMeans.sum() " << bignMeans.sum());

	return 0;
}


template int testMeansFile<float>::operator()(int argc, const char **argv) const;
template int testMeansFile<double>::operator()(int argc, const char **argv) const;
template int testMeansFile<ulong>::operator()(int argc, const char **argv) const;
template<typename T> int testMeansFile<T>::operator()(int argc, const char **argv) const {
	outln( "opening " << ANOMDET_SAMPLES_FILE);
	map<string, CuMatrix<T>*> results = CuMatrix<T>::parseOctaveDataFile(ANOMDET_SAMPLES_FILE,false, true);

	if(!results.size()) {
		outln("no " << ANOMDET_SAMPLES_FILE << "; exiting");
		return -1;
	}

	outln("loaded " << ANOMDET_SAMPLES_FILE);

	T fac20 = CuMatrix<T>::factorial(20);
	outln("fac20 " << fac20);
	outln("diff " << fac20-2.432902023e+18);
	if(sizeof(T) > 4) {
		dassert(fac20 == 2.43290200817664e+18);
	}else {
		dassert(util<T>::almostEquals(fac20,2.4329e+18,163674624) );
	}



	CuMatrix<T> seq =  CuMatrix<T>::sequence(1,1,10).extrude(5); // 1 row sequence + 5 row copies
	seq = seq /= CuMatrix<T>::ones(1,10); // plus row of ones

	outln("created seq " << seq.toShortString());
	T seqProd = seq.prod();
	outln("seqProd " << seqProd);
	seq.syncBuffers() ;
	outln("seq.syncBuffers()");
	outln("seq " << seq << ", prod " << seqProd);

	CuMatrix<T> rowSm = seq.rowSum();
	outln("rowSm " << rowSm.syncBuffers());

	CuMatrix<T> chunk = CuMatrix<T>::ones(500,5);
	CuMatrix<T> mosaic;
	for(int i =1; i < 10; i++) {
		mosaic = mosaic.rightConcatenate( chunk * i);
	}
	outln("mosaic " << mosaic.syncBuffers());
	CuMatrix<T> mosaicLong(mosaic);
	for(int i =1; i < 10; i++) {
		mosaicLong = mosaicLong.bottomConcatenate( mosaic * i);
	}
	outln("mosaicLong " << mosaicLong.toShortString());
	outln("mosaicLong " << mosaicLong.syncBuffers());


	outln("rowSm " << rowSm.syncBuffers());
	CuMatrix<T>& x = *results["X"];
	CuMatrix<T> tx = x.transpose();
	x.syncBuffers();
	outln("x " << x);
	outln("tx " << tx.syncBuffers());
	CuMatrix<T> xb = x.copy(true);
	for(T i = 2 ; i < 10; i ++) {
		xb = xb |= (i * x);
	}
	xb.syncBuffers();
	CuMatrix<T> txxb = xb.transpose();
	outln("xb " << xb);
	outln("txxb " << txxb.syncBuffers());
	outln("load x of " << x.m << "x" << x.n);
	CuMatrix<T>& xval = *results["Xval"];
	outln("load xval of " << xval.m << "x" << xval.n);
	CuMatrix<T>& xr = x;
	CuMatrix<T>& txr = tx;
	CuMatrix<T>& yval = *results["yval"];
	yval.syncBuffers();
	outln("got yval " << yval);

	CuMatrix<T> col = xr.columnMatrix(0);
	CuMatrix<T> col2 = xb.columnMatrix(21);
	outln("col2 " << col2.syncBuffers());
	outln("col.sum() " << col.sum());
	outln("col2.sum() " << col2.sum());
	CuMatrix<T> fmeansf2(1,x.n,false,true);
	CuMatrix<T> fmeanst2 = CuMatrix<T>::zeros(1,x.n);
	CuMatrix<T> bfmeans2(1,xb.n,false,true);
	CuMatrix<T> fmeansf(1,x.n,false,true),fmeanst(1,x.n,false,true);
	CuMatrix<T> bfmeans(1,xb.n,false,true);
	txr.featureMeansTx(fmeanst2);
	xr.featureMeans(fmeanst,true);
	outln("fmeanst " << fmeanst.syncBuffers());
	outln("fmeanst2 " << fmeanst2.syncBuffers());

	txxb.featureMeansTx(bfmeans2);
	xb.featureMeans(bfmeans,true);
	outln("bfmeans " << bfmeans.syncBuffers());
	outln("bfmeans2 " << bfmeans2.syncBuffers());
	T bdiff = bfmeans2.sumSqrDiff(bfmeans);
	outln("bfmeans2.sumSqrDiff(bfmeans) " << bdiff);
	assert(bdiff < util<T>::epsilon());

	util<CuMatrix<T> >::deletePtrMap(results);

	return 0;
}


template int testKmeans<float>::operator()(int argc, const char **argv) const;
template int testKmeans<double>::operator()(int argc, const char **argv) const;
template int testKmeans<ulong>::operator()(int argc, const char **argv) const;
template<typename T> int testKmeans<T>::operator()(int argc, const char **argv) const {
	outln("testKmeans start");
	checkCudaErrors(cudaGetLastError());
	outln( "opening " << KMEANS_FILE);
	map<string, CuMatrix<T>*> results = CuMatrix<T>::parseOctaveDataFile(KMEANS_FILE,false, true);
	checkCudaError(cudaGetLastError());

	int count = b_util::getCount(argc,argv,10);

	if(!results.size()) {
		outln("no " << KMEANS_FILE << "; exiting");
		return -1;
	}

	outln("\n\nloaded file: " << KMEANS_FILE);

	CuMatrix<T>& x = *results["X"];
	x.syncBuffers();
	DMatrix<T> dm;
	x.tile0(dm,true);
	x.lastMod = mod_synced;

	outln("x.lastMod " << b_util::modStr(x.lastMod));
	outln("x.tiler.hasDmemQ() " << x.tiler.hasDmemQ());
	assert(x.gpuReadyQ());
	CuMatrix<T> wx = x |= x |= x;

	setCurrGpuDebugFlags( debugMeans,true,false);
	CuMatrix<T> orgmeans = x.featureMeans(true);
	setCurrGpuDebugFlags( ~debugMeans,false,true);
	outln("\n\nx\n"<< x.syncBuffers());
	outln("orgmeans\n"<< orgmeans.syncBuffers());
	outln("wx\n"<< wx.syncBuffers());
	//int k = 3;
	//T spread = 10;
	T arry[] = {(T)3., (T)3., (T)6., (T)2., (T)8., (T)5.};
	CuMatrix<T> means(arry, 3,2,2, true);
	checkCudaError(cudaGetLastError());
	outln("means\n"<< means.syncBuffers());
	CuMatrix<T> wmeans = means |= means |= means;
	outln("wmeans\n"<< wmeans.syncBuffers());
	checkCudaError(cudaGetLastError());
	//CuMatrix<T> means = CuMatrix<T>::randn(k,x.n,spread);
	CuMatrix<T> nuMeans = CuMatrix<T>::zeros(3,2);
	outln("nuMeans " << nuMeans);
	checkCudaError(cudaGetLastError());
	CuMatrix<T> wNuMeans = nuMeans |= nuMeans|= nuMeans;

	cherr(cudaPeekAtLastError());
	outln("initial means\n" << means.syncBuffers());
/*
	srand (time(null));
	for(int i = 0; i < k * x.n; i++) {
		means.elements[i]=  spread * (RAND_MAX - rand())/(1.0 *RAND_MAX);
	}
	means.invalidateDevice();
	means.syncBuffers();
	outln("revised means\n" << means.syncBuffers());
*/
	uint indx[x.m];
	uint windx[x.m];
	IndexArray indices(indx, x.m, false);
	IndexArray windices(windx, x.m, false);
	T delta = util<T>::maxValue();
	T wdelta = util<T>::maxValue();
	int iterations;
	cherr(cudaPeekAtLastError());
	for(iterations = 0; iterations < count && wdelta > util<T>::epsilon(); iterations++ ) {

		cherr(cudaPeekAtLastError());
		outln("finding closet indices for means " << means.syncBuffers());
		Kmeans<T>::findClosest(indices, means, x);
		cherr(cudaPeekAtLastError());
		outln("distortion " << Kmeans<T>::distortion(indices, means, x));
		cherr(cudaPeekAtLastError());
		outln("\n\n\n\n\n\n");
		outln("finding closet indices for wmeans " << wmeans.syncBuffers());
		Kmeans<T>::findClosest(windices, wmeans, wx);
		outln("found closest");

		cherr(cudaPeekAtLastError());

		CuMatrix<T> mind = CuMatrix<T>::fromBuffer(indices.indices, sizeof(uint), toT<T>::fromUint, x.m, 1,1);
		outln("mind " << mind.toShortString());
		CuMatrix<T> mwind = CuMatrix<T>::fromBuffer(windices.indices, sizeof(uint), toT<T>::fromUint, x.m, 1,1);
		outln("mwind " << mwind.toShortString());
		CuMatrix<T> compIndices =  mind |= mwind;
		outln("compIndices " << compIndices.toShortString());
		CuMatrix<T> diffInd = mind-mwind;
		outln("diffInd " << diffInd.toShortString());


		setCurrGpuDebugFlags( debugVerbose,true,false);
		outln("compIndices " << compIndices.syncBuffers());
		outln("diffInd" << diffInd.syncBuffers());
		setCurrGpuDebugFlags( ~debugVerbose,false,true);

		Kmeans<T>::calcMeansColThread(indices, nuMeans, x);
		Kmeans<T>::calcMeansColThread(windices, wNuMeans, wx);
		delta = means.sumSqrDiff(nuMeans);
		wdelta = wmeans.sumSqrDiff(wNuMeans);
		outln("diff " << delta );
		outln("wdiff " << wdelta );

		nuMeans.copy(means);
		wNuMeans.copy(wmeans);
		means.invalidateHost();
		wmeans.invalidateHost();
		nuMeans.zero();
		wNuMeans.zero();
		//outln("after zero, nuMeans " << nuMeans.syncBuffers());
	}

	if(iterations < count) {
		outln("converged after " << iterations << " to within " << delta);
	}

	util<CuMatrix<T> >::deletePtrMap(results);

	return 0;
}


template int testMeansPitch<float>::operator()(int argc, const char **argv) const;
template int testMeansPitch<double>::operator()(int argc, const char **argv) const;
template int testMeansPitch<ulong>::operator()(int argc, const char **argv) const;
template<typename T> int testMeansPitch<T>::operator()(int argc, const char **argv) const {
	outln("testMeansPitch start");

	CuMatrix<T> src = CuMatrix<T>::increasingColumns(1000,1000,1);
	printArray( src.currBuffer(), 100);

	//outln("src.sum " << src.sum());

	CuMatrix<T> col;
	col.m = src.m;
	col.n = 1;
	col.p = src.p;
	col.size = col.m * col.p * sizeof(T);
	col.ownsDBuffers = col.ownsHBuffers = false;
	col.tiler.set( src.tiler.buffers);
	col.elements = src.elements;
	col.tiler.reset(col);
	//col.lastMod = mod_synced;
	col.invalidateHost();
	col.syncBuffers();

	outln("col ss " << col.toShortString());

	outln("col  " << col);
	T colKahan = col.kahanSum();
	outln("col kahanSum()  " << colKahan );
	T colSum = col.sum();
	outln("col.sum " << colSum);

	assert(colSum == colKahan);

	CuMatrix<T> col2 = src.columnMatrix(2);

	outln("col2.sum " << col2.sum());
}


#include "tests.cc"

