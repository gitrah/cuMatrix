/*
 * testanom.cc
 *
 *  Created on: Sep 6, 2012
 *      Author: reid
 */


#include "../Matrix.h"
#include "../util.h"
#include "../AnomalyDetection.h"
#include "tests.h"
const char* ANOMDET_SAMPLES_FILE = "ex8data1.txt";

template int testAnomDet<float>::operator()(int argc, char const ** args) const;
template int testAnomDet<double>::operator()(int argc, char const ** args) const;
template<typename T> int testAnomDet<T>::operator()(int argc, const char** args) const {
	outln( "opening " << ANOMDET_SAMPLES_FILE);
	map<string, Matrix<T>*> results = util<T>::parseOctaveDataFile(ANOMDET_SAMPLES_FILE,false, true);

	if(!results.size()) {
		outln("no " << ANOMDET_SAMPLES_FILE << "; exiting");
		return -1;
	}
	typedef typename map<string, Matrix<T>*>::iterator iterator;
	iterator it;
	it = results.begin();

	Matrix<T> seq = Matrix<T>::sequence(1,1,10);
	T seqProd = seq.prod();
	outln("seq " << seq.syncBuffers() << ", prod " << seqProd);
	Matrix<T> rowSm = seq.rowSum();
	outln("rowSm " << rowSm.syncBuffers());
	Matrix<T>& x = *results["X"];
	x.syncBuffers();
	Matrix<T> xb = x.copy(true);
	for(T i = 1.0 ; i < 2; i += 0.1) {
		xb = xb |= (i * x);
	}
	xb.syncBuffers();
	outln("load x of " << x.m << "x" << x.n);
	Matrix<T>& xval = *results["Xval"];
	outln("load xval of " << xval.m << "x" << xval.n);

	Matrix<T>& yval = *results["yval"];
	yval.syncBuffers();
	outln("got yval " << yval);

	Matrix<T> fmeansf(1,x.n,false,true),fmeanst(1,x.n,false,true);
	Matrix<T> bfmeans(1,xb.n,false,true);
	x.featureMeans(fmeanst,true);
	xb.featureMeans(bfmeans,true);
	outln("fmeanst " << fmeanst.syncBuffers());
	x.featureMeans(fmeansf,false);
	outln("fmeansf " << fmeansf.syncBuffers());
	Matrix<T> means, sqrdSigmas;
	Matrix<T> bmeans, bsqrdSigmas;
	Matrix<T> bsSigma2(xb.n,1,false,true);

	outln("bfmeans " << bfmeans.syncBuffers());

	int count = b_util::getCount(argc,args,1);
	CuTimer timer;
	timer.start();
	for(int i =0; i < count;i++) {
		AnomalyDetection<T>::fitGaussians(means, sqrdSigmas, x);
	}
	float exeTime = timer.stop();
	outln(count << " fit gaussians  took " <<exeTime << " flow of " << x.flow(count,2,exeTime));
	outln("mus\n" << means.syncBuffers());
	outln("sigmas\n" << sqrdSigmas.syncBuffers());

	AnomalyDetection<T>::fitGaussians(bmeans, bsqrdSigmas, xb);
	outln("direct");
	xb.variance(bsSigma2, bfmeans);
	outln("bmus\n" << bmeans.syncBuffers());
	outln("bsigmas\n" << bsqrdSigmas.syncBuffers());
	outln("bsSigma2\n" << bsSigma2.syncBuffers());

	Matrix<T> pdens(x.m,x.n,false,true);
	Matrix<T> bpdens(xb.m,xb.n,false,true);
	x.multivariateGaussianFeatures(pdens,sqrdSigmas,means);
	xb.multivariateGaussianFeatures(bpdens,bsqrdSigmas,bmeans);
	outln("pdens " << pdens.syncBuffers());
	outln("bpdens " << bpdens.syncBuffers());

	xval.syncBuffers();
	Matrix<T> pdensval(xval.m,xval.n,false,true);
	xval.multivariateGaussianFeatures(pdensval,sqrdSigmas,means);

	Matrix<T> pdensvalprod(xval.m, 1,false,true);
	pdensval.rowProductTx(pdensvalprod);
	outln("pdensvalprod ('row product via transpose ')" << pdensvalprod.syncBuffers());

	Matrix<T> pvalVector(xval.m, 1,false,true);
	pdensval.mvGaussianVectorFromFeatures(pvalVector);
	outln("pvalVector " << pvalVector.syncBuffers());

	Matrix<T> pvecDirect(xval.m, 1,false,true);
	xval.multivariateGaussianVector(pvecDirect,sqrdSigmas,means);
	outln("pvecDirect " << pvecDirect.syncBuffers());

	Matrix<T> bpVector(xb.m, 1,false,true);
	outln("sample prob from feature probs bpdens " << bpdens.toShortString());
	bpdens.mvGaussianVectorFromFeatures(bpVector);
	//Matrix<T>::verbose=true;
	outln("bpVector " << bpVector.syncBuffers());
	//Matrix<T>::verbose=false;

	Matrix<T> bpVectorByRowProductTx(xb.m, 1,false,true);
	bpdens.rowProductTx(bpVectorByRowProductTx);
	outln("bpVectorByRowProductTx " << bpVectorByRowProductTx.syncBuffers());

	Matrix<T> pvecByCovariance = x.multivariateGaussianVectorM(sqrdSigmas,means);
	outln("pvecByCovariance " << pvecByCovariance.syncBuffers());

	Matrix<T> pvecValByCovariance = xval.multivariateGaussianVectorM(sqrdSigmas,means);
	outln("pvecByCovariance " << pvecByCovariance.syncBuffers());

	pair<T,T> minMax = x.bounds();
	outln("bounds " << pp(minMax) << "...");

	pair<T,T> thresh = AnomalyDetection<T>::selectThreshold(yval, pvalVector);
	outln("(bestEpsilon, bestF1) : " << pp(thresh) << "...");

	util<Matrix<T> >::deleteMap(results);

	return 0;
}

#include "tests.cc"
