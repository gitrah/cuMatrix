/*
 * testanom.cc
 *
 *  Created on: Sep 6, 2012
 *      Author: reid
 */


#include "../CuMatrix.h"
#include "../util.h"
#include "../AnomalyDetection.h"

#include "tests.h"
const char* ANOMDET_SAMPLES_FILE = "ex8data1.txt";
const char* ANOMDET_SAMPLES2_FILE = "ex8data2.txt";
const char* REDWINE_SAMPLES_FILE = "winequality-red.csv";
const char* KMEANS_FILE = "ex7data2.txt";

template int testAnomDet<float>::operator()(int argc, const char **argv) const;
template int testAnomDet<double>::operator()(int argc, const char **argv) const;
template int testAnomDet<ulong>::operator()(int argc, const char **argv) const;
template<typename T> int testAnomDet<T>::operator()(int argc, const char **argv) const {
	outln( "opening " << ANOMDET_SAMPLES_FILE);
	CuTimer timer;
	timer.start();
	map<string, CuMatrix<T>*> results = CuMatrix<T>::parseOctaveDataFile(ANOMDET_SAMPLES_FILE,false, true);

	if(!results.size()) {
		outln("no " << ANOMDET_SAMPLES_FILE << "; exiting");
		return -1;
	}else {
		outln("parsing took " << fromMicros(timer.stop()));
	}

	outln("loaded " << ANOMDET_SAMPLES_FILE);

	outln("20! " << CuMatrix<T>::factorial(20));

	uint replicationFactor = 5097;
	CuMatrix<T> seq =  CuMatrix<T>::sequence(1,1,10);
	outln("created seq " << seq.toShortString());
	T seqProd = seq.prod();
	outln("seqProd " << seqProd);
	seq.syncBuffers() ;
	outln("seq.syncBuffers()");
	outln("seq " << seq << ", prod " << seqProd);
	CuMatrix<T> rowSm = seq.rowSum();
	outln("rowSm " << rowSm.syncBuffers());
	CuMatrix<T>& xR = *results["X"];
	xR.syncBuffers();
	outln("read xR " << xR);

	timer.start();
	CuMatrix<T>x = xR.replicateTiled(replicationFactor,1);
	outln("replicating x took " << fromMicros(timer.stop()));
	x.syncBuffers();
	outln("replicated x " << x);
	//CuMatrix<T> bigX = CuMatrix<T>::zeros(x.m * 256, x.n);

	/*CuMatrix<T> xb = x.copy(true);
	for(T i = 1.0 ; i < 2; i += 0.1) {
		xb = xb |= (i * x);
	}
	xb.syncBuffers();*/
	outln("load x of " << x.m << "x" << x.n);
	CuMatrix<T>& xvalR = *results["Xval"];
	CuMatrix<T>xval = xvalR.replicateTiled(replicationFactor,1);
	outln("load xval of " << xval.m << "x" << xval.n);
	//outln("xb of " << xb);

	CuMatrix<T>& yvalR = *results["yval"];
	CuMatrix<T>yval = yvalR.replicateTiled(replicationFactor,1);
	yval.syncBuffers();
	//outln("got yval " << yval);

	CuMatrix<T> fmeansf(1,x.n,true,true),fmeanst(1,x.n,true,true);
	CuMatrix<T> fmeansfR(1,xR.n,true,true),fmeanstR(1,xR.n,true,true);
	//CuMatrix<T> bfmeans(1,xb.n,false,true);
	timer.start();
	x.featureMeans(fmeanst,true);
	float exeTime = timer.stop();
	outln("fmeanst (took" << exeTime<<"): " << fmeanst.syncBuffers());
	xR.featureMeans(fmeanstR,true);
	outln("fmeanstR: " << fmeanstR.syncBuffers());
	//xb.featureMeans(bfmeans,true);

	timer.start();
	x.featureMeans(fmeansf,false);
	exeTime = timer.stop();
	outln("fmeansf (took" << exeTime<<"): " << fmeansf.syncBuffers());
	outln("getTypeName<T>() " << getTypeName<T>());
	outln(" util<float>::epsilon() " <<  util<float>::epsilon());
	outln(" util<T>::epsilon() " <<  util<T>::epsilon());
	assert(fmeanst.almostEq(fmeansf, util<T>::epsilon()));
	outln("passed assert(fmeanst.almostEq(fmeansf");
	CuMatrix<T> means, sqrdSigmas;
	//CuMatrix<T> bmeans, bsqrdSigmas;
	//CuMatrix<T> bsSigma2(xb.n,1,false,true);

	//outln("bfmeans " << bfmeans.syncBuffers());

	int count = b_util::getCount(argc,argv,1);
	outln("count " << count);
	timer.start();
	cherr(cudaPeekAtLastError());
	for(int i =0; i < count;i++) {
		AnomalyDetection<T>::fitGaussians(means, sqrdSigmas, x);
	}
	cherr(cudaPeekAtLastError());
	exeTime = timer.stop()/1000;
	outln(count << " fit gaussians  took " <<exeTime << " flow of " << x.flow(count,2,exeTime));
	outln("mus\n" << means.syncBuffers());
	outln("sigmas\n" << sqrdSigmas.syncBuffers());
	assert(util<T>::almostEquals(means.get(0), 14.1122,.001));
	assert(util<T>::almostEquals(means.get(1), 14.998,.001));
	assert(util<T>::almostEquals(sqrdSigmas.get(0), 1.8326,.0001));
	assert(util<T>::almostEquals(sqrdSigmas.get(1), 1.7097,.0001));
	/*AnomalyDetection<T>::fitGaussians(bmeans, bsqrdSigmas, xb);
	outln("direct");
	xb.variance(bsSigma2, bfmeans);
	outln("bmus\n" << bmeans.syncBuffers());
	outln("bsigmas\n" << bsqrdSigmas.syncBuffers());
	outln("bsSigma2\n" << bsSigma2.syncBuffers());
*/
	outln("after almosteq(s)");

	CuMatrix<T> pdens(x.m,x.n,true,true);
	CuMatrix<T> pvec(x.m,1,true,true);
	DMatrix<T> d_pdens = pdens.asDmatrix();
	DMatrix<T> d_pvec = pvec.asDmatrix();
	//CuMatrix<T> bpdens(xb.m,xb.n,false,true);
	outln("b sigmas " << sqrdSigmas.toShortString());
	x.multivariateGaussianFeatures(pdens,sqrdSigmas,means);
	outln("pdens " << pdens.syncBuffers());
	CuMatrix<T>::mvGaussianVectorFromFeatures(d_pvec, d_pdens);
	pvec.invalidateHost();
	outln("d_pvec " << pvec.syncBuffers());

	outln("a sigmas " << sqrdSigmas.toShortString());
	//xb.multivariateGaussianFeatures(bpdens,bsqrdSigmas,bmeans);
	//outln("pdens " << pdens.syncBuffers());
	//outln("pvec " << pvec.syncBuffers());
	//outln("bpdens " << bpdens.syncBuffers());

	xval.syncBuffers();
	CuMatrix<T> pdensval(xval.m,xval.n,true,true);
	xval.multivariateGaussianFeatures(pdensval,sqrdSigmas,means);
	outln("a2 sigmas " << sqrdSigmas.toShortString());

	CuMatrix<T> pvalVector(xval.m, 1,true,true);
	pdensval.mvGaussianVectorFromFeatures(pvalVector);
	outln("pvalVector " << pvalVector.syncBuffers());

	CuMatrix<T> pvecDirect(xval.m, 1,true,true);
	xval.multivariateGaussianVector(pvecDirect,sqrdSigmas,means);
	outln("a3 sigmas " << sqrdSigmas.toShortString());
	outln("pvecDirect " << pvecDirect.syncBuffers());
	CuMatrix<T> diff = pvalVector - pvecDirect;
	outln("diff " << diff.syncBuffers());
	outln("pvalVector.sumSqrDiff(pvecDirect) " <<pvalVector.sumSqrDiff(pvecDirect));
	assert(pvalVector.almostEq(pvecDirect));

	outln("pvalVector.min() " << pvalVector.min());
	outln("pvalVector.max() " << pvalVector.max());
	pair<T,T> bounds = pvalVector.bounds();
	outln("bounds " << bounds.first << ", " << bounds.second);

	timer.start();
	if( 1 == 0) {
		pair<T,T> thresh = AnomalyDetection<T>::selectThreshold(yval, pvalVector);
		outln("(bestEpsilon, bestF1) : " << pp(thresh) << "... and took " << fromMicros(timer.stop()));
		assert(util<T>::almostEquals(thresh.first, 8.99e-5, 1e-8));
		assert(util<T>::almostEquals(thresh.second, .875));
		checkCudaErrors(cudaDeviceSynchronize());
		timer.start();
		pair<T,T> threshOmp = AnomalyDetection<T>::selectThresholdOmp(yval, pvalVector);
		outln("Omp(bestEpsilon, bestF1) : " << pp(threshOmp) << "... and took " << fromMicros(timer.stop()));
		assert(util<T>::almostEquals(threshOmp.first, 8.99e-5, 1e-8));
		assert(util<T>::almostEquals(threshOmp.second, .875));
	}


	ltUnaryOp<T> lt =  Functory<T,ltUnaryOp>::pinch();
	lt.comp() = 8.99e-5; // thresh.first;

	//IndexArray outliers = pvec.find(lt);
	int outlierCount = 20;
	uint idxs[outlierCount];
	memset(idxs,0, outlierCount);
	IndexArray outliers(idxs,outlierCount,false);
	outln("init outliers " << outliers);
	//int outlier = pvec.count(lt);
	pvec.findFirstN(outliers,lt);
	outln("1st " << outlierCount <<" outliers " << outliers);
/*
	CuMatrix<T> bpVector(xb.m, 1,false,true);
	outln("sample prob from feature probs bpdens " << bpdens.toShortString());
	bpdens.mvGaussianVectorFromFeatures(bpVector);
	//debugVerbose=true;
	outln("bpVector " << bpVector.syncBuffers());
	//debugVerbose=false;
*/
	checkCudaErrors(cudaDeviceSynchronize());
	outln("b2 sigmas " << sqrdSigmas.toShortString());
	CuMatrix<T> pvecByCovariance = x.multivariateGaussianVectorM(sqrdSigmas,means);

	outln("pvecByCovariance " << pvecByCovariance.syncBuffers());

	CuMatrix<T> pvecValByCovariance = xval.multivariateGaussianVectorM(sqrdSigmas,means);
	outln("pvecValByCovariance " << pvecValByCovariance.syncBuffers());

	pair<T,T> minMax = x.bounds();
	outln("bounds " << pp(minMax) << "...");

	util<CuMatrix<T> >::deletePtrMap(results);

	return 0;
}

template int testMVAnomDet<float>::operator()(int argc, const char **argv) const;
template int testMVAnomDet<double>::operator()(int argc, const char **argv) const;
template<typename T> int testMVAnomDet<T>::operator()(int argc, const char **argv) const {
	outln( "opening " << ANOMDET_SAMPLES2_FILE);
	map<string, CuMatrix<T>*> results = CuMatrix<T>::parseOctaveDataFile(ANOMDET_SAMPLES2_FILE,false, true);
	checkCudaError(cudaGetLastError());

	if(!results.size()) {
		outln("no " << ANOMDET_SAMPLES2_FILE << "; exiting");
		return -1;
	}

	outln("loaded " << ANOMDET_SAMPLES2_FILE);


	CuMatrix<T>& x = *results["X"];
	x.syncBuffers();
	outln("load x of " << x.m << "x" << x.n);
	CuMatrix<T>& xval = *results["Xval"];
	outln("load xval of " << xval.m << "x" << xval.n);

	CuMatrix<T>& yval = *results["yval"];
	yval.syncBuffers();
	//outln("got yval " << yval);

	CuMatrix<T> fmeansf(1,x.n,false,true),fmeanst(1,x.n,false,true);
	x.featureMeans(fmeanst,true);
	outln("fmeanst " << fmeanst.syncBuffers());
	x.featureMeans(fmeansf,false);
	outln("fmeansf " << fmeansf.syncBuffers());
	CuMatrix<T> means, sqrdSigmas;
	CuMatrix<T> bmeans, bsqrdSigmas;

	checkCudaError(cudaGetLastError());
	int count = b_util::getCount(argc,argv,1);
	CuTimer timer;
	timer.start();
	for(int i =0; i < count;i++) {
		AnomalyDetection<T>::fitGaussians(means, sqrdSigmas, x);
	}
	checkCudaError(cudaGetLastError());
	float exeTime = timer.stop();
	outln(count << " fit gaussians  took " <<exeTime << " flow of " << x.flow(count,2,exeTime));
	outln("mus\n" << means.syncBuffers());
	outln("sqrdSigmas\n" << sqrdSigmas.syncBuffers());
	outln("sqrdSigmas " << sqrdSigmas.toShortString());

	CuMatrix<T> p;
	timer.start();
	for(int i =0; i < count;i++) {
		flprintf("%dth call to multivariateProbDensity\n", i);
		AnomalyDetection<T>::multivariateProbDensity(p,x,means,sqrdSigmas);
	}
	checkCudaError(cudaGetLastError());
	exeTime = timer.stop();
	outln(count << " multivariateProbDensity took " <<exeTime );
	CuMatrix<T> p2;
	timer.start();
	for(int i =0; i < count;i++) {
		AnomalyDetection<T>::multivariateProbDensity2(p2,x,means,sqrdSigmas);
	}
	checkCudaError(cudaGetLastError());
	exeTime = timer.stop();
	outln(count << " multivariateProbDensity2 took " <<exeTime );
	outln("prob density p2\n" << p2.syncBuffers());
	//p.transpose();

	xval.syncBuffers();

	CuMatrix<T> pdensval = AnomalyDetection<T>::multivariateProbDensity(xval,means,sqrdSigmas).transpose();
	checkCudaError(cudaGetLastError());
	outln("pdensval " << pdensval.syncBuffers());

	pair<T,T> thresh0 = AnomalyDetection<T>::selectThreshold(yval, pdensval);
	outln("(bestEpsilon, bestF1) : " << pp(thresh0) << "...");
	CuMatrix<T> predictions = p < thresh0.first;
	outln("predictions.sum " << predictions.sum());


	CuMatrix<T> pvalVector(xval.m, 1,false,true);
	pdensval.mvGaussianVectorFromFeatures(pvalVector);
	outln("pvalVector " << pvalVector.syncBuffers());
	checkCudaError(cudaGetLastError());

	CuMatrix<T> pvecDirect(xval.m, 1,false,true);
	xval.multivariateGaussianVector(pvecDirect,sqrdSigmas,means);
	//checkCudaError(cudaGetLastError());
	outln("a3 sigmas " << sqrdSigmas.toShortString());
	outln("pvecDirect " << pvecDirect.syncBuffers());
	checkCudaError(cudaGetLastError());

/*
	CuMatrix<T> pvecByCovariance = x.multivariateGaussianVectorM(sqrdSigmas,means);
	outln("pvecByCovariance " << pvecByCovariance.syncBuffers());

	CuMatrix<T> pvecValByCovariance = xval.multivariateGaussianVectorM(sqrdSigmas,means);
	outln("pvecValByCovariance " << pvecValByCovariance.syncBuffers());

	pair<T,T> minMax = x.bounds();
	outln("bounds " << pp(minMax) << "...");
*/

	const T bestEpsilon_MV = 1.377228890761358e-18;
	const T bestF1_MV = 0.615384615384615;
	pair<T,T> thresh = AnomalyDetection<T>::selectThreshold(yval, pvalVector);
	outln("(bestEpsilon, bestF1) : " << pp(thresh) << "...");
	outln("eps : " << util<T>::epsilon());
	outln("epsf : " << util<float>::epsilon());
	outln("(T)fabs(thresh.first - bestEpsilon_MV) : " << (T)fabs(thresh.first - bestEpsilon_MV));
	outln("(T)fabs(thresh.second - bestF1_MV) : " << (T)fabs(thresh.second - bestF1_MV));
	dassert((T)fabs(thresh.first - bestEpsilon_MV) < util<T>::epsilon());
	dassert((T)fabs(thresh.second - bestF1_MV) < util<T>::epsilon());

	util<CuMatrix<T> >::deletePtrMap(results);

	return 0;
}


template int testRedWineScSv<float>::operator()(int argc, const char **argv) const;
template int testRedWineScSv<double>::operator()(int argc, const char **argv) const;
template<typename T> int testRedWineScSv<T>::operator()(int argc, const char **argv) const {
	outln( "opening " << REDWINE_SAMPLES_FILE);
	map<string, CuMatrix<T>*> results = CuMatrix<T>::parseCsvDataFile(REDWINE_SAMPLES_FILE,";",false, true,true);

	if(!results.size()) {
		outln("no " << REDWINE_SAMPLES_FILE << "; exiting");
		return -1;
	}

	outln("loaded " << REDWINE_SAMPLES_FILE);

	CuMatrix<T>& xAndY = *results["xAndY"];
	outln("xAndY " << xAndY.toShortString());
	xAndY.syncBuffers();
	CuMatrix<T>& rxy = xAndY;
	CuMatrix<T> x = rxy.dropLast(true);
	CuMatrix<T> y = rxy.columnMatrix(rxy.n-1);
	outln("x\n" << x.syncBuffers());
	outln("y\n" << y.syncBuffers());
	util<CuMatrix<T> >::deletePtrMap(results);

	return 0;
}

template int testCsv<float>::operator()(int argc, const char **argv) const;
template int testCsv<double>::operator()(int argc, const char **argv) const;
template<typename T> int testCsv<T>::operator()(int argc, const char **argv) const {
	string path = b_util::getPath(argc,argv,REDWINE_SAMPLES_FILE);
	outln( "opening " << path);
	map<string, CuMatrix<T>*> results = CuMatrix<T>::parseCsvDataFile(path.c_str(),",",false, true, false);

	if(!results.size()) {
		outln("no " << path << "; exiting");
		return -1;
	}

	outln("loaded " << path);
	CuMatrix<T>& x = *results["x"];
	outln("x " << x.toShortString());
	CuMatrix<T>means(1,x.n,true,true);
	CuTimer timer;
	timer.start();
	x.featureMeans(means,false);
	outln("mean calc took " << timer.stop()/1000 << "s, means: " << means.syncBuffers());
	util<CuMatrix<T> >::deletePtrMap(results);

	return 0;
}


#include "tests.cc"
