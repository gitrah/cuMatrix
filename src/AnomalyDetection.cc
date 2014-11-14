/*
 *
 *  Created on: Aug 24, 2012
 *      Author: reid
 */

#include "CuMatrix.h"
#include "AnomalyDetection.h"
#include "LuDecomposition.h"
//  val twoPi = 2 * math.Pi
#include <math.h>
#include <utility>
#include <omp.h>
const double oneOverSqrt2Pi =  1. / sqrt(2. * Pi);
const double twoPi= 2 * Pi;

// sqrdSigma is a column matrix, by convention
template void AnomalyDetection<ulong>::fitGaussians( CuMatrix<ulong>& means, CuMatrix<ulong>& sqrdSigmas, const CuMatrix<ulong>& x);
template void AnomalyDetection<float>::fitGaussians( CuMatrix<float>& means, CuMatrix<float>& sqrdSigmas, const CuMatrix<float>& x);
template void AnomalyDetection<double>::fitGaussians( CuMatrix<double>& means, CuMatrix<double>& sqrdSigmas, const CuMatrix<double>& x);
template <typename T> void AnomalyDetection<T>::fitGaussians( CuMatrix<T>& means, CuMatrix<T>& sqrdSigmas, const CuMatrix<T>& x) {
	if(means.n != x.n) {
		if( means.d_elements) {
			outln("means matrix is wrong sized ");
			dthrow( matricesOfIncompatibleShape());
		}else {
			means.n = means.p = x.n;
			means.m = 1;
			means.updateSize();
			means.allocDevice();
		}
	}
	if(sqrdSigmas.m != x.n) {
		if( sqrdSigmas.d_elements) {
			outln("sqrdSigmas matrix is wrong sized ");
			dthrow(matricesOfIncompatibleShape());
		}else {
			sqrdSigmas.m =  x.n;
			sqrdSigmas.n = sqrdSigmas.p = 1;
			sqrdSigmas.updateSize();
			sqrdSigmas.allocDevice();
		}
	}
	if(checkDebug(debugAnomDet))outln("ad.fitGaussians " << sqrdSigmas.toShortString());
	x.fitGaussians(sqrdSigmas, means);
}

template  pair<ulong,ulong>  AnomalyDetection<ulong>::selectThreshold( const CuMatrix<ulong>& yValidation, const CuMatrix<ulong>& probabilityDensityValidation);
template  pair<float,float>  AnomalyDetection<float>::selectThreshold( const CuMatrix<float>& yValidation, const CuMatrix<float>& probabilityDensityValidation);
template  pair<double,double>  AnomalyDetection<double>::selectThreshold( const CuMatrix<double>& yValidation, const CuMatrix<double>& probabilityDensityValidation);
template <typename T> pair<T,T>  AnomalyDetection<T>::selectThreshold(const CuMatrix<T>& yValidation, const CuMatrix<T>& probabilityDensityValidation) {
	T bestEpsilon = 0, bestF1 = 0, f1 = 0;
	uint truePos = yValidation.sum();
	if(checkDebug(debugAnomDet))outln("yValidation truePos " << truePos);
	if(checkDebug(debugAnomDet))outln("yValidation " << yValidation.toShortString());
	if(checkDebug(debugAnomDet))outln("probabilityDensityValidation " << probabilityDensityValidation.toShortString());
	uint falsePos = 0, falseNeg = 0;
	T precision, recall;
	pair<T,T> prBounds = probabilityDensityValidation.bounds();
	if(checkDebug(debugAnomDet))outln("prBounds " << pp(prBounds));
	T stepSize = (prBounds.second - prBounds.first)/1000.0;
	for(T epsilon = prBounds.first + stepSize; epsilon <= prBounds.second; epsilon += stepSize) {
		CuMatrix<T> cvPredictions = probabilityDensityValidation < epsilon;
		truePos = (  (cvPredictions == 1.0) && (yValidation == 1)).sum();
		falsePos = (  (cvPredictions == 1.0) && (yValidation == 0)).sum();
		falseNeg = (  (cvPredictions == 0.0) && (yValidation == 1)).sum();
		precision = 1.0 * truePos / (truePos + falsePos);
		recall = 1.0 * truePos / (truePos + falseNeg);
		f1 = 2 * precision * recall / (precision + recall );
		if(checkDebug(debugAnomDet | debugVerbose))outln("for eps " << epsilon );
		if(checkDebug(debugAnomDet| debugVerbose))outln("\tgot truePos "<< truePos << ", falsePos " << falsePos << ", falseNeg " << falseNeg);
		if(checkDebug(debugAnomDet| debugVerbose))outln("\tgot precision "<< precision << ", recall " << recall << ", f1 " << f1);
		if(f1 > bestF1) {
			bestF1 = f1;
			bestEpsilon = epsilon;
		}
	}
	return pair<T,T>(bestEpsilon, bestF1);
}

template  pair<ulong,ulong>  AnomalyDetection<ulong>::selectThresholdOmp( const CuMatrix<ulong>& yValidation, const CuMatrix<ulong>& probabilityDensityValidation);
template  pair<float,float>  AnomalyDetection<float>::selectThresholdOmp( const CuMatrix<float>& yValidation, const CuMatrix<float>& probabilityDensityValidation);
template  pair<double,double>  AnomalyDetection<double>::selectThresholdOmp( const CuMatrix<double>& yValidation, const CuMatrix<double>& probabilityDensityValidation);
template <typename T> pair<T,T>  AnomalyDetection<T>::selectThresholdOmp(const CuMatrix<T>& yValidation, const CuMatrix<T>& probabilityDensityValidation) {
    int num_gpus = 0;   // number of CUDA GPUs
    T bestEpsilon = 0, bestF1 = 0, f1 = 0;
	uint truePos = yValidation.sum();
	checkCudaErrors(cudaGetDeviceCount(&num_gpus));
	if(checkDebug(debugAnomDet))outln("yValidation truePos " << truePos);
	if(checkDebug(debugAnomDet))outln("yValidation " << yValidation.toShortString());
	if(checkDebug(debugAnomDet))outln("probabilityDensityValidation " << probabilityDensityValidation.toShortString());
	uint falsePos = 0, falseNeg = 0;
	T precision, recall;
	pair<T,T> prBounds = probabilityDensityValidation.bounds();
    outln("prBounds " << pp(prBounds));
	T stepSize = (prBounds.second - prBounds.first)/1000.0;
	outln("stepSize " << stepSize);
    uint step = 0;
    uint tid;
    T epsilon = prBounds.first;
    uint nthreads=0;
    int dev = ExecCaps::currDev();
#pragma omp parallel private(tid, epsilon, step, nthreads)
    {
		tid = omp_get_thread_num();
		nthreads = omp_get_num_threads();
		if(checkDebug(debugAnomDet))printf("tid %d of %d\n", tid, nthreads);
		epsilon = prBounds.first + tid * stepSize;
		if(ExecCaps::currDev() != dev) {
			flprintf("thread %d lost current device %d, restoring...\n", tid,dev);
			checkCudaError(cudaSetDevice(dev));
		}
		while(epsilon <= prBounds.second) {
			if(checkDebug(debugAnomDet))printf("thread %d step %d from epsilon %f\n", tid, step, epsilon);
			CuMatrix<T> cvPredictions = probabilityDensityValidation < epsilon;
			truePos = (  (cvPredictions == 1.0) && (yValidation == 1)).sum();
			falsePos = (  (cvPredictions == 1.0) && (yValidation == 0)).sum();
			falseNeg = (  (cvPredictions == 0.0) && (yValidation == 1)).sum();
			precision = 1.0 * truePos / (truePos + falsePos);
			recall = 1.0 * truePos / (truePos + falseNeg);
			f1 = 2 * precision * recall / (precision + recall );
			if(checkDebug(debugAnomDet | debugVerbose))outln("thread " << tid << " for eps " << epsilon );
			if(checkDebug(debugAnomDet| debugVerbose))outln("\tgot truePos "<< truePos << ", falsePos " << falsePos << ", falseNeg " << falseNeg);
			if(checkDebug(debugAnomDet| debugVerbose))outln("\tgot precision "<< precision << ", recall " << recall << ", f1 " << f1);
			if(f1 > bestF1) {
				bestF1 = f1;
				bestEpsilon = epsilon;
			}
			step++;
			epsilon = prBounds.first + (step * nthreads + tid ) * stepSize;
		}
    }
	return pair<T,T>(bestEpsilon, bestF1);
}

/*
template <typename T> CuMatrix<T> AnomalyDetection<T>::multivariateGaussian(const CuMatrix<T>& x, const CuMatrix<T>& mu, const CuMatrix<T>& sigmaSqrd) {
	CuMatrix<T> s2 = sigmaSqrd.vectorQ() ? sigmaSqrd.diagonal() : sigmaSqrd;
	CuMatrix<T> xNormed = x.subMeans(mu);
}
*/

/**
  * @param x sample to be tested
  * @param params Gaussian/Normal distribution parameters, an array of feature means and the total variance
  * @param n feature count of x
  * @return probability sample is an anomaly
  *          n     1             (xj - μj)^2
  * p(x) = 	Π   -------- exp( - ---------   )
  *         j=1   √(2Piσ^2)           2σ^2
  */

template ulong AnomalyDetection<ulong>::probabilityDensityFunction( const ulong* x, const ulong* mus, const ulong* sigmas, uint n);
template float AnomalyDetection<float>::probabilityDensityFunction( const float* x, const float* mus, const float* sigmas, uint n);
template double AnomalyDetection<double>::probabilityDensityFunction( const double* x, const double* mus, const double* sigmas, uint n);
template <typename T> T AnomalyDetection<T>::probabilityDensityFunction( const T* x, const T* mus, const T* sigmas, uint n) {
  T pc = static_cast<T>( 1);
  unsigned int i = 0;
  T dist = 0;
  T sigma = 0;
  while (i < n) {
    sigma = sigmas[i];
    dist = (x[i] - mus[i] / sigma);
    pc *= 1 / (sqrt(twoPi * sigma)) * exp(-(dist * dist / 2));
    i++;
  }
  return (pc);
}

template <typename T> T AnomalyDetection<T>::p( const T* x, const T* mus, const T* sigmas, uint n) {
	return probabilityDensityFunction(x, mus, sigmas,n);
}

template void AnomalyDetection<ulong>::multivariateProbDensity( CuMatrix<ulong>&,const CuMatrix<ulong>& x, const CuMatrix<ulong>&  means, const CuMatrix<ulong>&  sqrdSigmas);
template void AnomalyDetection<float>::multivariateProbDensity( CuMatrix<float>&,const CuMatrix<float>& x, const CuMatrix<float>&  means, const CuMatrix<float>&  sqrdSigmas);
template void AnomalyDetection<double>::multivariateProbDensity( CuMatrix<double>&,const CuMatrix<double>& x, const CuMatrix<double>&  means, const CuMatrix<double>&  sqrdSigmas);
template <typename T> void AnomalyDetection<T>::multivariateProbDensity( CuMatrix<T> & ret, const CuMatrix<T>& x, const CuMatrix<T>&  means, const CuMatrix<T>&  sqrdSigmas){
	// turn vector into (eigenmatrix?)
  CuMatrix<T> sigmas;
  if(sqrdSigmas.columnVectorQ() || sqrdSigmas.rowVectorQ()) {
	  sigmas = sqrdSigmas.vectorToDiagonal();
  } else {
	  sigmas = sqrdSigmas;
  }
  sigmas.syncBuffers();
  if(checkDebug(debugAnomDet))
	  outln("sigmas " << sigmas);

  uint n = means.n;
  CuMatrix<T> xMinusMu = x.subMeans(means);
  if(checkDebug(debugAnomDet))outln("xMinusMu " << xMinusMu.syncBuffers());

  LUDecomposition<T> luSigmas(sigmas);

  T det = luSigmas.determinant();
  if(checkDebug(debugAnomDet))
	  outln("det " << det);


  T fac1 = 1/pow(twoPi, n / 2.) / sqrt(det);
  if(checkDebug(debugAnomDet))outln("fac1 " << fac1);

  CuMatrix<T> idSigmas = CuMatrix<T>::identity(sigmas.m).syncBuffers();
  if(checkDebug(debugAnomDet))
	  outln("idsigmas " << idSigmas);
  CuMatrix<T> sigmasInv = luSigmas.solve(idSigmas);
  if(checkDebug(debugAnomDet))
	  outln("sigmasInv " << sigmasInv.syncBuffers());
/*
 p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
      exp(-0.5 * sum(bsxfun(@times, Xnormed * pinv(Sigma2), Xnormed), 2));
*/

  ret = fac1 * (-.5* (((xMinusMu * sigmasInv) % xMinusMu).transpose().featureMeans(1) * sigmas.m).transpose()).exp();
  if(checkDebug(debugAnomDet))
	  outln("ret " << ret.syncBuffers());

}
template CuMatrix<ulong> AnomalyDetection<ulong>::multivariateProbDensity( const CuMatrix<ulong>& x, const CuMatrix<ulong>&  means, const CuMatrix<ulong>&  sqrdSigmas);
template CuMatrix<float> AnomalyDetection<float>::multivariateProbDensity( const CuMatrix<float>& x, const CuMatrix<float>&  means, const CuMatrix<float>&  sqrdSigmas);
template CuMatrix<double> AnomalyDetection<double>::multivariateProbDensity( const CuMatrix<double>& x, const CuMatrix<double>&  means, const CuMatrix<double>&  sqrdSigmas);
template <typename T> CuMatrix<T>  AnomalyDetection<T>::multivariateProbDensity( const CuMatrix<T>& x, const CuMatrix<T>&  means, const CuMatrix<T>&  sqrdSigmas){
	CuMatrix<T> ret;
	multivariateProbDensity(ret,x,means, sqrdSigmas);
	return ret;
}

template void AnomalyDetection<ulong>::multivariateProbDensity2( CuMatrix<ulong>&,const CuMatrix<ulong>& x, const CuMatrix<ulong>&  means, const CuMatrix<ulong>&  sqrdSigmas);
template void AnomalyDetection<float>::multivariateProbDensity2( CuMatrix<float>&,const CuMatrix<float>& x, const CuMatrix<float>&  means, const CuMatrix<float>&  sqrdSigmas);
template void AnomalyDetection<double>::multivariateProbDensity2( CuMatrix<double>&,const CuMatrix<double>& x, const CuMatrix<double>&  means, const CuMatrix<double>&  sqrdSigmas);
template <typename T> void AnomalyDetection<T>::multivariateProbDensity2( CuMatrix<T> & ret, const CuMatrix<T>& x, const CuMatrix<T>&  means, const CuMatrix<T>&  sqrdSigmas){
	// turn vector into (eigenmatrix?)
	CuMatrix<T> sigmas;
	if (sqrdSigmas.columnVectorQ() || sqrdSigmas.rowVectorQ()) {
		sigmas = sqrdSigmas.vectorToDiagonal();
	} else {
		sigmas = sqrdSigmas;
	}
	sigmas.syncBuffers();
	if (checkDebug(debugAnomDet))
		outln("sigmas " << sigmas);

	uint n = means.n;
	CuMatrix<T> xMinusMu = x.subMeans(means);
	if (checkDebug(debugAnomDet))
		outln("xMinusMu " << xMinusMu.syncBuffers());

	LUDecomposition<T> luSigmas(sigmas);

	T det = luSigmas.determinant();
	if (checkDebug(debugAnomDet))
		outln("det " << det);

	T fac1 = 1 / pow(twoPi, n / 2.) / sqrt(det);
	if (checkDebug(debugAnomDet))
		outln("fac1 " << fac1);

	CuMatrix<T> idSigmas = CuMatrix<T>::identity(sigmas.m).syncBuffers();
	CuMatrix<T> sigmasInv = luSigmas.solve(idSigmas);
	if (checkDebug(debugAnomDet)) {
		CuMatrix<T> prod = sigmasInv * sigmas;
		outln("prod  " << prod.syncBuffers());
		T tError = prod.sumSqrDiff(idSigmas);
		outln("total error " << tError);
		outln("eps - total error " << ( util<T>::epsilon() - tError));
		assert(tError <= util<T>::epsilon());
	}
	/*
	 p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
	 exp(-0.5 * sum(bsxfun(@times, Xnormed * pinv(Sigma2), Xnormed), 2));
	 */
	CuMatrix<T> prod1 = xMinusMu * sigmasInv;
	if (checkDebug(debugAnomDet))
		outln("xMinusMu * sigmas.inverse() " << prod1.syncBuffers());

	CuMatrix<T> hprod1 = prod1 % xMinusMu;
	if (checkDebug(debugAnomDet))outln("(xMinusMu * sigmas.inverse()) % xMinusMu (hprod1) " << hprod1.syncBuffers());

	CuMatrix<T> sumv(1, hprod1.n, false, true);
	hprod1.transpose().featureMeans(sumv, true);
	if (checkDebug(debugAnomDet))outln("(xMinusMu * sigmas.inverse()) % xMinusMu " << hprod1.syncBuffers());

	CuMatrix<T> colSum = (sumv * hprod1.n).transpose();
	if (checkDebug(debugAnomDet))outln("colSum " << colSum.syncBuffers());

	CuMatrix<T> half = -.5 * colSum;
	CuMatrix<T> ehalf = half.exp();

	ret= ehalf * fac1;
	//return fac1 * (-.5* (((xMinusMu * sigmasInv) % xMinusMu).transpose().featureMeans(1) * sigmas.m).transpose()).exp();
}

template CuMatrix<ulong> AnomalyDetection<ulong>::multivariateProbDensity2( const CuMatrix<ulong>& x, const CuMatrix<ulong>&  means, const CuMatrix<ulong>&  sqrdSigmas);
template CuMatrix<float> AnomalyDetection<float>::multivariateProbDensity2( const CuMatrix<float>& x, const CuMatrix<float>&  means, const CuMatrix<float>&  sqrdSigmas);
template CuMatrix<double> AnomalyDetection<double>::multivariateProbDensity2( const CuMatrix<double>& x, const CuMatrix<double>&  means, const CuMatrix<double>&  sqrdSigmas);
template <typename T> CuMatrix<T>  AnomalyDetection<T>::multivariateProbDensity2( const CuMatrix<T>& x, const CuMatrix<T>&  means, const CuMatrix<T>&  sqrdSigmas){
	CuMatrix<T> ret;
	multivariateProbDensity2(ret,x,means, sqrdSigmas);
	return ret;
}


template <typename T> CuMatrix<T> AnomalyDetection<T>::sigma(CuMatrix<T> x, T* mus){
  uint m = x.m;
  uint n = x.n;
  CuMatrix<T> sigma = CuMatrix<T>::zeros(n, n);
  uint i = 0;
  CuMatrix<T> mat =  CuMatrix<T>::zeros(n, 1);
  while (i < m) {
    checkCudaErrors(cudaMemcpy(mat.elements, x.elements + i * n, n, cudaMemcpyHostToHost));
	CuMatrix<T>::HHCopied++;
	CuMatrix<T>::MemHhCopied += n * sizeof(T);
    mat = mat.subMeans(mus);
    sigma = sigma + mat * mat.transpose();
    i++;
  }
  return sigma / m;
}
template <typename T> CuMatrix<T> AnomalyDetection<T>::sigmaVector(CuMatrix<T> x){
	CuMatrix<T> normed = x.normalize();
	return normed.featureMeans(true);
}
