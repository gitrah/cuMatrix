/*
 *
 *  Created on: Aug 24, 2012
 *      Author: reid
 */

#include "Matrix.h"
#include "AnomalyDetection.h"
//  val twoPi = 2 * math.Pi
#include <math.h>
#include <utility>
#define Pi 3.141592653589793
#define OneOverSqrt2Pi 2.5066282746310002

const double oneOverSqrt2Pi =  1. / sqrt(2. * Pi);
const double twoPi= 2 * Pi;

// sqrdSigma is a column matrix, by convention
template void AnomalyDetection<float>::fitGaussians( Matrix<float>& means, Matrix<float>& sqrdSigmas, const Matrix<float>& x);
template void AnomalyDetection<double>::fitGaussians( Matrix<double>& means, Matrix<double>& sqrdSigmas, const Matrix<double>& x);
template <typename T> void AnomalyDetection<T>::fitGaussians( Matrix<T>& means, Matrix<T>& sqrdSigmas, const Matrix<T>& x) {
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
	outln("ad.fitGaussians " << sqrdSigmas.toShortString());
	x.fitGaussians(sqrdSigmas, means);
}

template  pair<float,float>  AnomalyDetection<float>::selectThreshold( const Matrix<float>& yValidation, const Matrix<float>& probabilityDensityValidation);
template  pair<double,double>  AnomalyDetection<double>::selectThreshold( const Matrix<double>& yValidation, const Matrix<double>& probabilityDensityValidation);
template <typename T> pair<T,T>  AnomalyDetection<T>::selectThreshold(const Matrix<T>& yValidation, const Matrix<T>& probabilityDensityValidation) {
	T bestEpsilon = 0, bestF1 = 0, f1 = 0;
	uint truePos = yValidation.sum();
	outln("yValidation truePos " << truePos);
	uint falsePos = 0, falseNeg = 0;
	T precision, recall;
	pair<T,T> prBounds = probabilityDensityValidation.bounds();
	outln("prBounds " << pp(prBounds));
	T stepSize = (prBounds.second - prBounds.first)/1000.0;
	for(T epsilon = prBounds.first + stepSize; epsilon <= prBounds.second; epsilon += stepSize) {
		Matrix<T> cvPredictions = probabilityDensityValidation < epsilon;
		truePos = (  (cvPredictions == 1.0) && (yValidation == 1)).sum();
		falsePos = (  (cvPredictions == 1.0) && (yValidation == 0)).sum();
		falseNeg = (  (cvPredictions == 0.0) && (yValidation == 1)).sum();
		precision = 1.0 * truePos / (truePos + falsePos);
		recall = 1.0 * truePos / (truePos + falseNeg);
		f1 = 2 * precision * recall / (precision + recall );
		//outln("for eps " << epsilon );
		//outln("\tgot truePos "<< truePos << ", falsePos " << falsePos << ", falseNeg " << falseNeg);
		//outln("\tgot precision "<< precision << ", recall " << recall << ", f1 " << f1);
		if(f1 > bestF1) {
			bestF1 = f1;
			bestEpsilon = epsilon;
		}
	}
	return pair<T,T>(bestEpsilon, bestF1);
}

/*
template <typename T> Matrix<T> AnomalyDetection<T>::multivariateGaussian(const Matrix<T>& x, const Matrix<T>& mu, const Matrix<T>& sigmaSqrd) {
	Matrix<T> s2 = sigmaSqrd.vectorQ() ? sigmaSqrd.diagonal() : sigmaSqrd;
	Matrix<T> xNormed = x.subMeans(mu);
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

template Matrix<float> AnomalyDetection<float>::multiVariateProbDensity( const Matrix<float>& x, const Matrix<float>&  means, const Matrix<float>&  sqrdSigmas);
template Matrix<double> AnomalyDetection<double>::multiVariateProbDensity( const Matrix<double>& x, const Matrix<double>&  means, const Matrix<double>&  sqrdSigmas);
template <typename T> Matrix<T>  AnomalyDetection<T>::multiVariateProbDensity( const Matrix<T>& x, const Matrix<T>&  means, const Matrix<T>&  sqrdSigmas){
	// turn vector into (eigenmatrix?)
  Matrix<T> sigmas;
  if(sqrdSigmas.columnVectorQ() || sqrdSigmas.rowVectorQ()) {
	  sigmas = sqrdSigmas.vectorToDiagonal();
	  sigmas.syncBuffers();
  } else {
	  sigmas = sqrdSigmas;
  }
  outln("sigmas " << sigmas);

  uint n = means.n;
  Matrix<T> xMinusMu = x.subMeans(means);
  outln("xMinusMu " << xMinusMu.syncBuffers());

  T det = sigmas.determinant();
  outln("det " << det << ", n " << n);

  outln("twoPi " << twoPi);
  outln("pow(twoPi, -n / 2.) " << 1/pow(twoPi, n / 2.)); // cuda pow no like negative exponent
  T fac1 = 1/pow(twoPi, n / 2.) / sqrt(det);
  outln("fac1 " << fac1);
  Matrix<T> sigmasInv = sigmas.inverse();
  Matrix<T> prod1 = xMinusMu * sigmasInv;
  outln("xMinusMu * sigmas.inverse() " << prod1.syncBuffers());
  Matrix<T> hprod1 = prod1 % xMinusMu;
  outln("(xMinusMu * sigmas.inverse()) % xMinusMu " << hprod1.syncBuffers());
  Matrix<T> sumv(1,hprod1.n,false,true);
  hprod1.transpose().featureMeans(sumv,true);
  outln("(xMinusMu * sigmas.inverse()) % xMinusMu " << hprod1.syncBuffers());
  Matrix<T> colSum = (sumv * hprod1.m).transpose();
  outln("colSum " << colSum.syncBuffers());
  Matrix<T> half =  (-.5 * sumv * hprod1.m).exp();

  //return ((((xMinusMu * sigmas.inverse()) % xMinusMu).toColumnSumVector() * (-.5)).exp() * fac1);
  return half * fac1;
}

template <typename T> Matrix<T> AnomalyDetection<T>::sigma(Matrix<T> x, T* mus){
  uint m = x.m;
  uint n = x.n;
  Matrix<T> sigma = Matrix<T>::zeros(n, n);
  uint i = 0;
  Matrix<T> mat =  Matrix<T>::zeros(n, 1);
  while (i < m) {
    checkCudaErrors(cudaMemcpy(mat.elements, x.elements + i * n, n, cudaMemcpyHostToHost));
    mat = mat.subMeans(mus);
    sigma = sigma + mat * mat.transpose();
    i++;
  }
  return sigma / m;
}
template <typename T> Matrix<T> AnomalyDetection<T>::sigmaVector(Matrix<T> x){
	Matrix<T> normed = x.normalize();
	return normed.featureMeans(true);
}
