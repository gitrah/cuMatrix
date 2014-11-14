/*
 * anomaly.h
 *
 *  Created on: Sep 5, 2012
 *      Author: reid
 */

#ifndef ANOMALY_DETECTION_H_
#define ANOMALY_DETECTION_H_

#include "CuMatrix.h"

template <typename T> struct f1score {
	uint truePos, falsePos, falseNeg;
	T epsilon, f1;
	T precision, recall;

	void zero() {
		truePos = falsePos = falseNeg = 0;
		epsilon = f1 = precision = recall = 0;
	}
} ;

template <typename T> class AnomalyDetection {
public:

	static void fitGaussians(CuMatrix<T>& mean, CuMatrix<T>& variance, const CuMatrix<T>& x);
	static pair<T,T> selectThreshold(const CuMatrix<T>& yValidation, const CuMatrix<T>& probabilityDensityValidation);
	static pair<T,T> selectThresholdOmp(const CuMatrix<T>& yValidation, const CuMatrix<T>& probabilityDensityValidation);
	static T probabilityDensityFunction( const T* x, const T* mus, const T* sigmas, uint n);
	static T p( const T* x, const T* mus, const T* sigmas, uint n);
	static void multivariateProbDensity( CuMatrix<T>& ret, const CuMatrix<T>& x, const CuMatrix<T>&  means, const CuMatrix<T>&  sqrdSigmas);
	static CuMatrix<T>  multivariateProbDensity( const CuMatrix<T>& x, const CuMatrix<T>&  means, const CuMatrix<T>&  sqrdSigmas);
	static void multivariateProbDensity2( CuMatrix<T>& ret, const CuMatrix<T>& x, const CuMatrix<T>&  means, const CuMatrix<T>&  sqrdSigmas);
	static CuMatrix<T>  multivariateProbDensity2( const CuMatrix<T>& x, const CuMatrix<T>&  means, const CuMatrix<T>&  sqrdSigmas);
	static CuMatrix<T> sigma(CuMatrix<T> x, T* mus);
	static CuMatrix<T> sigmaVector(CuMatrix<T> x);
};

#ifdef CuMatrix_Enable_Cdp
template <typename T> __global__ void selectThresholdKernel(T* bestEpsilon, T* bestF1, DMatrix<T> yValidation, DMatrix<T> probabilityDensityValidation, T pdvMin, T pdvMax);
#endif

#endif /* ANOMALY_DETECTION_H_ */
