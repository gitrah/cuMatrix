/*
 * anomaly.h
 *
 *  Created on: Sep 5, 2012
 *      Author: reid
 */

#ifndef ANOMALY_DETECTION_H_
#define ANOMALY_DETECTION_H_

#include "Matrix.h"

template <typename T> class AnomalyDetection {
public:
	static void fitGaussians(Matrix<T>& mean, Matrix<T>& variance, const Matrix<T>& x);
	static pair<T,T> selectThreshold(const Matrix<T>& yValidation, const Matrix<T>& probabilityDensityValidation);
	static T probabilityDensityFunction( const T* x, const T* mus, const T* sigmas, uint n);
	static T p( const T* x, const T* mus, const T* sigmas, uint n);
	static Matrix<T>  multiVariateProbDensity( const Matrix<T>& x, const Matrix<T>&  means, const Matrix<T>&  sqrdSigmas);
	static Matrix<T> sigma(Matrix<T> x, T* mus);
	static Matrix<T> sigmaVector(Matrix<T> x);
};

#endif /* ANOMALY_DETECTION_H_ */
