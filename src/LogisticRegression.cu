/*
 * LogisticRegression.cu
 *
 *  Created on: Nov 26, 2014
 *      Author: reid
 */

#include "LogisticRegression.h"


template<typename T> __host__ CUDART_DEVICE
void LogisticRegression<T>::gradCostFunction(CuMatrix<T>& grad, T& cost,
		const CuMatrix<T>& x, const CuMatrix<T>& y,
		const CuMatrix<T>& theta, T lambda) {
	const int m = y.m;
	CuMatrix<T> hThetaT = ( theta.transpose() * x.transpose()).sigmoid();
	CuMatrix<T> yT = y.transpose();
	cost = costFunctionNoReg2(hThetaT, yT, m);
	CuMatrix<T> gradNoReg = ((hThetaT - yT) * x) / m;

	// regularization component requires theta with first param zero-d
	CuMatrix<T> thetaCopy = theta.copy();
	thetaCopy.set(0, 0, 0); // elements[0] = 0;
	CuMatrix<T> gradReg = lambda * thetaCopy.transpose() / m;
	T jDel = lambda / (2. * m) * (thetaCopy ^ ((T)2)).sum();
	cost += jDel;
	grad = (gradNoReg + gradReg).transpose(); // want as column
}
template __host__ CUDART_DEVICE void LogisticRegression<float>::gradCostFunction(
		CuMatrix<float>& grad, float& cost,
		const CuMatrix<float>& x, const CuMatrix<float>& y, const CuMatrix<float>& theta, float lambda);

template __host__ CUDART_DEVICE void LogisticRegression<double>::gradCostFunction(
		CuMatrix<double>& grad, double& cost,
		const CuMatrix<double>& x, const CuMatrix<double>& y, const CuMatrix<double>& theta,
		double lambda);

template __host__ CUDART_DEVICE void LogisticRegression<ulong>::gradCostFunction(
		CuMatrix<ulong>& grad, ulong& cost,
		const CuMatrix<ulong>& x, const CuMatrix<ulong>& y, const CuMatrix<ulong>& theta,
		ulong lambda);
