/*
 * LinearRegression.h
 *
 *  Created on: Aug 22, 2012
 *      Author: reid
 */
#pragma once

#include "CuMatrix.h"
#include "util.h"

template<typename T> class LinearRegression {
public:
//
	static void gradCostFunction(CuMatrix<T>& grad, T& cost,
			const CuMatrix<T>& x, const CuMatrix<T>& y,
			const CuMatrix<T>& theta, T lambda);

	// operator% => hadamardProduct (elementwise product)
	static T costFunctionNoRegNu(CuMatrix<T>& hThetaTxT, CuMatrix<T>& yT, int m) ;

	static CuMatrix<T> gradientDescent(T alpha, CuMatrix<T>& theta, CuMatrix<T>& x, CuMatrix<T>& y,CuMatrix<T>& jHistory, int iters) ;
	static CuMatrix<T> gradientDescentL(T alpha, CuMatrix<T>& theta, CuMatrix<T>& x, CuMatrix<T>& y,CuMatrix<T>& jHistory, int iters);
	static CuMatrix<T> gradientDescentLoop(T alpha, CuMatrix<T>& theta, CuMatrix<T>& x, CuMatrix<T>& y,CuMatrix<T>& jHistory, int iters) ;

	static T costFunctionNoReg(CuMatrix<T>& hThetaTxT, CuMatrix<T>& yT, int m) ;
	static T costFunction(CuMatrix<T>& a, CuMatrix<T>& y, T lambda,
			vector<CuMatrix<T> > thetas) ;

	static T costFunctionNoReg2(CuMatrix<T>& hThetaT, CuMatrix<T>& yT, int m) ;

	template<template <typename> class CostFunction > static CuMatrix<T> gradientApprox(
			CostFunction<T> costFn, CuMatrix<T> theta, T epsilon);
};

