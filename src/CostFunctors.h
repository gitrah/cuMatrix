/*
 * CostFunctors.h
 *
 *  Created on: Sep 18, 2013
 *      Author: reid
 */
#pragma once

#include <helper_cuda.h>
#include <helper_math.h>
#include "UnaryOpF_Gen.h"

template<typename T> struct regularizationF : public UnaryOpF<T,1> {
   __host__ __device__ T&  lambda() { return CuFunctor<T,1>::state; }
   __host__ __device__ const T& lambda_ro() const { return CuFunctor<T,1>::state; }
};

template<typename T>
struct lambdaThetaSqrd: public regularizationF<T>  {
	lambdaThetaSqrd(T _lambda) { lambda() = _lambda;}

	__host__ __device__
	T operator()(const T xi) const {
		return (lambda_ro()* xi * xi);
	}
};

template<typename T> struct costF : public UnaryOpF<T,1> {

};

template<typename T> struct hypothesisF : public UnaryOpF<T,1> {
};

template<typename T>
struct linearRegressionCostF : public costF<T>  {
	const hypothesisF<T>& hyp;

	explicit linearRegressionCostF(const hypothesisF<T>& hyp) : hyp(hyp) {}
	__host__ __device__
	T operator()(const T& xi, const T& yi) const {
		T delta = hypothesisF(xi) - yi;
		return delta * delta;
	}
};

template<typename T>
struct reglinearRegressionCostF : public linearRegressionCostF<T>  {
	const hypothesisF<T>& hyp;
	const regularizationF<T>& reg;

	explicit reglinearRegressionCostF(const hypothesisF<T>& hyp, const regularizationF<T>& reg) :  hyp(hyp), reg(reg){}

	using unaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi, const T& yi) const {
		T delta = hypothesisF(xi) - yi;
		return delta * delta + regF();
	}
};
