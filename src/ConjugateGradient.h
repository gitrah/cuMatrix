/*
 * conjugategradient.h
 *
 *  Created on: Oct 13, 2012
 *      Author: reid
 */
#pragma once

#include "CuMatrix.h"

template<typename T> class ConjugateGradient {
public:
	static T rho;
	static T sig;
	static T int0;
	static T ext;
	static T max;
	static T ratio;

	static T MinPositiveValue;

	// 'ISO C++ forbids in-class initialization of non-const static member,' so...
	static void init() {
		rho = 0.01;
		sig = .5;
		int0 = .1;
		ext = 3.0;
		max = 20;
		ratio = 100;
	}
	template<typename CostFunction> static std::pair<CuMatrix<T>, std::pair<CuMatrix<T>, int> >
	fmincg(CostFunction& f, CuMatrix<T>& x, int length = 50, int red = 1);
	static inline bool nanQ(T value);

// requires #include <limits>

	static inline bool infQ(T value);

	static inline bool realQ(T t);
};
template <typename T> T ConjugateGradient<T>::rho;
template <typename T> T ConjugateGradient<T>::sig;
template <typename T> T ConjugateGradient<T>::int0;
template <typename T> T ConjugateGradient<T>::ext;
template <typename T> T ConjugateGradient<T>::max;
template <typename T> T ConjugateGradient<T>::ratio;
