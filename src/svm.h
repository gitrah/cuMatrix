/*
 * svm.h
 *
 *  Created on: Jun 10, 2014
 *      Author: reid
 */

#pragma once

#include "CuMatrix.h"

template <typename TTT, template <typename  TT> class T>
class MyClass
{
   T<TTT> func(TTT p);
};


template <typename TTT, template <typename  TT> class T> T<TTT> MyClass<TTT,T>::func(TTT p)
{
  // blah
}


template<typename T> class svm {
public:

	template< template <typename> class KernelFunction >
	static void svmTrain(CuMatrix<T>& model, const CuMatrix<T>& x, T c, KernelFunction<T> kernel, int maxPasses, T tol = util<T>::epsilon());

};
