/*
 * svm.cc
 *
 *  Created on: Jun 10, 2014
 *      Author: reid
 */

#ifndef SVM_CC_
#define SVM_CC_

#include "svm.h"

template <typename T> template< template <typename> class KernelFunction >
void svm<T>::svmTrain(CuMatrix<T>& model, const CuMatrix<T>& x, T c, KernelFunction<T> kernel, int maxPasses, T tol) {

}


#endif /* SVM_CC_ */
