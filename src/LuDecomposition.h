/*
 * ludecomposition.h
 *
 *  Created on: Oct 3, 2012
 *      Author: reid
 */
#include "Matrix.h"

#ifndef LUDECOMPOSITION_H_
#define LUDECOMPOSITION_H_


template <typename T> class LUDecomposition {
  T* lu;
  uint m,n;
  int* pivots;
  Matrix<T>& mRef;

  int pivsign;
  T* luRowi;
  T* luColj;

public:
  LUDecomposition(Matrix<T>& x);

  void compute();
  bool singularQ();
  T determinant();
  Matrix<T> solve(const Matrix<T>& b);

};
#endif /* LUDECOMPOSITION_H_ */
