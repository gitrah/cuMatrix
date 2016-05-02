/*
 * ludecomposition.h
 *
 *  Created on: Oct 3, 2012
 *      Author: reid
 */

#ifndef LUDECOMPOSITION_H_
#define LUDECOMPOSITION_H_

#include "CuMatrix.h"

template <typename T> class LUDecomposition {
  T* lu;
  int m,n;
  int* pivots;
  CuMatrix<T>& mRef;

  int pivsign;
  T* luRowi;
  T* luColj;

public:
  LUDecomposition(CuMatrix<T>& x);
  ~LUDecomposition();

  void compute();
  bool singularQ();
  T determinant();
  CuMatrix<T> solve(const CuMatrix<T>& b);

};
#endif /* LUDECOMPOSITION_H_ */
