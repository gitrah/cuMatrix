/*
 * ludecomposition.cc
 *
 *  Created on: Oct 5, 2012
 *      Author: reid
 *   Algorithm originally from JAMA I think
 */

#include "LuDecomposition.h"
#include "MatrixExceptions.h"
#include <climits>

template<typename T> LUDecomposition<T>::LUDecomposition(CuMatrix<T>& x) :
		 m(x.m), n(x.n), mRef(x), pivsign(1),luRowi (null){
	ulong len = m*n*sizeof(T);
	checkCudaErrors(cudaHostAlloc((void**)&lu,len,0));
	if(mRef.lastMod == mod_device) {
		mRef.syncBuffers();
	}
	checkCudaErrors(cudaMemcpy(lu, mRef.elements, len,cudaMemcpyHostToHost));
	pivots = new int[m];
	for (int i = 0; i < m; i++) {
		pivots[i] = i;
	}
	checkCudaErrors(cudaHostAlloc((void**)&luColj,m*sizeof(T),0));
	compute();
}

template<typename T> LUDecomposition<T>::~LUDecomposition() {
	checkCudaErrors(cudaFreeHost(lu));
	checkCudaErrors(cudaFreeHost(luColj));
}

template<typename T> void LUDecomposition<T>::compute() {
	int j = 0;
	int i = 0;
	int k = 0;
	int p = 0;
	int pidx=0;
	int jidx=0;
	T s = 0;
	int kMax = 0;
	T t = 0;
	while (j < n) {
		// Make a copy of the j-th column to localize references.
		i = 0;
		while (i < m) {
			luColj[i] = lu[i * n + j];
			i++;
		}
		i = 0;
		// Apply previous transformations.
		while (i < m) {
			luRowi = lu + i * n;
			// Most of the time is spent in the following dot product.
			kMax = min(i, j);
			s = 0;
			k = 0;
			while (k < kMax) {
				s += luRowi[ k] * luColj[k];
				k++;
			}
			luColj[i] -= s;
			luRowi [j] = luColj[i];
			i++;
		}
		// Find pivot and exchange if necessary.
		p = j;
		i = j + 1;
		while (i < m) {
			if (abs(luColj[i]) > abs(luColj[p])) {
				p = i;
			}
			i++;
		}
		if (p != j) {
			k = 0;
			while (k < n) {
				pidx = jidx = k;
				pidx += p * n;
				jidx += j * n;
				t = lu[pidx];
				lu[pidx] = lu[jidx];
				lu[jidx] = t;
				k++;
			}

			k = pivots[p];
			pivots[p] = pivots[j];
			pivots[j] = k;
			pivsign = -pivsign;
		}

		// Compute multipliers.
		jidx = j * n + j;
		if (j < m && lu[jidx] != 0.0) {
			i = j + 1;
			while (i < m) {
				lu[i * n + j] /= lu[jidx];
				i++;
			}
		}
		j++;
	}
	mRef.invalidateDevice();
}

template<typename T> bool LUDecomposition<T>::singularQ() {
	int j = 0;
	while (j < n) {
		if ( abs(lu[j * (n+1)]) < 1e-6) {
			return true;
		}
		j++;
	}
	return false;
}

template<typename T> T LUDecomposition<T>::determinant() {
	if (m != n) {
		dthrow (notSquare());
	}
	T d = pivsign;
	int j = 0;
	while (j < n) {
		d *= lu[j * (n + 1)];
		j++;
	}
	return d;
}

template<typename T> CuMatrix<T> LUDecomposition<T>::solve(const CuMatrix<T>& b) {
	if (b.m != m) {
		dthrow ( rowDimsDisagree());
	}

	if (m == n && singularQ()) {
		dthrow ( singularMatrix());
	}

	// Copy right hand side with pivoting
	int nx = b.n;
	CuMatrix<T> xm = b.clippedRowSubset(pivots, m, std::pair<uint,uint>(0, nx - 1));
	int nm = xm.n;
	T* x = xm.elements;
	//outln("xm0 " << xm.syncBuffers());
	// Solve L*Y = B(piv,:)
	int k = 0;
	int i = 0;
	int j = 0;
	while (k < n) {
		i = k + 1;
		while (i < n) {
			j = 0;
			while (j < nx) {
				x[i * nm + j] -= x[k * nm + j] * lu[i * n + k];
				j++;
			}
			i++;
		}
		k++;
	}
	//outln("xm1 " << xm.syncBuffers());
	// Solve U*X = Y;
	if(n > 1) {
		k = n - 1;
		while (true) {
			j = 0;
			while (j < nx) {
				x[k * nm + j] /= lu[k * (n + 1)];
				j++;
			}
			i = 0;
			while (i < k) {
				j = 0;
				while (j < nx) {
					x[i * nm + j] -= x[k * nm + j] * lu[i * n + k];
					j++;
				}
				i++;
			}
			if(k==0){
				break;
			} else {
 				k -= 1;
			}
		}
	}
	xm.invalidateDevice();

	return xm;
}

template class LUDecomposition<float>;
template class LUDecomposition<double>;
template class LUDecomposition<ulong>;
