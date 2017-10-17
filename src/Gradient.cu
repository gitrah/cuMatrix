#include "Gradient.h"


template <typename T> __host__ __device__ void gradPix(CuMatrix<T>& i,CuMatrix<T>&j, CuMatrix<T>& src ) {

}
template <typename T> __host__ __device__ pair<CuMatrix<T>&,CuMatrix<T>&> grad( CuMatrix<T>& src) {
	CuMatrix<T> i = CuMatrix<T>::zeros(src.m,src.n);
	CuMatrix<T> j = CuMatrix<T>::zeros(src.m,src.n);
	DMatrix<T> d_src, d_i, d_j;

}
