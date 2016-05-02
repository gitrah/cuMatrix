#include "PCA.h"

template<typename T>
	CuMatrix<T> PCA<T>::pca(const CuMatrix<T>& src, int k) {
	CuMatrix<T> normed = src.normalize();
}
