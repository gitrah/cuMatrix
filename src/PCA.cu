#include "PCA.h"

template<typename T> __host__ __device__ __inline__ T hypot(T a, T b) {
	T r = 0;
	if (abs(a) > abs(b)) {
		r = b / a;
		return abs(a) * sqrt(1 + r * r);
	} else if (b != 0) {
		r = a / b;
		return abs(b) * sqrt(1 + r * r);
	} else {
		return r;
	}
}

template<typename T>  __host__ __device__ void assignsa(T* s, int k, int m, int n, T* a) {
	s[k] = 0;
	int i = k;
	while (i < m) {
		s[k] = hypot(s[k], a[i * n + k]);
		i++;
	}

	if (s[k] != 0.0) {
		if (a[k * n + k] < 0.0) {
			s[k] = -s[k];
		}
		i = k;
		while (i < m) {
			a[i * n + k] /= s[k];
			i++;
		}
		a[k * n + k] += 1.0;
	}
	s[k] = -s[k];

}

template<typename T>  CuMatrix<T> PCA<T>::pca(const CuMatrix<T>& src, int k) {
	CuMatrix<T> normed = src.normalize();
}
