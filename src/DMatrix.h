/*
 * DMatrix.h
 *
 *  Created on: Oct 13, 2013
 *      Author: reid
 */

#ifndef DMATRIX_H_
#define DMATRIX_H_

template<typename T> struct DMatrix {
public:
	T* elements;
	uint m, n, p;

	__host__ __device__ DMatrix() :
			elements(null), m(0), n(0), p(0) {
	}
	__host__ __device__ DMatrix(uint _m, uint _n) :
			elements(null), m(_m), n(_n), p(_n) {
	}
	__host__ __device__ DMatrix(T* _elements, uint _m, uint _n) :
			elements(_elements), m(_m), n(_n), p(_n) {
	}
	__host__ __device__ DMatrix(T* _elements, uint _m, uint _n, uint _p) :
			elements(_elements), m(_m), n(_n), p(_p) {
	}


	static DMatrix<T> ZeroMatrix;
};

// Device Matrix
template<class T>
struct SharedMemory
{
	__device__ inline operator T *() {
		extern __shared__ int __smem[];
		return (T*) __smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T*) __smem;
	}
};

template<typename T> __host__ __device__ inline void prdm(const char* msg, const DMatrix<T>& m) {
	printf("%s %x %d*%d*%d\n",msg, m.elements, m.m, m.n,m.p);
}
template<typename T> __host__ __device__ inline void prdm( const DMatrix<T>& m) {
	prdm("",m);
}

template<typename T> extern inline __device__ T get(const DMatrix<T>& dm, uint l) {
	//	if(dm.n == dm.p) {
//		return dm.elements[l];
//	}
	uint div = l /dm.n;
	uint idx = div * dm.p;
	idx += l - div * dm.n;
	//printf("offset l %u -> %u\n",l ,idx);
	return dm.elements[idx ];
}

template<typename T> struct MatProd {
	typedef void (*MatProdKptr) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int);
};

#endif /* DMATRIX_H_ */
