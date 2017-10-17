/*
 * CuSet.h
 *
 */

#pragma once

#include <set>
#include <functional>
#include "util.h"

using std::set;
using  std::binary_function;

extern string OrderedIntersectionTypeStr[];

template<typename T> __global__ void setMakerKernel( T* ptrset, int* len, const int pset, const T* atts, const int patts, long n);
template<typename T> __global__ void setMakerIterativeKernel( T* ptrset, int* len, const int p,const T* atts, const int patts, long n);
template <typename T> __global__ void mergeSetsKernel( T* ptrset, int* len, const int pset, const T* src1, const T* src2,const int p1, const int p2, const int n1, const int n2);

enum OrderedIntersectionType {
	AbeforeB, // A first
	AonB,
	BinA,
	BonA,  // B first
	AinB,
	BbeforeA
};

enum AllocType {
	NonAlloced,
	CudaHostAlloced,
	CudaMalloced,
	DevMalloced
};

//enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice
__host__ __device__ enum cudaMemcpyKind copyKind(AllocType trg,  AllocType src);

template <typename T> struct Counter {
	T val;
	int count;

	Counter(T val): val(val), count(0) {}
	Counter(T val, int count): val(val), count(count) {}
	int inc() { return ++count;}

	int test( T v ) {
		if( fabs(val - v) <= util<T>::epsilon())
			return inc();
		return 0;
	}
	int test( Counter<T> o ) {
		if( fabs(val - o.val) <= util<T>::epsilon()) {
			return count += o.count;
		} else {
			val = o.val;  count= o.count;
			return count;
		}
	}
	Counter<T>&  operator=( Counter<T> o) { test(o); return *this;}

	static __host__ __device__ int size(int n) { return  n * sizeof(Counter<T>); }
};

template <typename T> struct OrderedCountedSet {
	T* vals;
	int pvals;
	int* counts;
	int pcounts;
	int n;
	AllocType valType;
	AllocType countType;

	__host__ __device__ OrderedCountedSet(T* vals, int n, int p =1): vals(vals), n(n), pvals(p), pcounts(p), valType(NonAlloced) {
#ifdef __CUDA_ARCH__
		counts = malloc(n * sizeof(int));
		countType = DevMalloced;
#else
		struct cudaPointerAttributes ptrAtts;
		cherr(cudaPointerGetAttributes(&ptrAtts, vals ));
		if(ptrAtts.memoryType == cudaMemoryTypeHost) {
			cherr(cudaMallocHost( &counts, n * sizeof(int)));
			countType = CudaHostAlloced;
		}else {
			cherr(cudaMalloc( &counts, n * sizeof(int)));
			countType = CudaMalloced;
		}
#endif
	}

	__host__ __device__ OrderedCountedSet(T* vals, int* counts, int n, int pvals, int pcounts,
			AllocType valType, AllocType countType) :
			vals(vals), counts(counts), n(n), pvals(pvals),pcounts(pcounts), valType(valType), countType(
					countType) {
	}

	__host__ __device__ OrderedCountedSet(int n, AllocType valType, AllocType countType): n(n), pvals(1), pcounts(1), valType(valType), countType(countType) {
		allocBuffers();
	}

	/*
	 * no cudaRealloc, but THIS IS JUST AS GOOD
	 */
	__host__ __device__ void resize(int newN) {
		OrderedCountedSet<T> prime(newN, valType, countType);
		memcpy(prime, *this, MIN(newN, n));
		freeBuffers();
		prime.valType = prime.countType = NonAlloced; // to keep its buffers for meown
		vals = prime.vals;
		counts = prime.counts;
		n = prime.n;
	}

	OrderedCountedSet<T> operator+(int off) {
		return OrderedCountedSet<T>(vals + off, counts + off, n - off, NonAlloced, NonAlloced);
	}

	__host__ __device__ void allocBuffers() {
#ifdef __CUDA_ARCH__
		//assert(valType == DevMalloced && countType == DevMalloced);
		vals = (T*)malloc(n * sizeof(T));
		counts = (int*)malloc(n * sizeof(int));
#else
		if(valType == CudaHostAlloced) {
			cherr(cudaMallocHost( &vals, n * sizeof(T)));
		}else {
			cherr(cudaMalloc( &vals, n * sizeof(T)));
		}
		if(countType == CudaHostAlloced) {
			cherr(cudaMallocHost( &counts, n * sizeof(int)));
		}else {
			cherr(cudaMalloc( &counts, n * sizeof(int)));
		}
#endif
	}

	__host__ __device__ void freeBuffers() {
#ifdef __CUDA_ARCH__
		if(vals) {
			//assert(valType == DevMalloced|| valType == NonAlloced);
			if(valType == DevMalloced)
				free(vals);
		}
		if(counts) {
			//assert(countType == DevMalloced || countType == NonAlloced);
			if(countType == DevMalloced)
				free(counts);
		}
#else
		if(vals) {
			assert(valType !=DevMalloced );
			if(valType == CudaHostAlloced) {
				cherr(cudaFreeHost(vals));
			}else if(valType == CudaMalloced) {
				cherr(cudaFree(vals));
			}
		}
		if(counts) {
			assert(countType !=DevMalloced );
			if(countType == CudaHostAlloced) {
				cherr(cudaFreeHost(counts));
			}
			else if(countType == CudaMalloced) {
				cherr(cudaFree(counts));
			}
		}
#endif
	}

	__host__ __device__ ~OrderedCountedSet() {
		freeBuffers();
	}

	__host__ __device__ void set(int idx, Counter<T> ctr) {
		T v = vals[idx * pvals];
		if( fabs(ctr.val - v) <= util<T>::epsilon()) {
			counts[idx * pcounts]+= ctr.count;
		} else {
			 vals[idx* pvals] = ctr.val;
			 counts[idx * pcounts] = ctr.count;
		}
	}

	__host__ __device__ Counter<T> get(int idx) {
		return Counter<T>(vals[idx* pvals],counts[idx* pcounts]);
	}

	static __host__ __device__ void memcpy(OrderedCountedSet<T>& trg, const OrderedCountedSet<T>& src, int len, cudaStream_t stream = 0) {
		//::memcpy(trg.vals, src.vals, len * sizeof(T));\
		//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
		cherr(cudaMemcpy2DAsync(trg.vals, 	trg.pvals * sizeof(T), 	src.vals, 	src.pvals * sizeof(T), sizeof(T), len, copyKind(trg.valType, src.valType), stream));
		//::memcpy(trg.counts, src.counts, len * sizeof(int));
		cherr(cudaMemcpy2DAsync(trg.counts, 	trg.pcounts*sizeof(int),src.counts, src.pcounts*sizeof(int), sizeof(int), len, copyKind(trg.countType,src.countType),stream));
	}
	static __host__ __device__ void memcpy(OrderedCountedSet<T>& trg, const OrderedCountedSet<T>& src) {
		OrderedCountedSet<T>::memcpy(trg,src,src.n);
	}

	__host__ __device__ T& operator[](ptrdiff_t ofs) {
		counts[ofs * pcounts]++;
		return vals[ofs * pvals];
	}

	__host__ __device__ const T& operator[](ptrdiff_t ofs) const {
		counts[ofs * pcounts]++;
		return vals[ofs * pvals];
	}

/*
	__host__ __device__ T& operator[](int i) { return ary[i+offset]; }
	__host__ __device__ const T& operator[](int i) const { return ary[i+offset]; }
*/
	//__device__ inline operator T *() {

	__host__ __device__ OrderedCountedSet<T>& operator=(const OrderedCountedSet<T>& o) {

		//vals(vals), counts(counts), n(n), pvals(pvals),pcounts(pcounts), valType(valType), countType(countType)
		freeBuffers();
		vals = o.vals;
		counts = o.counts;
		n = o.n;
		pvals = o.pvals;
		pcounts= o.pcounts;
		valType = NonAlloced;
		countType = NonAlloced;

		return *this;
	}


};

template<typename T> __global__ void mergeCASetsKernel(OrderedCountedSet<T>* pset,
		int* len, const T* src1, const T* src2, int p1, int p2, int n1, int n2);

template <typename T> class CuSet {
	static T Delta;
	/*
	 * two ordered sets [Aa,Ab] & [Ba,Bb] may intersect (or not, as in AbeforeB)
	 * 'x then y' ===> 'x <= y'
	 *
	 * todo setReduction -> (v1,v2,pool) -> first v1 comp'd to v2 and result(s) placed in ordered pool
	 * shuffleSet (warp's worth of values)
	 * 	each call each thread compares two sorted lists la lb (and output a single sorted list lo) (first call lists are 1 elem)
	 * 		s.t. lo.len <= la.len + lb.len ( = for the disjoint case)
	 *  when there are 64 list remaining, shuffle version
	 */

public:
	__inline__ __host__ __device__ static int partition(T* a, int lo, int hi);
	__inline__ __host__ __device__ static int partition(OrderedCountedSet<T>&, int lo, int hi);
	__host__ __device__ static void dedup(T* a, int* newN, int oldN);

	/* out alloced for oldN elements */
	template<template<typename > class OrderedCountedSet> __host__ __device__ static void dedupOCS(
			OrderedCountedSet<T>& out, const T* a, int oldN);
	__host__ __device__ static void merge(T** buffer, int* newN, int scratchLen, int oldN1, int oldN2);
	__host__ __device__ static void quicksort(T* a, int lo, int hi);
	__host__ __device__ static void quicksort(OrderedCountedSet<T>& ca);
	__host__ __device__ static void quicksort(OrderedCountedSet<T>& ca, int lo, int hi);
	__host__ __device__ static void quicksortIterative(T* a, int* scratch, int lo, int hi);

	template <template <typename> class OrderedCountedSet> __host__ __device__ static int mergeSortedCtrAry(OrderedCountedSet<T>& out, const OrderedCountedSet<T>& sorted1, const OrderedCountedSet<T>& sorted2);
	__inline__ __host__ __device__ static int mergeSorted(T* out,  T const * const sorted1, T const * const  sorted2, const int len1, const int len2, enum cudaMemcpyKind aOutKind= cudaMemcpyDefault, enum cudaMemcpyKind bOutKind = cudaMemcpyDefault, const int pout=1, const int p1 =1, const int p2=1, cudaStream_t stream = 0);
	__inline__ __host__ __device__ static int mergeSorted2(T* out,  T const * const sorted1, T const * const  sorted2, const int len1, const int len2, const int pout=1,const  int p1=1, const int p2=1);

	__host__ __device__ static OrderedIntersectionType getOrderedIntersectionType(T firstA, T lastA, T firstB, T lastB);
	template<typename PackType> __host__ __device__ static OrderedIntersectionType getOrderedIntersectionType(const PackType abBounds);
	template <template <typename> class OrderedCountedSet> __host__ __device__ static OrderedIntersectionType getOrderedIntersectionType(const OrderedCountedSet<T>& s1, OrderedCountedSet<T>& s2) {
		return getOrderedIntersectionType( s1.vals, s2.vals, s1.n, s2.n);
	}
/*
	__host__ static auto ordXsectTypeStr(OrderedIntersectionType oxt)  {
		return OrderedIntersectionTypeStr[oxt];
	}
*/
	//template <typename F> __host__ static F jaccard(set<T> s1, set<T> s2); // ||intersect(s1,s2)||/||union(s1,s2)||
	__host__ static double jaccard(set<T> s1, set<T> s2); // ||intersect(s1,s2)||/||union(s1,s2)||

	static __host__ __device__ void countSort(T* orderSet, int* counts, long*len, int ptout, int piout, const T* input, int pin, long n);

	static __host__ __device__ int bpSearch(  T const * const elems, int n, const T& trg, int p = 1 );
	template <template <typename> class OrderedCountedSet, template <typename> class Counter> static __host__ __device__ int bpSearch(const OrderedCountedSet<T>& elems, const Counter<T>& trg ) {
		return bpSearch(elems.vals,elems.n, trg.val);
	}


	template<template<typename > class Counter> static __host__ __device__ void toCounters( Counter<T>** ctts, AllocType type, const T* src, int n, int p = 1);
	/*
	 * 32  threads have sorted lists
	 * lower 16 compare to upper 16, 16 lists (merged) survive
	 */
	__device__  void shflset();

	/*
	 *
	 */
	__host__ __device__ static void toSortedColumnSets( CuMatrix<T>& sortedVals, CuMatrix<int>& counts, const CuMatrix<T>& m);

	__host__ __device__ static void columnHistogram( CuMatrix<T>& out, const CuMatrix<T>& m) {
		// for each column, create counter

	}

};

