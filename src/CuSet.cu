/*
 *
 * CuSet.cu
 */

#include "CuSet.h"
#include "CuMatrix.h"
#include <algorithm>

string OrderedIntersectionTypeStr[] { "AbeforeB", // A first
		"AonB", "BinA", "BonA",  // B first
		"AinB", "BbeforeA" };

__host__ __device__ enum cudaMemcpyKind copyKind(AllocType trg,  AllocType src) {
	if(trg==NonAlloced||src == NonAlloced) {
		return cudaMemcpyDefault;
	}
	if(trg == CudaHostAlloced ) {
		return src == CudaHostAlloced ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost;
	}else {
		return src == CudaHostAlloced ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
	}
}

template<typename T> __host__ __device__ int CuSet<T>::
bpSearch(
		T const * const elems,
		int n,
		const T& trg,
		int p) {
	int idx = n / 2;
	int idxP1 = idx + 1;
	int idxM1 = idx - 1;
	int nM1 = n - 1;
	T mid = elems[idx* p];
	if (trg - mid > util<T>::epsilon()) {		//( mid.before(trg)
			// search upper half
		if (trg >= elems[idxP1* p] && trg <= elems[nM1* p]) {
			if (n - idxP1 > 0) {
				return idxP1 + bpSearch(elems + idxP1 * p, n - idxP1, trg, p);
			} else
				return n;
		}

		else if (trg < elems[idxP1 * p]) {
			return idxP1;
		} else if (trg > elems[nM1 * p]) {
			return n;
		}

	} else if (mid - trg > util<T>::epsilon()) { // (mid.after(trg(
		if (trg >= elems[0] && idxM1 > -1 && trg <= elems[idxM1* p]) {
			return bpSearch(elems, idx, trg, p); // search lower half
		} else if (trg < elems[0]) {
			return 0;
		}
	}
	return idx;
}

template<typename T> __inline__ __host__ __device__ int CuSet<T>::partition(
		T* a, int lo, int hi) {
	T pivot = a[hi];
	int i = lo - 1;
	T tmp = a[0];
	for (int j = lo; j < hi; j++) {
		if (a[j] < pivot) {
			i++;
			tmp = a[i];
			a[i] = a[j];
			a[j] = tmp;
		}
	}
	i++;
	tmp = a[i];
	a[i] = a[hi];
	a[hi] = tmp;
	return i;
}

template<typename T> __inline__ __host__ __device__ int CuSet<T>::partition(
		OrderedCountedSet<T>& ca, int lo, int hi) {
	T pivot = ca.vals[hi];
	int i = lo - 1;
	T tmp = ca.vals[0];
	int tcount = ca.counts[0];
	for (int j = lo; j < hi; j++) {
		if (ca.vals[j] < pivot) {
			i++;
			tmp = ca.vals[i];
			tcount = ca.counts[i];
			ca.vals[i] = ca.vals[j];
			ca.counts[i] = ca.counts[j];
			ca.vals[j] = tmp;
			ca.counts[j] = tcount;
		}
	}
	i++;
	tmp = ca.vals[i];
	tcount = ca.counts[i];
	ca.vals[i] = ca.vals[hi];
	ca.counts[i] = ca.counts[hi];
	ca.vals[hi] = tmp;
	ca.counts[hi] = tcount;
	return i;
}


template<typename T> __host__ __device__ void CuSet<T>::quicksort(T* a, int lo,
		int hi) {
	if (lo < hi) {
		int p = partition(a, lo, hi);
		quicksort(a, lo, p - 1);
		quicksort(a, p + 1, hi);
	}
}

template<typename T> __host__ __device__ void CuSet<T>::quicksort(OrderedCountedSet<T>& ca, int lo,
		int hi) {
	if (lo < hi) {
		int p = partition(ca, lo, hi);
		quicksort(ca, lo, p - 1);
		quicksort(ca, p + 1, hi);
	}
}
template<typename T> __host__ __device__ void CuSet<T>::quicksort(OrderedCountedSet<T>& ca) {
	return CuSet<T>::quicksort(ca,0,ca.n-1);
}
template<typename T> __host__ __device__ void CuSet<T>::quicksortIterative(T* a,
		int* stack, int l, int h) {
	int top = -1;

	// push initial values of l and h to stack
	stack[++top] = l;
	stack[++top] = h;

	// Keep popping from stack while is not empty
	while (top >= 0) {
		// Pop h and l
		h = stack[top--];
		l = stack[top--];

		// Set pivot element at its correct position
		// in sorted array
		int p = partition(a, l, h);

		// If there are elements on left side of pivot,
		// then push left side to stack
		if (p - 1 > l) {
			stack[++top] = l;
			stack[++top] = p - 1;
		}

		// If there are elements on right side of pivot,
		// then push right side to stack
		if (p + 1 < h) {
			stack[++top] = p + 1;
			stack[++top] = h;
		}
	}
	stack[0] = top;
}

template<typename T> __host__ __device__ int3 ffmerge(T* out, const T* sorted1,
		const T* sorted2, int len1, int len2) {
	T c1, c2;
	int i1 = 0, i2 = 0, io = 0;
	c1 = sorted1[i1];
	c2 = sorted2[i2];
	bool working = true;
	while (working) {
		if (c1 < c2) {
			out[io++] = c1;
			i1++;
			if (i1 < len1)
				c1 = sorted1[i1];
			else {
				if (i2 < len2) {
					memcpy(out + io, sorted2 + i2, tSz<T>(len2 - i2));
					io += len2 - i2;
					i2 = len2;
				}
				working = false;
			}
		} else if (c2 < c1) {
			out[io++] = c2;
			i2++;
			if (i2 < len2)
				c2 = sorted2[i2];
			else {
				if (i1 < len1) {
					memcpy(out + io, sorted1 + i1, tSz<T>(len1 - i1));
					io += len1 - i1;
					i1 = len1;
				}
				working = false;
			}
		} else {
			out[io++] = c1;
			i1++;
			i2++;
			if (i1 < len1 && i2 < len2) {
				c1 = sorted1[i1];
				c2 = sorted2[i2];
			} else {
				if (i1 < len1) {
					memcpy(out + io, sorted1 + i1, tSz<T>(len1 - i1));
					io += len1 - i1;
					i1 = len1;
				} else if (i2 < len2) {
					memcpy(out + io, sorted2 + i2, tSz<T>(len2 - i2));
					io += len2 - i2;
					i2 = len2;
				}
				working = false;
			}
		}
	}
	int3 res;
	res.x = io;
	res.y = i1;
	res.z = i2;
	return res;
}

template<typename T, template<typename > class OrderedCountedSet> __host__ __device__ int3 ffmergeCtrAry(
		OrderedCountedSet<T>& out, const OrderedCountedSet<T>& sorted1,
		const OrderedCountedSet<T>& sorted2) {
	Counter<T> c1, c2;
	int i1 = 0, i2 = 0, io = 0;
	c1 = sorted1.get(i1);
	c2 = sorted2.get(i2);
	bool working = true;
	while (working) {
		if (c1 < c2) {
			out.set(io++, c1);
			i1++;
			if (i1 < sorted1.n)
				c1 = sorted1.get(i1);
			else {
				if (i2 < sorted2.n) {
					OrderedCountedSet<T>::memcpy(out + io, sorted2 + i2,
							sorted2.n - i2);
					io += sorted2.n - i2;
					i2 = sorted2.n;
				}
				working = false;
			}
		} else if (c2 < c1) {
			out.set(io++, c2);
			i2++;
			if (i2 < sorted2.n)
				c2 = sorted2.get(i2);
			else {
				if (i1 < sorted1.n) {
					OrderedCountedSet<T>::memcpy(out + io, sorted1 + i1,
							sorted1.n - i1);
					io += sorted1.n - i1;
					i1 = sorted1.n;
				}
				working = false;
			}
		} else {
			out.set(io++, c1);
			i1++;
			i2++;
			if (i1 < sorted1.n && i2 < sorted2.n) {
				c1 = sorted1.get(i1);
				c2 = sorted2.get(i2);
			} else {
				if (i1 < sorted1.n) {
					OrderedCountedSet<T>::memcpy(out + io, sorted1 + i1,
							sorted1.n - i1);
					io += sorted1.n - i1;
					i1 = sorted1.n;
				} else if (i2 < sorted2.n) {
					OrderedCountedSet<T>::memcpy(out + io, sorted2 + i2,
							sorted2.n - i2);
					io += sorted2.n - i2;
					i2 = sorted2.n;
				}
				working = false;
			}
		}
	}
	int3 res;
	res.x = io;
	res.y = i1;
	res.z = i2;
	return res;
}

template<typename T> __host__ __device__ OrderedIntersectionType CuSet<T>::getOrderedIntersectionType(
		T firstA, T lastA, T firstB, T lastB) {
	if (firstA <= firstB) {
		if (lastA <= firstB) {
			// 		firstA ... lastA
			//						    firstB ... lastB
			return AbeforeB;
		} else {
			if (lastA <= lastB) {
				// 		firstA ... lastA
				//			    firstB ... lastB
				return AonB;
			} else {
				// 		firstA  ...  lastA
				//		 firstB ... lastB
				return BinA;
			}
		}
	} else {
		if (firstA > lastB) {
			// 						firstA ... lastA
			//	firstB ... lastB
			return BbeforeA;
		} else {
			if (lastB <= lastA) {
				// 			firstA ... lastA
				//	firstB ... lastB
				return BonA;
			}
		}
	}
	// 		firstA ... lastA
	//	firstB 		...	 lastB
	return AinB;
}

/*
 * caller's responsibility to point out at a large enough buffer (at least lenA+lenB for end cases
 */
template<typename T> __host__ __device__ int CuSet<T>::mergeSorted(T* out,
		T const * const  sortedA,  T const * const  sortedB, const int lenA, const int lenB, enum cudaMemcpyKind aOutKind, enum cudaMemcpyKind bOutKind, const int pout, const int pA, const int pB, cudaStream_t stream) {
	T firstA = sortedA[0];
	T firstB = sortedB[0];
	T lastA = sortedA[(lenA - 1)*pA];
	T lastB = sortedB[(lenB - 1)*pB];

	OrderedIntersectionType oxsectType = getOrderedIntersectionType(firstA, lastA, firstB, lastB);
	int3 mergices;
	int firstBinAidx = -1;
	int firstAinBidx = -1;


	const int szT = sizeof(T);
	const int tlena = szT *lenA;
	const int tlenb = szT *lenB;
	switch (oxsectType) {
	case AbeforeB:
		//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
		cherr(cudaMemcpy2DAsync(out, pout, sortedA, pA, 1, tlena, aOutKind, stream));
		cherr(cudaMemcpy2DAsync(out + lenA, pout, sortedB, pB, 1, tlenb, bOutKind, stream));
		return lenA + lenB;
	case BbeforeA:
		cherr(cudaMemcpy2DAsync(out, pout, sortedB, pB, 1, tlenb,bOutKind, stream));
		cherr(cudaMemcpy2DAsync(out + lenB, pout, sortedA, pA, 1, tlena, aOutKind , stream));
		return lenA + lenB;
	case AonB:
	case BinA:
		firstBinAidx = bpSearch(sortedA, lenA, firstB);
		// copy begining of A up to index where firstB would insert
		//memcpy(out, sortedA, firstBinAidx * szT);
		cherr(cudaMemcpy2DAsync(out, pout, sortedA, pA, 1, firstBinAidx * szT,aOutKind, stream));
		if (oxsectType == AonB) {
			int lastAinBidx = bpSearch(sortedB, lenB, lastA); // last elem in sortedA falls sumairz inside sortedB
			mergices = ffmerge(out + firstBinAidx, sortedA + firstBinAidx,
					sortedB, lenA - firstBinAidx, lastAinBidx);
			//memcpy(out + firstBinAidx + mergices.x, sortedB + lastAinBidx, (lenB - lastAinBidx) * 1);
			cherr(
					cudaMemcpy2DAsync(out + firstBinAidx + mergices.x, pout,
							sortedB + lastAinBidx, pB, 1,
							(lenB - lastAinBidx) * szT, bOutKind,stream));
			return firstBinAidx + mergices.x + lenB - mergices.z;
		}
		// both firstB and lastB fall inside sortedA
		mergices = ffmerge<T>(out + firstBinAidx, sortedA + firstBinAidx,
				sortedB, lenA - firstBinAidx, lenB);
		//memcpy(out + firstBinAidx + mergices.x, sortedB + mergices.z, (lenB - mergices.z) * szT);
		cherr(
				cudaMemcpy2DAsync(out + firstBinAidx + mergices.x, pout,
						sortedB + mergices.z, pB, 1,
						(lenB - mergices.z) * szT, bOutKind,stream));
		return firstBinAidx + mergices.x + lenB - mergices.z;
	case BonA:
	case AinB:
		firstAinBidx = bpSearch(sortedB, lenB, firstA);
		// copy begining of B up to index where firstA would insert
		//memcpy(out, sortedB, firstAinBidx * szT);
		cherr(cudaMemcpy2DAsync(out, pout, sortedB, pB, 1, firstAinBidx * szT, bOutKind,stream));
		if (oxsectType == BonA) {
			int lastBinAidx = bpSearch(sortedA, lenA, lastB);
			mergices = ffmerge(out + firstAinBidx, sortedA,
					sortedB + firstAinBidx, lenA, lenB - lastBinAidx);
			// copy remainder of A
			//memcpy(out + firstAinBidx + mergices.x, sortedA + mergices.y, (lenA - mergices.y) * szT);
			cherr(cudaMemcpy2DAsync(out + firstAinBidx + mergices.x, pout, sortedA + mergices.y, pA, 1, (lenA - mergices.y) * szT, aOutKind, stream));
			return firstAinBidx + mergices.x + lenA - mergices.y;
		}
		// AinB
		mergices = ffmerge(out + firstAinBidx, sortedA, sortedB + firstAinBidx,
				lenA, lenB - firstAinBidx);
		//memcpy(out + firstAinBidx + mergices.x, sortedA + mergices.y, (lenA - mergices.y) * szT);
		cherr(cudaMemcpy2DAsync(out + firstAinBidx + mergices.x, pout, sortedA + mergices.y, pA, 1, (lenA - mergices.y) * szT, aOutKind, stream));
		return firstAinBidx + mergices.x + lenA - mergices.y;
	}
	return -1;
}

template<> template<> __host__ __device__ OrderedIntersectionType CuSet<double>::getOrderedIntersectionType<double4>( const double4 ptype) {
	return getOrderedIntersectionType(ptype.x, ptype.y, ptype.z, ptype.w);
}

template<typename T> __host__ __device__ int CuSet<T>::mergeSorted2(T* dst,
		T const * const  sortedA,  T const * const  sortedB, int lenA, int lenB, int pout, int p1, int p2) {
	const T* a_start = sortedA;
	const T* a_end = a_start + lenA;
	const T* b_start = sortedB;
	const T* b_end = b_start + lenB;

	while (a_start < a_end && b_start < b_end) {
		if (*a_start <= *b_start)
			*dst++ = *a_start++;// if elements are equal, then a[] element is output
		else
			*dst++ = *b_start++;
	}
	while (a_start < a_end)
		*dst++ = *a_start++;
	while (b_start < b_end)
		*dst++ = *b_start++;
	return 0;
}

//inline void merge_ptr( const _Type* a_start, const _Type* a_end, const _Type* b_start, const _Type* b_end, _Type* dst )
//{

/**
 * populate counters
 */
template<typename T, template<typename > class Counter>
__global__ void counterPopKernel(Counter<T>* ctrs, const T*src, int n, int p) {

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		ctrs[idx].val = src[idx * p];
		ctrs[idx].count = 1;
	}
}

template<typename T> template<template<typename > class Counter>
__host__ __device__ void CuSet<T>::toCounters(Counter<T>** ctrs, AllocType type,
		const T* src, int n, int p) {
	assert(type > NonAlloced);
	switch (type) {
	case CudaHostAlloced:
		cherr(cudaHostAlloc(ctrs, n * sizeof(Counter<T> )))
		;
		for (int i = 0; i < n; i++) {
			*ctrs[i].val = src[i * p];
			*ctrs[i].count = 1;
		}
		return;
	default:
		cherr(cudaMalloc(ctrs, n * sizeof(Counter<T> )))
		;
	}

	int threads;
	int blocks;

	::getReductionExecContext(blocks, threads, n);
	counterPopKernel<<<blocks, threads>>>(*ctrs, src, n, p);
	cherr(cudaDeviceSynchronize());
}

template<typename T> template<template<typename > class OrderedCountedSet>
__host__ __device__ int CuSet<T>::mergeSortedCtrAry(OrderedCountedSet<T>& out,
		const OrderedCountedSet<T>& sortedA, const OrderedCountedSet<T>& sortedB) {
	Counter<T> firstA = sortedA.get(0);
	Counter<T> firstB = sortedB.get(0);
	Counter<T> lastA = sortedA.get(sortedA.n - 1);
	Counter<T> lastB = sortedB.get(sortedB.n - 1);

	OrderedIntersectionType pxsectType = getOrderedIntersectionType(sortedA, sortedB);
	int3 mergices;
	int firstBinAidx = -1;
	int firstAinBidx = -1;
	switch (pxsectType) {
	case AbeforeB:
		OrderedCountedSet<T>::memcpy(out, sortedA);
		OrderedCountedSet<T>::memcpy(out + sortedA.n, sortedB);
		return sortedA.n + sortedB.n;
	case BbeforeA:
		OrderedCountedSet<T>::memcpy(out, sortedB);
		OrderedCountedSet<T>::memcpy(out + sortedB.n, sortedA);
		return sortedA.n + sortedB.n;
	case AonB:
	case BinA:
		firstBinAidx = bpSearch(sortedA, firstB);
		// copy begining of A up to index where firstB would insert
		OrderedCountedSet<T>::memcpy(out, sortedA, firstBinAidx);
		if (pxsectType == AonB) {
			int lastAinBidx = bpSearch(sortedB, lastA); // last elem in sortedA falls sumarez inside sortedB
			mergices = ffmergeCtrAry(out + firstBinAidx, sortedA + firstBinAidx,
					sortedB, sortedA.n - firstBinAidx, lastAinBidx);
			OrderedCountedSet<T>::memcpy(out + firstBinAidx + mergices.x,
					sortedB + lastAinBidx, (sortedB.n - lastAinBidx));
			return firstBinAidx + mergices.x + sortedB.n - mergices.z;
		}
		// both firstB and lastB fall inside sortedA
		mergices = ffmergeCtrAry<T>(out + firstBinAidx, sortedA + firstBinAidx,
				sortedB, sortedA.n - firstBinAidx, sortedB.n);
		OrderedCountedSet<T>::memcpy(out + firstBinAidx + mergices.x,
				sortedB + mergices.z, (sortedB.n - mergices.z));
		return firstBinAidx + mergices.x + sortedB.n - mergices.z;
	case BonA:
	case AinB:
		firstAinBidx = bpSearch(sortedB, firstA);
		// copy beginning of B up to index where firstA would insert
		OrderedCountedSet<T>::memcpy(out, sortedB, firstAinBidx);
		if (pxsectType == BonA) {
			int lastBinAidx = bpSearch(sortedA, lastB);
			mergices = ffmergeCtrAry(out + firstAinBidx, sortedA,
					sortedB + firstAinBidx, sortedA.n, sortedB.n - lastBinAidx);
			// copy remainder of A
			OrderedCountedSet<T>::memcpy(out + firstAinBidx + mergices.x,
					sortedA + mergices.y, (sortedA.n - mergices.y));
			return firstAinBidx + mergices.x + sortedA.n - mergices.y;
		}
		// AinB
		mergices = ffmergeCtrAry(out + firstAinBidx, sortedA,
				sortedB + firstAinBidx, sortedA.n, sortedB.n - firstAinBidx);
		OrderedCountedSet<T>::memcpy(out + firstAinBidx + mergices.x,
				sortedA + mergices.y, (sortedA.n - mergices.y));
		return firstAinBidx + mergices.x + sortedA.n - mergices.y;
	}
	return -1;
}

template<typename T> __host__ __device__ void CuSet<T>::dedup(T* a, int* newN,
		int oldN) {
	*newN = 0;
	T curr;
	for (int i = 0; i < oldN; i++) {
		int j = 0;
		curr = a[i];
		for (; j < *newN; j++) {
			if (a[j] == curr)
				break;
		}
		if (j == *newN) {
			a[j] = curr;
			(*newN)++;
		}
	}
}

template<typename T> template<template<typename > class OrderedCountedSet> __host__ __device__ void CuSet<
		T>::dedupOCS(OrderedCountedSet<T>& out, const T* a, int oldN) {
	int newN = 0;
	T curr;
	for (int i = 0; i < oldN; i++) {
		int j = 0;
		curr = a[i];
		for (; j < newN; j++) {
			if (fabs(out.vals[j] - curr) < util<T>::epsilon()) {
				out.counts[j]++;
				break;
			}
		}
		if (j == newN) {
			out.vals[j] = curr;
			out.counts[j] = 1;
			newN++;
		}
	}
	if (out.n > newN) {
		out.resize(newN);
	}
}
template __host__ __device__ void CuSet<float>::dedupOCS<OrderedCountedSet>(
		OrderedCountedSet<float>&, float const*, int);
template __host__ __device__ void CuSet<double>::dedupOCS<OrderedCountedSet>(
		OrderedCountedSet<double>&, double const*, int);
template __host__ __device__ void CuSet<unsigned long>::dedupOCS<OrderedCountedSet>(
		OrderedCountedSet<unsigned long>&, unsigned long const*, int);

/*
 * buffers offset from buffer by scratchN ('a') and scratchN + aN ('b") are already sorted sets
 * if they intersect,  zip them together
 * don't call if
 * 		a < b, can return buffer = a &  *mergedN =  aN + bN
 * 		b > a, can copy b to  buffer & (!using ultraslo memcpy)		    )
 */
template<typename T> __host__ __device__ void CuSet<T>::merge(T** buffer,
		int* mergedN, int scratchN, int aN, int bN) {
	T curr1, curr2;
	const T* a = *buffer + scratchN;
	const T* b = a + aN;
	int aIdx = 0, bIdx = 0, scratchIdx = 0;
	bool moreAsQ;
	bool moreBsQ;

	while ((moreAsQ = aIdx < aN) || (moreBsQ = bIdx < bN)) {
		if (moreAsQ)
			curr1 = a[aIdx];
		if (moreBsQ)
			curr2 = b[bIdx];
		if ((curr1 <= curr2) || (!moreBsQ && moreAsQ)) {
			*(*buffer + scratchIdx++) = curr1;
			aIdx++;
			if (curr1 == curr2) {
				bIdx++;
			}
		} else if ((curr2 < curr1) || (!moreAsQ && moreBsQ)) {
			*(*buffer + scratchIdx++) = curr2;
			bIdx++;
		}
	}
	flprintf("exiting final merged len (scratchIdx) %d\n", scratchIdx);
	*mergedN = scratchIdx;
}

/*
 * unsorted array -> OrderedCountedSet array
 * 		threaded over chunks that will fit in shmem (n *(size(source elem type T) +size(CA elem = type T + int)
 * 		chunk -> OrderedCountedSet
 *
 * OrderedCountedSet array
 * 		each thread launching k 4 merging two CAs
 */

template<typename T> __global__ void toOrderedCountedSet( ) {
}

template<typename T> __host__  __device__ void toOrderedCountedSetL( const T  *  src) {

}

template<typename T> __global__ void setMakerKernel(T* setPtr, int pset, int* len,
		const T* atts, int patts, long n) {
	// each block gets as big a chunk of an attribute columnn as will fit in shmem
	T* pshmem = SharedMemory<T>();
	T* psortedShmem = pshmem + blockDim.x;
	// copy globtoshmem
	int idx_g = blockDim.x * blockIdx.x + threadIdx.x;
	int idx_s = threadIdx.x;
	if (idx_g < n)
		pshmem[idx_s] = atts[idx_g * patts];
	else {
		flprintf("thread b %d g %d exceeded n %d\n", idx_s, idx_g, n);
		pshmem[idx_s] = (T) 0;
	}
	__syncthreads();
	*len = 0;
	if (threadIdx.x == 0) {
		CuSet<T>::quicksort(pshmem, 0, blockDim.x - 1);
		CuSet<T>::dedup(pshmem, len, blockDim.x - 1);
	}

	__syncthreads();

	if (idx_g < n)
		setPtr[idx_g * pset] = pshmem[idx_s];
	else
		setPtr[idx_g * pset] = (T) 0;
}

template<typename T> __global__ void setMakerIterativeKernel(T* setPtr, int* len, const int pset,
		const T* atts, const int patts,const long n) {
	// each block gets as big a chunk of an attribute columnn as will fit in shmem
	T* pshmem = SharedMemory<T>();
	//int* pscratch = (int*) (pshmem + blockDim.x);
	// copy globtoshmem
	int idx_g = blockDim.x * blockIdx.x + threadIdx.x;
	int idx_s = threadIdx.x;
	if (idx_g < n)
		pshmem[idx_s] = atts[idx_g];
	else
		pshmem[idx_s] = (T) 0;

	__syncthreads();
	/*	int* blockLen = len + blockIdx.x;

	if (threadIdx.x == 0) {
		CuSet<T>::quicksortIterative(pshmem, pscratch, 0, blockDim.x - 1);
		CuSet<T>::dedup(pshmem, blockLen, blockDim.x - 1);
	}


	__syncthreads();

	if (idx_g < n && idx_s <= *blockLen)
		setPtr[idx_g] = pshmem[idx_s];*/
}
template __global__ void setMakerIterativeKernel<float>(float*, int*,const int,
		float const*, const int,const long);
template __global__ void setMakerIterativeKernel<double>(double*, int*,const int,
		double const*, const int,const long);
template __global__ void setMakerIterativeKernel<unsigned long>(ulong*,	int*, const int,
		ulong const*, const int,const long);

template<typename T> __global__ void mergeSetsKernel(T* pset, int* len, const int ps, const T* src1,
		const T* src2, const int p1,  const int p2, const int n1, const int n2) {
	// each block gets as big a chunk of an attribute columnn as will fit in shmem
	T* pscratch = SharedMemory<T>();
	T* pshmem1 = pscratch + blockDim.x;
	T* pshmem2 = pshmem1 + blockDim.x;
	// copy globtoshmem
	int idx_g = blockDim.x * blockIdx.x + threadIdx.x;
	int idx_s = threadIdx.x;
	//T s1,s2;

	if (idx_g < n1) {
		pshmem1[idx_s] = src1[idx_g * p1];
	}
	if (idx_g < n2) {
		pshmem2[idx_s] = src2[idx_g * p2];
	}

	__syncthreads();

	// best perf if pscratch is copied to pset
	if (threadIdx.x == 0) {
		if (blockIdx.x == 0) {
			flprintf("len %p\n", len);
		}
		*(len + blockIdx.x) = 0;
		CuSet<T>::merge(&pscratch, len + blockIdx.x, blockDim.x, n1, n2);
	}

	__syncthreads();

	if (idx_g < *len)
		pset[idx_g] = pshmem1[idx_s];

}
template __global__ void mergeSetsKernel<float>(float*, int*, const int, const float*, const float*,
		const  int, const int, const int, const int);
template __global__ void mergeSetsKernel<double>(double*, int*, const int, const double*,
		const double*, const int, const int, const int, const int);
template __global__ void mergeSetsKernel<unsigned long>(unsigned long*, int*, const int,
		const unsigned long*, const unsigned long*, const int, const int, const int, const int);

template<typename T> __host__ double CuSet<T>::jaccard(set<T> s1, set<T> s2) {
	set<T> xsect;
	set<T> younion;
	set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
			std::inserter(xsect, xsect.begin()));
	set_union(s1.begin(), s1.end(), s2.begin(), s2.end(),
			std::inserter(younion, younion.begin()));
	return static_cast<double>(1.0 * (xsect.size()) / (younion.size()));
}

// todo
template<typename T> __host__ __device__ void CuSet<T>::countSort(T* orderSet, int* counts, long*len, int ptout, int piout, const T* input, int pin, long n) {
	int threads = n / 2;

	while (threads > 1) {

		threads /= 2;
	}

}


template<typename T> __host__ __device__ void CuSet<T>::toSortedColumnSets( CuMatrix<T>& sortedVals, CuMatrix<int>& counts,  const CuMatrix<T>& m) {
	CuMatrix<T> mTxp = m.transpose();
	// check out mat size
	// for each column, (now row) launch kernel with 32x1 blocks
	//int initialWarps = DIV_UP(m.m, WARP_SIZE);
	for(int col = 0; col < m.n; col++) {
		DMatrix<T> d_m, d_trgv;
		m.tile0(d_m,m.lastMod == mod_host);
		sortedVals.tile0(d_trgv,sortedVals.lastMod == mod_host);
		DMatrix<int> d_counts;
		counts.tile0(d_counts,counts.lastMod == mod_host);

	}
}

template struct CuSet<float> ;
template struct CuSet<double> ;
template struct CuSet<ulong> ;
template struct CuSet<int> ;
