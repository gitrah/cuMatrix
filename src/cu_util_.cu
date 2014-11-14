/*
 * cu_util_.cu
 *
 *  Created on: Jul 19, 2014
 *      Author: reid
 */

#include "cu_util.h"
#include <helper_cuda.h>
#include "caps.h"
#include <string>
#include <sstream>
#include "DMatrix.h"
#include "CuMatrix.h"
#include <float.h>
#include <limits>
#include "Maths.h"

/*


__host__ __device__ int b_util::countSpanrows( uint m, uint n, uint warpSize ) {
	uint num = MAX(n,warpSize), den = MIN(n,warpSize);
	int div = num/den;
	if( div* den == num) {
		//flprintf("div %d * num %d (== %d)\n", div, num, (div * num));
		return 0;
	}
	uint sf = smallestFactor(n);
	uint warps = DIV_UP(m * n,warpSize);

	//flprintf("sf %u n/sf %u warps %d\n", sf, n/sf, warps);
	uint factor = sf == n ? n : n/sf;
	return warps * (factor-1)/factor;
	//flprintf("factor (%u) s.t. (uint) (  m * (1. * (factor-1)/(factor))) == %u\n", factor, sr);
}

__host__ __device__ bool b_util::spanrowQ( uint row, uint n, uint warpSize) {
#ifdef __CUDA_ARCH__
	return ::spanrowQ(row, n);
#else
	uint warpS = row * n / warpSize;
	uint warpE = (row + 1 ) * n / warpSize;
	return warpS != warpE;
#endif
}

template<typename T> __host__ __device__ void cu_util<T>::prdm(const DMatrix<T>& md) {
	printf("dmat d: %p (%u*%u*%u)\n", md.elements, md.m,md.n,md.p);
}
template  __host__ __device__ void cu_util<float>::prdm(const DMatrix<float>& md);
template  __host__ __device__ void cu_util<double>::prdm(const DMatrix<double>& md);
template  __host__ __device__ void cu_util<ulong>::prdm(const DMatrix<ulong>& md);

template<typename T> __host__ __device__ void cu_util<T>::printDm( const DMatrix<T>& dm , const char* msg) {
	uint size = dm.m*dm.p;
	printf("%s (%d*%d*%d) %d &dmatrix=%p elements %p\n",msg,dm.m,dm.n,dm.p,size, &dm,dm.elements);
	T * elems = NULL;
#ifndef __CUDA_ARCH__
	if(dm.elements) {
		checkCudaError(cudaHostAlloc(&elems, size,0));
		checkCudaError(cudaMemcpy(elems,dm.elements, size, cudaMemcpyDeviceToHost));
	}
#else
	elems = dm.elements;
#endif
	if(!elems) {
		printf("printDm nothing to see here\n");
		return;
	}

	bool header = false;
	if (checkDebug(debugVerbose) || (dm.m < CuMatrix<T>::getMaxRowsDisplayed() && dm.n < CuMatrix<T>::getMaxColsDisplayed())) {
		for (uint i1 = 0; i1 < dm.m; i1++) {
			if(!header) {
				printf("-");
				for (uint j1 = 0; j1 < dm.n; j1++) {
					if(j1 % 10 == 0) {
						printf(" %d", j1/10);
					}else {
						printf("  ");
					}
					printf(" ");
				}
				printf("\n");
				header = true;
			}
			printf("[");
			for (uint j1 = 0; j1 < dm.n; j1++) {

				if(sizeof(T) == 4)
					printf("% 2.10g", elems[i1 * dm.p + j1]);  //get(i1,j1) );
				else
					printf("% 2.16g", elems[i1 * dm.p + j1]); // get(i1,j1) );
						//);
				if (j1 < dm.n - 1) {
					printf(" ");
				}
			}
			printf("] ");
			if(i1 % 10 == 0) {
				printf("%d", i1);
			}

			printf("\n");
		}
		if(header) {
			printf("+");
			for (uint j1 = 0; j1 < dm.n; j1++) {
				if(j1 % 10 == 0) {
					printf(" %d",j1/10);
				}else {
					printf("  ");
				}
				printf(" ");
			}
			printf("\n");
			header = false;
		}

	} else {
		for (uint i2 = 0; i2 < CuMatrix<T>::getMaxRowsDisplayed() + 1 && i2 < dm.m; i2++) {
			if (i2 == CuMatrix<T>::getMaxRowsDisplayed()) {
				printf(".\n.\n.\n");
				continue;
			}
			for (uint j2 = 0; j2 < CuMatrix<T>::getMaxColsDisplayed() + 1 && j2 < dm.n; j2++) {
				if (j2 == CuMatrix<T>::getMaxColsDisplayed()) {
					printf("...");
					continue;
				}
				if(sizeof(T) == 4)
					printf("% 2.10g", elems[i2 * dm.p + j2]); //get(i2,j2));
				else
					printf("% 2.16g", elems[i2 * dm.p + j2]); //get(i2,j2));
						//elements[i2 * p + j2]);
				if (j2 < dm.n - 1) {
					printf(" ");
				}
			}
			printf("\n");
		}
		if (dm.m > CuMatrix<T>::getMaxRowsDisplayed()) {
			for (uint i3 =dm.m - CuMatrix<T>::getMaxRowsDisplayed(); i3 < dm.m; i3++) {
				if (dm.n > CuMatrix<T>::getMaxColsDisplayed()) {
					for (uint j3 = dm.n - CuMatrix<T>::getMaxColsDisplayed(); j3 < dm.n; j3++) {
						if (j3 == dm.n - CuMatrix<T>::getMaxColsDisplayed()) {
							printf("...");
							continue;
						}
						if(sizeof(T) == 4)
							printf("% 2.10g", elems[i3 * dm.p + j3]);//get(i3, j3));
						else
							printf("% 2.16g", elems[i3 * dm.p + j3]); //get(i3,j3));
								//elements[i3 * p + j3]);
						if (j3 < dm.n - 1) {
							printf(" ");
						}
					}
				} else {
					for (uint j4 = 0; j4 < dm.n; j4++) {
						if(sizeof(T) == 4)
							printf("% 2.10g", elems[i3 * dm.p + j4]); // get(i3,j4));
						else
							printf("% 2.16g", elems[i3 * dm.p + j4]); //get(i3,j4));
								//elements[i3 * p + j4]);

						if (j4 < dm.n - 1) {
							printf(" ");
						}
					}

				}
				printf("\n");
			}
		} else { //if(dm.m > 10) -> dm.n > 10
			for (uint i5 = 0; i5 < CuMatrix<T>::getMaxRowsDisplayed() + 1 && i5 < dm.m; i5++) {

				if (dm.n > CuMatrix<T>::getMaxColsDisplayed()) {
					for (uint j5 = dm.n - CuMatrix<T>::getMaxColsDisplayed(); j5 < dm.n; j5++) {
						if (j5 == dm.n - CuMatrix<T>::getMaxColsDisplayed()) {
							printf("...");
							continue;
						}
						T t = elems[i5 * dm.p + j5];

						if(sizeof(T) == 4)
							printf("% 2.10g", t);
						else
							printf("% 2.16g", t);
						if (j5 < dm.n - 1) {
							printf(" ");
						}
					}
				} else {
					for (uint j4 = 0; j4 < dm.n; j4++) {
						if(sizeof(T) == 4)
							printf("% 2.10g", elems[i5 * dm.p + j4]); //get(i5,j4));
						else
							printf("% 2.16g", elems[i5 * dm.p + j4]); //get(i5,j4));

						if (j4 < dm.n - 1) {
							printf(" ");
						}
					}
				}

				printf("\n");
			}

		}
	}
#ifndef __CUDA_ARCH__
	if(elems) {
		checkCudaErrors(cudaFreeHost(elems));
	}
#endif

}
template void cu_util<float>::printDm(DMatrix<float> const&,char const*);
template void cu_util<double>::printDm(DMatrix<double> const&,char const*);
template void cu_util<ulong>::printDm(DMatrix<ulong> const&,char const*);

template<typename T> __host__ __device__ void cu_util<T>::printRow(const DMatrix<T>& dm, uint row) {
	prlocf("printRow\n");
	if(!dm.elements) {
		printf("row %d: null elements\n", row);
		return;
	}
	ulong idx = row * dm.p;
	T* elems;
#ifndef __CUDA_ARCH__
	uint rowSize = dm.n*sizeof(T);
	checkCudaError(cudaHostAlloc(&elems, rowSize,0));
	checkCudaError(cudaMemcpy(elems, dm.elements + idx, rowSize, cudaMemcpyHostToDevice));
#else
	elems = dm.elements;
#endif
	printf("row %d: ", row);
	for(int c = 0; c < dm.n; c++) {
		printf("%5.2f ", elems[idx + c]);
	}
	printf("\n");
#ifndef __CUDA_ARCH__
	checkCudaError(cudaFreeHost(elems));
#endif
}
template void cu_util<float>::printRow(DMatrix<float> const&,uint);
template void cu_util<double>::printRow(DMatrix<double> const&,uint);
template void cu_util<ulong>::printRow(DMatrix<ulong> const&,uint);
*/
