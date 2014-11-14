/*
 * AnomalyDetection_.cu
 *
 *  Created on: Oct 27, 2013
 *      Author: reid
 */

#include "CuMatrix.h"
#include "AnomalyDetection.h"
#include "LuDecomposition.h"
#include "Kernels.h"
//  val twoPi = 2 * math.Pi
#include <math.h>
#include <utility>

#ifdef CuMatrix_Enable_Cdp

template<typename T> __global__
void selectThresholdKernel(T* f1s, CuMatrix<T> yValidation, CuMatrix<T> pdv, T pdvMin,
		T pdvMax, ulong n) {

	ulong gIdx = threadIdx.x + blockIdx.x * blockDim.x; // 1d mapped to probability spectrum
	T epsilon = pdvMin + (pdvMax - pdvMin) * gIdx / n; //  this is epsilon
	//T f1 = 0;
	cudaStream_t stream[4];
	for(int i = 0; i < 4; i++) {
		cherr(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
	}
	cudaEvent_t start_event, stop_event;

	cherr(cudaEventCreateWithFlags(&start_event, cudaEventDisableTiming));
	cherr(cudaEventCreateWithFlags(&stop_event, cudaEventDisableTiming));

	CuMatrix<T> cvPredictions = pdv < epsilon;

	//T truePos = yValidation.sum();
	if (checkDebug(debugAnomDet)) {
		ostreamlike s;
	//	soutln("yValidation truePos " << truePos);
		sout("yValidation " );
		yValidation.printShortString();
		sout("pdv ");
		pdv.printShortString();
	}
	T falsePos = 0, falseNeg = 0;
	T precision, recall;
    // these all launches kernes
	eqUnaryOp<T> eqOnef = Functory<T, eqUnaryOp>::pinch((T)1);
	eqUnaryOp<T> eqZerof = Functory<T, eqUnaryOp>::pinch((T)0);
	CuMatrix<T> cvEq1 =  cvPredictions.unaryOp(eqOnef,stream[0]); // cvPredictions == 1;
	CuMatrix<T> cvEq0 =  cvPredictions.unaryOp(eqZerof,stream[1]); // cvPredictions == 1;
	CuMatrix<T> yvEq1 =  yValidation.unaryOp(eqOnef,stream[2]); // cvPredictions == 1;
	CuMatrix<T> yvEq0 =  yValidation.unaryOp(eqZerof,stream[3]); // cvPredictions == 1;

	cherr(cudaEventRecord(stop_event, 0));
#ifndef __CUDA_ARCH__
	cherr(cudaDeviceSynchronize());   // block until the event is actually recorded
#else
	__syncthreads();
#endif

	T truePos = ((cvPredictions == 1) && (yValidation == 1)).sum();
	falsePos = ((cvPredictions == 1) && (yValidation == 0)).sum();
	falseNeg = ((cvPredictions == 0) && (yValidation == 1)).sum();


	precision = 1.0 * truePos / (truePos + falsePos);
	recall = 1.0 * truePos / (truePos + falseNeg);
	f1s[gIdx] = 2 * precision * recall / (precision + recall);
	// and then a Max reduction on fs[] (saving index in fs of max and using that to find corr. epsilon)
}
template __global__ void selectThresholdKernel(float*, CuMatrix<float>,
		CuMatrix<float>, float, float, ulong);
template __global__ void selectThresholdKernel(double*, CuMatrix<double>,
		CuMatrix<double>, double, double, ulong);
#endif
