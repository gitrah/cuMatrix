/*
 * testrowredux.cu
 *
 *  Created on: May 26, 2014
 *      Author: reid
 */
#include "tests.h"

#include "../CuMatrix.h"
#include "../util.h"
#include "../MatrixExceptions.h"
#include "../Maths.h"
#include "testKernels.h"

template int testReduceRows<float>::operator()(int argc, const char **argv) const;
template int testReduceRows<double>::operator()(int argc, const char **argv) const;
template int testReduceRows<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testReduceRows<T>::operator()(int argc, const char **argv) const {
	outln("testReduceRows start " );
	plusBinaryOp<T> plus = Functory<T,plusBinaryOp>::pinch();
	int start = b_util::getStart(argc,argv,1);

	for(int i = 0; i < 65; i++ ) {
		outln( "scanSumL(" << i <<") " << scanSumL(i));
	}
	for(int i = start; i < 65; i++ ) {
		outln("i " << i << ", inclusum " << scanSumL(i-1));
		CuMatrix<T> m1 = CuMatrix<T>::increasingColumns(0,1024,i);
		outln("m1 " << m1.syncBuffers());

		CuMatrix<T> resVec = CuMatrix<T>::zeros(m1.m,1);

		DMatrix<T> d_res, d_m1;
		m1.tile0(d_m1,true);
		resVec.tile0(d_res,true);

		CuMatrix<T>::reduceRows(d_res,d_m1,plus);

		T rvSum = resVec.sum();
		uint inclusum = scanSumL(i-1);
		outln("resVec " << resVec.syncBuffers() << "\nresVec.sum() " << rvSum);
		outln("should equals rows X inclusive sum (" << m1.m << " X " << inclusum << ")");
		assert(resVec.sum() ==  inclusum * d_res.m);
	}

	ulong len = 2 * Mega;
	CuMatrix<T> bigm1 = CuMatrix<T>::increasingColumns(len,64,0);
	outln("bigm1 " << bigm1.syncBuffers());
	T bigm1sum = bigm1.sum();
	T check  = 2l * Mega * 2016l;
	outln("bigm1.sum " << bigm1sum << ", check " << check);
	assert(bigm1sum  == check);

	CuMatrix<T> bigResVec = CuMatrix<T>::zeros(bigm1.m,1);

	DMatrix<T> d_bigres, d_bigm1;
	bigm1.tile0(d_bigm1,true);
	bigResVec.tile0(d_bigres,true);

	setCurrGpuDebugFlags( debugRedux,true,false);
	CuMatrix<T>::reduceRows(d_bigres,d_bigm1,plus);
	setCurrGpuDebugFlags( ~debugRedux,false,true);

	outln("bigResVec " << bigResVec.syncBuffers());
	assert(bigResVec.sum() == 2016 * d_bigres.m);


	/*

	CuMatrix<T> tinyOnes = CuMatrix<T>::ones(50,1);
	CuMatrix<T> tiny = tinyOnes |= (2 * tinyOnes);
	outln("tiny " << tiny.syncBuffers());
	outln("tiny col 0 sum " << tiny.reduceColumn(plus,0,0));
	outln("tiny col 1 sum " << tiny.reduceColumn(plus,0,1));

	CuMatrix<T> ones = CuMatrix<T>::ones(len,1);
	checkCudaError(cudaGetLastError());
	T colOneSum = ones.columnSum(0);
	checkCudaError(cudaGetLastError());
	outln("ones.colSum(0) " << colOneSum);

	T onesum = ones.sum();
	assert(colOneSum == onesum);
	outln("passed assert(colOneSum == onesum)");
*/
	return 0;
}


