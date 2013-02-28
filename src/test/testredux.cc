#include "../Matrix.h"
#include "../util.h"
#include "tests.h"

template int testRedux<float>::operator()(int argc, char const ** args) const;
template int testRedux<double>::operator()(int argc, char const ** args) const;
template <typename T> int testRedux<T>::operator()(int argc, const char** args) const {
	Matrix<T> incrRows = Matrix<T>::increasingRows(0, 200, 100);
	outln("incrRows " << incrRows.syncBuffers());
	T incrRowsSum = incrRows.sum();
	outln("incrRows sum " << incrRowsSum);
	Matrix<T> col1, col3;
	incrRows.submatrix(col1,incrRows.m,1,0,0);
	incrRows.submatrix(col3,incrRows.m,1,0,2);
	T col1Sum = col1.sum();
	T col3Sum = col3.sum();
	outln("col1Sum sum " << col1Sum);
	outln("col3Sum sum " << col3Sum);

	Matrix<T> b = Matrix<T>::zeros(60,1);
	for(int i = 1; i < 60; i++) {
		b = b |= (Matrix<T>::ones(60,1) * i);
	}
	outln("b " << b.syncBuffers());
	Matrix<T> bcol1, bcol3;
	b.submatrix(bcol1,b.m,1,0,0);
	b.submatrix(bcol3,b.m,1,0,2);
	T bcol1Sum = bcol1.sum();
	T bcol3Sum = bcol3.sum();
	outln("bcol1Sum sum " << bcol1Sum);
	outln("bcol3Sum sum " << bcol3Sum);

	return 0;
}

template int testColumnAndRowSum<float>::operator()(int argc, char const ** args) const;
template int testColumnAndRowSum<double>::operator()(int argc, char const ** args) const;
template <typename T> int testColumnAndRowSum<T>::operator()(int argc, const char** args) const {
	Matrix<T> incrRows = Matrix<T>::increasingRows(0, 200, 100);
	int count = b_util::getCount(argc,args,1000);
    CuTimer timer;
    float exeTime;

    timer.start();
    T sumSum = 0;
	for(int i = 0; i < count; i++) {
		sumSum += incrRows.columnSum(5);
	}
    exeTime = timer.stop();

	outln(count << " runs of columnsum == " << sumSum << " and took " << exeTime);

    timer.start();
    sumSum = 0;
    Matrix<T> col5;
    incrRows.submatrix(col5,incrRows.m,1,0,5);
	for(int i = 0; i < count; i++) {
		sumSum += col5.sum();
	}
	exeTime = timer.stop();

	outln(count << " runs of col5 sum == " << sumSum << " and took " << exeTime);

    timer.start();
    sumSum = 0;
	for(int i = 0; i < count; i++) {
		sumSum += incrRows.rowSum(5);
	}
	exeTime = timer.stop();

	outln(count << " runs of rowSum == " << sumSum << " and took " << exeTime);

    timer.start();
    sumSum = 0;
    Matrix<T> row5;
    incrRows.submatrix(row5,1,incrRows.n,5,0);
	for(int i = 0; i < count; i++) {
		sumSum += row5.sum();
	}
	exeTime = timer.stop();

	outln(count << " runs of row5 sum == " << sumSum << " and took " << exeTime);

	return 0;
}

template int testSumLoop<float>::operator()(int argc, char const ** args) const;
template int testSumLoop<double>::operator()(int argc, char const ** args) const;
template <typename T> int testSumLoop<T>::operator()(int argc, const char** args) const {
	Matrix<T> m = Matrix<T>::ones(5000,400);
	ulong matBytes = m.size;
    CuTimer timer;

    T lastSumSum = 0;

    string max;

	int count = b_util::getCount(argc,args,1000);
	float memFlowIter =0, maxFlow = 0;
	for(int grids = 16; grids < 256; grids *= 2) {
		for(int blocks = 32; blocks <=1024; blocks *= 2) {
		    timer.start();
		    T sumSum = 0;
			for (int i = 0; i < count; i++) {
				sumSum += m.sum();
			}
			if(lastSumSum == 0) {
				lastSumSum = sumSum;
			} else {
				dassert(util<T>::almostEquals(lastSumSum, sumSum));
				lastSumSum = sumSum;
			}
			float exeTime = timer.stop();

		    memFlowIter = 2.f * 1000.f * matBytes / Giga / (exeTime / count);
		    stringstream ss;
			ss  << "grids " << grids << ", blocks " << blocks << ", sumsum " << sumSum << " took exeTime " << (exeTime /1000) << "s or flow of " << memFlowIter << "GB/s" << endl;
			if(memFlowIter > maxFlow) {
				max = ss.str();
				maxFlow = memFlowIter;
			}
			cout << ss.str();
		}
	}
	cout << "max " << max;
	return 0;
}

template int testNneqP<float>::operator()(int argc, char const ** args) const;
template int testNneqP<double>::operator()(int argc, char const ** args) const;
template <typename T> int testNneqP<T>::operator()(int argc, const char** args) const {
	Matrix<T> bigSeq = Matrix<T>::sequence(0,20,20).syncBuffers();
	Matrix<T> chunk1, chunk2;
	bigSeq.submatrix(chunk1,4,4,2,2);
	bigSeq.submatrix(chunk2,4,4,6,6);
	outln("bigSeq " << bigSeq);
	outln("chunk1 " << chunk1);
	outln("chunk2 " << chunk2);
	Matrix<T> chunks = chunk1 + chunk2;
	outln("chunks " << chunks.syncBuffers());
	return 0;
}

template int testEqualsEtc<float>::operator()(int argc, char const ** args) const;
template int testEqualsEtc<double>::operator()(int argc, char const ** args) const;
template <typename T> int testEqualsEtc<T>::operator()(int argc, const char** args) const {
	Matrix<T> ms = Matrix<T>::sin(40, 39);
	Matrix<T> ms2 = Matrix<T>::sin(40, 39);
	Matrix<T> ms3 = Matrix<T>::sin(40, 39);
	outln("\nms " << ms.toShortString() << "\nms2 " << ms2.toShortString() << "\nms3 " << ms3.toShortString());
	//ms3.set(5, 5, 5);
	CuMatrix<T>::set(ms3.d_elements, ms3.m, ms3.n, ms3.p, 5, 5, 5);
	outln("ms == ms2 " << tOrF(ms == ms2));
	outln("ms == ms3 " << tOrF(ms == ms3));
	assert( (ms == ms2));
	assert( !( ms == ms3));
	Matrix<T> bigZ = Matrix<T>::zeros(929,987);
	Matrix<T> medZ = Matrix<T>::zeros(500,500);
	assert(bigZ.zeroQ());
	assert(medZ.zeroQ());
	//medZ.set(303,452,1);
	CuMatrix<T>::set(medZ.d_elements, medZ.m, medZ.n, medZ.p, 303, 452, 1);

	outln("modified medZ != 0 " << tOrF(!medZ.zeroQ()));
	outln("\n\n\n");
	assert(!medZ.zeroQ());

	Matrix<T> bigOnes = Matrix<T>::ones(1000,500);
	Matrix<T> bigI = Matrix<T>::identity(1000);
	T sumBigOnes_0 = bigOnes.columnSum(0);
	T sumBigOnes_50 = bigOnes.columnSum(50);
	bool excepted = false;
	try {
		 bigOnes.columnSum(733);
	} catch (const columnOutOfBounds& me) {
		excepted = true;
	}
	outln("excepted " << tOrF(excepted) );
	assert(excepted);
	T sumBigOnes_433 = bigOnes.columnSum(433);
	T sumBigI = bigI.columnSum(0);
	outln("sumBigOnes_0 " << sumBigOnes_0 );
	outln("sumBigOnes_50 " << sumBigOnes_50 );
	outln("sumBigOnes_433 " << sumBigOnes_433 );
	outln("sumBigI " << sumBigI );

	outln("bigOnes.hasBiasColumn() " << tOrF(bigOnes.hasBiasColumn()));
	outln("bigI.hasBiasColumn() " << tOrF(bigI.hasBiasColumn()));
	assert(!bigI.hasBiasColumn());
	Matrix<T> withBias = bigI.addBiasColumn();
	outln("withBias.hasBiasColumn() " << tOrF(withBias.hasBiasColumn()));
	return 0;

}

#include "tests.cc"
