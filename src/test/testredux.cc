#include "../CuMatrix.h"
#include "../util.h"
#include "../Kernels.h"
#include "tests.h"
#include "../MatrixExceptions.h"
#include "testKernels.h"

template int testRedux<float>::operator()(int argc, const char **argv) const;
template int testRedux<double>::operator()(int argc, const char **argv) const;
template int testRedux<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testRedux<T>::operator()(int argc, const char **argv) const {
/*	outln("testRedux start " );
	checkCudaError(cudaGetLastError());
	for(ulong i = 16777215l; i < 128* Mega + 9; i += 1) {
		CuMatrix<T> onz = CuMatrix<T>::ones(i, 1);
		ulong sum = onz.sum();
		ulong ksum = onz.syncBuffers().kahanSum();
		outln("i "<<i << ", sum " << sum <<", ksum " << ksum);
		assert(i == sum);
	}

	CuMatrix<T> incrRows = CuMatrix<T>::increasingRows(200, 100,(T)0);
	CuMatrix<T> onez = CuMatrix<T>::ones(200, 100);
	outln("incrRows " << incrRows.syncBuffers());
	outln("onez " << onez.syncBuffers());
	T incrRowsSum = incrRows.sum();
	T onezSum = onez.sum();
	outln("incrRows sum " << incrRowsSum);
	outln("onez sum " << onezSum);
	//CuMatrix<T> longy = CuMatrix<T>::randn(1,33554432/16, 10);

	CuMatrix<T> longy = CuMatrix<T>::ones(1, 33554432/16);
	outln("longy " << longy.syncBuffers());
	CuMatrix<T> col1, col3;
	incrRows.submatrix(col1,incrRows.m,1,0,0);
	incrRows.submatrix(col3,incrRows.m,1,0,2);
	T col1Sum = col1.sum();
	T col3Sum = col3.sum();
	T longySum  = longy.sum();
	outln("col1Sum sum " << col1Sum);
	outln("col3Sum sum " << col3Sum);
	outln("longy sum " << longySum);

	CuMatrix<T> b = CuMatrix<T>::zeros(60,1);
	for(int i = 1; i < 60; i++) {
		b = b |= (CuMatrix<T>::ones(60,1) * i);
	}
	outln("b " << b.syncBuffers());
	CuMatrix<T> bcol1, bcol3;
	b.submatrix(bcol1,b.m,1,0,0);
	b.submatrix(bcol3,b.m,1,0,2);
	T bcol1Sum = bcol1.sum();
	T bcol3Sum = bcol3.sum();
	outln("bcol1Sum sum " << bcol1Sum);
	outln("bcol3Sum sum " << bcol3Sum);

	return 0;
	*/
	outln("testRedux start " );
	checkCudaError(cudaGetLastError());
	CuMatrix<T> incrRows = CuMatrix<T>::increasingRows(200, 100,(T)0);
	CuMatrix<T> onez = CuMatrix<T>::ones(200, 100);
	outln("incrRows " << incrRows.syncBuffers());
	outln("onez " << onez.syncBuffers());
	T incrRowsSum = incrRows.sum();
	T onezSum = onez.sum();
	outln("incrRows sum " << incrRowsSum);
	assert( util<T>::almostEquals(1990000, incrRowsSum));
	outln("onez sum " << onezSum);
	cherr(cudaGetLastError());
	CuMatrix<T> longy = CuMatrix<T>::randn(1,33554432/64, 10);
	outln("longy " << longy.syncBuffers());
	CuMatrix<T> col0, col2;
	incrRows.submatrix(col0,incrRows.m,1,0,0);
	outln("col0.toss " << col0.toss());
	outln("col0 " << col0);
	incrRows.submatrix(col2,incrRows.m,1,0,2);
	outln("col2.toss " << col2.toss());
	outln("col2 " << col2);
	T col0Sum = col0.sum();
	T col2Sum = col2.sum();
	outln("col0Sum sum " << col0Sum);
	outln("col2Sum sum " << col2Sum);
	assert(col0Sum == 19900);
	assert(col2Sum == 19900);
	T longySum  = longy.sum();
	outln("longy sum " << longySum);

	CuMatrix<T> b = CuMatrix<T>::zeros(60,1);
	for(int i = 1; i < 60; i++) {
		b = b |= (CuMatrix<T>::ones(60,1) * i);
	}
	outln("b " << b.syncBuffers());
	CuMatrix<T> bcol1, bcol3;
	b.submatrix(bcol1,b.m,1,0,0);
	b.submatrix(bcol3,b.m,1,0,2);
	T bcol1Sum = bcol1.sum();
	T bcol3Sum = bcol3.sum();
	outln("bcol1Sum sum " << bcol1Sum);
	outln("bcol3Sum sum " << bcol3Sum);

	return 0;

}


template int testColumnRedux<float>::operator()(int argc, const char **argv) const;
template int testColumnRedux<double>::operator()(int argc, const char **argv) const;
template int testColumnRedux<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testColumnRedux<T>::operator()(int argc, const char **argv) const {
	outln("testColumnRedux start " );

	//plusBinaryOp<T> pbo = Functory<T,plusBinaryOp>::pinch();

	plusBinaryOp<T> pbo = Functory<T, plusBinaryOp>::pinch();

/*
	CuMatrix<T> tinyOnes = CuMatrix<T>::ones(50,1);
	CuMatrix<T> tiny = tinyOnes |= (2 * tinyOnes);
	outln("tiny " << tiny.syncBuffers());
	outln("tiny col 0 sum " << tiny.reduceColumn(plus,0,0));
	outln("tiny col 1 sum " << tiny.reduceColumn(plus,0,1));

*/
	ulong len =  16 * Mega;
	CuMatrix<T> ones = CuMatrix<T>::ones(len,1);
	checkCudaError(cudaGetLastError());
	ones.syncBuffers();
	T colOneSum = ones.columnSum(0);
	checkCudaError(cudaGetLastError());
	outln("ones.colSum(0) " << colOneSum);

	T onesum = ones.sum();
	assert(colOneSum == onesum);
	outln("passed assert(colOneSum == onesum)");
	outln("ones.m " << ones.m << " ones.sum " << onesum);
	T reduceColOneSum = ones.reduceColumn(pbo,0,0);
	outln("ones.reduceColumn(plus,0,0) " << reduceColOneSum);
	assert(colOneSum == reduceColOneSum);

	checkCudaError(cudaGetLastError());
	CuMatrix<T> twos = ones * 2;
	T twosum = twos.sum();
	assert(twosum == 2 * onesum);
	checkCudaError(cudaGetLastError());
	CuMatrix<T> oneTwos = ones |= twos;
	oneTwos.syncBuffers();
	T reduceOneTwoColOneSum = oneTwos.reduceColumn(pbo,0,0);
	T reduceOneTwoColTwoSum = oneTwos.reduceColumn(pbo,0,1);
	outln("oneTwos.reduceColumn(plus,0,0) " << reduceOneTwoColOneSum);
	outln("oneTwos.reduceColumn(plus,0,1) " << reduceOneTwoColTwoSum);

	outln("twos.m " << twos.m << " twos.sum " << twos.sum());
	checkCudaError(cudaGetLastError());
	outln("twos.colSum(0) " << twos.columnSum(0));
	checkCudaError(cudaGetLastError());
	T reduceColOneTwosSum = twos.reduceColumn(pbo,0,0);
	outln("twos.reduceColumn(plus,0,0) " << reduceColOneTwosSum);

	outln("oneTwos " << oneTwos.syncBuffers());
	outln("oneTwos.sum " << oneTwos.sum());

	assert(onesum == colOneSum && colOneSum == reduceColOneSum);

	checkCudaError(cudaGetLastError());
	CuMatrix<T> oneTwoThrees = oneTwos |= (ones * 3);
	oneTwoThrees.syncBuffers();
	outln("onwTwoThrees " << oneTwoThrees);
	checkCudaError(cudaGetLastError());

	uint count = b_util::getCount(argc,argv,10);
    CuTimer timer;
    float colSumTime, reduceColTime;

    T sumSum;

    sumSum = 0;
    timer.start();
	for(uint i = 0; i < count; i++) {
		sumSum += oneTwoThrees.columnSum(2);
	}
	colSumTime = timer.stop();
	outln(count << " columSums took " << colSumTime << " sumSum " << sumSum);
	outln( " count  " <<  count );
	outln( " len  " <<  len );
	outln( " count * 3 * len " <<  count * 3 * len);
	assert(sumSum == count * 3 * len);
	checkCudaError(cudaGetLastError());

    sumSum = 0;
    timer.start();
	for(uint i = 0; i < count; i++) {
		sumSum += oneTwoThrees.columnReduceIndexed( pbo, 2, 0);
		//sumSum += oneTwoThrees.reduceColumn(pbo,0,2);
	}
	reduceColTime = timer.stop();
	outln(count << " reduceColum took " << reduceColTime << " sumSum " << sumSum);
	assert(sumSum == count * 3 * len);
	outln("reduceCol takes " << 100*(1 - (colSumTime-reduceColTime)/colSumTime) << "% of time of colSum");
	return 0;
}


template int testShuffle<float>::operator()(int argc, const char **argv) const;
template int testShuffle<double>::operator()(int argc, const char **argv) const;
template int testShuffle<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testShuffle<T>::operator()(int argc, const char **argv) const {
	int len = 32;
	//size_t size = len * sizeof(T);
	T ary[len];
	T total = 0;
	for(int i = 0; i < len; i++) {
		ary[i] = 2 * i;
		total += 2 * i;
	}
	outln("total " << total << ", ary " << util<T>::parry( ary,len));
	plusBinaryOp<T> plus = Functory<T,plusBinaryOp>::pinch();
	multBinaryOp<T> mult = Functory<T,multBinaryOp>::pinch();

	outln("shuffln len " << len << " bop " << typeid(plus).name() << " nice " << nicen(plus));
	outln("nice " << nicen(util<T>));
	T res = shuffle(ary,len, plus);
	outln("res " << res);
	assert(res == total);
	T ary2[5] = {1,3,5,7,9};
	outln("ary2 " << util<T>::parry( ary2, 5));
	T res2 = shuffle(ary2,5, plus);
	outln("calcing pres2....");
	T pres2 = shuffle(ary2,5, mult);
	T total2 = 1 + 3 + 5 + 7 + 9;
	T ptotal2 = 1 * 3 * 5 * 7 * 9;
	outln("total2 " << total2);
	outln("ptotal2 " << ptotal2);
	outln("res2 " << res2);
	outln("pres2 " << pres2);
	assert(res2 ==total2);
	T ary3[8] = {1,3,5,7,9,11,13,0};
	T res3 = shuffle(ary3,8, plus);
	T total3 = total2 + 11 + 13;
	outln("total3 " << total3);
	outln("res3 " << res3);
	assert(res3 == total3);
//	T ary3b[7] = {1,3,5,7,9,11,13};
	T res3b = shuffle(ary3,7, plus);
	T total3b = total2 + 11 + 13;
	outln("total3b " << total3b);
	outln("res3b " << res3b);
	assert(res3b == total3b);


	launchTestShuffleKernel<T>();

	return 0;
}

template int testColumnAndRowSum<float>::operator()(int argc, const char **argv) const;
template int testColumnAndRowSum<double>::operator()(int argc, const char **argv) const;
template <typename T> int testColumnAndRowSum<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> incrRows = CuMatrix<T>::increasingRows(200, 100,(T)0);
	uint count = b_util::getCount(argc,argv,1000);
    CuTimer timer;
    float exeTime;

    timer.start();
    T sumSum = 0;
	for(uint i = 0; i < count; i++) {
		sumSum += incrRows.columnSum(5);
	}
    exeTime = timer.stop();

	outln(count << " runs of columnsum == " << sumSum << " and took " << exeTime);

    timer.start();
    sumSum = 0;
    CuMatrix<T> col5;
    incrRows.submatrix(col5,incrRows.m,1,0,5);
	for(uint i = 0; i < count; i++) {
		sumSum += col5.sum();
	}
	exeTime = timer.stop();

	outln(count << " runs of col5 sum == " << sumSum << " and took " << exeTime);

    timer.start();
    sumSum = 0;
	for(uint i = 0; i < count; i++) {
		sumSum += incrRows.rowSum(5);
	}
	exeTime = timer.stop();

	outln(count << " runs of rowSum == " << sumSum << " and took " << exeTime);

    timer.start();
    sumSum = 0;
    CuMatrix<T> row5;
    incrRows.submatrix(row5,1,incrRows.n,5,0);
	for(uint i = 0; i < count; i++) {
		sumSum += row5.sum();
	}
	exeTime = timer.stop();

	outln(count << " runs of row5 sum == " << sumSum << " and took " << exeTime);

	return 0;
}

template int testSumLoop<float>::operator()(int argc, const char **argv) const;
template int testSumLoop<double>::operator()(int argc, const char **argv) const;
template <typename T> int testSumLoop<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> m = CuMatrix<T>::ones(5000,400);
	ulong matBytes = m.size;
    CuTimer timer;

    T lastSumSum = 0;

    string max;

	int count = b_util::getCount(argc,argv,1000);
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

template int testNneqP<float>::operator()(int argc, const char **argv) const;
template int testNneqP<double>::operator()(int argc, const char **argv) const;
template <typename T> int testNneqP<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> bigSeq = CuMatrix<T>::sequence(0,20,20).syncBuffers();
	CuMatrix<T> chunk1, chunk2;
	bigSeq.submatrix(chunk1,4,4,2,2);
	bigSeq.submatrix(chunk2,4,4,6,6);
	outln("bigSeq " << bigSeq);
	outln("chunk1 " << chunk1);
	outln("chunk2 " << chunk2);
	CuMatrix<T> chunks = chunk1 + chunk2;
	outln("chunks " << chunks.syncBuffers());
	return 0;
}

template int testEqualsEtc<float>::operator()(int argc, const char **argv) const;
template int testEqualsEtc<double>::operator()(int argc, const char **argv) const;
template <typename T> int testEqualsEtc<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> ms = CuMatrix<T>::sin(40, 39);
	CuMatrix<T> ms2 = CuMatrix<T>::sin(40, 39);
	CuMatrix<T> ms3 = CuMatrix<T>::sin(40, 39);
	outln("\nms " << ms.toShortString() << "\nms2 " << ms2.toShortString() << "\nms3 " << ms3.toShortString());
	//ms3.set(5, 5, 5);
	setL<T>(ms3.tiler.currBuffer(), ms3.m, ms3.n, ms3.p, 5, 5, 5);
	outln("ms == ms2 " << tOrF(ms == ms2));
	outln("ms == ms3 " << tOrF(ms == ms3));
	assert( (ms == ms2));
	assert( !( ms == ms3));
	CuMatrix<T> bigZ = CuMatrix<T>::zeros(929,987);
	CuMatrix<T> medZ = CuMatrix<T>::zeros(500,500);
	assert(bigZ.zeroQ());
	assert(medZ.zeroQ());
	//medZ.set(303,452,1);
	setL<T>(medZ.tiler.currBuffer(), medZ.m, medZ.n, medZ.p, 303, 452, 1);

	outln("modified medZ != 0 " << tOrF(!medZ.zeroQ()));
	outln("\n\n\n");
	assert(!medZ.zeroQ());

	CuMatrix<T> bigOnes = CuMatrix<T>::ones(1000,500);
	CuMatrix<T> bigI = CuMatrix<T>::identity(1000);
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

	outln("bigOnes.biasedQ() " << tOrF(bigOnes.biasedQ()));
	outln("bigI.biasedQ() " << tOrF(bigI.biasedQ()));
	assert(!bigI.biasedQ());
	CuMatrix<T> withBias = bigI.addBiasColumn();
	outln("withBias.biasedQ() " << tOrF(withBias.biasedQ()));
	return 0;

}


template int testCount<float>::operator()(int argc, const char **argv) const;
template int testCount<double>::operator()(int argc, const char **argv) const;
template int testCount<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testCount<T>::operator()(int argc, const char **argv) const {
	outln("int testCount<T>::operator enter");
	CuMatrix<T> longC = CuMatrix<T>::ones(1000,1);
	T longCSum = longC.sum();
	outln("longCSum " << longCSum);
	checkCudaErrors(cudaDeviceSynchronize());
	bool longCsumIsCorrect = longCSum == longC.m * longC.n;
	assert(longCsumIsCorrect);
	outln("after asert");

	CuMatrix<T> mSin = CuMatrix<T>::sin(200,200, (T)10, (T)1,(T)0);
	outln("mSin" << mSin.syncBuffers());

	CuMatrix<T> ones = CuMatrix<T>::ones(200,200);
	CuMatrix<T> fives = 5 * ones;

	CuMatrix<T> mSinOdd = CuMatrix<T>::sin(199,201);
	CuMatrix<T> onesOdd = CuMatrix<T>::ones(199,201);
	CuMatrix<T> fivesOdd = 5 * onesOdd;

	CuMatrix<T> longy = CuMatrix<T>::ones(1,16000000);
	CuMatrix<T> longCol = CuMatrix<T>::ones(16000000,1);
	CuMatrix<T> longR = CuMatrix<T>::ones(1,1000);
	CuMatrix<T> longR2 = CuMatrix<T>::ones(1,1002);
	outln("longR " << longR.sum());
	outln("longR2 " << longR2.sum());

	T longySum = longy.sum();
	T longyMax = longy.max();
	T longyMin = longy.min();
	outln("longySum "<< longySum << ", "<< longyMax << ", " << longyMin);

	T mSinSum = mSin.sum();
	outln("mSinSum " << mSinSum);
	T onesSum = ones.sum();
	outln("onesSum " << onesSum);
	assert(onesSum == 40000);
	T fivesSum = fives.sum();
	outln("fivesSum " << fivesSum);
	assert(fivesSum == 200000);

	T mSinOddSum = mSinOdd.sum();
	outln("mSinOddSum " << mSinOddSum);
	T onesOddSum = onesOdd.sum();
	assert(onesOddSum == 39999);
	outln("onesOddSum " << onesOddSum);
	T fivesOddSum = fivesOdd.sum();
	outln("fivesOddSum " << fivesOddSum);
	assert(fivesOddSum == 199995);

	ltUnaryOp<T> lt = Functory<T,ltUnaryOp>::pinch(.5);
	gtUnaryOp<T> gt = Functory<T,gtUnaryOp>::pinch(10);

	almostEqUnaryOp<T> almeq =  Functory<T,almostEqUnaryOp>::pinch(1,0.0005);
	outln("mSin.max() " << mSin.max());
	outln("mSin.in() " << mSin.min());

	outln("mSin.any(almeq) " << mSin.any(almeq));
	outln("mSin.all(almeq) " << mSin.all(almeq));
	outln("mSin.none(almeq) " << mSin.none(almeq));

	outln("mSin.any(lt " << lt.comp_ro()<< ") " << mSin.any(lt));
	assert(mSin.any(lt));
	outln("mSin.all(lt " << lt.comp_ro()<< ") " << mSin.all(lt));
	assert(mSin.all(lt));
	outln("mSin.none(lt " << lt.comp_ro()<< ") " << mSin.none(lt));
	assert(!mSin.none(lt));

	assert(mSin.none(gt));

	outln("ones.any(almeq) " << ones.any(almeq));
	assert(ones.any(almeq));
	outln("ones.all(almeq) " << ones.all(almeq));
	assert(ones.all(almeq));
	outln("ones.none(almeq) " << ones.none(almeq));
	assert(!ones.none(almeq));

	outln("fives.any(almeq) " << fives.any(almeq));
	assert(!fives.any(almeq));
	outln("fives.all(almeq) " << fives.all(almeq));
	assert(!fives.all(almeq));
	outln("fives.none(almeq) " << fives.none(almeq));
	assert(fives.none(almeq));

	long  mSin_count = mSin.count(almeq);
	long ones_count = ones.count(almeq);
	outln("mSin.count(almeq(1)) " << mSin_count);
	outln("ones.count(almeq(1)) " << ones_count);
	assert(ones_count == ones.m * ones.n);
	assert(ones.count(almeq) == ones.m * ones.n);
	outln("fives.count(almeq(1)) " << fives.count(almeq));

	lt.comp() = .5;
	outln("mSin.count(lt(.5)) " << mSin.count(lt));
	outln("ones.count(lt(.5)) " << ones.count(lt));
	outln("fives.count(lt(.5)) " << fives.count(lt));

	return 0;

}
template int testEtoX<float>::operator()(int argc, const char **argv) const;
template int testEtoX<double>::operator()(int argc, const char **argv) const;
template int testEtoX<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testEtoX<T>::operator()(int argc, const char **argv) const {
/*
	CuTimer timer;
	timer.start();
	T smalle5 = eXL((T)5, 5);
	outln("smalle5 " << smalle5 << " took " << timer.stop());

	timer.start();
	T mede5 = eXL((T)5, 50);
	outln("mede5 " << mede5 << " took " << timer.stop());

	timer.start();
	T bige5 = eXL((T)5, 200);
	outln("bige5 " << bige5 << " took " << timer.stop());

*/
	return 0;
}


#include "tests.cc"
