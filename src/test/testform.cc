#include "tests.h"
#include "testKernels.h"
#include "../CuMatrix.h"
#include "../util.h"
#include "../caps.h"

template int testTiler<float>::operator()(int argc, const char **argv) const;
template int testTiler<double>::operator()(int argc, const char **argv) const;
template int testTiler<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testTiler<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> m = CuMatrix<T>::ones(2,2);

	Tiler<T>& tiler = m.tiler;

	tiler.gpuMask = 1 | 1 << 2 | 1 << 7;  // gpus 0, 3 & 7
	tiler.dumpMask( tiler.gpuMask);

	outln("index of 0 " << tiler.indexInMask(0));
	outln("index of 1" << tiler.indexInMask(1));
	outln("index of 2 " << tiler.indexInMask(2));
	outln("index of 3 " << tiler.indexInMask(3));
	outln("index of 4 " << tiler.indexInMask(4));
	outln("index of 5 " << tiler.indexInMask(5));
	outln("index of 6 " << tiler.indexInMask(6));
	outln("index of 7 " << tiler.indexInMask(7));
	outln("index of 8 " << tiler.indexInMask(8));
	return 0;
}

template int testDTiler<float>::operator()(int argc, const char **argv) const;
template int testDTiler<double>::operator()(int argc, const char **argv) const;
template int testDTiler<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testDTiler<T>::operator()(int argc, const char **argv) const {


	DMatrix<T> src((T*)0x10000, 1000,1000, 1000);
	flprintf("src: %d*%d*%d\n",src.m, src.n, src.p);

	DTiler<T> dtiler(src, tdRows, 7);
	DMatrix<T> trg;
	dtiler.nextTile1D(trg);
	flprintf("tile deetz: trg %d*%d*%d\n",trg.m, trg.n, trg.p);
	int soFar=0;
	while( !trg.zeroMatrixQ()) {
		flprintf("step %d: \ntrg %s\n", soFar, util<T>::pdm(trg).c_str());
		printf("\n");
		dtiler.nextTile1D(trg);
		soFar++;
	}
}


template int testBinCat<float>::operator()(int argc, const char **argv) const;
template int testBinCat<double>::operator()(int argc, const char **argv) const;
template int testBinCat<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testBinCat<T>::operator()(int argc, const char **argv) const {
	setCurrGpuDebugFlags( debugVerbose,true,false);
	CuMatrix<T> a = CuMatrix<T>::seqMod(1, 32, 40, 1).syncBuffers();
	outln("a " << a.syncBuffers().toString());
	CuMatrix<T> b = a.toBinaryCategoryMatrix();
	outln("b " << b.syncBuffers().toString());
	CuMatrix<T> c = b.toMaxColumnIndexVector();
	outln("c " << c.syncBuffers().toString());
	CuMatrix<T> a_c = a-c;
	outln("a_c " << a_c.syncBuffers());
	//outln( "a == c " << (a == c));
	//outln( "a != c " << (a != c));
	outln("a minus c complete");
	outln( "a almostEq c " << (a.almostEq( c)));
	//outln( "c almostEq a " << (c.almostEq( a)));
	almostEqualsBinaryOp<T> op =  Functory<T,almostEqualsBinaryOp>::pinch((T)1e-6);

	outln(" op.epsilon " << op.epsilon_ro());
	CuMatrix<T> a_c2 = a.binaryOp(c, op);
	outln("a_c2 " << a_c2.syncBuffers());
	outln(" op.epsilon2 " << op.epsilon_ro());
	dassert(a.almostEq(c));
	outln("passed 1");
	CuMatrix<T> a2 = CuMatrix<T>::seqMod(1, 70, 40, 1);
	outln(a2.syncBuffers().toString());
	CuMatrix<T> b2 = a2.toBinaryCategoryMatrix();
	outln(b2.syncBuffers().toString());
	CuMatrix<T> c2 = b2.toMaxColumnIndexVector();
	outln(c2.syncBuffers().toString());
	dassert(a2.almostEq(c2 + static_cast<T>(1)));
	setCurrGpuDebugFlags( ~debugVerbose,false,true);
	return 0;
}

template int testFillXvsY<float>::operator()(int argc, const char **argv) const;
template int testFillXvsY<double>::operator()(int argc, const char **argv) const;
template <typename T> int testFillXvsY<T>::operator()(int argc, const char **argv) const {
	setCurrGpuDebugFlags( debugVerbose,true,false);
	int count = b_util::getCount(argc,argv,1000);
    CuTimer timer;
    timer.start();
	clock_t lastTime = clock();

	// rows
    for(int i = 0; i < count; i++) {
		CuMatrix<T> m = CuMatrix<T>::ones(5000,1);
	}

	double delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("rows " << count << " took " << delta << " secs");
    float exeTime = timer.stop();
    float memFlowIter = 1000.f * 5000 / Giga / (exeTime / count);
	outln("N " << count << " took exeTime " << (exeTime /1000) << "s or flow of " << memFlowIter << "GB/s");

	lastTime = clock();
	timer.start();
	// columns
	for(int i = 0; i < count; i++) {
		CuMatrix<T> m = CuMatrix<T>::ones(1,5000);
	}
	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("columns " << count << " took " << delta << " secs");
	exeTime = timer.stop();
    memFlowIter = 1000.f * 5000 / Giga / (exeTime / count);
	outln("N " << count << " took exeTime " << (exeTime /1000) << "s or flow of " << memFlowIter << "GB/s");

	return 0;
}

template int testCat<float>::operator()(int argc, const char **argv) const;
template int testCat<double>::operator()(int argc, const char **argv) const;
template int testCat<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testCat<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> s = CuMatrix<T>::ones(40, 1);
	outln("s.sum " << s.sum());
	CuMatrix<T> ss2 = CuMatrix<T>::ones(40, 40);
	outln("ss2.sum " << ss2.sum());
	CuMatrix<T> ss23 = ss2 * 3;
	outln("ss23.sum " << ss23.sum());
	CuMatrix<T> ss3 = CuMatrix<T>::ones(40, 40) * 3;
	outln("ss3.sum " << ss3.sum());
	CuMatrix<T> t = s |= CuMatrix<T>::ones(40, 40) * 3;
	outln("t.sum " << t.sum());
	outln("t.ss " << t.toShortString());
	outln("t after sum " << t.sum() << " mat " << t.syncBuffers());
	CuMatrix<T> s2 = CuMatrix<T>::ones(40,40) * 2;
	T s2sum = s2.sum();
	outln("s2sum " << s2sum);
	CuMatrix<T> t2 = s2 |= CuMatrix<T>::ones(40,1);
	outln("t2 " << (t2.syncBuffers()));
	T t2sum = t2.sum();
	outln("t2sum " << t2sum);
	assert(t2sum == 40 + s2sum);
	CuMatrix<T> t3= s2 /= CuMatrix<T>::ones(1,40);
	outln("t3 " << t3.syncBuffers());
	T t3sum = t3.sum();
	outln("t3sum " << t3sum);
	assert(t3sum == t2sum);
	CuMatrix<T> s3 = s.rightConcatenate(CuMatrix<T>::ones(40, 1));
	outln("s final " << (s3.syncBuffers()).toString());

	CuMatrix<T> big = CuMatrix<T>::ones(400,400) * 2;
	T bs = big.sum();

	CuMatrix<T> bigger = big /= big;
	outln("bigger " << bigger.toShortString());
	//outln("bigger " << bigger.syncBuffers());
	CuMatrix<T> bigger2 = big |= big;
	outln("bigger2 " << bigger2.syncBuffers().toShortString());
	T bs2 = big.sum();

	T bbs = bigger.sum();
	T bbs2 = bigger2.sum();
	outln("bs " << bs << ", bs2 " << bs2);
	outln("bbs " << bbs);
	outln("bbs2 " << bbs2);
	dassert(bs * 2 == bbs);

	CuMatrix<T> lside = CuMatrix<T>::ones(200,100);
	CuMatrix<T> rside = 3 * lside;
	lside.syncBuffers();
	rside.syncBuffers();
	CuMatrix<T> comb = CuMatrix<T>::zeros(1,200*200);
	const CuMatrix<T>* pair[2];
	pair[0] = &lside;
	pair[1] = &rside;

	CuMatrix<T>::concat(comb, 2, pair);
	outln("comb.sum()  " << comb.sum() );
	outln("lside.sum()  " << lside.sum() );
	outln("rside.sum()  " << rside.sum() );
	dassert(comb.sum() == lside.sum() + rside.sum());



	int input_layer_size = 3;
	int hidden_layer_size = 5;
	int num_labels = 3;
	int m = 5;

	// We generate some 'random' test data
	CuMatrix<T> theta1 = CuMatrix<T>::sequence((T)0.5, hidden_layer_size, input_layer_size + 1).syncBuffers();
	outln("theta1\n" << theta1);
	T theta1Sum = theta1.sum();
	printDevArray<T>(theta1.currBuffer(), __FILE__ " imm bef concat theta1 dev ", __LINE__, MAX(theta1.m,theta1.n));
	printArray(theta1.currBuffer(), 20);
	CuMatrix<T> theta2 = CuMatrix<T>::sequence(hidden_layer_size * (input_layer_size + 1), num_labels, hidden_layer_size + 1).syncBuffers();
	T theta2Sum = theta2.sum();
	outln("theta2\n" << theta2);
	printDevArray<T>(theta1.currBuffer(), __FILE__ " imm bef concat theta2 dev ", __LINE__, MAX(theta2.m,theta2.n));
	CuMatrix<T> nn_params(1,(theta1.size + theta2.size)/sizeof(T),false,true);
	outln("nn_params " << nn_params.toShortString());
	const CuMatrix<T> * parts[] = {&theta1,&theta2};
	CuMatrix<T>::concat(nn_params,2, parts);
	printDevArray<T>(nn_params.currBuffer(), __FILE__ " imm after concat nn_params dev ", __LINE__, MAX(nn_params.m,nn_params.n));
	if(!nn_params.hostReadyQ()) {
		nn_params.syncBuffers();
	}
	outln("nn_params " << nn_params);
	T nn_paramSum= nn_params.sum();
	outln("theta1Sum " << theta1Sum);
	outln("theta2Sum " << theta2Sum);
	outln("nn_paramSum " << nn_paramSum);
	outln("diff " << (nn_paramSum - theta1Sum - theta2Sum));

	assert(util<T>::almostEquals(nn_paramSum, theta1Sum + theta2Sum, util<T>::epsilon()));


	return 0;
}

template int testMaxColIdxs<float>::operator()(int argc, const char **argv) const;
template int testMaxColIdxs<double>::operator()(int argc, const char **argv) const;
template <typename T> int testMaxColIdxs<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> a = CuMatrix<T>::randn(1000,10);
	outln(a.syncBuffers());
	CuMatrix<T> b = a.toMaxColumnIndexVector();
	outln(b.syncBuffers());
	return 0;
}

template int testAnonMatrices<float>::operator()(int argc, const char **argv) const;
template int testAnonMatrices<double>::operator()(int argc, const char **argv) const;
template int testAnonMatrices<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testAnonMatrices<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> a = CuMatrix<T>::ones(1000,1000);
	CuMatrix<T> b = 2 * CuMatrix<T>::ones(1000,1000);
	CuMatrix<T> c = a.transpose() * b.transpose();
	CuMatrix<T> d = a * b;
	T csum = c.sum();
	T dsum = d.sum();
	outln("csum " << csum << "  dsum " << dsum);
	assert(c.sum() == d.sum());
	return 0;
}

const char * txNames[] = {"transposeNaive","transposeCoalesced","transposeNoBankConflicts","transposeDiagonalKernel"};
template <typename T> __global__ void transposeNaive( T*, const T*, int, int, int, int);
template <typename T> __global__ void transposeCoalesced( T*, const T*, int, int, int, int);
template <typename T> __global__ void transposeNoBankConflictsPitch( T*, const T*, int, int, int, int);
template <typename T> __global__ void transposeDiagonalPitchKernel( T*, const T*, int, int, int, int);




template int testTranspose<float>::operator()(int argc, const char **argv) const;
template int testTranspose<double>::operator()(int argc, const char **argv) const;
template int testTranspose<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testTranspose<T>::operator()(int argc, const char **argv) const {

	CuMatrix<T> small = CuMatrix<T>::increasingColumns(40,40,1);
	outln("small " << small.syncBuffers());

	CuMatrix<T> smallT = small.transpose();
	outln("smallT " << smallT.syncBuffers());


	CuMatrix<T> smallNs = CuMatrix<T>::increasingColumns(40,20,1);
	outln("smallNs " << smallNs.syncBuffers());

	CuMatrix<T> smallTns = smallNs.transpose();
	outln("smallTns " << smallTns.syncBuffers());

	CuMatrix<T> twi = CuMatrix<T>::increasingColumns(1000,500,1);
	CuMatrix<T> ttwi = twi.transpose();
	outln("twi " << twi.syncBuffers());
	outln("ttwi " << ttwi.syncBuffers());
	CuMatrix<T> row0 = twi.rowMatrix(0);
	outln("row0 " << row0.syncBuffers());
	T twiRow0sum= row0.sum();
	CuMatrix<T> row0t = ttwi.columnMatrix(0);
	outln("row0' " << row0t.syncBuffers());
	T ttwiCol0sum = row0t.sum();
	outln("twiRow0sum " << twiRow0sum);
	outln("ttwiCol0sum " << ttwiCol0sum);
	assert(twiRow0sum == ttwiCol0sum );


	void (*kernels[4])( T*, const T*, int, int, int, int);
    void (*kernel)( T*, const T*, int, int, int, int);
	kernels[0] = &transposeNaive;
	kernels[1] = &transposeCoalesced;
	kernels[2] = &transposeNoBankConflictsPitch;
	kernels[3] = &transposeDiagonalPitchKernel;
	for(int i = 0; i < 4;i++) {
//	for(int i = 3; i < 4;i++) {
		kernel = kernels[i];
		outln("kernel " << txNames[i]);
		CuMatrix<T> ms = CuMatrix<T>::sequence(1,11,13);
		outln("ms\n" << ms.syncBuffers());

		CuMatrix<T> tms = ms.transposePitchKernelPtr(kernel);
		//printArray(tms.currBuffer(), 100);
		//printColoArray(tms.elements, 100);
		tms.syncBuffers();
		outln("tms\n" << tms);
		//printColoArray(tms.elements, 100);
		outln("tms ss " << tms.toShortString());
		CuMatrix<T> ttms = tms.transposePitchKernelPtr(kernel);
		if(checkDebug(debugCg))outln("cg1");

		setCurrGpuDebugFlags( debugVerbose,true,false);
		if(checkDebug(debugCg))outln("cg2");
		outln("ms " << ms);
		outln("ttms " << ttms);
		//printArray(ttms.currBuffer(), 100);
		ttms.syncBuffers();
		//printColoArray(ttms.elements, 100);
		tms.syncBuffers();
		outln("ttms2 " << ttms);
		outln("tms " << tms);
		outln("s- " << (ms - ttms).syncBuffers());
		setCurrGpuDebugFlags( ~debugVerbose,false,true);
		if(checkDebug(debugCg))outln("cg3");
		assert( ttms.almostEq(ms));

		CuMatrix<T> ms2a = CuMatrix<T>::sequence(2,22,24).syncBuffers();
		outln("ms2a " << ms2a);
		CuMatrix<T> ms2 = ms2a.addBiasColumn();
		outln("ms2 ss " << ms2.toShortString());
		ms2.syncBuffers();
		outln("ms2 " << ms2);
		CuMatrix<T> tms2 = ms2.transposePitchKernelPtr(kernel).syncBuffers();
		CuMatrix<T> ttms2 = tms2.transposePitchKernelPtr(kernel).syncBuffers();
		if(checkDebug(debugCg))outln("cg4");
		setCurrGpuDebugFlags( debugVerbose,true,false);
		if(checkDebug(debugCg))outln("cg5");
		tms2.syncBuffers();
		outln("tms2 " << tms2);
		outln("s2- " << (ms2 - ttms2).syncBuffers());
		setCurrGpuDebugFlags( ~debugVerbose,false,true);
		assert( ttms2.almostEq(ms2));

		CuMatrix<T> m = CuMatrix<T>::sequence(0,32*3+1-16, 32*3-16).addBiasColumn().syncBuffers();
		CuMatrix<T> tm = m.transposePitchKernelPtr(kernel).syncBuffers();
		CuMatrix<T> ttm = tm.transposePitchKernelPtr(kernel).syncBuffers();
		setCurrGpuDebugFlags( debugVerbose,true,false);
		if(checkDebug(debugCg))outln("cg6");
		outln("m " << m);
		outln("tm " << tm);
		outln("- " << (m - ttm).syncBuffers());
		setCurrGpuDebugFlags( ~debugVerbose,false,true);
		assert( ttm.almostEq(m));


		CuMatrix<T> m2 = CuMatrix<T>::sequence(0,16, 34).addBiasColumn().syncBuffers();
		CuMatrix<T> tm2 = m2.transposePitchKernelPtr(kernel).syncBuffers();
		CuMatrix<T> ttm2 = tm2.transposePitchKernelPtr(kernel).syncBuffers();
		setCurrGpuDebugFlags( debugVerbose,true,false);
		if(checkDebug(debugCg))outln("cg7");
		outln("m2 " << m2);
		outln("tm2 " << tm2);
		outln("- " << (m2 - ttm2).syncBuffers());
		setCurrGpuDebugFlags( ~debugVerbose,false,true);
		if(checkDebug(debugCg))outln("cg8");
		assert( ttm2.almostEq(m2));

		CuMatrix<T> dm = CuMatrix<T>::sequence(0, 65, 62).addBiasColumn().syncBuffers();
		CuMatrix<T> dtm = dm.transposePitchKernelPtr(kernel).syncBuffers();
		CuMatrix<T> dttm = dtm.transposePitchKernelPtr(kernel).syncBuffers();
		setCurrGpuDebugFlags( debugVerbose,true,false);
		outln("dm " << dm);
		outln("dtm " << dtm);
		outln("- " << (dm - dttm).syncBuffers());
		setCurrGpuDebugFlags( ~debugVerbose,false,true);
		assert( dttm.almostEq(dm));

		CuMatrix<T> mBig = CuMatrix<T>::sin(5000, 399).syncBuffers().addBiasColumn().syncBuffers();
		CuMatrix<T> tmBig = mBig.transposePitchKernelPtr(kernel).syncBuffers();
		CuMatrix<T> ttmBig = tmBig.transposePitchKernelPtr(kernel).syncBuffers();
		outln("mBig " << mBig);
		outln("tmBig " << tmBig);
		outln("- " << (mBig - ttmBig).syncBuffers());
		outln("sum " << (mBig - ttmBig).sum());
		assert( ttmBig.almostEq(mBig));
		outln("after assert( ttmBig.almostEq(mBig))" );

		CuMatrix<T> mTOdd= CuMatrix<T>::sin(23, 23).syncBuffers();
		outln("before tmTOdd");
		CuMatrix<T> tmTOdd= mTOdd.transposePitchKernelPtr(kernel);
		outln("before tttmOdd tmTOdd ss " << tmTOdd.toShortString());
		CuMatrix<T> tttmOdd = tmTOdd.transposePitchKernelPtr(kernel);
		outln("before tttmOdd.almostEq(mTOdd)");
		outln("tttmOdd " << tttmOdd.almostEq(mTOdd));

		CuMatrix<T> mTOddNs= CuMatrix<T>::sin(13, 23).syncBuffers();
		CuMatrix<T> tmTOddNs= mTOddNs.transposePitchKernelPtr(kernel);
		CuMatrix<T> tttmOddNs = tmTOddNs.transposePitchKernelPtr(kernel);
		outln("tttmOddNs " << tttmOddNs.almostEq(mTOddNs));

		CuMatrix<T> mBig2 = CuMatrix<T>::zeros(3220, 399).addBiasColumn();
		CuMatrix<T> tmBig2= mBig2.transposePitchKernelPtr(kernel);
		CuMatrix<T> ttmBig2 = tmBig2.transposePitchKernelPtr(kernel);
		outln("big2 " << ttmBig2.almostEq(mBig2));

		CuMatrix<T> mNonSquare= CuMatrix<T>::sin(512, 256).syncBuffers();
		CuMatrix<T> tmNonSquare = mNonSquare.transposePitchKernelPtr(kernel);
		CuMatrix<T> ttmNonSquare = tmNonSquare.transposePitchKernelPtr(kernel);
		outln("ttmNonSquare " << ttmNonSquare.almostEq(mNonSquare));

		CuMatrix<T> mOdd= CuMatrix<T>::sin(333, 333).syncBuffers();
		CuMatrix<T> tmOdd = mOdd.transposePitchKernelPtr(kernel);
		CuMatrix<T> ttmOdd = tmOdd.transposePitchKernelPtr(kernel);
		outln("ttmOdd " << ttmOdd.almostEq(mOdd));

		CuMatrix<T> mOddNs= CuMatrix<T>::sin(533, 333).syncBuffers();
		CuMatrix<T> tmOddNs = mOddNs.transposePitchKernelPtr(kernel);
		CuMatrix<T> ttmOddNs = tmOddNs.transposePitchKernelPtr(kernel);
		outln("ttmOddNS " << ttmOddNs.almostEq(mOddNs));

		CuMatrix<T> mSOdd= CuMatrix<T>::sin(33, 33).syncBuffers();
		CuMatrix<T> tmSOdd= mSOdd.transposePitchKernelPtr(kernel);
		CuMatrix<T> ttsmOdd = tmSOdd.transposePitchKernelPtr(kernel);
		outln("ttsmOdd " << ttsmOdd.almostEq(mSOdd));

		CuMatrix<T> mSOddNs= CuMatrix<T>::sin(33, 53).syncBuffers();
		CuMatrix<T> tmSOddNs= mSOddNs.transposePitchKernelPtr(kernel);
		CuMatrix<T> ttsmOddNs = tmSOddNs.transposePitchKernelPtr(kernel);
		outln("ttsmOddNs " << ttsmOddNs.almostEq(mSOddNs));
	}
	return 0;
}

extern int blockH;
template int testTransposeLoop<float>::operator()(int argc, const char **argv) const;
template int testTransposeLoop<double>::operator()(int argc, const char **argv) const;
template int testTransposeLoop<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testTransposeLoop<T>::operator()(int argc, const char **argv) const {
    CuTimer timer;
	outln("testTransposeLoop start");
	CuMatrix<T> seq = CuMatrix<T>::sequence(0,1024, 1024);
	outln("seq " << seq.syncBuffers());
	//
	CuMatrix<T> tx = CuMatrix<T>::zeros(1024, 1024);
	outln("tx " << tx.toss());
	ulong matB = tx.size;
	DMatrix<T> tx_d;
	tx.tile0(tx_d,false);
	int count = b_util::getCount(argc,argv,10000);
	void (*kernels[4])( T*, const T*,int,int,int,int);
    void (*kernel)(T*,const T*,int,int,int,int);
	kernels[0] = &transposeNaive;
	kernels[1] = &transposeCoalesced;
	kernels[2] = &transposeNoBankConflictsPitch;
	kernels[3] = &transposeDiagonalPitchKernel;
	outln("blockDim.y " << blockH);
	for(int k = 0; k < 4;k++) {
		kernel = kernels[k];
		outln("kernel " << txNames[k] << " at " << kernel);
	    outln(debugMem << debugFtor << debugCopy);
	    timer.start();
		clock_t lastTime = clock();

		for (int i = 0; i < count; i++) {
			seq.transposePitchKernelPtr(tx_d, kernel);
		}
		double delta = b_util::diffclock(clock(), lastTime) / 1000;
		outln("s " << count << " took " << delta << " secs");
	    float exeTime = timer.stop();
	    float memFlowIter = 2.f * 1000.f * matB / Giga / (exeTime / count);
		outln("N " << count << " took exeTime " << (exeTime /1000) << "s or flow of " << memFlowIter << "GB/s");
		T s1 = seq.sum();
		T s2 = tx.sum();
		outln("sanity " << s1 << " should == " << s2 << " (abs(delta)) =" << abs( s2-s1));
		assert(abs(s2 -s1) < 5e-5);
	}
	outln("testTransposeLoop finish");
	return 0;
}

template int testTransposeHuge<float>::operator()(int argc, const char **argv) const;
template int testTransposeHuge<double>::operator()(int argc, const char **argv) const;
template int testTransposeHuge<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testTransposeHuge<T>::operator()(int argc, const char **argv) const {
    CuTimer timer;
	outln("testTransposeHuge start");
	CuMatrix<T> m = CuMatrix<T>::increasingColumns(10000, 12000,1);

	outln("m " << m.syncBuffers());
	timer.start();
	clock_t lastTime = clock();

	CuMatrix<T> tm = m.transpose();

	double delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("m' " << m.toShortString() << " took " << delta << " secs");
    float exeTimeMs = timer.stop();
    outln("cutimer exeTime s " << exeTimeMs/1000);

	outln("tm " << tm.syncBuffers());

	T ms = m.sum();
	T tms = tm.sum();

	outln("ms " << ms);
	outln("tms " << tms);

	assert(util<T>::almostEqualsNormalized(  ms , tms, util<T>::epsilon()* 10));
	return 0;
}


template int testReshape<float>::operator()(int argc, const char **argv) const;
template int testReshape<double>::operator()(int argc, const char **argv) const;
template <typename T> int testReshape<T>::operator()(int argc, const char **argv) const {
	int width = 500;
	int height = 500;
	CuMatrix<T> ms = CuMatrix<T>::sin(height, width-1).addBiasColumn();
	CuMatrix<T> mc = CuMatrix<T>::cos(height, width-1).addBiasColumn();
	CuMatrix<T> flat = ms.poseAsCol() /= mc.poseAsCol();
	CuMatrix<T> msP = flat.reshape(height,width,0);
	CuMatrix<T> mcP = flat.reshape(height,width,height*width);
	dassert(ms.almostEq(msP));
	outln("ms sum " << ms.sum() << ", msP.sum " << msP.sum());
	dassert(mc.almostEq(mcP));
	outln("mc sum " << mc.sum() << ", mcP.sum " << mcP.sum());
	return 0;
}

template int testTransposeNneqP<float>::operator()(int argc, const char **argv) const;
template int testTransposeNneqP<double>::operator()(int argc, const char **argv) const;
template <typename T> int testTransposeNneqP<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> incrRows = CuMatrix<T>::increasingRows( 1000, 900,5).syncBuffers();
	CuMatrix<T> incrRowsCM = CuMatrix<T>::increasingRows( 1000, 900,5, true).syncBuffers();
	outln("a row of incrRows");
	for(int col = 0; col < incrRows.n; col++ ) {
		cout << incrRows.get(0,col);
	}
	cout << endl;
	outln("a row of incrRowsCM");
	for(int col = 0; col < incrRowsCM.n; col++ ) {
		cout << incrRowsCM.get(0,col);
	}
	cout << endl;

	CuMatrix<T> m1 = CuMatrix<T>::zeros(1,10,false).syncBuffers();
	CuMatrix<T> m2 = CuMatrix<T>::zeros(1,10,true).syncBuffers();
	for(int col = 0; col < 10; col++) {
		m1.set(0,col,col);
		m2.set(0,col,col);
	}
	m1.syncBuffers();
	m2.syncBuffers();
	outln("m1 " << m1);
	outln("m2 " << m2);

	outln("incrRows " << incrRows.toShortString() << ", " << incrRows.sum());
	outln("incrRows " << incrRows.syncBuffers());
	outln("incrRowsCM " << incrRowsCM.syncBuffers());
	CuMatrix<T> incrCols = CuMatrix<T>::increasingColumns( 1000, 900,5);
	outln("incrCols " << incrCols.toShortString() << ", " << incrCols.sum());
	CuMatrix<T> incrColsCM = CuMatrix<T>::increasingColumns(100, 90,5, true);
	outln("incrCols " << incrCols.syncBuffers());
	outln("colsCM " << incrColsCM.syncBuffers());

	return 0;
}

template int testSubmatrices<float>::operator()(int argc, const char **argv) const;
template int testSubmatrices<double>::operator()(int argc, const char **argv) const;
template int testSubmatrices<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testSubmatrices<T>::operator()(int argc, const char **argv) const {
	outln("testSubmatrices ent");
	CuMatrix<T> theta1 = CuMatrix<T>::sequence(100, 10, 25);
	CuMatrix<T> theta2 = CuMatrix<T>::sequence(2000, 20,26);
	theta1.syncBuffers();
	theta2.syncBuffers();
	outln("theta1 " << theta1.toShortString());
	outln("theta2 " << theta2.toShortString());
	T sTheta1= theta1.sum();
	T sTheta2= theta2.sum();
	outln("theta1 sum " << sTheta1 );
	outln("theta2 sum " << sTheta2 );
	outln("theta1 sum " << sTheta1 << " "<< theta1);
	outln("theta2 sum " << sTheta2 << " " << theta2);
	theta1.poseAsRow();
	theta2.poseAsRow();
	const CuMatrix<T>* parts[] = {&theta1,&theta2};
	const CuMatrix<T>* parts21[] = {&theta2,&theta1};
	CuMatrix<T> thetas12(1,(theta1.size + theta2.size)/sizeof(T),false,true);
    outln("\n\nthetas12.size " << thetas12.size << ", thetas12.tiler.m_size " << thetas12.tiler.m_size << ", thetas12.tiler.tileSize " << thetas12.tiler.tileSize);
	CuMatrix<T>::concat(thetas12, 2,parts);
	CuMatrix<T> thetas21(1,thetas12.size/sizeof(T),false,true);
	CuMatrix<T>::concat(thetas21, 2,parts21);
	theta1.unPose();
	theta2.unPose();
	thetas12.syncBuffers();
	thetas21.syncBuffers();
	T s_thetas12 = thetas12.sum();
	T s_thetas21= thetas21.sum();
	assert(util<T>::almostEquals(s_thetas12,s_thetas21));
	assert(util<T>::almostEquals(s_thetas12, sTheta1 + sTheta2));
    outln("\n\nthetas12 sum " << s_thetas12 << " "<< thetas12);
    outln("thetas21 sum " << s_thetas21 << " "<< thetas21);
	CuMatrix<T> unpackedTheta1, unpackedTheta2;
	CuMatrix<T> unpackedTheta1b, unpackedTheta2b;
	// submatrix(CuMatrix<T>& v, int rows, int cols, int pitch, ulong offset)
	thetas12.unconcat(unpackedTheta1,theta1.m,theta1.n,theta1.p,0);
	thetas12.unconcat(unpackedTheta2,theta2.m,theta2.n,theta2.p, theta1.size/sizeof(T));
	thetas21.unconcat(unpackedTheta1b,theta2.m,theta2.n,theta2.p,0);
	thetas21.unconcat(unpackedTheta2b,theta1.m,theta1.n,theta1.p,theta2.size/sizeof(T));
	outln("\n\nunpackedTheta1.ss " << unpackedTheta1.toShortString());
	outln("unpackedTheta2 " << unpackedTheta2.toShortString());
	outln("unpackedTheta2b" << unpackedTheta2b.toShortString());
	outln("unpackedTheta1b.ss " << unpackedTheta1b.toShortString());
	outln("unpackedTheta2b " << unpackedTheta2b.syncBuffers());
	outln("unpackedTheta2b.sum " << unpackedTheta2b.sum() );
	assert(theta1.almostEq(unpackedTheta1));
	assert(theta1.almostEq(unpackedTheta2b));

    uint input_layer_size = 400; // 20x20 Input Images of Digits
    uint hidden_layer_size = 25; //   25 hidden units
    uint num_labels = 10; // 10 labels, from 1 to 10
    CuMatrix<T> initial_Theta1 = CuMatrix<T>::sequence(5, hidden_layer_size, input_layer_size).addBiasColumn();
    CuMatrix<T> initial_Theta2 = CuMatrix<T>::sequence(50000,num_labels, hidden_layer_size).addBiasColumn();
    outln("initial_Theta1 " << initial_Theta1.syncBuffers());
    outln("initial_Theta2 " << initial_Theta2.syncBuffers());
    outln("initial_Theta1 sum " << initial_Theta1.sum());
    outln("initial_Theta2 sum " << initial_Theta2.sum());
    const CuMatrix<T>* pieces[] = {&initial_Theta1, &initial_Theta2};
    CuMatrix<T> nn_params(1, (initial_Theta1.size + initial_Theta2.size)/sizeof(T),true,true);
    CuMatrix<T>::concat(nn_params,2, pieces);
    outln("nn_params ss " << nn_params.toShortString());
    outln("nn_params " << nn_params);
    nn_params.syncBuffers();
 	CuMatrix<T> second_Theta1, second_Theta2;
	nn_params.unconcat(second_Theta1, hidden_layer_size,input_layer_size + 1,input_layer_size + 1, 0);
	outln("second_Theta1 " << second_Theta1.toShortString());
	outln("second_Theta1 " << second_Theta1.syncBuffers());
	outln("second_Theta1 sum " << second_Theta1.sum());
	//outln("nn_params " << nn_params);
	//outln("theta1.sum " << theta1.sum());
	nn_params.unconcat(second_Theta2, num_labels, hidden_layer_size + 1, hidden_layer_size + 1,(hidden_layer_size * (input_layer_size + 1)));
	outln("second_Theta2 " << second_Theta2.syncBuffers());
	outln("second_Theta2 sum " << second_Theta2.sum());

	assert(initial_Theta1.almostEq(second_Theta1));
	assert(initial_Theta2.almostEq(second_Theta2));

	return 0;
}

template int testDropFirstAlts<float>::operator()(int argc, const char **argv) const;
template int testDropFirstAlts<double>::operator()(int argc, const char **argv) const;
template int testDropFirstAlts<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testDropFirstAlts<T>::operator()(int argc, const char **argv) const {
	outln("testDropFirstAlts start");
	CuMatrix<T> b = CuMatrix<T>::ones(5000, 999) * 2;
	outln("b.sum " << b.sum());
	CuMatrix<T> base = b.addBiasColumn();
	CuMatrix<T> col1 = base.columnMatrix(0);
	outln("base.sum " << base.sum());
	outln("col1.sum " << col1.sum());
	assert(base.biasedQ());
	outln("base " << base.toShortString() << " sum " << base.sum());
	CuMatrix<T> baseM1 = base.dropFirst(true);
	CuMatrix<T> baseM2 = base.dropFirst(false);
	outln("baseM1 " << baseM1.toShortString());
	outln("baseM1 sum " << baseM1.sum());
	outln("baseM2 " << baseM2.toShortString());
	outln("baseM2 sum " << baseM2.sum());
	//outln("baseM2 " << baseM2.syncBuffers());
	CuMatrix<T> s1 = baseM1.sigmoid();
	outln("s1 " << s1.sum());
	CuMatrix<T> s2 = baseM2.sigmoid();
	outln("s2 " << s2.sum());
	return 0;
}

template int testSigmoidNneqP<float>::operator()(int argc, const char **argv) const;
template int testSigmoidNneqP<double>::operator()(int argc, const char **argv) const;
template <typename T> int testSigmoidNneqP<T>::operator()(int argc, const char **argv) const {
	outln("testDropFirstAlts start");
	CuMatrix<T> b = CuMatrix<T>::ones(40, 39) * 2;

	T bsum = b.sum();
	outln("b.sum " << bsum);
	//outln("b " << b.syncBuffers());
	CuMatrix<T> biased = b.addBiasColumn().syncBuffers();
	outln("biased.sum " << biased.sum());
	CuMatrix<T> smb;
	biased.submatrix(smb,b.m, b.n, 0,1); // should be equiv to b
	//base2.invalidateHost();
	//outln("smb " << smb);

	T smbsum = smb.sum();
	outln("smb.sum " << smbsum);
	assert(util<T>::almostEquals(bsum,smbsum));

	CuMatrix<T> sigb = b.sigmoid();
	outln("sigb " << sigb.syncBuffers());
	T sigbsum = sigb.sum();
	outln("sigb sum " << sigbsum);
	CuMatrix<T> sigsmb = smb.sigmoid();
	outln("sigsmb " << sigsmb.syncBuffers());
	T sigsmbsum = sigsmb.sum();
	outln("sigsmb sum " << sigsmbsum);
	assert(util<T>::almostEquals(sigbsum,sigsmbsum));

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

template int testSubmatrices2<ulong>::operator()(int argc, const char **argv) const;
template int testSubmatrices2<float>::operator()(int argc, const char **argv) const;
template int testSubmatrices2<double>::operator()(int argc, const char **argv) const;
template <typename T> int testSubmatrices2<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> b = CuMatrix<T>::zeros(60,1);
	for(int i = 1; i < 60; i++) {
		b = b |= (CuMatrix<T>::ones(60,1) * i);
	}
	outln("b " << b.syncBuffers());

	CuMatrix<T> sub1, sub2;
	CuMatrix<T> sub3(40,40,false,true);
	b.submatrix(sub1,40,40,2,2);
	b.submatrix(sub2,40,40,20,20);
	sub1.copy(sub3,0,0);
	sub3.syncBuffers();
	outln("sub1 " << sub1);
	outln("sub2 " << sub2);
	outln("sub3 " << sub3.syncBuffers());
	CuMatrix<T> sigsub1 = sub1.sigmoid();
	outln("sigsub1 " << sigsub1.syncBuffers());
	CuMatrix<T> sigsub2 = sub2.sigmoid();

	outln("sigsub2 " << sigsub2.syncBuffers());

	return 0;
}

bool testAttFreqsDev1 = false;

template int testAttFreqsSmall<float>::operator()(int argc, const char **argv) const;
template int testAttFreqsSmall<double>::operator()(int argc, const char **argv) const;
template int testAttFreqsSmall<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testAttFreqsSmall<T>::operator()(int argc, const char **argv) const {
	if(testAttFreqsDev1)
		ExecCaps_setDevice(1);
	{
		// test small

		T data[]= { 	's', 85, 85, 'f', 'n',
						's', 80, 90, 't', 'n',
						'o', 83, 78, 'f', 'y',
						'r', 70, 96, 'f', 'y',
						'r', 68, 80, 'f', 'y',
						'r', 65, 70, 't', 'n',
						'o', 64, 65, 't', 'y',
						's', 72, 95, 'f', 'n',
						's', 69, 70, 'f', 'y',
						'r', 75, 80, 'f', 'y',
						's', 75, 70, 't', 'y',
						'o', 72, 90, 't', 'y',
						'o', 81, 75, 'f', 'y',
						'r', 71, 80, 't', 'n' };
		CuMatrix<T> golfx(data, 14, 5, true);
		outln("golfx " << golfx);
		CuMatrix<T> freqs = CuMatrix<T>::zeros(golfx.m, golfx.n);
		//outln("freqs " << freqs);
		CuMatrix<T> atts = CuMatrix<T>::zeros(golfx.m, golfx.n);
		//outln("atts " << atts);
		CuMatrix<T> entropies = CuMatrix<T>::zeros(golfx.m, golfx.n);
		//outln("entropies " << entropies);
		CuMatrix<T> dummy = CuMatrix<T>::zeros(1, golfx.n);
		//outln("dummy " << dummy);
		CuMatrix<int> counts = CuMatrix<int>::zeros(golfx.m, golfx.n);
		CuMatrix<int> depths = CuMatrix<int>::zeros(1,golfx.n);
		CuMatrix<int> distances = CuMatrix<int>::zeros(1,golfx.n);
		//outln("distances " << distances);
		CuTimer timer;
		timer.start();
		golfx.attributeFrequencies(freqs, atts, entropies, counts, depths, distances);
		float attFreqsTimeS = timer.stop()/1000.0;


		outln("freqs.toss() " << freqs.toss() );
		CuMatrix<T> freqSums = CuMatrix<T>::zeros(1,golfx.n);// freqs.columnSums();
		outln("freqSums " << freqSums.syncBuffers());

		outln("golfx attributeFrequencies took " << attFreqsTimeS << "s");
		depths.syncBuffers();
		counts.syncBuffers();
		outln("golfx depths\n");
		depths.print("depths\n");
		outln("golfx distances\n");
		depths.print("distances\n");
		int totalDist = distances.sum();
		outln("total distance " << totalDist);

		outln("golfx atts\n" << atts.syncBuffers());

		outln("golfx freqs " << freqs.syncBuffers());
		outln("golfx entrops " << entropies.syncBuffers());

		for(int i= 0; i < golfx.n; i++ ) {
			outln("col " << i << " ent-sum " << entropies.columnSum(i) << ", freq sum " << freqs.columnSum(i));
		}
		T allFreqs = freqs.sum();
		outln("freqsum "<< allFreqs);
	}

	{
	}


	{
		CuMatrix<T> shtep = CuMatrix<T>::increasingColumns(14, 5,1);
		outln("shtep " << shtep.syncBuffers());
		for(int i= 0; i < shtep.n; i++ ) {
			outln("shtep col " << i << " sum " << shtep.columnSum(i));
		}
		map<string, CuMatrix<T>*> f;

		f = CuMatrix<T>::parseOctaveDataFile(SAMPLES_FILE, false, true);
		CuMatrix<T>* x = f["X"];
		outln("x "<< x->toShortString());
		x->syncBuffers();

		CuMatrix<T> freqs = CuMatrix<T>::zeros(x->m,x->n);
		CuMatrix<T> atts = CuMatrix<T>::zeros(x->m,x->n);
		CuMatrix<T> entropies = CuMatrix<T>::zeros(x->m,x->n);
		CuMatrix<int> counts = CuMatrix<int>::zeros(x->m,x->n);
		CuMatrix<int> depths = CuMatrix<int>::zeros(1,x->n);
		CuMatrix<int> distances = CuMatrix<int>::zeros(1,x->n);

		CuTimer timer;
		timer.start();
		cudaStream_t stream;

		checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

		x->attributeFrequencies(freqs, atts, entropies, counts,depths,distances);
		float attFreqsTimeS = timer.stop()/1000.0;
		outln("x->attributeFrequencies took " << attFreqsTimeS << "s");
		depths.syncBuffers();
		counts.syncBuffers();
		outln("depths\n");
		depths.print("depths\n");
		outln("distances\n");
		depths.print("distances\n");
		int totalDist = distances.sum();
		outln("total distance " << totalDist);
	//	counts.print("counts");

		outln("atts\n" << atts.syncBuffers());

		outln("freqs " << freqs.syncBuffers());
		T allFreqs = freqs.sum();
		outln("freqsum "<< allFreqs);
		assert(util<T>::almostEqualsNormalized(allFreqs, x->n)); // since every column should sum to 1, total should be = to num cols
		util<CuMatrix<T> >::deletePtrMap(f);
	}




	return 0;
}

template int testAttFreqsMed<float>::operator()(int argc, const char **argv) const;
template int testAttFreqsMed<double>::operator()(int argc, const char **argv) const;
template int testAttFreqsMed<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testAttFreqsMed<T>::operator()(int argc, const char **argv) const {

}

#include "tests.cc"
