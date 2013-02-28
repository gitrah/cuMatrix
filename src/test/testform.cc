#include "tests.h"
#include "../util.h"
#include "../caps.h"

template int testBinCat<float>::operator()(int argc, char const ** args) const;
template int testBinCat<double>::operator()(int argc, char const ** args) const;
template <typename T> int testBinCat<T>::operator()(int argc, const char** args) const {
	Matrix<T>::verbose = true;
	Matrix<T> a = Matrix<T>::seqMod(1, 32, 40, 1);
	outln(a.toString());
	Matrix<T> b = a.toBinaryCategoryMatrix();
	outln(b.toString());
	Matrix<T> c = b.toMaxColumnIndexVector();
	outln(c.toString());
	dassert(a.almostEq(c));
	outln("passed 1");
	Matrix<T> a2 = Matrix<T>::seqMod(1, 70, 40, 1);
	outln(a2.toString());
	Matrix<T> b2 = a2.toBinaryCategoryMatrix();
	outln(b2.toString());
	Matrix<T> c2 = b2.toMaxColumnIndexVector();
	outln(c2.toString());
	dassert(a2.almostEq(c2 + static_cast<T>(1)));
	Matrix<T>::verbose = false;
	return 0;
}

template int testFillXvsY<float>::operator()(int argc, char const ** args) const;
template int testFillXvsY<double>::operator()(int argc, char const ** args) const;
template <typename T> int testFillXvsY<T>::operator()(int argc, const char** args) const {
	Matrix<T>::verbose = true;
	int count = b_util::getCount(argc,args,1000);
    CuTimer timer;
    timer.start();
	clock_t lastTime = clock();

	// rows
    for(int i = 0; i < count; i++) {
		Matrix<T> m = Matrix<T>::ones(5000,1);
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
		Matrix<T> m = Matrix<T>::ones(1,5000);
	}
	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("columns " << count << " took " << delta << " secs");
	exeTime = timer.stop();
    memFlowIter = 1000.f * 5000 / Giga / (exeTime / count);
	outln("N " << count << " took exeTime " << (exeTime /1000) << "s or flow of " << memFlowIter << "GB/s");

	return 0;
}

template int testCat<float>::operator()(int argc, char const ** args) const;
template int testCat<double>::operator()(int argc, char const ** args) const;
template <typename T> int testCat<T>::operator()(int argc, const char** args) const {
	Matrix<T> s = Matrix<T>::ones(40, 1);
	Matrix<T> t = s |= Matrix<T>::ones(40, 40) * 3;
	outln("t.ss " << t.toShortString());
	outln("t after sum " << t.sum() << " mat " << t.syncBuffers());
	Matrix<T> s2 = Matrix<T>::ones(40,40) * 2;
	Matrix<T> t2 = s2 |= Matrix<T>::ones(40,1);
	outln("t2 " << (t2.syncBuffers()));
	Matrix<T> s3 = s.rightConcatenate(Matrix<T>::ones(40, 1));
	outln("s final " << (s3.syncBuffers()).toString());

	Matrix<T> big = Matrix<T>::ones(400,400) * 2;
	Matrix<T> bigger = big /= big;
	T bs = big.sum();
	T bbs = bigger.sum();
	outln("big.sum() " << bs << ", bigger " << bbs);
	dassert(bs * 2 == bbs);
	return 0;
}

template int testMaxColIdxs<float>::operator()(int argc, char const ** args) const;
template int testMaxColIdxs<double>::operator()(int argc, char const ** args) const;
template <typename T> int testMaxColIdxs<T>::operator()(int argc, const char** args) const {
	Matrix<T> a = Matrix<T>::randn(1000,10);
	outln(a.syncBuffers());
	Matrix<T> b = a.toMaxColumnIndexVector();
	outln(b.syncBuffers());
	return 0;
}

const char * txNames[] = {"transposeNaive","transposeCoalesced","transposeNoBankConflicts","transposeDiagonalKernel"};
template <typename T> __global__ void transposeNaive(const T*, T*, uint, uint);
template <typename T> __global__ void transposeCoalesced(const T*, T*, uint, uint);
template <typename T> __global__ void transposeNoBankConflicts(const T*, T*, uint, uint);
template <typename T> __global__ void transposeDiagonalKernel(const T*, T*, uint, uint);

template int testTranspose<float>::operator()(int argc, char const ** args) const;
template int testTranspose<double>::operator()(int argc, char const ** args) const;
template <typename T> int testTranspose<T>::operator()(int argc, const char** args) const {

	void (*kernels[4])(const T*, T*, uint, uint);
    void (*kernel)(const T*, T*, uint, uint);
	kernels[0] = &transposeNaive;
	kernels[1] = &transposeCoalesced;
	kernels[2] = &transposeNoBankConflicts;
	kernels[3] = &transposeDiagonalKernel;
	for(int i = 0; i < 4;i++) {
//	for(int i = 3; i < 4;i++) {
		kernel = kernels[i];
		outln("kernel " << txNames[i]);
		Matrix<T> ms = Matrix<T>::sequence(1,11,13);
		ms.syncBuffers();
		Matrix<T> tms = ms.transposeKernelPtr(kernel);
		Matrix<T> ttms = tms.transposeKernelPtr(kernel);
		Matrix<T>::verbose = true;
		outln("ms " << ms);
		tms.syncBuffers();
		outln("tms " << tms);
		outln("s- " << (ms - ttms).syncBuffers());
		Matrix<T>::verbose = false;
		assert( ttms.almostEq(ms));

		Matrix<T> ms2a = Matrix<T>::sequence(2,22,24).syncBuffers();\
		outln("ms2a " << ms2a);
		Matrix<T> ms2 = ms2a.addBiasColumn();
		outln("ms2 " << ms2.toShortString());
		ms2.syncBuffers();
		outln("ms2 " << ms2);
		Matrix<T> tms2 = ms2.transposeKernelPtr(kernel).syncBuffers();
		Matrix<T> ttms2 = tms2.transposeKernelPtr(kernel).syncBuffers();
		Matrix<T>::verbose = true;
		outln("tms2 " << tms2);
		outln("s2- " << (ms2 - ttms2).syncBuffers());
		Matrix<T>::verbose = false;
		assert( ttms2.almostEq(ms2));

		Matrix<T> m = Matrix<T>::sequence(0,32*3+1-16, 32*3-16).addBiasColumn().syncBuffers();
		Matrix<T> tm = m.transposeKernelPtr(kernel).syncBuffers();
		Matrix<T> ttm = tm.transposeKernelPtr(kernel).syncBuffers();
		Matrix<T>::verbose = true;
		outln("m " << m);
		outln("tm " << tm);
		outln("- " << (m - ttm).syncBuffers());
		Matrix<T>::verbose = false;
		assert( ttm.almostEq(m));


		Matrix<T> m2 = Matrix<T>::sequence(0,16, 34).addBiasColumn().syncBuffers();
		Matrix<T> tm2 = m2.transposeKernelPtr(kernel).syncBuffers();
		Matrix<T> ttm2 = tm2.transposeKernelPtr(kernel).syncBuffers();
		Matrix<T>::verbose = true;
		outln("m2 " << m2);
		outln("tm2 " << tm2);
		outln("- " << (m2 - ttm2).syncBuffers());
		Matrix<T>::verbose = false;
		assert( ttm2.almostEq(m2));

		Matrix<T> dm = Matrix<T>::sequence(0, 65, 62).addBiasColumn().syncBuffers();
		Matrix<T> dtm = dm.transposeKernelPtr(kernel).syncBuffers();
		Matrix<T> dttm = dtm.transposeKernelPtr(kernel).syncBuffers();
		Matrix<T>::verbose = true;
		outln("dm " << dm);
		outln("dtm " << dtm);
		outln("- " << (dm - dttm).syncBuffers());
		Matrix<T>::verbose = false;
		assert( dttm.almostEq(dm));

		Matrix<T> mBig = Matrix<T>::sin(5000, 399).syncBuffers().addBiasColumn().syncBuffers();
		Matrix<T> tmBig = mBig.transposeKernelPtr(kernel).syncBuffers();
		Matrix<T> ttmBig = tmBig.transposeKernelPtr(kernel).syncBuffers();
		outln("mBig " << mBig);
		outln("tmBig " << tmBig);
		outln("- " << (mBig - ttmBig).syncBuffers());
		outln("sum " << (mBig - ttmBig).sum());
		assert( ttmBig.almostEq(mBig));

		Matrix<T> mTOdd= Matrix<T>::sin(23, 23).syncBuffers();
		Matrix<T> tmTOdd= mTOdd.transposeKernelPtr(kernel);
		Matrix<T> tttmOdd = tmTOdd.transposeKernelPtr(kernel);
		outln("tttmOdd " << tttmOdd.almostEq(mTOdd));

		Matrix<T> mTOddNs= Matrix<T>::sin(13, 23).syncBuffers();
		Matrix<T> tmTOddNs= mTOddNs.transposeKernelPtr(kernel);
		Matrix<T> tttmOddNs = tmTOddNs.transposeKernelPtr(kernel);
		outln("tttmOddNs " << tttmOddNs.almostEq(mTOddNs));

		Matrix<T> mBig2 = Matrix<T>::zeros(3220, 399).addBiasColumn();
		Matrix<T> tmBig2= mBig2.transposeKernelPtr(kernel);
		Matrix<T> ttmBig2 = tmBig2.transposeKernelPtr(kernel);
		outln("big2 " << ttmBig2.almostEq(mBig2));

		Matrix<T> mNonSquare= Matrix<T>::sin(512, 256).syncBuffers();
		Matrix<T> tmNonSquare = mNonSquare.transposeKernelPtr(kernel);
		Matrix<T> ttmNonSquare = tmNonSquare.transposeKernelPtr(kernel);
		outln("ttmNonSquare " << ttmNonSquare.almostEq(mNonSquare));

		Matrix<T> mOdd= Matrix<T>::sin(333, 333).syncBuffers();
		Matrix<T> tmOdd = mOdd.transposeKernelPtr(kernel);
		Matrix<T> ttmOdd = tmOdd.transposeKernelPtr(kernel);
		outln("ttmOdd " << ttmOdd.almostEq(mOdd));

		Matrix<T> mOddNs= Matrix<T>::sin(533, 333).syncBuffers();
		Matrix<T> tmOddNs = mOddNs.transposeKernelPtr(kernel);
		Matrix<T> ttmOddNs = tmOddNs.transposeKernelPtr(kernel);
		outln("ttmOddNS " << ttmOddNs.almostEq(mOddNs));

		Matrix<T> mSOdd= Matrix<T>::sin(33, 33).syncBuffers();
		Matrix<T> tmSOdd= mSOdd.transposeKernelPtr(kernel);
		Matrix<T> ttsmOdd = tmSOdd.transposeKernelPtr(kernel);
		outln("ttsmOdd " << ttsmOdd.almostEq(mSOdd));

		Matrix<T> mSOddNs= Matrix<T>::sin(33, 53).syncBuffers();
		Matrix<T> tmSOddNs= mSOddNs.transposeKernelPtr(kernel);
		Matrix<T> ttsmOddNs = tmSOddNs.transposeKernelPtr(kernel);
		outln("ttsmOddNs " << ttsmOddNs.almostEq(mSOddNs));
	}
	return 0;
}

extern int blockH;
template int testTransposeLoop<float>::operator()(int argc, char const ** args) const;
template int testTransposeLoop<double>::operator()(int argc, char const ** args) const;
template <typename T> int testTransposeLoop<T>::operator()(int argc, const char** args) const {
    CuTimer timer;
	outln("testTransposeLoop start");
	Matrix<T> seq = Matrix<T>::sequence(0,1024, 1024);
	//
	Matrix<T> tx = Matrix<T>::zeros(1024, 1024);
	ulong matB = tx.size;
	DMatrix<T> tx_d;
	tx.asDmatrix(tx_d);
	int count = b_util::getCount(argc,args,10000);
	void (*kernels[4])(const T*,T*,uint,uint);
    void (*kernel)(const T*,T*,uint,uint);
	kernels[0] = &transposeNaive;
	kernels[1] = &transposeCoalesced;
	kernels[2] = &transposeNoBankConflicts;
	kernels[3] = &transposeDiagonalKernel;
	outln("blockDim.y " << blockH);
	for(int k = 0; k < 4;k++) {
		kernel = kernels[k];
		outln("kernel " << txNames[k]);
	    outln(debugMem << debugLife << debugCopy);
	    timer.start();
		clock_t lastTime = clock();
		for (int i = 0; i < count; i++) {
			seq.transposeKernelPtr(tx_d, kernel);
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

template int testReshape<float>::operator()(int argc, char const ** args) const;
template int testReshape<double>::operator()(int argc, char const ** args) const;
template <typename T> int testReshape<T>::operator()(int argc, const char** args) const {
	uint width = 500;
	uint height = 500;
	Matrix<T> ms = Matrix<T>::sin(height, width-1).addBiasColumn();
	Matrix<T> mc = Matrix<T>::cos(height, width-1).addBiasColumn();
	Matrix<T> flat = ms.poseAsCol() /= mc.poseAsCol();
	Matrix<T> msP = flat.reshape(height,width,0);
	Matrix<T> mcP = flat.reshape(height,width,height*width);
	dassert(ms.almostEq(msP));
	outln("ms sum " << ms.sum() << ", msP.sum " << msP.sum());
	dassert(mc.almostEq(mcP));
	outln("mc sum " << mc.sum() << ", mcP.sum " << mcP.sum());
	return 0;
}

template int testTransposeNneqP<float>::operator()(int argc, char const ** args) const;
template int testTransposeNneqP<double>::operator()(int argc, char const ** args) const;
template <typename T> int testTransposeNneqP<T>::operator()(int argc, const char** args) const {
	Matrix<T> incrRows = Matrix<T>::increasingRows(5, 1000, 900).syncBuffers();
	Matrix<T> incrRowsCM = Matrix<T>::increasingRows(5, 1000, 900, true).syncBuffers();
	outln("a row of incrRows");
	for(uint col = 0; col < incrRows.n; col++ ) {
		cout << incrRows.get(0,col);
	}
	cout << endl;
	outln("a row of incrRowsCM");
	for(uint col = 0; col < incrRowsCM.n; col++ ) {
		cout << incrRowsCM.get(0,col);
	}
	cout << endl;

	Matrix<T> m1 = Matrix<T>::zeros(1,10,false).syncBuffers();
	Matrix<T> m2 = Matrix<T>::zeros(1,10,true).syncBuffers();
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
	Matrix<T> incrCols = Matrix<T>::increasingColumns(5, 1000, 900);
	outln("incrCols " << incrCols.toShortString() << ", " << incrCols.sum());
	Matrix<T> incrColsCM = Matrix<T>::increasingColumns(5, 100, 90, true);
	outln("incrCols " << incrCols.syncBuffers());
	outln("colsCM " << incrColsCM.syncBuffers());

	return 0;
}

template int testFillers<float>::operator()(int argc, char const ** args) const;
template int testFillers<double>::operator()(int argc, char const ** args) const;
template <typename T> int testFillers<T>::operator()(int argc, const char** args) const {
	outln("testFillers start");
	Matrix<T> seq = Matrix<T>::sequence(1, 100, 1);
	outln("seq\n" << seq.toString());
	Matrix<T> msin = Matrix<T>::sin(10,1);
	Matrix<T> mcos = Matrix<T>::cos(10,1);
	outln("mcos\n" << mcos.toString());
	outln("msin\n" << msin.toString());

	return 0;
}

template int testSubmatrices<float>::operator()(int argc, char const ** args) const;
template int testSubmatrices<double>::operator()(int argc, char const ** args) const;
template <typename T> int testSubmatrices<T>::operator()(int argc, const char** args) const {
	Matrix<T> theta1 = Matrix<T>::sequence(100, 10, 25);
	Matrix<T> theta2 = Matrix<T>::sequence(2000, 20,26);
	theta1.syncBuffers();
	theta2.syncBuffers();
	T sTheta1= theta1.sum();
	T sTheta2= theta2.sum();
	outln("theta1 sum " << sTheta1 << " "<< theta1);
	outln("theta2 sum " << sTheta2 << " " << theta2);
	theta1.poseAsRow();
	theta2.poseAsRow();
	const Matrix<T>* parts[] = {&theta1,&theta2};
	const Matrix<T>* parts21[] = {&theta2,&theta1};
	Matrix<T> thetas12(1,(theta1.size + theta2.size)/sizeof(T),false,true);
	Matrix<T>::concat(thetas12, 2,parts);
	Matrix<T> thetas21(1,thetas12.size/sizeof(T),false,true);
	Matrix<T>::concat(thetas21, 2,parts21);
	theta1.unPose();
	theta2.unPose();
	thetas12.syncBuffers();
	thetas21.syncBuffers();
	T s_thetas12 = thetas12.sum();
	T s_thetas21= thetas21.sum();
	assert(util<T>::almostEquals(s_thetas12,s_thetas21));
	assert(util<T>::almostEquals(s_thetas12, sTheta1 + sTheta2));
    outln("thetas12 sum " << s_thetas12 << " "<< thetas12);
    outln("thetas21 sum " << s_thetas21 << " "<< thetas21);
	Matrix<T> unpackedTheta1, unpackedTheta2;
	Matrix<T> unpackedTheta1b, unpackedTheta2b;
	// submatrix(Matrix<T>& v, uint rows, uint cols, uint pitch, ulong offset)
	thetas12.unconcat(unpackedTheta1,theta1.m,theta1.n,theta1.p,0);
	thetas12.unconcat(unpackedTheta2,theta2.m,theta2.n,theta2.p, theta1.size/sizeof(T));
	thetas21.unconcat(unpackedTheta1b,theta2.m,theta2.n,theta2.p,0);
	thetas21.unconcat(unpackedTheta2b,theta1.m,theta1.n,theta1.p,theta2.size/sizeof(T));
	outln("unpackedTheta1.ss " << unpackedTheta1.toShortString());
	outln("unpackedTheta2 " << unpackedTheta2);
	outln("unpackedTheta1b.ss " << unpackedTheta1b.toShortString());
	outln("unpackedTheta2b " << unpackedTheta2b);
	assert(theta1.almostEq(unpackedTheta1));
	assert(theta1.almostEq(unpackedTheta2b));

    uint input_layer_size = 400; // 20x20 Input Images of Digits
    uint hidden_layer_size = 25; //   25 hidden units
    uint num_labels = 10; // 10 labels, from 1 to 10
    Matrix<T> initial_Theta1 = Matrix<T>::sequence(5, hidden_layer_size, input_layer_size).addBiasColumn();
    Matrix<T> initial_Theta2 = Matrix<T>::sequence(50000,num_labels, hidden_layer_size).addBiasColumn();
    outln("initial_Theta1 " << initial_Theta1.syncBuffers());
    outln("initial_Theta2 " << initial_Theta2.syncBuffers());
    outln("initial_Theta1 sum " << initial_Theta1.sum());
    outln("initial_Theta2 sum " << initial_Theta2.sum());
    const Matrix<T>* pieces[] = {&initial_Theta1, &initial_Theta2};
    Matrix<T> nn_params(1, (initial_Theta1.size + initial_Theta2.size)/sizeof(T),false,true);
    Matrix<T>::concat(nn_params,2, pieces);
 	Matrix<T> second_Theta1, second_Theta2;
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

template int testDropFirstAlts<float>::operator()(int argc, char const ** args) const;
template int testDropFirstAlts<double>::operator()(int argc, char const ** args) const;
template <typename T> int testDropFirstAlts<T>::operator()(int argc, const char** args) const {
	outln("testDropFirstAlts start");
	Matrix<T> b = Matrix<T>::ones(5000, 999) * 2;
	outln("b.sum " << b.sum());
	Matrix<T> base = b.addBiasColumn();
	outln("base " << base.toShortString() << " sum " << base.sum());
	Matrix<T> baseM1 = base.dropFirst(true);
	Matrix<T> baseM2 = base.dropFirst(false);
	outln("baseM1 " << baseM1.toShortString());
	outln("baseM1 sum " << baseM1.sum());
	outln("baseM2 " << baseM2.toShortString());
	outln("baseM2 sum " << baseM2.sum());
	//outln("baseM2 " << baseM2.syncBuffers());
	Matrix<T> s1 = baseM1.sigmoid();
	outln("s1 " << s1.sum());
	Matrix<T> s2 = baseM2.sigmoid();
	outln("s2 " << s2.sum());
	return 0;
}

template int testSigmoidNneqP<float>::operator()(int argc, char const ** args) const;
template int testSigmoidNneqP<double>::operator()(int argc, char const ** args) const;
template <typename T> int testSigmoidNneqP<T>::operator()(int argc, const char** args) const {
	outln("testDropFirstAlts start");
	Matrix<T> b = Matrix<T>::ones(40, 39) * 2;

	T bsum = b.sum();
	outln("b.sum " << bsum);
	//outln("b " << b.syncBuffers());
	Matrix<T> biased = b.addBiasColumn().syncBuffers();
	outln("biased.sum " << biased.sum());
	Matrix<T> smb;
	biased.submatrix(smb,b.m, b.n, 0,1); // should be equiv to b
	//base2.invalidateHost();
	//outln("smb " << smb);

	T smbsum = smb.sum();
	outln("smb.sum " << smbsum);
	assert(util<T>::almostEquals(bsum,smbsum));

	Matrix<T> sigb = b.sigmoid();
	outln("sigb " << sigb.syncBuffers());
	T sigbsum = sigb.sum();
	outln("sigb sum " << sigbsum);
	Matrix<T> sigsmb = smb.sigmoid();
	outln("sigsmb " << sigsmb.syncBuffers());
	T sigsmbsum = sigsmb.sum();
	outln("sigsmb sum " << sigsmbsum);
	assert(util<T>::almostEquals(sigbsum,sigsmbsum));

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

template int testSubmatrices2<float>::operator()(int argc, char const ** args) const;
template int testSubmatrices2<double>::operator()(int argc, char const ** args) const;
template <typename T> int testSubmatrices2<T>::operator()(int argc, const char** args) const {
	Matrix<T> b = Matrix<T>::zeros(60,1);
	for(int i = 1; i < 60; i++) {
		b = b |= (Matrix<T>::ones(60,1) * i);
	}
	outln("b " << b.syncBuffers());

	Matrix<T> sub1, sub2;
	Matrix<T> sub3(40,40,false,true);
	b.submatrix(sub1,40,40,2,2);
	b.submatrix(sub2,40,40,20,20);
	sub1.copy(sub3,0,0);
	outln("sub1 " << sub1);
	outln("sub2 " << sub2);
	outln("sub3 " << sub3.syncBuffers());
	Matrix<T> sigsub1 = sub1.sigmoid();
	Matrix<T> sigsub2 = sub2.sigmoid();

	outln("sigsub1 " << sigsub1.syncBuffers());
	outln("sigsub1 " << sigsub2.syncBuffers());

	return 0;
}
#include "tests.cc"
