#include "tests.h"
#include "../CuMatrix.h"
#include "../util.h"
#include "../caps.h"
#include "../Kernels.h"


template int testSqrMatsMultSmall<float>::operator()(int argc, const char **argv) const;
template int testSqrMatsMultSmall<double>::operator()(int argc, const char **argv) const;
template int testSqrMatsMultSmall<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testSqrMatsMultSmall<T>::operator()(int argc, const char **argv) const {
	outln("testSqrMatsMult start");
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,argv,1);
	outln("creating big matrices");
	checkCudaErrors(cudaDeviceSynchronize());
	timer.start();
	CuMatrix<T> ms = CuMatrix<T>::increasingRows(1,128,128) ;
	DMatrix<T> d_ms = ms.asDmatrix(true);
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 128,128);
	CuMatrix<T> txmc = mc.transpose();
	outln("txmc\n"<< txmc.syncHost());
	DMatrix<T> d_mc = mc.asDmatrix(false);
	DMatrix<T> d_txmc = txmc.asDmatrix(true);
	CuMatrix<T> msc(128,128,false,true);
	DMatrix<T> d_msc;
	msc.tile0(d_msc,false);
	exeTime = timer.stop()/1000.0f;
	outln("creating big matrices " << exeTime << " secs");

	dim3 blocks[]  = { dim3(32,32),dim3(16,16), dim3(8,8)};
	map<string,float> runTimes;
	typedef pair<string,float> runpair;
	const char* names[] = {"k1 ","k2 ", "ktx ", "ktx2 ", "ktxred "};
	void (*matProdKptr[]) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int) =
	{matrixProductKernel,matrixProductKernel2,matrixProductKernelTxdB,matrixProductKernelTxdB2
#ifdef CuMatrix_Enable_Cdp
			, matrixProductReductionTxdBKernel
#endif
			};
    for(int kernel = 0; kernel < 1; kernel++) {
    	clock_t lastTime = clock();
		for(int i = 0; i < 3; i++) {
			dim3* blockP = blocks + i;
			outln("using block dim " << b_util::pd3(blockP));
			outln("using d_ms " << util<T>::pdm(d_ms));
			timer.start();
			for (int i = 0; i < total; i++) {
				matrixProductKPtrL(d_msc,matProdKptr[kernel], d_ms, kernel>1 ? d_txmc : d_mc, blockP);
			}
			exeTime = timer.stop();
			runTimes.insert(runTimes.end(),runpair(names[kernel] + b_util::pd3(blockP, util<T>::pdm(d_ms) + "*" + util<T>::pdm(d_mc)), exeTime));
			int factor = 1; // (reads are of diff sizes so ignored; this is 'write' throughput)
			outln( total << " of 1000x1000 took " << exeTime << "ms or flow of " << ms.flow(total,factor,exeTime) << "GB/s");
			outln( "perElem msc " << exeTime/msc.size);
			cudaDeviceSynchronize();
			outln("\n\n" << msc.syncHost());
			outln( "sanity msc.sum " << msc.sum());

		}
	    double delta = b_util::diffclock(clock(), lastTime) / 1000;
	    outln("Completed! s " << (3 * total) << " took " << delta << " secs\n\n\n");

	    outln("results");
		typedef typename map<string, float>::iterator iterator;
		for(iterator it = runTimes.begin();it != runTimes.end();it++) {
			outln((*it).first << " took " << (*it).second << "ms");
		}
    }
	return 0;
}

template int testSqrMatsMult<float>::operator()(int argc, const char **argv) const;
template int testSqrMatsMult<double>::operator()(int argc, const char **argv) const;
template int testSqrMatsMult<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testSqrMatsMult<T>::operator()(int argc, const char **argv) const {
	outln("testSqrMatsMult start");
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,argv,1);

	outln("creating big matrices");

	CuMatrix<T> b1s = CuMatrix<T>::ones(1024,1024);
	CuMatrix<T> b2s = 2 * CuMatrix<T>::ones(1024,1024);
	outln("b2s " << b2s.syncBuffers());
	CuMatrix<T> bmult = b1s * b2s;
	outln("bmult " << bmult.syncBuffers());
	outln("bmult sum " << bmult.sum());
	timer.start();
	CuMatrix<T> ms = CuMatrix<T>::increasingRows(1,1024,1024) ;
	outln("ms " << ms.syncBuffers());
	DMatrix<T> d_ms = ms.asDmatrix(true);
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 1024,1024);
	CuMatrix<T> txmc = mc.transpose();
	DMatrix<T> d_mc = mc.asDmatrix( true);
	DMatrix<T> d_txmc = txmc.asDmatrix( true);
	CuMatrix<T> msc(1024,1024,false,true);
	DMatrix<T> d_msc;
	msc.tile0(d_msc,false);
	exeTime = timer.stop()/1000.0f;
	outln("creating big matrices " << exeTime << " secs");

	dim3 blocks[]  = { dim3(32,32),dim3(16,16), dim3(8,8)};
	map<string,float> runTimes;
	typedef pair<string,float> runpair;
	const char* names[] = {"k1 ","k2 ", "ktx ", "ktx2 "};

	void (*matProdKptr[]) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int) =  {matrixProductKernel,matrixProductKernel2,matrixProductKernel3,matrixProductKernel4,matrixProductKernelTxdB,matrixProductKernelTxdB2
#ifdef CuMatrix_Enable_Cdp
, matrixProductReductionTxdBKernel
#endif
	};

    for(int kernel = 0; kernel < 3; kernel++) {
    	clock_t lastTime = clock();
		for(int i = 0; i < 3; i++) {
			dim3* blockP = blocks + i;
			outln("kernel " << kernel << " using block dim " << b_util::pd3(blockP));
			timer.start();
			for (int i = 0; i < total; i++) {
				matrixProductKPtrL(d_msc,matProdKptr[kernel], d_ms, kernel>1 ? d_txmc : d_mc, blockP);
				checkCudaError(cudaDeviceSynchronize());
			  	 b_util::usedDmem();
  			}
			checkCudaError(cudaGetLastError());
			exeTime = timer.stop();
			runTimes.insert(runTimes.end(),runpair(names[kernel] + b_util::pd3(blockP, util<T>::pdm(d_ms) + "*" + util<T>::pdm(d_mc)), exeTime));
			int factor = 1; // (reads are of diff sizes so ignored; this is 'write' throughput)
			outln( total << " of 1000x1000 took " << exeTime << "ms or flow of " << ms.flow(total,factor,exeTime) << "GB/s");
			outln( "perElem msc " << exeTime/msc.size);
			outln( "sanity msc.sum " << msc.sum());

		}
	    double delta = b_util::diffclock(clock(), lastTime) / 1000;
	    outln("Completed! s " << (3 * total) << " took " << delta << " secs\n\n\n");

	    outln("results");
		typedef typename map<string, float>::iterator iterator;
		for(iterator it = runTimes.begin();it != runTimes.end();it++) {
			outln((*it).first << " took " << (*it).second << "ms");
		}
    }
	return 0;
}

template int testProductKernel3<float>::operator()(int argc, const char **argv) const;
template int testProductKernel3<double>::operator()(int argc, const char **argv) const;
template int testProductKernel3<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testProductKernel3<T>::operator()(int argc, const char **argv) const {
	outln("testProductShapes start");
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,argv,1);

	CuMatrix<T> m13by17= CuMatrix<T>::ones(13,17);
	CuMatrix<T> m17by1 = CuMatrix<T>::fill(2, 17,1);
	CuMatrix<T> mm2 = m13by17 * m17by1;
 	outln("m13by17 * m17by1 " << mm2.toShortString());

	T v[] = {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3};
	T v2[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5};

	CuMatrix<T> m1(v, 3,5, true);
	outln("m1 ");
	outln(m1.toString());
	CuMatrix<T> m2(v2,5,4, true);
	outln("m2 ");
	outln(m2.toString());
	CuMatrix<T> mm1 = m1 * m2;
	outln("m1 * m2 " << mm1.syncBuffers().toShortString());
	outln(mm1.toString());
	dim3 d(4,4);
	CuMatrix<T> mm1b = m1.matrixProduct(m2,&d).syncBuffers();
	outln("m1 * m2 d(4,4) " << mm1b.toShortString());
	outln(mm1b.toString());
	CuMatrix<T> m33 = CuMatrix<T>::ones(10,33);
	CuMatrix<T> m332 = CuMatrix<T>::ones(33,10) * 2;

	CuMatrix<T> mm3 = m33 * m332;
	outln(" m33 * m332 " << mm3.toShortString());
	outln(mm3.syncBuffers().toString());

	outln("creating big matrices");
	checkCudaErrors(cudaDeviceSynchronize());
	timer.start();
	CuMatrix<T> ms = CuMatrix<T>::ones(1000,1000) ;
	DMatrix<T> d_ms = ms.asDmatrix(true);
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 1000,200);
	DMatrix<T> d_mc = mc.asDmatrix(true);
	CuMatrix<T> msc(1000,200,false,true);
	DMatrix<T> d_msc;
	msc.tile0(d_msc,false);
	exeTime = timer.stop()/1000.0f;
	outln("creating big matrices " << exeTime << " secs");

	dim3 d8( 8,8);
	dim3 d16( 16,16);
	dim3 d32( 32,32);
	outln("null");
	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductKPtrL(d_msc,matrixProductKernel3,d_ms, d_mc, null);
	}
	T sum = msc.sum();
	msc.invalidateHost();
	outln(msc.toShortString() << " with default dims sum " << sum << " from \n" << msc.syncBuffers());

	exeTime = timer.stop()/1000.0f;
	outln("null took " << exeTime << " secs");
	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductKPtrL(d_msc,matrixProductKernel3,d_ms, d_mc, &d8);
		//ms.matrixProductL(d_msc,d_ms, d_mc, &d8);
	}
	exeTime = timer.stop()/1000.0f;
	outln("8x8 took " << exeTime << " secs");
	T sum2 = msc.sum();
	msc.invalidateHost();
	outln("8x8 dims sum " << sum << ", sum2 " << sum2 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum2));

	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductKPtrL(d_msc,matrixProductKernel3,d_ms, d_mc, &d16);
		//ms.matrixProductL(d_msc, d_ms, d_mc, &d16);
	}
	exeTime = timer.stop()/1000.0f;
	outln("16x16 took " << exeTime << " secs");
	T sum3 = msc.sum();
	msc.invalidateHost();
	outln("16x16 dims sum " << sum << ", sum3 " << sum3 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum3));

	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductKPtrL(d_msc,matrixProductKernel3,d_ms, d_mc, &d32);
		//sms.matrixProductL(d_msc, d_ms, d_mc,&d32);
	}
	exeTime = timer.stop()/1000.0f;
	outln("32x32 took " << exeTime << " secs");
	T sum4 = msc.sum();
	outln("32x32 dims sum4 " << sum4 );
	msc.invalidateHost();
	outln("32x32 dims sum " << sum << " sum4" << sum4 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum4));
	msc.invalidateHost();
	outln(msc.syncBuffers().toString());

	return 0;
}

template int testProductShapes<float>::operator()(int argc, const char **argv) const;
template int testProductShapes<double>::operator()(int argc, const char **argv) const;
template int testProductShapes<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testProductShapes<T>::operator()(int argc, const char **argv) const {
	outln("testProductShapes start");
	checkCudaError(cudaGetLastError());
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,argv,1);


    CuMatrix<T> x = CuMatrix<T>::increasingRows(1,100,1);
	outln("x:\n" << x.syncBuffers());
    CuMatrix<T> x2 =  2 * x |= 3 * x;
	outln("x2:\n" << x2.syncBuffers());
    CuMatrix<T> xb = x2.addBiasColumn();
    outln("xb:\n" << xb.syncBuffers());
    printDevArray(xb.tiler.buff(), "xb.tiler.buff()",-1, xb.m * xb.n);
    CuMatrix<T> justB = CuMatrix<T>::ones(100,1);
    outln("justB:\n" << justB.syncBuffers());
    CuMatrix<T> xjustB = justB |= x2;
    outln("xjustB:\n" << xjustB.syncBuffers());


    CuMatrix<T> m1z = CuMatrix<T>::ones(3,1);
    outln("m1z:\n" << m1z.syncBuffers());

/*
    CuMatrix<T> txp =  mz.transpose() ;
    outln("txp\n" << txp.syncBuffers());
    CuMatrix<T> txm1z =   m1z.transpose() ;
    outln("txm1z\n" << txm1z.syncBuffers());
*/
    //CuMatrix<T> xbt= xb.transpose();

    //outln("xbt:\n" << xbt.syncBuffers());

    CuMatrix<T> m1zt = m1z.transpose();
    outln("m1zt:\n" <<m1zt.toShortString());
    CuMatrix<T> prod20 = m1zt * xb.transpose();
    CuMatrix<T> prod2 = m1z.transpose() * xb.transpose();

    outln("prod2\n" << prod2.syncBuffers());
    assert(25350 == prod2.sum());

	CuMatrix<T> m13by17= CuMatrix<T>::ones(13,17);
	outln("m13by17\n" << m13by17.syncBuffers());

	CuMatrix<T> m17by1 = CuMatrix<T>::fill(2, 17,1);
	outln("m17by1\n" << m17by1.syncBuffers());
	CuMatrix<T> mm2 = m13by17 * m17by1;
	outln("m13by17 * m17by1 " << mm2.toShortString());
	outln("m13by17 * m17by1 sum " << mm2.sum());

	T v[] = {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3};
	T v2[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5};

	CuMatrix<T> m1(v, 3,5, true);
	outln("m1 ");
	outln(m1.toString());
	CuMatrix<T> m2(v2,5,4, true);
	outln("m2 ");
	outln(m2.toString());
	CuMatrix<T> mm1 = m1 * m2;
	outln("m1 * m2 " << mm1.syncBuffers().toShortString());
	outln(mm1.toString());
	checkCudaError(cudaGetLastError());
	dim3 d(4,4);
	CuMatrix<T> mm1b = m1.matrixProduct(m2,&d).syncBuffers();
	outln("m1 * m2 d(4,4) " << mm1b);
	assert(mm1b.sum() == 360);
	CuMatrix<T> m33 = CuMatrix<T>::ones(10,33);
	outln("m33 " << m33.toShortString());
	CuMatrix<T> m332 = CuMatrix<T>::ones(33,10) * 2;
	outln("m332 " << b_util::modStr(m332.lastMod) << ", "<< m332.syncBuffers());

	CuMatrix<T> mm3 = m33 * m332;
	outln(" m33 * m332 " << mm3.toShortString());
	outln(mm3.syncBuffers().toString());
	assert(mm3.sum() == 6600);
	outln("creating big matrices");
	CuMatrix<T> ms = CuMatrix<T>::ones(1000,1000) ;
	outln("created ms " <<ms.toShortString());
	DMatrix<T> d_ms = ms.asDmatrix(true);
	outln("created d_ms " << util<T>::pdm(d_ms));
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 1000,200);
	outln("created mc " <<mc.syncBuffers());
	DMatrix<T> d_mc = mc.asDmatrix(true);
	outln("created d_mc " << util<T>::pdm(d_mc));
	CuMatrix<T> msc(1000,200,true,true);
	DMatrix<T> d_msc;
	msc.tile0(d_msc,false);
	outln("after msc.tile0");

	dim3 d8( 8,8);
	dim3 d16( 16,16);
	dim3 d32( 32,32);
	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductL(d_msc,d_ms, d_mc, null);
	}
	msc.invalidateHost();
	outln("msc " << msc.syncBuffers());
	T sum = msc.sum();
	//msc.invalidateHost();
	outln(msc << " with default dims sum " << sum );
	assert(sum == 20100000000);

	exeTime = timer.stop()/1000.0f;
	outln("null took " << exeTime << " secs");
	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductL(d_msc,d_ms, d_mc, &d8);
	}
	exeTime = timer.stop()/1000.0f;
	outln("8x8 took " << exeTime << " secs");
	T sum2 = msc.sum();
	msc.invalidateHost();
	outln("8x8 dims sum " << sum << ", sum2 " << sum2 );
	dassert(util<T>::almostEquals(sum ,sum2));

	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductL(d_msc, d_ms, d_mc, &d16);
	}
	exeTime = timer.stop()/1000.0f;
	outln("16x16 took " << exeTime << " secs");
	T sum3 = msc.sum();
	msc.invalidateHost();
	outln("16x16 dims sum " << sum << ", sum3 " << sum3 );
	dassert(util<T>::almostEquals(sum ,sum3));

	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductL(d_msc, d_ms, d_mc,&d32);
	}
	exeTime = timer.stop()/1000.0f;
	outln("32x32 took " << exeTime << " secs");
	T sum4 = msc.sum();
	outln("32x32 dims sum4 " << sum4 );
	msc.invalidateHost();
	outln("32x32 dims sum " << sum << " sum4" << sum4 );
	dassert(util<T>::almostEquals(sum ,sum4));


	return 0;
}
template int testLargeMatProds<float>::operator()(int argc, const char **argv) const;
template int testLargeMatProds<double>::operator()(int argc, const char **argv) const;
template int testLargeMatProds<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testLargeMatProds<T>::operator()(int argc, const char **argv) const {
	dim3 d16( 16,16);
	ExecCaps* caps = ExecCaps::currCaps();
	CuMatrix<T> mSnug = CuMatrix<T>::fill((T) 2, caps->maxGrid.y * d16.y, 2);
	T snugSum = mSnug.sum();
	assert(snugSum == 2 * ( caps->maxGrid.y * d16.y * 2 ));
	T ba[] = {1,2,3,4};
	int factor = 5;
	CuMatrix<T> b(ba, 2,2, true);
	outln("mSnug " << mSnug.toShortString());
	outln("b " << b.toShortString());
	CuMatrix<T> snugRes = mSnug * b;
	outln("snugRes " << snugRes.syncBuffers());
	CuMatrix<T> mOver = CuMatrix<T>::fill((T) 2, factor * caps->maxGrid.y * d16.y, 2);
	outln("mOver " << mOver.toShortString());
	T overSum = mOver.sum();
	assert(snugSum * factor == overSum);
	outln("overRes beg mOver ss " << mOver.toShortString() );
	outln("overRes beg b ss " << b.toShortString() );
	CuMatrix<T> overRes = mOver * b;
	outln("overRes " << overRes.syncBuffers());
	T snugResSum = snugRes.sum();
	T overResSum = overRes.sum();
	outln("snugResSum " << snugResSum);
	outln("overResSum " << overResSum);
	assert(snugResSum * factor == overResSum );
	return 0;
}

template int testHugeMatProds<float>::operator()(int argc, const char **argv) const;
template int testHugeMatProds<double>::operator()(int argc, const char **argv) const;
template int testHugeMatProds<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testHugeMatProds<T>::operator()(int argc, const char **argv) const {
	dim3 d16( 16,16);
	ExecCaps* caps = ExecCaps::currCaps();
    int total = b_util::getCount(argc,argv,5);

	outln("testHugeMatProds start");
	float exeTime;
    CuTimer timer;

    outln("creating big matrices");
	CuMatrix<T> ms = CuMatrix<T>::ones(12248,12248) ;
	memblo;
	outln("ms\n" << ms);
	outln("ms.tiler\n" << ms.tiler);

	timer.start();
	{
		CuMatrix<T> prodOnes = ms * ms;
		outln("prodOnes.sum\n" << prodOnes.sum() << " took " << timer.stop()/1000);
	}

	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 12248, 12248);
	{
		CuMatrix<T> prodMc = mc * mc;
		outln("prodMc.sum\n"<< prodMc.sum());
	}


	outln("mc\n"<< mc);
	outln("mc.tiler\n" << mc.tiler);
	timer.start();
	for(int i= 0; i < total; i++) {
		CuMatrix<T> mp = ms * mc;
		if(i == total-1) {
			outln("mp took " << (timer.stop()/1000.0f) << "s\n" << b_util::modStr(mp.lastMod));
			outln("mp\n" << mp);
			//mp.invalidateDevice();
			outln("mp.sum\n" << mp.sum());
		}
	}


	return 0;
}


template int testHugeMatProds2<float>::operator()(int argc, const char **argv) const;
template int testHugeMatProds2<double>::operator()(int argc, const char **argv) const;
template int testHugeMatProds2<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testHugeMatProds2<T>::operator()(int argc, const char **argv) const {
	dim3 d16( 16,16);
	ExecCaps* caps = ExecCaps::currCaps();

	outln("testHugeMatProds2 start");
	float exeTime;
    CuTimer timer;
   // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 80 * Mega);  // had no apparent effect

    outln("creating big matrices");
	CuMatrix<T> ms = CuMatrix<T>::identity(10000) ;
	CuMatrix<T> ms1 = CuMatrix<T>::identity(1000) ;
	outln("ms\n" << ms);
	printColoArrayDiagNe( ms.elements, ms.p, ms.n, (T)1);

	outln("ms1\n" << ms1);
	outln("ms.tiler\n" << ms.tiler);
	CuMatrix<T> mc = CuMatrix<T>::diagonal(10000, 5);
	outln("mc\n"<< mc);
	CuMatrix<T> mc1 = CuMatrix<T>::diagonal(1000, 5);
	mc1.syncBuffers();
	CuMatrix<T> mp1 = ms1 * mc1;
	printDevArrayDiag(mp1.tiler.buff(), "testHugeMatProds2 in " __FILE__ , __LINE__ ,mp1.p, 100 );
	outln("mc1\n"<< mc1);
	T mp1Sum = mp1.sum();
	outln("mp1.sum\n" << mp1Sum);
	assert(mp1Sum == 5000);

	outln("mc.tiler\n" << mc.tiler);
	timer.start();
	CuMatrix<T> mp = ms * mc;
	printDevArrayDiag(mp.tiler.buff(), "testHugeMatProds2 in " __FILE__ , __LINE__ ,mp.p, 100 );
	outln("\n\n\nafter printDevArrayDiag(mp.tiler.buff()\n\n\n");

	prtColoArrayDiag(mp.elements,"testHugeMatProds2 in " __FILE__ , __LINE__ ,  mp.p, 10000);
	outln("\n\n\nafter prtColoArrayDiag(mp.elements\n\n\n");
	/*
	 * 0-2875 : 5
	 * 2876-5362: nan -inf etc
	 * 5363-7512 : 0
	 * 7513- 8238 : 5
	 * 8239-9999 : blech
	 */

	outln("mp\n" << mp);
	outln("mp took " << (timer.stop()/1000.0f) << "s\n" << b_util::modStr(mp.lastMod));
	//mp.invalidateDevice();
	//mp1.invalidateDevice();
	outln("mp1\n" << mp1);
	T mpSum = mp.sum();
	assert(mpSum == 50000);
	outln("mp.sum\n" << mpSum);

	return 0;
}

template int testHugeMatProds3<float>::operator()(int argc, const char **argv) const;
template int testHugeMatProds3<double>::operator()(int argc, const char **argv) const;
template int testHugeMatProds3<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testHugeMatProds3<T>::operator()(int argc, const char **argv) const {
	dim3 d16( 16,16);
	ExecCaps* caps = ExecCaps::currCaps();
    int total = b_util::getCount(argc,argv,1);

	outln("testProductShapes start");
	float exeTime;
    CuTimer timer;

    outln("creating big matrices");
	CuMatrix<T> ms = CuMatrix<T>::ones(10000,10000) ;
	outln("ms\n" << ms);
	outln("ms.tiler\n" << ms.tiler);
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 10000, 10000);
	outln("mc\n"<< mc);
	outln("mc.tiler\n" << mc.tiler);
	timer.start();
	CuMatrix<T> mp = ms * mc;
	outln("mp took " << (timer.stop()/1000.0f) << "s\n" << b_util::modStr(mp.lastMod));
	outln("mp\n" << mp);

	mp.invalidateDevice();
	outln("mp.sum\n" << mp.sum());

	return 0;
}



template int testProductShapesTxB<float>::operator()(int argc, const char **argv) const;
template int testProductShapesTxB<double>::operator()(int argc, const char **argv) const;
template int testProductShapesTxB<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testProductShapesTxB<T>::operator()(int argc, const char **argv) const {
	outln("testProductShapesTxB first org");
	testProductShapes<T> org;
	org(argc,argv);

	outln("\n\n\ntestProductShapesTxB start");
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,argv,1);

	CuMatrix<T> m13by17= CuMatrix<T>::ones(13,17);
	CuMatrix<T> m17by1 = CuMatrix<T>::fill(2, 17,1);
	CuMatrix<T> mm2 = m13by17 * m17by1;
 	outln("m13by17 * m17by1 " << mm2.toShortString());

	T v[] = {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3};
	T v2[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5};

	CuMatrix<T> m1(v, 3,5, true);
	outln("m1 ");
	outln(m1.toString());
	CuMatrix<T> m2(v2,5,4, true);
	outln("m2 ");
	outln(m2.toString());
	CuMatrix<T> mm1 = m1 * m2;
	outln("m1 * m2 " << mm1.syncBuffers().toShortString());
	outln(mm1.toString());
	dim3 d(4,4);
	CuMatrix<T> mm1b = m1.matrixProduct(m2,&d).syncBuffers();
	outln("m1 * m2 d(4,4) " << mm1b.toShortString());
	outln(mm1b.toString());
	CuMatrix<T> m33 = CuMatrix<T>::ones(10,33);
	CuMatrix<T> m332 = CuMatrix<T>::ones(33,10) * 2;

	CuMatrix<T> mm3 = m33 * m332;
	outln(" m33 * m332 " << mm3.toShortString());
	outln(mm3.syncBuffers().toString());

	outln("creating big matrices...");
	checkCudaErrors(cudaDeviceSynchronize());
	timer.start();
	CuMatrix<T> ms = CuMatrix<T>::ones(1000,1000);
	DMatrix<T> d_ms = ms.asDmatrix(true);
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 1000,200).transpose();
	DMatrix<T> d_mc = mc.asDmatrix(true);
	CuMatrix<T> msc(1000,200,false,true);
	DMatrix<T> d_msc;
	msc.tile0(d_msc,false);
	exeTime = timer.stop()/1000.0f;
	outln("...creating big matrices took " << exeTime << "s");

	dim3 d8( 8,8);
	dim3 d16( 16,16);
	dim3 d32( 32,32);
	timer.start();
	for (int i = 0; i < total; i++) {
		ms.matrixProductTxdbL( d_msc,d_ms, d_mc, null);
	}
	T sum = msc.sum();
	msc.invalidateHost();
	outln(msc.toShortString() << " with default dims sum " << sum << " from \n" << msc.syncBuffers());

	exeTime = timer.stop()/1000.0f;
	outln("null took " << exeTime << " secs");
	timer.start();
	for (int i = 0; i < total; i++) {
		ms.matrixProductTxdbL(d_msc,d_ms, d_mc, &d8);
	}
	exeTime = timer.stop()/1000.0f;
	outln("8x8 took " << exeTime << " secs");
	T sum2 = msc.sum();
	msc.invalidateHost();
	outln("8x8 dims sum " << sum << ", sum2 " << sum2 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum2));

	timer.start();
	for (int i = 0; i < total; i++) {
		ms.matrixProductTxdbL(d_msc, d_ms, d_mc, &d16);
	}
	exeTime = timer.stop()/1000.0f;
	outln("16x16 took " << exeTime << " secs");
	T sum3 = msc.sum();
	msc.invalidateHost();
	outln("16x16 dims sum " << sum << ", sum3 " << sum3 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum3));

	timer.start();
	for (int i = 0; i < total; i++) {
		ms.matrixProductTxdbL(d_msc, d_ms, d_mc, &d32);
	}
	exeTime = timer.stop()/1000.0f;
	outln("32x32 took " << exeTime << " secs");
	T sum4 = msc.sum();
	outln("32x32 dims sum4 " << sum4 );
	msc.invalidateHost();
	outln("32x32 dims sum " << sum << " sum4 " << sum4 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum4));
	msc.invalidateHost();
	outln(msc.syncBuffers().toString());

	CuMatrix<T> sqr1 = CuMatrix<T>::ones(1000, 1000);
	uint quantity = sqr1.size/sizeof(T);
	uint wWide = 1250;
	uint hWide = quantity / wWide;
	CuMatrix<T> wide = CuMatrix<T>::ones(hWide, wWide);
	CuMatrix<T> mShort = CuMatrix<T>::ones(wWide, hWide) * 2;
	CuMatrix<T> rWide = CuMatrix<T>::zeros(hWide,hWide);
	CuMatrix<T>  tmshort = mShort.transpose();
	DMatrix<T> dRwide,dwide,txdDshort;

	rWide.tile0( dRwide, true);
	wide.tile0( dwide, true);
	tmshort.tile0( txdDshort,true);
	outln("wWide " << wWide <<", hWide " << hWide);
	outln("rWide prod matrix -> " << rWide.toShortString());
	outln("mShort " << mShort.toShortString());
	outln("tmshort " << tmshort.toShortString());
	outln("wide " << wide.toShortString());
	timer.start();
	for (int i = 0; i < total; i++) {
		wide.matrixProductTxdbL(dRwide, dwide,  txdDshort,&d32);
	}
	exeTime = timer.stop();
	T flow = rWide.flow(total,1,exeTime) ;
	outln( "count " << total << " of wide*mShort took " << exeTime << "ms or flow of " <<flow << "GB/s");
	T flelem = exeTime/rWide.size;
	outln( "perElem rWide " << flelem);

	return 0;
}


template int testProductShapesKernelPtr<float>::operator()(int argc, const char **argv) const;
template int testProductShapesKernelPtr<double>::operator()(int argc, const char **argv) const;
template int testProductShapesKernelPtr<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testProductShapesKernelPtr<T>::operator()(int argc, const char **argv) const {

	outln("testProductShapesTxB first org");
	testProductShapes<T> org;
	org(argc,argv);

	outln("\n\n\ntestProductShapesTxB start");
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,argv,1);

	void (*matProdKptr) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int) = &matrixProductKernelTxdB;

	CuMatrix<T> m13by17= CuMatrix<T>::ones(13,17);
	CuMatrix<T> m17by1 = CuMatrix<T>::fill(2, 17,1);
	CuMatrix<T> mm2 = m13by17 * m17by1;
 	outln("m13by17 * m17by1 " << mm2.toShortString());

	T v[] = {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3};
	T v2[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5};

	CuMatrix<T> m1(v, 3,5, true);
	outln("m1 ");
	outln(m1.toString());
	CuMatrix<T> m2(v2,5,4, true);
	outln("m2 ");
	outln(m2.toString());
	CuMatrix<T> mm1 = m1 * m2;
	outln("m1 * m2 " << mm1.syncBuffers().toShortString());
	outln(mm1.toString());
	dim3 d(4,4);
	CuMatrix<T> mm1b = m1.matrixProduct(m2,&d).syncBuffers();
	outln("m1 * m2 d(4,4) " << mm1b.toShortString());
	outln(mm1b.toString());
	CuMatrix<T> m33 = CuMatrix<T>::ones(10,33);
	CuMatrix<T> m332 = CuMatrix<T>::ones(33,10) * 2;

	CuMatrix<T> mm3 = m33 * m332;
	outln(" m33 * m332 " << mm3.toShortString());
	outln(mm3.syncBuffers().toString());

	outln("creating big sin/cos matrices");
	checkCudaErrors(cudaDeviceSynchronize());
	timer.start();
	CuMatrix<T> ms = CuMatrix<T>::ones(1000,1000);
	DMatrix<T> d_ms = ms.asDmatrix(true);
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 1000,200).transpose();
	CuMatrix<T> mcT = mc.transpose();
	CuMatrix<T> mcProd = ms * mcT;

	DMatrix<T> d_mc = mc.asDmatrix(true);
	DMatrix<T> d_mcT = mcT.asDmatrix(true);
    CuMatrix<T> msc(1000,200,false,true);
	DMatrix<T> d_msc;
	msc.tile0(d_msc,false);
	exeTime = timer.stop()/1000.0f;
	outln("creating big matrices " << exeTime << " secs");

	dim3 d8( 8,8);
	dim3 d16( 16,16);
	dim3 d32( 32,32);
	outln("null");
	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductKPtrL(d_msc,matProdKptr, d_ms, d_mc, null);
	}
	T sum = msc.sum();
	msc.invalidateHost();
	outln(msc.toShortString() << " with default dims sum " << sum << " from \n" << msc.syncBuffers());

	exeTime = timer.stop()/1000.0f;
	outln("null took " << exeTime << " secs");
	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductKPtrL(d_msc,matrixProductKernelTxdB,d_ms, d_mc, &d8);
	}
	exeTime = timer.stop()/1000.0f;
	outln("8x8 took " << exeTime << " secs");
	T sum2 = msc.sum();
	msc.invalidateHost();
	CuMatrix<T> diff = mcProd - msc;
	T diffSum = diff.sum();
	outln("diffSum " << diffSum);
	outln("8x8 dims sum " << sum << ", sum2 " << sum2 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum2));

	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductKPtrL(d_msc,matrixProductKernelTxdB, d_ms, d_mc, &d16);
	}
	exeTime = timer.stop()/1000.0f;
	outln("16x16 took " << exeTime << " secs");
	T sum3 = msc.sum();
	msc.invalidateHost();
	CuMatrix<T> diff2 = mcProd - msc;
	T diffSum2 = diff2.sum();
	outln("diffSum2 " << diffSum);
	outln("16x16 dims sum " << sum << ", sum3 " << sum3 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum3));

	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductKPtrL(d_msc,matrixProductKernelTxdB,d_ms, d_mc, &d32);
	}
	exeTime = timer.stop()/1000.0f;
	outln("32x32 took " << exeTime << " secs");
	T sum4 = msc.sum();
	outln("32x32 dims sum4 " << sum4 );
	msc.invalidateHost();
	outln("32x32 dims sum " << sum << " sum4" << sum4 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum4));
	msc.invalidateHost();
	outln(msc.syncBuffers().toString());

	CuMatrix<T> sqr1 = CuMatrix<T>::ones(1000, 1000);
	outln("sqr1 " << sqr1.toShortString());
	CuMatrix<T> sqr2 = 2 * sqr1;
	outln("sqr2 " << sqr2.toShortString());
	DMatrix<T> dsqr1,dsqr2, txdDsqr2, dRsqr;
	sqr1.tile0(dsqr1, true);
	sqr2.tile0(dsqr2, true);
	CuMatrix<T> txSqr2 = sqr2.transpose();

	txSqr2.tile0(txdDsqr2, true);
	CuMatrix<T> rSqr = sqr1 * sqr2;
	rSqr.tile0(dRsqr, true);

	outln("rSqr " << rSqr.syncBuffers());
	outln("txSqr2 " << txSqr2.syncBuffers());
	T rSqrSum = rSqr.sum();
	outln("rSqrSum " << rSqrSum);

	matrixProductKPtrL(dRsqr,matrixProductKernelTxdB,dsqr1, txdDsqr2, &d16);
	rSqr.invalidateHost();

	outln("txd rSqr " << rSqr.syncBuffers());
	T txRSqrSum = rSqr.sum();
	outln("txRSqrSum " << txRSqrSum);
	assert(txRSqrSum == rSqrSum);
	return 0;
}



template int testProductShapesLoop<float>::operator()(int argc, const char **argv) const;
template int testProductShapesLoop<double>::operator()(int argc, const char **argv) const;
template int testProductShapesLoop<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testProductShapesLoop<T>::operator()(int argc, const char **argv) const {
	outln("testProductShapesLoop start");
	int count = b_util::getCount(argc,argv,100);
	float exeTime;

	CuMatrix<T> sqr1 = CuMatrix<T>::ones(1000, 1000);
	outln("sqr1 " << sqr1.toShortString());
	CuMatrix<T> sqr2 = 2 * sqr1;
	outln("sqr2 " << sqr2.toShortString());
	CuMatrix<T> rSqr = sqr1 * sqr2;
	T rSqrSum = rSqr.sum();
	DMatrix<T> dsqr1,dsqr2, txdDsqr2, dRsqr;
	sqr1.tile0(dsqr1, sqr1.lastMod == mod_host);
	sqr2.tile0(dsqr2,  sqr2.lastMod == mod_host);
	CuMatrix<T> txSqr2 = sqr2.transpose();
	txSqr2.tile0(txdDsqr2,  txSqr2.lastMod == mod_host);
	rSqr.tile0(dRsqr,  rSqr.lastMod == mod_host);

	uint quantity = sqr1.size/sizeof(T);
	uint wWide = 1250;
	uint hWide = quantity / wWide;
	uint wWider = 1600;
	uint hWider = quantity / wWider;
	uint wWidest = 2000;
	uint hWidest = quantity / wWidest;
	uint wXWidest = 3125;
	uint hXWidest = quantity / wXWidest;
	uint wXXWidest = 4000;
	uint hXXWidest = quantity / wXXWidest;
	CuMatrix<T> wide = CuMatrix<T>::ones(hWide, wWide);
	outln("wide "<< wide.toShortString());
	CuMatrix<T> mShort = CuMatrix<T>::ones(wWide, hWide) * 2;
	outln("mShort "<< mShort.toShortString());
	CuMatrix<T> widemShort = wide * mShort;
	outln("widemShort " << widemShort.sum());

	CuMatrix<T> wider = CuMatrix<T>::ones(hWider, wWider);
	outln("wider "<< wider.toShortString());
	CuMatrix<T> mShorter = CuMatrix<T>::ones(wWider, hWider) * 2;
	outln("mShorter "<< mShorter.toShortString());
	CuMatrix<T> widermShorter = wider * mShorter;
	outln("widermShorter " << widermShorter.sum());

	CuMatrix<T> widest = CuMatrix<T>::ones(hWidest,wWidest);
	outln("widest "<< widest.toShortString());
	CuMatrix<T> mShortest = CuMatrix<T>::ones(wWidest,hWidest) * 2;
	outln("mShortest "<< mShortest.toShortString());

	CuMatrix<T> xwidest = CuMatrix<T>::ones(hXWidest,wXWidest);
	outln("xwidest "<< xwidest.toShortString());
	CuMatrix<T> xmShortest = CuMatrix<T>::ones(wXWidest,hXWidest) * 2;
	outln("xmShortest "<< xmShortest.toShortString());

	CuMatrix<T> xxwidest = CuMatrix<T>::ones(hXXWidest,wXXWidest);
	outln("xxwidest "<< xxwidest.toShortString());
	CuMatrix<T> xxmShortest = CuMatrix<T>::ones(wXXWidest,hXXWidest) * 2;
	outln("xxmShortest "<< xxmShortest.toShortString());

	CuMatrix<T> rWide = CuMatrix<T>::zeros(hWide,hWide);
	CuMatrix<T> trWide = CuMatrix<T>::zeros(wWide,wWide);
	CuMatrix<T> rWider = CuMatrix<T>::zeros(hWider,hWider);
	CuMatrix<T> trWider = CuMatrix<T>::zeros(wWider,wWider);
	CuMatrix<T> rWidest = CuMatrix<T>::zeros(hWidest,hWidest);
	CuMatrix<T> trWidest = CuMatrix<T>::zeros(wWidest,wWidest);
	CuMatrix<T> rxWidest = CuMatrix<T>::zeros(hXWidest,hXWidest);
	CuMatrix<T> trxWidest = CuMatrix<T>::zeros(wXWidest,wXWidest);
	CuMatrix<T> rxxWidest = CuMatrix<T>::zeros(hXXWidest,hXXWidest);
	CuMatrix<T> trxxWidest = CuMatrix<T>::zeros(wXXWidest,wXXWidest);

	DMatrix<T> dwide,txdDwide,dshort,txdDshort,dwider,txdDwider,dshorter,txdDshorter,dwidest,txdDwidest,dshortest,txdDshortest;
	DMatrix<T> dxwidest,dxshortest,dxxwidest,dxxshortest;
	DMatrix<T> dRwide,dRwider,dRwidest,tdRwide,tdRwider,tdRwidest;
	DMatrix<T> dRxwidest,dRxxwidest,tdRxwidest,tdRxxwidest;
	wide.tile0(dwide);

	CuMatrix<T> wideShort = wide * mShort;
	outln("wideShort.sum " << wideShort.sum());

	CuMatrix<T> txdWide = wide.transpose();
	txdWide.tile0(txdDwide);

	mShort.tile0(dshort);
	CuMatrix<T>  tmshort = mShort.transpose();
	tmshort.tile0( txdDshort);
	wider.tile0(dwider);

	// todo An aspect that errorered would be perfect
	// problem is the (anonymous) transposed matrix get deleted after the asDmatrix call,
	// which eventually results in the corruption of txDwider
	//  wider.transpose().asDmatrix(txdDwider);
	CuMatrix<T> txdWider = wider.transpose();
	txdWider.tile0(txdDwider);

	mShorter.tile0(dshorter);
	outln("mShorter " << mShorter.toShortString());
	CuMatrix<T>  tmShorter = mShorter.transpose();
	outln("tmShorter " << tmShorter.toShortString());
	dassert(mShorter.sum() == tmShorter.sum());
	tmShorter.tile0(txdDshorter);

	widest.tile0(dwidest);
	outln("widest " << widest.toShortString());
	CuMatrix<T> txWidest = widest.transpose();
	outln("widest.transpose() " << txWidest.toShortString());
	dassert(widest.sum() == txWidest.sum());
	txWidest.tile0(txdDwidest);

	mShortest.tile0(dshortest);
	outln("dshortest " << util<T>::pdm(dshortest));
	CuMatrix<T> txShortest =mShortest.transpose();
	dassert(mShortest.sum() == txShortest.sum());
	txShortest.tile0(txdDshortest);
	outln("txdDshortest " << util<T>::pdm(txdDshortest));

	xwidest.tile0(dxwidest);
	xmShortest.tile0(dxshortest);
	xxwidest.tile0(dxxwidest);
	xxmShortest.tile0(dxxshortest);
	rWide.tile0(dRwide);
	rWider.tile0(dRwider);
	rWidest.tile0(dRwidest);
	trWide.tile0(tdRwide);
	trWider.tile0(tdRwider);
	trWidest.tile0(tdRwidest);
	rxWidest.tile0(dRxwidest);
	trxWidest.tile0(tdRxwidest);
	rxxWidest.tile0(dRxxwidest);
	trxxWidest.tile0(tdRxxwidest);


	dim3 blocks[]  = { dim3(8,8),dim3(16,16),dim3(32,32)};
	map<string,float> runTimes;
	typedef pair<string,float> runpair;
    CuTimer timer;
	void (*matProdKptr[]) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int) =
		{	//matrixProductBandwidthKernel,
			matrixProductKernel,matrixProductKernel,
			matrixProductKernel2,
			matrixProductKernel3,
			matrixProductKernelTxdB,
			matrixProductKernelTxdB2};
	const char* names[] = {/*"kbandwdth ",*/"k1 ","k1 ","k2 ", "k3 ", "ktx ", "ktx2 "};
	T sum1 = -1,sum2=-1,sum3=-1,sum4=-1,sum5=-1,sum6=-1,sum7=-1;
    for(int kernel = 0; kernel < 6; kernel++) {
    	clock_t lastTime = clock();
    	sum1 = -1;
		outln("kernel " << kernel << "  " << matProdKernelName(matProdKptr[kernel]));
		for(int i = 0; i < 3; i++) {
			dim3* blockP = blocks + i;
			outln("using block dim " << b_util::pd3(blockP));
			outln("rSqr prod matrix -> " << rSqr.toShortString());
			timer.start();
			for (int i = 0; i < count; i++) {
				matrixProductKPtrL(dRsqr,matProdKptr[kernel], dsqr1, kernel > 3 ? txdDsqr2 : dsqr2, blockP);
				checkCudaError(cudaGetLastError());
			}
			if(sum1<0) {
				sum1 = rSqr.sum();
				outln("sum1 " << sum1 << ", rSqrSum " <<rSqrSum);
				if(kernel != 0) assert(rSqrSum == sum1);
			}
			exeTime = timer.stop();
			runTimes.insert(runTimes.end(),runpair(names[kernel] + b_util::pd3(blockP, util<T>::pdm(dsqr1) + "*" + util<T>::pdm(dsqr2)), exeTime/rSqr.size));
			int factor = 1; // (reads are of diff sizes so ignored; this is 'write' throughput)
			outln( count << " of sqr1*sqr2 took " << exeTime << "ms or flow of " << rSqr.flow(count,factor,exeTime) << "GB/s");
			outln( "perElem rSqr " << exeTime/rSqr.size);

			timer.start();
			for (int i = 0; i < count; i++) {
				// res, a, b
				matrixProductKPtrL(dRwide,matProdKptr[kernel], dwide,  kernel > 3 ? txdDshort: dshort, blockP);
			}
			if(sum2<0) {
				sum2 = rWide.sum();
				outln("sum2 rWide " << sum2);
			} else {
				assert(sum2 == rWide.sum());
			}
			exeTime = timer.stop();
			runTimes.insert(runTimes.end(),runpair(names[kernel] +b_util::pd3(blockP, util<T>::pdm(dwide) + "*" + util<T>::pdm(dshort)), exeTime/rWide.size));
			outln( count << " of wide*mShort took " << exeTime << "ms or flow of " << rWide.flow(count,factor,exeTime) << "GB/s");

			timer.start();
			for (int i = 0; i < count; i++) {
				matrixProductKPtrL(dRwider,matProdKptr[kernel], dwider, kernel > 3 ? txdDshorter : dshorter, blockP);
			}
			if(sum3<0) {
				sum3 = rWider.sum();
				outln("sum3 rWider " << sum3);
			} else {
				assert(sum3 == rWider.sum());
			}
			exeTime = timer.stop();
			runTimes.insert(runTimes.end(),runpair(names[kernel] +b_util::pd3(blockP, util<T>::pdm(dwider) + "*" + util<T>::pdm(dshorter) ), exeTime/rWider.size));
			outln( count << " of wider*mShorter took " << exeTime << "ms or flow of " << rWider.flow(count,factor,exeTime) << "GB/s");
			outln( "perElem rWider " << exeTime/rWider.size);

			timer.start();
			for (int i = 0; i < count; i++) {
				matrixProductKPtrL(dRwidest,matProdKptr[kernel], dwidest, kernel > 3 ? txdDshortest : dshortest, blockP);
			}
			if(sum4<0) {
				sum4 = rWidest.sum();
				outln("sum4 rWidest " << sum4);
			} else {
				assert(sum4 == rWidest.sum());
			}
			exeTime = timer.stop();
			runTimes.insert(runTimes.end(),runpair(names[kernel] +b_util::pd3(blockP, util<T>::pdm(dwidest) + "*" + util<T>::pdm(dshortest)), exeTime/rWidest.size));
			outln( count << " of widest*mShortest took " << exeTime << "ms or flow of " << rWidest.flow(count,factor,exeTime) << "GB/s");
			outln( "perElem rWidest " << exeTime/rWidest.size);
			outln( "sanity rWidest.sum " << rWidest.sum());

			outln(trWide.toShortString());

			outln("trWide prod matrix -> " << trWide.toShortString());
			timer.start();
			for (int i = 0; i < count; i++) {
				matrixProductKPtrL(tdRwide,matProdKptr[kernel], dshort, kernel > 3 ? txdDwide : dwide, blockP);
			}
			if(sum5<0) {
				sum5 = trWide.sum();
				outln("sum5 trWide " << sum5);
			} else {
				outln("sum5 trWide checking " << sum5 << " == " << trWide.sum());
				assert(sum5 == trWide.sum());
			}
			exeTime = timer.stop();
			runTimes.insert(runTimes.end(),runpair(names[kernel] +b_util::pd3(blockP, util<T>::pdm(dshort) + "*" + util<T>::pdm(dwide) ), exeTime/trWide.size));
			outln( count << " of mShort*wide took " << exeTime << "ms or flow of " << trWide.flow(count,factor,exeTime) << "GB/s");
			outln( "perElem trWide " << exeTime/trWide.size);
			outln( "sanity trWide.sum " << trWide.sum());

			timer.start();
			for (int i = 0; i < count; i++) {
				matrixProductKPtrL(tdRwider,matProdKptr[kernel], dshorter, kernel > 3 ? txdDwider : dwider, blockP);
			}
			if(sum6<0) {
				sum6= trWider.sum();
				outln("trWider sum6 " << sum6);
			} else {
				outln("trWider sum6 checking " << sum6 << " == " << trWider.sum());
				assert(sum6 == trWider.sum());
			}
			exeTime = timer.stop();
			runTimes.insert(runTimes.end(),runpair(names[kernel] +b_util::pd3(blockP, util<T>::pdm(dshorter) + "*" + util<T>::pdm(dwider) ), exeTime/trWider.size));
			outln( count << " of mShorter*wider took " << exeTime << "ms or flow of " << trWider.flow(count,factor,exeTime) << "GB/s");
			outln( "perElem trWider " << exeTime/trWider.size);
			outln( "sanity trWider.sum " << trWider.sum());

			timer.start();
			for (int i = 0; i < count; i++) {
				matrixProductKPtrL(tdRwidest,matProdKptr[kernel], dshortest, kernel > 3 ? txdDwidest : dwidest, blockP);
			}
			if(sum7<0) {
				sum7 = trWidest.sum();
				outln("sum7 trWidest" << sum7);
			} else {
				outln("sum7 trWidest checking " << sum7 << " ==  " << trWidest.sum());
				assert(sum7 == trWidest.sum());
			}
			exeTime = timer.stop();
			runTimes.insert(runTimes.end(),runpair(names[kernel] +b_util::pd3(blockP, util<T>::pdm(dshortest) + "*" + util<T>::pdm(dwidest) ), exeTime/trWidest.size));
			outln( count << " of mShortest*widest took " << exeTime << "ms or flow of " << trWidest.flow(count,factor,exeTime) << "GB/s");
			outln( "perElem trWidest " << exeTime/trWidest.size);
		}
	    double delta = b_util::diffclock(clock(), lastTime) / 1000;
	    outln("Completed! s " << (3 * count) << " took " << delta << " secs\n\n\n");

	    outln("results");
		typedef typename map<string, float>::iterator iterator;
		for(iterator it = runTimes.begin();it != runTimes.end();it++) {
			outln((*it).first << " took " << (*it).second << "ms");
		}
    }

	outln("testProductShapesLoop finish");
	return 0;
}


template int testAutodot<float>::operator()(int argc, const char **argv) const;
template int testAutodot<double>::operator()(int argc, const char **argv) const;
template int testAutodot<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testAutodot<T>::operator()(int argc, const char **argv) const {
	outln("testAutodot start");
	int count = b_util::getCount(argc,argv,10000);
	float exeTime;
	CuMatrix<T> m = CuMatrix<T>::sin(1000, 1000);
	const float sizeG= 1. * m.size / Giga;
	const uint xfer = count * sizeG;
	//const uint lengthInTs = src.size/sizeof(T);
	float memFlowIter = 0;
    CuTimer timer;

    outln("m " << m.syncBuffers());
	outln("m.sum " << m.sum());
	T sqrsum = (m % m).sum();
	outln("m.sqrsum " << sqrsum);
	dassert(util<T>::almostEquals(sqrsum,m.autoDot()));
	clock_t lastTime = clock();
	outln("\n\n\nPerforming autoDot on 1000x1000 matrx " << count << " times");
	T s = 0;
	timer.start();
	for (int i = 0; i < count; i++) {
		s += m.autoDot();
	}
    exeTime = timer.stop();
    memFlowIter = xfer * 1000/exeTime;
	outln("m.autoDot() N " << count << " took exeTime " << (exeTime /1000) << "s or flow (w) of " << memFlowIter << "GB/s");
	dassert( util<T>::almostEquals(sqrsum*count,  s));
	double delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("Completed! s " << count << " took " << delta << " secs\n\n\n");
	outln("testAutodot finish");
	return 0;
}

template int testMultLoop<float>::operator()(int argc, const char **argv) const;
template int testMultLoop<double>::operator()(int argc, const char **argv) const;
template int testMultLoop<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testMultLoop<T>::operator()(int argc, const char **argv) const {
	outln("testMultLoop start");
	CuMatrix<T> m1 = CuMatrix<T>::ones(999, 999);
	outln("m1 " << m1.syncBuffers());
	CuMatrix<T> m1b = m1 * 2;
	outln("m1b " << m1b.syncBuffers());
	CuMatrix<T> m2 = CuMatrix<T>::ones(999, 999) * 2;
	checkCudaErrors(cudaDeviceSynchronize());
	outln("made mats m1 " << m1.toShortString() << ", m2 " << m2.toShortString() << "\n");
	outln("m2 " << m2.syncBuffers());
	int blocks;
	int threads;
	int n = m1.m * m1.n;
	getReductionExecContext(blocks, threads, n);
	outln("blocks " << blocks << "\n");

	CuMatrix<T> buffer = CuMatrix<T>::zeros(m1.m, m2.n);

	DMatrix<T> m1_d;
	m1.tile0(m1_d);
	DMatrix<T> m2_d;
	m2.tile0(m2_d);
	//outln("after sync");
	DMatrix<T> m3_d;
	buffer.tile0(m3_d, false);
	checkCudaErrors(cudaDeviceSynchronize());
	outln("m1 " << m1.toShortString() << ", m1_d " << util<T>::pdm(m1_d));
	outln(m1.syncBuffers().toString());
	outln("m2 " << m2.toShortString() << ", m2_d " << util<T>::pdm(m2_d));
	outln(m2.syncBuffers().toString());
	outln("buffer " << buffer.toShortString() << ", m3_d " << util<T>::pdm(m3_d));

	outln("\n\n\ntestMultLoop -- this will take a moment (listen for the GPU fan)\n\nm1 " <<
			&m1 << ", m2 " << &m2 << ", buff " << &buffer << "\nm1.d " << m1.tiler.currBuffer() <<
			", m2.d " << m2.tiler.currBuffer() << ", buff.d " << buffer.tiler.currBuffer() << "\n\n");

	clock_t lastTime = clock();
	int count = b_util::getCount(argc,argv,10);
	for (int i = 0; i < count; i++) {
		matrixProductL(m3_d, m1_d, m2_d, 0);
	}
	double delta = b_util::diffclock(clock(), lastTime) / 1000;
	T s = buffer.sum();
	outln("s " << s << " took " << delta << " secs");
	//assert(abs( 2e9 - s) <.001);

	buffer.invalidateHost();
	outln("buff ");
	outln(buffer.syncBuffers().toString());

	outln("testMultLoop finish");

	return 0;

}


#include "tests.cc"
