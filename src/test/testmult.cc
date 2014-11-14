#include "tests.h"
#include "../CuMatrix.h"
#include "../util.h"
#include "../caps.h"
#include "../Kernels.h"


template int testSqrMatsMultSmall<float>::operator()(int argc, char const ** args) const;
template int testSqrMatsMultSmall<double>::operator()(int argc, char const ** args) const;
template int testSqrMatsMultSmall<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testSqrMatsMultSmall<T>::operator()(int argc, const char** args) const {
	outln("testSqrMatsMult start");
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,args,1);
	outln("creating big matrices");
	checkCudaErrors(cudaDeviceSynchronize());
	timer.start();
	CuMatrix<T> ms = CuMatrix<T>::increasingRows(1,128,128) ;
	DMatrix<T> d_ms = ms.asDmatrix();
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 128,128);
	CuMatrix<T> txmc = mc.transpose();
	outln("txmc\n"<< txmc.syncHost());
	DMatrix<T> d_mc = mc.asDmatrix();
	DMatrix<T> d_txmc = txmc.asDmatrix();
	CuMatrix<T> msc(128,128,false,true);
	DMatrix<T> d_msc;
	msc.asDmatrix(d_msc,false,false);
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

template int testSqrMatsMult<float>::operator()(int argc, char const ** args) const;
template int testSqrMatsMult<double>::operator()(int argc, char const ** args) const;
template int testSqrMatsMult<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testSqrMatsMult<T>::operator()(int argc, const char** args) const {
	outln("testSqrMatsMult start");
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,args,1);

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
	DMatrix<T> d_ms = ms.asDmatrix();
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 1024,1024);
	CuMatrix<T> txmc = mc.transpose();
	DMatrix<T> d_mc = mc.asDmatrix();
	DMatrix<T> d_txmc = txmc.asDmatrix();
	CuMatrix<T> msc(1024,1024,false,true);
	DMatrix<T> d_msc;
	msc.asDmatrix(d_msc,false,false);
	exeTime = timer.stop()/1000.0f;
	outln("creating big matrices " << exeTime << " secs");

	dim3 blocks[]  = { dim3(32,32),dim3(16,16), dim3(8,8)};
	map<string,float> runTimes;
	typedef pair<string,float> runpair;
	const char* names[] = {"k1 ","k2 ", "ktx ", "ktx2 "};

	void (*matProdKptr[]) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int) =  {matrixProductKernel,matrixProductKernel2,matrixProductKernelTxdB,matrixProductKernelTxdB2
#ifdef CuMatrix_Enable_Cdp
, matrixProductReductionTxdBKernel
#endif
	};

    for(int kernel = 0; kernel < 1; kernel++) {
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

template int testProductKernel3<float>::operator()(int argc, char const ** args) const;
template int testProductKernel3<double>::operator()(int argc, char const ** args) const;
template int testProductKernel3<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testProductKernel3<T>::operator()(int argc, const char** args) const {
	outln("testProductShapes start");
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,args,1);

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
	DMatrix<T> d_ms = ms.asDmatrix();
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 1000,200);
	DMatrix<T> d_mc = mc.asDmatrix();
	CuMatrix<T> msc(1000,200,false,true);
	DMatrix<T> d_msc;
	msc.asDmatrix(d_msc,false,false);
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

template int testProductShapes<float>::operator()(int argc, char const ** args) const;
template int testProductShapes<double>::operator()(int argc, char const ** args) const;
template int testProductShapes<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testProductShapes<T>::operator()(int argc, const char** args) const {
	outln("testProductShapes start");
	checkCudaError(cudaGetLastError());
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,args,1);


    CuMatrix<T> x = CuMatrix<T>::increasingRows(1,100,1);
    CuMatrix<T> x2 =  2 * x |= 3 * x;
    CuMatrix<T> xb = x2.addBiasColumn();
    outln("xb\n" << xb.syncBuffers());

    CuMatrix<T> mz = CuMatrix<T>::zeros(3,1);
    CuMatrix<T> m1z = CuMatrix<T>::ones(3,1);

    CuMatrix<T> prod = mz.transpose() * xb.transpose();
    outln("prod\n" << prod.syncBuffers());
    CuMatrix<T> prod2 = m1z.transpose() * xb.transpose();
    outln("prod2\n" << prod2.syncBuffers());

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
	checkCudaError(cudaGetLastError());
	dim3 d(4,4);
	CuMatrix<T> mm1b = m1.matrixProduct(m2,&d).syncBuffers();
	outln("m1 * m2 d(4,4) " << mm1b.toShortString());
	outln(mm1b.toString());
	CuMatrix<T> m33 = CuMatrix<T>::ones(10,33);
	outln("m33 " << m33.toShortString());
	CuMatrix<T> m332 = CuMatrix<T>::ones(33,10) * 2;
	outln("m332 " << m332.toShortString());

	CuMatrix<T> mm3 = m33 * m332;
	outln(" m33 * m332 " << mm3.toShortString());
	outln(mm3.syncBuffers().toString());

	outln("creating big matrices");
	checkCudaErrors(cudaDeviceSynchronize());
	CuMatrix<T> ms = CuMatrix<T>::ones(1000,1000) ;
	DMatrix<T> d_ms = ms.asDmatrix();
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 1000,200);
	DMatrix<T> d_mc = mc.asDmatrix();
	CuMatrix<T> msc(1000,200,false,true);
	DMatrix<T> d_msc;
	msc.asDmatrix(d_msc,false,false);

	dim3 d8( 8,8);
	dim3 d16( 16,16);
	dim3 d32( 32,32);
	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductL(d_msc,d_ms, d_mc, null);
	}
	T sum = msc.sum();
	msc.invalidateHost();
	outln(msc.toShortString() << " with default dims sum " << sum );

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


template int testProductShapesTxB<float>::operator()(int argc, char const ** args) const;
template int testProductShapesTxB<double>::operator()(int argc, char const ** args) const;
template int testProductShapesTxB<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testProductShapesTxB<T>::operator()(int argc, const char** args) const {
	outln("testProductShapesTxB first org");
	testProductShapes<T> org;
	org(argc,args);

	outln("\n\n\ntestProductShapesTxB start");
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,args,1);

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
	DMatrix<T> d_ms = ms.asDmatrix();
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 1000,200).transpose();
	DMatrix<T> d_mc = mc.asDmatrix();
	CuMatrix<T> msc(1000,200,false,true);
	DMatrix<T> d_msc;
	msc.asDmatrix(d_msc,false,false);
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

	rWide.asDmatrix( dRwide);
	wide.asDmatrix( dwide);
	tmshort.asDmatrix( txdDshort);
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


template int testProductShapesKernelPtr<float>::operator()(int argc, char const ** args) const;
template int testProductShapesKernelPtr<double>::operator()(int argc, char const ** args) const;
template int testProductShapesKernelPtr<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testProductShapesKernelPtr<T>::operator()(int argc, const char** args) const {

	outln("testProductShapesTxB first org");
	testProductShapes<T> org;
	org(argc,args);

	outln("\n\n\ntestProductShapesTxB start");
	float exeTime;
    CuTimer timer;
    int total = b_util::getCount(argc,args,1);

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
	DMatrix<T> d_ms = ms.asDmatrix();
	CuMatrix<T> mc = CuMatrix<T>::increasingColumns(1, 1000,200).transpose();
	DMatrix<T> d_mc = mc.asDmatrix();
	CuMatrix<T> msc(1000,200,false,true);
	DMatrix<T> d_msc;
	msc.asDmatrix(d_msc,false,false);
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
	outln("16x16 dims sum " << sum << ", sum3 " << sum3 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum3));

	timer.start();
	for (int i = 0; i < total; i++) {
		matrixProductKPtrL(d_msc,matrixProductKernelTxdB,d_mc, d_msc,&d32);
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



//template <typename T> struct  : public Test<T> {	int operator()(int argc, const char** args)const;};


template int testProductShapesLoop<float>::operator()(int argc, char const ** args) const;
template int testProductShapesLoop<double>::operator()(int argc, char const ** args) const;
template int testProductShapesLoop<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testProductShapesLoop<T>::operator()(int argc, const char** args) const {
	outln("testProductShapesLoop start");
	int count = b_util::getCount(argc,args,100);
	float exeTime;

	CuMatrix<T> sqr1 = CuMatrix<T>::ones(1000, 1000);
	outln("sqr1 " << sqr1.toShortString());
	CuMatrix<T> sqr2 = 2 * sqr1;
	outln("sqr2 " << sqr2.toShortString());
	CuMatrix<T> rSqr = sqr1 * sqr2;
	T rSqrSum = rSqr.sum();
	DMatrix<T> dsqr1,dsqr2, txdDsqr2, dRsqr;
	sqr1.asDmatrix(dsqr1);
	sqr2.asDmatrix(dsqr2);
	sqr2.transpose().asDmatrix(txdDsqr2);
	rSqr.asDmatrix(dRsqr);

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
	wide.asDmatrix(dwide);
	wide.transpose().asDmatrix(txdDwide);

	mShort.asDmatrix(dshort);
	CuMatrix<T>  tmshort = mShort.transpose();
	tmshort.asDmatrix( txdDshort);
	wider.asDmatrix(dwider);
	wider.transpose().asDmatrix(txdDwider);

	mShorter.asDmatrix(dshorter);
	outln("mShorter " << mShorter.toShortString());
	CuMatrix<T>  tmShorter = mShorter.transpose();
	outln("tmShorter " << tmShorter.toShortString());
	dassert(mShorter.sum() == tmShorter.sum());
	tmShorter.asDmatrix(txdDshorter);

	widest.asDmatrix(dwidest);
	outln("widest " << widest.toShortString());
	outln("widest.transpose() " << widest.transpose().toShortString());
	CuMatrix<T> txWidest = widest.transpose();
	dassert(widest.sum() == txWidest.sum());
	txWidest.asDmatrix(txdDwidest);

	mShortest.asDmatrix(dshortest);
	outln("dshortest " << util<T>::pdm(dshortest));
	CuMatrix<T> txShortest =mShortest.transpose();
	dassert(mShortest.sum() == txShortest.sum());
	txShortest.asDmatrix(txdDshortest);
	outln("txdDshortest " << util<T>::pdm(txdDshortest));

	xwidest.asDmatrix(dxwidest);
	xmShortest.asDmatrix(dxshortest);
	xxwidest.asDmatrix(dxxwidest);
	xxmShortest.asDmatrix(dxxshortest);
	rWide.asDmatrix(dRwide);
	rWider.asDmatrix(dRwider);
	rWidest.asDmatrix(dRwidest);
	trWide.asDmatrix(tdRwide);
	trWider.asDmatrix(tdRwider);
	trWidest.asDmatrix(tdRwidest);
	rxWidest.asDmatrix(dRxwidest);
	trxWidest.asDmatrix(tdRxwidest);
	rxxWidest.asDmatrix(dRxxwidest);
	trxxWidest.asDmatrix(tdRxxwidest);


	dim3 blocks[]  = { dim3(32,32),dim3(16,16), dim3(8,8)};
	map<string,float> runTimes;
	typedef pair<string,float> runpair;
    CuTimer timer;
	void (*matProdKptr[]) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int) =
		{	matrixProductBandwidthKernel,
			matrixProductKernel,
			matrixProductKernel2,
			matrixProductKernel3,
			matrixProductKernelTxdB,
			matrixProductKernelTxdB2};
	const char* names[] = {"kbandwdth ","k1 ","k2 ", "k3 ", "ktx ", "ktx2 "};
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


template int testAutodot<float>::operator()(int argc, char const ** args) const;
template int testAutodot<double>::operator()(int argc, char const ** args) const;
template int testAutodot<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testAutodot<T>::operator()(int argc, const char** args) const {
	outln("testAutodot start");
	int count = b_util::getCount(argc,args,10000);
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

template int testMultLoop<float>::operator()(int argc, char const ** args) const;
template int testMultLoop<double>::operator()(int argc, char const ** args) const;
template int testMultLoop<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testMultLoop<T>::operator()(int argc, const char** args) const {
	outln("testMultLoop start");
	CuMatrix<T> m1 = CuMatrix<T>::ones(999, 999);
	outln("m1 " << m1.syncBuffers());
	CuMatrix<T> m1b = m1 * 2;
	outln("m1b " << m1b.syncBuffers());
	CuMatrix<T> m2 = CuMatrix<T>::ones(999, 999) * 2;
	checkCudaErrors(cudaDeviceSynchronize());
	outln("made mats m1 " << m1.toShortString() << ", m2 " << m2.toShortString() << "\n");
	outln("m2 " << m2.syncBuffers());
	uint blocks;
	uint threads;
	uint n = m1.m * m1.n;
	getReductionExecContext(blocks, threads, n);
	outln("blocks " << blocks << "\n");

	CuMatrix<T> buffer = CuMatrix<T>::zeros(m1.m, m2.n);

	DMatrix<T> m1_d;
	m1.asDmatrix(m1_d, true);
	DMatrix<T> m2_d;
	m2.asDmatrix(m2_d, true);
	//outln("after sync");
	DMatrix<T> m3_d;
	buffer.asDmatrix(m3_d, false);
	checkCudaErrors(cudaDeviceSynchronize());
	outln("m1 " << m1.toShortString() << ", m1_d " << util<T>::pdm(m1_d));
	outln(m1.syncBuffers().toString());
	outln("m2 " << m2.toShortString() << ", m2_d " << util<T>::pdm(m2_d));
	outln(m2.syncBuffers().toString());
	outln("buffer " << buffer.toShortString() << ", m3_d " << util<T>::pdm(m3_d));

	outln("\n\n\ntestMultLoop -- this will take a moment (listen for the GPU fan)\n\nm1 " << &m1 << ", m2 " << &m2 << ", buff " << &buffer << "\nm1.d " << m1.d_elements << ", m2.d " << m2.d_elements << ", buff.d " << buffer.d_elements << "\n\n");

	clock_t lastTime = clock();
	int count = b_util::getCount(argc,args,10);
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
