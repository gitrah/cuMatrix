#include "tests.h"
#include "../util.h"
#include "../caps.h"

template int testProductShapes<float>::operator()(int argc, char const ** args) const;
template int testProductShapes<double>::operator()(int argc, char const ** args) const;
template <typename T> int testProductShapes<T>::operator()(int argc, const char** args) const {
	outln("testProductShapes start");
	char *count = null;
	getCmdLineArgumentString(argc, (const char **) args, "count", &count);
	int total = 1;
	if(count) {
		std::stringstream(count) >> total;
	}

	Matrix<T> m13by17= Matrix<T>::ones(13,17);
	Matrix<T> m17by1 = Matrix<T>::fill(2, 17,1);
	Matrix<T> mm2 = m13by17 * m17by1;
 	outln("m13by17 * m17by1 " << mm2.toShortString());

	T v[] = {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3};
	T v2[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5};

	Matrix<T> m1(v, 3,5, true);
	outln("m1 ");
	outln(m1.toString());
	Matrix<T> m2(v2,5,4, true);
	outln("m2 ");
	outln(m2.toString());
	Matrix<T> mm1 = m1 * m2;
	outln("m1 * m2 " << mm1.syncBuffers().toShortString());
	outln(mm1.toString());
	dim3 d(4,4);
	Matrix<T> mm1b = m1.matrixProduct(m2,&d).syncBuffers();
	outln("m1 * m2 d(4,4) " << mm1b.toShortString());
	outln(mm1b.toString());
	Matrix<T> m33 = Matrix<T>::ones(10,33);
	Matrix<T> m332 = Matrix<T>::ones(33,10) * 2;

	Matrix<T> mm3 = m33 * m332;
	outln(" m33 * m332 " << mm3.toShortString());
	outln(mm3.syncBuffers().toString());

	clock_t lastTime;
	double delta;
	outln("creating big sin/cos matrices");
	checkCudaErrors(cudaDeviceSynchronize());
	lastTime = clock();
	Matrix<T> ms = Matrix<T>::ones(1000,1000) * 2;
	DMatrix<T> d_ms = ms.asDmatrix();
	Matrix<T> mc = Matrix<T>::ones(1000,1000) * 5;
	DMatrix<T> d_mc = mc.asDmatrix();
	Matrix<T> msc(1000,1000,false,true);
	DMatrix<T> d_msc;
	msc.asDmatrix(d_msc,false,false);
	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("creating big matrices " << delta << " secs");

	dim3 d8( 8,8);
	dim3 d16( 16,16);
	dim3 d32( 32,32);
	outln("null");
	lastTime = clock();
	for (int i = 0; i < total; i++) {
		ms.matrixProductL(d_msc,d_ms, d_mc, null);
	}
	T sum = msc.sum();
	msc.invalidateHost();
	outln(msc.toShortString() << " with default dims sum " << sum << " from \n" << msc.syncBuffers());

	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("null took " << delta << " secs");
	lastTime = clock();
	for (int i = 0; i < total; i++) {
		ms.matrixProductL(d_msc,d_ms, d_mc, &d8);
	}
	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("8x8 took " << delta << " secs");
	T sum2 = msc.sum();
	msc.invalidateHost();
	outln("8x8 dims sum " << sum2 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum2));

	lastTime = clock();
	for (int i = 0; i < total; i++) {
		ms.matrixProductL(d_msc, d_ms, d_mc, &d16);
	}
	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("16x16 took " << delta << " secs");
	T sum3 = msc.sum();
	msc.invalidateHost();
	outln("16x16 dims sum " << sum3 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum3));

	lastTime = clock();
	for (int i = 0; i < total; i++) {
		ms.matrixProductL(d_ms, d_mc, d_msc,&d32);
	}
	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("32x32 took " << delta << " secs");
	T sum4 = msc.sum();
	msc.invalidateHost();
	outln("32x32 dims sum " << sum4 << " from \n" << msc.syncBuffers());
	dassert(util<T>::almostEquals(sum ,sum4));
	msc.invalidateHost();
	outln(msc.syncBuffers().toString());

	return 0;
}

template int testProductShapesLoop<float>::operator()(int argc, char const ** args) const;
template int testProductShapesLoop<double>::operator()(int argc, char const ** args) const;
template <typename T> int testProductShapesLoop<T>::operator()(int argc, const char** args) const {
	outln("testProductShapesLoop start");
	int count = b_util::getCount(argc,args,100);
	clock_t lastTime = clock();
	float exeTime;

	Matrix<T> sqr1 = Matrix<T>::ones(1000, 1000);
	Matrix<T> sqr2 = 2 * sqr1;
	Matrix<T> rSqr = Matrix<T>::zeros(1000,1000);
	DMatrix<T> dsqr1,dsqr2, dRsqr;
	sqr1.asDmatrix(dsqr1);
	sqr2.asDmatrix(dsqr2);
	rSqr.asDmatrix(dRsqr);

	uint quantity = sqr1.size/sizeof(T);
	uint wWide = 1250;
	uint hWide = quantity / wWide;
	outln("wWide " << wWide <<", hWide " << hWide);
	uint wWider = 1600;
	uint hWider = quantity / wWider;
	outln("wWider " << wWider <<", hWider " << hWider);
	uint wWidest = 2000;
	uint hWidest = quantity / wWidest;
	outln("wWidest " << wWidest <<", hWidest " << hWidest);
	uint wXWidest = 3125;
	uint hXWidest = quantity / wXWidest;
	outln("wXWidest " << wXWidest <<", hXWidest " << hXWidest);
	uint wXXWidest = 4000;
	uint hXXWidest = quantity / wXXWidest;
	outln("wXXWidest " << wXXWidest <<", hXXWidest " << hXXWidest);
	Matrix<T> wide = Matrix<T>::ones(hWide, wWide);
	Matrix<T> mShort = Matrix<T>::ones(wWide, hWide) * 2;
	Matrix<T> wider = Matrix<T>::ones(hWider, wWider);
	Matrix<T> mShorter = Matrix<T>::ones(wWider, hWider) * 2;
	Matrix<T> widest = Matrix<T>::ones(hWidest,wWidest);
	Matrix<T> mShortest = Matrix<T>::ones(wWidest,hWidest) * 2;
	Matrix<T> xwidest = Matrix<T>::ones(hXWidest,wXWidest);
	Matrix<T> xmShortest = Matrix<T>::ones(wXWidest,hXWidest) * 2;
	Matrix<T> xxwidest = Matrix<T>::ones(hXXWidest,wXXWidest);
	Matrix<T> xxmShortest = Matrix<T>::ones(wXXWidest,hXXWidest) * 2;

	Matrix<T> rWide = Matrix<T>::zeros(hWide,hWide);
	Matrix<T> trWide = Matrix<T>::zeros(wWide,wWide);
	Matrix<T> rWider = Matrix<T>::zeros(hWider,hWider);
	Matrix<T> trWider = Matrix<T>::zeros(wWider,wWider);
	Matrix<T> rWidest = Matrix<T>::zeros(hWidest,hWidest);
	Matrix<T> trWidest = Matrix<T>::zeros(wWidest,wWidest);
	Matrix<T> rxWidest = Matrix<T>::zeros(hXWidest,hXWidest);
	Matrix<T> trxWidest = Matrix<T>::zeros(wXWidest,wXWidest);
	Matrix<T> rxxWidest = Matrix<T>::zeros(hXXWidest,hXXWidest);
	Matrix<T> trxxWidest = Matrix<T>::zeros(wXXWidest,wXXWidest);

	DMatrix<T> dwide,dshort,dwider,dshorter,dwidest,dshortest;
	DMatrix<T> dxwidest,dxshortest,dxxwidest,dxxshortest;
	DMatrix<T> dRwide,dRwider,dRwidest,tdRwide,tdRwider,tdRwidest;
	DMatrix<T> dRxwidest,dRxxwidest,tdRxwidest,tdRxxwidest;
	wide.asDmatrix(dwide);
	mShort.asDmatrix(dshort);
	wider.asDmatrix(dwider);
	mShorter.asDmatrix(dshorter);
	widest.asDmatrix(dwidest);
	mShortest.asDmatrix(dshortest);
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

    CuTimer timer;
    for(int i = 0; i < 3; i++) {
    	dim3* blockP = blocks + i;
    	outln("using block dim " << b_util::pd3(blockP));

		outln("rSqr prod matrix -> " << rSqr.toShortString());
		timer.start();
		for (int i = 0; i < count; i++) {
			sqr1.matrixProductL(dRsqr, dsqr1, dsqr2, 0);
		}
		exeTime = timer.stop();
		int factor = 1; // (reads are of diff sizes so ignored; this is 'write' throughput)
		outln( count << " of sqr1*sqr2 took " << exeTime << "ms or flow of " << rSqr.flow(count,factor,exeTime) << "GB/s");
		outln( "perElem rSqr " << exeTime/rSqr.size);
		outln( "sanity rSqr.sum " << rSqr.sum());

		outln("rWide prod matrix -> " << rWide.toShortString());
		timer.start();
		for (int i = 0; i < count; i++) {
			wide.matrixProductL(dRwide, dwide, dshort, blockP);
		}
		exeTime = timer.stop();
		outln( count << " of wide*mShort took " << exeTime << "ms or flow of " << rWide.flow(count,factor,exeTime) << "GB/s");
		outln( "perElem rWide " << exeTime/rWide.size);
		outln( "sanity rWide.sum " << rWide.sum());

		outln("rWider prod matrix -> " << rWider.toShortString());
		timer.start();
		for (int i = 0; i < count; i++) {
			wider.matrixProductL(dRwider, dwider, dshorter, blockP);
		}
		exeTime = timer.stop();
		outln( count << " of wider*mShorter took " << exeTime << "ms or flow of " << rWider.flow(count,factor,exeTime) << "GB/s");
		outln( "perElem rWider " << exeTime/rWider.size);
		outln( "sanity rWider.sum " << rWider.sum());

		outln("rWidest prod matrix -> " << rWidest.toShortString());
		timer.start();
		for (int i = 0; i < count; i++) {
			widest.matrixProductL(dRwidest, dwidest, dshortest, blockP);
		}
		exeTime = timer.stop();
		outln( count << " of widest*mShortest took " << exeTime << "ms or flow of " << rWidest.flow(count,factor,exeTime) << "GB/s");
		outln( "perElem rWidest " << exeTime/rWidest.size);
		outln( "sanity rWidest.sum " << rWidest.sum());


		outln("trWide prod matrix -> " << trWide.toShortString());
		timer.start();
		for (int i = 0; i < count; i++) {
			mShort.matrixProductL(tdRwide, dshort, dwide, blockP);
		}
		exeTime = timer.stop();
		outln( count << " of mShort*wide took " << exeTime << "ms or flow of " << trWide.flow(count,factor,exeTime) << "GB/s");
		outln( "perElem trWide " << exeTime/trWide.size);
		outln( "sanity trWide.sum " << trWide.sum());

		outln("trWider prod matrix -> " << trWider.toShortString());
		timer.start();
		for (int i = 0; i < count; i++) {
			mShorter.matrixProductL(tdRwider, dshorter, dwider, blockP);
		}
		exeTime = timer.stop();
		outln( count << " of mShorter*wider took " << exeTime << "ms or flow of " << trWider.flow(count,factor,exeTime) << "GB/s");
		outln( "perElem trWider " << exeTime/trWider.size);
		outln( "sanity trWider.sum " << trWider.sum());

		outln("trWidest prod matrix -> " << trWidest.toShortString());
		timer.start();
		for (int i = 0; i < count; i++) {
			mShortest.matrixProductL(tdRwidest, dshortest, dwidest, blockP);
		}
		exeTime = timer.stop();
		outln( count << " of mShortest*widest took " << exeTime << "ms or flow of " << trWidest.flow(count,factor,exeTime) << "GB/s");
		outln( "perElem trWidest " << exeTime/trWidest.size);
		outln( "sanity trWidest.sum " << trWidest.sum());
    }

    double delta = b_util::diffclock(clock(), lastTime) / 1000;
    outln("Completed! s " << (3 * count) << " took " << delta << " secs\n\n\n");
	outln("testProductShapesLoop finish");
	return 0;
}


template int testAutodot<float>::operator()(int argc, char const ** args) const;
template int testAutodot<double>::operator()(int argc, char const ** args) const;
template <typename T> int testAutodot<T>::operator()(int argc, const char** args) const {
	outln("testAutodot start");
	int count = b_util::getCount(argc,args,10000);
	float exeTime;
	Matrix<T> m = Matrix<T>::sin(1000, 1000);
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
template <typename T> int testMultLoop<T>::operator()(int argc, const char** args) const {
	outln("testMultLoop start");
	Matrix<T> m1 = Matrix<T>::ones(999, 999);
	outln("m1 " << m1.syncBuffers());
	Matrix<T> m1b = m1 * 2;
	outln("m1b " << m1b.syncBuffers());
	Matrix<T> m2 = Matrix<T>::ones(999, 999) * 2;
	checkCudaErrors(cudaDeviceSynchronize());
	outln("made mats m1 " << m1.toShortString() << ", m2 " << m2.toShortString() << "\n");
	outln("m2 " << m2.syncBuffers());
	int blocks;
	int threads;
	uint n = m1.m * m1.n;
	m1.getReductionExecContext(blocks, threads, n);
	outln("blocks " << blocks << "\n");

	Matrix<T> buffer = Matrix<T>::zeros(m1.m, m2.n);

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
		m1.matrixProductL(m3_d, m1_d, m2_d, 0);
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
