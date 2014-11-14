#include "tests.h"
#include "../CuMatrix.h"
#include "../util.h"
#include "../caps.h"
#include "../Migrator.h"
#include "../MatrixExceptions.h"

template int testMultiGPUMemcpy<float>::operator()(int argc, char const ** args) const;
template int testMultiGPUMemcpy<double>::operator()(int argc, char const ** args) const;
template <typename T> int testMultiGPUMemcpy<T>::operator()(int argc, const char** args) const {

	outln("testMultiGPUMemcpy start");

	ExecCaps::findDevCaps();
	outln("found " << g_devCount << " gpus");

    checkCudaErrors(cudaSetDevice(0));
    CuMatrix<T> m50d0 = CuMatrix<T>::ones(50,50).syncBuffers();
    outln("m50d0\n" << m50d0);

    cudaPointerAttributes ptrAtts;
    checkCudaErrors(cudaPointerGetAttributes(&ptrAtts, m50d0.d_elements));
    outln("m50d0 d_elements ptr info " << b_util::sPtrAtts(ptrAtts));
    checkCudaErrors(cudaSetDevice(1));
    CuMatrix<T> m50d1 = CuMatrix<T>::sin(50,50).syncBuffers();
    checkCudaErrors(cudaPointerGetAttributes(&ptrAtts, m50d1.d_elements));
    outln("m50d1 d_elements ptr info " << b_util::sPtrAtts(ptrAtts));
    outln("m50d1\n" << m50d1);

    checkCudaErrors(cudaSetDevice(0));
    m50d1.getMgr().migrate(0, m50d1);
    outln("after migrate(0) m50d1 " << m50d1.toShortString());
    CuMatrix<T> sumd0 = m50d0 + m50d1;
    outln("sumd0\n" << sumd0.syncBuffers());

    m50d1.toDevice(1);
    outln("after move m50d1 " << m50d1.toShortString());
    m50d0.toDevice(1);
    outln("after move m50d0 " << m50d0.toShortString());
    outln("currdev " << b_util::currDevice());
    checkCudaErrors(cudaSetDevice(1));
    CuMatrix<T> sumd1 = m50d0 + m50d1;
    outln("sumd1 short\n" << sumd1.toShortString());
    outln("sumd1\n" << sumd1.syncBuffers());

    int total = b_util::getCount(argc,args,1000);
/*
	T s = 0;
	for(int i = 0; i < total; i++ ){
		CuMatrix<T> m = CuMatrix<T>::sin(500,500);
		CuMatrix<T> mc =CuMatrix<T>::cos(500,500);
		s += ((m * mc) * mc / 2.).sum();
		outln(i << "th iter; sum " << s << " usage: " << usedMemRatio() << "%");
	}

*/
    checkCudaErrors(cudaSetDevice(0));
	CuMatrix<T> col = CuMatrix<T>::sin(500,500).syncBuffers();

	util<T>::deletePtrArray(g_devCaps, g_devCount);
	util<T>::deleteDevPtrArray(gd_devCaps, g_devCount);

	return 0;
}

template int testMultiGPUMath<float>::operator()(int argc, char const ** args) const;
template int testMultiGPUMath<double>::operator()(int argc, char const ** args) const;
template <typename T> int testMultiGPUMath<T>::operator()(int argc, const char** args) const {

	int count = b_util::getCount(argc,args,100);
	float exeTime1,exeTime2,exeTime3;
	int devCount;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(devCount < 2) {
		dthrow(insufficientGPUCount());
	}

	outln("testMultiGPUMath start");
	//clock_t lastTime = clock();

	outln("found " << devCount << " gpus");
	cudaStream_t d0Stream, d1Stream;

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaStreamCreate(&d0Stream));
	checkCudaErrors(cudaSetDevice(1));
	checkCudaErrors(cudaStreamCreate(&d1Stream));
	checkCudaErrors(cudaSetDevice(0));

	// create mat pairs on two gpus and mult them in a loop

	CuTimer timerD0(d0Stream);
    CuMatrix<T> m5001d0 = CuMatrix<T>::ones(500,500);
    CuMatrix<T> m5002d0 = CuMatrix<T>::sin(500,500);

    checkCudaErrors(cudaSetDevice(1));
	CuTimer timerD1(d1Stream);
    CuMatrix<T> m5001d1 = CuMatrix<T>::ones(500,500);
    CuMatrix<T> m5002d1 = CuMatrix<T>::sin(500,500);

    outln(count << " iterations of dev 0 mult ");
    checkCudaErrors(cudaSetDevice(0));
    timerD0.start();
    T sum1;
	for(int i = 0; i < count; i++) {
		CuMatrix<T> res = m5001d0 * m5002d0;
		if(i == count -1) {
			sum1 = res.sum();
		}
	}
	exeTime1 = timerD0.stop();
	outln("for dev 0, " << count << " took " << exeTime1 );

    outln(count << " iterations of dev 1 mult ");
    checkCudaErrors(cudaSetDevice(1));
    timerD1.start();
    T sum2;
	for(int i = 0; i < count; i++) {
		CuMatrix<T> res = m5001d1 * m5002d1;
		if(i == count -1) {
			sum2 = res.sum();
		}
	}
	exeTime2 = timerD1.stop();
	outln("for dev 1, " << count << " took " << exeTime2 );

	timerD1.start();
    T sum3;
	for(int i = 0; i < count; i++) {
	    checkCudaErrors(cudaSetDevice(0));
		CuMatrix<T> res0 = m5001d0 * m5002d0;
	    checkCudaErrors(cudaSetDevice(1));
		CuMatrix<T> res1 = m5001d1 * m5002d1;
		if(i == count -1) {
			sum3 = res1.sum();
		}
	}
	exeTime3 = timerD1.stop();
	Migrator<T> migrate0(0);
	migrate0 << m5001d1 << m5002d1;
    checkCudaErrors(cudaSetDevice(0));
	CuMatrix<T> res1 = m5001d1 * m5002d1;
	outln("after migrate res1.sum() " << res1.sum());

	outln("for dev 0 + 1, " << count << " took " << exeTime3 );

	outln("sum1 " << sum1);
	outln("sum2 " << sum2);
	outln("sum3 " << sum3);

	return 0;
}



