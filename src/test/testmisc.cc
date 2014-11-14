#include "tests.h"
#include "../util.h"
#include "../caps.h"
#include "../Validation.h"

template int testRandSequence<float>::operator()(int argc, char const ** args) const;
template int testRandSequence<double>::operator()(int argc, char const ** args) const;
template <typename T>  int testRandSequence<T>::operator()(int argc, const char** args) const {

	vector<uint> l1;
	b_util::randSequence(l1, 30,5);
	vector<uint> l2;
	b_util::randSequence(l2, 40,100);

	outln("l1\n" << b_util::pvec(l1));
	outln("l2\n" << b_util::pvec(l2));

	return 0;
}

template int testRandAsFnOfSize<float>::operator()(int argc, char const ** args) const;
template int testRandAsFnOfSize<double>::operator()(int argc, char const ** args) const;
template <typename T>  int testRandAsFnOfSize<T>::operator()(int argc, const char** args) const {

	int size[] = {10, 100, 1000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 20000000};
	uint count = b_util::getCount(argc,args,5);

    CuTimer timer;
    float exeTime;

	for(int i = 0; i < 12; i++) {
	    timer.start();
		for(int j = 0; j < count; j++) {
			CuMatrix<T> mat = CuMatrix<T>::randn(1, size[i], (T) 10.);
		}
		exeTime = timer.stop();
		outln(count << " randn fills of size " << size[i] << " took " << exeTime <<  "s total, or " << 1./(exeTime/size[i]) << "elems/s");
	}

	return 0;
}

template int testMontePi<float>::operator()(int argc, char const ** args) const;
template int testMontePi<double>::operator()(int argc, char const ** args) const;
template <typename T>  int testMontePi<T>::operator()(int argc, const char** args) const {

	int size[] = {10, 100, 1000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000};
	uint count = b_util::getCount(argc,args,5);

    CuTimer timer;
    float exeTime;
    T edgeOfSquareLength = 2;

	for(int i = 0; i < 12; i++) {
	    timer.start();
		for(int j = 0; j < count; j++) {
			CuMatrix<T> xs = CuMatrix<T>::randn(1, size[i], edgeOfSquareLength);
			CuMatrix<T> ys = CuMatrix<T>::randn(1, size[i], edgeOfSquareLength);
		}
		exeTime = timer.stop();
		outln(count << " randn fills of size " << size[i] << " took " << exeTime <<  "s total, or " << 1./(exeTime/size[i]) << "elems/s");
	}

	return 0;
}




template int testRandomizingCopyRows<float>::operator()(int argc, char const ** args) const;
template int testRandomizingCopyRows<double>::operator()(int argc, char const ** args) const;
template <typename T>  int testRandomizingCopyRows<T>::operator()(int argc, const char** args) const {

	CuMatrix<T> src = CuMatrix<T>::increasingRows(0, 100,100);
	CuMatrix<T> y = CuMatrix<T>::increasingRows(0, 100,1);
	outln("src " << src.syncBuffers());

	vector<uint>indices, lindices;
	CuMatrix<T> shuffled, leftovers;
	src.shuffle(shuffled, leftovers, (T) .5,indices);

	outln("shuffled " << shuffled.syncBuffers());
	outln("leftovers " << leftovers.syncBuffers());

	indices.clear();
	CuMatrix<T> training, cv, testm;
	CuMatrix<T> ytraining, ycv, ytestm;
	Validation<T>::toValidationSets(training,cv,testm,src,.7,.15,indices, lindices);
	Validation<T>::toValidationSets(ytraining,ycv,ytestm,y,.7,.15,indices, lindices);
	outln("training\n" << training.syncBuffers());
	outln("cv\n" << cv.syncBuffers());
	outln("testm\n" << testm.syncBuffers());
	outln("yyraining\n" << ytraining.syncBuffers());
	outln("ycv\n" << ycv.syncBuffers());
	outln("ytestm\n" << ytestm.syncBuffers());
	return 0;
}

#include "tests.cc"
