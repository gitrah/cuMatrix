#include "tests.h"
#include "../util.h"
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

template int testShuffleCopyRows<float>::operator()(int argc, char const ** args) const;
template int testShuffleCopyRows<double>::operator()(int argc, char const ** args) const;
template <typename T>  int testShuffleCopyRows<T>::operator()(int argc, const char** args) const {

	Matrix<T> src = Matrix<T>::increasingRows(0, 100,100);
	Matrix<T> y = Matrix<T>::increasingRows(0, 100,1);
	outln("src " << src.syncBuffers());

	vector<uint>indices, lindices;
	Matrix<T> shuffled, leftovers;
	src.shuffle(shuffled, leftovers, (T) .5,indices);

	outln("shuffled " << shuffled.syncBuffers());
	outln("leftovers " << leftovers.syncBuffers());

	indices.clear();
	Matrix<T> training, cv, testm;
	Matrix<T> ytraining, ycv, ytestm;
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
