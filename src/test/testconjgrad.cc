/*
 * testconjgrad.cc
 *
 *  Created on: Sep 6, 2012
 *      Author: reid
 */
#include "../CuMatrix.h"
#include "../util.h"
#include "../AnomalyDetection.h"

template <typename T> int testConjGrad(int argc, char** args) {
	std::map<std::string, CuMatrix<T>*> f= CuMatrix<T>::parseOctaveDataFile("ex4data1.txt",false, true);
	std::map<std::string, CuMatrix<T>*> fw= CuMatrix<T>::parseOctaveDataFile("ex4weights.txt",false, true);


	std::cout << "found " << f.size() << " octave objects\n";

	CuMatrix<T>& x = *f["X"];
	outln("load x of " << x.m << "x" << x.n);
	CuMatrix<T>& y = *f["y"];
	outln("got y " << y.toShortString());
	CuMatrix<T>& theta1 = *fw["Theta1"];
	outln("got theta1 " << theta1.toShortString());
	CuMatrix<T>& theta2 = *fw["Theta2"];
	outln("got theta2 " << theta2.toShortString());

    int m = x.m;
    int n= x.n;
    CuMatrix<T> mus = x.featureMeans(true);

    int input_layer_size = 400; // 20x20 Input Images of Digits
    int hidden_layer_size = 25; //   25 hidden units
    int num_labels = 10; // 10 labels, from 1 to 10
    uint nfeatures = theta1.n;
    dassert(num_labels == nfeatures); // feature mismatch

    CuMatrix<T> thetas = theta1.poseAsRow() |= theta2.poseAsRow();
    theta1.unPose();
    theta2.unPose();
    T lambda = 0;
}
