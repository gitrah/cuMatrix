#include "tests.h"
#include "../mnist.h"
#include "../NeuralNet.h"
#include "../CuMatrix.h"
#include "../ConjugateGradient.h"
#include "../debug.h"
#include "../util.h"
#include "../Cycler.h"
#include "../Kernels.h"
#include <cuda_profiler_api.h>
#include <random>


const char* MNIST_SAMPLES_FILE = "train-images-idx3-ubyte";
const char* MNIST_LABELS_FILE = "train-labels-idx1-ubyte";
const char* MNIST_FILES[2] = {MNIST_SAMPLES_FILE,MNIST_LABELS_FILE};


template int testNeuralMnistHw<float>::operator()(int argc, const char **argv) const;
template int testNeuralMnistHw<double>::operator()(int argc, const char **argv) const;
template int testNeuralMnistHw<ulong>::operator()(int argc, const char **argv) const;
template<typename T> int testNeuralMnistHw<T>::operator()(int argc,
		const char** args) const {
	const int input_layer_size = 28*28; // 20x20 Input Images of Digits
	const int hidden_layer_size = 300; //   25 hidden units
	const int hidden_layer_size2 = 100; //   25 hidden units
	const int num_labels = 10; // 10 labels, from 1 to 10
	const int num_samples = 60000;

	bool checkGrad = false;
	T lambda = 3;
	if (checkGrad) {
		NeuralNet<T>::checkNnGradients(lambda);
	}

	int iterations = b_util::getParameter(argc, args, "its", 1000);

	int tid;


	CuMatrix<T> x =readMnistImages<T>(MNIST_SAMPLES_FILE).syncBuffers();
	outln("x " << x.toShortString());

	CuMatrix<T> y =readMnistLables<T>(MNIST_LABELS_FILE).syncBuffers();
	outln("y " << y.toShortString());

	CuMatrix<T> sT;

    srand (time(NULL));
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(1,60000);
	auto dice = std::bind ( distribution, generator );
	int num = rand() % 20;
	outln("num " << num);
	x.submatrix(sT,1,28 * 28, num, 0);
	outln("sT " << sT);
	CuMatrix<T> s = sT.reshape(28,28,0);
	T label = y.get(num,0);
	outln("random digit (should be " << label << ") " << s.syncBuffers());


	outln("x " << x);
	outln("y " << y);
	CuTimer timer;
	timer.start();

	T sumx = x.sum();
	outln("sumx " << sumx);
	//assert(util<T>::almostEquals(sumx, 2.626782601596818e05));

	T cost;
	CuMatrix<T> xBiased = x.addBiasColumn();
	assert(xBiased.biasedQ());
	CuMatrix<T> training, training2, cv, cv2;


	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	vector<int> vIndices;
	CuMatrix<T> yBiased = y.toBinaryCategoryMatrix().syncBuffers();

	xBiased.shuffle(&training, &cv, (T) .75, vIndices);
	y.shuffle(&ytraining, &ycv, (T) .75, vIndices);

	yBiased.shuffle(&ytrainingBiased, &ycvBiased, (T) .75, vIndices);

	outln("after shuffling xBiased sum %ld " << xBiased.sum());
	outln("after shuffling y sum %ld " << y.sum());

	outln("after shuffling training sum %ld " << training.sum());
	outln("after shuffling cv sum %ld " << cv.sum());

	outln("after shuffling h"
			"ytraining sum %ld " << ytraining.sum());
	outln("after shuffling ycv sum %ld " << ycv.sum());

	outln("after shuffling yBiased " << yBiased.toShortString());

	ConjugateGradient<T>::init();

	vector<reference_wrapper<CuMatrix<T>>> mats;
	mats.push_back(training);
	mats.push_back(ytrainingBiased);


	bool repeatable = true;

	CuMatrix<T> initial_Theta1 =
			repeatable ?
					CuMatrix<T>::sin(hidden_layer_size, input_layer_size).syncBuffers().addBiasColumn() :
					CuMatrix<T>::randn(hidden_layer_size, input_layer_size).syncBuffers().addBiasColumn();
	CuMatrix<T> initial_Theta2 =
			repeatable ?
					CuMatrix<T>::sin(hidden_layer_size, hidden_layer_size).syncBuffers().addBiasColumn() :
					CuMatrix<T>::randn(hidden_layer_size, hidden_layer_size).syncBuffers().addBiasColumn();
	CuMatrix<T> initial_Theta3 =
			repeatable ?
					CuMatrix<T>::sin(num_labels, hidden_layer_size).syncBuffers().addBiasColumn() :
					CuMatrix<T>::randn(num_labels, hidden_layer_size).syncBuffers().addBiasColumn();

	outln(
			"hs " << hidden_layer_size << ": " << "initial_Theta1 " << initial_Theta1.syncBuffers().toShortString());
	outln(
			"hs " << hidden_layer_size << ": " << "initial_Theta2 " << initial_Theta2.syncBuffers().toShortString());
	outln(
			"hs " << hidden_layer_size << ": " << "initial_Theta3 " << initial_Theta3.syncBuffers().toShortString());
	const CuMatrix<T>* parts3[] = { &initial_Theta1, &initial_Theta2,
			&initial_Theta3 };
	CuMatrix<T> initial_nn_params(
			(initial_Theta1.size + initial_Theta2.size
					+ initial_Theta3.size) / sizeof(T), 1, false, true);
	outln("hs " << hidden_layer_size << ": " << "initial_nn_params bef concat " << initial_nn_params.toShortString());

	CuMatrix<T>::concat(initial_nn_params, 3, parts3);

	uint2 dims[3];

	dims[0].x = initial_Theta1.m;
	dims[0].y = initial_Theta1.n;
	dims[1].x = initial_Theta2.m;
	dims[1].y = initial_Theta2.n;
	dims[2].x = initial_Theta3.m;
	dims[2].y = initial_Theta3.n;
	outln("\n\ninitial_nn_params aft concat " << initial_nn_params.toShortString());

	nnCostFtorPm<T> pmFtor3(dims, 3, training, ytrainingBiased, lambda);
	CuTimer justFmincg;
	justFmincg.start();
	outln("justFmincg.start()");

	pair<CuMatrix<T>, pair<CuMatrix<T>, int> > tup2_3 =
			ConjugateGradient<T>::fmincg(pmFtor3, initial_nn_params,
					iterations);
	outln( "back from initial_nn_params pmfmincg, took " << justFmincg.stop()/1000 << "s");
	//
	//
	CuMatrix<T> nn_parms3PmGrad = tup2_3.first;
	outln("nn_parms3PmGrad " << nn_parms3PmGrad.toShortString());
	outln("nn_parms3PmGrad.sum " << nn_parms3PmGrad.sum());

	CuMatrix<T> nTheta1Pm3(dims[0].x, dims[0].y, false, true);
	CuMatrix<T> nTheta2Pm3(dims[1].x, dims[1].y, false, true);
	CuMatrix<T> nTheta3Pm3(dims[2].x, dims[2].y, false, true);
					// mat, m, n, p, off
	nn_parms3PmGrad.unconcat(nTheta1Pm3, hidden_layer_size,
			(input_layer_size + 1), (input_layer_size + 1), 0);
	outln("nTheta1Pm3 " << nTheta1Pm3.toShortString());
	outln("nTheta1Pm3,sum " << nTheta1Pm3.sum());

	nn_parms3PmGrad.unconcat(nTheta2Pm3, hidden_layer_size, (hidden_layer_size + 1),
			(hidden_layer_size + 1), hidden_layer_size * (input_layer_size + 1));
	outln("nTheta2Pm3 " << nTheta2Pm3.toShortString());
	outln("nTheta2Pm3,sum " << nTheta2Pm3.sum());

	nn_parms3PmGrad.unconcat(nTheta3Pm3, num_labels, (hidden_layer_size + 1),
			(hidden_layer_size + 1),
			hidden_layer_size * (input_layer_size + 1 + hidden_layer_size)
					+ hidden_layer_size);
	outln("nTheta3Pm3 " << nTheta3Pm3.toShortString());
	//  outln("nTheta2Pm " << nTheta2Pm.syncBuffers());

	vector<CuMatrix<T> > thetaV;
	thetaV.push_back(nTheta1Pm3);
	thetaV.push_back(nTheta2Pm3);
	thetaV.push_back(nTheta3Pm3);

	NNPrediction<T> pred3 = NeuralNet<T>::forward(thetaV, training,
			ytrainingBiased, lambda);
	CuMatrix<T> p1pm3 = pred3.hTheta;
	//    outln("p1Pm " << p1Pm.syncBuffers());
	outln("p1pm3 sum " << p1pm3.sum());

	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1Pm3 = p1pm3.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	outln("forwarding");
	for(auto &im : thetaV) outln("thetaV." << im.toShortString());
	for(auto &im : {cv,
			ycvBiased} )outln("cv/y " << im.toShortString());
	NNPrediction<T> pred3cv = NeuralNet<T>::forward(thetaV, cv,
			ycvBiased, lambda);
	CuMatrix<T> p1Pmcv3 = pred3cv.hTheta;
	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1Pmcv3 = p1Pmcv3.toMaxColumnIndexVector() + 1;
	outln("h1Pmcv3 " << h1Pmcv3.toShortString());

	CuMatrix<T> p1cv = NeuralNet<T>::predictCg(nTheta1Pm3, nTheta2Pm3, cv);
	outln("p1cv " << p1cv.toShortString());
	//b_util::allDevices([]() { setCurrGpuDebugFlags(debugCons | debugDestr,true,false,0 );});

	T resPm3 = ytraining.accuracy(h1Pm3);
	checkCudaError(cudaGetLastError());
	outln("Pm3 training accuracy : " << ( resPm3 * 100));

	T rescvPm3 = ycv.accuracy(h1Pmcv3);
	outln("cv accuracy : " << ( rescvPm3 * 100));
	checkCudaError(cudaGetLastError());


	return 0;
}

