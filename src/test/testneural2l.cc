#include "../NeuralNet.h"
#include "../CuMatrix.h"
#include "../ConjugateGradient.h"
#include "../debug.h"
#include "../util.h"
#include "../Cycler.h"
#include "../Kernels.h"
#include "tests.h"
#include <cuda_profiler_api.h>
#include <map>

template int testNeural2l<float>::operator()(int argc,
		const char ** args) const;
template int testNeural2l<double>::operator()(int argc,
		const char ** args) const;
template int testNeural2l<ulong>::operator()(int argc,
		const char ** args) const;
template<typename T> int testNeural2l<T>::operator()(int argc,
		const char** args) const {

	bool checkGrad = true;
	T lambda = 3;
	if (checkGrad) {
		//NeuralNet<T>::checkNnGradients(0);
		NeuralNet<T>::checkNnGradients(lambda);
	}

	CuMatrix<T> onesies = CuMatrix<T>::ones(513,513);
	outln(onesies.syncBuffers());

	assert(onesies.biasedQ());

	int iterations = b_util::getParameter(argc, args, "its", 50);

	map<string, CuMatrix<T>*> f = CuMatrix<T>::parseOctaveDataFile(SAMPLES_FILE,
			false, true);
	map<string, CuMatrix<T>*> fw = CuMatrix<T>::parseOctaveDataFile(
			WEIGHTS_FILE, false, true);
	if (!f.size()) {
		outln("no " << SAMPLES_FILE << "; exiting");
		return -1;
	}
	if (!fw.size()) {
		outln("no " << WEIGHTS_FILE << "; exiting");
		return -1;
	}
	CuTimer timer;
	timer.start();
//	int ret = testNeuralKPtr(iterations, f,fw);
	int ret = testNeuralH2N(iterations, f, fw, false);
	outln("testNeuralKPtr(" <<iterations <<") took " << timer.stop() << "ms");
	util<CuMatrix<T> >::deletePtrMap(f);
	util<CuMatrix<T> >::deletePtrMap(fw);
	return ret;
}

template int testNeural2lCsv<float>::operator()(int argc,
		const char ** args) const;
template int testNeural2lCsv<double>::operator()(int argc,
		const char ** args) const;
template<typename T> int testNeural2lCsv<T>::operator()(int argc,
		const char** args) const {

	//int iterations = b_util::getParameter(argc, args, "its", 50);

	map<string, CuMatrix<T>*> f = CuMatrix<T>::parseOctaveDataFile(
			REDWINE_CSV_FILE, false, true);
	if (!f.size()) {
		outln("no " << REDWINE_CSV_FILE << "; exiting");
		return -1;
	}
	//int ret = testNeuralKPtr(iterations, f,fw);
	util<CuMatrix<T> >::deletePtrMap(f);
	return 0;
}

template int testNeural2lAdult<float>::operator()(int argc,
		const char ** args) const;
template int testNeural2lAdult<double>::operator()(int argc,
		const char ** args) const;
template int testNeural2lAdult<ulong>::operator()(int argc,
		const char ** args) const;
template<typename T> int testNeural2lAdult<T>::operator()(int argc,
		const char** args) const {

	bool checkGrad = true;
	T lambda = 3;
	if (checkGrad) {
		//NeuralNet<T>::checkNnGradients(0);
		NeuralNet<T>::checkNnGradients(lambda);
	}

	int iterations = b_util::getParameter(argc, args, "its", 50);

	map<string, CuMatrix<T>*> f = CuMatrix<T>::parseOctaveDataFile(ADULT_FILE,
			false, true);
	if (!f.size()) {
		outln("no " << ADULT_FILE << "; exiting");
		return -1;
	}
	CuTimer timer;
	timer.start();

	CuMatrix<T>* x = f["x"];
	outln("x "<< x->toShortString());
	x->syncBuffers();
	assert(x->lastMod == mod_synced);
	T sumx = x->sum();
	outln("sumx " << sumx);
	//assert(util<T>::almostEquals(sumx, 2.626782601596818e05));
	CuMatrix<T> xT = x->transpose();
	T sumxT = xT.sum();
	outln("sumxT " << sumxT);
	assert(util<T>::almostEquals(sumx, sumxT));
	//outln("0x " << x->sum());
	CuMatrix<T>* y = f["y"];
	y->syncBuffers();
	outln("y " << y->toShortString());

	CuMatrix<T> mus = x->featureMeans(true);

	outln("yBiasing");
	CuMatrix<T> yBiased = y->toBinaryCategoryMatrix().syncBuffers();
	outln("yBiased " << yBiased.toShortString());
	checkCudaErrors(cudaGetLastError());

	int input_layer_size = 14; // 20x20 Input Images of Digits
	int hidden_layer_size = 7; //   25 hidden units
	int num_labels = 1; // 10 labels, from 1 to 10

	outln("adding bias");
	CuMatrix<T> xBiased = x->addBiasColumn();
	outln(
			"added; x " << x->toShortString() << " --> " << xBiased.toShortString());

	T sigArry[] = { (T) 1, (T) -.5, (T) 0, (T) .5, (T) 1 };
	CuMatrix<T> sigTest1(sigArry, 1, 5, true);
	CuMatrix<T> aSigTest = sigTest1.syncBuffers().sigmoidGradient();
	outln("aSigTest " << aSigTest.syncBuffers());
	checkCudaError(cudaGetLastError());

	CuMatrix<T> training, training2, cv, cv2;
	vector<int> vIndices;

	T xbs = xBiased.sum();
	outln("xBiased.sum " << xbs);
	xBiased.shuffle(&training, &cv, (T) .75, vIndices);
	outln("shuffled ");

	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	y->shuffle(&ytraining, &ycv, (T) .75, vIndices);
	yBiased.shuffle(&ytrainingBiased, &ycvBiased, (T) .75, vIndices);

	nnCostFtor<T> costFn(input_layer_size, hidden_layer_size, num_labels,
			training, ytrainingBiased, lambda);
	outln("explicit costFn(thetas)");
//	outln("cost " << cost);
	costFn.lambda = 1;
//	costFn(grad,cost, thetas);
	checkCudaError(cudaGetLastError());
	//("cost lambada = 1 " << cost);

	CuMatrix<T> initial_Theta1 = CuMatrix<T>::sin(hidden_layer_size,
			input_layer_size).syncBuffers().addBiasColumn();
	CuMatrix<T> initial_Theta2 =
			CuMatrix<T>::sin(num_labels, hidden_layer_size).syncBuffers().addBiasColumn();

	outln("initial_Theta1 " << initial_Theta1.syncBuffers());
	outln("initial_Theta2 " << initial_Theta2.syncBuffers());

	const CuMatrix<T>* parts[] = { &initial_Theta1, &initial_Theta2 };

	CuMatrix<T> initial_nn_params(
			(initial_Theta1.size + initial_Theta2.size) / sizeof(T), 1, false,
			true);
	outln("initial_nn_params bef concat " << initial_nn_params.toShortString());
	flprintf("initial_nn_params %.20g\n", initial_nn_params.sum());
	CuMatrix<T>::concat(initial_nn_params, 2, parts);
	outln("initial_nn_params aft concat " << initial_nn_params.toShortString());
	CuMatrix<T> grad(initial_nn_params.size / sizeof(T), 1, false, true);
	outln("created grad " << grad.toShortString());
	outln("created2 grad " << grad.toShortString());

	//CuMatrix<T> initial_nn_params = CuMatrix<T>::zeros(1,(initial_Theta1.size + initial_Theta2.size)/sizeof(T));

	nnCostFtor<T> ftor(input_layer_size, hidden_layer_size, num_labels,
			training, ytrainingBiased, lambda);

	ConjugateGradient<T>::init();
	ftor.verbose = true;
	outln("post init last err " << b_util::lastErrStr());

	CuTimer justFmincg;
	justFmincg.start();
	//  cherr(cudaProfilerStart());
	pair<CuMatrix<T>, pair<CuMatrix<T>, int> > tup2 =
			ConjugateGradient<T>::fmincg(ftor, initial_nn_params, iterations);
	outln("back from fmincg, took " << justFmincg.stop()/1000 << "s");
	checkCudaError(cudaGetLastError());

	CuMatrix<T> nn_parms = tup2.first;

	CuMatrix<T> nTheta1;
	CuMatrix<T> nTheta2;
	nn_parms.unconcat(nTheta1, hidden_layer_size, (input_layer_size + 1),
			(input_layer_size + 1), 0);
	nn_parms.unconcat(nTheta2, num_labels, (hidden_layer_size + 1),
			(hidden_layer_size + 1),
			hidden_layer_size * (input_layer_size + 1));
	outln("nTheta1,sum " << nTheta1.sum());
	outln("nTheta2,sum " << nTheta2.sum());
	checkCudaError(cudaGetLastError());
	//  outln("nTheta2 " << nTheta2.syncBuffers());

	CuMatrix<T> p1 = NeuralNet<T>::predictCg(nTheta1, nTheta2, training);
//    outln("p1 " << p1.syncBuffers());
	outln("p1 sum " << p1.sum());

	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1 = p1.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	CuMatrix<T> p1cv = NeuralNet<T>::predictCg(nTheta1, nTheta2, cv);
	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1cv = p1cv.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	outln("h1 " << h1.toShortString());
	outln("h1 s " << h1.sum());
	outln("ytraining " << ytraining.toShortString());
	outln("ytraining s " << ytraining.sum());

	T res = ytraining.accuracy(h1);
	checkCudaError(cudaGetLastError());
	outln("training accuracy : " << ( res * 100));

	T rescv = ycv.accuracy(h1cv);
	checkCudaError(cudaGetLastError());
	outln("cv accuracy : " << ( rescv * 100));

	T cost;
	nnCostFtor<T> costFnTraining(input_layer_size, hidden_layer_size,
			num_labels, training, ytrainingBiased, lambda);
	costFnTraining(grad, cost, nn_parms);
	outln("cost on training " << cost);
	nnCostFtor<T> costFnCv(input_layer_size, hidden_layer_size, num_labels, cv,
			ycvBiased, lambda);
	costFnCv(grad, cost, nn_parms);
	outln("cost on cv " << cost);
//    cherr(cudaProfilerStop());

	util<CuMatrix<T> >::deletePtrMap(f);
	return 0;
}

template<typename T> int testNeural5Pm(const CuMatrix<T>& training,
		const CuMatrix<T>& cv, const CuMatrix<T>& ytraining,
		const CuMatrix<T>& ytrainingBiased, const CuMatrix<T>& ycv);

template<typename T> int testNeuralKPtrPm(int iterations, CuMatrix<T> *theta1,
		CuMatrix<T> * theta2, T thetasSum, T initCost, CuMatrix<T>& xBiased,
		CuMatrix<T>& yBiased, CuMatrix<T>& training, CuMatrix<T>& cv,
		CuMatrix<T>& ytraining, CuMatrix<T>& ycv, CuMatrix<T>& ycvBiased,
		CuMatrix<T>& ytrainingBiased, CuMatrix<T>& p1, CuMatrix<T>& h1) {
	//setCurrGpuDebugFlags( debugVerbose,true,false);

	T lambda = 3;

	// test vector-based pack/unpack
	vector<CuMatrix<T>> vthta;
	vthta.push_back(*theta1);
	vthta.push_back(*theta2);
	PackedMat<T> pmv = PackedMat<T>::pack(vthta);
	pmv.owner = false;
	outln("pmv dims " << pmv.dumpDims());
	outln("pmv nn_params " << pmv.nn_params->syncBuffers());
	T packedThetasSum = pmv.nn_params->sum();
	outln("pmv nn_params sum " << packedThetasSum);
	assert(util<T>::almostEquals(thetasSum, packedThetasSum, 0.01));

	vector<CuMatrix<T>> outies;

	int offset = 0;
	for (int i = 0; i < pmv.layers; i++) {
		CuMatrix<T> layer;
		pmv.nn_params->unconcat(layer, pmv.dims[i].x, pmv.dims[i].y,
				pmv.dims[i].y, offset, false);
		outln("outies adding layer " << i << ": " << layer.toShortString());
		outies.push_back(layer);
		offset += pmv.dims[i].x * pmv.dims[i].y;
	}

	checkCudaErrors(cudaGetLastError());

	T cost = 0;
	lambda = 1;
	int input_layer_size = 400; // 20x20 Input Images of Digits
	int hidden_layer_size = 25; //   25 hidden units
	int num_labels = 10; // 10 labels, from 1 to 10

	vector<CuMatrix<T> > thetaV;
	thetaV.push_back(*theta1);
	thetaV.push_back(*theta2);
	CuMatrix<T> fbGrad;
	T fbCost;
	NeuralNet<T>::forwardAndBack(fbGrad, fbCost, training, ytrainingBiased,
			thetaV, lambda);
	outln(
			"pm lambda " << lambda << ", fbGrad " << fbGrad.syncBuffers() << ", fbCost " << fbCost << "\n\n");
	assert(util<T>::almostEquals(initCost, fbCost, 1e-8));

	/*

	 CuMatrix<T> initial_Theta1 = CuMatrix<T>::sin(hidden_layer_size, input_layer_size).syncBuffers().addBiasColumn();
	 CuMatrix<T> initial_Theta2 = CuMatrix<T>::sin(num_labels, hidden_layer_size).syncBuffers().addBiasColumn();


	 */

	outln("with packed layers, fbCost " << fbCost);

	CuMatrix<T> theta_1b = CuMatrix<T>::sin(hidden_layer_size,
			hidden_layer_size).syncBuffers().addBiasColumn();
	const CuMatrix<T>* parts3[] = { theta1, &theta_1b, theta2 };

	thetaV.clear();
	thetaV.push_back(*theta1);
	CuMatrix<T> theta_1c = CuMatrix<T>::sin(hidden_layer_size,
			hidden_layer_size).syncBuffers().addBiasColumn();
	CuMatrix<T> theta_1d = CuMatrix<T>::sin(hidden_layer_size,
			hidden_layer_size).syncBuffers().addBiasColumn();
	thetaV.push_back(theta_1c);
	thetaV.push_back(theta_1d);
	thetaV.push_back(*theta2);
	NeuralNet<T>::forwardAndBack(fbGrad, fbCost, xBiased, yBiased, thetaV,
			lambda);
	outln("with extra layers, fbCost " << fbCost);

	lambda = 1;

	//util<T>::timeReps(&CuMatrix<T>::sigmoidGradient,"sigmoidGradient", &sigTest2, 10000);

	//CuMatrix<T> initial_Theta1 = CuMatrix<T>::randn(hidden_layer_size, input_layer_size).addBiasColumn();
	//CuMatrix<T> initial_Theta2 = CuMatrix<T>::randn(num_labels, hidden_layer_size).addBiasColumn();

	CuMatrix<T> initial_Theta1 = CuMatrix<T>::sin(hidden_layer_size,
			input_layer_size).syncBuffers().addBiasColumn();
	CuMatrix<T> initial_Theta2 =
			CuMatrix<T>::sin(num_labels, hidden_layer_size).syncBuffers().addBiasColumn();
	outln("initial_Theta1 " << initial_Theta1.syncBuffers().toShortString());
	outln("initial_Theta2 " << initial_Theta2.syncBuffers().toShortString());
	const CuMatrix<T>* parts[] = { &initial_Theta1, &initial_Theta2 };

	CuMatrix<T> initial_nn_params(
			(initial_Theta1.size + initial_Theta2.size) / sizeof(T), 1, false,
			true);
	outln("initial_nn_params bef concat " << initial_nn_params.toShortString());
	flprintf("initial_nn_params %.20g\n", initial_nn_params.sum());
	CuMatrix<T>::concat(initial_nn_params, 2, parts);
	outln("initial_nn_params aft concat " << initial_nn_params.toShortString());

	//CuMatrix<T> initial_nn_params = CuMatrix<T>::zeros(1,(initial_Theta1.size + initial_Theta2.size)/sizeof(T));

	outln("ycv " << ycv.syncBuffers().toShortString());
	testNeural5Pm(training, cv, ytraining, ytrainingBiased, ycv);
	//
	//
	//		conjugate gradient
	/////////////////////////////////
	//
	//
	//
	const CuMatrix<T>* parts2[] = { theta1, theta2 };
	CuMatrix<T> nn_params_2((theta1->size + theta2->size) / sizeof(T), 1, false,
			true);
	CuMatrix<T>::concat(nn_params_2, 2, parts2);
	nnCostFtor<T> ftor(input_layer_size, hidden_layer_size, num_labels,
			training, ytrainingBiased, lambda);

	uint2 dims[2];
	dims[0].x = theta1->m;
	dims[0].y = theta1->n;
	dims[1].x = theta2->m;
	dims[1].y = theta2->n;
	nnCostFtorPm<T> pmFtor(dims, 2, training, ytrainingBiased, lambda);

	thetaV.clear();
	thetaV.push_back(*theta1);
	thetaV.push_back(theta_1b);
	thetaV.push_back(*theta2);

	ConjugateGradient<T>::init();
	ftor.verbose = true;
	outln("post init last err " << b_util::lastErrStr());

	CuTimer justFmincg;
	justFmincg.start();

	//
	//	 ConjugateGradient<T>::fmincg
	//
	pair<CuMatrix<T>, pair<CuMatrix<T>, int> > tup2_2 =
			ConjugateGradient<T>::fmincg(pmFtor, nn_params_2, iterations);
	outln(
			"back from nn_params_2 pmfmincg, took " << justFmincg.stop()/1000 << "s");

	outln("back from pmfmincg, nn_params_2 " <<nn_params_2.toShortString());
	CuMatrix<T> nn_parmsPmGrad = tup2_2.first;
	outln("back from pmfmincg, nn_parmsPmGrad " <<nn_params_2.toShortString());

	CuMatrix<T> nTheta1Pm;
	CuMatrix<T> nTheta2Pm;
	nn_parmsPmGrad.unconcat(nTheta1Pm, hidden_layer_size,
			(input_layer_size + 1), (input_layer_size + 1), 0);
	nn_parmsPmGrad.unconcat(nTheta2Pm, num_labels, (hidden_layer_size + 1),
			(hidden_layer_size + 1),
			hidden_layer_size * (input_layer_size + 1));
	checkCudaError(cudaGetLastError());
	outln("nTheta1Pm,sum " << nTheta1Pm.sum());
	outln("nTheta2Pm,sum " << nTheta2Pm.sum());
//  outln("nTheta2Pm " << nTheta2Pm.syncBuffers());

	CuMatrix<T> p1Pm = NeuralNet<T>::predictCg(nTheta1Pm, nTheta2Pm, training);
//    outln("p1Pm " << p1Pm.syncBuffers());
	outln("p1Pm sum " << p1Pm.sum());

	outln("p1Pm.sumSqrDiff(p1) " << p1Pm.sumSqrDiff(p1));
	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1Pm = p1Pm.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	CuMatrix<T> p1Pmcv = NeuralNet<T>::predictCg(nTheta1Pm, nTheta2Pm, cv);
	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1Pmcv = p1Pmcv.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	outln("h1Pm " << h1Pm.toShortString());
	outln("h1Pm s " << h1Pm.sum());

	outln("h1Pm.sumSqrDiff(h1) " << h1Pm.sumSqrDiff(h1));

	T resPm = ytraining.accuracy(h1Pm);
	checkCudaError(cudaGetLastError());
	outln("Pm training accuracy : " << ( resPm * 100));

	T rescvPm = ycv.accuracy(h1Pmcv);
	checkCudaError(cudaGetLastError());
	outln("cv Pmaccuracy : " << ( rescvPm * 100));

	CuMatrix<T> grad;
	nnCostFtor<T> costFnTrainingPm(input_layer_size, hidden_layer_size,
			num_labels, training, ytrainingBiased, lambda);
	costFnTrainingPm(grad, cost, nn_parmsPmGrad);
	outln("cost on Pmtraining " << cost);
	nnCostFtor<T> costFnCv(input_layer_size, hidden_layer_size, num_labels, cv,
			ycvBiased, lambda);
	costFnCv(grad, cost, nn_parmsPmGrad);
	outln("cost on  Pmcv " << cost);

//    cherr(cudaProfilerStop());

	CuMatrix<T> nn_params_3(
			(theta1->size + theta_1b.size + theta2->size) / sizeof(T), 1, false,
			true);
	outln("\n\nnn_params_3 bef concat " << nn_params_3.toShortString());
	//flprintf("nn_params_2 %.20g\n", nn_params_2.sum());
	CuMatrix<T>::concat(nn_params_3, 3, parts3);
	//ok if(1==1) return 0;
	dims[0].x = theta1->m;
	dims[0].y = theta1->n;
	dims[1].x = theta_1b.m;
	dims[1].y = theta_1b.n;
	dims[2].x = theta2->m;
	dims[2].y = theta2->n;
	outln("\n\nnn_params_3 aft concat " << nn_params_3.toShortString());
	nnCostFtorPm<T> pmFtor3(dims, 3, training, ytrainingBiased, lambda);

	//
	//
	justFmincg.start();

	pair<CuMatrix<T>, pair<CuMatrix<T>, int> > tup2_3 =
			ConjugateGradient<T>::fmincg(pmFtor3, nn_params_3, iterations);
	outln(
			"back from nn_params_3 pmfmincg, took " << justFmincg.stop()/1000 << "s");

	//
	//
	CuMatrix<T> nn_parms3PmGrad = tup2_3.first;
	outln("nn_parms3PmGrad " << nn_parms3PmGrad.toShortString());
	outln("nn_parms3PmGrad.sum " << nn_parms3PmGrad.sum());
	/*
	 * 										rows	cols 	pitch		offset
	 CuMatrix<T> sm_Theta1 = CuMatrix<T>::sin(12, input_layer_size).syncBuffers().addBiasColumn();

	 CuMatrix<T> sm_Theta1b = CuMatrix<T>::sin(12, 12).syncBuffers().addBiasColumn();
	 CuMatrix<T> sm_Theta2 = CuMatrix<T>::sin(num_labels, 12).syncBuffers().addBiasColumn();

	 CuMatrix<T> nn_params_3((sm_Theta1.size + sm_Theta1b.size + sm_Theta2.size)/sizeof(T),1,false,true);
	 outln("\n\nnn_params_3 bef concat " << nn_params_3.toShortString());
	 //flprintf("nn_params_2 %.20g\n", nn_params_2.sum());
	 CuMatrix<T>::concat(nn_params_3, 3, parts3);
	 //ok if(1==1) return 0;
	 dims[0].x = sm_Theta1.m;dims[1].y =sm_Theta1.n;
	 dims[1].x = sm_Theta1b.m;dims[1].y =sm_Theta1b.n;
	 dims[2].x = sm_Theta2.m;dims[1].y =sm_Theta2.n;
	 outln("\n\nnn_params_3 aft concat " << nn_params_3.toShortString());
	 nnCostFtorPm<T> pmFtor3(dims,3, training, ytrainingBiased, lambda);

	 */
	CuMatrix<T> nTheta1Pm3;
	CuMatrix<T> nTheta1bPm3;
	CuMatrix<T> nTheta2Pm3;
	nn_parms3PmGrad.unconcat(nTheta1Pm3, hidden_layer_size,
			(input_layer_size + 1), (input_layer_size + 1), 0);
	outln("nTheta1Pm3 " << nTheta1Pm3.toShortString());
	outln("nTheta1Pm3,sum " << nTheta1Pm3.sum());

	nn_parms3PmGrad.unconcat(nTheta1bPm3, hidden_layer_size,
			(hidden_layer_size + 1), (hidden_layer_size + 1),
			(hidden_layer_size + 1),
			hidden_layer_size * (input_layer_size + 1));
	outln("nTheta1bPm3 " << nTheta1bPm3.toShortString());
	outln("nTheta1bPm3,sum " << nTheta1bPm3.sum());

	nn_parms3PmGrad.unconcat(nTheta2Pm3, num_labels, (hidden_layer_size + 1),
			(hidden_layer_size + 1),
			hidden_layer_size * (input_layer_size + hidden_layer_size + 2));
	outln("nTheta2Pm3 " << nTheta2Pm3.toShortString());
	outln("nTheta2Pm3,sum " << nTheta2Pm3.sum());
	checkCudaError(cudaGetLastError());
	//  outln("nTheta2Pm " << nTheta2Pm.syncBuffers());

	thetaV.clear();
	thetaV.push_back(nTheta1Pm3);
	thetaV.push_back(nTheta1bPm3);
	thetaV.push_back(nTheta2Pm3);

	NNPrediction<T> pred3 = NeuralNet<T>::forward(thetaV, training,
			ytrainingBiased, lambda);
	CuMatrix<T> p1pm3 = pred3.hTheta;
//    outln("p1Pm " << p1Pm.syncBuffers());
	outln("p1pm3 sum " << p1pm3.sum());

	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1Pm3 = p1pm3.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	NNPrediction<T> pred3cv = NeuralNet<T>::forward(thetaV, cv, ycvBiased,
			lambda);
	CuMatrix<T> p1Pmcv3 = pred3cv.hTheta;
	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1Pmcv3 = p1Pmcv3.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	outln("h1Pmcv3 " << h1Pmcv3.toShortString());
	outln("h1Pmcv3 s " << h1Pmcv3.sum());

	T resPm3 = ytraining.accuracy(h1Pm3);
	checkCudaError(cudaGetLastError());
	outln("Pm3 training accuracy : " << ( resPm3 * 100));

	T rescvPm3 = ycv.accuracy(h1Pmcv3);
	checkCudaError(cudaGetLastError());
	outln("cv P3 maccuracy : " << ( rescvPm3 * 100));

	return 0;

}
template<typename T> int testNeuralH1N(int iterations,
		map<string, CuMatrix<T>*>& f, map<string, CuMatrix<T>*>& fw) {
	T lambda = 3;

	CuMatrix<T>* x = f["X"];
	outln("x "<< x->toShortString());
	x->syncBuffers();
	assert(x->lastMod == mod_synced);
	T sumx = x->sum();
	outln("sumx " << sumx);
	assert(util<T>::almostEquals(sumx, 2.626782601596818e05));

	CuMatrix<T>* y = f["y"];
	y->syncBuffers();
	outln("y " << y->toShortString());

	int input_layer_size = 400; // 20x20 Input Images of Digits
	int hidden_layer_size_s = 1; //
	int hidden_layer_size_e = 10; //
	int num_labels = 10; // 10 labels, from 1 to 10
	T cost;
	CuMatrix<T> xBiased = x->addBiasColumn();
	CuMatrix<T> training, training2, cv, cv2;
	vector<int> vIndices;
	xBiased.shuffle(&training, &cv, (T) .75, vIndices);
	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	y->shuffle(&ytraining, &ycv, (T) .75, vIndices);
	CuMatrix<T> yBiased = y->toBinaryCategoryMatrix().syncBuffers();
	yBiased.shuffle(&ytrainingBiased, &ycvBiased, (T) .75, vIndices);

	ConjugateGradient<T>::init();

	for (int hiddenSize = hidden_layer_size_s; hiddenSize < hidden_layer_size_e;
			hiddenSize++) {
		CuMatrix<T> initial_Theta1 = CuMatrix<T>::sin(hiddenSize,
				input_layer_size).syncBuffers().addBiasColumn();
		CuMatrix<T> initial_Theta2 =
				CuMatrix<T>::sin(num_labels, hiddenSize).syncBuffers().addBiasColumn();
		outln(
				"hs " << hiddenSize << ": " << "initial_Theta1 " << initial_Theta1.syncBuffers().toShortString());
		outln(
				"hs " << hiddenSize << ": " << "initial_Theta2 " << initial_Theta2.syncBuffers().toShortString());
		const CuMatrix<T>* parts[] = { &initial_Theta1, &initial_Theta2 };
		CuMatrix<T> initial_nn_params(
				(initial_Theta1.size + initial_Theta2.size) / sizeof(T), 1,
				false, true);
		outln(
				"hs " << hiddenSize << ": " << "initial_nn_params bef concat " << initial_nn_params.toShortString());
		CuMatrix<T>::concat(initial_nn_params, 2, parts);
		outln(
				"initial_nn_params aft concat " << initial_nn_params.syncBuffers());
		flprintf("initial_nn_params sum %.20g\n", initial_nn_params.sum());

		nnCostFtor<T> costFn(input_layer_size, hiddenSize, num_labels, training,
				ytrainingBiased, lambda);
		CuMatrix<T> grad(initial_nn_params.size / sizeof(T), 1, false, true);
		outln(
				"hs " << hiddenSize << ": " <<"created grad " << grad.toShortString());
		costFn(grad, cost, initial_nn_params);
		checkCudaError(cudaGetLastError());
		outln("hs " << hiddenSize << ": " <<"cost " << cost);
		nnCostFtor<T> ftor(input_layer_size, hiddenSize, num_labels, training,
				ytrainingBiased, lambda);

		CuMatrix<T> initGrad;
		T initCost;
		ftor(initGrad, initCost, initial_nn_params);
		outln("initGrad " << initGrad.syncBuffers());
		outln("initCost (lambda = " << lambda << "):  "<< initCost);

		ftor.verbose = true;
		outln("post init last err " << b_util::lastErrStr());

		CuTimer justFmincg;
		justFmincg.start();
		//  cherr(cudaProfilerStart());
		pair<CuMatrix<T>, pair<CuMatrix<T>, int> > tup2 =
				ConjugateGradient<T>::fmincg(ftor, initial_nn_params,
						iterations);
		outln("back from fmincg, took " << justFmincg.stop()/1000 << "s");
		checkCudaError(cudaGetLastError());
		CuMatrix<T> nn_parms = tup2.first;

		CuMatrix<T> nTheta1;
		CuMatrix<T> nTheta2;
		// mat, m, n, p, off
		nn_parms.unconcat(nTheta1, hiddenSize, (input_layer_size + 1),
				(input_layer_size + 1), 0);
		nn_parms.unconcat(nTheta2, num_labels, (hiddenSize + 1),
				(hiddenSize + 1), hiddenSize * (input_layer_size + 1));
		checkCudaError(cudaGetLastError());
		//  outln("nTheta2 " << nTheta2.syncBuffers());

		CuMatrix<T> p1 = NeuralNet<T>::predictCg(nTheta1, nTheta2, training);
		//    outln("p1 " << p1.syncBuffers());
		outln("hs " << hiddenSize << ": " <<"p1 sum " << p1.sum());

		checkCudaError(cudaGetLastError());
		CuMatrix<T> h1 = p1.toMaxColumnIndexVector() + 1;
		checkCudaError(cudaGetLastError());
		CuMatrix<T> p1cv = NeuralNet<T>::predictCg(nTheta1, nTheta2, cv);
		checkCudaError(cudaGetLastError());
		CuMatrix<T> h1cv = p1cv.toMaxColumnIndexVector() + 1;
		checkCudaError(cudaGetLastError());
		outln("hs " << hiddenSize << ": " <<"h1 " << h1.toShortString());
		outln("hs " << hiddenSize << ": " <<"h1 s " << h1.sum());
		outln(
				"hs " << hiddenSize << ": " <<"ytraining " << ytraining.toShortString());
		outln("hs " << hiddenSize << ": " <<"ytraining s " << ytraining.sum());
		T res = ytraining.accuracy(h1);
		checkCudaError(cudaGetLastError());
		outln(
				"hs " << hiddenSize << ": " <<"training accuracy : " << ( res * 100));

		T rescv = ycv.accuracy(h1cv);
		checkCudaError(cudaGetLastError());
		outln("hs " << hiddenSize << ": " <<"cv accuracy : " << ( rescv * 100));
		outln("cvacc " << hiddenSize << " " << (rescv * 100) << ";");
	}
	return 0;
}

// H2N - 2 hidden / n repetitions
template<typename T> int testNeuralH2N(int iterations,
		map<string, CuMatrix<T>*>& f, map<string, CuMatrix<T>*>& fw,
		bool repeatable) {
	T lambda = 3;

	CuMatrix<T>* x = f["X"];
	outln("x "<< x->toShortString());
	x->syncBuffers();
	assert(x->lastMod == mod_synced);
	T sumx = x->sum();
	outln("sumx " << sumx);
	assert(util<T>::almostEquals(sumx, 2.626782601596818e05));

	CuMatrix<T>* y = f["y"];
	y->syncBuffers();
	outln("y " << y->toShortString());

	int input_layer_size = 400; // 20x20 Input Images of Digits
	int hidden_layer_size_s = 1; //
	int hidden_layer_size_e = 20; //
	int hidden_layer2_size_s = 1; //
	int hidden_layer2_size_e = 20; //
	int num_labels = 10; // 10 labels, from 1 to 10
	T cost;
	CuMatrix<T> xBiased = x->addBiasColumn();
	assert(xBiased.biasedQ());
	CuMatrix<T> training, training2, cv, cv2;
	vector<int> vIndices;
	xBiased.shuffle(&training, &cv, (T) .75, vIndices);
	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	y->shuffle(&ytraining, &ycv, (T) .75, vIndices);
	CuMatrix<T> yBiased = y->toBinaryCategoryMatrix().syncBuffers();
	//assert(yBiased.biasedQ());
	yBiased.shuffle(&ytrainingBiased, &ycvBiased, (T) .75, vIndices);
	outln("after shuffling yBiased " << yBiased.toShortString());

	// n = h1 dim, h2 dim, cv acc)
	CuMatrix<T> resultTable = CuMatrix<T>::zeros(
			(hidden_layer_size_e - hidden_layer_size_s)
					* (hidden_layer2_size_e - hidden_layer2_size_s), 3);
	int resultDev = b_util::deviceOfResidence( resultTable );

	outln("zerod: resultTeble " << resultTable.toShortString() << " on device " << resultDev);
	checkCudaError(cudaGetLastError());

	ConjugateGradient<T>::init();
	int resultIdx = 0;
	Cycler cycler(ExecCaps::countGpus());

	vector<reference_wrapper<CuMatrix<T>>> mats;
	mats.push_back(training);
	mats.push_back(ytrainingBiased);


	for (int hiddenSize = hidden_layer_size_s; hiddenSize < hidden_layer_size_e;
			hiddenSize++) {
		for (int hiddenSize2 = hidden_layer2_size_s;
				hiddenSize2 < hidden_layer2_size_e; hiddenSize2++) {

			b_util::allDevices([]() { setCurrGpuDebugFlags( !(debugCons | debugDestr),false,true,0 );});
			int device = cycler.next();
			outln("device (cycler.next) " << device);
			outln("training b "  << training.toShortString());
			training.getMgr().migrate(device, mats);
			outln("training a "  << training.toShortString());

			ExecCaps_setDevice(device);

			CuMatrix<T> initial_Theta1 =
					repeatable ?
							CuMatrix<T>::sin(hiddenSize, input_layer_size).syncBuffers().addBiasColumn() :
							CuMatrix<T>::randn(hiddenSize, input_layer_size).syncBuffers().addBiasColumn();
			CuMatrix<T> initial_Theta2 =
					repeatable ?
							CuMatrix<T>::sin(hiddenSize2, hiddenSize).syncBuffers().addBiasColumn() :
							CuMatrix<T>::randn(hiddenSize2, hiddenSize).syncBuffers().addBiasColumn();
			CuMatrix<T> initial_Theta3 =
					repeatable ?
							CuMatrix<T>::sin(num_labels, hiddenSize2).syncBuffers().addBiasColumn() :
							CuMatrix<T>::randn(num_labels, hiddenSize2).syncBuffers().addBiasColumn();

			outln(
					"hs " << hiddenSize << ": " << "initial_Theta1 " << initial_Theta1.syncBuffers().toShortString());
			outln(
					"hs " << hiddenSize << ": " << "initial_Theta2 " << initial_Theta2.syncBuffers().toShortString());
			outln(
					"hs " << hiddenSize << ": " << "initial_Theta3 " << initial_Theta3.syncBuffers().toShortString());
			const CuMatrix<T>* parts3[] = { &initial_Theta1, &initial_Theta2,
					&initial_Theta3 };
			CuMatrix<T> initial_nn_params(
					(initial_Theta1.size + initial_Theta2.size
							+ initial_Theta3.size) / sizeof(T), 1, false, true);
			outln("hs " << hiddenSize << ": " << "initial_nn_params bef concat " << initial_nn_params.toShortString());

			assert(ExecCaps::currDev() == device);
			CuMatrix<T>::concat(initial_nn_params, 3, parts3);
			//ok if(1==1) return 0;
			uint2 dims[3];

			dims[0].x = initial_Theta1.m;
			dims[0].y = initial_Theta1.n;
			dims[1].x = initial_Theta2.m;
			dims[1].y = initial_Theta2.n;
			dims[2].x = initial_Theta3.m;
			dims[2].y = initial_Theta3.n;
			outln("\n\ninitial_nn_params aft concat " << initial_nn_params.toShortString());

			//
			//
			//
			//
			Migrator<T> migr(device);
			migr << training << ytraining << ytrainingBiased << ycv << ycvBiased << cv ;
			//
			//
			//
			//

			nnCostFtorPm<T> pmFtor3(dims, 3, training, ytrainingBiased, lambda);
			CuTimer justFmincg;
			justFmincg.start();
			outln("justFmincg.start()");

			pair<CuMatrix<T>, pair<CuMatrix<T>, int> > tup2_3 =
					ConjugateGradient<T>::fmincg(pmFtor3, initial_nn_params,
							iterations);
			outln( "back from initial_nn_params pmfmincg, took " << justFmincg.stop()/1000 << "s");
			assert(ExecCaps::currDev() == device);

			//
			//
			CuMatrix<T> nn_parms3PmGrad = tup2_3.first;
			outln("nn_parms3PmGrad " << nn_parms3PmGrad.toShortString());
			outln("nn_parms3PmGrad.sum " << nn_parms3PmGrad.sum());

			CuMatrix<T> nTheta1Pm3(dims[0].x, dims[0].y, false, true);
			CuMatrix<T> nTheta2Pm3(dims[1].x, dims[1].y, false, true);
			CuMatrix<T> nTheta3Pm3(dims[2].x, dims[2].y, false, true);
							// mat, m, n, p, off
			nn_parms3PmGrad.unconcat(nTheta1Pm3, hiddenSize,
					(input_layer_size + 1), (input_layer_size + 1), 0);
			outln("nTheta1Pm3 " << nTheta1Pm3.toShortString());
			outln("nTheta1Pm3,sum " << nTheta1Pm3.sum());

			nn_parms3PmGrad.unconcat(nTheta2Pm3, hiddenSize2, (hiddenSize + 1),
					(hiddenSize + 1), hiddenSize * (input_layer_size + 1));
			outln("nTheta2Pm3 " << nTheta2Pm3.toShortString());
			outln("nTheta2Pm3,sum " << nTheta2Pm3.sum());

			nn_parms3PmGrad.unconcat(nTheta3Pm3, num_labels, (hiddenSize2 + 1),
					(hiddenSize2 + 1),
					hiddenSize * (input_layer_size + 1 + hiddenSize2)
							+ hiddenSize2);
			outln("nTheta3Pm3 " << nTheta3Pm3.toShortString());
			outln("nTheta3Pm3,sum " << nTheta3Pm3.sum());
			checkCudaError(cudaGetLastError());
			//  outln("nTheta2Pm " << nTheta2Pm.syncBuffers());

			vector<CuMatrix<T> > thetaV;
			thetaV.push_back(nTheta1Pm3);
			thetaV.push_back(nTheta2Pm3);
			thetaV.push_back(nTheta3Pm3);

			int destDev = ExecCaps::currDev();

			bool sameDev = b_util::onDeviceQ( destDev, nTheta1Pm3,nTheta2Pm3,nTheta3Pm3,training,
					ytrainingBiased);
			outln("all on " << destDev << ": " << tOrF(sameDev));
			if(! sameDev) {
				map<int,long> devOx;
				auto allMats = { nTheta1Pm3,nTheta2Pm3,nTheta3Pm3,training,
						ytrainingBiased};
				for(auto & im : allMats )
					outln("before " << im.toShortString());
				int maxDev = b_util::devByMaxOccupancy(devOx, nTheta1Pm3,nTheta2Pm3,nTheta3Pm3,training,
						ytrainingBiased);
				outln("maxDev " << maxDev);
				outln("devOx:");
				for(auto &p : devOx)
					outln("devOx "<< p.first << ": " << p.second);
				int migCnt = b_util::migrate(destDev,nTheta1Pm3,nTheta2Pm3,nTheta3Pm3,training,
					ytrainingBiased  );
				outln("migCnt " << migCnt);
				for(auto & im : allMats )
					outln("afteur " << im.toShortString());

			}

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

			b_util::allDevices([]() { setCurrGpuDebugFlags(debugCons | debugDestr,true,false,0 );});

			CuMatrix<T> other = CuMatrix<T>::ones(1260,1) ;
			outln("other " << other.toShortString());
			checkCudaError(cudaGetLastError());
			outln("h1Pmcv3 " << h1Pmcv3.toShortString());
			outln(hiddenSize2 << " other " << other.toShortString());
			outln("h1Pmcv3 s " << h1Pmcv3.sum());

			T resPm3 = ytraining.accuracy(h1Pm3);
			checkCudaError(cudaGetLastError());
			outln("Pm3 training accuracy : " << ( resPm3 * 100));

			T rescvPm3 = ycv.accuracy(h1Pmcv3);
			checkCudaError(cudaGetLastError());
			outln(
					"cvacc " << hiddenSize << " " << hiddenSize2 << " "<< (rescvPm3 * 100) << ";  curr dev " << ExecCaps::currDev());
			if(!b_util::onDeviceQ( destDev, resultTable)) {
				ExecCaps_visitDevice(  b_util::deviceOfResidence( resultTable ) );
				resultTable.set(resultIdx, 0, hiddenSize);
				resultTable.set(resultIdx, 1, hiddenSize2);
				resultTable.set(resultIdx, 2, rescvPm3);
				ExecCaps_restoreDevice( destDev );
				checkCudaError(cudaGetLastError());
			}
			resultIdx++;


			outln("end of inner block");
		}
	}

	resultTable.toFile(NN_DIM_RESULTS_FILE);

	return 0;
}

