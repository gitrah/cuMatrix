/*
 * testneural.cc
 *
 *  Created on: Oct 15, 2012
 *      Author: reid
 */

#include "../NeuralNet.h"
#include "../CuMatrix.h"
#include "../ConjugateGradient.h"
#include "../debug.h"
#include "../util.h"
#include "../Cycler.h"
#include "../Kernels.h"
#include "tests.h"
#include <cuda_profiler_api.h>
#include <set>
#include <omp.h>
#include <thread>

const char* SAMPLES_FILE = "ex4data1.txt";
const char* WEIGHTS_FILE = "ex4weights.txt";
const char* FILES[2] = {SAMPLES_FILE,WEIGHTS_FILE};
const char* REDWINE_CSV_FILE = "winequality-red.csv";
const char* ADULT_FILE = "exadultdata.txt";
const char* YEAR_PRED_FILE = "exYearPredictioMatdata0.txt";
const char* NN_DIM_RESULTS_FILE = "exNNdim-data.txt";
extern template class ConjugateGradient<float> ;
extern template class ConjugateGradient<double> ;

template int testNormalize<float>::operator()(int argc,
		const char ** args) const;
template int testNormalize<double>::operator()(int argc,
		const char ** args) const;
template int testNormalize<ulong>::operator()(int argc,
		const char ** args) const;
template<typename T> int testNormalize<T>::operator()(int argc,
		const char** args) const {
	CuTimer timer;
	timer.start();
	string name = "appCats";
	if (argc > 1) {
		name = args[1];
	}
	string fname = "ct" + b_util::cap(name) + ".txt";
	map<string, CuMatrix<T>*> f = CuMatrix<T>::parseOctaveDataFile(
			fname.c_str(), false, true);
	CuMatrix<T>* rawVec = f[name];
	outln("reading rawVec took " << fromSeconds(timer.stop()/1000.));
	outln("rawVec " << rawVec->toShortString());
	timer.start();
	rawVec->syncBuffers();
	outln(
			"rawVec syncBuffers took " << timer.stop() << "s and result is:\n" << *rawVec);
	timer.start();
	T sum = rawVec->sum();
	outln("sum took "<< fromSeconds(timer.stop()/1000.));
	outln("sum " << sum);
	T avg = sum / rawVec->length();
	outln("avg " << avg);
	timer.start();
	CuMatrix<T> meanNormed = *rawVec - avg;
	outln("meanNormed " << meanNormed.syncBuffers());
	outln("meanNormed took " << fromSeconds(timer.stop()/1000));

	// move mnormed off device
	meanNormed.getMgr().freeTiles(meanNormed);

	timer.start();
	T sqrAvg = avg * avg;
	CuMatrix<T> sqrRawVec = rawVec->sqr();
	outln("sqrRawVec " << sqrRawVec.syncBuffers());

	rawVec->getMgr().freeTiles(*rawVec);

	sqrRawVec = sqrRawVec - sqrAvg;
	cherr(cudaPeekAtLastError());

	outln("sqrRawVec ss " << sqrRawVec.toShortString());

	T stDev = sqrtf(sqrRawVec.sum() / (rawVec->length() - 1));
	cherr(cudaPeekAtLastError());
	outln("stDev  " << stDev);

	meanNormed.getMgr().freeTiles(sqrRawVec);
	cherr(cudaPeekAtLastError());
	// restore dev meanNormed
	meanNormed.invalidateDevice();
	cherr(cudaPeekAtLastError());
	meanNormed.syncBuffers();
	meanNormed = meanNormed / stDev;
	cherr(cudaPeekAtLastError());
	outln("norming rawVec took " << fromSeconds(timer.stop()/1000));

	T plus = meanNormed.reduce(Functory<T, plusBinaryOp>::pinch(), 0);
	cherr(cudaPeekAtLastError());
	outln("meanNormed.sum (should be ~ zero) " << plus);
	meanNormed.syncBuffers();
	cherr(cudaPeekAtLastError());
	string saveName = name + "Normed";
	CuMatrix<T>::toOctaveFile(saveName.c_str(), meanNormed);

	// free org rawVec
	util<CuMatrix<T> >::deletePtrMap(f);
	return 0;
}

template int testNeural<float>::operator()(int argc, const char **argv) const;
template int testNeural<double>::operator()(int argc, const char **argv) const;
template int testNeural<ulong>::operator()(int argc, const char **argv) const;
template<typename T> int testNeural<T>::operator()(int argc,
		const char** args) const {

	bool ompq = checkCmdLineFlag(argc,args, "omp");

	bool checkGrad = true;
	T lambda = 3;
	if (checkGrad) {
		NeuralNet<T>::checkNnGradients(lambda);
	}

	int iterations = b_util::getParameter(argc, args, "its", 100);

	map<string, CuMatrix<T>*> f_s[2]; //f, fw;
	map<string, CuMatrix<T>*> f,fw;
	int tid;

	if(ompq) {
#ifdef CuMatrix_UseOmp

#pragma omp parallel  private(tid) num_threads(2)
	{
		std::thread::id curr_thread_id = std::this_thread::get_id();

		outln("omp_in_parallel  " << tOrF(omp_in_parallel()));
		outln("curr_thread_id == main_thread_id  " << tOrF(curr_thread_id == main_thread_id));
		outln("omp_get_num_procs " << omp_get_num_procs());
		tid = omp_get_thread_num();
		outln("tid " << tid << " open " << FILES[tid %2]);
		f_s[tid %2] = CuMatrix<T>::parseOctaveDataFile(FILES[tid %2], false, true);

		outln("omp_get_level (nesting level) " << omp_get_level());
		outln("omp_in_parallel  " << tOrF(omp_in_parallel()));
	}
	f = f_s[0];
	fw = f_s[1];
#else
	{
		f = CuMatrix<T>::parseOctaveDataFile(SAMPLES_FILE, false, true);
		fw = CuMatrix<T>::parseOctaveDataFile(WEIGHTS_FILE, false, true);
	}
#endif
	} else {
		f = CuMatrix<T>::parseOctaveDataFile(SAMPLES_FILE, false, true);
		fw = CuMatrix<T>::parseOctaveDataFile(WEIGHTS_FILE, false, true);
	}

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
	int hidden_layer_size = 25; //   25 hidden units
	int num_labels = 10; // 10 labels, from 1 to 10

	T cost;
	CuMatrix<T> xBiased = x->addBiasColumn();
	assert(xBiased.biasedQ());
	CuMatrix<T> training, training2, cv, cv2;


	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	vector<int> vIndices;
	CuMatrix<T> yBiased = y->toBinaryCategoryMatrix().syncBuffers();

	if( ompq) {

#ifdef CuMatrix_UseOmp
	CuMatrix<T>* srcs[] = {&xBiased, y};
	CuMatrix<T>* trgPtrs[] = {&training, &ytraining};
	CuMatrix<T>* trgPtrs2[] = {&cv, &ycv};
	vector<int> vIndicesAr[] = {vector<int>(), vector<int>()};
	int idx;
#pragma omp parallel  private(tid, idx) num_threads(2)
	{
		tid = omp_get_thread_num();
		idx = tid %2;
		outln("tid " << tid);
		srcs[idx]->shuffle(trgPtrs[idx], trgPtrs2[idx], (T) .75, vIndicesAr[idx]);
		outln("tid " << tid << " back from shuffle ");
		cherr(cudaDeviceSynchronize());
	}
#else
	xBiased.shuffle(&training, &cv, (T) .75, vIndices);
	y->shuffle(&ytraining, &ycv, (T) .75, vIndices);
	//assert(yBiased.biasedQ());
#endif
	} else {
		outln("noomp");
		xBiased.shuffle(&training, &cv, (T) .75, vIndices);
		y->shuffle(&ytraining, &ycv, (T) .75, vIndices);
	}
	yBiased.shuffle(&ytrainingBiased, &ycvBiased, (T) .75, vIndices);

	outln("after shuffling xBiased sum %ld " << xBiased.sum());
	outln("after shuffling y sum %ld " << y->sum());

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


	util<CuMatrix<T> >::deletePtrMap(f);
	util<CuMatrix<T> >::deletePtrMap(fw);
	return 0;
}




template<typename T> int testNeuralKPtr(int iterations,
		map<string, CuMatrix<T>*>& f, map<string, CuMatrix<T>*>& fw) {
	//setCurrGpuDebugFlags( debugVerbose,true,false);

	T lambda = 3;

	CuMatrix<T>* x = f["X"];
	outln("x "<< x->toShortString());
	x->syncBuffers();
	assert(x->lastMod == mod_synced);
	T sumx = x->sum();
	outln("sumx " << sumx);
	assert(util<T>::almostEquals(sumx, 2.626782601596818e05));
	CuMatrix<T> xT = x->transpose();
	T sumxT = xT.sum();
	outln("sumxT " << sumxT);
	assert(util<T>::almostEquals(sumx, sumxT));
	//outln("0x " << x->sum());
	CuMatrix<T>* y = f["y"];
	y->syncBuffers();
	outln("y " << y->toShortString());

	CuMatrix<T>* theta1 = fw["Theta1"];
	CuMatrix<T>* theta2 = fw["Theta2"];
	theta1->syncBuffers();
	theta2->syncBuffers();

	T theta1Sum = theta1->sum();
	outln("theta1.sum " << theta1Sum);
	assert(util<T>::almostEquals(theta1Sum, 9.2426, 0.0001));

	T theta2Sum = theta2->sum();
	outln("theta2.sum " << theta2Sum);
	assert(util<T>::almostEquals(theta2Sum, -100.08, 0.01));

	CuMatrix<T> mus = x->featureMeans(true);

	outln("yBiasing");
	CuMatrix<T> yBiased = y->toBinaryCategoryMatrix().syncBuffers();
	outln("yBiased " << yBiased.toShortString());
	checkCudaErrors(cudaGetLastError());

	int input_layer_size = 400; // 20x20 Input Images of Digits
	int hidden_layer_size = 25; //   25 hidden units
	int num_labels = 10; // 10 labels, from 1 to 10

	const CuMatrix<T>* thetaParts[] = { theta1, theta2 };
	outln(
			"(theta1->size + theta2->size)/sizeof(T) " << (theta1->size + theta2->size)/sizeof(T));

	CuMatrix<T> thetas((theta1->size + theta2->size) / sizeof(T), 1, false,
			true);
	CuMatrix<T>::concat(thetas, 2, thetaParts);
	outln("created thetas " << thetas.toShortString());
	thetas.syncBuffers();
	T thetasSum = thetas.sum();
	assert(util<T>::almostEquals(thetasSum, theta1Sum + theta2Sum, 0.01));
	outln("thetas.sum " << thetasSum);
	lambda = 0;
	CuMatrix<T> gradl0;
	T costL0 = 0;

	outln("adding bias");
	CuMatrix<T> xBiased = x->addBiasColumn();
	outln(
			"added; x " << x->toShortString() << " --> " << xBiased.toShortString());

	T z02sum = (*theta1 * xBiased.transpose()).sum();
	outln("z02sum " << z02sum);

	NeuralNet<T>::nnCostFunction(gradl0, costL0, thetas, input_layer_size,
			hidden_layer_size, num_labels, xBiased, yBiased, lambda);

	outln("costL0 " << costL0);

	outln("gradl0 " << gradl0.syncBuffers());
	outln("gradl0.sum " << gradl0.sum());
	flprintf("costL0 %f\n", (double )costL0);
	assert(util<T>::almostEquals(costL0, 0.287629, 0.0001));

	T cost;
	lambda = 3;
	CuMatrix<T> gradl1; // = CuMatrix<T>::zeros( gradl0.m,gradl0.n );
	outln("thetas " << thetas.toShortString());
	outln("xBiased " << xBiased.toShortString());
	outln("yBiased " << yBiased.toShortString());
	outln("gradl1 " << gradl1.toShortString());

	NeuralNet<T>::nnCostFunction(gradl1, cost, thetas, input_layer_size,
			hidden_layer_size, num_labels, xBiased, yBiased, lambda);
	outln("cost lambada = 3 " << cost);
	assert(util<T>::almostEquals(cost, 0.576051, 0.003));

	lambda = 1;
	NeuralNet<T>::nnCostFunction(gradl1, cost, thetas, input_layer_size,
			hidden_layer_size, num_labels, xBiased, yBiased, lambda);
	outln("cost lambada = 1 " << cost);
	assert(util<T>::almostEquals(cost, 0.383770, 0.001));

	T sigArry[] = { 1, -.5, 0, .5, 1 };
	CuMatrix<T> sigTest1(sigArry, 1, 5, true);
	CuMatrix<T> aSigTest = sigTest1.syncBuffers().sigmoidGradient();
	outln("aSigTest " << aSigTest.syncBuffers());
	checkCudaError(cudaGetLastError());

	CuMatrix<T> training, training2, cv, cv2;
	vector<int> vIndices;

	//outln("shuffling");
	//outln("xBiased " << xBiased);
	T xbs = xBiased.sum();
	outln("xBiased.sum " << xbs);
	xBiased.shuffle(&training, &cv, (T) .75, vIndices);
	//x->shuffle(training2,cv2,(T).75,vIndices);
	outln("shuffled ");
	//outln( training);

	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	y->shuffle(&ytraining, &ycv, (T) .75, vIndices);
	yBiased.shuffle(&ytrainingBiased, &ycvBiased, (T) .75, vIndices);

	nnCostFtor<T> costFn(input_layer_size, hidden_layer_size, num_labels,
			training, ytrainingBiased, lambda);
	outln("explicit costFn(thetas)");
	CuMatrix<T> grad(thetas.size / sizeof(T), 1, false, true);
	outln("created grad " << grad.toShortString());
	outln("created2 grad " << grad.toShortString());
	costFn(grad, cost, thetas);
	checkCudaError(cudaGetLastError());
	outln("cost " << cost);
	costFn.lambda = 1;
	costFn(grad, cost, thetas);
	checkCudaError(cudaGetLastError());
	outln("cost lambada = 1 " << cost);

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

	T initThetas1Sum = initial_Theta1.sum();
	T initThetas2Sum = initial_Theta2.sum();
	T initThetasSum = initThetas1Sum + initThetas2Sum;
	outln("initThetas1Sum " << initThetas1Sum);
	outln("initThetas2Sum " << initThetas2Sum);
	outln("initThetasSum " << initThetasSum);

	CuMatrix<T> initial_nn_params(
			(initial_Theta1.size + initial_Theta2.size) / sizeof(T), 1, false,
			true);
	outln("initial_nn_params bef concat " << initial_nn_params.toShortString());
	CuMatrix<T>::concat(initial_nn_params, 2, parts);
	outln("initial_nn_params aft concat " << initial_nn_params.syncBuffers());
	flprintf("initial_nn_params sum %.20g\n", initial_nn_params.sum());

	//CuMatrix<T> initial_nn_params = CuMatrix<T>::zeros(1,(initial_Theta1.size + initial_Theta2.size)/sizeof(T));

	//
	//
	//		conjugate gradient
	/////////////////////////////////
	//
	//
	//

	nnCostFtor<T> ftor(input_layer_size, hidden_layer_size, num_labels,
			training, ytrainingBiased, lambda);

	CuMatrix<T> initGrad;
	T initCost;
	ftor(initGrad, initCost, initial_nn_params);
	outln("initGrad " << initGrad.syncBuffers());
	outln("initCost (lambda = " << lambda << "):  "<< initCost);

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

	return 0;
//    return testNeuralKPtrPm( iterations, &initial_Theta1,  &initial_Theta2, initThetasSum, initCost, xBiased, yBiased,
//    		training, cv, ytraining,  ycv, ycvBiased, ytrainingBiased, p1, h1);
}

template<typename T> int testNeuralKPtrMulti(int iterations,
		map<string, CuMatrix<T>*>& f, map<string, CuMatrix<T>*>& fw) {
	//setCurrGpuDebugFlags( debugVerbose,true,false);

	T lambda = 1;

	CuMatrix<T>* x = f["X"];
	outln("x "<< x->toShortString());
	x->syncBuffers();
	assert(x->lastMod == mod_synced);
	T sumx = x->sum();
	outln("sumx " << sumx);
	assert(util<T>::almostEquals(sumx, 2.626782601596818e05));
	CuMatrix<T> xT = x->transpose();
	T sumxT = xT.sum();
	outln("sumxT " << sumxT);
	assert(util<T>::almostEquals(sumx, sumxT));
	//outln("0x " << x->sum());
	CuMatrix<T>* y = f["y"];
	y->syncBuffers();
	outln("y " << y->toShortString());

	CuMatrix<T>* theta1 = fw["Theta1"];
	CuMatrix<T>* theta2 = fw["Theta2"];
	theta1->syncBuffers();
	theta2->syncBuffers();

	T theta1Sum = theta1->sum();
	outln("theta1.sum " << theta1Sum);
	assert(util<T>::almostEquals(theta1Sum, 9.2426, 0.0001));

	T theta2Sum = theta2->sum();
	outln("theta2.sum " << theta2Sum);
	assert(util<T>::almostEquals(theta2Sum, -100.08, 0.01));

	CuMatrix<T> mus = x->featureMeans(true);
	outln("yBiasing");
	CuMatrix<T> yBiased = y->toBinaryCategoryMatrix().syncBuffers();
	outln("yBiased " << yBiased.toShortString());
	checkCudaErrors(cudaGetLastError());

	int input_layer_size = 400; // 20x20 Input Images of Digits
	int hidden_layer_size = 25; //   25 hidden units
	int hidden_layer_size2 = 25; //   25 hidden units
	int num_labels = 10; // 10 labels, from 1 to 10

	const CuMatrix<T>* thetaParts[] = { theta1, theta2 };
	CuMatrix<T> thetas((theta1->size + theta2->size) / sizeof(T), 1, false,
			true);
	CuMatrix<T>::concat(thetas, 2, thetaParts);
	outln("created thetas " << thetas.toShortString());
	thetas.syncBuffers();
	T thetasSum = thetas.sum();
	assert(util<T>::almostEquals(thetasSum, theta1Sum + theta2Sum, 0.01));
	outln("thetas.sum " << thetasSum);
	lambda = 0;
	CuMatrix<T> gradl0;
	T costL0 = 0;

	checkCudaErrors(cudaGetLastError());

	outln("adding bias");
	CuMatrix<T> xBiased = x->addBiasColumn();
	outln(
			"added; x " << x->toShortString() << " --> " << xBiased.toShortString());

	T z02sum = (*theta1 * xBiased.transpose()).sum();
	outln("z02sum " << z02sum);

	NeuralNet<T>::nnCostFunction(gradl0, costL0, thetas, input_layer_size,
			hidden_layer_size, num_labels, xBiased, yBiased, lambda);

	outln("costL0 " << costL0);

	outln("gradl0 " << gradl0.syncBuffers());
	outln("gradl0.sum " << gradl0.sum());
	flprintf("costL0 %f\n", (double )costL0);
	assert(util<T>::almostEquals(costL0, 0.287629, 0.0001));

	T cost;
	CuMatrix<T> gradl1; // = CuMatrix<T>::zeros( gradl0.m,gradl0.n );
	lambda = 3;
	outln("thetas " << thetas.toShortString());
	outln("xBiased " << xBiased.toShortString());
	outln("yBiased " << yBiased.toShortString());
	outln("gradl1 " << gradl1.toShortString());

	NeuralNet<T>::nnCostFunction(gradl1, cost, thetas, input_layer_size,
			hidden_layer_size, num_labels, xBiased, yBiased, lambda);
	outln("cost lambada = 3 " << cost);

	vector<CuMatrix<T> > thetaV;
	thetaV.push_back(*theta1);
	thetaV.push_back(*theta2);
	CuMatrix<T> fbGrad;
	T fbCost;
	NeuralNet<T>::forwardAndBack(fbGrad, fbCost, xBiased, yBiased, thetaV,
			lambda);
	outln("fbCost " << fbCost);

	assert(util<T>::almostEquals(cost, 0.576051, 0.003));

	lambda = 1;

	NeuralNet<T>::nnCostFunction(gradl1, cost, thetas, input_layer_size,
			hidden_layer_size, num_labels, xBiased, yBiased, lambda);
	outln("cost lambada = 1 " << cost);
	assert(util<T>::almostEquals(cost, 0.383770, 0.001));

	T sigArry[] = { 1, -.5, 0, .5, 1 };
	CuMatrix<T> sigTest1(sigArry, 1, 5, true);
	CuMatrix<T> aSigTest = sigTest1.syncBuffers().sigmoidGradient();
	outln("aSigTest " << aSigTest.syncBuffers());
	checkCudaError(cudaGetLastError());

	CuMatrix<T> training, training2, cv, cv2;
	vector<int> vIndices;

	//outln("shuffling");
	//outln("xBiased " << xBiased);
	T xbs = xBiased.sum();
	outln("xBiased.sum " << xbs);
	xBiased.shuffle(training, cv, (T) .75, vIndices);
	//x->shuffle(training2,cv2,(T).75,vIndices);
	outln("shuffled ");
	//outln( training);

	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	y->shuffle(&ytraining, &ycv, (T) .75, vIndices);
	yBiased.shuffle(&ytrainingBiased, &ycvBiased, (T) .75, vIndices);

	nnCostFtor<T> costFn(input_layer_size, hidden_layer_size, num_labels,
			training, ytrainingBiased, lambda);
	outln("explicit costFn(thetas)");
	CuMatrix<T> grad(thetas.size / sizeof(T), 1, false, true);
	outln("created grad " << grad.toShortString());
	outln("created2 grad " << grad.toShortString());
	costFn(grad, cost, thetas);
	checkCudaError(cudaGetLastError());
	outln("cost " << cost);
	costFn.lambda = 1;
	costFn(grad, cost, thetas);
	checkCudaError(cudaGetLastError());
	outln("cost lambada = 1 " << cost);

	//util<T>::timeReps(&CuMatrix<T>::sigmoidGradient,"sigmoidGradient", &sigTest2, 10000);

	//CuMatrix<T> initial_Theta1 = CuMatrix<T>::randn(hidden_layer_size, input_layer_size).addBiasColumn();
	//CuMatrix<T> initial_Theta2 = CuMatrix<T>::randn(num_labels, hidden_layer_size).addBiasColumn();

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

	nnCostFtor<T> costFnTraining(input_layer_size, hidden_layer_size,
			num_labels, training, ytrainingBiased, lambda);
	costFnTraining(grad, cost, nn_parms);
	outln("cost on training " << cost);
	nnCostFtor<T> costFnCv(input_layer_size, hidden_layer_size, num_labels, cv,
			ycvBiased, lambda);
	costFnCv(grad, cost, nn_parms);
	outln("cost on cv " << cost);
//    cherr(cudaProfilerStop());

	return 0;
}

template int testNeural2Loop<float>::operator()(int argc,
		const char ** args) const;
template int testNeural2Loop<double>::operator()(int argc,
		const char ** args) const;
template int testNeural2Loop<ulong>::operator()(int argc,
		const char ** args) const;
template<typename T> int testNeural2Loop<T>::operator()(int argc,
		const char** args) const {
	CuTimer timer;
	map<string, float> runTimes;
	timer.start();
	int count = b_util::getCount(argc, args, 5);
	outln("exp test " << b_util::expNotation(1000l) );
	outln("plz test " << b_util::plz("run",count) );
	float exeTime;
	outln("starting trial of " << count << " " << b_u::plz("run",count) );
	void (*matProdKptr[])(DMatrix<T>, const DMatrix<T>, const DMatrix<T>,
			int) = {matrixProductKernel,matrixProductKernel2,matrixProductKernel3,matrixProductKernel4 /*,matrixProductKernelTxdB,matrixProductKernelTxdB2*/};
	const char* names[] = { "k1 ", "k2 ", "k3 ", "k4 ", "ktx ", "ktx2 " };
	int iterations = b_util::getParameter(argc, args, "its", 50);
	map<string, CuMatrix<T>*> f = CuMatrix<T>::parseOctaveDataFile(SAMPLES_FILE,
			false, true);
	map<string, CuMatrix<T>*> fw = CuMatrix<T>::parseOctaveDataFile(
			WEIGHTS_FILE, false, true);
	exeTime = timer.stop();
	outln(
			count << "loading took a total of " <<exeTime << " secs, avg t/rec " << exeTime/count);
	typedef pair<string, float> runpair;
	// DefaultMatProdBlock
	checkCudaErrors(cudaGetLastError());
	for (int kernel = 0; kernel < 4 * 2; kernel++) {
		CuMatrix<T>::g_matrix_product_kernel = matProdKptr[kernel/2];
		if (kernel & 1) {
			CuMatrix<T>::DefaultMatProdBlock = dim3(32, 32);
		} else {
			CuMatrix<T>::DefaultMatProdBlock = dim3(16, 16);
		}
		outln("starting nn with kernel " << names[kernel/2]);
		timer.start();
		testNeuralKPtr(iterations, f, fw);
		exeTime = timer.stop();
		//CuMatrix<T>::mgr->dumpLeftovers();
		runTimes.insert(runTimes.end(),
				runpair(
						names[kernel/2] + string(" ")
								+ b_util::pd3(CuMatrix<T>::DefaultMatProdBlock),
						exeTime));
	}
	outln("results");
	typedef typename map<string, float>::iterator iterator;
	for (iterator it = runTimes.begin(); it != runTimes.end(); it++) {
		outln((*it).first << " took " << (*it).second << "ms");
	}
	return 0;
}

template int testCheckNNGradients<float>::operator()(int argc,
		const char ** args) const;
template int testCheckNNGradients<double>::operator()(int argc,
		const char ** args) const;
template<typename T> int testCheckNNGradients<T>::operator()(int argc,
		const char** args) const {
	outln(
			"NeuralNet<T>::checkNnGradients() " << NeuralNet<T>::checkNnGradients());
	return 0;
}

template int testNeural2lYrPred<float>::operator()(int argc,
		const char ** args) const;
template int testNeural2lYrPred<double>::operator()(int argc,
		const char ** args) const;
template int testNeural2lYrPred<ulong>::operator()(int argc,
		const char ** args) const;
template<typename T> int testNeural2lYrPred<T>::operator()(int argc,
		const char** args) const {

	bool checkGrad = true;
	T lambda = 3;
	if (checkGrad) {
		//NeuralNet<T>::checkNnGradients(0);
		NeuralNet<T>::checkNnGradients(lambda);
	}

	int iterations = b_util::getParameter(argc, args, "its", 50);

	CuTimer timer;
	timer.start();
	map<string, CuMatrix<T>*> f = CuMatrix<T>::parseOctaveDataFile(
			YEAR_PRED_FILE, false, true);
	if (!f.size()) {
		outln("no " << YEAR_PRED_FILE << "; exiting");
		return -1;
	}
	outln(
			"loading " << YEAR_PRED_FILE << " took " << (timer.stop() / 1000) << " s");

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
	T yMin = y->min();
	CuMatrix<T> yTared = *y - yMin;
	CuMatrix<T> yBiased = yTared.toBinaryCategoryMatrix().syncBuffers();
	outln("yBiased " << yBiased.toShortString());
	checkCudaErrors(cudaGetLastError());

	int input_layer_size = 90; // 20x20 Input Images of Digits
	int hidden_layer_size = 55; //   25 hidden units
	int num_labels = 89; // 1922 to 2011

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

	//outln("shuffling");
	//outln("xBiased " << xBiased);
	T xbs = xBiased.sum();
	outln("xBiased.sum " << xbs);
	xBiased.shuffle(&training, &cv, (T) .75, vIndices);
	//x->shuffle(training2,cv2,(T).75,vIndices);
	outln("shuffled ");
	outln(training);

	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	yTared.shuffle(&ytraining, &ycv, (T) .75, vIndices);
	yBiased.shuffle(&ytrainingBiased, &ycvBiased, (T) .75, vIndices);

	nnCostFtor<T> costFn(input_layer_size, hidden_layer_size, num_labels,
			training, ytrainingBiased, lambda);
	outln("explicit costFn(thetas)");
//	costFn(grad,cost,thetas);
//	checkCudaError(cudaGetLastError());
//	outln("cost " << cost);
	costFn.lambda = 1;
//	costFn(grad,cost, thetas);
	checkCudaError(cudaGetLastError());
	//("cost lambada = 1 " << cost);

	//util<T>::timeReps(&CuMatrix<T>::sigmoidGradient,"sigmoidGradient", &sigTest2, 10000);

	//CuMatrix<T> initial_Theta1 = CuMatrix<T>::randn(hidden_layer_size, input_layer_size).addBiasColumn();
	//CuMatrix<T> initial_Theta2 = CuMatrix<T>::randn(num_labels, hidden_layer_size).addBiasColumn();

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
	checkCudaError(cudaGetLastError());
	//  outln("nTheta2 " << nTheta2.syncBuffers());

	CuMatrix<T> p1 = NeuralNet<T>::predictCg(nTheta1, nTheta2, training);
//    outln("p1 " << p1.syncBuffers());
	outln("p1 sum " << p1.sum());

	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1 = p1.toMaxColumnIndexVector() + yMin;
	checkCudaError(cudaGetLastError());
	CuMatrix<T> p1cv = NeuralNet<T>::predictCg(nTheta1, nTheta2, cv);
	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1cv = p1cv.toMaxColumnIndexVector() + yMin;
	checkCudaError(cudaGetLastError());
	outln("h1 " << h1.toShortString());
	outln("h1 s " << h1.sum());
	outln("ytraining " << ytraining.toShortString());
	outln("ytraining s " << ytraining.sum());

	outln("h1 " << h1);
	T res = ytraining.accuracy(h1);
	CuMatrix<T>::toOctaveFile("h1data.txt", h1);

	std::set<T> distinctH1 = h1.distinct();
	typename std::set<T>::const_iterator values = distinctH1.begin();
	outln("distinct h1\n"<< *values);

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

template int testNeural2lYrPredMulti<float>::operator()(int argc,
		const char ** args) const;
template int testNeural2lYrPredMulti<double>::operator()(int argc,
		const char ** args) const;
template int testNeural2lYrPredMulti<ulong>::operator()(int argc,
		const char ** args) const;
template<typename T> int testNeural2lYrPredMulti<T>::operator()(int argc,
		const char** args) const {

	bool checkGrad = true;
	T lambda = 3;
	if (checkGrad) {
		//NeuralNet<T>::checkNnGradients(0);
		NeuralNet<T>::checkNnGradients(lambda);
	}

	int iterations = b_util::getParameter(argc, args, "its", 50);

	CuTimer timer;
	timer.start();
	map<string, CuMatrix<T>*> f = CuMatrix<T>::parseOctaveDataFile(
			YEAR_PRED_FILE, false, true);
	if (!f.size()) {
		outln("no " << YEAR_PRED_FILE << "; exiting");
		return -1;
	}
	outln(
			"loading " << YEAR_PRED_FILE << " took " << (timer.stop() / 1000) << " s");

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

	//////////////////////////////

	int input_layer_size = 90; // 20x20 Input Images of Digits
	int hidden_layer_size = 100; //   25 hidden units
	int hidden_layer_size2 = 55; //   25 hidden units
	int num_labels = 89; // 1922 to 2011

	//////////////////////////////

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

	//outln("shuffling");
	//outln("xBiased " << xBiased);
	T xbs = xBiased.sum();
	outln("xBiased.sum " << xbs);
	xBiased.shuffle(&training, &cv, (T) .75, vIndices);
	//x->shuffle(training2,cv2,(T).75,vIndices);
	outln("shuffled ");
	outln(training);

	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	y->shuffle(&ytraining, &ycv, (T) .75, vIndices);
	yBiased.shuffle(&ytrainingBiased, &ycvBiased, (T) .75, vIndices);
	//outln("")
	nnCostFtor<T> costFn(input_layer_size, hidden_layer_size, num_labels,
			training, ytrainingBiased, lambda);
	outln("explicit costFn(thetas)");
//	costFn(grad,cost,thetas);
//	checkCudaError(cudaGetLastError());
//	outln("cost " << cost);
	costFn.lambda = 1;
//	costFn(grad,cost, thetas);
	checkCudaError(cudaGetLastError());
	//("cost lambada = 1 " << cost);

	//util<T>::timeReps(&CuMatrix<T>::sigmoidGradient,"sigmoidGradient", &sigTest2, 10000);

	//CuMatrix<T> initial_Theta1 = CuMatrix<T>::randn(hidden_layer_size, input_layer_size).addBiasColumn();
	//CuMatrix<T> initial_Theta2 = CuMatrix<T>::randn(num_labels, hidden_layer_size).addBiasColumn();

	CuMatrix<T> initial_Theta1 = CuMatrix<T>::sin(hidden_layer_size,
			input_layer_size).syncBuffers().addBiasColumn();
	CuMatrix<T> initial_Theta2 = CuMatrix<T>::sin(hidden_layer_size2,
			hidden_layer_size).syncBuffers().addBiasColumn();
	CuMatrix<T> initial_Theta3 = CuMatrix<T>::sin(num_labels,
			hidden_layer_size2).syncBuffers().addBiasColumn();

	outln("initial_Theta1 " << initial_Theta1.syncBuffers());
	outln("initial_Theta2 " << initial_Theta2.syncBuffers());
	outln("initial_Theta3 " << initial_Theta3.syncBuffers());

	const CuMatrix<T>* parts[] = { &initial_Theta1, &initial_Theta2,
			&initial_Theta3 };

	CuMatrix<T> initial_nn_params(
			(initial_Theta1.size + initial_Theta2.size + initial_Theta3.size)
					/ sizeof(T), 1, false, true);
	outln("initial_nn_params bef concat " << initial_nn_params.toShortString());
	flprintf("initial_nn_params %.20g\n", initial_nn_params.sum());
	CuMatrix<T>::concat(initial_nn_params, 3, parts);
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
	std::set<T> distinctH1 = h1.distinct();
	typename std::set<T>::const_iterator values = distinctH1.begin();
	outln("distinct h1\n"<< *values);
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
		const CuMatrix<T>& ytrainingBiased, const CuMatrix<T>& ycv) {
	T cost = 0;
	T lambda = 1;
	int input_layer_size = 400; // 20x20 Input Images of Digits
	int num_labels = 10; // 10 labels, from 1 to 10

	int layer1 = 15;
	int layer2 = 5;
	vector<CuMatrix<T> > thetaV;

	CuMatrix<T> thta1 =
			CuMatrix<T>::sin(layer1, input_layer_size).syncBuffers().addBiasColumn();
	CuMatrix<T> thta2 =
			CuMatrix<T>::sin(layer2, layer1).syncBuffers().addBiasColumn();
	CuMatrix<T> thta3 =
			CuMatrix<T>::sin(layer2, layer2).syncBuffers().addBiasColumn();
	CuMatrix<T> thta4 =
			CuMatrix<T>::sin(layer2, layer2).syncBuffers().addBiasColumn();
	CuMatrix<T> thta5 =
			CuMatrix<T>::sin(num_labels, layer2).syncBuffers().addBiasColumn();
	thetaV.clear();
	thetaV.push_back(thta1);
	thetaV.push_back(thta2);
	thetaV.push_back(thta3);
	thetaV.push_back(thta4);
	thetaV.push_back(thta5);

	CuMatrix<T> fbGrad;
	T fbCost;

	NeuralNet<T>::forwardAndBack(fbGrad, fbCost, training, ytrainingBiased,
			thetaV, lambda);
	outln("with 5 (thta1-5) layers, fbCost " << fbCost);

	const CuMatrix<T>* parts5[] = { &thta1, &thta2, &thta3, &thta4, &thta5 };
	//ok here  if(1==1) return 0;

	CuMatrix<T> nn_params_5(
			(thta1.size + thta2.size + thta3.size + thta4.size + thta5.size)
					/ sizeof(T), 1, false, true);
	outln("\n\nn_params_5 bef concat " << nn_params_5.toShortString());
	//flprintf("nn_params_2 %.20g\n", nn_params_2.sum());
	CuMatrix<T>::concat(nn_params_5, 5, parts5);
	//ok if(1==1) return 0;
	uint2 dims5[5];
	dims5[0].x = thta1.m;
	dims5[0].y = thta1.n;
	dims5[1].x = thta2.m;
	dims5[1].y = thta2.n;
	dims5[2].x = thta3.m;
	dims5[2].y = thta3.n;
	dims5[3].x = thta4.m;
	dims5[3].y = thta4.n;
	dims5[4].x = thta5.m;
	dims5[4].y = thta5.n;

	T nn_params_5Sum = nn_params_5.sum();
	//ok  if(1==1) return 0;

	outln("bef NeuralNet<T>::forwardAndBac with pm of nn_params_5 ");
	outln("bef NeuralNet<T>::forwardAndBac nn_params_5Sum " << nn_params_5Sum);
	//assert(util<T>::almostEquals(nn_params_5Sum, theta1Sum+theta2Sum, 1e-8));

	NeuralNet<T>::forwardAndBack(fbGrad, fbCost, training, ytrainingBiased,
			nn_params_5, 5, dims5, lambda);
	// NNPrediction<T> tupl3 = NeuralNet<T>::predict( pm, *x);
	outln("aft NeuralNet<T>::forwardAndBac nn_params_2; cost: " << fbCost);
	//assert(util<T>::almostEquals(initCost, fbCost, 1e-8));
	/*
	 * uint2* _dims,
	 int _layers,
	 CuMatrix<T>& _xBiased,
	 CuMatrix<T>& _y,
	 T _lambda
	 */
	nnCostFtorPm<T> pmFtor5(dims5, 5, training, ytrainingBiased, lambda);
	CuTimer justFmincg;
	justFmincg.start();

	pair<CuMatrix<T>, pair<CuMatrix<T>, int> > tup5_2 =
			ConjugateGradient<T>::fmincg(pmFtor5, nn_params_5, 100);
	outln(
			"back from nn_params_5 pmfmincg, took " << justFmincg.stop()/1000 << "s");

	CuMatrix<T> nn_parms = tup5_2.first;

	CuMatrix<T> nThta1;
	CuMatrix<T> nThta2;
	CuMatrix<T> nThta3;
	CuMatrix<T> nThta4;
	CuMatrix<T> nThta5;
	int ts = sizeof(T);
	nn_parms.unconcat(nThta1, dims5[0].x, dims5[0].y, dims5[0].y, 0);
	nn_parms.unconcat(nThta2, dims5[1].x, dims5[1].y, dims5[1].y, 0,
			thta1.size / ts);
	nn_parms.unconcat(nThta3, dims5[2].x, dims5[2].y, dims5[2].y, 0,
			(thta1.size + thta2.size) / ts);
	nn_parms.unconcat(nThta4, dims5[3].x, dims5[3].y, dims5[3].y, 0,
			(thta1.size + thta2.size + thta3.size) / ts);
	nn_parms.unconcat(nThta5, dims5[4].x, dims5[4].y, dims5[4].y, 0,
			(thta1.size + thta2.size + thta3.size + thta4.size) / ts);
	checkCudaError(cudaGetLastError());
	//  outln("nTheta2 " << nTheta2.syncBuffers());
	thetaV.clear();
	thetaV.push_back(nThta1);
	thetaV.push_back(nThta2);
	thetaV.push_back(nThta3);
	thetaV.push_back(nThta4);
	thetaV.push_back(nThta5);

	NNPrediction<T> pred5 = NeuralNet<T>::forward(thetaV, training,
			ytrainingBiased, lambda);
	CuMatrix<T> p1 = pred5.hTheta;
	outln("p1 " << p1.toShortString());
	outln("p1 sum " << p1.sum());

	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1 = p1.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	outln("h1 " << h1.toShortString());
	NNPrediction<T> pred5cv = NeuralNet<T>::forward(thetaV, cv, ycv, lambda);
	CuMatrix<T> p1cv = pred5cv.hTheta;
	outln("p1cv " << p1cv.syncBuffers().toShortString());
	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1cv = p1cv.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	outln("h1 s " << h1.sum());
	outln("ytraining " << ytraining.toShortString());
	outln("ytraining s " << ytraining.sum());

	T res = ytraining.accuracy(h1);
	std::set<T> distinctH1 = h1.distinct();
	typename std::set<T>::const_iterator values = distinctH1.begin();
	outln("distinct h1\n"<< *values);
	checkCudaError(cudaGetLastError());
	outln("training accuracy : " << ( res * 100));

	T rescv = ycv.accuracy(h1cv);
	checkCudaError(cudaGetLastError());
	outln("cv accuracy : " << ( rescv * 100));

}

template int testDim3Octave<float>::operator()(int argc, const char **argv) const;
template int testDim3Octave<double>::operator()(int argc, const char **argv) const;
template int testDim3Octave<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testDim3Octave<T>::operator()(int argc, const char **argv) const{


	std::map<std::string, dim3> results = CuMatrix<T>::parseOctaveMatrixSizes( SAMPLES_FILE );

	std::cout << "found " << results.size() << " octave objects\n";
	typedef typename std::map<std::string, dim3>::iterator iterator;
	iterator it;
	it = results.begin();

	while (it != results.end()) {
		dim3 m = (*it).second;
		outln("m " << m.x << ", "<< m.y << ", " << m.z);
		std::cout << (*it).first << std::endl;
		it++;
	}
	//results[std::string("X")];
	//std::cout << << "\n";
	return (0);
}
