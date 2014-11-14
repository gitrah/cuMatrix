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
#include "../Kernels.h"
#include "tests.h"

const char* SAMPLES_FILE = "ex4data1.txt";
const char* WEIGHTS_FILE = "ex4weights.txt";
const char* REDWINE_CSV_FILE = "winequality-red.csv";
extern template class  ConjugateGradient<float>;
extern template class  ConjugateGradient<double>;


template int testNeural2l<float>::operator()(int argc, char const ** args) const;
template int testNeural2l<double>::operator()(int argc, char const ** args) const;
template <typename T> int testNeural2l<T>::operator()(int argc, const char** args) const {

	int iterations = b_util::getParameter(argc, args, "its", 50);

	map<string, CuMatrix<T>*> f = util<T>::parseOctaveDataFile(SAMPLES_FILE,
    			false, true);
	map<string, CuMatrix<T>*> fw = util<T>::parseOctaveDataFile(WEIGHTS_FILE,
    			false, true);
	if(!f.size()) {
		outln("no " << SAMPLES_FILE << "; exiting");
		return -1;
	}
	if(!fw.size()) {
		outln("no " << WEIGHTS_FILE << "; exiting");
		return -1;
	}
	CuTimer timer;
	timer.start();
	int ret = testNeuralKPtr(iterations, f,fw);
	outln("testNeuralKPtr(" <<iterations <<") took " << timer.stop() << "ms");
    util<CuMatrix<T> >::deletePtrMap(f);
    util<CuMatrix<T> >::deletePtrMap(fw);
    return ret;
}

template int testNeural2lCsv<float>::operator()(int argc, char const ** args) const;
template int testNeural2lCsv<double>::operator()(int argc, char const ** args) const;
template <typename T> int testNeural2lCsv<T>::operator()(int argc, const char** args) const {

	//int iterations = b_util::getParameter(argc, args, "its", 50);

	map<string, CuMatrix<T>*> f = util<T>::parseOctaveDataFile(REDWINE_CSV_FILE,
    			false, true);
	if(!f.size()) {
		outln("no " << REDWINE_CSV_FILE << "; exiting");
		return -1;
	}
	//int ret = testNeuralKPtr(iterations, f,fw);
    util<CuMatrix<T> >::deletePtrMap(f);
    return 0;
}

template <typename T> int testNeuralKPtr(int iterations, map<string, CuMatrix<T>*>& f, map<string, CuMatrix<T>*>& fw ) {
	//setCurrGpuDebugFlags( debugVerbose,true,false);
    CuMatrix<T>* x = f["X"];
    x->syncBuffers();
    //outln("x from file " << *x);
    //outln("0x " << x->sum());
    CuMatrix<T>* y = f["y"];
    y->syncBuffers();
    outln("y " << y->toShortString());

    CuMatrix<T>* theta1 = fw["Theta1"];
    CuMatrix<T>* theta2 = fw["Theta2"];
    theta1->syncBuffers();
    theta2->syncBuffers();
    outln("theta1.sum " << theta1->sum());
    outln("theta2.sum " << theta2->sum());
    CuMatrix<T> mus = x->featureMeans(true);
    CuMatrix<T> yBiased = y->toBinaryCategoryMatrix();
	checkCudaErrors(cudaGetLastError());

    uint input_layer_size = 400; // 20x20 Input Images of Digits
    uint hidden_layer_size = 25; //   25 hidden units
    uint num_labels = 10; // 10 labels, from 1 to 10
    const CuMatrix<T>* thetaParts[] = {theta1, theta2};
    CuMatrix<T> thetas(1, (theta1->size + theta2->size)/sizeof(T),false,true);
    outln("created thetas " << thetas.toShortString());
    CuMatrix<T>::concat(thetas, 2, thetaParts);
   // outln("thetas.sum " << thetas.sum());
    T lambda = 0;

	checkCudaErrors(cudaGetLastError());

    CuMatrix<T> xBiased = x->addBiasColumn();
	CuMatrix<T> training, cv;
	vector<uint> vIndices;
	xBiased.shuffle(training,cv,(T).75,vIndices);
	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	y->shuffle(ytraining,ycv,(T).75,vIndices);
	yBiased.shuffle(ytrainingBiased,ycvBiased,(T).75,vIndices);

    nnCostFtor<T> costFn(input_layer_size,hidden_layer_size,num_labels, training, ytrainingBiased,lambda );
    outln("explicit costFn(thetas)");
    CuMatrix<T> grad(1,thetas.size/sizeof(T),false,true);
    outln("created grad " << grad.toShortString());
    T cost;
    costFn(grad,cost,thetas);
	checkCudaError(cudaGetLastError());
	outln("cost " << cost);
    costFn.lambda = 1;
    costFn(grad,cost, thetas);
	checkCudaError(cudaGetLastError());
	outln("cost lambada = 1 " << cost);

    T sigArry[] = {1, -.5, 0, .5, 1};
    CuMatrix<T> sigTest1(sigArry, 5, 1, true);
    CuMatrix<T>  sigTest2 = CuMatrix<T>::randn(50, 50, 25);
	checkCudaError(cudaGetLastError());
    DMatrix<T> dm;
    sigTest2.asDmatrix(dm);
    sigTest2.invalidateHost();
    outln("sigTest2 " << sigTest2.toShortString());

    //util<T>::timeReps(&CuMatrix<T>::sigmoidGradient,"sigmoidGradient", &sigTest2, 10000);

    bool check = false;
    lambda = 3;
    if(check) {
    	NeuralNet<T>::checkNnGradients(0);
        NeuralNet<T>::checkNnGradients(lambda);
    }

    //CuMatrix<T> initial_Theta1 = CuMatrix<T>::randn(hidden_layer_size, input_layer_size).addBiasColumn();
    //CuMatrix<T> initial_Theta2 = CuMatrix<T>::randn(num_labels, hidden_layer_size).addBiasColumn();
    CuMatrix<T> initial_Theta1 = CuMatrix<T>::sin(hidden_layer_size, input_layer_size).addBiasColumn();
    CuMatrix<T> initial_Theta2 = CuMatrix<T>::cos(num_labels, hidden_layer_size).addBiasColumn();
    const CuMatrix<T>* parts[] = {&initial_Theta1, &initial_Theta2};
    CuMatrix<T> initial_nn_params(1,(initial_Theta1.size + initial_Theta2.size)/sizeof(T),false,true);
    CuMatrix<T>::concat(initial_nn_params, 2, parts);

    nnCostFtor<T> ftor(input_layer_size, hidden_layer_size, num_labels, training, ytrainingBiased, lambda);

    ConjugateGradient<T>::init();
    ftor.verbose=true;
    outln("post init last err " << b_util::lastErrStr());

    CuTimer justFmincg;
    justFmincg.start();
    pair<CuMatrix<T>, pair<CuMatrix<T>, int > > tup2 = ConjugateGradient<T>::fmincg(ftor,initial_nn_params, iterations);
    outln("back from fmincg, took " << justFmincg.stop());
	checkCudaError(cudaGetLastError());

    CuMatrix<T> nn_parms = tup2.first;

    CuMatrix<T> nTheta1;
    CuMatrix<T> nTheta2;
    nn_parms.unconcat(nTheta1, hidden_layer_size, (input_layer_size + 1),(input_layer_size + 1),0);
	nn_parms.unconcat(nTheta2, num_labels, (hidden_layer_size + 1), (hidden_layer_size + 1), hidden_layer_size * (input_layer_size + 1));
	checkCudaError(cudaGetLastError());


    CuMatrix<T> p1 = NeuralNet<T>::predictCg(nTheta1, nTheta2, training);
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
    outln("training accuracy : " << ( res  * 100));

    T rescv = ycv.accuracy(h1cv);
	checkCudaError(cudaGetLastError());
    outln("cv accuracy : " << ( rescv  * 100));

    nnCostFtor<T> costFnTraining(input_layer_size,hidden_layer_size,num_labels, training, ytrainingBiased,lambda );
    costFnTraining(grad,cost,nn_parms);
    outln("cost on training " << cost);
    nnCostFtor<T> costFnCv(input_layer_size,hidden_layer_size,num_labels, cv, ycvBiased,lambda );
    costFnCv(grad,cost, nn_parms);
    outln("cost on cv " << cost);

    return 0;
}


template int testNeural2Loop<float>::operator()(int argc, char const ** args) const;
template int testNeural2Loop<double>::operator()(int argc, char const ** args) const;
template <typename T> int testNeural2Loop<T>::operator()(int argc, const char** args) const {
	CuTimer timer;
	map<string,float> runTimes;
	timer.start();
	int count = b_util::getCount(argc,args,5);
	outln("exp test " << b_util::expNotation(1000l) ) ;
	outln("plz test " << b_util::plz("run",count) ) ;
	float exeTime;
	outln("starting trial of " << count << " " << b_u::plz("run",count) ) ;
	void (*matProdKptr[]) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int) =  {matrixProductKernel,matrixProductKernel2,matrixProductKernel3 /*,matrixProductKernelTxdB,matrixProductKernelTxdB2*/};
	const char* names[] = {"k1 ","k1 ","k2 ","k2 ", "ktx ", "ktx2 "};
	int iterations = b_util::getParameter(argc, args, "its", 50);
	map<string, CuMatrix<T>*> f = util<T>::parseOctaveDataFile(SAMPLES_FILE,
    			false, true);
	map<string, CuMatrix<T>*> fw = util<T>::parseOctaveDataFile(WEIGHTS_FILE,
    			false, true);
	exeTime = timer.stop();
	outln(count << "loading took a total of " <<exeTime << " secs, avg t/rec " << exeTime/count);
	typedef pair<string,float> runpair;
	// DefaultMatProdBlock
	checkCudaErrors(cudaGetLastError());
	for(int kernel = 0; kernel < 3; kernel++) {
    	CuMatrix<T>::g_matrix_product_kernel = matProdKptr[kernel];
    	if(kernel & 1) {
    		CuMatrix<T>::DefaultMatProdBlock = dim3(32,32);
    	} else {
    		CuMatrix<T>::DefaultMatProdBlock = dim3(16,16);
    	}
    	outln("starting nn with kernel " << names[kernel]);
		timer.start();
		testNeuralKPtr(iterations, f,fw);
		exeTime = timer.stop();
		CuMatrix<T>::mgr->dumpLeftovers();
		runTimes.insert(runTimes.end(),runpair(names[kernel] + string(" ") + b_util::pd3(CuMatrix<T>::DefaultMatProdBlock), exeTime));
	}
    outln("results");
	typedef typename map<string, float>::iterator iterator;
	for(iterator it = runTimes.begin();it != runTimes.end();it++) {
		outln((*it).first << " took " << (*it).second << "ms");
	}
	return 0;
}

template int testCheckNNGradients<float>::operator()(int argc, char const ** args) const;
template int testCheckNNGradients<double>::operator()(int argc, char const ** args) const;
template <typename T> int testCheckNNGradients<T>::operator()(int argc, const char** args) const {
    outln("NeuralNet<T>::checkNnGradients() " << NeuralNet<T>::checkNnGradients());
    return 0;
}

#include "tests.cc"
