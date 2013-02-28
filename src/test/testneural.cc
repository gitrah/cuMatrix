/*
 * testneural.cc
 *
 *  Created on: Oct 15, 2012
 *      Author: reid
 */


#include "../NeuralNet.h"
#include "../ConjugateGradient.h"
#include "../debug.h"
#include "../util.h"
#include "tests.h"

const char* SAMPLES_FILE = "ex4data1.txt";
const char* WEIGHTS_FILE = "ex4weights.txt";
extern template class  ConjugateGradient<float>;
extern template class  ConjugateGradient<double>;


template int testNeural2l<float>::operator()(int argc, char const ** args) const;
template int testNeural2l<double>::operator()(int argc, char const ** args) const;
template <typename T> int testNeural2l<T>::operator()(int argc, const char** args) const {

	int iterations = b_util::getParameter(argc, args, "its", 50);

	map<string, Matrix<T>*> f = util<T>::parseOctaveDataFile(SAMPLES_FILE,
    			false, true);
	map<string, Matrix<T>*> fw = util<T>::parseOctaveDataFile(WEIGHTS_FILE,
    			false, true);
	if(!f.size()) {
		outln("no " << SAMPLES_FILE << "; exiting");
		return -1;
	}
	if(!fw.size()) {
		outln("no " << WEIGHTS_FILE << "; exiting");
		return -1;
	}
	//Matrix<T>::verbose = true;
    Matrix<T>* x = f["X"];
    x->syncBuffers();
    //outln("x from file " << *x);
    //outln("0x " << x->sum());
    Matrix<T>* y = f["y"];
    y->syncBuffers();
    outln("y " << y->toShortString());

    Matrix<T>* theta1 = fw["Theta1"];
    Matrix<T>* theta2 = fw["Theta2"];
    theta1->syncBuffers();
    theta2->syncBuffers();
    outln("theta1.sum " << theta1->sum());
    outln("theta2.sum " << theta2->sum());
    Matrix<T> mus = x->featureMeans(true);
    Matrix<T> yBiased = y->toBinaryCategoryMatrix();

    uint input_layer_size = 400; // 20x20 Input Images of Digits
    uint hidden_layer_size = 25; //   25 hidden units
    uint num_labels = 10; // 10 labels, from 1 to 10
    const Matrix<T>* thetaParts[] = {theta1, theta2};
    Matrix<T> thetas(1, (theta1->size + theta2->size)/sizeof(T),false,true);
    outln("created thetas " << thetas.toShortString());
    Matrix<T>::concat(thetas, 2, thetaParts);
   // outln("thetas.sum " << thetas.sum());
    T lambda = 0;


    Matrix<T> xBiased = x->addBiasColumn();
	Matrix<T> training, cv;
	vector<uint> vIndices;
	xBiased.shuffle(training,cv,(T).75,vIndices);
	Matrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	y->shuffle(ytraining,ycv,(T).75,vIndices);
	yBiased.shuffle(ytrainingBiased,ycvBiased,(T).75,vIndices);

    nnCostFtor<T> costFn(input_layer_size,hidden_layer_size,num_labels, training, ytrainingBiased,lambda );
    outln("explicit costFn(thetas)");
    Matrix<T> grad(1,thetas.size/sizeof(T),false,true);
    outln("created grad " << grad.toShortString());
    T cost;
    costFn(grad,cost,thetas);
    outln("cost " << cost);
    costFn.lambda = 1;
    costFn(grad,cost, thetas);
    outln("cost lambada = 1 " << cost);

    T sigArry[] = {1, -.5, 0, .5, 1};
    Matrix<T> sigTest1(sigArry, 5, 1, true);
    Matrix<T>  sigTest2 = Matrix<T>::randn(50, 50, 25);
    DMatrix<T> dm;
    sigTest2.asDmatrix(dm);
    sigTest2.invalidateHost();
    outln("sigTest2 " << sigTest2.toShortString());

    //util<T>::timeReps(&Matrix<T>::sigmoidGradient,"sigmoidGradient", &sigTest2, 10000);

    bool check = false;
    lambda = 3;
    if(check) {
    	NeuralNet<T>::checkNnGradients(0);
        NeuralNet<T>::checkNnGradients(lambda);
    }

    //Matrix<T> initial_Theta1 = Matrix<T>::randn(hidden_layer_size, input_layer_size).addBiasColumn();
    //Matrix<T> initial_Theta2 = Matrix<T>::randn(num_labels, hidden_layer_size).addBiasColumn();
    Matrix<T> initial_Theta1 = Matrix<T>::sin(hidden_layer_size, input_layer_size).addBiasColumn();
    Matrix<T> initial_Theta2 = Matrix<T>::cos(num_labels, hidden_layer_size).addBiasColumn();
    const Matrix<T>* parts[] = {&initial_Theta1, &initial_Theta2};
    Matrix<T> initial_nn_params(1,(initial_Theta1.size + initial_Theta2.size)/sizeof(T),false,true);
    Matrix<T>::concat(initial_nn_params, 2, parts);

    nnCostFtor<T> ftor(input_layer_size, hidden_layer_size, num_labels, training, ytrainingBiased, lambda);

    ConjugateGradient<T>::init();
    ftor.verbose=true;
    outln("post init last err " << b_util::lastErrStr());

    CuTimer justFmincg;
    justFmincg.start();
    pair<Matrix<T>, pair<Matrix<T>, int > > tup2 = ConjugateGradient<T>::fmincg(ftor,initial_nn_params, iterations);
    outln("back from fmincg, took " << justFmincg.stop());

    Matrix<T> nn_parms = tup2.first;

    Matrix<T> nTheta1;
    Matrix<T> nTheta2;
    nn_parms.unconcat(nTheta1, hidden_layer_size, (input_layer_size + 1),(input_layer_size + 1),0);
	nn_parms.unconcat(nTheta2, num_labels, (hidden_layer_size + 1), (hidden_layer_size + 1), hidden_layer_size * (input_layer_size + 1));


    Matrix<T> p1 = NeuralNet<T>::predictCg(nTheta1, nTheta2, training);
    Matrix<T> h1 = p1.toMaxColumnIndexVector() + 1;
    Matrix<T> p1cv = NeuralNet<T>::predictCg(nTheta1, nTheta2, cv);
    Matrix<T> h1cv = p1cv.toMaxColumnIndexVector() + 1;
    T res = ytraining.accuracy(h1);
    outln("training accuracy : " << ( res  * 100));

    T rescv = ycv.accuracy(h1cv);
    outln("cv accuracy : " << ( rescv  * 100));

    nnCostFtor<T> costFnTraining(input_layer_size,hidden_layer_size,num_labels, training, ytrainingBiased,lambda );
    costFnTraining(grad,cost,nn_parms);
    outln("cost on training " << cost);
    nnCostFtor<T> costFnCv(input_layer_size,hidden_layer_size,num_labels, cv, ycvBiased,lambda );
    costFnCv(grad,cost, nn_parms);
    outln("cost on cv " << cost);


    util<Matrix<T> >::deleteMap(f);
    util<Matrix<T> >::deleteMap(fw);
    return 0;
}

template int testCheckNNGradients<float>::operator()(int argc, char const ** args) const;
template int testCheckNNGradients<double>::operator()(int argc, char const ** args) const;
template <typename T> int testCheckNNGradients<T>::operator()(int argc, const char** args) const {
    outln("NeuralNet<T>::checkNnGradients() " << NeuralNet<T>::checkNnGradients());
    return 0;
}

#include "tests.cc"
