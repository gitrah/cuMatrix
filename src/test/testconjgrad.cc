/*
 * testconjgrad.cc
 *
 *  Created on: Sep 6, 2012
 *      Author: reid
 */
#include "../Matrix.h"
#include "../util.h"
#include "../AnomalyDetection.h"

template <typename T> int testConjGrad(int argc, char** args) {
	std::map<std::string, Matrix<T>*> f= util<T>::parseOctaveDataFile("ex4data1.txt",false, true);
	std::map<std::string, Matrix<T>*> fw= util<T>::parseOctaveDataFile("ex4weights.txt",false, true);


	std::cout << "found " << f.size() << " octave objects\n";
	typedef typename std::map<std::string, Matrix<T>*>::iterator iterator;
	iterator it;
	it = f.begin();

	Matrix<T>& x = *f["X"];
	outln("load x of " << x.m << "x" << x.n);
	Matrix<T>& y = *f["y"];
	outln("got y " << y.toShortString());
	Matrix<T>& theta1 = *fw["Theta1"];
	outln("got theta1 " << theta1.toShortString());
	Matrix<T>& theta2 = *fw["Theta2"];
	outln("got theta2 " << theta2.toShortString());

    int m = x.m;
    int n= x.n;
    Matrix<T> mus = x.featureMeans(true);

    int input_layer_size = 400; // 20x20 Input Images of Digits
    int hidden_layer_size = 25; //   25 hidden units
    int num_labels = 10; // 10 labels, from 1 to 10
    uint nfeatures = theta1.n;
    dassert(num_labels == nfeatures); // feature mismatch

    Matrix<T> thetas = theta1.poseAsRow() |= theta2.poseAsRow();
    theta1.unPose();
    theta2.unPose();
    T lambda = 0;
/*    var (j, grad) = nnCostFunction(thetas, input_layer_size, hidden_layer_size, num_labels, x, y, lambda)

    println("cost " + j)
    lambda = 1f
    val tup = nnCostFunction(thetas, input_layer_size, hidden_layer_size, num_labels, x, y, lambda)
    println("cost (lambada = 1) " + tup._1)

    val sigTest1 = new MatrixF(Array(1, -.5f, 0, .5f, 1), 5)
    val sigTest2 = MatrixF.randn(5000, 500, 25)
    var g1: MatrixF = null
    var g2: MatrixF = null
    var gdc1: MatrixF = null
    var gdc2: MatrixF = null
    time("sigmoidGradient1", g1 = sigmoidGradient(sigTest1), 5000)
    // time("sigmoidGradient2", g2= sigmoidGradient(sigTest2),100)

    time("sigmoidGradientDc1", gdc1 = sigmoidGradientDc(sigTest1), 5000)
    //time("sigmoidGradientDc2", gdc2= sigmoidGradientDc(sigTest2),100)

    val initial_Theta1 = MatrixF.randn(hidden_layer_size, input_layer_size).addBiasCol()
    val initial_Theta2 = MatrixF.randn(num_labels, hidden_layer_size).addBiasCol()

    // Unroll parameters
    val initial_nn_params = initial_Theta1.poseAsCol +/ initial_Theta2.poseAsCol
    initial_Theta1.unPose
    initial_Theta2.unPose

    checkNnGradients()

    lambda = 3
    checkNnGradients(lambda)

    // def fmincg(f: (MatrixF) => (Float, MatrixF), xin: MatrixF, length: Int = 100, red: Int = 1) {
    var tupcg: (MatrixF, MatrixF, Int) = null
    time("fmincg 50", tupcg = fmincg(nnCostFunction(_, input_layer_size, hidden_layer_size, num_labels, x, y, lambda), initial_nn_params))
    val nn_parms = tupcg._1

    val nn_j = tupcg._2

    val nTheta1 = nn_parms.reshape(hidden_layer_size, (input_layer_size + 1))
    val nTheta2 = nn_parms.reshape(num_labels, (hidden_layer_size + 1), hidden_layer_size * (input_layer_size + 1))
    val p1 = predictCg(nTheta1, nTheta2, x)
    val h1 = p1.maxColIdxs + 1
    val res = (y - h1).elementOpDc(x => if (x == 0d) 1 else 0).featureAverages
    println("accuracy : " + res(0) * 100)
    val p2 = predict(Array(nTheta1, nTheta2), x)*/
  //}
}
