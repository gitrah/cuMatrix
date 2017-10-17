/* *
 *
 */
#include "NeuralNet.h"
#include "util.h"
#include "MemMgr.h"
#include "debug.h"
#include "Kernels.h"
#include "LogisticRegression.h"
#include <algorithm>

using std::for_each;

template<typename T>
T NeuralNet<T>::nnCostFunctionSansGradient(const CuMatrix<T>& nn_params,
		int input_layer_size, int hidden_layer_size, int num_labels,
		const CuMatrix<T>& xBiased, const CuMatrix<T>& yBiased, T lambda) {
	CuMatrix<T> grad(nn_params.size / sizeof(T), 1, false,true);
	T cost;
	nnCostFunction(grad,cost,nn_params, input_layer_size,
			hidden_layer_size, num_labels, xBiased, yBiased, lambda);
	return cost;
}

template<typename T>
 void NeuralNet<T>::nnCostFunction(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& nn_params,
		int input_layer_size, int hidden_layer_size, int num_labels,
		const CuMatrix<T>& xBiased, const CuMatrix<T>& yBiased, T lambda, bool colMajor) {

	if(checkDebug(debugNn | debugCg)) {
		flprintf("entre xBiased %s\n", xBiased.toss().c_str());
		flprintf("entre grad %s\n", grad.toss().c_str());
		flprintf("entre lambda %.2f\n", (double) lambda);
		flprintf("entre nn_params %s\n" , nn_params.toss().c_str());
		outln("input_layer_size " <<input_layer_size);
		outln("hidden_layer_size " <<hidden_layer_size);
		outln("num_labels " <<num_labels);
	}
 	CuMatrix<T> theta1(hidden_layer_size,input_layer_size + 1,true,true);
 	CuMatrix<T> theta2(num_labels, hidden_layer_size + 1,true,true);
	nn_params.unconcat(theta1, hidden_layer_size,input_layer_size + 1,input_layer_size + 1, 0, colMajor);
	theta1.syncBuffers();
	if(checkDebug(debugNn))if(theta1.size < (long) 100 * (int)sizeof(T))outln("theta1 " << theta1);
	if(checkDebug(debugNn| debugCg))outln("aft uncon theta1.sum " << theta1.sum());
	if(checkDebug(debugNn))outln("xBiased.sum " << xBiased.sum());

	nn_params.unconcat(theta2, num_labels, hidden_layer_size + 1, hidden_layer_size + 1,(hidden_layer_size * (input_layer_size + 1)), colMajor);
	theta2.syncBuffers();
	if(checkDebug(debugNn| debugCg))outln("aft uncon theta2.sum " << theta2.sum());
	if(checkDebug(debugNn| debugCg))outln("aft uncon xt.sum " << (xBiased.transpose()).sum());

	checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn) && theta2.size < 100 * (int)sizeof(T))outln("theta2 " << theta2);
	//if(checkDebug(debugNn))outln("theta2 col sums " << ((T)theta2.m * theta2.featureMeans(true)).syncBuffers());
	if(checkDebug(debugNn))outln("theta2.sum " << theta2.sum());

	int m = xBiased.m;

	CuTimer timer;

	//timer.start();

	if(checkDebug(debugNn))flprintf("z02 = theta1 ( %uX%u) * xBiased' (%u x %u)\n", theta1.m, theta1.n, xBiased.n, xBiased.m);
	CuMatrix<T> z02 = (theta1 * xBiased.transpose());
	if(checkDebug(debugNn))flprintf("z02  %u X %u\n", z02.m, z02.n);
	if(checkDebug(debugNn | debugCg))flprintf("z02.sum %.20g\n", (double)z02.sum());

	CuMatrix<T> z2 = z02.transpose();
	if(checkDebug(debugNn))flprintf("z2  %u X %u\n", z2.m, z2.n);
	checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn))flprintf("z2.sum %.20g\n", (double)z2.sum());

	if(checkDebug(debugNn))flprintf("a2 == ones(%u,1) | z2.sigmoid (%u x %u)\n", m, z2.m, z2.n);
	CuMatrix<T> a2 = CuMatrix<T>::ones(m, 1) |= z2.sigmoid();
	if(checkDebug(debugNn))outln("a2 " << a2.toShortString());
	checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn | debugCg))flprintf("a2.sum %.20g\n", (double)a2.sum());

	if(checkDebug(debugNn))flprintf("z3 = (theta2 ( %uX%u) * a2' (%u x %u))'\n", theta2.m, theta2.n, a2.n, a2.m);
	CuMatrix<T> z3 = (theta2 * a2.transpose()).transpose();
	if(checkDebug(debugNn))flprintf("z3 (%u X %u)\n", z3.m, z3.n);
	//float tMulti = timer.stop();

	//timer.start();

	//CuMatrix<T> z3 = (theta2 * ( CuMatrix<T>::ones(m, 1) |= (((theta1 * xBiased.transpose())).transpose()).sigmoid()).transpose()).transpose();
	//float tOne = timer.stop();

	//outln("tdelta " << ( tOne - tMulti)/ tMulti);

	//checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn))flprintf("z3.sum %.20g\n",(double) z3.sum());
	CuMatrix<T> a3 = z3.sigmoid();
	//checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn))flprintf("a3.sum %.20g\n", (double)a3.sum());
//	if(checkDebug(debugNn))outln(" a3 ss " << a3.toShortString());

	cost  = ( -1./m * (yBiased % a3.log() + (1. - yBiased) % ((1. - a3).log()))).sum();

	if(checkDebug(debugNn))flprintf("cost no reg %.20g\n",(double) cost);

	// these should be views
	CuMatrix<T> tempTheta2 = theta2.dropFirst(true).syncBuffers();

	if(checkDebug(debugNn| debugCg))flprintf("tempTheta2.sum() %.20g\n",(double) tempTheta2.sum());
	//if(checkDebug(debugNn))outln("tempTheta2 " <<tempTheta2);
	//CuMatrix<T> tempTheta2Means = tempTheta2.featureMeans(true) * ((T)theta2.m);
	//if(checkDebug(debugNn))outln("tempTheta2 col sums " << tempTheta2Means.syncBuffers());

	T jreg3 =  ((tempTheta2 ^ ((T)2)) * (lambda / (2. * m))).sum();
	if(checkDebug(debugNn| debugCg))outln("jreg3 " << jreg3);
	cost  += jreg3;
	CuMatrix<T> tempTheta1 = theta1.dropFirst();

	T jreg2 =   ((tempTheta1 ^ ((T)2)) * (lambda / (2. * m))).sum();
	if(checkDebug(debugNn| debugCg))outln("jreg2 " << jreg2);
	cost  += jreg2;
	if(checkDebug(debugNn))outln("ybiased " << yBiased.toShortString());


	CuMatrix<T> delta_3 = a3 - yBiased;
	if(checkDebug(debugNn| debugCg))outln("delta_3 " << delta_3.sum());
	if(checkDebug(debugNn))outln("z2.sigmoidGradient() " << z2.sigmoidGradient().sum());

	CuMatrix<T> delta_2 = (delta_3 * tempTheta2) % z2.sigmoidGradient();
	if(checkDebug(debugNn | debugCg))outln("delta_2 " << delta_2.sum());

	CuMatrix<T> bigDelta2 = delta_3.transpose() * a2;
	if(checkDebug(debugNn))outln("bigDelta2 " << bigDelta2.sum());
	CuMatrix<T> bigDelta1 = delta_2.transpose() * xBiased;
	if(checkDebug(debugNn))outln("bigDelta1 " << bigDelta1.sum());

	CuMatrix<T> temp = CuMatrix<T>::zeros(theta2.m, 1) |= tempTheta2;
	CuMatrix<T> theta2_grad = (bigDelta2 + (temp * lambda)) / m;

	temp = CuMatrix<T>::zeros(theta1.m, 1) |= tempTheta1;

	if(checkDebug(debugNn))outln("temp " << temp.sum());
	CuMatrix<T> theta1_grad = (bigDelta1 + (temp * lambda)) / m;

	const CuMatrix<T>* parts[] = {&theta1_grad,&theta2_grad};

	if(checkDebug(debugNn))outln("theta1_grad " << theta1_grad);
	if(checkDebug(debugNn))outln("theta2_grad " << theta2_grad);
	if(checkDebug(debugNn))flprintf("theta1_grads %.20g\n",(double)theta1_grad.sum());
	if(checkDebug(debugNn))flprintf("theta2_grads %.20g\n", (double)theta2_grad.sum());
	if(checkDebug(debugNn))outln("grad " << grad.toShortString());
	CuMatrix<T>::concat(grad, 2, parts);

	if(checkDebug(debugNn)) {
		outln("grad.elements");
		printColoArray(grad.elements, 38);
		outln("grad.currBuffer()[0]@38");
		printArray(grad.currBuffer(), 38);
	}

	if(checkDebug(debugNn))outln("grad.sum " << grad.sum());
	if(checkDebug(debugNn))b_util::dumpStack();

}

template<typename T>
 void NeuralNet<T>::nnCostFunctionN(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& nn_params,
		int input_layer_size, int hidden_layer_size[], int hidden_layer_count, int num_labels,
		const CuMatrix<T>& xBiased, const CuMatrix<T>& yBiased, T lambda, bool colMajor) {

	if(checkDebug(debugNn)) {
		outln("entre xBiased " <<xBiased.toShortString());
		outln("entre lambda " << lambda);
		outln("entre nn_params " <<nn_params);
		outln("input_layer_size " <<input_layer_size);
		outln("hidden_layer_count " <<hidden_layer_count);
		outln("num_labels " <<num_labels);
	}
 	CuMatrix<T> thetas[hidden_layer_count + 1];
 	int matOffset = 0;
 	for(int i =0; i < hidden_layer_count; i++) {
 		if(i == 0) {
 	 		thetas[i] = CuMatrix<T>(hidden_layer_size[i],input_layer_size + 1,false,true);
 			nn_params.unconcat(thetas[i], hidden_layer_size[i],input_layer_size + 1,input_layer_size + 1, 0, colMajor);
 		} else {
 	 		thetas[i] = CuMatrix<T>(hidden_layer_size[i],hidden_layer_size[i-1]+ 1,false,true);
 			nn_params.unconcat(thetas[i], hidden_layer_size[i],hidden_layer_size[i-1]+ 1,hidden_layer_size[i-1]+ 1, matOffset, colMajor);
 		}
 		thetas[i].syncBuffers();
 		matOffset += thetas[i].m * thetas[i].n;
 		if(checkDebug(debugNn))if(thetas[i].size < 100 * (long)sizeof(T))outln("thetas["<< i << "] " << thetas[i]);
 	}
	//if(checkDebug(debugNn))outln("theta1.sum " << theta1.sum());

 	thetas[hidden_layer_count] = CuMatrix<T>(num_labels, hidden_layer_size[hidden_layer_count-1] + 1,false,true);
	nn_params.unconcat(thetas[hidden_layer_count], num_labels, hidden_layer_size[hidden_layer_count-1] + 1, hidden_layer_size[hidden_layer_count-1]+ 1,matOffset, colMajor);
	thetas[hidden_layer_count].syncBuffers();

	checkCudaError(cudaGetLastError());
	//if(checkDebug(debugNn))if(thetas[hidden_layer_count].size < 100 * sizeof(T))outln("thetas[hidden_layer_count (== " + hidden_layer_count + ")] " << thetas[hidden_layer_count]);
	//if(checkDebug(debugNn))outln("theta2 col sums " << ((T)theta2.m * theta2.featureMeans(true)).syncBuffers());
	//if(checkDebug(debugNn))outln("theta2.sum " << theta2.sum());

	CuTimer timer;
	/*

	//timer.start();

 	//for(int i =0; i < hidden_layer_count; i++) {
	CuMatrix<T> z02 = (theta1 * xBiased.transpose());
	if(checkDebug(debugNn))flprintf("z02.sum %.20g\n", z02.sum());

	CuMatrix<T> z2 = z02.transpose();
	checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn))flprintf("z2.sum %.20g\n", z2.sum());

	CuMatrix<T> a2 = CuMatrix<T>::ones(m, 1) |= z2.sigmoid();
	checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn))flprintf("a2.sum %.20g\n", a2.sum());

	CuMatrix<T> z3 = (theta2 * a2.transpose()).transpose();

	//checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn))flprintf("z3.sum %.20g\n", z3.sum());
	CuMatrix<T> a3 = z3.sigmoid();
	//checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn))flprintf("a3.sum %.20g\n", a3.sum());
//	if(checkDebug(debugNn))outln(" a3 ss " << a3.toShortString());

	cost  = ( -1./m * (yBiased % a3.log() + (1. - yBiased) % ((1. - a3).log()))).sum();

	if(checkDebug(debugNn))flprintf("cost no reg %.20g\n", cost);
	//CuMatrix<T> h  = ((theta2 * a2.transpose()).transpose()).sigmoid();
	//h1 = sigmoid( (Theta2 * [ ones(m,1) sigmoid(  Theta1 * X' )]')'  )
	//
	//(theta2 *( CuMatrix<T>::ones(m,1) |= (  theta1 * X.transpose() ).sigmoid().transpose()).transpose()
	//if(checkDebug(debugNn))if(h.size < 100 * sizeof(T))outln("h " << h.syncBuffers());
	//if(checkDebug(debugNn))flprintf("h.sum %.20g\n", h.sum());

	//CuMatrix<T> h1 = (theta2 * ( CuMatrix<T>::ones(m,1) |= (  theta1 * xBiased.transpose() ).sigmoid().transpose()).transpose()).transpose().sigmoid();
	//if(checkDebug(debugNn))if(h1.size < 100 * sizeof(T))outln("h1 " << h1.syncBuffers());
	//if(checkDebug(debugNn))flprintf("h1.sum %.20g\n", h1.sum());


	// these should be views
	CuMatrix<T> tempTheta2 = theta2.dropFirst(true).syncBuffers();

	if(checkDebug(debugNn))flprintf("tempTheta2.sum() %.20g\n", tempTheta2.sum());
	if(checkDebug(debugNn))outln("tempTheta2 " <<tempTheta2);
	//CuMatrix<T> tempTheta2Means = tempTheta2.featureMeans(true) * ((T)theta2.m);
	//if(checkDebug(debugNn))outln("tempTheta2 col sums " << tempTheta2Means.syncBuffers());

	T jreg3 =  ((tempTheta2 ^ ((T)2)) * (lambda / (2. * m))).sum();
	if(checkDebug(debugNn))outln("jreg3 " << jreg3);
	cost  += jreg3;
	CuMatrix<T> tempTheta1 = theta1.dropFirst();

	T jreg2 =   ((tempTheta1 ^ ((T)2)) * (lambda / (2. * m))).sum();
	if(checkDebug(debugNn))outln("jreg2 " << jreg2);
	cost  += jreg2;
	if(checkDebug(debugNn))outln("ybiased " << yBiased.toShortString());


	outln("a3 " << a3.toShortString());

	T* pa3_currBuffer = a3.currBuffer();
	T* pyBiased = yBiased.tiler.currBuffer();

	MemMgr<T>::checkValid(pa3_currBuffer, "pa3_currBuffer");
	outln("pa3_currBuffer passed");
	MemMgr<T>::checkValid(pa3_currBuffer + a3.m * a3.n - 1, "pa3_currBuffer + m * n");
	outln("pa3_currBuffer + m * n passed");
	MemMgr<T>::checkValid(pyBiased, "pyBiased");
	outln("pyBiased passed");
	MemMgr<T>::checkValid(pyBiased+ yBiased.m * yBiased.n - 1, "pyBiased + yBiased.m * yBiased.n");
	outln("pyBiased + passed");



	CuMatrix<T> delta_3 = a3 - yBiased;
	if(checkDebug(debugNn))outln("delta_3 " << delta_3.sum());
	if(checkDebug(debugNn))outln("z2.sigmoidGradient() " << z2.sigmoidGradient().sum());

	CuMatrix<T> delta_2 = (delta_3 * tempTheta2) % z2.sigmoidGradient();
	if(checkDebug(debugNn))outln("delta_2 " << delta_2.sum());

	CuMatrix<T> bigDelta2 = delta_3.transpose() * a2;
	if(checkDebug(debugNn))outln("bigDelta2 " << bigDelta2.sum());
	CuMatrix<T> bigDelta1 = delta_2.transpose() * xBiased;
	if(checkDebug(debugNn))outln("bigDelta1 " << bigDelta1.sum());

	CuMatrix<T> temp = CuMatrix<T>::zeros(theta2.m, 1) |= tempTheta2;
	CuMatrix<T> theta2_grad = (bigDelta2 + (temp * lambda)) / m;

	temp = CuMatrix<T>::zeros(theta1.m, 1) |= tempTheta1;

	if(checkDebug(debugNn))outln("temp " << temp.sum());
	CuMatrix<T> theta1_grad = (bigDelta1 + (temp * lambda)) / m;

	const CuMatrix<T>* parts[] = {&theta1_grad,&theta2_grad};

	if(checkDebug(debugNn))outln("theta1_grad " << theta1_grad);
	if(checkDebug(debugNn))outln("theta2_grad " << theta2_grad);
	if(checkDebug(debugNn))flprintf("theta1_grads %.20g\n",theta1_grad.sum());
	if(checkDebug(debugNn))flprintf("theta2_grads %.20g\n", theta2_grad.sum());
	if(checkDebug(debugNn))outln("grad " << grad.toShortString());
	CuMatrix<T>::concat(grad, 2, parts);

	if(checkDebug(debugNn)) {
		outln("grad.elements");
		printColoArray(grad.elements, 38);
		outln("grad.currBuffer()[0]@38");
		printArray(grad.currBuffer(), 38);
	}
*/

	if(checkDebug(debugNn))outln("grad " << grad.sum());
	if(checkDebug(debugNn))b_util::dumpStack();

}

template<typename T>
T NeuralNet<T>::checkNnGradients(T lambda) {
	nnCostFunctionSanGradientOp op;
	op.input_layer_size = 3;
	op.hidden_layer_size = 5;
	op.num_labels = 3;
	int m = 5;

	// We generate some 'random' test data
	CuMatrix<T> theta1 = CuMatrix<T>::sin(op.hidden_layer_size,
			op.input_layer_size + 1, 1/10.0, 2.0 * Pi, 1).syncBuffers();
	if(checkDebug(debugNn))outln("theta1\n" << theta1);
	T theta1Sum = theta1.sum();
	printDevArray<T>(theta1.currBuffer(), __FILE__ " imm bef concat theta1 dev ", __LINE__, MAX(theta1.m,theta1.n));
	if(checkDebug(debugNn))printArray(theta1.currBuffer(), 20);
	CuMatrix<T> theta2 = CuMatrix<T>::sin(op.num_labels, op.hidden_layer_size + 1, 1/10.0, 2.0 * Pi, 1).syncBuffers();
	T theta2Sum = theta2.sum();
	if(checkDebug(debugNn))outln("theta2\n" << theta2);
	printDevArray<T>(theta1.currBuffer(), __FILE__ " imm bef concat theta2 dev ", __LINE__, MAX(theta2.m,theta2.n));
	CuMatrix<T> nn_params(1,(theta1.size + theta2.size)/sizeof(T),false,true);
	const CuMatrix<T> * parts[] = {&theta1,&theta2};
	CuMatrix<T>::concat(nn_params,2, parts);
	printDevArray<T>(nn_params.currBuffer(), __FILE__ " imm after concat nn_params dev ", __LINE__, MAX(nn_params.m,nn_params.n));
	if(!nn_params.hostReadyQ()) {
		nn_params.syncBuffers();
	}
	T nn_paramSum= nn_params.sum();
	if(checkDebug(debugNn))outln("theta1Sum " << theta1Sum);
	if(checkDebug(debugNn))outln("theta2Sum " << theta2Sum);
	if(checkDebug(debugNn))outln("nn_paramSum " << nn_paramSum);
	if(checkDebug(debugNn))outln("diff " << (nn_paramSum - theta1Sum - theta2Sum));

	assert(util<T>::almostEquals(nn_paramSum, theta1Sum + theta2Sum, util<T>::epsilon()));

	//outln("nn_params\n" << nn_params);
	//theta1.unPose();
	//theta2.unPose();
	// Reusing debugInitializeWeights to generate X
	CuMatrix<T> x = CuMatrix<T>::sin(m, op.input_layer_size, 1/10.0, 2.0 * Pi, 1).syncBuffers();
	if(checkDebug(debugNn))outln("x " << x);
	CuMatrix<T> xBiased = x.addBiasColumn();
	//if(checkDebug(debugNn))outln("xBiased " << xBiased);
	CuMatrix<T> y(m, 1,true,true);
	for (int i = 0; i < m; i++) {
		y.set(i, (i + 1) % op.num_labels + 1);
	}
	y.syncBuffers();
	//if(checkDebug(debugNn))outln("y " << y);
	//CuMatrix<T> yBiased = y.addBiasColumn();
	CuMatrix<T> yBiased = y.toBinaryCategoryMatrix();
	//if(checkDebug(debugNn))outln("yBiased " << yBiased.syncBuffers());
	T epsilon = 1e-4;
	op._x = xBiased;
	op.y = yBiased;
	op.lambda = 3;
	CuMatrix<T> grad = CuMatrix<T>::zeros(nn_params.m,nn_params.n);
	T cost;
	//if(checkDebug(debugNn))outln("calcing grad");
	nnCostFunction(grad, cost, nn_params,
			op.input_layer_size, op.hidden_layer_size, op.num_labels, xBiased, yBiased,
			lambda,true);
	if(checkDebug(debugNn))outln("gradsum " << grad.sum());
		nn_params.invalidateDevice();
	if(checkDebug(debugNn))	outln("nn_params " << nn_params.syncBuffers());
		printDevArray<T>(nn_params.currBuffer(), __FILE__ " nn_params dev ", __LINE__, MAX(nn_params.m,nn_params.n));
	CuMatrix<T> numgrad = NeuralNet<T>::gradientApprox(op, nn_params, epsilon, true);
	//if(checkDebug(debugNn))
	if(checkDebug(debugNn))outln("numgradssum " << numgrad.sum());

	if(checkDebug(debugNn))outln("\n\nnumgrad  | grad\n" << (numgrad |= grad).syncBuffers());
	//T delta = ((numgrad - grad).vectorLength() / (numgrad + grad).vectorLength());

	CuMatrix<T> m1 = numgrad - grad;
	CuMatrix<T> m2 = numgrad + grad;

	CuMatrix<T> mm1 = m1.transpose() * m1;
	CuMatrix<T> mm2 = m2.transpose() * m2;

	if(checkDebug(debugNn))outln("mm1 " << mm1.syncBuffers());

	if(checkDebug(debugNn))outln("mm2 " << mm2.syncBuffers());

	T s1 = mm1.sum();
	T s2 = mm2.sum();

	if(checkDebug(debugNn))
		outln("s1 " << s1);
	if(checkDebug(debugNn))
		outln("s2 " << s2);

	T d2 = sqrt(s1)/sqrt(s2);
	if(checkDebug(debugNn))outln("d2 " << d2);

	T delta = (numgrad - grad).norm(2) / (numgrad + grad ).norm(2);
	if(checkDebug(debugNn))outln("delta " << delta);

	if(checkDebug(debugNn))outln("grads " << grad.sum());
	if(checkDebug(debugNn))outln("delta " << delta);
	if(sizeof(T) > 4)
		assert( numgrad.almostEq(grad, 0.001));
	else
		assert( numgrad.almostEq(grad, 0.01));
	return delta;
}

// costFn: MatrixD => Double,
// todo 3.5 impl
template<typename T> template<typename CostFunction> CuMatrix<T> NeuralNet<T>::gradientApprox(
		CostFunction costFn, CuMatrix<T> theta, T epsilon, bool colMajor) {
	const uint l = theta.m * theta.n;
	long i = 0;
	CuMatrix<T> perturb = CuMatrix<T>::zeros(theta.m, theta.n);
	CuMatrix<T> gradApprox = CuMatrix<T>::zeros(theta.m, theta.n);
	T jMinus = 0, jPlus = 0;
	while (i < l) {
		perturb.set(i, epsilon);
		//outln("perturb " << perturb.syncBuffers());
		//::set(perturb.d_elements, perturb.m, perturb.n, perturb.p, i, static_cast<T>(epsilon));
		//outln("theta " << theta.toShortString());
		jMinus = costFn(theta - perturb);
		printf("jMinus %.20g\n",(double) jMinus);
		jPlus = costFn(theta + perturb);
		printf("jPlus %.20g\n", (double)jPlus);
		printf("numgrad(%ld) %.20g\n", i,(double)((jPlus - jMinus) / (2. * epsilon)));
		gradApprox.set(i, (jPlus - jMinus) / (2. * epsilon));
		//::set(gradApprox.d_elements, gradApprox.m, gradApprox.n,gradApprox.p, i, static_cast<T>((jPlus - jMinus) / (2. * epsilon)));
		perturb.set(i, 0);
		//::set(perturb.d_elements, perturb.m, perturb.n, perturb.p, i, static_cast<T>(0));
		i += 1;
	}
	if(checkDebug(debugNn)) {
		outln("gradApprox host/dev");
		printColoArray(gradApprox.elements, l);
		printArray(gradApprox.currBuffer(), l);
	}

	gradApprox.invalidateHost();
	return gradApprox;
}

template<typename T>
void NeuralNet<T>::indicesFromRange(uint*& indices, uint& count, uint start,
		uint end) {
	count = end - start + 1;
	outln("indices b " << indices);
	indices = new uint[count];
	outln("indices a " << indices);
	for (uint i = 0; i < count; i++) {
		indices[i] = start + i;
	}

}

template<typename T> void NeuralNet<T>::forwardAndBack(CuMatrix<T>& grad,
		T& cost, const CuMatrix<T>& x, const CuMatrix<T>& y,
		vector<CuMatrix<T> >& thetas, T lambda) {
	// forward
	NNPrediction<T> tupl = forward(thetas, x,y, lambda);
	cost = tupl.cost;

	uint2* dims = null;
	NeuralNet<T>::back(grad, dims, tupl, x, y, thetas, lambda);

}

template<typename T> PackedMat<T>::PackedMat(const PackedMat<T>& o) :
		layers(o.layers), owner(false), nn_params(o.nn_params) {
	uint2* tmp = new uint2[layers];
	memcpy(tmp,o.dims, layers * sizeof(uint2));
	dims = tmp;
	outln("PackedMat from " << &o << " created " << this);
}

template <typename T> PackedMat<T> PackedMat<T>::pack( const vector<CuMatrix<T> >& v) {
	const int layers = v.size();

	uint2* dims = new uint2[layers];
	typedef typename vector< CuMatrix<T> >::const_iterator vectorator;

	long size = 0;
	int idx = 0;
	for(vectorator i = v.begin(); i < v.end(); i++) {
		CuMatrix<T> m = *i;
		outln("m " << m.toShortString());
		size += m.size;
		dims[idx].x = m.m; dims[idx].y = m.n;
		idx++;
	}

	CuMatrix<T>* packed = new CuMatrix<T>(size/sizeof(T),1,false,true);

	outln("packed" << packed->toShortString());
	CuMatrix<T>::concat(*packed, v);

	return 	PackedMat<T>(packed, true, layers, dims);
}
template PackedMat<float> PackedMat<float>::pack( const vector<CuMatrix<float> >&) ;
template PackedMat<double> PackedMat<double>::pack( const vector<CuMatrix<double> >&) ;
template PackedMat<int> PackedMat<int>::pack( const vector<CuMatrix<int> >&) ;
template PackedMat<uint> PackedMat<uint>::pack( const vector<CuMatrix<uint> >&) ;
template PackedMat<long> PackedMat<long>::pack( const vector<CuMatrix<long> >&) ;
template PackedMat<ulong> PackedMat<ulong>::pack( const vector<CuMatrix<ulong> >&) ;

template<typename T> void PackedMat<T>::pack(
		CuMatrix<T>& nn_params,
		uint2*& dims,
		int& layers,
		const vector<CuMatrix<T> >& v) {
	layers = v.size();

	dims = new uint2[layers];
	typedef typename vector< CuMatrix<T> >::const_iterator vectorator;

	long size = 0;
	int idx = 0;
	for(vectorator i = v.begin(); i < v.end(); i++) {
		CuMatrix<T> m = *i;
//		outln("m " << m.toShortString());
		size += m.size;
		dims[idx].x = m.m; dims[idx].y = m.n;
		idx++;
	}

	nn_params = CuMatrix<T>::zeros(size/sizeof(T),1);

	//outln("packed" << nn_params.toShortString());
	CuMatrix<T>::concat(nn_params, v);

	//PackedMat<T>(&nn_params, true, layers, dims);
}
template void PackedMat<float>::pack(CuMatrix<float>&, uint2*&, int&, const vector<CuMatrix<float> >&) ;
template void PackedMat<double>::pack(CuMatrix<double>&, uint2*&, int&, const vector<CuMatrix<double> >&) ;
template void PackedMat<int>::pack(CuMatrix<int>&, uint2*&, int&, const vector<CuMatrix<int> >&) ;
template void PackedMat<uint>::pack(CuMatrix<uint>&, uint2*&, int&, const vector<CuMatrix<uint> >&) ;
template void PackedMat<long>::pack(CuMatrix<long>&, uint2*&, int&, const vector<CuMatrix<long> >&) ;
template void PackedMat<ulong>::pack(CuMatrix<ulong>&, uint2*&, int&, const vector<CuMatrix<ulong> >&) ;

template<typename T> void NeuralNet<T>::back(CuMatrix<T>& grad, uint2*& dims,
		NNPrediction<T>& tupl, const CuMatrix<T>& x, const CuMatrix<T>& y,
		vector<CuMatrix<T> >& thetas, T lambda) {
	int m = x.m;

	// back
	CuMatrix<T> hTheta = tupl.hTheta;
	stack<CuMatrix<T> >  zs = tupl.zs;
	stack<CuMatrix<T> >  as = tupl.as;
	CuMatrix<T> yb = y.isBinaryCategoryMatrix() ? y : y.toBinaryCategoryMatrix();

	vector< CuMatrix<T> > sigmas;
	vector< CuMatrix<T> > grads;
	if(checkDebug(debugNn))outln("last sigma hTheta " << hTheta.toShortString() );
	if(checkDebug(debugNn))outln("last sigma yb " << yb.toShortString() );
	// this is backward; needs to start from end
	// remove last z ('used' in calculation of sigma
	zs.pop();

	int thetaCount = thetas.size();
	if(checkDebug(debugNn))outln("thetaCount " << thetaCount);
	int i = thetaCount - 1;
	CuMatrix<T> curr_sigma = hTheta - yb;
	CuMatrix<T> curr_big_delta;
	CuMatrix<T> curr_theta;
	CuMatrix<T> curr_z;
	CuMatrix<T> curr_a;
	if(checkDebug(debugNn))outln("pushing sigma"<<thetaCount +1 << ": " << curr_sigma.toShortString());
	sigmas.push_back(curr_sigma);

	//  A * B has A.m, B.n dims

	//  sigma i = theta_i' * sigma(i+1) * sigGrad(z_i)
	while (i > 0) {
		curr_theta = thetas.at(i);
		if(checkDebug(debugNn))outln("theta" << i + i<< ": " << curr_theta.toss() );
		curr_z = zs.top();
		zs.pop();
		if(checkDebug(debugNn))outln("z" << i + i<< ": "  << curr_z.toShortString() );
		CuMatrix<T> temp = curr_theta.dropFirst().syncBuffers();
		if(checkDebug(debugNn))outln("temp "<< i +1 <<": " << temp.toShortString() );
		if(checkDebug(debugNn))outln("tempTheta"<< i +1 <<".sum " << temp.sum() );
		CuMatrix<T> sigmaTheta = curr_sigma * temp;
		if(checkDebug(debugNn))outln("sigmaTheta" << i + i<< ": "  << sigmaTheta.toShortString() );
		if(checkDebug(debugNn))outln("sigmaTheta" << i + i<< ".sum " << sigmaTheta.sum() );
		if(checkDebug(debugNn))outln("z" << i + i<< "siggradient.sum " << curr_z.sigmoidGradient().sum() );
		curr_sigma =  sigmaTheta  % curr_z.sigmoidGradient();
		if(checkDebug(debugNn))outln("sigma" << i + 1 << curr_sigma.toShortString() );
		if(checkDebug(debugNn))outln("sigma" << i + 1 << ".sum " << curr_sigma.sum() );
		sigmas.insert(sigmas.begin(), curr_sigma);
		i--;
	}
	i = 0;
	if(checkDebug(debugNn))outln("bigDeltas");
	as.pop();

	//vector< char > cs;
	//cs.clear();
	int currSigIdx;
	while (!sigmas.empty()) {
		// wierd that pop returns no data
		curr_sigma = sigmas.back();
		currSigIdx = sigmas.size() + 1;
		if(checkDebug(debugNn))outln("sigma " << currSigIdx << ": " << curr_sigma.toShortString() );
		if(checkDebug(debugNn))outln("sigma" << currSigIdx << ".sum " << curr_sigma.sum() );
		sigmas.erase(sigmas.end());
		if(as.size() > 0 ) {
			curr_a = as.top();
			if(checkDebug(debugNn))outln("a" << thetaCount -i << ": " << curr_a.toShortString() );
			as.pop();
		} else {
			curr_a = x;
		}
		curr_big_delta = curr_sigma.transpose() * curr_a;
		if(checkDebug(debugNn))outln("big_delta" << thetaCount -i << ": " << curr_big_delta.toShortString() );
		if(checkDebug(debugNn))outln("big_delta" << thetaCount -i << ".sum " << curr_big_delta.sum() );
		//indicesFromRange(indices, count, 1, thetas[i + 1].n - 1);/
		curr_theta = thetas.back();
		thetas.pop_back();
		if(checkDebug(debugNn))outln("theta " << i + 1 << ": " << curr_theta.toShortString() );
		CuMatrix<T> temp = CuMatrix<T>::zeros(curr_theta.m, 1) |= (curr_theta.dropFirst());
		if(checkDebug(debugNn))outln("temp " << temp.toShortString() );
		if(checkDebug(debugNn))outln("temp.sum " << temp.sum() );
		CuMatrix<T> curr_grad = ( curr_big_delta+ temp * lambda ) / m;
		if(checkDebug(debugNn))outln("theta" << thetaCount -i << "_grad " << curr_grad.toShortString() );
		if(checkDebug(debugNn))outln("theta" << thetaCount -i<< "_grad.sum " << curr_grad.sum() );

		grads.insert(grads.begin(), curr_grad);
		i++;
	}
/*
	for (i = thetaCount - 1; i > -1; i--) {
		// wierd that pop returns no data
		curr_sigma = sigmas.at(i);
		currSigIdx = i + 2;
		if(checkDebug(debugNn))outln("sigma " << currSigIdx << ": " << curr_sigma.toShortString() );
		if(checkDebug(debugNn))outln("sigma" << currSigIdx << ".sum " << curr_sigma.sum() );
		sigmas.erase(sigmas.end());
		if(as.size() > 0 ) {
			curr_a = as.top();
			if(checkDebug(debugNn))outln("a" << i+1 << ": " << curr_a.toShortString() );
			as.pop();
		} else {
			curr_a = x;
		}
		curr_big_delta = curr_sigma.transpose() * curr_a;
		if(checkDebug(debugNn))outln("big_delta" << thetaCount -i << ": " << curr_big_delta.toShortString() );
		if(checkDebug(debugNn))outln("big_delta" << thetaCount -i << ".sum " << curr_big_delta.sum() );
		//indicesFromRange(indices, count, 1, thetas[i + 1].n - 1);/
		curr_theta = thetas.back();
		thetas.pop_back();
		if(checkDebug(debugNn))outln("theta " << i + 1 << ": " << curr_theta.toShortString() );
		CuMatrix<T> temp = CuMatrix<T>::zeros(curr_theta.m, 1) |= (curr_theta.dropFirst());
		if(checkDebug(debugNn))outln("temp " << temp.toShortString() );
		if(checkDebug(debugNn))outln("temp.sum " << temp.sum() );
		CuMatrix<T> curr_grad = ( curr_big_delta+ temp * lambda ) / m;
		if(checkDebug(debugNn))outln("theta" << i + 1 << "_grad " << curr_grad.toShortString() );
		if(checkDebug(debugNn))outln("theta" << i + 1 << "_grad.sum " << curr_grad.sum() );

		grads.push_back(curr_grad);
		i++;
	}
*/
	int dummyLayers; // dummy because layers = grads.size
	PackedMat<T>::pack(grad, dims,dummyLayers, grads);
}


template<typename T> CuMatrix<T> NeuralNet<T>::predictCg(const CuMatrix<T>& theta1,
		const CuMatrix<T>& theta2, const CuMatrix<T>& xBiased) {
	int m = xBiased.m;

	CuMatrix<T> p = CuMatrix<T>::zeros(m, 1);
	CuMatrix<T> h1 = (xBiased * theta1.transpose()).sigmoid();
	CuMatrix<T> h2 = (h1.addBiasColumn() * theta2.transpose()).sigmoid();
	return h2;

}

// todo x? inputs? fix
// aka 'forward'
template<typename T>
NNPrediction<T> NeuralNet<T>::forward(const vector<CuMatrix<T> >& weights,
		const CuMatrix<T>& inputs, const CuMatrix<T>& y, T lambda) {
	CuMatrix<T> ones = CuMatrix<T>::ones(10,10);
	auto mats = {inputs, y, ones};
	map<int,long> devOx;
	bool coQ=  b_util::onCurrDeviceQ(inputs, y, ones);
	if(checkDebug(debugNn))outln("coQ " << coQ);
/*

	bool coQb = b_util::onDeviceQ( ExecCaps::currDev(), inputs, y, ones);
	if(checkDebug(debugNn))outln("coQb " << coQb);

	bool coQgen = b_util::colocatedQ(inputs, y, ones);
	if(checkDebug(debugNn))outln("coQgen " << coQgen);

	bool coQ2=  CuMatrix<T>::colocatedQ(mats);
	if(checkDebug(debugNn))outln("coQ2 " << coQ2);

	bool coQ3=  CuMatrix<T>::colocatedQ(weights);
	if(checkDebug(debugNn))outln("coQ3 " << coQ3);

	if(!coQ) {
		for_each(mats.begin(), mats.end(), [](const CuMatrix<T>& mm) {	outln(mm.toShortString()); });
		outln("b_util::devByMaxOccupancy " << coQ3);
		int maxDev = b_util::devByMaxOccupancy(devOx, inputs, y, ones);
		int destDev = ExecCaps::currDev();
		outln("migrd " << migCount << " to " << destDev);
	}

*/
/*
	int waytzOx = 	CuMatrix<T>::devByOccupancy( weights );
	if(checkDebug(debugNn))outln("CuMatrix<T>::devByOccupancy " << waytzOx);

	int onDev = y.tiler.deviceOfResidence();
	bool sameDev = true;
	//mats.for_each(  mats.begin(), mats.end(),[&sameDev,&m](){ sameDev &= m.tiler.deviceOfResidence() == onDev;});
	for( auto const &m : mats)
		[&sameDev,&m,onDev](){ sameDev &= m.tiler.deviceOfResidence() == onDev;};

	//mats.for_each(  mats.begin(), mats.end(),[&sameDev,&m](){ sameDev &= m.tiler.deviceOfResidence() == onDev;});

	if(checkDebug(debugNn))outln("enter inputs " << inputs.toShortString() << " samedev " << sameDev);
	CuMatrix<T> x = inputs.biasedQ() ? inputs : inputs.addBiasColumn();
*/
	CuMatrix<T> x = inputs; //.biasedQ() ? inputs : inputs.addBiasColumn();
	CuMatrix<T> yBiased = y.n > 1 ? y : y.toBinaryCategoryMatrix();
	uint idx = 0;
	CuMatrix<T> lastA = x;
	stack<CuMatrix<T> > zs;
	stack<CuMatrix<T> > as;
	//as.insert(as.end(),lastA);
	uint weightCount = weights.size();
	int m = inputs.m;
	if(checkDebug(debugNn))outln("current Dev " << ExecCaps::currDev() << ", weightCount " << weightCount);
	for(auto const& weight : weights) {
		if(checkDebug(debugNn))outln("idx " << idx+1);
		if(checkDebug(debugNn))outln("b_util::onCurrDeviceQ(weight, x, yBiased) " << b_util::onCurrDeviceQ(weight, x, yBiased));
		if(checkDebug(debugNn))outln("theta" << idx+1 << ".sum " << weight.sum());
/*
		bool lastaABiasedQ = lastA.biasedQ();
		lastA = lastaABiasedQ ? lastA : lastA.addBiasColumn();
*/
		CuMatrix<T> z = (weight * lastA.transpose()).transpose();
		if(checkDebug(debugNn))flprintf("z%d (%u x %u) = theta%d ( %uX%u) * xBiased' (%u x %u)\n",  idx+2, z.m, z.n, idx+1, weight.m, weight.n, x.n, x.m);
		if(checkDebug(debugNn))outln("z" << idx+2 << ".sum " << z.sum());
		if(checkDebug(debugNn))outln("zs end was " << (zs.size() -1));
		//zs.insert(zs.end(), z);
		zs.push(z);
		if(checkDebug(debugNn))outln("zs end aftins is " << (zs.size() -1));
		//lastA = zs[idx].sigmoid();
		lastA = z.sigmoid();
		if (idx < weightCount - 1) {
			lastA = lastA.addBiasColumn();
			if(checkDebug(debugNn))outln("added bias to a"<< idx+2);
		}
		if(checkDebug(debugNn))outln("(a"<< idx+2<< ") (=z) " << lastA.toShortString() << "\na"<<idx+2<< ".sum " << lastA.sum());
		as.push(lastA);
		idx += 1;
		//as.insert(as.end(), lastA);
	}
	if(checkDebug(debugNn))outln("done, computing cost");
//	cost  = ( -1./m * (yBiased % a3.log() + (1. - yBiased) % ((1. - a3).log()))).sum();

	//T cost  = ( -1./lastA.m * (yBiased % lastA.log() + (1. - yBiased) % ((1. - lastA).log()))).sum();
	T j  = ( -1./m * (yBiased % lastA.log() + (1. - yBiased) % ((1. - lastA).log()))).sum();
	T jreg = LogisticRegression<T>::costReg(m, lambda, weights);
	if(checkDebug(debugNn))outln("j " << j << ", jreg " << jreg);

	return NNPrediction<T>(lastA, j + jreg, zs, as);

}
template<typename T>
NNPrediction<T> NeuralNet<T>::forward(PackedMat<T> pm,
		 const CuMatrix<T>& inputs, const CuMatrix<T>& y) {
	if(checkDebug(debugNn))outln("enter");
	CuMatrix<T> x = inputs.biasedQ() ? inputs : inputs.addBiasColumn();
	CuMatrix<T> yBiased = y.n > 1 ? y : y.toBinaryCategoryMatrix();
	uint idx = 0;
	CuMatrix<T> lastA = x;
	stack<CuMatrix<T> > zs;
	stack<CuMatrix<T> > as;

	vector<CuMatrix<T> > v_nn_params;


	// unconcat(CuMatrix<T>& v, int rows, int cols, int pitch, uint offset, bool colMajor)
	//	nn_params.unconcat(theta1, hidden_layer_size,input_layer_size + 1,input_layer_size + 1, 0, colMajor);
	uint offset = 0;
	for(int i =0; i < pm.layers; i++ ) {
		CuMatrix<T> layer(pm.dims[i].x, pm.dims[i].y,false,true);
		pm.nn_params->unconcat(layer , pm.dims[i].x, pm.dims[i].y , pm.dims[i].y, offset, false);
		if(checkDebug(debugNn))outln("adding layer " << i << ": " << layer.toShortString());
		v_nn_params.push_back(layer);
		offset += pm.dims[i].x * pm.dims[i].y;
	}
	//as.insert(as.end(),lastA);
	uint weightCount = v_nn_params.size();
	if(checkDebug(debugNn))outln("weightCount " << weightCount);
	typedef typename vector< CuMatrix<T> >::iterator vecterator;
	for(vecterator vecti = v_nn_params.begin(); vecti != v_nn_params.end(); vecti++) {
		//if(checkDebug(debugNn))outln("idx " << idx+1);
		CuMatrix<T> weight = (*vecti);
		lastA = lastA.biasedQ() ? lastA : lastA.addBiasColumn();
		if(checkDebug(debugNn))outln("idx " << idx << " weight " << weight.toShortString());
		if(checkDebug(debugNn))outln("idx " << idx << " lastA " << lastA.toShortString());
		CuMatrix<T> z = (weight * lastA.transpose()).transpose();
		if(checkDebug(debugNn))outln("idx " << idx << " z " << z.toShortString());
		if(checkDebug(debugNn))outln("zs end was " << (zs.size() -1));
		//zs.insert(zs.end(), z);
		zs.push(z);
		if(checkDebug(debugNn))outln("zs end aftins is " << (zs.size() -1));
		//lastA = zs[idx].sigmoid();
		lastA = z.sigmoid();
		if (idx < weightCount - 1) {
			lastA = lastA.addBiasColumn();
			if(checkDebug(debugNn))outln("added bias to lastA");
		}
		if(checkDebug(debugNn))outln("lastA (=z) " << lastA.toShortString());
		as.push(lastA);
		idx += 1;
		//as.insert(as.end(), lastA);
	}
	T cost  = ( -1./lastA.m * (yBiased % lastA.log() + (1. - yBiased) % ((1. - lastA).log()))).sum();
	return NNPrediction<T>(lastA, cost, zs, as);
}

template<typename T> void NeuralNet<T>::forwardAndBack(CuMatrix<T>& grad,
		T& cost, const CuMatrix<T>& x, const CuMatrix<T>& y,
		const CuMatrix<T>& nn_params, int layers, const uint2* dims, T lambda) {
	vector<CuMatrix<T> > v_nn_params;

	if(checkDebug(debugNn))outln("nn_params-> " << nn_params.toShortString());
	if(checkDebug(debugNn))outln("layers " <<   layers);

	uint offset = 0;
	for(int i =0; i < layers; i++ ) {
		CuMatrix<T> layer(dims[i].x, dims[i].y, false,true );
		if(checkDebug(debugNn))outln("i " << i << ": dims[i].x " << dims[i].x << ", dims[i].y " << dims[i].y << ", offset " << offset);

		//outln("layer " << i << ": " << layer.toShortString());
		nn_params.unconcat(layer , dims[i].x, dims[i].y , dims[i].y, offset, false);
		if(checkDebug(debugNn))outln("adding layer " << i << ": " << layer.toShortString());
		if(checkDebug(debugNn))outln("layer " << i << ".sum()  " << layer.sum());
		v_nn_params.push_back(layer);
		offset += dims[i].x * dims[i].y;
	}
	uint weightCount = v_nn_params.size();
	if(checkDebug(debugNn))outln("weightCount " << weightCount);

	NNPrediction<T> tupl = forward(v_nn_params, x, y, lambda);
	cost = tupl.cost;
	uint2* dummyDims = null;
	NeuralNet<T>::back(grad, dummyDims, tupl, x, y, v_nn_params, lambda);
}


template<typename T, template<typename > class OutT> void nnPermUtil<T, OutT >::mapPermutations(list<OutT<T>>&  retVals, nnPermUtil::permFunction fn, const NnRunInfo<T>& nnri, list<list<uint> > listOfLists, list<uint> instance) {

	if (!listOfLists.size()) {
		outln("listOfLists empty, calling fn " );
		retVals.push_back( fn( instance, nnri));
		return;
	}
	outln("listOfLists.size() " << listOfLists.size());
	list<uint> currentList = listOfLists.front();
	listOfLists.pop_front();

	for(typename list<uint>::iterator i = currentList.begin(); i != currentList.end(); i++) {
		instance.push_back(*i);
		outln("pushed " << instance.back());
		mapPermutations(retVals, fn, nnri, listOfLists, instance); //recursively invoking with a "smaller" problem
		instance.pop_back();
	}
	listOfLists.push_front(currentList);
}
template void nnPermUtil<float, CuMatrix>::mapPermutations(std::list<CuMatrix<float>, std::allocator<CuMatrix<float> > >&, CuMatrix<float> (*)(std::list<unsigned int, std::allocator<unsigned int> >, NnRunInfo<float> const&), NnRunInfo<float> const&, std::list<std::list<unsigned int, std::allocator<unsigned int> >, std::allocator<std::list<unsigned int, std::allocator<unsigned int> > > >, std::list<unsigned int, std::allocator<unsigned int> >);

template void nnPermUtil<double, CuMatrix>::mapPermutations(std::list<CuMatrix<double>, std::allocator<CuMatrix<double> > >&, CuMatrix<double> (*)(std::list<unsigned int, std::allocator<unsigned int> >, NnRunInfo<double> const&), NnRunInfo<double> const&, std::list<std::list<unsigned int, std::allocator<unsigned int> >, std::allocator<std::list<unsigned int, std::allocator<unsigned int> > > >, std::list<unsigned int, std::allocator<unsigned int> >);

template void nnPermUtil<unsigned long, CuMatrix>::mapPermutations(std::list<CuMatrix<unsigned long>, std::allocator<CuMatrix<unsigned long> > >&, CuMatrix<unsigned long> (*)(std::list<unsigned int, std::allocator<unsigned int> >, NnRunInfo<unsigned long> const&), NnRunInfo<unsigned long> const&, std::list<std::list<unsigned int, std::allocator<unsigned int> >, std::allocator<std::list<unsigned int, std::allocator<unsigned int> > > >, std::list<unsigned int, std::allocator<unsigned int> >);



template class NeuralNet<float>;
template class NeuralNet<double>;
template class NeuralNet<ulong>;
