/* *
 *
 */
#include "NeuralNet.h"
#include "debug.h"
#include "Kernels.h"
#include "LogisticRegression.h"

template<typename T>
T NeuralNet<T>::nnCostFunctionSansGradient(const CuMatrix<T>& nn_params,
		int input_layer_size, int hidden_layer_size, int num_labels,
		const CuMatrix<T>& xBiased, const CuMatrix<T>& yBiased, T lambda) {
	CuMatrix<T> grad(1,nn_params.size / sizeof(T), false,true);
	T cost;
	nnCostFunction(grad,cost,nn_params, input_layer_size,
			hidden_layer_size, num_labels, xBiased, yBiased, lambda);
	return cost;
}

template<typename T>
 void NeuralNet<T>::nnCostFunction(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& nn_params,
		int input_layer_size, int hidden_layer_size, int num_labels,
		const CuMatrix<T>& xBiased, const CuMatrix<T>& yBiased, T lambda) {
	checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn))outln("entre xBiased " <<xBiased.toShortString());
	if(checkDebug(debugNn))outln("entre nn_params " <<nn_params.toShortString());
 	CuMatrix<T> theta1, theta2;
	nn_params.unconcat(theta1, hidden_layer_size,input_layer_size + 1,input_layer_size + 1, 0);
	checkCudaError(cudaGetLastError());
	//outln("theta1 " << theta1.toShortString());
	if(checkDebug(debugNn))outln("theta1.sum " << theta1.sum());
	//outln("nn_params " << nn_params);
	//outln("theta1.sum " << theta1.sum());
	nn_params.unconcat(theta2, num_labels, hidden_layer_size + 1, hidden_layer_size + 1,(hidden_layer_size * (input_layer_size + 1)));
	checkCudaError(cudaGetLastError());
	if(checkDebug(debugNn))outln("theta2.sum " << theta2.sum());
	uint m = xBiased.m;
	//CuMatrix<T> yb = y.toBinaryCategoryMatrix();
	CuMatrix<T> z2 = (theta1 * xBiased.transpose()).transpose();
	checkCudaError(cudaGetLastError());
	//if(checkDebug(debugNn))outln("z2.sum " << z2.sum());
	CuMatrix<T> a2 = CuMatrix<T>::ones(m, 1) |= z2.sigmoid();
	checkCudaError(cudaGetLastError());
	//if(checkDebug(debugNn))outln("a2.sum " << a2.sum());
	CuMatrix<T> z3 = (theta2 * a2.transpose()).transpose();
	checkCudaError(cudaGetLastError());
	//if(checkDebug(debugNn))outln("z3.sum " << z3.sum());
	CuMatrix<T> a3 = z3.sigmoid();
	checkCudaError(cudaGetLastError());
	//if(checkDebug(debugNn))outln("a3.sum " << a3.sum());
	cost  = ( -1./m * (yBiased % a3.log() + (1. - yBiased) % ((1. - a3).log()))).sum();
	if(checkDebug(debugNn))outln("j " << cost);
	checkCudaError(cudaGetLastError());
	// these should be views
	CuMatrix<T> tempTheta2 = theta2.dropFirst();
	if(checkDebug(debugNn))outln("tempTheta2 " <<tempTheta2.syncBuffers());
	//CuMatrix<T> tempTheta2;
	//theta2.submatrix(tempTheta2,theta2.m, theta2.n-1,0,1);
	//tempTheta2.syncBuffers();
	//outln("tempTheta2.sum "  << tempTheta2.sum());
	T jreg3 =  ((tempTheta2 ^ ((T)2)) * (lambda / (2. * m))).sum();
	cost  += jreg3;
	if(checkDebug(debugNn))outln("jreg3 " << jreg3);
	CuMatrix<T> tempTheta1 = theta1.dropFirst();
	//CuMatrix<T> tempTheta1;
	//theta1.submatrix(tempTheta1,theta1.m, theta1.n-1,0,1);

	//tempTheta1.syncBuffers();
	//outln("tempTheta1.sum "  << tempTheta1.sum());
	T jreg2 =   ((tempTheta1 ^ ((T)2)) * (lambda / (2. * m))).sum();
	if(checkDebug(debugNn))outln("jreg2 " << jreg2);
	cost  += jreg2;
	CuMatrix<T> delta_3 = a3 - yBiased;
	CuMatrix<T> delta_2 = (delta_3 * tempTheta2) % z2.sigmoidGradient();
	CuMatrix<T> bigDelta2 = delta_3.transpose() * a2;
	CuMatrix<T> bigDelta1 = delta_2.transpose() * xBiased;

	CuMatrix<T> temp = CuMatrix<T>::zeros(theta2.m, 1) |= tempTheta2;
	CuMatrix<T> theta2_grad = (bigDelta2 + (temp * lambda)) / m;
	temp = CuMatrix<T>::zeros(theta1.m, 1) |= tempTheta1;
	CuMatrix<T> theta1_grad = (bigDelta1 + lambda * temp) / m;
	//CuMatrix<T> grad = theta1_grad.poseAsCol() /= theta2_grad.poseAsCol();
	const CuMatrix<T>* parts[] = {&theta1_grad,&theta2_grad};
	//CuMatrix<T> grad = CuMatrix<T>::concat(2, parts);
	CuMatrix<T>::concat(grad, 2, parts);
	//theta1.unPose();
	//theta2.unPose();
	//if(checkDebug(debugNn))outln("grad.sum " << grad.sum());
	if(checkDebug(debugNn))outln("grad " << grad.toShortString());
	//return std::pair<T, CuMatrix<T> >(j, grad);
}

template<typename T>
T NeuralNet<T>::checkNnGradients(T lambda) {
	nnCostFunctionSanGradientOp op;
	op.input_layer_size = 3;
	op.hidden_layer_size = 5;
	op.num_labels = 3;
	uint m = 5;

	// We generate some 'random' test data
	CuMatrix<T> theta1 = CuMatrix<T>::sin(op.hidden_layer_size,
			op.input_layer_size + 1, false).syncBuffers();
	//outln("theta1\n" << theta1);
	CuMatrix<T> theta2 = CuMatrix<T>::sin(op.num_labels, op.hidden_layer_size + 1,
			false).syncBuffers();
	//outln("theta2\n" << theta2);
	CuMatrix<T> nn_params(1,(theta1.size + theta2.size)/sizeof(T),false,true);
	const CuMatrix<T> * parts[] = {&theta1,&theta2};
	 CuMatrix<T>::concat(nn_params,2, parts);
	//outln("nn_params\n" << nn_params);
	//theta1.unPose();
	//theta2.unPose();
	// Reusing debugInitializeWeights to generate X
	CuMatrix<T> x = CuMatrix<T>::sin(m, op.input_layer_size, false).syncBuffers();
	CuMatrix<T> xBiased = x.addBiasColumn();
	CuMatrix<T> y(m, 1,true,true);
	for (uint i = 0; i < m; i++) {
		y.set(i, (i + 1) % op.num_labels + 1);
	}
	y.syncBuffers();
	CuMatrix<T> yBiased = y.addBiasColumn();

	T epsilon = 1e-4;
	op._x = x;
	op.y = y;
	op.lambda = 3;
	CuMatrix<T> grad(nn_params.m,nn_params.n, false,true);
	T cost;
	nnCostFunction(grad, cost, nn_params,
			op.input_layer_size, op.hidden_layer_size, op.num_labels, xBiased, yBiased,
			lambda);
	CuMatrix<T> numgrad = NeuralNet<T>::gradientApprox(op, nn_params, epsilon);
	//outln("\n\ngrad  | numgrad\n" << (grad |= numgrad));
/*
	CuMatrix<T> ngtheta1 = numgrad.reshape(op.hidden_layer_size,
			op.input_layer_size + 1, 0);
	CuMatrix<T> ngtheta2 = numgrad.reshape(op.num_labels, op.hidden_layer_size + 1,
			(op.hidden_layer_size * (op.input_layer_size + 1)));
	CuMatrix<T> gtheta1 = grad.reshape(op.hidden_layer_size,
			op.input_layer_size + 1, 0);
	CuMatrix<T> gtheta2 = grad.reshape(op.num_labels, op.hidden_layer_size + 1,
			(op.hidden_layer_size * (op.input_layer_size + 1)));
*/
	T costApprox = ((numgrad - grad).vectorLength()
			/ (numgrad + grad).vectorLength());
	return costApprox;
}

// costFn: MatrixD => Double,
// todo 3.5 impl
template<typename T> template<typename CostFunction> CuMatrix<T> NeuralNet<T>::gradientApprox(
		CostFunction costFn, CuMatrix<T> theta, T epsilon) {
	const uint l = theta.m * theta.n;
	ulong i = 0;
	CuMatrix<T> perturb = CuMatrix<T>::zeros(theta.m, theta.n);
	CuMatrix<T> gradApprox = CuMatrix<T>::zeros(theta.m, theta.n);
	T jMinus = 0, jPlus = 0;
	while (i < l) {
		//perturb.set(i, epsilon);
		::set(perturb.d_elements, perturb.m, perturb.n, perturb.p, i, static_cast<T>(epsilon));
		jMinus = costFn(theta - perturb);
		jPlus = costFn(theta + perturb);
		//gradApprox.set(i, (jPlus - jMinus) / (2. * epsilon));
		::set(gradApprox.d_elements, gradApprox.m, gradApprox.n,gradApprox.p, i, static_cast<T>((jPlus - jMinus) / (2. * epsilon)));
		//perturb.set(i, 0);
		::set(perturb.d_elements, perturb.m, perturb.n, perturb.p, i, static_cast<T>(0));
		i += 1;
	}
	return gradApprox;
}

template<typename T>
void NeuralNet<T>::indicesFromRange(uint** indices, uint& count, uint start,
		uint end) {
	const uint l = end - start + 1;
	*indices = new uint[l];
	for (uint i = 0; i < l; i++) {
		*indices[i] = start + i;
	}
}

template<typename T> pair<T, vector<CuMatrix<T> > > NeuralNet<T>::forwardAndBack(CuMatrix<T>& x, CuMatrix<T>& y, vector<CuMatrix<T> > thetas, T lambda) {
	uint m = x.m;
	NNPrediction<T> tupl = predict(thetas, x);
	CuMatrix<T> hTheta = tupl.hTheta;
	vector<CuMatrix<T> >  zs = tupl.zs;
	vector<CuMatrix<T> >  as = tupl.as;
	CuMatrix<T> yb = y.isBinaryCategoryMatrix() ? y : y.toBinaryCategoryMatrix();

	T j = LogisticRegression<T>::costFunction(hTheta, yb, lambda, thetas);
	vector< CuMatrix<T> > deltas;
	vector< CuMatrix<T> > bigDeltas;
	vector< CuMatrix<T> > grads;
	deltas.insert(deltas.end(), hTheta - yb);
	int i = 0;
	uint* indices;
	int thetaCount = thetas.size();
	uint count;
	while (i < thetaCount - 1) {
		indicesFromRange(&indices, count, 1, thetas[i + 1].n - 1);
		CuMatrix<T> theta = thetas.at(i + 1).columnSubset(indices, count);
		delete[] indices;
		deltas[i] = (deltas[i + 1] * theta)
				% (zs[i].sigmoidGradient().transpose());
		i++;
	}
	i = 0;
	while (i < thetaCount) {
		bigDeltas[i] = deltas[i].transpose() * as[i];
		indicesFromRange(&indices, count, 1, thetas[i + 1].n - 1);
		CuMatrix<T> theta = CuMatrix<T>::zeros(thetas.at(i).m, 1) |= (thetas.at(i).columnSubset(indices, count));
		delete[] indices;
		grads[i] = (bigDeltas[i] + theta * lambda) / m;
		i++;
	}
	return pair<T, vector<CuMatrix<T> > >(j, grads);

}

template<typename T> CuMatrix<T> NeuralNet<T>::predictCg(const CuMatrix<T>& theta1,
		const CuMatrix<T>& theta2, const CuMatrix<T>& xBiased) {
	uint m = xBiased.m;

	CuMatrix<T> p = CuMatrix<T>::zeros(m, 1);
	CuMatrix<T> h1 = (xBiased * theta1.transpose()).sigmoid();
	CuMatrix<T> h2 = (h1.addBiasColumn() * theta2.transpose()).sigmoid();
	return h2;

}

// todo x? inputs? fix
template<typename T>
NNPrediction<T> NeuralNet<T>::predict(vector<CuMatrix<T> >weights,
		CuMatrix<T> inputs) {
	CuMatrix<T> x = inputs.biasedQ() ? inputs : inputs.addBiasColumn();
	uint idx = 0;
	CuMatrix<T> lastA = zerom;
	vector<CuMatrix<T> > zs;
	vector<CuMatrix<T> > as;
	as.insert(as.end(),lastA);
	uint weightCount = weights.size();
	typedef typename vector< CuMatrix<T> >::iterator vecterator;
	for(vecterator vecti = weights.begin(); vecti != weights.end(); vecti++) {
		zs.insert(zs.end(), weights[idx] * lastA.transpose());
		lastA = zs[idx].sigmoid().transpose();
		if (idx < weightCount - 1)
			lastA = lastA.addBiasColumn();
		idx += 1;
		as.insert(as.end(), lastA);
	}
	return NNPrediction<T>(lastA, zs, as);

}

template class NeuralNet<float>;
template class NeuralNet<double>;
template class NeuralNet<ulong>;
