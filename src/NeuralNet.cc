/* *
 *
 */
#include "NeuralNet.h"
#include "debug.h"

#include "LogisticRegression.h"

template<typename T>
T NeuralNet<T>::nnCostFunctionSansGradient(const Matrix<T>& nn_params,
		int input_layer_size, int hidden_layer_size, int num_labels,
		const Matrix<T>& xBiased, const Matrix<T>& yBiased, T lambda) {
	Matrix<T> grad(1,nn_params.size / sizeof(T), false,true);
	T cost;
	nnCostFunction(grad,cost,nn_params, input_layer_size,
			hidden_layer_size, num_labels, xBiased, yBiased, lambda);
	return cost;
}

template<typename T>
 void NeuralNet<T>::nnCostFunction(Matrix<T>& grad, T& cost, const Matrix<T>& nn_params,
		int input_layer_size, int hidden_layer_size, int num_labels,
		const Matrix<T>& xBiased, const Matrix<T>& yBiased, T lambda) {
	if(debugNn)outln("entre xBiased " <<xBiased.toShortString());
	if(debugNn)outln("entre nn_params " <<nn_params.toShortString());
 	Matrix<T> theta1, theta2;
	nn_params.unconcat(theta1, hidden_layer_size,input_layer_size + 1,input_layer_size + 1, 0);
	//outln("theta1 " << theta1.toShortString());
	if(debugNn)outln("theta1.sum " << theta1.sum());
	//outln("nn_params " << nn_params);
	//outln("theta1.sum " << theta1.sum());
	nn_params.unconcat(theta2, num_labels, hidden_layer_size + 1, hidden_layer_size + 1,(hidden_layer_size * (input_layer_size + 1)));
	if(debugNn)outln("theta2.sum " << theta2.sum());
	uint m = xBiased.m;
	//Matrix<T> yb = y.toBinaryCategoryMatrix();
	Matrix<T> z2 = (theta1 * xBiased.transpose()).transpose();
	Matrix<T> a2 = Matrix<T>::ones(m, 1) |= z2.sigmoid();
	Matrix<T> z3 = (theta2 * a2.transpose()).transpose();
	Matrix<T> a3 = z3.sigmoid();
	cost  = ( -1./m * (yBiased % a3.log() + (1. - yBiased) % ((1. - a3).log()))).sum();
	if(debugNn)outln("j " << cost);
	// these should be views
	Matrix<T> tempTheta2 = theta2.dropFirst();
	//Matrix<T> tempTheta2;
	//theta2.submatrix(tempTheta2,theta2.m, theta2.n-1,0,1);
	//tempTheta2.syncBuffers();
	//outln("tempTheta2.sum "  << tempTheta2.sum());
	T jreg3 =  ((tempTheta2 ^ 2) * (lambda / (2. * m))).sum();
	cost  += jreg3;
	if(debugNn)outln("jreg3 " << jreg3);
	Matrix<T> tempTheta1 = theta1.dropFirst();
	//Matrix<T> tempTheta1;
	//theta1.submatrix(tempTheta1,theta1.m, theta1.n-1,0,1);

	//tempTheta1.syncBuffers();
	//outln("tempTheta1.sum "  << tempTheta1.sum());
	T jreg2 =   ((tempTheta1 ^ 2) * (lambda / (2. * m))).sum();
	if(debugNn)outln("jreg2 " << jreg2);
	cost  += jreg2;
	Matrix<T> delta_3 = a3 - yBiased;
	Matrix<T> delta_2 = (delta_3 * tempTheta2) % z2.sigmoidGradient();
	Matrix<T> bigDelta2 = delta_3.transpose() * a2;
	Matrix<T> bigDelta1 = delta_2.transpose() * xBiased;

	Matrix<T> temp = Matrix<T>::zeros(theta2.m, 1) |= tempTheta2;
	Matrix<T> theta2_grad = (bigDelta2 + (temp * lambda)) / m;
	temp = Matrix<T>::zeros(theta1.m, 1) |= tempTheta1;
	Matrix<T> theta1_grad = (bigDelta1 + lambda * temp) / m;
	//Matrix<T> grad = theta1_grad.poseAsCol() /= theta2_grad.poseAsCol();
	const Matrix<T>* parts[] = {&theta1_grad,&theta2_grad};
	//Matrix<T> grad = Matrix<T>::concat(2, parts);
	Matrix<T>::concat(grad, 2, parts);
	//theta1.unPose();
	//theta2.unPose();
	//if(debugNn)outln("grad.sum " << grad.sum());
	if(debugNn)outln("grad " << grad.toShortString());
	//return std::pair<T, Matrix<T> >(j, grad);
}

template<typename T>
T NeuralNet<T>::checkNnGradients(T lambda) {
	nnCostFunctionSanGradientOp op;
	op.input_layer_size = 3;
	op.hidden_layer_size = 5;
	op.num_labels = 3;
	uint m = 5;

	// We generate some 'random' test data
	Matrix<T> theta1 = Matrix<T>::sin(op.hidden_layer_size,
			op.input_layer_size + 1, false).syncBuffers();
	//outln("theta1\n" << theta1);
	Matrix<T> theta2 = Matrix<T>::sin(op.num_labels, op.hidden_layer_size + 1,
			false).syncBuffers();
	//outln("theta2\n" << theta2);
	Matrix<T> nn_params(1,(theta1.size + theta2.size)/sizeof(T),false,true);
	const Matrix<T> * parts[] = {&theta1,&theta2};
	 Matrix<T>::concat(nn_params,2, parts);
	//outln("nn_params\n" << nn_params);
	//theta1.unPose();
	//theta2.unPose();
	// Reusing debugInitializeWeights to generate X
	Matrix<T> x = Matrix<T>::sin(m, op.input_layer_size, false).syncBuffers();
	Matrix<T> xBiased = x.addBiasColumn();
	Matrix<T> y(m, 1,true,true);
	for (uint i = 0; i < m; i++) {
		y.set(i, (i + 1) % op.num_labels + 1);
	}
	y.syncBuffers();
	Matrix<T> yBiased = y.addBiasColumn();

	T epsilon = 1e-4;
	op._x = x;
	op.y = y;
	op.lambda = 3;
	Matrix<T> grad(nn_params.m,nn_params.n, false,true);
	T cost;
	nnCostFunction(grad, cost, nn_params,
			op.input_layer_size, op.hidden_layer_size, op.num_labels, xBiased, yBiased,
			lambda);
	Matrix<T> numgrad = NeuralNet<T>::gradientApprox(op, nn_params, epsilon);
	//outln("\n\ngrad  | numgrad\n" << (grad |= numgrad));
/*
	Matrix<T> ngtheta1 = numgrad.reshape(op.hidden_layer_size,
			op.input_layer_size + 1, 0);
	Matrix<T> ngtheta2 = numgrad.reshape(op.num_labels, op.hidden_layer_size + 1,
			(op.hidden_layer_size * (op.input_layer_size + 1)));
	Matrix<T> gtheta1 = grad.reshape(op.hidden_layer_size,
			op.input_layer_size + 1, 0);
	Matrix<T> gtheta2 = grad.reshape(op.num_labels, op.hidden_layer_size + 1,
			(op.hidden_layer_size * (op.input_layer_size + 1)));
*/
	T costApprox = ((numgrad - grad).vectorLength()
			/ (numgrad + grad).vectorLength());
	return costApprox;
}

// costFn: MatrixD => Double,
// todo 3.5 impl
template<typename T> template<typename CostFunction> Matrix<T> NeuralNet<T>::gradientApprox(
		CostFunction costFn, Matrix<T> theta, T epsilon) {
	const uint l = theta.m * theta.n;
	ulong i = 0;
	Matrix<T> perturb = Matrix<T>::zeros(theta.m, theta.n);
	Matrix<T> gradApprox = Matrix<T>::zeros(theta.m, theta.n);
	T jMinus = 0, jPlus = 0;
	while (i < l) {
		//perturb.set(i, epsilon);
		CuMatrix<T>::set(perturb.d_elements, perturb.m, perturb.n, perturb.p, i, static_cast<T>(epsilon));
		jMinus = costFn(theta - perturb);
		jPlus = costFn(theta + perturb);
		//gradApprox.set(i, (jPlus - jMinus) / (2. * epsilon));
		CuMatrix<T>::set(gradApprox.d_elements, gradApprox.m, gradApprox.n,gradApprox.p, i, static_cast<T>((jPlus - jMinus) / (2. * epsilon)));
		//perturb.set(i, 0);
		CuMatrix<T>::set(perturb.d_elements, perturb.m, perturb.n, perturb.p, i, static_cast<T>(0));
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

template<typename T> pair<T, vector<Matrix<T> > > NeuralNet<T>::forwardAndBack(Matrix<T>& x, Matrix<T>& y, vector<Matrix<T> > thetas, T lambda) {
	uint m = x.m;
	NNPrediction<T> tupl = predict(thetas, x);
	Matrix<T> hTheta = tupl.hTheta;
	vector<Matrix<T> >  zs = tupl.zs;
	vector<Matrix<T> >  as = tupl.as;
	Matrix<T> yb = y.isBinaryCategoryMatrix() ? y : y.toBinaryCategoryMatrix();

	T j = LogisticRegression<T>::costFunction(hTheta, yb, lambda, thetas);
	vector< Matrix<T> > deltas;
	vector< Matrix<T> > bigDeltas;
	vector< Matrix<T> > grads;
	deltas.insert(deltas.end(), hTheta - yb);
	int i = 0;
	uint* indices;
	int thetaCount = thetas.size();
	uint count;
	while (i < thetaCount - 1) {
		indicesFromRange(&indices, count, 1, thetas[i + 1].n - 1);
		Matrix<T> theta = thetas.at(i + 1).columnSubset(indices, count);
		delete[] indices;
		deltas[i] = (deltas[i + 1] * theta)
				% (zs[i].sigmoidGradient().transpose());
		i++;
	}
	i = 0;
	while (i < thetaCount) {
		bigDeltas[i] = deltas[i].transpose() * as[i];
		indicesFromRange(&indices, count, 1, thetas[i + 1].n - 1);
		Matrix<T> theta = Matrix<T>::zeros(thetas.at(i).m, 1) |= (thetas.at(i).columnSubset(indices, count));
		delete[] indices;
		grads[i] = (bigDeltas[i] + theta * lambda) / m;
		i++;
	}
	return pair<T, vector<Matrix<T> > >(j, grads);

}

template<typename T> Matrix<T> NeuralNet<T>::predictCg(const Matrix<T>& theta1,
		const Matrix<T>& theta2, const Matrix<T>& xBiased) {
	uint m = xBiased.m;

	Matrix<T> p = Matrix<T>::zeros(m, 1);
	Matrix<T> h1 = (xBiased * theta1.transpose()).sigmoid();
	Matrix<T> h2 = (h1.addBiasColumn() * theta2.transpose()).sigmoid();
	return h2;

}

// todo x? inputs? fix
template<typename T>
NNPrediction<T> NeuralNet<T>::predict(vector<Matrix<T> >weights,
		Matrix<T> inputs) {
	Matrix<T> x = inputs.hasBiasColumn() ? inputs : inputs.addBiasColumn();
	uint idx = 0;
	Matrix<T> lastA = zerom;
	vector<Matrix<T> > zs;
	vector<Matrix<T> > as;
	as.insert(as.end(),lastA);
	uint weightCount = weights.size();
	typedef typename vector< Matrix<T> >::iterator vecterator;
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
