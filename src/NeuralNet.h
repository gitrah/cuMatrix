/*
 * neuralnet.h
 *
 *  Created on: Oct 13, 2012
 *      Author: reid
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "Matrix.h"

template <typename T> struct NNPrediction{
	Matrix<T> hTheta;
	vector<Matrix<T> > zs;
	vector<Matrix<T> > as;
	NNPrediction(Matrix<T> hTheta, vector<Matrix<T> > zs, vector<Matrix<T> > as) : hTheta(hTheta), zs(zs), as(as) {}
};

template <typename T> class NeuralNet {
public:
	static T nnCostFunctionSansGradient(const Matrix<T>& nn_params, int input_layer_size,
	    int hidden_layer_size,
	    int num_labels,
	    const Matrix<T>& _x,
	    const Matrix<T>& y,
	    T lambda);

	struct nnCostFunctionSanGradientOp {
		int input_layer_size;
		int hidden_layer_size;
		int num_labels;
		Matrix<T> _x;
		Matrix<T> y;
		T lambda;
		__host__ __device__
		T operator()(const Matrix<T>& nn_params)  {
			return nnCostFunctionSansGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, _x,y,lambda);
		}
	};
	/**
	 * yb is 'binary category matrix' a map of y m x 1 ∈ Rn values to m x n binary
	 */

	static void nnCostFunction(Matrix<T>& grad, T& cost, const Matrix<T>& nn_params,
			int input_layer_size,
		    int hidden_layer_size,
		    int num_labels,
		    const Matrix<T>&_xBiased,
		    const Matrix<T>& yBiased,
		    T lambda);

	static T checkNnGradients( T lambda = 0);
	template <typename CostFunction> static Matrix<T> gradientApprox(
			  CostFunction costFn,
			  Matrix<T> theta,
			  T epsilon);
	static pair<T, vector<Matrix<T> > > forwardAndBack(Matrix<T>& x, Matrix<T>& y, vector<Matrix<T> > thetas, T lambda);

	static void indicesFromRange(uint** indices, uint& count, uint start, uint end);

	static Matrix<T> predictCg(const Matrix<T>& theta1, const Matrix<T>& theta2, const Matrix<T>& xBiased);
	static NNPrediction<T> predict(vector<Matrix<T> > weights, Matrix<T> inputs);
};

template <typename T> 	struct nnCostFtor {
	int input_layer_size;
    int hidden_layer_size;
    int num_labels;
    const Matrix<T>& xBiased;
    const Matrix<T>& y;
    T lambda;
    bool verbose;
    __host__ __device__	nnCostFtor(int _input_layer_size,
    int _hidden_layer_size,
    int _num_labels,
    Matrix<T>& _xBiased,
    Matrix<T>& _y,
    T _lambda ) : input_layer_size(_input_layer_size), hidden_layer_size(_hidden_layer_size),num_labels(_num_labels), xBiased( _xBiased), y(_y),lambda(_lambda) {}
    __host__ __device__	void operator()(Matrix<T>& grad, T& cost, const Matrix<T>& nn_params) {
		NeuralNet<T>::nnCostFunction(grad,cost, nn_params, input_layer_size, hidden_layer_size, num_labels, xBiased, y, lambda);
	}
};



#endif /* NEURALNET_H_ */
