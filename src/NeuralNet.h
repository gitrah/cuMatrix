/*
 * neuralnet.h
 *
 *  Created on: Oct 13, 2012
 *      Author: reid
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "CuMatrix.h"

template <typename T> struct NNPrediction{
	CuMatrix<T> hTheta;
	vector<CuMatrix<T> > zs;
	vector<CuMatrix<T> > as;
	NNPrediction(CuMatrix<T> hTheta, vector<CuMatrix<T> > zs, vector<CuMatrix<T> > as) : hTheta(hTheta), zs(zs), as(as) {}
};

template <typename T> class NeuralNet {
public:
	static T nnCostFunctionSansGradient(const CuMatrix<T>& nn_params, int input_layer_size,
	    int hidden_layer_size,
	    int num_labels,
	    const CuMatrix<T>& _x,
	    const CuMatrix<T>& y,
	    T lambda);

	struct nnCostFunctionSanGradientOp {
		int input_layer_size;
		int hidden_layer_size;
		int num_labels;
		CuMatrix<T> _x;
		CuMatrix<T> y;
		T lambda;
		__host__ __device__
		T operator()(const CuMatrix<T>& nn_params)  {
			return nnCostFunctionSansGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, _x,y,lambda);
		}
	};
	/**
	 * yb is 'binary category matrix' a map of y m x 1 ∈ Rn values to m x n binary
	 */

	static void nnCostFunction(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& nn_params,
			int input_layer_size,
		    int hidden_layer_size,
		    int num_labels,
		    const CuMatrix<T>&_xBiased,
		    const CuMatrix<T>& yBiased,
		    T lambda);

	static T checkNnGradients( T lambda = 0);
	template <typename CostFunction> static CuMatrix<T> gradientApprox(
			  CostFunction costFn,
			  CuMatrix<T> theta,
			  T epsilon);
	static pair<T, vector<CuMatrix<T> > > forwardAndBack(CuMatrix<T>& x, CuMatrix<T>& y, vector<CuMatrix<T> > thetas, T lambda);

	static void indicesFromRange(uint** indices, uint& count, uint start, uint end);

	static CuMatrix<T> predictCg(const CuMatrix<T>& theta1, const CuMatrix<T>& theta2, const CuMatrix<T>& xBiased);
	static NNPrediction<T> predict(vector<CuMatrix<T> > weights, CuMatrix<T> inputs);
};

template <typename T> 	struct nnCostFtor {
	int input_layer_size;
    int hidden_layer_size;
    int num_labels;
    const CuMatrix<T>& xBiased;
    const CuMatrix<T>& y;
    T lambda;
    bool verbose;
    __host__ __device__	nnCostFtor(int _input_layer_size,
    int _hidden_layer_size,
    int _num_labels,
    CuMatrix<T>& _xBiased,
    CuMatrix<T>& _y,
    T _lambda ) : input_layer_size(_input_layer_size), hidden_layer_size(_hidden_layer_size),num_labels(_num_labels), xBiased( _xBiased), y(_y),lambda(_lambda),verbose(false) {}
    __host__ __device__	void operator()(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& nn_params) {
		NeuralNet<T>::nnCostFunction(grad,cost, nn_params, input_layer_size, hidden_layer_size, num_labels, xBiased, y, lambda);
	}
};



#endif /* NEURALNET_H_ */
