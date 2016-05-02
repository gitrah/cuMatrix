/*
 * neuralnet.h
 *
 *  Created on: Oct 13, 2012
 *      Author: reid
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "CuMatrix.h"
using std::stack;

template <typename T> struct NNPrediction{
	CuMatrix<T> hTheta;
	T cost;
	stack<CuMatrix<T> > zs;
	stack<CuMatrix<T> > as;
	NNPrediction(CuMatrix<T> hTheta, T cost, stack<CuMatrix<T> > zs, stack<CuMatrix<T> > as) : hTheta(hTheta),cost(cost), zs(zs), as(as) {}
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
		    T lambda, bool colMajor = false);

	static void nnCostFunctionN(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& nn_params,
			int input_layer_size,
		    int hidden_layer_size[],
		    int hidden_layer_count,
		    int num_labels,
		    const CuMatrix<T>&_xBiased,
		    const CuMatrix<T>& yBiased,
		    T lambda, bool colMajor = false);

	static T checkNnGradients( T lambda = 0);
	template <typename CostFunction> static CuMatrix<T> gradientApprox(
			  CostFunction costFn,
			  CuMatrix<T> theta,
			  T epsilon, bool colMajor);
	static void forwardAndBack(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& x,
			const CuMatrix<T>& y, vector<CuMatrix<T> >& thetas, T lambda);
	static void forwardAndBack(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& x,
			const CuMatrix<T>& y, const CuMatrix<T>& nn_params, int layers,
			const uint2* dims, T lambda);

	static void back(CuMatrix<T>& grad, uint2*& dims,
			NNPrediction<T>& tupl, const CuMatrix<T>& x, const CuMatrix<T>& y,
			vector<CuMatrix<T> >& thetas, T lambda);


	static void indicesFromRange(uint*& indices, uint& count, uint start, uint end);

	static CuMatrix<T> predictCg(const CuMatrix<T>& theta1, const CuMatrix<T>& theta2, const CuMatrix<T>& xBiased);
	static NNPrediction<T> forward(const vector<CuMatrix<T> >& weights,
			const CuMatrix<T>& inputs, const CuMatrix<T>& y, T lambda);
	static NNPrediction<T> forward(const PackedMat<T> pm, const CuMatrix<T>& inputs, const CuMatrix<T>& y);
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

template <typename T> 	struct nnCostFtorPm {
	uint2* dims;
    int layers;
    const CuMatrix<T>& xBiased;
    const CuMatrix<T>& y;
    T lambda;
    bool verbose;
    __host__ __device__	nnCostFtorPm(uint2* _dims,
    int _layers,
    const CuMatrix<T>& _xBiased,
    const CuMatrix<T>& _y,
    T _lambda ) : dims(_dims), layers(_layers), xBiased( _xBiased), y(_y),lambda(_lambda),verbose(false) {
    }
    /*
     * template<typename T> void NeuralNet<T>::forwardAndBack(CuMatrix<T>& grad,
		T& cost, const CuMatrix<T>& x, const CuMatrix<T>& y,
		const CuMatrix<T>& nn_params, int layers, const uint2* dims, T lambda) {
     *
     */
    __host__ __device__	void operator()(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& nn_params) {
     //   PackedMat<T> pm(&nn_params, layers, dims);
		NeuralNet<T>::forwardAndBack(grad, cost,  xBiased, y, nn_params, layers, dims, lambda);
	}

};


template <typename T> 	struct nnCostFtorN {
	int input_layer_size;
    int hidden_layer_size[];
    int hidden_layer_count;
    int num_labels;
    const CuMatrix<T>& xBiased;
    const CuMatrix<T>& y;
    T lambda;
    bool verbose;
    __host__ __device__	nnCostFtorN(int _input_layer_size,
    int _hidden_layer_size[],
    int _hidden_layer_count,
    int _num_labels,
    CuMatrix<T>& _xBiased,
    CuMatrix<T>& _y,
    T _lambda ) : input_layer_size(_input_layer_size), hidden_layer_size(_hidden_layer_size),hidden_layer_count(_hidden_layer_count),num_labels(_num_labels), xBiased( _xBiased), y(_y),lambda(_lambda),verbose(false) {}
    __host__ __device__	void operator()(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& nn_params) {
		NeuralNet<T>::nnCostFunction(grad,cost, nn_params, input_layer_size, hidden_layer_size, hidden_layer_count, num_labels, xBiased, y, lambda);
	}
};

/*
 * to support testing over permutations of NN topologies
 *
 */

template<typename T> struct NnRunInfo {
	int maxIterations;
	map<string, CuMatrix<T>*>& f;
	map<string, CuMatrix<T>*>& fw;
	bool repeatable;
	NnRunInfo( int maxIterations, map<string, CuMatrix<T>*>& f,	map<string, CuMatrix<T>*>& fw,	bool repeatable ) :
		  maxIterations(maxIterations), f(f), fw(fw), repeatable(repeatable) {}
};
template <typename T, template <typename> class OutT> struct nnPermUtil {

	/*
	 * a function that returns a 'OutT' and operates on a list of network layer dimensions
	 */

	typedef OutT<T> (*permFunction)( list<uint> layerDimPermInst, const NnRunInfo<T>& nnri );


	  /**
	   *  recursively applies fn to all the permutations of nn topologies
	   *
	 * @param retVals  		list of return values
	 * @param fn 			the fn to be applied to each permutation instance
	 * @param listOfLists	the list of layer size lists
	 * @param inst			holds particular permutation ('permutation instance') of topology
	 *
	 */
	static void mapPermutations( list<OutT<T> >& retVals,  permFunction fn, const NnRunInfo<T>& nnri, list<list<uint>> listOfLists, list<uint> inst);
};

#endif /* NEURALNET_H_ */
