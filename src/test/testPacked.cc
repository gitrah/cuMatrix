#include "../NeuralNet.h"
#include "../CuMatrix.h"
#include "tests.h"

template int testPackedMat<float>::operator()(int argc, const char **argv) const;
template int testPackedMat<double>::operator()(int argc, const char **argv) const;
template int testPackedMat<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testPackedMat<T>::operator()(int argc, const char **argv) const {

	T lambda = 3;

    uint num_samples = 5000;
    uint input_layer_size = 400; // 20x20 Input Images of Digits
    uint hidden_layer_size = 25; //   25 hidden units
    uint num_labels = 10; // 10 labels, from 1 to 10
 	vector<CuMatrix<T> > thetaV;

    CuMatrix<T> theta1 = CuMatrix<T>::sin(hidden_layer_size, input_layer_size).syncBuffers().addBiasColumn();
    CuMatrix<T> theta1b = CuMatrix<T>::sin(hidden_layer_size, hidden_layer_size).syncBuffers().addBiasColumn();
    CuMatrix<T> theta2 = CuMatrix<T>::sin(num_labels, hidden_layer_size).syncBuffers().addBiasColumn();
    outln("theta1 " << theta1.toShortString());
    outln("theta1b " << theta1b.toShortString());
    outln("theta2 " << theta2.toShortString());


    T thetasSum = theta1.sum() + theta2.sum();

    // test vector-based pack/unpack
    vector< CuMatrix<T>> vthta;
    vthta.push_back(theta1);
    vthta.push_back(theta2);

    PackedMat<T> pmv = PackedMat<T>::pack(vthta);
    outln("pmv owner " << tOrF(pmv.owner));
    outln("pmv dims " << pmv.dumpDims());
    outln("pmv nn_params " << pmv.nn_params->toShortString());
    T packedThetasSum = pmv.nn_params->sum();
    outln("pmv nn_params sum " << packedThetasSum);
    assert(util<T>::almostEquals( thetasSum,packedThetasSum, 0.01));

    vector< CuMatrix<T>> outies;

	uint offset = 0;
	for(int i =0; i < pmv.layers; i++ ) {
		CuMatrix<T> layer(pmv.dims[i].x, pmv.dims[i].y,false,true);
		pmv.nn_params->unconcat(layer , pmv.dims[i].x, pmv.dims[i].y , pmv.dims[i].y, offset, false);
		outln("outies adding layer " << i << ": " << layer.toShortString());
		outies.push_back(layer);
		offset += pmv.dims[i].x * pmv.dims[i].y;
	}

	checkCudaErrors(cudaGetLastError());

	CuMatrix<T> x = CuMatrix<T>::sin(num_samples, input_layer_size).syncBuffers();
	CuMatrix<T> xBiased = x.addBiasColumn();
	CuMatrix<T> y = (CuMatrix<T>::randn(num_samples,1) * 5 + 5).floor() ;
	outln("y " << y.syncBuffers());
	CuMatrix<T> yBiased = y.toBinaryCategoryMatrix();
    T cost=12.20335035028618; //0.5760512456162463;
 	thetaV.push_back(theta1);
    thetaV.push_back(theta2);
	CuMatrix<T> fbGrad;
	T fbCost;
	outln("pre fAndB with xBiased " << xBiased.toShortString());
	outln("pre fAndB with yBiased " << yBiased.toShortString());
	NeuralNet<T>::forwardAndBack(fbGrad, fbCost, xBiased, yBiased, thetaV, lambda);
	outln("fbGrad " << fbGrad << ", fbCost " << fbCost << "\n\n");
	assert(util<T>::almostEquals(cost, fbCost, 1e-8));

    thetaV.clear();
    thetaV.push_back(theta1);
	thetaV.push_back(theta1b);
	thetaV.push_back(theta2);
	NeuralNet<T>::forwardAndBack(fbGrad, fbCost, xBiased, yBiased, thetaV, lambda);
	outln("with extra layer, fbCost " << fbCost);

    const CuMatrix<T>* parts3[] = {&theta1, &theta1b, &theta2};

	CuMatrix<T> nn_params_3((theta1.size + theta1b.size + theta2.size)/sizeof(T),1,false,true);
	outln("\n\nnn_params_3 bef concat " << nn_params_3.toShortString());
	//flprintf("nn_params_3 %.20g\n", nn_params_3.sum());
    CuMatrix<T>::concat(nn_params_3, 3, parts3);
	//ok if(1==1) return 0;
    uint2 dims[3];
    dims[0].x = theta1.m;dims[0].y = theta1.n;
    dims[1].x = theta1b.m;dims[1].y =theta1b.n;
    dims[2].x = theta2.m;dims[2].y = theta2.n;
    outln("\n\nnn_params_3 aft concat " << nn_params_3.toShortString());

}
