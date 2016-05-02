#include "tests.h"
#include "../util.h"
#include "../NeuralNet.h"
#include "../ConjugateGradient.h"

/*

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
*/

template <typename T> CuMatrix<T> testNNListDim(  list<uint> layerDims, const NnRunInfo<T>& runInfo ) {
	prlocf("testNNListDim enter: " );
	b_util::print(layerDims);
    T lambda = 3;
    map<string, CuMatrix<T>*>& f = runInfo.f;
    map<string, CuMatrix<T>*>& fw = runInfo.fw;
    bool repeatable = runInfo.repeatable;
    int maxIterations = runInfo.maxIterations;
	CuMatrix<T>* x = f["X"];
    outln("x "<< x->toShortString());
    x->syncBuffers();
    assert(x->lastMod == mod_synced);
    T sumx = x->sum();
    outln("sumx " << sumx);
    assert(util<T>::almostEquals(sumx, 2.626782601596818e05));

    CuMatrix<T>* y = f["y"];
    y->syncBuffers();
    outln("y " << y->toShortString());

    int input_layer_size = 400; // 20x20 Input Images of Digits
    int num_labels = 10; // 10 labels, from 1 to 10
    T cost;
    CuMatrix<T> xBiased = x->addBiasColumn();
	CuMatrix<T> training,training2, cv,cv2;
	vector<int> vIndices;
	xBiased.shuffle(&training,&cv,(T).75,vIndices);
	outln("refcounts training: " << training.hRefCount() << ", " << training.dRefCount());
	outln( "cv " << cv.hRefCount() << ", " << cv.dRefCount());
	CuMatrix<T> ytraining, ytrainingBiased, ycv, ycvBiased;
	outln( "ytraining befshuf " << ytraining.hRefCount() << ", " << ytraining.dRefCount());
	y->shuffle(&ytraining,&ycv,(T).75,vIndices);
	outln( "ytraining " << ytraining.hRefCount() << ", " << ytraining.dRefCount());
	outln( "ycv" << ycv.hRefCount() << ", " << ycv.dRefCount());
    CuMatrix<T> yBiased = y->toBinaryCategoryMatrix().syncBuffers();
    yBiased.shuffle(&ytrainingBiased,&ycvBiased,(T).75,vIndices);

    ConjugateGradient<T>::init();
	int resultIdx = 0;

	int lastDim = input_layer_size;
	int currDim = 0;
	vector<CuMatrix<T> > thetas;
	int numLayers = layerDims.size() + 1; // output layer is sized by numlabels
	const CuMatrix<T>* parts[numLayers];
	uint2 dims[numLayers];
	int offsets[numLayers];
	int layerIdx = 0;
	int totalElems= 0;
	T componentSum = 0;
    for(typename list<uint>::iterator i = layerDims.begin(); i != layerDims.end(); i++) {
    	currDim = *i;
    	thetas.push_back(
    		repeatable
				? CuMatrix<T>::sin(currDim,lastDim).syncBuffers().addBiasColumn()
				: CuMatrix<T>::randn(currDim,lastDim).syncBuffers().addBiasColumn()
    	);
    	outln("created " << thetas.back().toShortString());
    	offsets[layerIdx] = totalElems;
    	componentSum +=	thetas.back().sum();
    	dims[layerIdx].x = currDim; dims[layerIdx].y = lastDim+1;
    	totalElems += currDim * (lastDim + 1);
    	outln("totalElems after " << layerIdx  << ": " << totalElems);
    	parts[layerIdx++] = &(thetas.back());
    	lastDim = currDim;
    }
	offsets[layerIdx] = totalElems;

    outln("created "  << thetas.size() << " thetas");
    // now the output layer
	thetas.push_back(
		repeatable
			? CuMatrix<T>::sin(num_labels,lastDim).syncBuffers().addBiasColumn()
			: CuMatrix<T>::randn(num_labels,lastDim).syncBuffers().addBiasColumn()
	);
	totalElems += num_labels * (lastDim + 1);
	outln("totalElems " << totalElems);
	parts[layerIdx] = &(thetas.back());
	dims[layerIdx].x = num_labels; dims[layerIdx].y = lastDim+1;
	componentSum +=	thetas.back().sum();

	CuMatrix<T> initial_nn_params(totalElems,1,false,true);
	outln("initial_nn_params " << initial_nn_params.toShortString());

	//CuMatrix<T>::concat(initial_nn_params, numLayers, parts);
	CuMatrix<T>::concat(initial_nn_params, thetas);
	T aggregateSum = initial_nn_params.sum();

	assert(util<T>::almostEquals(aggregateSum, componentSum));

	nnCostFtorPm<T> pmFtor(dims, numLayers, training, ytrainingBiased, lambda);

	CuTimer justFmincg;
	justFmincg.start();

	pair<CuMatrix<T>, pair<CuMatrix<T>, int> > cgTup =
			ConjugateGradient<T>::fmincg(pmFtor, initial_nn_params,
					maxIterations);
	outln(
			"back from initial_nn_params pmfmincg, took " << justFmincg.stop()/1000 << "s");

	CuMatrix<T> nn_parmsPmGrad = cgTup.first;
	outln("nn_parmsPmGrad " << nn_parmsPmGrad.toShortString());
	outln("nn_parmsPmGrad.sum " << nn_parmsPmGrad.sum());

	thetas.clear();

	for(int i = 0; i < numLayers; i++ ) {
		thetas.push_back( CuMatrix<T>( dims[i].x, dims[i].y, false,true) );
		CuMatrix<T>& nTheta= thetas.back();
		outln("unconcat i " << i << ", dims[i].x " << dims[i].x << " dims[i].y  " << dims[i].y << ", offsets[i] " << offsets[i]);
		nn_parmsPmGrad.unconcat(nTheta, dims[i].x, dims[i].y, dims[i].y, offsets[i]);
		outln("new mat " << &nTheta << ":  helem " << nTheta.elements << ", buff " << nTheta.tiler.buff());
	}

	NNPrediction<T> pred = NeuralNet<T>::forward(thetas, training,
			ytrainingBiased, lambda);
	CuMatrix<T> p1 = pred.hTheta;
	//    outln("p1Pm " << p1Pm.syncBuffers());
	outln("p1 sum " << p1.sum());

	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1 = p1.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	NNPrediction<T> predCv = NeuralNet<T>::forward(thetas, cv,
			ycvBiased, lambda);
	CuMatrix<T> p1Cv = predCv.hTheta;
	checkCudaError(cudaGetLastError());
	CuMatrix<T> h1Cv = p1Cv.toMaxColumnIndexVector() + 1;
	checkCudaError(cudaGetLastError());
	outln("h1Cv " << h1Cv.toShortString());
	outln("h1Cv s " << h1Cv.sum());

	T res = ytraining.accuracy(h1);
	checkCudaError(cudaGetLastError());
	string dimsStr = b_util::toString(layerDims);
	outln("nnDims " << dimsStr << ": training accuracy : " << ( res * 100));

	T rescv = ycv.accuracy(h1Cv);
	checkCudaError(cudaGetLastError());
	outln("nnDims " << dimsStr << ": cvacc " << (rescv * 100) << ";");

	CuMatrix<T>output = CuMatrix<T>::zeros(1, numLayers);
	for(int i = 0; i < numLayers-1; i++ ) {
		output.set(0, i, dims[i].x);
	}
	output.set(0, numLayers-1, rescv );


	prlocf("testNNListDim exit: " );
	thetas.clear();

	return output;
}


template int testPermus<float>::operator()(int argc, const char **argv) const;
template int testPermus<double>::operator()(int argc, const char **argv) const;
template int testPermus<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testPermus<T>::operator()(int argc, const char **argv) const {
	uint2 dims[3];
	uint2 dims1[1];
	uint2 dims2[2];
	uint2 dims5[5];

	dims[0].x = 15; dims[0].y = 36;
	dims[1].x = 15; dims[1].y = 36;
	dims[2].x = 15; dims[2].y = 36;

	dims1[0].x = 25; dims1[0].y = 31;

	dims2[0].x = 20; dims2[0].y = 21;
	dims2[1].x = 10; dims2[1].y = 11;

	dims5[0].x = 10; dims5[0].y = 16;
	dims5[1].x = 10; dims5[1].y = 16;
	dims5[2].x = 10; dims5[2].y = 16;
	dims5[3].x = 10; dims5[3].y = 16;
	dims5[4].x = 10; dims5[4].y = 16;

	list<list<uint>> ranges = b_util::rangeLL(dims, 3,5);
	list<list<uint>> ranges1 = b_util::rangeLL(dims1, 1,5);
	list<list<uint>> ranges2 = b_util::rangeLL(dims2, 2,5);
	list<list<uint>> ranges5 = b_util::rangeLL(dims5, 5, 5);
	list<uint> sol;
	b_util::printAll(ranges, sol);
	b_util::printAll(ranges5, sol);

	outln("sol after " << sol.size());
	b_util::print(sol);

	int iterations = b_util::getParameter(argc, argv, "its", 50);

	map<string, CuMatrix<T>*> f = CuMatrix<T>::parseOctaveDataFile(SAMPLES_FILE,
    			false, true);
	map<string, CuMatrix<T>*> fw = CuMatrix<T>::parseOctaveDataFile(WEIGHTS_FILE,
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


	NnRunInfo<T> runInfo(50,f,fw, true);
	typename nnPermUtil<T,CuMatrix>::permFunction pf = testNNListDim;

	list<CuMatrix<T>> result1Rows;

	nnPermUtil<T, CuMatrix >::mapPermutations(result1Rows, pf, runInfo, ranges1, sol);
	outln("testPermus(" <<iterations <<") took " << timer.stop() << "ms");
	int idx = 0;
	for(typename list<CuMatrix<T>>::iterator i  =result1Rows.begin();  i != result1Rows.end(); i++) {
		outln("testPermus result1 i " << idx++ << ": "<< (*i).syncBuffers());
	}


	list<CuMatrix<T>> resultRows;
	timer.start();

	nnPermUtil<T, CuMatrix >::mapPermutations(resultRows, pf, runInfo, ranges2, sol);
	idx = 0;
	for(typename list<CuMatrix<T>>::iterator i  =resultRows.begin();  i != resultRows.end(); i++) {
		outln("testPermus result i " << idx++ << ": "<< (*i).syncBuffers());
	}
//	int ret = testNeuralKPtr(iterations, f,fw);
	//int ret = testNeuralH2N(iterations, f,fw, false);
	outln("testPermus(" <<iterations <<") took " << timer.stop() << "ms");
    util<CuMatrix<T> >::deletePtrMap(f);
    util<CuMatrix<T> >::deletePtrMap(fw);

	return 0;
}
