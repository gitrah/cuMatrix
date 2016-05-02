/*
 * testlogreg.cc
 *
 *  Created on: Sep 14, 2012
 *      Author: reid
 */

#include "../CuMatrix.h"
#include "../util.h"
#include "../LogisticRegression.h"
#include "../ConjugateGradient.h"
#include "tests.h"

template int testParseOctave<float>::operator()(int argc, const char **argv) const;
template int testParseOctave<double>::operator()(int argc, const char **argv) const;
template <typename T> int testParseOctave<T>::operator()(int argc, const char **argv) const{

	if (argc < 2) {
		std::cout << "a" << argc << "usage: " << argv[0] << " <<filename>> "
				<< std::endl;

		exit(-1);
	}

	std::cout << "opening " << argv[1] << std::endl;
	std::map<std::string, CuMatrix<T>*> results = CuMatrix<T>::parseOctaveDataFile(argv[1],
			false, true);

	std::cout << "found " << results.size() << " octave objects\n";
	typedef typename std::map<std::string, CuMatrix<T>*>::iterator iterator;
	iterator it;
	it = results.begin();

	CuMatrix<T>* x = results["X"];
	CuMatrix<T>* y = results["y"];
	outln("loaded x " << x->toShortString());
	outln("loaded y " << y->toShortString());
	CuMatrix<T> mx = x->featureMeans(false);
	outln("mx " << mx.syncBuffers());
	while (it != results.end()) {
		CuMatrix<T>& m = *(*it).second;
		outln("m " << m);
		std::cout << (*it).first << std::endl;
		CuMatrix<T> means = m.featureMeans(false);
		outln("means " << means.syncBuffers());
		for (unsigned int i = 0; i < m.n; i++) {
			std::cout << i << ": " << means.elements[i] << std::endl;
		}
		std::cout << "m.get(0,0) " << m.get(0, 0) << std::endl;
		try {
			unsigned int mm = m.m;
			unsigned int mn = m.n;
			std::cout << "m.get(m/2,n/2) " << m.get(mm / 2 - 1, mn / 2 - 1)
					<< std::endl;
			std::cout << "m.get(m/3,n/3) " << m.get(mm / 3 - 1, mn / 3 - 1)
					<< std::endl;
			std::cout << "m.get(m-1,n-1) " << m.get(mm - 1, mn - 1)
					<< std::endl;
		} catch (MatrixException& c) {
			outln("ignoring exception " << c);
		}
		it++;
	}
    util<CuMatrix<T> >::deletePtrMap(results);
	//results[std::string("X")];
	//std::cout << << "\n";
	return (0);
}

template int testLogRegCostFunctionNoRegMapFeature<float>::operator()(int argc, const char **argv) const;
template int testLogRegCostFunctionNoRegMapFeature<double>::operator()(int argc, const char **argv) const;
template int testLogRegCostFunctionNoRegMapFeature<ulong>::operator()(int argc, const char **argv) const;

template <typename T> int testLogRegCostFunctionNoRegMapFeature<T>::operator()(int argc, const char **argv) const{

	const char* filename = "ex2data2.txt";

	std::cout << "opening " << filename << std::endl;
	std::map<std::string, CuMatrix<T>*> results = CuMatrix<T>::parseOctaveDataFile(filename,
			false, true);

	std::cout << "found " << results.size() << " octave objects\n";

	CuMatrix<T>& x = *results["X"];
	outln("x " << x.sum());

	CuMatrix<T>& y = *results["y"];
	outln("y\n" <<y);
	CuMatrix<T> x1 = x.columnVector(0);
	outln("x1\n" <<x1.syncBuffers());
	CuMatrix<T> x2 = x.columnVector(1);
	outln("x2\n" <<x2.syncBuffers());
	CuMatrix<T> xm = CuMatrix<T>::mapFeature(x1, x2, (float) 6);
	outln("xm\n" <<xm.syncBuffers());

	CuMatrix<T> init_theta = CuMatrix<T>::randn(xm.n, 1, .25);

	T lambda = 1;
	CuMatrix<T> grad;
	T cost;
	LogisticRegression<T>::gradCostFunction(grad,cost,xm, y, init_theta, lambda);
	outln("j " << cost);
	outln("grad " << grad.syncHost());

    util<CuMatrix<T> >::deletePtrMap(results);
	//results[std::string("X")];
	//std::cout << << "\n";
	return (0);
}

int innerFunc() {
    outln(print_stacktrace());
	return 0;
}

template <typename T> void howAboutMe() {
	T res = (T) innerFunc();
	outln("res " << res);
}

// TODO:  where is outerFn? Doesn't show in stack trace.
template <typename T> int outerFn() {
	howAboutMe<T>();
	return 2;
}

template int testCostFunction<float>::operator()(int argc, const char **argv) const;
template int testCostFunction<double>::operator()(int argc, const char **argv) const;
template int testCostFunction<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testCostFunction<T>::operator()(int argc, const char **argv) const{

	const char* filename = null;
	if (argc != 2 || argv[1][0] == '-') {
		filename = "ex2data2.txt";
		outln("using default filename " << filename);
	} else {
		filename = argv[1];
	}

	int iterations = b_util::getParameter(argc, argv, "its", 400);

	std::cout << "opening " << filename << std::endl;

	std::map<std::string, CuMatrix<T>*> results = CuMatrix<T>::parseOctaveDataFile(
			filename, false, true);

	outln( "found " << results.size() << " octave objects");

	CuMatrix<T>& x = CuMatrix<T>::getMatrix(results,"X");
	CuMatrix<T> x1 = x.columnVector(0);
	outln("x1\n" <<x1.syncBuffers());
	CuMatrix<T> x2 = x.columnVector(1);
	outln("x2\n" <<x2.syncBuffers());
	// map feature adds 'bias'/intercept column
	CuMatrix<T> xm = CuMatrix<T>::mapFeature(x1, x2, (float) 6);
	//outln("xm\n" <<xm.syncBuffers());

	T xmSum = xm.sum();
	outln("xm sum " <<xmSum );
	assert(util<T>::almostEquals(328.71,xmSum, .01));
	int m = xm.m;

	CuMatrix<T>& y = CuMatrix<T>::getMatrix(results,"y");
	//outln("y " << y.syncBuffers());

	CuMatrix<T> init_theta = CuMatrix<T>::zeros(xm.n, 1);
	outln("init_theta " << init_theta.syncHost());
	T lambda = 0;
	CuMatrix<T> grad,gradZeroLambda;
	T cost;

	LogisticRegression<T>::gradCostFunction(gradZeroLambda, cost, xm, y, init_theta, lambda);

	VerbOn();
	outln("gradZeroLambda " << gradZeroLambda.syncBuffers() << ", cost " << cost);
	VerbOff();

	T gradZeroLambdaA[] = { (T) 0.008475, (T)0.01879, (T)7.777e-05, (T)0.05034, (T)0.0115, (T)0.03766,
			(T)0.01836, (T)0.007324, (T)0.008192, (T)0.02348, (T)0.03935, (T)0.002239, (T)0.01286,
			(T)0.003096, (T)0.0393, (T)0.01997, (T)0.00433, (T)0.003386, (T)0.005838, (T)0.004476,
			(T)0.03101, (T)0.03103, (T)0.001097, (T)0.006316, (T)0.0004085, (T)0.007265, (T)0.001376,
			(T)0.03879 };
	CuMatrix<T> expGradZeroLambda(gradZeroLambdaA,init_theta.m,init_theta.n,true);
	T diff = ((expGradZeroLambda - gradZeroLambda) ^ (T)2).sum();
	outln("diff " << diff);
	assert(diff < util<T>::epsilon());
	assert(diff == expGradZeroLambda.sumSqrDiff(gradZeroLambda));
	//au = prev;

	outln("j (should be ~0.6931: " << cost);
	assert(Math<T>::aboutEq(cost, .6931f, .0001f));

	lambda=1;

	LogisticRegression<T>::gradCostFunction(grad, cost, xm, y, init_theta, lambda);
	// should still equal expGradZeroLambda because init_theta is a ZeroMatrix
	assert(diff == expGradZeroLambda.sumSqrDiff(grad));

	logRegCostFtor<T> ftor(y,xm, lambda);

    ConjugateGradient<T>::init();
    outln("post init last err " << b_util::lastErrStr());

    CuTimer justFmincg;
    justFmincg.start();
   // CuMatrix<T> init_thetaT = init_theta.transpose();
    pair<CuMatrix<T>, pair<CuMatrix<T>, int > > tup2 = ConjugateGradient<T>::fmincg(ftor,init_theta, iterations);
    outln("back from fmincg, took " << justFmincg.stop());

    CuMatrix<T> theta_reg = tup2.first;
    outln("theta_reg " << theta_reg.syncBuffers());
    CuMatrix<T> tup2_2m  = tup2.second.first;
    outln("tup2_2m " << tup2_2m.syncBuffers());
    outln(" tup2.second.second " << tup2.second.second);

    ftor(grad, cost, theta_reg);
    int iters = tup2.second.second;

    outln("converged after iters " << iters);
	assert(Math<T>::aboutEq(cost, .529f, .0001f));

	outln("j lambda = 1 " << cost);
	outln("grad \n" << grad.syncHost());
	T accuracy =  LogisticRegression<T>::predictionAccuracy(theta_reg, xm, y);
	outln("yields accuracy " <<accuracy);

	CuMatrix<T> predicted = (theta_reg.transpose() * xm.transpose()).sigmoid() >= (T).5;
	outln("predicted " << predicted.syncBuffers());
	// worth it to add a 'hadamard equals' operation?
	CuMatrix<T> truePos =  predicted.transpose().binaryOp( y, Functory<T, equalsBinaryOp>::pinch());
	outln("truePos " << truePos.syncBuffers());
	assert( util<T>::almostEquals(100 * truePos.sum()/(truePos.m * truePos.n),accuracy, util<T>::epsilon())) ;

    util<CuMatrix<T> >::deletePtrMap(results);
	return 0;
}


#include "tests.cc"
