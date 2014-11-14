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

template int testParseOctave<float>::operator()(int argc, char const ** args) const;
template int testParseOctave<double>::operator()(int argc, char const ** args) const;
template <typename T> int testParseOctave<T>::operator()(int argc, char const ** args) const{

	if (argc < 2) {
		std::cout << "a" << argc << "usage: " << args[0] << " <<filename>> "
				<< std::endl;

		exit(-1);
	}

	std::cout << "opening " << args[1] << std::endl;
	std::map<std::string, CuMatrix<T>*> results = util<T>::parseOctaveDataFile(args[1],
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

template int testLogRegCostFunctionNoRegMapFeature<float>::operator()(int argc, char const ** args) const;
template int testLogRegCostFunctionNoRegMapFeature<double>::operator()(int argc, char const ** args) const;
template <typename T> int testLogRegCostFunctionNoRegMapFeature<T>::operator()(int argc, char const ** args) const{

	if (argc < 2) {
		std::cout << "usage: " << args[0] << " <<filename>> " << std::endl;
		exit(-1);
	}

	std::cout << "opening " << args[1] << std::endl;
	std::map<std::string, CuMatrix<T>*> results = util<T>::parseOctaveDataFile(args[1],
			false, true);

	std::cout << "found " << results.size() << " octave objects\n";
	typedef typename std::map<std::string, CuMatrix<T>*>::iterator iterator;
	iterator it;
	it = results.begin();

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

template <typename T> int outerFn() {
	return innerFunc();
}
template int testCostFunction<float>::operator()(int argc, char const ** args) const;
template int testCostFunction<double>::operator()(int argc, char const ** args) const;
template int testCostFunction<ulong>::operator()(int argc, char const ** args) const;
template <typename T> int testCostFunction<T>::operator()(int argc, char const ** args) const{

	//const char* filename = "ex2data2m.txt";
	if (argc < 2) {
		outln("argc " << argc);
		std::cout << "usage: " << args[0] << " <<filename>> " << std::endl;
		exit(-1);
	}

	int iterations = b_util::getParameter(argc, args, "its", 400);

	std::cout << "opening " << args[1] << std::endl;

	std::map<std::string, CuMatrix<T>*> results = util<T>::parseOctaveDataFile(
			args[1], false, true);

	outln( "found " << results.size() << " octave objects");
	typedef typename std::map<std::string, CuMatrix<T>*>::iterator iterator;
	iterator it;
	it = results.begin();

	CuMatrix<T>& x = *results["X"];
	outln("x " << x.sum());
	CuMatrix<T> xb = x.addBiasColumn();

	CuMatrix<T>& y = *results["y"];
/*
	outln("y\n" <<y.toString();
	CuMatrix<T> xm = CuMatrix<T>::mapFeature(x.columnVector(0), x.columnVector(1));
	outln("xm " << xm.toShortString()));
*/
	int ans = outerFn<T>();

	CuMatrix<T> init_theta = CuMatrix<T>::zeros(xb.n, 1);
	outln("init_theta " << init_theta.syncHost());
	T lambda = 0;
	CuMatrix<T> grad;
	T cost;

	LogisticRegression<T>::gradCostFunction(grad,cost,xb, y, init_theta,
			lambda);
	//au = prev;

	outln("j " << cost);
	assert(Math<T>::aboutEq(cost, .6931f, .0001f));

	outln("grad\n" <<grad.syncHost());

	lambda=1;
	logRegCostFtor<T> ftor(y,init_theta.syncBuffers(), lambda);
   // nnCostFtor<T> ftor(input_layer_size, hidden_layer_size, num_labels, training, ytrainingBiased, lambda);

    ConjugateGradient<T>::init();
    outln("post init last err " << b_util::lastErrStr());

    CuTimer justFmincg;
    justFmincg.start();
    pair<CuMatrix<T>, pair<CuMatrix<T>, int > > tup2 = ConjugateGradient<T>::fmincg(ftor,init_theta, iterations);
    outln("back from fmincg, took " << justFmincg.stop());

    CuMatrix<T> grad_reg = tup2.first;

	outln("j lambda = 1 " << cost);
	outln("grad_reg \n" << grad_reg.syncHost());

	assert(Math<T>::aboutEq(cost, .9081f, .0001f));

    util<CuMatrix<T> >::deletePtrMap(results);
	//results[std::string("X")];
	//std::cout << << "\n";
	return ans;
}


#include "tests.cc"
