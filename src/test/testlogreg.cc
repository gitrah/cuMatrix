/*
 * testlogreg.cc
 *
 *  Created on: Sep 14, 2012
 *      Author: reid
 */

#include "../Matrix.h"
#include "../util.h"
#include "../LogisticRegression.h"
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
	std::map<std::string, Matrix<T>*> results = util<T>::parseOctaveDataFile(args[1],
			false, true);

	std::cout << "found " << results.size() << " octave objects\n";
	typedef typename std::map<std::string, Matrix<T>*>::iterator iterator;
	iterator it;
	it = results.begin();

	Matrix<T>* x = results["X"];
	Matrix<T>* y = results["y"];
	outln("loaded x " << x->toShortString());
	outln("loaded y " << y->toShortString());

	while (it != results.end()) {
		Matrix<T>& m = *(*it).second;
		std::cout << (*it).first << std::endl;
		Matrix<T> means = m.featureMeans(false);
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
		} catch (char const * c) {
			outln("ignoring exception " << c);
		}
		it++;
	}
	//results[std::string("X")];
	//std::cout << << "\n";
	return (0);
}

template int testCostFunctionNoReg0<float>::operator()(int argc, char const ** args) const;
template int testCostFunctionNoReg0<double>::operator()(int argc, char const ** args) const;
template <typename T> int testCostFunctionNoReg0<T>::operator()(int argc, char const ** args) const{

	if (argc < 2) {
		std::cout << "usage: " << args[0] << " <<filename>> " << std::endl;
		exit(-1);
	}

	std::cout << "opening " << args[1] << std::endl;
	std::map<std::string, Matrix<T>*> results = util<T>::parseOctaveDataFile(args[1],
			false, true);

	std::cout << "found " << results.size() << " octave objects\n";
	typedef typename std::map<std::string, Matrix<T>*>::iterator iterator;
	iterator it;
	it = results.begin();

	Matrix<T>& x = *results["X"];
	outln("x " << x.sum());

	Matrix<T>& y = *results["y"];
	outln("y\n" <<y.toString());
	Matrix<T> x1 = x.columnVector(0);
	outln("x1\n" <<x1.toString());
	Matrix<T> x2 = x.columnVector(1);
	outln("x2\n" <<x2.toString());
	Matrix<T> xm = Matrix<T>::mapFeature(x1, x2, (float) 6);
	outln("xm\n" <<xm.toString());

	Matrix<T> init_theta = Matrix<T>::randn(xm.n, 1, .25);

	T lambda = 1;
	Matrix<T> grad;
	T cost;
	LogisticRegression<T>::gradCostFunction(grad,cost,xm, y, init_theta, lambda);
	outln("j " << grad);

	//results[std::string("X")];
	//std::cout << << "\n";
	return (0);
}

template int testCostFunction<float>::operator()(int argc, char const ** args) const;
template int testCostFunction<double>::operator()(int argc, char const ** args) const;
template <typename T> int testCostFunction<T>::operator()(int argc, char const ** args) const{

	const char* filename = "ex2data2m.txt";

	outln( "opening " << filename);
	std::map<std::string, Matrix<T>*> results = util<T>::parseOctaveDataFile(
			filename, false, true);

	outln( "found " << results.size() << " octave objects");
	typedef typename std::map<std::string, Matrix<T>*>::iterator iterator;
	iterator it;
	it = results.begin();

	Matrix<T>& x = *results["X"];
	outln("x " << x.sum());

	Matrix<T>& y = *results["y"];
	outln("y\n" <<y.toString();
	Matrix<T> xm = Matrix<T>::mapFeature(x.columnVector(0), x.columnVector(1));
	outln("xm " << xm.toShortString()));

	Matrix<T> init_theta = Matrix<T>::zeros(xm.n, 1);

	T lambda = 0;
	Matrix<T> grad;
	T cost;

	LogisticRegression<T>::gradCostFunction(grad,cost,xm, y, init_theta,
			lambda);
	//au = prev;

	outln("j " << cost);
	assert(Math<T>::aboutEq(cost, .6931f, .0001f));

	outln("grad\n" <<grad);
	lambda=1;
	init_theta = Matrix<T>::ones(xm.n,1) * .25f;
	LogisticRegression<T>::gradCostFunction(grad,cost,xm, y, init_theta, lambda);

	outln("j lambda = 1 " << cost);
	outln("grad \n" << grad);

	assert(Math<T>::aboutEq(cost, .9081f, .0001f));

	//results[std::string("X")];
	//std::cout << << "\n";
	return (0);
}

#include "tests.cc"
