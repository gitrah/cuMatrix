/*
 * testlogreg.cc
 *
 *  Created on: Sep 14, 2012
 *      Author: reid
 */

#include "../CuMatrix.h"
#include "../util.h"
#include "../LinearRegression.h"
#include "tests.h"


template int testLinRegCostFunctionNoReg<float>::operator()(int argc, const char **argv) const;
template int testLinRegCostFunctionNoReg<double>::operator()(int argc, const char **argv) const;
template int testLinRegCostFunctionNoReg<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testLinRegCostFunctionNoReg<T>::operator()(int argc, const char **argv) const{

	if (argc < 2) {
		std::cout << "usage: " << argv[0] << " <<filename>> " << std::endl;
		exit(-1);
	}

	std::cout << "opening " << argv[1] << std::endl;
	std::map<std::string, CuMatrix<T>*> results = CuMatrix<T>::parseOctaveDataFile(argv[1],
			false, true);

	std::cout << "found " << results.size() << " octave objects\n";

	CuMatrix<T>& x = *results["X"];
	outln("x " << x.toShortString());

	CuMatrix<T>& y = *results["y"];
	int m = y.m;
	CuMatrix<T> xb = x.addBiasColumn();
	CuMatrix<T> init_theta = CuMatrix<T>::zeros(xb.n, 1);
	CuMatrix<T> hThetaTXbT = init_theta.transpose() * xb.transpose();
	outln("hThetaTXbT " << hThetaTXbT.syncBuffers());
	CuMatrix<T> yT = y.transpose();
	T jno2m = LinearRegression<T>::costFunctionNoRegNu(hThetaTXbT, yT, m);
	outln("jno2m " << jno2m);

	T j = LinearRegression<T>::costFunctionNoReg(hThetaTXbT, yT, m);
	outln("j " << j);

	T alpha = .01;
	int iters = 1500;
	CuMatrix<T> jHistory = CuMatrix<T>::zeros(iters,1);
	CuTimer timer;
	timer.start();
//	CuMatrix<T> theta = LinearRegression<T>::gradientDescentLoop(alpha, init_theta,  x,y, jHistory, iters);
	CuMatrix<T> theta = LinearRegression<T>::gradientDescent(alpha, init_theta,  x,y, jHistory, iters);
	outln("theta " << theta.syncBuffers() << " took " << timer.stop() << "ms");
	outln("jHistory\n" << jHistory.syncBuffers()  );
	T pr1[] = {1, (T) 3.5};
	T pr2[] = {1, 7};

	CuMatrix<T> predict1( pr1, 1,2,true);
	CuMatrix<T> predict2( pr2, 1,2,true);
	T prof1 = (10000* predict1 * theta).toScalar();
	T prof2 = (10000* predict2 * theta).toScalar();
	outln("For population = 35,000, we predict a profit of " << prof1);
	outln("For population = 70,000, we predict a profit of " << prof2);

/*
	T lambda = 0;
	CuMatrix<T> grad;
	T cost;
	LinearRegression<T>::gradCostFunction(grad,cost,xm, y, init_theta, lambda);
	outln("j " << grad);
*/

	//results[std::string("X")];
	//std::cout << << "\n";
    util<CuMatrix<T> >::deletePtrMap(results);
	return (0);
}

template int testLinRegCostFunction<float>::operator()(int argc, const char **argv) const;
template int testLinRegCostFunction<double>::operator()(int argc, const char **argv) const;
template int testLinRegCostFunction<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testLinRegCostFunction<T>::operator()(int argc, const char **argv) const{

	outln( "starting testLinRegCostFunction");
	const char* filename = "ex2data2m.txt";

	outln( "opening " << filename);
	std::map<std::string, CuMatrix<T>*> results = CuMatrix<T>::parseOctaveDataFile(
			filename, false, true);

	outln( "found " << results.size() << " octave objects");

	CuMatrix<T>& x = *results["X"];
	outln("x " << x.sum());

	CuMatrix<T>& y = *results["y"];
	outln("y\n" <<y.toString());
	CuMatrix<T> xm = CuMatrix<T>::mapFeature(x.columnVector(0), x.columnVector(1));

	outln("xm " << xm.toShortString());

	CuMatrix<T> init_theta = CuMatrix<T>::zeros(xm.n, 1);

	T lambda = 0;
	CuMatrix<T> grad;
	T cost;

	LinearRegression<T>::gradCostFunction(grad,cost,xm, y, init_theta,
			lambda);
	//au = prev;

	outln("j " << cost);
	assert(Math<T>::aboutEq(cost, .6931f, .0001f));

	outln("grad\n" <<grad);
	lambda=1;
	init_theta = CuMatrix<T>::ones(xm.n,1) * .25f;
	LinearRegression<T>::gradCostFunction(grad,cost,xm, y, init_theta, lambda);

	outln("j lambda = 1 " << cost);
	outln("grad \n" << grad);

	assert(Math<T>::aboutEq(cost, .9081f, .0001f));

	//results[std::string("X")];
	//std::cout << << "\n";
    util<CuMatrix<T> >::deletePtrMap(results);
	return (0);
}

#include "tests.cc"
