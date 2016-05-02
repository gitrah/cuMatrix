/*
 * LogisticRegression.h
 *
 *  Created on: Aug 22, 2012
 *      Author: reid
 */
#pragma once

#include "CuMatrix.h"
#include "util.h"
/*
 * z(theta)i = theta' * xi (where theta is col vector, xi is ith sample (also as row vector)
 * hTheta(xi) = g(zi) = 1/(1 + e-zi) = 1/(1 + e-(theta'xi)
 * m1 * m2 = m2' * m1
 */
template<typename T> class LogisticRegression {
public:

	/*
	 * gradCostFunction
	 * 		grad
	 * 		cost cost output
	 *
	 */
	static __host__ CUDART_DEVICE void gradCostFunction(CuMatrix<T>& grad, T& cost,
				const CuMatrix<T>& x, const CuMatrix<T>& y,
				const CuMatrix<T>& theta, T lambda);
/*
	static void gradCostFunction(CuMatrix<T>& grad, T& cost,
			const CuMatrix<T>& x, const CuMatrix<T>& y,
			const CuMatrix<T>& theta, T lambda) {
		const int m = y.m;
		outln("enter x " << x.toShortString() );
		CuMatrix<T> tThetaX = theta.transpose() * x.transpose();
		outln("tThetaX" << tThetaX.syncBuffers());
		CuMatrix<T> hThetaT = tThetaX.sigmoid();
		outln("hThetaT" << hThetaT.syncBuffers());
		CuMatrix<T> yT = y.transpose();
		cost = costFunctionNoReg(hThetaT, yT, m);
		outln("cost " << cost);
		CuMatrix<T> gradNoReg = ((hThetaT - yT) * x) / m;
		outln("gradNoReg " << gradNoReg.syncBuffers());
		CuMatrix<T> thetaCopy(theta);
		thetaCopy.set(0, 0, 0); // elements[0] = 0;
		CuMatrix<T> gradReg = lambda * thetaCopy.transpose() / m;
		outln("gradReg " << gradReg.syncBuffers());
		T jDel = lambda / (2. * m) * (thetaCopy ^ ((T)2)).sum();
		outln("jDel " << jDel);
		cost += jDel;
		grad = gradNoReg + gradReg;
	}

*/

	// operator% => hadamardProduct (element-wise product)
	/*
	 * hThetaT (h . theta')
	 * yT ( y')
	 */
	static T costFunctionNoReg(const CuMatrix<T>& hThetaT, const CuMatrix<T>& yT, int m) {
#ifndef __CUDA_ARCH__
		if(checkDebug(debugCg)) {
			printf("costFunctionNoReg hThetaT %dx%d d %p h %p,", hThetaT.m,hThetaT.n, hThetaT.tiler.currBuffer(), hThetaT.elements );
			printf("yT %dx%d d %p h %p\n", yT.m,yT.n, yT.tiler.currBuffer(), yT.elements );
		}
#endif
		return static_cast<T>((-1. / m)
				* (yT % hThetaT.log() + (1 - yT) % ((1 - hThetaT).log())).sum());
	}

	/*
	 * replaces cost function with count of classification errors
	 */
	static T misclassificationErrorCount(CuMatrix<T>& hThetaT, CuMatrix<T>& yT, cudaStream_t stream = 0) {
		return hThetaT.combineReduce(misclassificationErrorBinaryOp<T>(),plusBinaryOp<T>(), yT, 0, stream);
	}

	static T costFunction(const CuMatrix<T>& a, const CuMatrix<T>& y, T lambda,
			const vector<CuMatrix<T> > thetas) {
		int m = y.m;
		CuMatrix<T> yb =
				y.isBinaryCategoryMatrix() ? y : y.toBinaryCategoryMatrix();
		T jDel = 0.;
		if (lambda != 0) {
			uint i = 0;
			typedef typename vector<CuMatrix<T> >::const_iterator iterator;
			for (iterator it = thetas.begin(); it < thetas.end(); it++) {
				CuMatrix<T> thetaCopy = (*it).dropFirst();
				T jdeldel = lambda / (2. * m) * thetaCopy.autoDot();
				if(checkDebug(debugCg))outln( i << " lambda " << lambda << ", jdeldel " << jdeldel);
				jDel += jdeldel;
				i += 1;
			}
		}
		T jNoReg = costFunctionNoReg(a, yb, m);
		//outln("jNoReg " << jNoReg);
		return (jNoReg + jDel);
	}

	static T costReg(int m, T lambda, const vector<CuMatrix<T> > thetas) {
		T jDel = 0.;
		if (lambda != 0) {
			uint i = 0;
			typedef typename vector<CuMatrix<T> >::const_iterator iterator;
			for (iterator it = thetas.begin(); it < thetas.end(); it++) {
				CuMatrix<T> thetaCopy = (*it).dropFirst();
				T jdeldel = lambda / (2. * m) * thetaCopy.autoDot();
				if(checkDebug(debugNn))outln("i " << i << " lambda " << lambda << ", jdeldel " << jdeldel);
				jDel += jdeldel;
				i += 1;
			}
		}
		return (jDel);
	}

	static T __host__ __device__  costFunctionNoReg2(CuMatrix<T>& hThetaT, CuMatrix<T>& yT, int m) {
		CuMatrix<T> yThThetaLog = yT % hThetaT.log();
		if (checkDebug(debugCg)) {
			CuMatrix<T> oneMinusyTh = (1 - yT) % ((1 - hThetaT).log());
			CuMatrix<T> onMin = 1 - yT;
			CuMatrix<T> onMinh = 1 - hThetaT;
			CuMatrix<T> lonminh = onMinh.log();
			CuMatrix<T> ha = onMin.hadamardProduct(lonminh);
			CuMatrix<T> ha2 = onMin.binaryOp(lonminh,
					Functory<T, plusBinaryOp>::pinch());
#ifndef __CUDA_ARCH__
			outln("hThetaT " << hThetaT.syncBuffers());
			outln("yThThetaLog " << yThThetaLog.syncBuffers());
			outln("oneMinusyTh " << oneMinusyTh.syncBuffers());
			outln("onMin " << onMin.syncBuffers());
			outln("onMinh " << onMinh.syncBuffers());
			outln("lonminh " << lonminh.syncBuffers());
			outln("ha " << ha .syncBuffers());
			outln("ha2 " << ha2 .syncBuffers());
#endif
		}
		return static_cast<T>((-1. / m)
				* (yThThetaLog + (1 - yT) % ((1 - hThetaT).log())).sum());
	}

	template<typename CostFunction> static CuMatrix<T> gradientApprox(
			CostFunction costFn, CuMatrix<T> theta, T epsilon) {
		uint l = theta.m * theta.n;
		uint i = 0;
		CuMatrix<T> perturb = CuMatrix<T>::zeros(theta.dims());
		CuMatrix<T> gradApprox = CuMatrix<T>::zeros(theta.dims());
		T jMinus = 0, jPlus = 0;
		while (i < l) {
			perturb.set(i, epsilon);
			jMinus = costFn(theta - perturb);
			jPlus = costFn(theta + perturb);
			gradApprox.elements[i] = (jPlus - jMinus) / (2. * epsilon);
			perturb.set(i, 0);
			i++;
		}
		return gradApprox;
	}

	static T predictionAccuracy(const CuMatrix<T>& theta, const CuMatrix<T>& x,const CuMatrix<T>& y) {
		CuMatrix<T> predicted = (theta.transpose() * x.transpose()).sigmoid() >= (T).5;
		return 100 * predicted.transpose().accuracy(y);
	}

};

template<typename T> struct logRegCostFtor {
	const CuMatrix<T>& y;
	const CuMatrix<T>& x;
	T lambda;
	logRegCostFtor( const CuMatrix<T>& y,const CuMatrix<T>& x, T lambda) : y(y),x(x), lambda(lambda) {}
	 __host__ __device__ void operator()(CuMatrix<T>& grad, T& cost, const CuMatrix<T>& theta) const {
		LogisticRegression<T>::gradCostFunction(grad, cost,x, y,theta, lambda);
	}
};

