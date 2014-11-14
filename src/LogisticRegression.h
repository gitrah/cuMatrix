/*
 * LogisticRegression.h
 *
 *  Created on: Aug 22, 2012
 *      Author: reid
 */
#ifndef LOGISTICREGRESSION_H_
#define LOGISTICREGRESSION_H_

#include "CuMatrix.h"
#include "util.h"

template<typename T> class LogisticRegression {
public:
	static void gradCostFunction(CuMatrix<T>& grad, T& cost,
			const CuMatrix<T>& x, const CuMatrix<T>& y,
			const CuMatrix<T>& theta, T lambda) {
		const int m = y.m;
		CuMatrix<T> tThetaX = theta.transpose() * x.transpose();
		outln("tThetaX" << tThetaX.syncBuffers());
		CuMatrix<T> hThetaT = tThetaX.sigmoid();
		outln("hThetaT" << hThetaT.syncBuffers());
		CuMatrix<T> yT = y.transpose();
		cost = costFunctionNoReg(hThetaT, yT, m);
		outln("cost " << cost);
		CuMatrix<T> gradNoReg = ((hThetaT - yT) * x) / m;
		CuMatrix<T> thetaCopy(theta);
		thetaCopy.set(0, 0, 0); // elements[0] = 0;
		CuMatrix<T> gradReg = lambda * thetaCopy.transpose() / m;
		outln("gradReg " << gradReg.syncBuffers());
		T jDel = lambda / (2. * m) * (thetaCopy ^ ((T)2)).sum();
		outln("jDel " << jDel);
		cost += jDel;
		grad = gradNoReg + gradReg;
	}


	// operator% => hadamardProduct (elementwise product)
	/*
	 * aka error function
	 * hThetaT (h . theta')
	 * yT ( y')
	 */
	static T costFunctionNoReg(CuMatrix<T>& hThetaT, CuMatrix<T>& yT, int m) {
		return static_cast<T>((-1. / m)
				* (yT % hThetaT.log() + (1 - yT) % ((1 - hThetaT).log())).sum());
	}

	/*
	 * replaces cost function with count of classification errors
	 */
	static T misclassificationErrorCount(CuMatrix<T>& hThetaT, CuMatrix<T>& yT, cudaStream_t stream = 0) {
		return hThetaT.combineReduce(misclassificationErrorBinaryOp<T>(),plusBinaryOp<T>(), yT, 0, stream);
	}

	static T costFunction(CuMatrix<T>& a, CuMatrix<T>& y, T lambda,
			vector<CuMatrix<T> > thetas) {
		uint m = y.m;
		CuMatrix<T> yb =
				y.isBinaryCategoryMatrix() ? y : y.toBinaryCategoryMatrix();
		T jDel = 0.;
		if (lambda != 0) {
			uint i = 0;
			typedef typename vector<CuMatrix<T> >::iterator iterator;
			for (iterator it = thetas.begin(); it < thetas.end(); it++) {
				CuMatrix<T> thetaCopy = (*it).dropFirst();
				T jdeldel = lambda / (2. * m) * thetaCopy.autoDot();
				outln( i << " jdeldel " << jdeldel);
				jDel += jdeldel;
				i += 1;
			}
		}
		T jNoReg = costFunctionNoReg(a, yb, m);
		outln("jNoReg " << jNoReg);
		return (jNoReg + jDel);
	}

	static T costFunctionNoReg2(CuMatrix<T>& hThetaT, CuMatrix<T>& yT, int m) {
		CuMatrix<T> yThThetaLog = yT % hThetaT.log();
		outln("yThThetaLog");
		outln(yThThetaLog.toString());
		CuMatrix<T> oneMinusyTh = (1 - yT) % ((1 - hThetaT).log());
		outln("oneMinusyTh");
		outln(oneMinusyTh.toString());
		CuMatrix<T> onMin = 1 - yT;
		outln("onMin");
		outln(onMin.toString());
		CuMatrix<T> onMinh = 1 - hThetaT;
		outln("onMinh");
		outln(onMinh.toString());
		CuMatrix<T> lonminh = onMinh.log();
		outln("lonminh");
		outln(lonminh.toString());
		CuMatrix<T> ha = onMin.hadamardProduct(lonminh);
		outln("ha");
		outln(ha.toString());
		CuMatrix<T> ha2 = onMin.binaryOp(lonminh, Functory<T,plusBinaryOp>::pinch());
		outln("ha2");
		outln(ha2.toString());
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
};

template<typename T> struct logRegCostFtor {
	const CuMatrix<T>& y;
	const CuMatrix<T>& theta;
	T lambda;
	logRegCostFtor( const CuMatrix<T>& y,
			const CuMatrix<T>& theta, T lambda) : y(y),theta(theta), lambda(lambda) {}

	virtual __host__ __device__ void operator()(CuMatrix<T>& grad, T& cost,
			const CuMatrix<T>& x) const {
		LogisticRegression<T>::gradCostFunction(grad, cost,x, y,theta, lambda);
	}
};

template void LogisticRegression<float>::gradCostFunction(
		CuMatrix<float>& grad, float& cost,
		const CuMatrix<float>& x, const CuMatrix<float>& y, const CuMatrix<float>& theta, float lambda);

template void LogisticRegression<double>::gradCostFunction(
		CuMatrix<double>& grad, double& cost,
		const CuMatrix<double>& x, const CuMatrix<double>& y, const CuMatrix<double>& theta,
		double lambda);
#endif /* LOGISTICREGRESSION_H_ */
