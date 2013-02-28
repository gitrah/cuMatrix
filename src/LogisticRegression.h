/*
 * LogisticRegression.h
 *
 *  Created on: Aug 22, 2012
 *      Author: reid
 */

#include "Matrix.h"
#include "util.h"

template<typename T> class LogisticRegression {
public:
	static void gradCostFunction(Matrix<T>& grad, T& cost,
			const Matrix<T>& x, const Matrix<T>& y,
			const Matrix<T>& theta, T lambda) {
		const int m = y.m;
		Matrix<T> tThetaX = theta.transpose() * x.transpose();
		outln("tThetaX");
		outln(tThetaX.toString());
		Matrix<T> hThetaT = tThetaX.sigmoid();
		outln("hThetaT");
		outln(hThetaT.toString());
		Matrix<T> yT = y.transpose();
		cost = costFunctionNoReg(hThetaT, yT, m);
		outln("cost " << cost);
		Matrix<T> gradNoReg = ((hThetaT - yT) * x) / m;
		Matrix<T> thetaCopy(theta);
		thetaCopy.set(0, 0, 0); // elements[0] = 0;
		Matrix<T> gradReg = lambda * thetaCopy.transpose() / m;
		T jDel = lambda / (2. * m) * (thetaCopy ^ 2).sum();
		cost += jDel;
		grad = gradNoReg + gradReg;
	}

	static T costFunctionNoReg(Matrix<T>& hThetaT, Matrix<T>& yT, int m) {
		return static_cast<T>((-1. / m)
				* (yT % hThetaT.log() + (1 - yT) % ((1 - hThetaT).log())).sum());
	}

	static T costFunction(Matrix<T>& a, Matrix<T>& y, T lambda,
			vector<Matrix<T> > thetas) {
		uint m = y.m;
		Matrix<T> yb =
				y.isBinaryCategoryMatrix() ? y : y.toBinaryCategoryMatrix();
		T jDel = 0.;
		if (lambda != 0) {
			uint i = 0;
			typedef typename vector<Matrix<T> >::iterator iterator;
			for (iterator it = thetas.begin(); it < thetas.end(); it++) {
				Matrix<T> thetaCopy = (*it).dropFirst();
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

	static T costFunctionNoReg2(Matrix<T>& hThetaT, Matrix<T>& yT, int m) {
		Matrix<T> yThThetaLog = yT % hThetaT.log();
		outln("yThThetaLog");
		outln(yThThetaLog.toString());
		Matrix<T> oneMinusyTh = (1 - yT) % ((1 - hThetaT).log());
		outln("oneMinusyTh");
		outln(oneMinusyTh.toString());
		Matrix<T> onMin = 1 - yT;
		outln("onMin");
		outln(onMin.toString());
		Matrix<T> onMinh = 1 - hThetaT;
		outln("onMinh");
		outln(onMinh.toString());
		Matrix<T> lonminh = onMinh.log();
		outln("lonminh");
		outln(lonminh.toString());
		Matrix<T> ha = onMin.hadamardProduct(lonminh);
		outln("ha");
		outln(ha.toString());
		Matrix<T> ha2 = onMin.binaryOp(lonminh, plusBinaryOp<T>());
		outln("ha2");
		outln(ha2.toString());
		return static_cast<T>((-1. / m)
				* (yThThetaLog + (1 - yT) % ((1 - hThetaT).log())).sum());
	}

	template<typename CostFunction> static Matrix<T> gradientApprox(
			CostFunction costFn, Matrix<T> theta, T epsilon) {
		uint l = theta.m * theta.n;
		uint i = 0;
		Matrix<T> perturb = Matrix<T>::zeros(theta.dims());
		Matrix<T> gradApprox = Matrix<T>::zeros(theta.dims());
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

template void LogisticRegression<float>::gradCostFunction(
		Matrix<float>& grad, float& cost,
		const Matrix<float>& x, const Matrix<float>& y, const Matrix<float>& theta, float lambda);

template void LogisticRegression<double>::gradCostFunction(
		Matrix<double>& grad, double& cost,
		const Matrix<double>& x, const Matrix<double>& y, const Matrix<double>& theta,
		double lambda);
