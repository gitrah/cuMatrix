/*
 * LinearRegression.cc
 *
 *  Created on: Sep 27, 2013
 *      Author: reid
 */

#include "LinearRegression.h"

template CuMatrix<float> LinearRegression<float>::gradientDescentL(float,
		CuMatrix<float>&, CuMatrix<float>&, CuMatrix<float>&, CuMatrix<float>&,
		int);

template CuMatrix<double> LinearRegression<double>::gradientDescentL(double,
		CuMatrix<double>&, CuMatrix<double>&, CuMatrix<double>&,
		CuMatrix<double>&, int);

template<typename T> CuMatrix<T> LinearRegression<T>::gradientDescentL(T alpha, CuMatrix<T>& theta, CuMatrix<T>& x, CuMatrix<T>& y,CuMatrix<T>& jHistory, int iters) {
	CuMatrix<T> grad = CuMatrix<T>::zeros(theta.m,theta.n);
	const DMatrix<T> dx = x.asDmatrix();
	const DMatrix<T> dy = y.asDmatrix();
	const DMatrix<T> dtheta = theta.asDmatrix();
	DMatrix<T> dgrad = grad.asDmatrix();
	DMatrix<T> djHistory = jHistory.asDmatrix();

	//gradientDescentCu<<<1,1>>>(dgrad, alpha,djHistory,dtheta,dx,dy,iters);
	cudaDeviceSynchronize();
	grad.invalidateHost();

	return grad;
}
/*

template void __global__ gradientDescentCu(
		DMatrix<float>, float, DMatrix<float>, DMatrix<float>, DMatrix<float>, DMatrix<float>,
		int);

template void __global__ gradientDescentCu(
		DMatrix<double>,double, DMatrix<double>, DMatrix<double>, DMatrix<double>,
		DMatrix<double>, int);

template<typename T> void __global__ gradientDescentCu(
		DMatrix<T> dgrad, T alpha, DMatrix<T> djHistory, DMatrix<T> dtheta, DMatrix<T> dx, DMatrix<T> dy,
		 int iters) {
	CuMatrix<T> x(dx);
	CuMatrix<T> y(dy);
	CuMatrix<T> theta(dtheta);
	CuMatrix<T> grad(dgrad);
	int m = y.size / sizeof(T);
	const T alphaOverM = alpha / m;
	int n = theta.size / sizeof(T);
	CuMatrix<T> jHist = CuMatrix<T>::zeros(iters, 1);
	//outln("theta " << theta.toShortString());
	//theta.syncBuffers();
	//outln("x " << x.toShortString());
	CuMatrix<T> xb = x.addBiasColumn();
	//outln("xb " << xb.toShortString());
	CuMatrix<T> yt = y.transpose();
	//outln("yt " << yt.toShortString());
	CuMatrix<T> hthetaX = theta.transpose() * xb.transpose();
	//outln("hthetaX " << hthetaX.toShortString());

	CuMatrix<T> tempTheta = CuMatrix<T>::zeros(theta.m, theta.n);

	CuMatrix<T> xbTheta(xb.m, theta.n, true, true);
	DMatrix<T> dxbTheta;
	xbTheta.asDmatrix(dxbTheta, false);
	CuMatrix<T> txXbTheta = xbTheta.transpose();
	DMatrix<T> dTxbTheta = txXbTheta.asDmatrix(false);

	DMatrix<T> dTheta;
	theta.asDmatrix(dTheta, true);

	CuMatrix<T> xbTrans = xb.transpose();
	DMatrix<T> dxbTrans = xbTrans.asDmatrix(true);

	CuMatrix<T> scalar = CuMatrix<T>::zeros(1, 1);
	DMatrix<T> dscalar = scalar.asDmatrix();

	const DMatrix<T> dxb = xb.asDmatrix();

	dim3 block(16, 16);

	CuMatrix<T> subRes(y.m, y.n, true, true);
	DMatrix<T> dsubRes = subRes.asDmatrix(false);

	minusBinaryOp<T> minOp;
	plusBinaryOp<T> plusOp;
	for (int i = 0; i < iters; i++) {
		//T cost = costFunctionNoRegNu(hthetaX, yt, m);
		T col;
		//outln(i << "th theta " << theta.get(0,0) << ", " << theta.get(1,0));
		for (int j = 0; j < n; j++) {
			if (j == 0) {
				//col = theta.get(j,0) -  alpha/m * (xb * theta - y).sum();
				// \/ xb * theta
				xbTheta.matrixProductKPtrL(dxbTheta, matrixProductKernel3, dxb,
						dTheta, &block);
				// \/ - y
				xbTheta.binaryOpL(dsubRes, dxbTheta, dy, minOp);
				// \/  theta.get(j,0) - sum * alhpa/m
				col = theta.elements[j]
						- alphaOverM * subRes.reduce(dsubRes, plusOp, 0);
			} else {
				//col = theta.get(j,0) -  alpha/m * ((xb * theta - y)' * x).sum();
				// \/ xb * theta
				xbTheta.matrixProductKPtrL(dxbTheta, matrixProductKernel3, dxb,
						dTheta, &block);
				//outln("dxbTheta " << util<T>::pdm(dxbTheta));
				// \/ - y
				xbTheta.binaryOpL(dsubRes, dxbTheta, dy, minOp);
				//outln("dsubRes " << util<T>::pdm(dsubRes));
				// transpose
				xbTheta.transposeL(dTxbTheta, dsubRes);
				//outln("dTxbTheta " << util<T>::pdm(dTxbTheta));
				//outln("dx " << util<T>::pdm(dx));
				// * x)
				x.matrixProductKPtrL(dscalar, matrixProductKernel3, dTxbTheta,
						dx, &block);
				//outln("dscalar ");
				//outln("scalar " << scalar.get(0, 0));
				col = theta.elements[j] - alphaOverM * scalar.get(0, 0);
				//outln("col (j!=0) " << col);
			}
			//tempTheta.set(j, col);
			tempTheta.d_elements[j] = col;
			//outln("tempTheta.set(j, col) ");
		}
		for (int j = 0; j < n; j++) {
			theta.d_elements[j] = tempTheta.d_elements[j];
		}
		//outln("copying " << tempTheta.toShortString() << " to " << theta.toShortString());
		//tempTheta.syncBuffers().copy(theta, 0, 0);

		//theta.syncBuffers();
		//outln(i << "th theta " << theta.syncBuffers());

		//hthetaX = theta.transpose() * xb.transpose();
		//jHistory.set(i, costFunctionNoRegNu(hthetaX, yt, m));
	}
	hthetaX = theta.transpose() * xbTrans;

	grad = theta - alphaOverM * ((hthetaX - yt) * xb).sum();
}
*/

/*template CuMatrix<float> LinearRegression<float>::gradientDescentLoop(float,
		CuMatrix<float>&, CuMatrix<float>&, CuMatrix<float>&, CuMatrix<float>&,
		int);

template CuMatrix<double> LinearRegression<double>::gradientDescentLoop(double,
		CuMatrix<double>&, CuMatrix<double>&, CuMatrix<double>&,
		CuMatrix<double>&, int);
template<typename T> CuMatrix<T> LinearRegression<T>::gradientDescentLoop(
		T alpha, CuMatrix<T>& theta, CuMatrix<T>& x, CuMatrix<T>& y,
		CuMatrix<T>& jHistory, int iters) {
	int m = y.size / sizeof(T);
	const T alphaOverM = alpha / m;
	int n = theta.size / sizeof(T);
	CuMatrix<T> jHist = CuMatrix<T>::zeros(iters, 1);
	outln("theta " << theta.toShortString());
	theta.syncBuffers();
	outln("x " << x.toShortString());
	CuMatrix<T> xb = x.addBiasColumn();
	outln("xb " << xb.toShortString());
	CuMatrix<T> yt = y.transpose();
	outln("yt " << yt.toShortString());
	CuMatrix<T> hthetaX = theta.transpose() * xb.transpose();
	outln("hthetaX " << hthetaX.toShortString());

	CuMatrix<T> tempTheta = CuMatrix<T>::zeros(theta.m, theta.n);

	CuMatrix<T> xbTheta(xb.m, theta.n, true, true);
	DMatrix<T> dxbTheta;
	xbTheta.asDmatrix(dxbTheta, false);
	CuMatrix<T> txXbTheta = xbTheta.transpose();
	DMatrix<T> dTxbTheta = txXbTheta.asDmatrix(false);

	DMatrix<T> dTheta;
	theta.asDmatrix(dTheta, true);

	CuMatrix<T> xbTrans = xb.transpose();
	DMatrix<T> dxbTrans = xbTrans.asDmatrix(true);

	CuMatrix<T> scalar = CuMatrix<T>::zeros(1, 1);
	DMatrix<T> dscalar = scalar.asDmatrix();

	const DMatrix<T> dxb = xb.asDmatrix();
	const DMatrix<T> dx = x.asDmatrix();
	const DMatrix<T> dy = y.asDmatrix();

	dim3 block(16, 16);

	CuMatrix<T> subRes(y.m, y.n, true, true);
	DMatrix<T> dsubRes = subRes.asDmatrix(false);

	minusBinaryOp<T> minOp;
	plusBinaryOp<T> plusOp;
	void (*matProdKptr[])(DMatrix<T>, const DMatrix<T>, const DMatrix<T>,
			int) = {matrixProductBandwidthKernel,matrixProductKernel,matrixProductKernel2,matrixProductKernel3,matrixProductKernelTxdB2};
	CuTimer kTimer;
	for (int kernel = 0; kernel < 3; kernel++) {
		kTimer.start();
		for (int i = 0; i < iters; i++) {
			//T cost = costFunctionNoRegNu(hthetaX, yt, m);
			T col;
			//outln(i << "th theta " << theta.get(0,0) << ", " << theta.get(1,0));
			for (int j = 0; j < n; j++) {
				if (j == 0) {
					//col = theta.get(j,0) -  alpha/m * (xb * theta - y).sum();
					// \/ xb * theta
					xbTheta.matrixProductKPtrL(dxbTheta, matProdKptr[kernel],
							dxb, dTheta, &block);
					// \/ - y
					xbTheta.binaryOpL(dsubRes, dxbTheta, dy, minOp);
					// \/  theta.get(j,0) - sum * alhpa/m
					col = theta.elements[j]
							- alphaOverM * subRes.reduce(dsubRes, plusOp, 0);
				} else {
					//col = theta.get(j,0) -  alpha/m * ((xb * theta - y)' * x).sum();
					// \/ xb * theta
					xbTheta.matrixProductKPtrL(dxbTheta, matProdKptr[kernel],
							dxb, dTheta, &block);
					//outln("dxbTheta " << util<T>::pdm(dxbTheta));
					// \/ - y
					xbTheta.binaryOpL(dsubRes, dxbTheta, dy, minOp);
					//outln("dsubRes " << util<T>::pdm(dsubRes));
					// transpose
					xbTheta.transposeL(dTxbTheta, dsubRes);
					//outln("dTxbTheta " << util<T>::pdm(dTxbTheta));
					//outln("dx " << util<T>::pdm(dx));
					// * x)
					x.matrixProductKPtrL(dscalar, matProdKptr[kernel],
							dTxbTheta, dx, &block);
					//outln("dscalar ");
					//outln("scalar " << scalar.get(0, 0));
					col = theta.elements[j] - alphaOverM * scalar.get(0, 0);
					//outln("col (j!=0) " << col);
				}
				tempTheta.set(j, col);
				//outln("tempTheta.set(j, col) ");
			}
			//outln("copying " << tempTheta.toShortString() << " to " << theta.toShortString());
			tempTheta.syncBuffers().copy(theta, 0, 0);
			//theta.syncBuffers();
			//outln(i << "th theta " << theta.syncBuffers());

			//hthetaX = theta.transpose() * xb.transpose();
			//jHistory.set(i, costFunctionNoRegNu(hthetaX, yt, m));
		}
		outln("kernel " << kernel << " took " << kTimer.stop());
	}
	hthetaX = theta.transpose() * xbTrans;
	return theta - alphaOverM * ((hthetaX - yt) * xb).sum();
}

template<typename T> T LinearRegression<T>::costFunctionNoReg(
		CuMatrix<T>& hThetaTxT, CuMatrix<T>& yT, int m) {
	return (1. / (2 * m)) * ((hThetaTxT - yT) ^ 2).sum();
}

template<typename T> T LinearRegression<T>::costFunction(CuMatrix<T>& a,
		CuMatrix<T>& y, T lambda, vector<CuMatrix<T> > thetas) {
	uint m = y.m;
	T jDel = 0.;
	if (lambda != 0) {
		uint i = 0;
		typedef typename vector<CuMatrix<T> >::iterator iterator;
		for (iterator it = thetas.begin(); it < thetas.end(); it++) {
			CuMatrix<T> thetaCopy = (*it).dropFirst();
			T jdeldel = lambda / (2. * m) * thetaCopy.autoDot();
			outln(i << " jdeldel " << jdeldel);
			jDel += jdeldel;
			i += 1;
		}
	}
	T jNoReg = costFunctionNoReg(a, y, m);
	outln("jNoReg " << jNoReg);
	return (jNoReg + jDel);
}

template<typename T> T LinearRegression<T>::costFunctionNoReg2(
		CuMatrix<T>& hThetaT, CuMatrix<T>& yT, int m) {
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
	CuMatrix<T> ha2 = onMin.binaryOp(lonminh, plusBinaryOp<T>());
	outln("ha2");
	outln(ha2.toString());
	return static_cast<T>((-1. / m)
			* (yThThetaLog + (1 - yT) % ((1 - hThetaT).log())).sum());
}

template<typename T> template<typename CostFunction> CuMatrix<T> LinearRegression<
		T>::gradientApprox(CostFunction costFn, CuMatrix<T> theta, T epsilon) {
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
}*/
