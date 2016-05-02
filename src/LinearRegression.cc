/*
 * LinearRegression.cc
 *
 *  Created on: Sep 27, 2013
 *      Author: reid
 */

#include "LinearRegression.h"
#include "Kernels.h"

template void LinearRegression<float>::gradCostFunction(CuMatrix<float>& grad,
		float& cost, const CuMatrix<float>& x, const CuMatrix<float>& y,
		const CuMatrix<float>& theta, float lambda);

template void LinearRegression<double>::gradCostFunction(CuMatrix<double>& grad,
		double& cost, const CuMatrix<double>& x, const CuMatrix<double>& y,
		const CuMatrix<double>& theta, double lambda);
template void LinearRegression<ulong>::gradCostFunction(CuMatrix<ulong>& grad,
		ulong& cost, const CuMatrix<ulong>& x, const CuMatrix<ulong>& y,
		const CuMatrix<ulong>& theta, ulong lambda);

/*
 * x is input m (samples) * n (variables) matrix
 * y is target
 * theta 1 * n (parameters of hypothesis functions) hTheta(x) ( = y) = theta dot x
 */
template<typename T> void LinearRegression<T>::gradCostFunction(
		CuMatrix<T>& grad, T& cost, const CuMatrix<T>& x, const CuMatrix<T>& y,
		const CuMatrix<T>& theta, T lambda) {
	const int m = y.m;
	CuMatrix<T> hThetaTx = theta.transpose() * x.transpose();
	outln("hThetaTx");
	outln(hThetaTx.toString());
	CuMatrix<T> yT = y.transpose();
	cost = costFunctionNoReg(hThetaTx, yT, m);
	outln("cost " << cost);
	CuMatrix<T> gradNoReg = ((hThetaTx - yT) * x) / m;
	CuMatrix<T> thetaCopy(theta);
	thetaCopy.set(0, 0, 0); // elements[0] = 0;
	CuMatrix<T> gradReg = lambda * thetaCopy.transpose() / m;
	T jDel = lambda / (2. * m) * (thetaCopy ^ ((T)2)).sum();
	cost += jDel;
	grad = gradNoReg + gradReg;
}

// operator% => hadamardProduct (elementwise product)
template float LinearRegression<float>::costFunctionNoRegNu(CuMatrix<float>&,
		CuMatrix<float>&, int);

template double LinearRegression<double>::costFunctionNoRegNu(CuMatrix<double>&,
		CuMatrix<double>&, int);
template ulong LinearRegression<ulong>::costFunctionNoRegNu(CuMatrix<ulong>&,
		CuMatrix<ulong>&, int);
template<typename T> T LinearRegression<T>::costFunctionNoRegNu(
		CuMatrix<T>& hThetaTxT, CuMatrix<T>& yT, int m) {
	return ((hThetaTxT - yT) ^ ((T)2)).sum() / (2 * m);
}

template CuMatrix<float> LinearRegression<float>::gradientDescent(float,
		CuMatrix<float>&, CuMatrix<float>&, CuMatrix<float>&, CuMatrix<float>&,
		int);

template CuMatrix<double> LinearRegression<double>::gradientDescent(double,
		CuMatrix<double>&, CuMatrix<double>&, CuMatrix<double>&,
		CuMatrix<double>&, int);

template CuMatrix<ulong> LinearRegression<ulong>::gradientDescent(ulong,
		CuMatrix<ulong>&, CuMatrix<ulong>&, CuMatrix<ulong>&,
		CuMatrix<ulong>&, int);

template<typename T> CuMatrix<T> LinearRegression<T>::gradientDescent(T alpha,
		CuMatrix<T>& theta, CuMatrix<T>& x, CuMatrix<T>& y,
		CuMatrix<T>& jHistory, int iters) {
	int m = y.size / sizeof(T);
	const T alphaOverM = alpha / m;
	int n = theta.size / sizeof(T);
	CuMatrix<T> jHist = CuMatrix<T>::zeros(iters, 1);
	outln("theta " << theta.toShortString());
	theta.syncBuffers();
	assert(x.tiler.tileSize == x.tiler.m_size);
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
	xbTheta.tile0(dxbTheta, false);
	CuMatrix<T> txXbTheta = xbTheta.transpose();
	DMatrix<T> dTxbTheta = txXbTheta.asDmatrix(false);

	DMatrix<T> dTheta;
	theta.tile0(dTheta, true);

	CuMatrix<T> xbTrans = xb.transpose();
	//DMatrix<T> dxbTrans = xbTrans.asDmatrix(true);

	CuMatrix<T> scalar = CuMatrix<T>::zeros(1, 1);
	DMatrix<T> dscalar = scalar.asDmatrix();

	const DMatrix<T> dxb = xb.asDmatrix();
	const DMatrix<T> dx = x.asDmatrix();
	const DMatrix<T> dy = y.asDmatrix();

	dim3 block(16, 16);

	CuMatrix<T> subRes(y.m, y.n, true, true);
	DMatrix<T> dsubRes = subRes.asDmatrix(false);

	minusBinaryOp<T> minOp = Functory<T,minusBinaryOp>::pinch();
	plusBinaryOp<T> plusOp = Functory<T,plusBinaryOp>::pinch();

	// todo this is really slow
	for (int i = 0; i < iters; i++) {
		//T cost = costFunctionNoRegNu(hthetaX, yt, m);
		T col;
		//outln(i << "th theta " << theta.get(0,0) << ", " << theta.get(1,0));
		for (int j = 0; j < n; j++) {
			if (j == 0) {
				//col = theta.get(j,0) -  alpha/m * (xb * theta - y).sum();
				// \/ xb * theta
				matrixProductKPtrL(dxbTheta, matrixProductKernel3, dxb,
						dTheta, &block);
				// \/ - y
				binaryOpL(dsubRes, dxbTheta, dy, minOp);
				// \/  theta.get(j,0) - sum * alhpa/m
				col = theta.elements[j]
						- alphaOverM * subRes.reduce(dsubRes, plusOp, 0);
			} else {
				//col = theta.get(j,0) -  alpha/m * ((xb * theta - y)' * x).sum();
				// \/ xb * theta
				matrixProductKPtrL(dxbTheta, matrixProductKernel3, dxb,
						dTheta, &block);
				//outln("dxbTheta " << util<T>::pdm(dxbTheta));
				// \/ - y
				binaryOpL(dsubRes, dxbTheta, dy, minOp);
				//outln("dsubRes " << util<T>::pdm(dsubRes));
				// transpose
				xbTheta.transposeL(dTxbTheta, dsubRes);
				//outln("dTxbTheta " << util<T>::pdm(dTxbTheta));
				//outln("dx " << util<T>::pdm(dx));
				// * x)
				matrixProductKPtrL(dscalar, matrixProductKernel3, dTxbTheta,
						dx, &block);
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
	hthetaX = theta.transpose() * xbTrans;
	return theta - alphaOverM * ((hthetaX - yt) * xb).sum();
}

template CuMatrix<float> LinearRegression<float>::gradientDescentLoop(float,
		CuMatrix<float>&, CuMatrix<float>&, CuMatrix<float>&, CuMatrix<float>&,
		int);
template CuMatrix<double> LinearRegression<double>::gradientDescentLoop(double,
		CuMatrix<double>&, CuMatrix<double>&, CuMatrix<double>&,
		CuMatrix<double>&, int);
template CuMatrix<ulong> LinearRegression<ulong>::gradientDescentLoop(ulong,
		CuMatrix<ulong>&, CuMatrix<ulong>&, CuMatrix<ulong>&,
		CuMatrix<ulong>&, int);
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
	assert(x.tiler.tileSize == x.tiler.m_size);
	CuMatrix<T> xb = x.addBiasColumn();
	outln("xb " << xb.toShortString());
	CuMatrix<T> yt = y.transpose();
	outln("yt " << yt.toShortString());
	CuMatrix<T> hthetaX = theta.transpose() * xb.transpose();
	outln("hthetaX " << hthetaX.toShortString());

	CuMatrix<T> tempTheta = CuMatrix<T>::zeros(theta.m, theta.n);

	CuMatrix<T> xbTheta(xb.m, theta.n, true, true);
	DMatrix<T> dxbTheta;
	xbTheta.tile0(dxbTheta,   false);
	CuMatrix<T> txXbTheta = xbTheta.transpose();
	DMatrix<T> dTxbTheta = txXbTheta.asDmatrix(false);

	DMatrix<T> dTheta;
	theta.tile0(dTheta,  true);

	CuMatrix<T> xbTrans = xb.transpose();

	CuMatrix<T> scalar = CuMatrix<T>::zeros(1, 1);
	DMatrix<T> dscalar = scalar.asDmatrix();

	const DMatrix<T> dxb = xb.asDmatrix();
	const DMatrix<T> dx = x.asDmatrix();
	const DMatrix<T> dy = y.asDmatrix();

	dim3 block(16, 16);

	CuMatrix<T> subRes(y.m, y.n, true, true);
	DMatrix<T> dsubRes = subRes.asDmatrix(false);

	minusBinaryOp<T> minOp = Functory<T,minusBinaryOp>::pinch();
	plusBinaryOp<T> plusOp = Functory<T,plusBinaryOp>::pinch();

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
					matrixProductKPtrL(dxbTheta, matProdKptr[kernel],
							dxb, dTheta, &block);
					// \/ - y
					binaryOpL(dsubRes, dxbTheta, dy, minOp);
					// \/  theta.get(j,0) - sum * alhpa/m
					col = theta.elements[j]
							- alphaOverM * subRes.reduce(dsubRes, plusOp, 0);
				} else {
					//col = theta.get(j,0) -  alpha/m * ((xb * theta - y)' * x).sum();
					// \/ xb * theta
					matrixProductKPtrL(dxbTheta, matProdKptr[kernel],
							dxb, dTheta, &block);
					//outln("dxbTheta " << util<T>::pdm(dxbTheta));
					// \/ - y
					binaryOpL(dsubRes, dxbTheta, dy, minOp);
					//outln("dsubRes " << util<T>::pdm(dsubRes));
					// transpose
					xbTheta.transposeL(dTxbTheta, dsubRes);
					//outln("dTxbTheta " << util<T>::pdm(dTxbTheta));
					//outln("dx " << util<T>::pdm(dx));
					// * x)
					matrixProductKPtrL(dscalar, matProdKptr[kernel],
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
	return (1. / (2 * m)) * ((hThetaTxT - yT) ^ ((T)2)).sum();
}

template<typename T> T LinearRegression<T>::costFunction(CuMatrix<T>& a,
		CuMatrix<T>& y, T lambda, vector<CuMatrix<T> > thetas) {
	int m = y.m;
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
	CuMatrix<T> ha2 = onMin.binaryOp(lonminh, Functory<T,plusBinaryOp>::pinch());
	outln("ha2");
	outln(ha2.toString());
	return static_cast<T>((-1. / m)
			* (yThThetaLog + (1 - yT) % ((1 - hThetaT).log())).sum());
}

//template <typename T> template< template <typename> class KernelFunction >
template<typename T> template< template <typename> class CostFunction > CuMatrix<T> LinearRegression<
		T>::gradientApprox(CostFunction<T> costFn, CuMatrix<T> theta, T epsilon) {
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

