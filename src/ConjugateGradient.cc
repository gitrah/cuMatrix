#include "ConjugateGradient.h"
#include "NeuralNet.h"
#include "LogisticRegression.h"
#include <limits>

template<typename T>
bool ConjugateGradient<T>::nanQ(T value) {
	return value != value;
}

// requires #include <limits>
template<typename T>
bool ConjugateGradient<T>::infQ(T value) {
	return std::numeric_limits<T>::has_infinity
			&& (value == std::numeric_limits<T>::infinity()
					|| -value == std::numeric_limits<T>::infinity());
}

template<typename T>
bool ConjugateGradient<T>::realQ(T t) {
	return !(infQ(t) || nanQ(t));
}

template<typename T> void printCost(T* lastCost, char* buff, int iteration, T cost) {
	if(iteration > 0){
		T delta = (cost-lastCost[iteration-1])/lastCost[iteration-1];
		if(iteration > 3) {
			T lc = (2 * lastCost[iteration-1] + lastCost[iteration-2])/3;
			delta = (cost-lc)/lc;
		}
		printf("%4d | Cost: %4.6e delta %.5f%\r", iteration, cost, 100 * delta);
	}
}
template<> void printCost<ulong>(ulong* lastCost, char* buff, int iteration, ulong cost) {
	sprintf(buff, "%4d | Cost: %4.6lu\r", iteration, cost);
	lastCost[iteration] = cost;
}
/*
 * adapted from fmincg.m/octave
 * f: ( CuMatrix<T>& grad, T& cost, const CuMatrix<T>& x)
 */
template<typename T> template<typename CostFunction> std::pair<CuMatrix<T>,
		std::pair<CuMatrix<T>, int> > ConjugateGradient<T>::fmincg(CostFunction& f,
		CuMatrix<T>& x, int length, int red) {
	if(checkDebug(debugCg)) {
		outln("enter");
	}

	if(checkDebug(debugCg)) {
		flprintf("x %dx%d d %p h %p\n", x.m, x.n, x.tiler.currBuffer(), x.elements);
		flprintf("f.y %dx%d d %p h %p\n", f.y.m, f.y.n, f.y.tiler.currBuffer(), f.y.elements);
	}
	CuTimer timer;
	if(checkDebug(debugUseTimers)) {
		timer.start();
	}
	char buff[1024];
	T a = 0;
	T b = 0;
	CuMatrix<T> x0 ;
	CuMatrix<T> df0 = CuMatrix<T>::zeros(x.m,x.n);//zerom;
	CuMatrix<T> df1 = CuMatrix<T>::zeros(x.m,x.n);//zerom;
	CuMatrix<T> df2 ;
	if(checkDebug(debugNn)) {
		outln("x0 "<< x0.toShortString());
		outln("df0 "<< df0.toShortString());
		outln("df1 "<< df1.toShortString());
		outln("df2 "<< df2.toShortString());
	}
	T f0 = 0;
	T f1 = 0;
	T* lastCost = new T[length];
	T f2 = 0;
	T f3 = 0;
	CuMatrix<T> s ;
	T d1 = 0;
	T d2 = 0;
	T d3 = 0;
	T z1 = 0;
	T z2 = 0;
	T z3 = 0;
	bool success = false;
	bool outerLoop = true;
	T limit = 0;
	int i = 0;
	bool ls_failed = false;
	CuMatrix<T> fX ;
/*
	if(checkDebug(debugCg))outln("limits min " << std::numeric_limits<T>::min());
	if(checkDebug(debugCg))outln("0 x.sum() " << x.sum());
	if(checkDebug(debugCg))outln("0 x.ss() " << x.toShortString());
	if(checkDebug(debugCg))outln("f1 " << f1);
*/
	if(checkDebug(debugCg)){
		outln("0.0 f");
		//outln("0.0 x " << x.syncBuffers());
		//outln("0.0 x+x " << (x+x).syncBuffers());
	}
	f(df1, f1, x);
	if(checkDebug(debugCg)){
		//outln("0.0 df1 " << df1.syncBuffers());
		outln("0.0 f1 " << f1 );
	}
	if (length > 0)
		i++;
	anyErr();
	s = df1.negate();
	if(checkDebug(debugCg))outln("s after df1 " << s.toShortString());
	//d1 = -s.autoDot();
	d1 = - ( s.transpose() * s).toScalar();
	if(checkDebug(debugCg))outln("init df1.sum " << df1.sum() << ", s.sum " << s.sum() << ", d1 " << d1);

	z1 = red / (1 - d1);

	while (outerLoop && i < abs(length)) {
		//cout << x.getMgr().stats();
		if(checkDebug(debugCg))b_util::usedDmem();
		if (length > 0)
			i++;
		if(checkDebug(debugCg))outln("x bef copy " << x.toShortString());
		if(checkDebug(debugCg))outln("x0 bef copy " << x0.toShortString());
		x0 = x.copy(true);
		if(checkDebug(debugCg))outln("x0 after copy " << x0.toShortString());
		f0 = f1;
		df0 = df1.copy(true);
		if(checkDebug(debugCg)){
			outln("i  " << i);
			outln(" z1 " << z1 << "\n");
			if(s.size < 100 * (int)sizeof(T))
				outln(" s " << s.syncBuffers() << "\n");

			//outln("x " << x.toShortString());
			outln("s " << s.toShortString());
		}
		x = x + z1 * s;
		if(checkDebug(debugCg)){
			if(x.size < 100 * (int)sizeof(T)) {
				outln("0.1 x " << x.syncBuffers());
				outln("0.1 x+x " << (x+x).syncBuffers());
			} else {
				outln("x.1 ss " << x.toShortString());
			}
			flprintf("0.1 x.sum %.20g\n", (double)x0.sum());
		}
		f(df2,f2, x);
		if(checkDebug(debugCg)){
			//outln("0.1 df2\n" << df2.syncBuffers() );
			outln("0.1 f2 " << f2 << ", df2.sum " << df2.sum());
		}
		if (length < 0)
			i++;
		if(checkDebug(debugCg)){
			outln("df2 " << df2.dimsString());
			outln("s " << s.dimsString());
		}

		d2 = (df2.transpose() * s).toScalar();
		if(checkDebug(debugCg))outln("0.1 d2 " << d2);
		f3 = f1;
		d3 = d1;
		z3 = -z1;
		int m = (length > 0) ? ConjugateGradient<T>::max : MIN(ConjugateGradient<T>::max, (T)-length - i);
		success = false;
		limit = -1;
		bool innerLoop = true;
		while (innerLoop) {
			while (((f2 > f1 + z1 * rho * d1) || (d2 > -sig * d1)) && m > 0) {
				limit = z1;
				if (f2 > f1) {
					z2 = z3 - .5 * d3 * z3 * z3 / (d3 * z3 + f2 - f3);
					if(checkDebug(debugCg))outln("f2 > f1, z2 " << z2);
				} else {
					a = 6. * (f2 - f3) / z3 + 3. * (d2 + d3);
					b = 3. * (f3 - f2) - z3 * (d3 + 2. * d2);
					z2 = (sqrt(b * b - a * d2 * z3 * z3) - b) / a;
					if(checkDebug(debugCg))flprintf("a %g, b %g, z2 %g\n", (double)a, (double)b, (double)z2);
					//outln("f2 <= f1, a " << a << ", b " << b << ", z2 " << z2);
				}
				if (nanQ(z2) || infQ(z2)) {
					outln("bad z2");
					z2 = z3 / 2.;
				}
				z2 = MAX(MIN(z2, int0 * z3), (1 - int0) * z3);

				z1 = z1 + z2; // update the step;
				if(checkDebug(debugCg)){
					outln("1 z1 = z1 + z2 " << z1);
					//outln("0.2 x + z2 * s " << (x + z2 * s).syncBuffers());
				}
				x = x + z2 * s;
				if(checkDebug(debugCg)){
					//outln("0.2 x " << x.syncBuffers());
					outln("0.2 x.sum " << x.sum());
				}
				f(df2,f2,x);
				if(checkDebug(debugCg)){
					outln("0.2 f2 " << f2);
					//outln("0.2 df2 " << df2.syncBuffers());
				}
				m = m - 1;
				if (length < 0)
					i++;
				d2 = (df2.transpose() * s).toScalar();
				z3 -= z2;
				if(checkDebug(debugCg))outln("1 d2 " << d2 << ", z3 " << z3);
			}
			if ((f2 > f1 + z1 * rho * d1) || (d2 > -sig * d1)) {
				innerLoop = false;
			} else if (d2 > sig * d1) {
				success = true;
				innerLoop = false;
			} else if (m == 0) {
				innerLoop = false;
			} else {
				a = 6 * (f2 - f3) / z3 + 3 * (d2 + d3); // make cubic extrapolation
				b = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
				z2 = -d2 * z3 * z3 / (b + sqrt(b * b - a * d2 * z3 * z3)); // num. error possible - ok!
				if(checkDebug(debugCg))outln("a " << a << ", b " << b << ", z2 " << z2);
				if (!realQ(z2) || z2 < 0) // num prob or wrong sign?
					if (limit < -0.5) // if we have no upper limit
						z2 = z1 * (ext - 1.);
					// the extrapolate the maximum amount
					else
						z2 = (limit - z1) / 2.; // otherwise bisect
				else if ((limit > -0.5) && (z2 + z1 > limit)) // extraplation beyond max?
					z2 = (limit - z1) / 2.; // bisect
				else if ((limit < -0.5) & (z2 + z1 > z1 * ext)) // extrapolation beyond limit
					z2 = z1 * (ext - 1.0); // set to extrapolation limit
				else if (z2 < -z3 * int0)
					z2 = -z3 * int0;
				else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - int0))) // too close to limit?
					z2 = (limit - z1) * (1.0 - int0);
				if(checkDebug(debugCg))outln("z2b " << z2);
				if(checkDebug(debugCg))outln("f3 " << f3 << " f2 " << f2 << " d3 " << d3 );
				if(checkDebug(debugCg))outln(" d2 " << d2 << " z3 " << z3 << " z2 " << z2);

				// f3=f2
				f3 = f2; d3 = d2; z3 = -z2; // set point 3 equal to point 2
				z1 = z1 + z2;
				if(false && checkDebug(debugCg)){
					outln("2 z1 = z1 + z2 " << z1);
					outln("0.3a x " << x.syncBuffers());
					CuMatrix<T> xcopy = x.copy(true);
					CuMatrix<T> xones = CuMatrix<T>::ones(x.m,x.n);
					outln("x+x " << (x+x).syncBuffers());
					outln("x+xones " << (x+xones).syncBuffers());
					outln("xones+x " << (xones+x).syncBuffers());
					outln("xones+xones " << (xones+xones).syncBuffers());
					outln("xcopy+xcopy " << (xcopy+xcopy).syncBuffers());
					CuMatrix<T> xPlusz2S =  x + z2 * s;
					outln("xPlusz2S " << xPlusz2S.syncBuffers());
				}
				x = x + z2 * s; // update current estimates
				if(checkDebug(debugCg)){
					//outln("0.3 x " << x.syncBuffers());
					outln("0.3 x.sum " << x.sum());
				}
				f(df2,f2,x);
				if(checkDebug(debugCg)){
					outln("0.3 f2 " << f2);
					//outln("0.3 df2 " << df2.syncBuffers());
				}
				m = m - 1;
				if (length < 0)
					i++; // count epochs?!
				d2 = (df2.transpose() * s).toScalar();
			}
			if(checkDebug(debugCg))outln("repeat while (innerLoop)");
		}
		if(checkDebug(debugCg))outln("done innerLoop");
		if (success) { // if line search succeeded
			f1 = f2;
			if (fX.zeroDimsQ()) {
				fX = CuMatrix<T>::ones(1, 1) * f1;
			} else {
		        fX = (fX.transpose() |= (CuMatrix<T>::ones(1, 1) * f1)).transpose();
			}

			//sprintf(buff, "%4d | Cost: %4.6e\r", i, f1);
			//outln(i << " cost " << f1);
		    //printf("%s %4i | Cost: %4.6e   ... device %d\r", iterationMessage, i, f1, ExecCaps::currDev());
			//printCost<T>(buff, i, f1);
		    printCost<T>(lastCost, buff, i,f1);
			cout << buff << endl;

			//  s = s * (df2.autoDot() - (df1.tN() * df2).toScalar()) / (df1.autoDot) - df2 // Polack-Ribiere direction
			s = s * (df2.autoDot() - (df1.transpose() * df2).toScalar()) / df1.autoDot() - df2;
			// Polack-Ribiere direction
			CuMatrix<T> tmp = df1;
			df1 = df2;
			df2 = tmp;
			if(checkDebug(debugCg))outln("df1.sum " << df1.sum() << ", df2.sum " << df2.sum() << ", tmp.sum() " << tmp.sum());

			// swap derivatives
			d2 = (df1.transpose() * s).toScalar();
			if(checkDebug(debugCg))outln("P-R d2 " << d2);
			if (d2 > 0) { // new slope must be negative
				s = df1 * -1; // otherwise use steepest direction
				d2 = (s.transpose() * s * (static_cast<T>( -1.))).toScalar();
			}
			z1 = z1 * MIN(ratio, d1 / (d2 - MinPositiveValue)); // slope ratio but max RATIO
			d1 = d2;
			ls_failed = false; // this line search did not fail
		} else {
			x = x0;
			f1 = f0;
			df1 = df0; // restore point from before failed line search
			if (ls_failed || i > abs(length)) { // line search failed twice in a row
				outerLoop = false; // or we ran out of time, so we give up
				if(checkDebug(debugCg))outln("ls_failed " << ls_failed << ", abs(length) " << abs(length) << " i " << i);
			} else {
				CuMatrix<T> tmp = df1;
				df1 = df2;
				df2 = tmp; // swap derivatives
				s = df1 * -1.; // try steepest
				d1 = (s.transpose() * s).toScalar() * -1;
				z1 = 1. / (1. - d1);
				ls_failed = true; // this line search failed
				if(checkDebug(debugCg))outln("failed first time");
			}
		}
		if(i < length)
			lastCost[i] = f1;

	}
	if(checkDebug(debugUseTimers)) {
		outln("leaving fmincg, took " << timer.stop());
	} else 	{
		outln("leaving fmincg");
	}
	outln("x " << x.toShortString());
	outln("fX " << fX.toShortString());
	delete[] lastCost;
	return std::pair<CuMatrix<T>, std::pair<CuMatrix<T>, int> >(x,
			std::pair<CuMatrix<T>, int>(fX, i));
}
template class ConjugateGradient<float>;
template class ConjugateGradient<double>;
template class ConjugateGradient<ulong>;

template std::pair<CuMatrix<float>, std::pair<CuMatrix<float>, int> > ConjugateGradient<float>::fmincg<nnCostFtor<float> >(nnCostFtor<float> &, CuMatrix<float>&, int, int);
template std::pair<CuMatrix<double>, std::pair<CuMatrix<double>, int> > ConjugateGradient<double>::fmincg<nnCostFtor<double> >(nnCostFtor<double> &, CuMatrix<double>&, int, int);
template std::pair<CuMatrix<ulong>, std::pair<CuMatrix<ulong>, int> > ConjugateGradient<ulong>::fmincg<nnCostFtor<ulong> >(nnCostFtor<ulong> &, CuMatrix<ulong>&, int, int);

template std::pair<CuMatrix<float>, std::pair<CuMatrix<float>, int> > ConjugateGradient<float>::fmincg<nnCostFtorPm<float> >(nnCostFtorPm<float> &, CuMatrix<float>&, int, int);
template std::pair<CuMatrix<double>, std::pair<CuMatrix<double>, int> > ConjugateGradient<double>::fmincg<nnCostFtorPm<double> >(nnCostFtorPm<double> &, CuMatrix<double>&, int, int);
template std::pair<CuMatrix<ulong>, std::pair<CuMatrix<ulong>, int> > ConjugateGradient<ulong>::fmincg<nnCostFtorPm<ulong> >(nnCostFtorPm<ulong> &, CuMatrix<ulong>&, int, int);

template std::pair<CuMatrix<float>, std::pair<CuMatrix<float>, int> > ConjugateGradient<float>::fmincg<logRegCostFtor<float> >(logRegCostFtor<float>&, CuMatrix<float>&, int, int);
template std::pair<CuMatrix<double>, std::pair<CuMatrix<double>, int> > ConjugateGradient<double>::fmincg<logRegCostFtor<double> >(logRegCostFtor<double>&, CuMatrix<double>&, int, int);
template std::pair<CuMatrix<ulong>, std::pair<CuMatrix<ulong>, int> > ConjugateGradient<ulong>::fmincg<logRegCostFtor<ulong> >(logRegCostFtor<ulong>&, CuMatrix<ulong>&, int, int);

template <> double ConjugateGradient<double>::MinPositiveValue =  2.225073858507201e-308;
template <> float ConjugateGradient<float>::MinPositiveValue = 1.4E-45;
template <> ulong ConjugateGradient<ulong>::MinPositiveValue = 1;

