#include "Matrix.h"
#include "ConjugateGradient.h"
#include "NeuralNet.h"
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
/*
 * f: (MatrixD) => (Double, MatrixD)
 */
template<typename T> template<typename CostFunction> std::pair<Matrix<T>,
		std::pair<Matrix<T>, int> > ConjugateGradient<T>::fmincg(CostFunction& f,
		Matrix<T>& x, int length, int red) {

	char buff[1024];
	T a = 0;
	T b = 0;
	Matrix<T> x0 = zerom;
	Matrix<T> df0 = zerom;
	Matrix<T> df1 = zerom;
	Matrix<T> df2 = zerom;
	T f0 = 0;
	T f1 = 0;
	T f2 = 0;
	T f3 = 0;
	Matrix<T> s = zerom;
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
	Matrix<T> fX = zerom;
	if(debugCg)outln("limits min " << std::numeric_limits<T>::min());
	if(debugCg)outln("0 x.sum() " << x.sum());
	if(debugCg)outln("0 x.ss() " << x.toShortString());
	if(debugCg)outln("f1 " << f1);
	f(df1, f1, x);
	if (length > 0)
		i++;
	anyErr();
	s = df1.negate();
	if(debugCg)outln("s after df1 " << s.toShortString());
	d1 = -s.autoDot();
	if(debugCg)outln("init df1.sum " << df1.sum() << ", s.sum " << s.sum() << ", d1 " << d1);

	z1 = red / (1 - d1);

	while (outerLoop && i < abs(length)) {
		cout << x.getMgr().stats();
		b_util::usedDmem();
		if (length > 0)
			i++;
		x0 = x.copy(true);
		f0 = f1;
		df0 = df1.copy(true);
		tout("i  " << i);
		ot(" z1 " << z1 << "\n");
		x = x + z1 * s;
		if(debugCg)outln("0.1 x.sum " << x.sum());
		f(df2,f2, x);
		if(debugCg)outln("f2 " << f2 << ", df2.sum " << df2.sum());
		if (length < 0)
			i++;
		d2 = (df2.transpose() * s).toScalar();
		if(debugCg)outln("d2 " << d2);
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
					if(debugCg)outln("f2 > f1, z2 " << z2);
				} else {
					a = 6. * (f2 - f3) / z3 + 3. * (d2 + d3);
					b = 3. * (f3 - f2) - z3 * (d3 + 2. * d2);
					z2 = (sqrt(b * b - a * d2 * z3 * z3) - b) / a;
					if(debugCg)outln("f2 <= f1, a " << a << ", b " << b << ", z2 " << z2);
				}
				if (nanQ(z2) || infQ(z2)) {
					outln("bad z2");
					z2 = z3 / 2.;
				}
				z2 = MAX(MIN(z2, int0 * z3), (1 - int0) * z3);

				z1 = z1 + z2; // update the step;
				if(debugCg)outln("1 z1 = z1 + z2 " << z1);
				x = x + z2 * s;
				f(df2,f2,x);
				m = m - 1;
				if (length < 0)
					i++;
				d2 = (df2.transpose() * s).toScalar();
				z3 -= z2;
				if(debugCg)outln("1 d2 " << d2 << ", z3 " << z3);
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
				if(debugCg)outln("a " << a << ", b " << b << ", z2 " << z2);
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
				if(debugCg)outln("z2b " << z2);
				if(debugCg)outln("f3 " << f3 << " f2 " << f2 << " d3 " << d3 );
				if(debugCg)outln(" d2 " << d2 << " z3 " << z3 << " z2 " << z2);
				f3 = f2; d3 = d2; z3 = -z2; // set point 3 equal to point 2
				z1 = z1 + z2;
				if(debugCg)outln("2 z1 = z1 + z2 " << z1);
				x = x + z2 * s; // update current estimates
				f(df2,f2,x);
				m = m - 1;
				if (length < 0)
					i++; // count epochs?!
				d2 = (df2.transpose() * s).toScalar();
			}
			if(debugCg)outln("repeat while (innerLoop)");
		}
		if(debugCg)outln("done innerLoop");
		if (success) { // if line search succeeded
			f1 = f2;
			if (zerom == fX) {
				fX = Matrix<T>::ones(1, 1) * f1;
			} else {
		        fX = (fX.transpose() |= (Matrix<T>::ones(1, 1) * f1)).transpose();
			}

			sprintf(buff, "%4d | Cost: %4.6e\r", i, f1);
			cout << buff << endl;

			//  s = s * (df2.autoDot() - (df1.tN() * df2).toScalar()) / (df1.autoDot) - df2 // Polack-Ribiere direction
			s = s * (df2.autoDot() - (df1.transpose() * df2).toScalar()) / df1.autoDot() - df2;
			// Polack-Ribiere direction
			Matrix<T> tmp = df1;
			df1 = df2;
			df2 = tmp;
			if(debugCg)outln("df1.sum " << df1.sum() << ", df2.sum " << df2.sum() << ", tmp.sum() " << tmp.sum());

			// swap derivatives
			d2 = (df1.transpose() * s).toScalar();
			if(debugCg)outln("P-R d2 " << d2);
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
			} else {
				Matrix<T> tmp = df1;
				df1 = df2;
				df2 = tmp; // swap derivatives
				s = df1 * -1.; // try steepest
				d1 = (s.transpose() * s).toScalar() * -1;
				z1 = 1. / (1. - d1);
				ls_failed = true; // this line search failed
			}
		}
	}
	outln("leaving fmincg");
	return std::pair<Matrix<T>, std::pair<Matrix<T>, int> >(x,
			std::pair<Matrix<T>, int>(fX, i));
}
template class ConjugateGradient<float>;
template class ConjugateGradient<double>;

template std::pair<Matrix<float>, std::pair<Matrix<float>, int> > ConjugateGradient<float>::fmincg<nnCostFtor<float> >(nnCostFtor<float> &, Matrix<float>&, int, int);
template std::pair<Matrix<double>, std::pair<Matrix<double>, int> > ConjugateGradient<double>::fmincg<nnCostFtor<double> >(nnCostFtor<double> &, Matrix<double>&, int, int);

template <> double ConjugateGradient<double>::MinPositiveValue =  2.225073858507201e-308;
template <> float ConjugateGradient<float>::MinPositiveValue = 1.4E-45;

