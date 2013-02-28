/*
 * functors.h
 *
 *  Created on: Oct 4, 2012
 *      Author: reid
 */
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "MatrixExceptions.h"
//#include "util.h"

#ifndef FUNCTORS_H_
#define FUNCTORS_H_

extern __host__ __device__ __device_builtin__ float                  powf(float x, float y) __THROW;

// not especially useful; another unsuccessful attempt at
// polymorphinc
struct deviceable {
	__host__ int toDevice(void** devInstance) {
		int error = cudaMalloc(devInstance, sizeof(this));
		error += cudaErrorNotSupported * cudaMemcpy(*devInstance, this, sizeof(this), cudaMemcpyHostToDevice);
		return error;
	}
	__host__ cudaError_t free(void* devInstance) {
		return cudaFree(devInstance);
	}
};

/* unary operand functors */
template<typename T> struct unaryOpF : public deviceable {
	virtual __host__ __device__ T operator()(const T& xi) const {
		return 0;
	}
};
template struct unaryOpF<float>;
template struct unaryOpF<double>;

template<typename T>
struct sigmoidUnaryOp : public unaryOpF<T>  {
	using unaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi) const {
		return (static_cast<T>( 1.)/ (static_cast<T>( 1.) + static_cast<T>( exp(-xi))));
	}
};
template<typename T>
struct sigmoidGradientUnaryOp : public unaryOpF<T>  {
	using unaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi) const {
		T t = (static_cast<T>( 1.) / (static_cast<T>( 1) + static_cast<T>(  exp(-xi))));
		return t * (1. - t);
	}
};
template<typename T>
struct negateUnaryOp : public unaryOpF<T>  {
	using unaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi) const {
		return (static_cast<T>(  -xi));
	}
};
template<typename T> struct logUnaryOp : public unaryOpF<T>   {
	using unaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi) const {
		return (static_cast<T>( log(xi)));
	}
};
template<typename T> struct oneOverUnaryOp : public unaryOpF<T>  {
	using unaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi) const {
		return (static_cast<T>( 1.)/static_cast<T>( xi));
	}
};
template<typename T> struct expUnaryOp : public unaryOpF<T>  {
	//using unaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi) const {
		return (static_cast<T>(  exp(xi)));
	}
};

template<typename T> struct powUnaryOp : public unaryOpF<T>   {
	using unaryOpF<T>::operator();
	T power;
	__host__ __device__	T operator()(const T& xi) const {
		return (static_cast<T>( pow(xi, power)));
	}
};
template <> struct powUnaryOp<float> : public unaryOpF<float> {
	using unaryOpF::operator();
	float power;
	__host__ __device__	float operator()(const float& xi) const {
		return powf(xi, power);
	}
};

template<typename T> struct sqrtUnaryOp : public unaryOpF<T>  {
	using unaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi) const {
		return static_cast<T>(  sqrt(xi));
	}
};

template<typename T> struct sqrUnaryOp : public unaryOpF<T>  {
	using unaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi) const {
		return ( xi * xi);
	}
};

template<typename T> struct divSqrtUnaryOp : public unaryOpF<T>  {
	using unaryOpF<T>::operator();
	float divisor;
	__host__ __device__
	T operator()(const T& xi) const {
		return static_cast<T>( sqrt(xi / divisor));
	}
};

template<typename T> struct scaleUnaryOp : public unaryOpF<T>  {
	using unaryOpF<T>::operator();
	T multiplicand;
	__host__ __device__	T operator()(const T& xi) const {
		return (  xi * multiplicand );
	}
};

template<typename T> struct translationUnaryOp : public unaryOpF<T>  {
	using unaryOpF<T>::operator();
	T addend;
	virtual __host__ __device__	T operator()(const T& xi) const {
		return (  xi + addend );
	}
};

template<typename T> struct subFromUnaryOp : public unaryOpF<T>  {
	using unaryOpF<T>::operator();
	T source;
	__host__ __device__	T operator()(const T& subtrahend) const {
		return (  source - subtrahend );
	}
};

template<typename T> struct oneOrZeroBoolUnaryOp : public unaryOpF<T> {
	using unaryOpF<T>::operator();
	__host__ __device__	T operator()(const T& t) const {
		return ( t == 0 || t == 1 );
	}
};

template<typename T> struct ltUnaryOp : public unaryOpF<T> {
	using unaryOpF<T>::operator();
	T comp;
	__host__ __device__	T operator()(const T& t) const {
		return ( t < comp );
	}
};

template<typename T> struct lteUnaryOp : public unaryOpF<T> {
	using unaryOpF<T>::operator();
	T comp;
	__host__ __device__	T operator()(const T& t) const {
		return ( t <= comp );
	}
};

template<typename T> struct gtUnaryOp : public unaryOpF<T> {
	using unaryOpF<T>::operator();
	T comp;
	__host__ __device__	T operator()(const T& t) const {
		return ( t > comp );
	}
};

template<typename T> struct gteUnaryOp : public unaryOpF<T> {
	using unaryOpF<T>::operator();
	T comp;
	__host__ __device__	T operator()(const T& t) const {
		return ( t >= comp );
	}
};

template<typename T> struct eqUnaryOp : public unaryOpF<T> {
	using unaryOpF<T>::operator();
	T comp;
	__host__ __device__	T operator()(const T& t) const {
		return ( t == comp );
	}
};

template<typename T> struct almostEqUnaryOp : public unaryOpF<T> {
	using unaryOpF<T>::operator();
	T comp;
	T eps;
	__host__ __device__	T operator()(const T& t) const {
		return ( abs(t - comp) < eps);
	}
};

template<typename T> struct equalsBoolUnaryOp : public unaryOpF<T> {
	using unaryOpF<T>::operator();
	T target;
	__host__ __device__	T operator()(const T& t) const {
		return ( t == target );
	}
};

template<typename T> struct almostEqualsBoolUnaryOp : public unaryOpF<T> {
	using unaryOpF<T>::operator();
	T target;
	T epsilon;
	__host__ __device__	T operator()(const T& t) const {
		return (  abs(t - target) < epsilon );
	}
};

template<typename T> struct binaryOpF : public deviceable {
	virtual __host__ __device__ T operator()(const T& xi, const T& xi2)  const = 0;
};

/* binary operand functors for reductions */
template<typename T>
struct multBinaryOp : public binaryOpF<T> {
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return (xi * xi2);
	}
};
template<typename T>
struct plusBinaryOp : public binaryOpF<T> {
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return (xi + xi2);
	}
};
template<typename T>
struct sqrPlusBinaryOp : public binaryOpF<T>  {
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return (xi*xi + xi2*xi2);
	}
};
template<typename T>
struct diffSquaredBinaryOp : public binaryOpF<T> {
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return ( (xi2 - xi) * (xi2 - xi));
	}
};
template<typename T>
struct equalsBinaryOp : public binaryOpF<T> {
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return ( xi2 == xi );
	}
};

template<typename T>
struct almostEqualsBinaryOp : public binaryOpF<T> {
	using binaryOpF<T>::operator();
	T epsilon;
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return ( abs(xi2 - xi) < epsilon );
	}
};
template<typename T>
struct minusBinaryOp : public binaryOpF<T> {
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return (xi - xi2);
	}
};
template<typename T>
struct minBinaryOp : public binaryOpF<T> {
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return MIN(xi, xi2);
	}
};
template<typename T>
struct maxBinaryOp : public binaryOpF<T> {
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return MAX(xi, xi2);
	}
};

template<typename T>
struct quotientBinaryOp : public binaryOpF<T> {
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return (xi / xi2);
	}
};

template<typename T> struct andBinaryOp : public  binaryOpF<T> {
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const  {
		return (xi && xi2);
	}
};

template<typename T> struct orBinaryOp : public binaryOpF<T> {
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return (xi || xi2);
	}
};


template<typename T>
struct gtBinaryOp :  public binaryOpF<T> {
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return (xi > xi2);
	}
};

template<typename T>
struct ltBinaryOp  :  public binaryOpF<T>{
	using binaryOpF<T>::operator();
	__host__ __device__
	T operator()(const T& xi,const T& xi2) const {
		return (xi < xi2);
	}
};


/////////
// index bool functors
/////////
struct boolUnaryOpIndexF : public unaryOpF<ulong> {
	using unaryOpF::operator();
	virtual __host__ __device__ ulong operator()(const ulong& idx) const {
		return false;
	}
};

struct isColumnUnaryOp : public boolUnaryOpIndexF {
	using unaryOpF::operator();
	uint pitch;
	uint column;
	__host__ __device__
	ulong operator()(const ulong& idx) const {
		return (idx % pitch == column);
	}
};

struct isRowUnaryOp  : public boolUnaryOpIndexF{
	using unaryOpF::operator();
	uint pitch;
	uint row;
	__host__ __device__
	ulong operator()(const ulong& idx) const {
		return (idx / pitch == row);
	}
};


/////////////////////
// fillers
////////////////////
template<typename T> struct unaryOpIndexF {
	virtual __host__ __device__ T operator()(const ulong& idx) const = 0;
};

template<typename T> struct hostUnaryOpIndexF {
	virtual __host__ T operator()(const ulong& idx) const = 0;
};

template<typename T>
struct stepFiller : public unaryOpIndexF<T> {
	__host__ __device__ T operator()(const ulong& idx) const {
		return  1 / static_cast<T>(1 + idx);
	}
};

template<typename T>
struct sinFillerHost : public hostUnaryOpIndexF<T> {
	T amplitude;
	T period;
	T phase;
	__host__
	T operator()(const ulong& idx) const {
		return static_cast<T>(  amplitude * ::sin(  2. * Pi/ period * (idx + phase)));
		//return static_cast<T>( ::sin( idx + 1) / 10.);
	}
};

template<typename T>
struct sinFiller : public unaryOpIndexF<T> {
	T amplitude;
	T period;
	T phase;
	__host__ __device__
	T operator()(const ulong& idx) const {
		//return static_cast<T>(  amplitude * ::sin(  2. * Pi/ period * (idx + phase) / 10.));
		return static_cast<T>( sinf( idx + 1) / 10.);
	}
};


template<typename T>
struct cosFiller : public unaryOpIndexF<T> {
	T amplitude;
	T period;
	T phase;
	__host__ __device__
	T operator()(const ulong& idx) const {
		//return static_cast<T>( amplitude *  ::cos( 2. * Pi/ period * (idx + phase) / 10.));
		return static_cast<T>( cosf( idx + 1) / 10.);
	}
};

template<typename T>
struct cosFillerHost  : public hostUnaryOpIndexF<T> {
	T amplitude;
	T period;
	T phase;
	__host__
	T operator()(const ulong& idx) const {
		//return static_cast<T>( amplitude *  ::cos( 2. * Pi/ period * (idx + phase) / 10.));
		return static_cast<T>( ::cos( idx + 1) / 10.);
	}
};



template<typename T>
struct sequenceFiller : public unaryOpIndexF<T> {
	T phase;
	__host__ __device__
	T operator()(const ulong& idx) const {
		return static_cast<T>( idx + phase);
	}
};

template<typename T>
struct increasingColumnsFiller : public unaryOpIndexF<T> {
	T start;
	ulong cols;
	__host__ __device__
	T operator()(const ulong& idx) const {
		return static_cast<T>( start + (idx % cols));
	}
};

template<typename T>
struct increasingRowsFiller : public unaryOpIndexF<T> {
	T start;
	ulong cols;
	__host__ __device__
	T operator()(const ulong& idx) const {
		return static_cast<T>( start + (idx / cols));
	}
};

template<typename T>
struct seqModFiller : public unaryOpIndexF<T> {
	T phase;
	int mod;
	__host__ __device__
	T operator()(const ulong& idx) const {
		return static_cast<T>(  (idx + static_cast<ulong>( phase)) % static_cast<ulong>( mod));
	}
};


template<typename T>
struct constFiller : public unaryOpIndexF<T> {
	T value;
	constFiller() : value(0) {}
	constFiller(constFiller & f) {
		value = f.value;
	}
	__host__ __device__
	T operator()(const ulong& idx) const {
		return ( value);
	}
};

template<typename T>
struct randFiller {
	T epsilon;
	__host__ __device__
	T operator()(const ulong& idx) const {
		//return static_cast<T>( (2 * (1. * rand()) / RAND_MAX - 1) * epsilon);
		return 0.667;
	}
};

template<typename T>
struct diagonalFiller : public unaryOpIndexF<T> {
	T value;
	uint dim;
	__host__ __device__
	T operator()(const ulong& idx) const {
		return ( idx / dim == idx % dim ? value : 0 );
	}
};


#endif /* FUNCTORS_H_ */
