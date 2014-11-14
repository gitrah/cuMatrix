/*
 * Vutil.h
 *
 *  Created on: Sep 16, 2013
 *      Author: reid
 */

#ifndef VUTIL_H_
#define VUTIL_H_
#include <string.h>

template <typename T> struct Vutil {
	static const int vecSize  = 3 * sizeof(T);
	inline static void copy3(T* dst, const T* src) {
		memcpy(dst, src, vecSize);
	}
	inline static void set3(T* dst, T val) {
		memset(dst, val, vecSize);
	}
	inline static void scale(T* dst, T factor) { dst[0]*=factor;dst[1]*=factor;dst[2]*=factor;}

	static void sub3(T* out, const T* minuend, const T* subtrahend) {
		out[0] = minuend[0]-subtrahend[0];
		out[1] = minuend[1]-subtrahend[1];
		out[2] = minuend[2]-subtrahend[2];
	}
	static void add3(T* out, const T* v1, const T* v2) {
		out[0] = v1[0]+v2[0];
		out[1] = v1[1]+v2[1];
		out[2] = v1[2]+v2[2];
	}
	static void inc3(T* out, T v) {
		out[0] += v;
		out[1] += v;
		out[2] += v;
	}

	inline static T lenSqr(const T* src) { return src[0]*src[0] + src[1]*src[1] + src[2]*src[2]; }
	inline static T len(const T* src) { return sqrt(lenSqr(src));}

	inline static void norm(T* out, const T* in) { scale(out, in, 1.0/len(in));}

	inline static T dist3Sqr(const T* dst, const T* src) {
		T d1 = dst[0]-src[0];
		T d2 = dst[1]-src[1];
		T d3 = dst[2]-src[2];
		return d1 * d1 + d2*d2 + d3 * d3;
	}
	inline static T dist3(T* dst, const T* src) {return sqrt(dist3Sqr(dst,src));}
};




#endif /* VUTIL_H_ */
