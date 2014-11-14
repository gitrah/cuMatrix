/*
 * Arrow.cc
 *
 *  Created on: Aug 28, 2013
 *      Author: reid
 */
#include <GL/gl.h>
#include "Arrow.h"
#include "Vutil.h"
#include "../debug.h"

template Arrow<float>::Arrow();
template Arrow<double>::Arrow();
template<typename T> Arrow<T>::Arrow() {
		memset(beg, 0, vecSize);
		memset(end, 0, vecSize);
		Drawable<T>::addToDrawList(this);
}

template Arrow<float>::Arrow( const float* , const float*);
template Arrow<double>::Arrow(const double* , const double*);
template<typename T> Arrow<T>::Arrow(const T* pos, const T* dir) {
		memcpy(beg, pos, vecSize);
		Vutil<T>::add3(end,pos,dir);
		Drawable<T>::addToDrawList(this);
}

template Arrow<float>::Arrow(float, float, float);
template Arrow<double>::Arrow(double, double, double);
template<typename T> Arrow<T>::Arrow(T x, T y, T z) {
		beg[0] = x; beg[1] = y; beg[2] = z;
		memcpy(end, beg, vecSize);
		Drawable<T>::addToDrawList(this);
}

template Arrow<float>::Arrow(float, float, float,float, float, float);
template Arrow<double>::Arrow(double, double, double,double, double, double);
template<typename T> Arrow<T>::Arrow(T x, T y, T z,T nx, T ny, T nz) {
		beg[0] = x; beg[1] = y; beg[2] = z;
		end[0] =  x+ nx; end[1] = y+ny; end[2] = z+nz;
		Drawable<T>::addToDrawList(this);
}

template void Arrow<float>::set( const float* , const float*);
template void Arrow<double>::set( const double* , const double*);
template<typename T> void Arrow<T>::set( const T* pos, const T* dir)  {
		memcpy(beg, pos, vecSize);
		Vutil<T>::add3(end, pos, dir);
}

template<typename T> void Arrow<T>::step() {
}
template bool Arrow<float>::scale(Bbox<float>*, float);
template bool Arrow<double>::scale(Bbox<double>*, double);
template<typename T> bool Arrow<T>::scale(Bbox<T>* bbox, T growBy){return false;}

template<typename T> void Arrow<T>::draw() const {
	outln("beg " << niceVec(beg) << ", end " << niceVec(end));
	glColor3f (0.0, 1.0, 0.0);
	glBegin(GL_LINES);
		glVertex3f(beg[0],beg[1],beg[2]);
		glVertex3f(end[0],end[1],end[2]);
	glEnd();
}


