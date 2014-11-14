/*
 * Geom.cc
 *
 *  Created on: Feb 1, 2014
 *      Author: reid
 */




#include "Geom.h"

template <typename T> __host__ __device__ bool withinCircle(T cOrgX, T cOrgY, T cRad, T pX, T pY, T eps) {
	T distX = cOrgX - pX;
	T distY = cOrgY - pY;
	distX *= distX;
	distY *= distY;
	return eps * eps > cRad * cRad - distX - distY;
}
