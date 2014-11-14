/*
 * Perspective.cpp
 *
 *  Created on: Sep 8, 2013
 *      Author: reid
 */

#include "Perspective.h"
#include <GL/gl.h>
#include <GL/glu.h>

Perspective::Perspective()  : fovy(80),aspect(1),zNear(1.5),zFar(70) {};

Perspective::Perspective(double fovy, double aspect, double zNear, double zFar) : fovy(fovy),aspect(aspect),zNear(zNear),zFar(zFar) {};

void Perspective::set() {
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy,aspect,zNear,zFar);
}
