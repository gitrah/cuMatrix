/*
 * LookAt.cc
 *
 *  Created on: Sep 9, 2013
 *      Author: reid
 */

#include "LookAt.h"
#include <GL/gl.h>
#include <GL/glu.h>

LookAt::LookAt() :  eyeX(25),eyeY(25),eyeZ(40),centerX(0),centerY(0),centerZ(0),upX(0),upY(0),upZ(1)
{
	// TODO Auto-generated constructor stub

}

LookAt::~LookAt() {
	// TODO Auto-generated destructor stub
}

void LookAt::set() {
	//glMatrixMode (GL_PROJECTION);
//	glLoadIdentity();
	gluLookAt (eyeX,eyeY,eyeZ,centerX, centerY,  centerZ,upX,upY,upZ);
}
