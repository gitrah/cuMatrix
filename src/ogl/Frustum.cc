/*
 * Frustum.cc
 *
 *  Created on: Sep 8, 2013
 *      Author: reid
 */

#include "Frustum.h"
#include <GL/gl.h>

Frustum::~Frustum() {}

void Frustum::set() {
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();
    glFrustum (left, right, bottom, top, near_val, far_val);
}
