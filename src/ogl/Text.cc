/*
 * Text.cc
 *
 *  Created on: Sep 10, 2013
 *      Author: reid
 */

#include "Text.h"
#include <GL/gl.h>
#include <GL/glut.h>
template void Text<float>::draw() const;
template void Text<double>::draw() const;
template<typename T>void Text<T>::draw() const{
	int lengthOfQuote = strlen(text);

	glPushMatrix();
	glTranslatef(pos[0],pos[1],pos[2]);
	glColor3f(1, 1, 1);
	glNormal3f(normal[0],normal[1],normal[2]);
	glScalef(scaleU, scaleU, scaleU);
	for (int i = 0; i < lengthOfQuote; i++) {
		glutStrokeCharacter(GLUT_STROKE_ROMAN, text[i]);
	}
	glPopMatrix();
}
