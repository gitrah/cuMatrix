/*
 * Drawable.cc
 *
 *  Created on: Sep 5, 2013
 *      Author: reid
 */

#include "Drawable.h"
#include <GL/glut.h>
#include <GL/glu.h>


template<typename T> void Drawable<T>::drawAxes() {
	glBegin(GL_LINES);
	glVertex3f (0.0, 0.0, -10.0);
	glVertex3f (0.0, 0.0, 100.0);
	glVertex3f (-10.0, 0.0, 0.0);
	glVertex3f (100.0, 0.0, 0.0);
	glVertex3f (0.0, -10.0, 0.0);
	glVertex3f (0.0, 100.0, 0.0);
	glEnd();
}

template void Drawable<float>::drawCb();
template void Drawable<double>::drawCb();
template<typename T> void Drawable<T>::drawCb() {
	outln("drawCb here drawCount " << drawCount << ". drawList " <<drawList);
	outln("drawCb drawList[0] " << drawList[0]);

	glLoadIdentity ();             /* clear the matrix */
		   /* viewing transformation  */
	if(lookAt) {
		outln("lookat " << lookAt->eyeX << ","<< lookAt->eyeY << ","<< lookAt->eyeZ << "  center " << lookAt->centerX << ", "<< lookAt->centerY << ", "<< lookAt->centerZ );
		lookAt->set();
	}
	glScalef (1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
	glColor3f (1.0, 0.0, 0.0);

	for (int i = 0; i < drawCount; i++) {
		Drawable<T> * drawable = drawList[i];
		outln("drawable " << drawable);
		drawable->draw();
	}

	drawAxes();

	glutSwapBuffers();
}

template void Drawable<float>::reshapeCb(int,int);
template void Drawable<double>::reshapeCb(int,int);
template<typename T> void Drawable<T>::reshapeCb(int w, int h) {
	outln("reshapeCb here drawCount " << drawCount << ". drawList " <<drawList);
	outln("reshapeCb drawList[0] " << drawList[0]);

	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
	lastW = w;
	lastH = h;
	outln("lastW " << w << ", lastH " << h);
	if(vision) {
		vision->set();
	} else {
		glMatrixMode (GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(80.0, 1.0, 1.5, 70.0);
	}
	if (bbx != null) {
	//	  glOrtho(bbx->x0, bbx->x1, bbx->y0, bbx->y1, bbx->z0, bbx->z1);
	} else {
		//dthrow(new WTF())
//		glOrtho(-100.0, 100.0, -100.0, 100.0, -100.0, 100.0);
	}
	glMatrixMode (GL_MODELVIEW);
}

template void Drawable<float>::addToDrawList(Drawable<float>*);
template void Drawable<double>::addToDrawList(Drawable<double>*);
template<typename T> void Drawable<T>::addToDrawList( Drawable<T>* p) {
	outln("adding p " << p);
	outln("ptr size " << sizeof(Drawable<T> *));
	Drawable<T> ** newPtr = new Drawable<T> *[drawCount +1]; //(Drawable<T>  **)malloc((drawCount + 1) * sizeof(Drawable<T> *));
	outln("creating newPtr " << newPtr);
	if (drawList != NULL) {
		outln("copying old contents from " << drawList);
		for(int i = 0; i < drawCount; i++) {
			newPtr[i] = drawList[i];
		}
		delete [] drawList;
	}
	drawList = newPtr;
	drawList[drawCount++] = p;
	outln("drawList " << drawList);
	outln("drawList[0] " << drawList[0]);
}

template<typename T> bool Drawable<T>::scaleAll() {
	bool scaled = false;
	for (int i = 0; i < drawCount; i++) {
		scaled |= drawList[i]->scale(bbx, bbxGrowFactor);
	}
	return scaled;
}

template void Drawable<float>::freeDrawList();
template void Drawable<double>::freeDrawList();
template<typename T> void Drawable<T>::freeDrawList() {
	delete [] drawList;
}

template void Drawable<float>::enablePaths();
template void Drawable<double>::enablePaths();
template<typename T> void Drawable<T>::enablePaths() {
	for(int i = 0; i < drawCount; i++) {
		drawList[i]->setWithPath(enablePath);
	}
}

