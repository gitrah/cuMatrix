/*
 * Steppable.cc
 *
 *  Created on: Sep 16, 2013
 *      Author: reid
 */

#include "Steppable.h"
#include <GL/glut.h>

template void Steppable<float>::stepCb();
template void Steppable<double>::stepCb();
template<typename T> void Steppable<T>::stepCb() {
	outln("stepCb here stepCount " << stepCount << ". stepList " <<stepList);
	outln("stepCb stepList[0] " << stepList[0]);

	for (int i = 0; i < stepCount; i++) {
		Steppable<T> * steppable = stepList[i];
		steppable->step();
	}

	glutPostRedisplay(); // triggers drawCb

}

template void Steppable<float>::addToStepList(Steppable<float>*);
template void Steppable<double>::addToStepList(Steppable<double>*);
template<typename T> void Steppable<T>::addToStepList( Steppable<T>* p) {
	outln("adding p " << p);
	outln("ptr size " << sizeof(Steppable<T> *));
	Steppable<T> ** newPtr = new Steppable<T> *[stepCount +1]; //(Steppable<T>  **)malloc((stepCount + 1) * sizeof(Drawable<T> *));
	outln("creating newPtr " << newPtr);
	if (stepList != NULL) {
		outln("copying old contents from " << stepList);
		for(int i = 0; i < stepCount; i++) {
			newPtr[i] = stepList[i];
		}
		delete [] stepList;
	}
	stepList = newPtr;
	stepList[stepCount++] = p;
	outln("stepList " << stepList);
	outln("stepList[0] " << stepList[0]);
}

template void Steppable<float>::freeStepList();
template void Steppable<double>::freeStepList();
template<typename T> void Steppable<T>::freeStepList() {
	delete [] stepList;
}
