/*
 * Path.cc
 *
 *  Created on: Sep 13, 2013
 *      Author: reid
 */

#include "Path.h"
#include <GL/gl.h>
#include "Vutil.h"
#include "../debug.h"

template void Path<float>::draw()const;
template void Path<double>::draw()const;
template<typename T> void Path<T>::draw()const {
	outln("drawing a path");
	glBegin(GL_LINES);

	T sz = data.size();
	T clr[] = {0,0,0};

	typedef typename vector<T3<T> >::iterator iterator;
	iterator it = data.begin();
	while(it != data.end()) {
		Vutil<T>::inc3(clr, 1.0/sz);
		glColor3f (clr[0],clr[1],clr[2]);
		T3<T> pos = (*it);
		glVertex3f(pos.data[0],pos.data[1],pos.data[2]);
		it++;
	}
	glEnd();
}
template void Path<float>::add(float*);
template void Path<double>::add(double *);
template<typename T> void Path<T>::add(T* pos){
	T3<T> t3;
	Vutil<T>::copy3(t3.data, pos);
	data.push_back(t3);
}
