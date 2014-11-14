/*
 * Arrow.h
 *
 *  Created on: Aug 28, 2013
 *      Author: reid
 */

#ifndef ARROW_H_
#define ARROW_H_

#include "Drawable.h"

template<typename T> class Arrow : public Drawable<T> {
	T beg[3];
	T end[3];
public:
	static const int vecSize = 3 * sizeof(T);
	Arrow();
	Arrow(const T* pos, const T* dir);
	Arrow(T x, T y, T z);
	Arrow(T x, T y, T z, T nx, T ny, T nz);

	void set(const T* pos, const T* dir);
	void draw()const;
	void step();
	bool scale(Bbox<T>* bbox, T growBy);

};

#endif /* ARROW_H_ */
