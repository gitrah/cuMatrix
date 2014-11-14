/*
 * Perspective.h
 *
 *  Created on: Sep 8, 2013
 *      Author: reid
 */

#ifndef PERSPECTIVE_H_
#define PERSPECTIVE_H_
#include "Vision.h"

class Perspective : public Vision {
	double fovy, aspect, zNear, zFar;
public:
	Perspective();
	Perspective(double fovy, double aspect, double zNear, double zFar);
	void set();

	double getAspect() const {
		return aspect;
	}

	void setAspect(double aspect) {
		this->aspect = aspect;
	}

	double getFovy() const {
		return fovy;
	}

	void setFovy(double fovy) {
		this->fovy = fovy;
	}

	double getFar() const {
		return zFar;
	}

	void setFar(double far) {
		zFar = far;
	}

	double getNear() const {
		return zNear;
	}

	void setNear(double near) {
		zNear = near;
	}
};


#endif /* PERSPECTIVE_H_ */
