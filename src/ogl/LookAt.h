/*
 * LookAt.h
 *
 *  Created on: Sep 9, 2013
 *      Author: reid
 */

#ifndef LOOKAT_H_
#define LOOKAT_H_
#include "Vision.h"

struct LookAt : public Vision {
	//enum Values {eyeX,eyeY,eyeZ,centerX,centerY,centerZ,upX,upY,upZ};
	//double data[]
	double eyeX;
	double eyeY;
	double eyeZ;
	double centerX;
	double centerY;
	double centerZ;
	double upX;
	double upY;
	double upZ;
	LookAt();
	virtual ~LookAt();
	void set();
};

#endif /* LOOKAT_H_ */
