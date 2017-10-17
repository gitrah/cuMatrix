/*
 * Frustum.h
 *
 *  Created on: Sep 8, 2013
 *      Author: reid
 */


#include "Vision.h"

class Frustum  : public Vision{
	double left, right, bottom, top, near_val, far_val;
public:
	Frustum() : left(-20.0),right( 20.0),bottom(-20.0),top(20.0),near_val(1.5),far_val(50.0) {} ;
	Frustum(double left, double right, double bottom, double top, double near_val, double far_val) : left(left),right(right),bottom(bottom),top(top),near_val(near_val),far_val(far_val) {};
	virtual ~Frustum();
	void set();
};
