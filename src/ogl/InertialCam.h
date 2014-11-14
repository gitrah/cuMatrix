/*
 * InertialCam.h
 *
 *  Created on: Sep 16, 2013
 *      Author: reid
 */

#ifndef INERTIALCAM_H_
#define INERTIALCAM_H_

#include "LookAt.h"
#include "Steppable.h"

class InertialCam : public Vision, public Steppable<float> {
	double nx,ny,nz;
	double mx,my,mz;
	int x,y;
	int lastX,lastY;
public:
	InertialCam();
	virtual ~InertialCam();
	void step();
	int getLastX() const;
	void setLastX(int lastX);
	int getLastY() const;
	void setLastY(int lastY);
	double getMx() const;
	void setMx(double mx);
	double getMy() const;
	void setMy(double my);
	double getMz() const;
	void setMz(double mz);
	double getNx() const;
	void setNx(double nx);
	double getNy() const;
	void setNy(double ny);
	double getNz() const;
	void setNz(double nz);
	int getX() const;
	void setX(int x);
	int getY() const;
	void setY(int y);
};

#endif /* INERTIALCAM_H_ */
