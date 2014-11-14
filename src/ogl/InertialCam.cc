/*
 * InertialCam.cc
 *
 *  Created on: Sep 16, 2013
 *      Author: reid
 */

#include "InertialCam.h"

InertialCam::InertialCam() : nx(0), ny(0), nz(0), mx(0), my(0), mz(0), x(0), y(0), lastX(0), lastY(0){
	// TODO Auto-generated constructor stub

}

InertialCam::~InertialCam() {
	// TODO Auto-generated destructor stub
}

int InertialCam::getLastX() const {
	return lastX;
}

void InertialCam::setLastX(int lastX) {
	this->lastX = lastX;
}

int InertialCam::getLastY() const {
	return lastY;
}

void InertialCam::setLastY(int lastY) {
	this->lastY = lastY;
}

double InertialCam::getMx() const {
	return mx;
}

void InertialCam::setMx(double mx) {
	this->mx = mx;
}

double InertialCam::getMy() const {
	return my;
}

void InertialCam::setMy(double my) {
	this->my = my;
}

double InertialCam::getMz() const {
	return mz;
}

void InertialCam::setMz(double mz) {
	this->mz = mz;
}

double InertialCam::getNx() const {
	return nx;
}

void InertialCam::setNx(double nx) {
	this->nx = nx;
}

double InertialCam::getNy() const {
	return ny;
}

void InertialCam::setNy(double ny) {
	this->ny = ny;
}

double InertialCam::getNz() const {
	return nz;
}

void InertialCam::setNz(double nz) {
	this->nz = nz;
}

int InertialCam::getX() const {
	return x;
}

void InertialCam::setX(int x) {
	this->x = x;
}

int InertialCam::getY() const {
	return y;
}

void InertialCam::setY(int y) {
	this->y = y;
}

void InertialCam::step() {

}
