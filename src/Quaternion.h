/*
 * Quaternion.h
 *
 *  Created on: Aug 15, 2013
 *      Author: reid
 */

#pragma once

template<typename T> class Quaternion {
	T w,x,y,z;
public:
	Quaternion() :  w(0),x(0),y(0),z(0){}
	Quaternion(T w, T x, T y, T z) :  w(w),x(x),y(y),z(z){};

	Quaternion sum(const Quaternion& o) { return Quaternion(w + o.w, x + o.x, y + o.y, z + o.z); }
	Quaternion product(const Quaternion& o) {
		T c1,c2,c3;
		util<T>::cross(c1, c2, c3, x,y,z, o.x, o.y, o.z);
		return Quaternion(w * o.w - (x * o.x + y * o.y + z * o.z),
				c1 + w * o.x + o.w * x,
				c2 + w * o.y + o.w * y,
				c3 + w * o.z + o.w * z);

	}
	inline Quaternion conjugate() {return Quaternion(w, -x, -y, -z); }
	inline Quaternion inverse() {
		return conjugate() / norm();
	}
	inline T norm() { return w * w + x*x + y*y + z* z; }

};
