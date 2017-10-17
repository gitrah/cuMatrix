/*
 * Drawable.h
 *
 *  Created on: Aug 28, 2013
 *      Author: reid
 */

#pragma once

#include <GL/gl.h>
#include <string>
#include <sstream>
#include "../debug.h"
#include "Vision.h"
#include "LookAt.h"
#include "Path.h"

using std::string;
using std::stringstream;

template<typename T> struct Bbox {
	T x0, x1;
	T y0, y1;
	T z0, z1;
	Bbox() :
			x0(0), x1(0), y0(0), y1(0), z0(0), z1(0) {
	}
	explicit Bbox(const T* a) :
			x0(a[0]), x1(a[1]), y0(a[2]), y1(a[3]), z0(a[4]), z1(a[5]) {
	}
	explicit Bbox(T x0, T x1, T y0, T y1, T z0, T z1) :
			x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1) {
	}
	Bbox(const Bbox<T>& o) :
			x0(o.x0), x1(o.x1), y0(o.y0), y1(o.y1), z0(o.z0), z1(o.z1) {
	}

	inline void scale(T factor) {
		x0 *= factor;
		x1 *= factor;
		y0 *= factor;
		y1 *= factor;
		z0 *= factor;
		z1 *= factor;
	}

	string toString() const {
		stringstream ss;
		ss << "(" << x0 << ", " << x1 << ") " << "(" << y0 << ", " << y1 << ") "
				<< "(" << z0 << ", " << z1 << ")\n";
		return ss.str();
	}
};
template<typename T> class Path;
template<typename T> class Drawable {
protected:
	Path<T> path;
	bool withPath;
public:
	Drawable() : withPath(false){}
	virtual void draw() const = 0;
	virtual void set(const T* pos, const T* dir)=0;
	virtual bool scale(Bbox<T>* bbx, T growBy)=0;
	virtual ~Drawable() { outln("~Drawable() die " << this << " die");}

	bool isWithPath() const {
		return withPath;
	}

	void setWithPath(bool withPath) {
		this->withPath = withPath;
	}

	static int lastW, lastH;

	static Bbox<T>* bbx;
	static Vision* vision;
	static LookAt* lookAt;
	static T bbxGrowFactor;

	static Drawable<T> ** drawList;
	static int drawCount;
	static bool enablePath;

	static void drawCb();
	static void drawAxes();
	static void stepCb();
	static void reshapeCb(int w, int h);
	static void addToDrawList( Drawable<T>* p);
	static bool scaleAll();
	static void freeDrawList();
	static void enablePaths();
};

template <typename T> int Drawable<T>::lastW = -1;
template <typename T> int Drawable<T>::lastH = -1;
template <typename T> int Drawable<T>::drawCount = 0;
template int Drawable<float>::drawCount;
template int Drawable<double>::drawCount;
template <typename T> Bbox<T>* Drawable<T>::bbx = NULL;
template <typename T> Vision* Drawable<T>::vision = NULL;
template <typename T> LookAt* Drawable<T>::lookAt = NULL;
template <typename T> Drawable<T> ** Drawable<T>::drawList = NULL;
template Drawable<float> ** Drawable<float>::drawList;
template Drawable<double> ** Drawable<double>::drawList;
template <typename T> bool Drawable<T>::enablePath = false;
template bool Drawable<float>::enablePath;
template bool  Drawable<double>::enablePath;
template <typename T> T Drawable<T>::bbxGrowFactor = 1.2;
