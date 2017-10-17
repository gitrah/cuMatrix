/*
 * Text.h
 *
 *  Created on: Sep 10, 2013
 *      Author: reid
 */

#pragma once

#include <string>
#include "Vutil.h"
#include "Drawable.h"
#include "Steppable.h"

using std::string;

template<typename T> class Text : public Drawable<T> , public Steppable<T> {
	char* text;
	T scaleU;
	T pos[3];
	T normal[3];
public:
	Text() : text(null),scaleU(0.01){ Drawable<T>::addToDrawList(this);};
	Text(T scaleU) : text(null),scaleU(scaleU) { zero(); Drawable<T>::addToDrawList(this);}
	Text(string s) : text((char*)s.c_str()),scaleU(0.01) { zero();Drawable<T>::addToDrawList(this);}
	Text(string s, int scaleU, T * pos, T *n) : text(s.c_str()), scaleU(scaleU) {
		Vutil<T>::copy3(this->pos, pos);
		Vutil<T>::copy3(this->normal, n);
		Drawable<T>::addToDrawList(this);
	}
	Text(string s, T * pos, T *n) : text(s.c_str()), scaleU(20) {
		Vutil<T>::copy3(this->pos, pos);
		Vutil<T>::copy3(this->normal, n);
		Drawable<T>::addToDrawList(this);
	}
	virtual ~Text() {};
	inline void zero() {
		Vutil<T>::set3(pos, 0);
		Vutil<T>::set3(normal, 0);
	}

	const T* getNormal() const {
		return normal;
	}
	void setNormal(const T* nrm) {
		Vutil<T>::copy3(normal,nrm);
	}
	const T* getPos() const {
		return pos;
	}
	void setPos(const T* p) {
		Vutil<T>::copy3(pos,p);
	}

	int getScaleU() const {
		return scaleU;
	}

	void setScaleU(int scaleU) {
		this->scaleU = scaleU;
	}

	void set(const T* pos, const T* dir) {
		Vutil<T>::copy3(this->pos,pos);
		Vutil<T>::copy3(normal,dir);
	}
	void draw() const;
	void step() {};
	bool scale(Bbox<T>* bbx, T growBy) { return false; }

	char* getText() const {
		return text;
	}
	const char ** getTextAdr() {
		return (const char**) &text;
	}

	void setText(char* text) {
		this->text = text;
	}
};
