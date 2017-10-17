/*
 * Path.h
 *
 *  Created on: Sep 13, 2013
 *      Author: reid
 */

#pragma once
#include <vector>

using std::vector;

template<typename T> struct T3 {
	T data[3];
};


template<typename T> class Path {
	static vector<T3<T> > data;
public:
	Path() {};
	virtual ~Path() {};
	void draw()const;
	void add(T* pos);
};
template vector<T3<float> > Path<float>::data;
template vector<T3<double> > Path<double>::data;
template <typename T> vector<T3<T> > Path<T>::data;

