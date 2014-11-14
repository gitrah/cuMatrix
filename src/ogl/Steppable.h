/*
 * Steppable.h
 *
 *  Created on: Sep 16, 2013
 *      Author: reid
 */

#ifndef STEPPABLE_H_
#define STEPPABLE_H_
#include "../util.h"

template <typename T> class Steppable {
public:
	virtual void step() = 0;
	Steppable(){};
	virtual ~Steppable(){};
	static Steppable<T> ** stepList;
	static int stepCount;

	static void addToStepList( Steppable<T>* p);
	static void freeStepList();
	// callbacks
	static void stepCb();
};
template <typename T> int Steppable<T>::stepCount = 0;
template int Steppable<float>::stepCount;
template int Steppable<double>::stepCount;
template <typename T> Steppable<T> ** Steppable<T>::stepList = NULL;
template Steppable<float> ** Steppable<float>::stepList;
template Steppable<double> ** Steppable<double>::stepList;

#endif /* STEPPABLE_H_ */
