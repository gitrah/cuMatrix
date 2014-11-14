/*
 * Point.h
 *
 *  Created on: Aug 10, 2013
 *      Author: reid
 */

#ifndef POINT_H_
#define POINT_H_
#include "../util.h"
#include "AccelEvents.h"
#include "Drawable.h"
#include "Steppable.h"

template<typename T> class Point : public Drawable<T>, public Steppable<T>{
	Drawable<T>* twin;
public:
	T pos[3];
private:
	T vel[3];
	T gravity[3];
	T mass;
	long ltime;
	long dtime;
	AccelEvents<T>** events;
	int eventCount;
public:

	Point();
	Point(const T* pos);
	Point(const T* pos, const T* vel);

	void trans(T del[3]);
	void makeCurrent();
	void step();
	void play();
	bool scale(Bbox<T>* bbx, T growBy);

	void lowPass(T* linAcc, const T* acc, T alpha);
	void hiPass(T* linAcc, const T* acc, T alpha);

	long getDtime() const {
		return dtime;
	}

	void setDtime(long dtime) {
		this->dtime = dtime;
	}

	int getEventCount() const {
		return eventCount;
	}

	int addEvents(const AccelEvents<T>* ie);

	T getMass() const {
		return mass;
	}

	void setMass(T mass) {
		this->mass = mass;
	}

	const T* getPos() const {
		return pos;
	}

	void setPos(const T* pos) {
		this->pos[0] = pos[0];
		this->pos[1] = pos[1];
		this->pos[2] = pos[2];
	}

	const T* getGravity() const {
		return gravity;
	}

	long getTime() const {
		return ltime;
	}

	void setTime(long ltime) {
		this->ltime = ltime;
	}

	const T* getVel() const {
		return vel;
	}

	void set(const T* pos, const T* dir);

	void draw()const;

	// util fns
	static void randomize(T* trg,  T spread);
	static inline bool insideBboxQ(const T* pos, const T* bboxWhd);
	static inline bool outsideBboxQ(const T* pos, const T* bboxWhd);

	static inline bool scale(T& t0, T& t1, T test, T factor);
	// normalized measure of closeness of pos to a surface of the bbox
	static T extremityN(const T* pos, const T* bboxWhd);

	Drawable<T>* getTwin() {
		return twin;
	}

	void setTwin(Drawable<T>* twin) {
		this->twin = twin;
	}

};


#endif /* POINT_H_ */
