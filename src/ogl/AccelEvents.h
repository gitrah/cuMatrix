/*
 * AccelEvents.h
 *
 */

#pragma once

#include "../util.h"
#include "../MatrixExceptions.h"
#include "Vutil.h"

template <typename T> struct Consts {
	static constexpr T two = (T)2;
};

template <typename T> class AccelEvents {
	int samples;
	int currSample;
	int rank;
	long* times;
	T* accel;

public:
	AccelEvents() : samples(0), currSample(0), rank(3), times(null), accel(null) {
		if(This != null) {
			dthrow ( ThisAlreadySet());
		}
		This = this;
	};

	AccelEvents(int samples); // : samples(samples), rank(3), times(null), accel(null)  {};
	~AccelEvents();

	int duringQ(long time) const;
	void linterp(T * acc, long time, int idx) const;
	void event(T* pos, T* gravity, T* vel, long time, long dtime)const;
	string toString() const;

	void syncToNow(long plusMillis);
	long first()const{ return times ? times[0] : -1; }
	long last() const{ return times ? times[samples-1] : -1;}
	long averageDtime();

	//void play(Point<T>& pt);
	bool intersectQ(const AccelEvents<T>* o)const;

	void play(T* pos, T* graivty, T* vel );

//statische-tict
	static AccelEvents<T>* fromMatrix(const CuMatrix<T>& m );

	static inline void update(T* pos, T* gravity, T* vel, const T* acc, T durationMicros);
	static inline void average(T* out, const T* in1, const T* in2, int rank = 3);

	static bool DelayMillisForMicros;
	static AccelEvents NoEvents;

	static AccelEvents* This;

	static const T noEvent[];
	static const int vecSize  = 3 * sizeof(T);
	static int sequenceThresholdMicros;  // pause of greater than this amount between samples => new sequence

};

template<typename T> const T AccelEvents<T>::noEvent[] = {0,0,0,0};
template <typename T> AccelEvents<T>* AccelEvents<T>::This = null;
template <typename T> int AccelEvents<T>::sequenceThresholdMicros = 300;
template <typename T> bool AccelEvents<T>::DelayMillisForMicros = false;
