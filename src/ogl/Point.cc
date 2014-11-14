/*
 * Point.cc
 *
 */
#include "Point.h"
#include <GL/gl.h>
#include <GL/glut.h>
#include "../util.h"

extern template class Drawable<float>;
extern template class Drawable<double>;


template Point<float>::Point();
template Point<double>::Point();
template<typename T> Point<T>::Point() : twin(null),
		mass(1), ltime(0), dtime(0), events(null), eventCount(0){
	memset(pos, 0, sizeof(T) * 3);
	memset(vel, 0, sizeof(T) * 3);
	memset(gravity, 0, sizeof(T) * 3);
	outln("Point() this " << this << " pos " << niceVec(pos) << ", grav " << niceVec(gravity));
	Drawable<T>::addToDrawList(this);
	Steppable<T>::addToStepList(this);
}

template Point<float>::Point(const float*);
template Point<double>::Point(const double*);
template<typename T> Point<T>::Point(const T* pos) :
		twin(null),mass(1), ltime(0), dtime(0), events(null), eventCount(0) {
	this->pos[0] = pos[0];
	this->pos[1] = pos[1];
	this->pos[2] = pos[2];
	memset(vel, 0, sizeof(T) * 3);
	memset(gravity, 0, sizeof(T) * 3);
	outln("Point() this " << this << " pos " << niceVec(pos) << ", grav " << niceVec(gravity));
	Drawable<T>::addToDrawList(this);
	Steppable<T>::addToStepList(this);
}

template Point<float>::Point(const float*,const float*);
template Point<double>::Point(const double*,const double*);
template<typename T> Point<T>::Point(const T* pos, const T* vel) :
		twin(null), mass(1), ltime(0), dtime(0), events(null), eventCount(0) {
	this->pos[0] = pos[0];
	this->pos[1] = pos[1];
	this->pos[2] = pos[2];
	this->vel[0] = vel[0];
	this->vel[1] = vel[1];
	this->vel[2] = vel[2];
	memset(gravity, 0, sizeof(T) * 3);
	Drawable<T>::addToDrawList(this);
	Steppable<T>::addToStepList(this);
}

template void Point<float>::set( const float* , const float*);
template void Point<double>::set( const double* , const double*);
template<typename T> void Point<T>::set( const T* pos, const T* dir)  {
	//dthrow(notImplemented());
}


template int Point<float>::addEvents(const AccelEvents<float>* ie);
template int Point<double>::addEvents(const AccelEvents<double>* ie);
template<typename T> int Point<T>::addEvents(const AccelEvents<T>* ie) {
	if (events == null) {
		events = new AccelEvents<T>*[1];
	} else {
		for(int i =0; i < eventCount; i++ ){
			if(events[i]->intersectQ(ie)) {
				outln(events[i]->toString() << " overlaps " << ie->toString() << "; superpos not impld");
				dthrow(new SuperposNotImplemented());
			}
		}
		void* temp = malloc(eventCount+1 * sizeof(AccelEvents<T>*));
		memcpy(temp, events, eventCount * sizeof(AccelEvents<T>*));
		free(events);
		events = (AccelEvents<T>**)temp;
	}
	this->events[eventCount++] = ( AccelEvents<T>*)ie;
	return ie->duringQ(ltime);
}

template void Point<float>::step();
template void Point<double>::step();
template<typename T> void Point<T>::step() {
	outln("Point<T>::step en");

	Drawable<T>::path.add(pos);
	events[0]->play(pos,gravity,vel);
	Drawable<T>::path.add(pos);

	if(twin) {
		T bg[3];
		Vutil<T>::copy3(bg,gravity);
		Vutil<T>::scale(bg,50);
		twin->set(pos, bg);
	}
	outln("Point<T>::step ex");
}

template void Point<float>::draw()const;
template void Point<double>::draw()const;
template<typename T> void Point<T>::draw()const {

   glPushMatrix();
   glColor3f(1.0, 1.0, 1.0);
   outln("[" << pos[0] << "," << pos[1]<< "," << pos[2]<< "]");
   glTranslatef(pos[0],pos[1],pos[2]);
   glutWireCube(5);
   glPopMatrix();
   //
   if(twin) {
	   twin->draw();
   }
   if(Drawable<T>::withPath) {
	   Drawable<T>::path.draw();
   }
}

template void Point<float>::randomize(float*, float);
template void Point<double>::randomize(double*, double);
template<typename T> void Point<T>::randomize(T* trg,  T spread) {
	srand (time(null));
	trg[0] = spread * (RAND_MAX - rand())/(1.0 *RAND_MAX);
	trg[1] = spread * (RAND_MAX - rand())/(1.0 *RAND_MAX);
	trg[2] = spread * (RAND_MAX - rand())/(1.0 *RAND_MAX);
}

template bool Point<float>::insideBboxQ(const float*, const float*);
template bool Point<double>::insideBboxQ(const double*, const double*);
template<typename T> bool Point<T>::insideBboxQ(const T* pos, const T* bboxWhd){
	return pos[0] >= bboxWhd[0] && pos[0] <= bboxWhd[1] &&
			pos[1] >= bboxWhd[2] && pos[1] <= bboxWhd[3] &&
			pos[2] >= bboxWhd[4] && pos[2] <= bboxWhd[5];
}

template bool Point<float>::outsideBboxQ(const float*, const float*);
template bool Point<double>::outsideBboxQ(const double*, const double*);
template<typename T> bool Point<T>::outsideBboxQ(const T* pos, const T* bboxWhd){
	return pos[0] < bboxWhd[0] && pos[0] > bboxWhd[1] &&
			pos[1] < bboxWhd[2] && pos[1] > bboxWhd[3] &&
			pos[2] < bboxWhd[4] && pos[2] > bboxWhd[5];
}

template<typename T> bool Point<T>::scale(T& t0, T& t1, T test, T factor){
	T center,edge;
	bool changed = false;
	while(test < t0 || test > t1) {
		outln(test << " < " << t0 << " or > " << t1);
		changed = true;
		center = (t0 + t1)/Consts<T>::two;
		edge = (t1 - t0) * factor;
		t0 = center - edge/Consts<T>::two;
		t1 = center + edge/Consts<T>::two;
	}
	return changed;
}

template bool Point<float>::scale(Bbox<float>*, float);
template bool Point<double>::scale(Bbox<double>*, double);
template<typename T> bool Point<T>::scale(Bbox<T>* bbox, T growBy){
	bool changedx = scale(bbox->x0,bbox->x1, pos[0], growBy);
	bool changedy = scale(bbox->y0,bbox->y1, pos[1], growBy);
	bool changedz = scale(bbox->z0,bbox->z1, pos[2], growBy);
	if(changedx || changedy || changedz ) {
		outln("scale! [" << changedx << changedy << changedz << "] ---> " << bbox->toString() );
		return true;
	}
	return false;
}
template void Point<float>::lowPass(float*,const float*, float);
template void Point<double>::lowPass(double*, const double*, double);
template<typename T> void Point<T>::lowPass(T* linAcc, const T* acc, T alpha){
	// (currentAcc.x * lowPassFilteringFactor) + (lastAcc.x * (1.0f - lowPassFilteringFactor));
	linAcc[0] = acc[0]*alpha + gravity[0]* (1.0 - alpha);
	linAcc[1] = acc[1]*alpha + gravity[1]* (1.0 - alpha);
	linAcc[2] = acc[2]*alpha + gravity[2]* (1.0 - alpha);
}
template void Point<float>::hiPass(float*,const float*, float);
template void Point<double>::hiPass(double*, const double*, double);
template<typename T> void Point<T>::hiPass(T* linAcc, const T* acc, T alpha){
	//  currentAcc.x = lastAcc.x - ((lastAcc.x * highPassFilteringsFactor) + (currentAcc.x * (1.0f - highPassFilteringsFactor)));
	linAcc[0] = gravity[0] - ( gravity[0] * alpha + acc[0] * (1.0 - alpha));
	linAcc[1] = gravity[1] - ( gravity[1] * alpha + acc[1] * (1.0 - alpha));
	linAcc[2] = gravity[2] - ( gravity[2] * alpha + acc[2] * (1.0 - alpha));
}

template float Point<float>::extremityN(const float*, const float*);
template double Point<double>::extremityN(const double*, const double*);
template<typename T> T Point<T>::extremityN(const T* pos, const T* bboxWhd){
	// first find nearest face, then 'extremity' eN =
	// for dist to face (D) < average edge len (L):
	//  	eN = 1 / (1 + D/L)
	// else
	//		eN = 0
	return 0;
}

template void Point<float>::trans(float[3]);
template void Point<double>::trans(double[3]);
template<typename T> void Point<T>::trans(T del[3]){
	pos[0] += del[0];
	pos[1] += del[1];
	pos[2] += del[2];
}

template void Point<float>::makeCurrent();
template void Point<double>::makeCurrent();
template<typename T> void Point<T>::makeCurrent(){
	ltime = b_util::nowMillis();
	outln("makeCurrent ltime " << ltime);
}


template void Point<float>::play();
template void Point<double>::play();
template<typename T> void Point<T>::play(){
	ltime = b_util::nowMillis();
	outln("makeCurrent ltime " << ltime);
}
