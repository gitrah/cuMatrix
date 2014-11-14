/*
 * AccelEvents.cc
 *
 */
#include "AccelEvents.h"
#include <GL/glut.h>
#include "../CuMatrix.h"
template<typename T> AccelEvents<T>::AccelEvents(int samples) : samples(samples), currSample(0), rank(3), times(null), accel(null)  {
	times = new long[samples];
	cudaMallocHost((void**)&accel, samples * 3 * sizeof(T));
	if(This != null) {
		dthrow ( ThisAlreadySet());
	}
	This = this;
};

template AccelEvents<float>::~AccelEvents();
template AccelEvents<double>::~AccelEvents();
template<typename T> AccelEvents<T>::~AccelEvents()  {
	if(times != null) {
		delete[] times;
	}
	if(accel != null) {
		cudaFreeHost(accel);
	}
};

template int AccelEvents<float>::duringQ(long time)const ;
template int AccelEvents<double>::duringQ(long time)const ;
template<typename T> int AccelEvents<T>::duringQ(long time)const  {
	outln("duringQ time " << time);
	if(samples == 0){
		outln("duringQ no samples");
		return -1;
	}

	int i = 0;
	int timeBeforeIdx = -1;
	while( i < samples) {
		if(time >= times[i]) {
			//outln("duringQ i " << i << " time after " << times[i]);
			timeBeforeIdx = i;
		} else if (time < times[i]) {
			if(timeBeforeIdx > -1) {
				outln("duringQ timeBeforeIdx is " << timeBeforeIdx);
				break;
			}
		}
		i++;
	}
	return i == samples ? -1 : timeBeforeIdx;
}
template void AccelEvents<float>::linterp(float* , long, int)const ;
template void AccelEvents<double>::linterp(double* , long, int )const ;
template<typename T> void AccelEvents<T>::linterp(T* out, long time, int idx)const  {
	if(idx <0) return;
	if(idx == samples -1){
		out[0]=0;out[1]=0;out[2]=0;
	} else {
		//T slope[] = {0,0,0};
		//long dt = times[idx+1]-times[idx];
		out[0] = (accel[ (idx+1) * vecSize ] - accel[idx * vecSize]);
		out[1] = (accel[ (idx+1) * vecSize +1] - accel[idx * vecSize +1]);
		out[2] = (accel[ (idx+1) * vecSize +2] - accel[idx * vecSize +2]);
	//	outln("slope " << niceVec<T>(slope));
/*
		out[0] = slope[0] * ( time - times[idx]);// + accel[idx * vecSize];
		out[1] = slope[1] * ( time - times[idx + 1]);// + accel[idx * vecSize + 1];
		out[2] = slope[2] * ( time - times[idx + 2]);// + accel[idx * vecSize + 2];
*/
	}
}

template void AccelEvents<float>::update(float* ,float *, float*,const float*, float);
template void AccelEvents<double>::update(double* ,double *, double*, const double*, double);
template<typename T> void AccelEvents<T>::update(T* pos, T* gravity, T* vel, const T* acc, T durationMicros) {
	T linAcc[3];
	const T alpha = 0.8;
	outln("gravity " << niceVec(gravity));

	T duration = durationMicros/100.0;
	// low/hi filtering, ala android docs
	gravity[0] = alpha * gravity[0] + (1-alpha)*acc[0];
	gravity[1] = alpha * gravity[1] + (1-alpha)*acc[1];
	gravity[2] = alpha * gravity[2] + (1-alpha)*acc[2];

	Vutil<T>::sub3(linAcc, acc, gravity);
	outln("linAcc " << niceVec(linAcc));

	T dPos[3];
	T nuVel[3];
	Vutil<T>::copy3(nuVel, linAcc);
	Vutil<T>::scale(nuVel,duration);

	dPos[0] = (vel[0] + nuVel[0]) * duration / 2;
	dPos[1] = (vel[1] + nuVel[1]) * duration / 2;
	dPos[2] = (vel[2] + nuVel[2]) * duration / 2;
	outln("dpos " << niceVec(dPos));
	Vutil<T>::add3(pos, pos, dPos);
	vel[0] = nuVel[0];
	vel[1] = nuVel[1];
	vel[2] = nuVel[2];
}

template void AccelEvents<float>::average(float* ,const float *, const float*, int);
template void AccelEvents<double>::average(double* ,const double *, const double*, int);
template<typename T> void AccelEvents<T>::average(T* out, const T* in1, const T* in2, int rank) {
	if(rank == 3) {
		out[0] = (in1[0]+in2[0])/2;
		out[1] = (in1[1]+in2[1])/2;
		out[2] = (in1[2]+in2[2])/2;
	} else {
		for(int i = 0; i < rank; i++ ) {
			out[i] = (in1[i] + in2[i])/2;
		}
	}
}

template void AccelEvents<float>::event(float*, float *,float *, long,long)const;
template void AccelEvents<double>::event(double*, double *,double *,long,long)const;
template<typename T> void AccelEvents<T>::event(T* pos, T* gravity, T* vel, long time, long dtime)const  {
	outln("event time " << time);
	int idx = duringQ(time);
	int idx2 = duringQ(time+dtime);

	T acc[3];
	T duration;
	//long nextTime;
	if(idx > -1 && idx2 > -1) {
		// update vel
		if(idx == idx2) {
			if(idx == samples - 1) {
				//nextTime = times[idx] + (times[idx]-times[idx-1]);
				Vutil<T>::copy3(acc,accel+ idx * vecSize);
			} else {
				linterp(acc, time, idx);
			}
			duration = (time-times[idx] + dtime)/1000.;
			update(pos, gravity, vel, acc, duration);
		} else { // dtime spans several events
			long ctime = time;
			for(int i = 0; i < idx2-idx; i++) {
				linterp(acc, ctime, idx +i);  // get acc at curr time
				//average(acc, acc, accel+ (idx+i+1)*vecSize); // get avg acc over span
				duration = i < idx2-idx -1 ? (times[idx+i + 1] - ctime)/1000. : (time+dtime - times[idx+i])/1000.;
				update(pos,gravity,vel,acc,duration);
				ctime = times[idx+i];
			}
		}
	} else {
		// update pos w/ const vel
		pos[0] += vel[0] * dtime/1000000.0;
		pos[1] += vel[1] * dtime/1000000.0;
		pos[2] += vel[2] * dtime/1000000.0;
	}
}

template string AccelEvents<float>::toString() const;
template string AccelEvents<double>::toString()const;
template<typename T> string AccelEvents<T>::toString() const  {
	stringstream ss;
	ss << "Accel stream with " << samples << " samples from " << niceEpochMicros(times[0]) <<
			" to " << niceEpochMicros(times[samples-1]) <<
			" [ " << fromMicros(times[samples-1] - times[0]) << " ]\n" ;
	for(int i = 0; i < samples; i++) {
		ss << times[i] << ": " << accel[i * vecSize] << ", "<< accel[i * vecSize+1] << ", "<< accel[i * vecSize+2] << "\n";
	}
	return ss.str();
}

template void AccelEvents<float>::syncToNow(long);
template void AccelEvents<double>::syncToNow(long);
template<typename T> void AccelEvents<T>::syncToNow(long plusMillis)  {
	if(samples > 0){
		long nowMillis = b_util::nowMillis();
		long delta = nowMillis - times[0] + plusMillis;
		for(int i = 0; i < samples; i++) {
			times[i] += delta;
		}
	}
}

template long AccelEvents<float>::averageDtime();
template long AccelEvents<double>::averageDtime();
template<typename T> long AccelEvents<T>::averageDtime()  {
	long step =0;
	for(int i = 1; i < samples; i++) {
		step += times[i]-times[i-1];
	}
	return step/samples;
}

template bool AccelEvents<float>::intersectQ(const AccelEvents<float>*)const;
template bool AccelEvents<double>::intersectQ(const AccelEvents<double>*)const;
template<typename T> bool AccelEvents<T>::intersectQ(const AccelEvents<T>* o) const {
	long myf = first(), myl = last();
	long of = o->first(), ol = o->last();
	return (myf >= of && myf < ol) ||
		(myl >= of && myl < ol) ||
		(of >= myf && of < myl) ||
		(ol >= myf && ol < myl);
}

template void AccelEvents<float>::play(float*, float*, float*);
template void AccelEvents<double>::play(double*,double*,double*);
template<typename T> void AccelEvents<T>::play(T* pos, T* gravity, T* vel){
	if(currSample == samples - 1) {
		currSample = 0;
	}
	outln("currSample " << currSample);
	long delayMicros = (currSample < samples - 1) ? times[currSample + 1] -times[currSample] : averageDtime() ;
	if(delayMicros > sequenceThresholdMicros) {
		outln("assuming new sequence ( " << delayMicros << " >> " << sequenceThresholdMicros << ")");
	} else {
		outln("delay " << delayMicros <<" (" << times[currSample + 1]  << " - " << -times[currSample] << ")");
		if(DelayMillisForMicros)
			delayMillis(delayMicros % 300);
		else
			delayMics(delayMicros % 300);

		T acc[3];
		linterp(acc, times[currSample], currSample);
		outln("accel " << Vutil<T>::len(acc) << ", " << niceVec(acc));

		T posC[3];
		T velC[3];
		T accC[3];
		Vutil<T>::copy3(posC,pos);
		outln("pos " << niceVec(pos));
		assert(Vutil<T>::dist3(posC,pos) < 0.0000001);
		Vutil<T>::copy3(velC,vel);
		assert(Vutil<T>::dist3(velC,vel) < 0.0000001);
		update( pos, gravity, vel, acc, delayMicros);
		//T dvelLen = Vutil<T>::dist3(velC,vel);
		Vutil<T>::sub3(accC,vel,velC);
		Vutil<T>::scale(accC,1000000.0/delayMicros);
		outln("accC " << niceVec(accC));
		T accLen = Vutil<T>::len(accC);

		outln("delta2 pos " << Vutil<T>::dist3Sqr(posC,pos));
		outln("accLen " << accLen);
		outln("|calcAcc - actAcc| " << Vutil<T>::dist3(accC, acc));
		outln("vel " << niceVec(vel));
	}
	currSample++;
}

template AccelEvents<float>* AccelEvents<float>::fromMatrix(const CuMatrix<float>& m );
template AccelEvents<double>* AccelEvents<double>::fromMatrix(const CuMatrix<double>& m );
template<typename T> AccelEvents<T>* AccelEvents<T>::fromMatrix(const CuMatrix<T>& m ) {
	AccelEvents<T>* events = new AccelEvents<T>(m.m);
	for(uint i =0; i < m.m;i++) {
		events->times[i] = m.get(i,0);
		events->accel[i * vecSize] = m.get(i,1);
		events->accel[i * vecSize+1] = m.get(i,2);
		events->accel[i * vecSize+2] = m.get(i,3);
	}
	return events;
}



