/*
 * debug.cc
 *
 *  Created on: Dec 18, 2012
 *      Author: reid
 */

#include "debug.h"
#include <time.h>
#include "CuFunctor.h"

const long SECOND_MS = 1000l;

const long MINUTE_S = 60l;

const long MINUTE_MS = MINUTE_S * SECOND_MS;

const long HOUR_S = 60 * MINUTE_S;

const long HOUR_MS = HOUR_S * MINUTE_MS;

const long DAY_S = 24 * HOUR_S;

const long DAY_MS = DAY_S * SECOND_MS;

const float F_DAY_MS = DAY_MS * 1.f;

// todo:  generate this from enum CuMatrixException
string CuMatrixExceptionStrings[] = {
		"successEx",
		"illegalArgumentEx",
		"outOfBoundsEx",
		"columnOutOfBoundsEx",
		"rowOutOfBoundsEx",
		"notRowVectorEx",
		"notColumnVectorEx",
		"notSyncedEx",
		"notSyncedDevEx",
		"notSyncedHostEx",
		"cantSyncHostFromDeviceEx",
		"notSquareEx",
		"badDimensionsEx",
		"matricesOfIncompatibleShapeEx",
		"rowDimsDisagreeEx",
		"columnDimsDisagreeEx",
		"exceedsMaxBlockDimEx",
		"spansMultipleTileEx",
		"precisionErrorEx",
		"notImplementedEx",
		"nNeqPnotImplementedEx",
		"singularMatrixEx",
		"noDeviceBufferEx",
		"noHostBufferEx",
		"alreadyPointingDeviceEx",
		"hostAllocationFromDeviceEx",
		"hostReallocationEx",
		"smemExceededEx",
		"notResidentOnDeviceEx",
		"timerNotStartedEx",
		"timerAlreadyStartedEx",
		"wrongStreamEx",
		"insufficientGPUCountEx",
		"nullPointerEx",

};

string NvmlErrors[] = {
    "NVML_SUCCESS",
    "NVML_ERROR_UNINITIALIZED",
    "NVML_ERROR_INVALID_ARGUMENT",
    "NVML_ERROR_NOT_SUPPORTED",
   	"NVML_ERROR_NO_PERMISSION",
    "NVML_ERROR_ALREADY_INITIALIZED",
    "NVML_ERROR_NOT_FOUND",
    "NVML_ERROR_INSUFFICIENT_SIZE",
    "NVML_ERROR_INSUFFICIENT_POWER",
    "NVML_ERROR_DRIVER_NOT_LOADED",
    "NVML_ERROR_TIMEOUT",
    "NVML_ERROR_IRQ_ISSUE",
    "NVML_ERROR_LIBRARY_NOT_FOUND",
    "NVML_ERROR_FUNCTION_NOT_FOUND",
    "NVML_ERROR_CORRUPTED_INFOROM",
    "NVML_ERROR_GPU_IS_LOST",
    "NVML_ERROR_RESET_REQUIRED",
    "NVML_ERROR_OPERATING_SYSTEM",
};

const char* getNvmlErrorEnum(int en) {
	if(en > -1 && en < 17) {
		return NvmlErrors[en].c_str();
	}
	return "unknown nvml error";
}

string fromSeconds(double seconds) {
	double days = 0;
	if (seconds > DAY_S) {
		days = (int) (seconds / DAY_S);
	}
	double remainder = seconds - days * DAY_S;
	double hours = 0;
	if (remainder > HOUR_S) {
		hours = (int) (remainder / HOUR_S);
	}
	remainder = remainder - hours * HOUR_S;
	double minutes = 0;
	if (remainder > MINUTE_S) {
		minutes = (int) (remainder / MINUTE_S);
	}
	remainder = remainder - minutes * MINUTE_S;
	stringstream time;
	if(days > 0) {
		time << days << " days ";
	}
	if(hours > 0) {
		time << hours << "h ";
	}
	if(minutes > 0 ){
		time << minutes << "m ";
	}
	if  (remainder > 0 || (remainder == 0 && hours == 0 && minutes == 0)) {
		char buff[20];
		sprintf(buff,"%.3f", remainder);
		time << buff << "s";
	}
	return time.str();
}

string fromMillis(double millis) {
	return fromSeconds(millis);
}

string fromMicros(double micros) {
	return fromMillis(micros/1000.);
}

string niceEpochMicros(long micros) {
	time_t argT = micros/1000;
	struct tm* millisTm = localtime(&argT);
	return string(asctime(millisTm));
}

void dummy_1f1476811d0646db8cc4de3c21f85825() {}


float pctChg(float a, float b){  return .1*((int)((1 - a/b)*1000));}
///float pctChg(float a, float b){  return .1*((int)(((b-a)/b)*1000));}



template <typename T> void printAllDeviceGFlops() {
	int devCount, currDev;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	checkCudaErrors(cudaGetDevice(&currDev));
	    float gflorps=0;
//#pragma omp parallel for private(gflorps)
	for(int i = 0; i < devCount;i++) {
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
			unaryOpIndexMbrs<T>::setupAllFunctionTables(i);
	#else
			unaryOpIndexMbrs<T>::setupAllMethodTables(i);
	#endif
#endif
		gflorps = util<T>::vAddGflops(i);
		flprintf("\n\n%s vector add @ %f\n\n", gpuNames[i].c_str(), gflorps);
	}
}
template void printAllDeviceGFlops<float>();
template void printAllDeviceGFlops<double>();
template void printAllDeviceGFlops<ulong>();

