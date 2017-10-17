/*
 * debug.cc
 *
 *  Created on: Dec 18, 2012
 *      Author: reid
 */

#include "debug.h"
#include <time.h>
#include "CuFunctor.h"
#include "CuMatrix.h"

const long SECOND_MS = 1000l;

const long MINUTE_S = 60l;

const long MINUTE_MS = MINUTE_S * SECOND_MS;

const long HOUR_S = 60 * MINUTE_S;

const long HOUR_MS = HOUR_S * MINUTE_MS;

const long DAY_S = 24 * HOUR_S;

const long DAY_MS = DAY_S * SECOND_MS;

const float F_DAY_MS = DAY_MS * 1.f;

const long WEEK_S = 7 * DAY_S;

const long WEEK_MS = WEEK_S * SECOND_MS;

const float F_WEEK_MS = WEEK_MS * 1.f;

const char * allChoice = "all";
const char * anomChoice = "anom";
const char * memChoice = "mem";
const char * copyChoice = "copy";
const char * copyDhChoice = "copyDh";
const char * execChoice = "exec";
const char * fillChoice = "fill";
const char * ftorChoice = "ftor";
const char * matprodChoice = "matprod";
const char * debugMatStatsChoice = "stats";
const char * consChoice = "cons";
const char * destrChoice = "dstr";
const char * refcountChoice = "ref";
const char * verboseChoice = "verb";
const char * syncChoice = "sync";
const char * nnChoice = "nn";
const char * cgChoice = "cg";
const char * txpChoice = "txp";
const char * pmChoice = "pack";
const char * smallBlkChoice = "sblk";
const char * medBlkChoice = "mblk";
const char * lrgBlkChoice = "lblk";
const char * debugMultGPUChoice = "gpu";
const char * debugMillisForMicrosChoice = "m4m";
const char * debugReduxChoice = "rdx";
const char * debugTimerChoice = "tmr";
const char * debugTilerChoice = "tlr";
const char * debugUnaryOpChoice = "uny";
const char * debugPrecisionChoice = "prec";
const char * debugMeansChoice = "mean";
const char * debugBinOpChoice = "bnop";
const char * debugCheckValidChoice = "dcv";
const char * debugMembloChoice = "mbl";

string DebugFlagsStr[] = {
"debugUseTimers",
"debugExec",
"debugMem",
"debugCheckValid",
"debugFtor",
"debugCopy",
"debugCopyDh",
"debugFill",
"debugMatProd",
"debugCons",
"debugDestr",
"debugTxp",
"debugRefcount",
"debugVerbose",
"debugNn",
"debugCg",
"debugMultGPU",
"debugPm",
"debugMaths",
"debugAnomDet",
"debugStream",
"debugRedux",
"debugSpoofSetLastError",
"debugTimer",
"debugMatStats",
"debugTiler",
"debugUnaryOp",
"debugPrecision",
"debugMeans",
"debugBinOp",
"debugFile",
"debugMemblo"};

string UsageStrPreamble =
 "usage:\n\n\tcumatrel|cumatrest [options]\n\n"
 "Usage options:\n";

string UsagesStr[] = {
		"-usage\t\t---\t\tdisplay this",
		"-dbg=<list>\t---\t\tspecify comma-sep'd list of one of more debugging options (listed below)",
		"-dev=N\t\t---\t\tspecify which device to be default (for single device tests)",
		"-stopAt=N\t---\t\tstop after this test (numbers start at 0 in code order)",
		"-blas\t\t---\t\tuse cublas for matrix math (really just multiplication",
		"-cool\t\t---\t\tselect whichever device is coolest (requires NVML) for each test",
		"-test=N\t\t---\t\trun test #N only (0 based, ordered by appearance in code)",
		"-krn=N\t\t---\t\tuse matrix product kernel N for matrix mult"
};

string DebugOptionsPreamble ="List of Debug Options\n\t(usually to enable detailed log messages for classes of operation)\n";


string DebugOptionsStr[] = {
"debugUseTimers\t\ttmr",
"debugExec\t\texec",
"debugMem\t\tmem",
"debugCheckValid\t\tdcv",
"debugFtor\t\tftor",
"debugCopy\t\tcopy",
"debugCopyDh\t\tcopyDh",
"debugFill\t\tfill",
"debugMatProd\t\tmatprod",
"debugCons\t\tcons",
"debugDestr\t\tdstr",
"debugTxp\t\ttxp",
"debugRefcount\t\tref",
"debugVerbose\t\tverb",
"debugNn\t\t\tnn",
"debugCg\t\t\tcg",
"debugMultGPU\t\tgpu",
"debugPm\t\t\tpack",
"debugMaths\t\t<?>",
"debugAnomDet\t\tanom",
"debugStream\t\t<?>",
"debugRedux\t\trdx",
"debugSpoofSetLastError\t<?>",
"debugTimer\t\ttmr",
"debugMatStats\t\tstats",
"debugTiler\t\ttlr",
"debugUnaryOp\t\tuny",
"debugPrecision\t\tprec",
"debugMeans\t\tmean",
"debugBinOp\t\tbnop",
"debugFile\t\tfile",
"debugMemblo\t\tmbl"};

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
		"needsTilingEx",
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
		"nullPointerEx"
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
	return fromSeconds(millis/1000.0);
}

string fromMicros(double micros) {
	return fromMillis(micros/1000000.0);
}

string niceEpochMicros(long micros) {
	time_t argT = micros/1000;
	struct tm* millisTm = localtime(&argT);
	return string(asctime(millisTm));
}

string dbgStr( ) {
	string ret = "";
	for(int i=0; i < 32; i++) {
		if( (hDebugFlags >> i) & 1) {
			ret += DebugFlagsStr[i] + " ";
		}
	}
	return ret;
}

void memblyFlf(const char * file , int line, const char * func){
	size_t stFree, stTotal;
	cherr(cudaMemGetInfo(&stFree, &stTotal));
	double pct =  100 * (1 - 1.* (stTotal-stFree)/stTotal);
	if(checkDebug(debugMemblo))
		printf("memser %s : %d %s free %.2f%% %s total %s\n",
				file, line, func,
				pct, b_util::expNotationMem(stFree).c_str(), b_util::expNotationMem(stTotal).c_str());
}

void dummy_1f1476811d0646db8cc4de3c21f85825() {}

float pctChg(float a, float b){  return .1*((int)((1 - a/b)*1000));}

template <typename T> void printAllDeviceGFlops() {
	int devCount, currDev;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	checkCudaErrors(cudaGetDevice(&currDev));
	    float gflorps=0;
//#pragma omp parallel for private(gflorps)
	for(int i = 0; i < devCount;i++) {
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
			CuTimer timer;
			timer.start();
			unaryOpIndexMbrs<T>::setupAllFunctionTables(i);
			float time = timer.stop()/9;
/*
		    CuMatrix<T>::incDhCopy("FunctionTableMgr<T>::setup0sFunctionTables", sizeof(typename UnaryOpF<T,0>::uopFunction),time, Uop0sLast);
		    CuMatrix<T>::incDhCopy("FunctionTableMgr<T>::setup0sFunctionTables", sizeof(typename UnaryOpIndexF<T,0>::iopFunction),time, Iop0sLast);
		    CuMatrix<T>::incDhCopy("FunctionTableMgr<T>::setup0sFunctionTables", sizeof(typename BinaryOpF<T,0>::bopFunction),time, Bop0sLast);

		    CuMatrix<T>::incDhCopy("FunctionTableMgr<T>::setup1sFunctionTables", sizeof(typename UnaryOpF<T,1>::uopFunction),time, Uop1sLast);
		    CuMatrix<T>::incDhCopy("FunctionTableMgr<T>::setup1sFunctionTables", sizeof(typename UnaryOpIndexF<T,1>::iopFunction),time, Iop1sLast);
		    CuMatrix<T>::incDhCopy("FunctionTableMgr<T>::setup1sFunctionTables", sizeof(typename BinaryOpF<T,1>::bopFunction),time, Bop1sLast);
		    CuMatrix<T>::incDhCopy("FunctionTableMgr<T>::setup2sFunctionTables", sizeof(typename UnaryOpF<T,2>::uopFunction),time, Uop2sLast);
		    CuMatrix<T>::incDhCopy("FunctionTableMgr<T>::setup2sFunctionTables", sizeof(typename UnaryOpIndexF<T,2>::iopFunction),time, Iop2sLast);
		    CuMatrix<T>::incDhCopy("FunctionTableMgr<T>::setup3sFunctionTables", sizeof(typename UnaryOpIndexF<T,3>::iopFunction),time, Iop3sLast);
*/
	#else
			CuTimer timer;
			timer.start();
			unaryOpIndexMbrs<T>::setupAllMethodTables(i);
			float time = timer.stop()/9;
/*
		    CuMatrix<T>::incDhCopy("MethodTableMgr<T>::setup0sMethodTables",  sizeof(typename UnaryOpF<T,0>::uopMethod),time, Uop0sLast);
		    CuMatrix<T>::incDhCopy("MethodTableMgr<T>::setup0sMethodTables", sizeof(typename UnaryOpIndexF<T,0>::iopMethod),time, Iop0sLast);
		    CuMatrix<T>::incDhCopy("MethodTableMgr<T>::setup0sMethodTables", sizeof(typename BinaryOpF<T,0>::bopMethod),time, Bop0sLast);
		    CuMatrix<T>::incDhCopy("MethodTableMgr<T>::setup1sMethodTables",  sizeof(typename UnaryOpF<T,1>::uopMethod),time, Uop1sLast);
		    CuMatrix<T>::incDhCopy("MethodTableMgr<T>::setup1sMethodTables", sizeof(typename UnaryOpIndexF<T,1>::iopMethod),time, Iop1sLast);
		    CuMatrix<T>::incDhCopy("MethodTableMgr<T>::setup1sMethodTables", sizeof(typename BinaryOpF<T,1>::bopMethod),time, Bop1sLast);
		    CuMatrix<T>::incDhCopy("MethodTableMgr<T>::setup2sMethodTables", sizeof(typename UnaryOpF<T,2>::uopMethod),time, Uop2sLast);
		    CuMatrix<T>::incDhCopy("MethodTableMgr<T>::setup2sMethodTables", sizeof(typename UnaryOpIndexF<T,2>::iopMethod),time, Iop2sLast);
		    CuMatrix<T>::incDhCopy("MethodTableMgr<T>::setup3sMethodTables", sizeof(typename UnaryOpIndexF<T,3>::iopMethod),time, Iop3sLast);
*/
	#endif
#endif
		gflorps = util<T>::vAddGflops(i);
		flprintf("\n\n%s vector add @ %f\n\n", gpuNames[i].c_str(), gflorps);
	}
}
template void printAllDeviceGFlops<float>();
template void printAllDeviceGFlops<double>();
template void printAllDeviceGFlops<ulong>();

