/*
 * debug.h
 *
 *  Created on: Jul 24, 2012
 *      Author: reid
 */

#pragma once

//#define TESTTMPLT

#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include "util.h"
#include "caps.h"
#include "CMap.h"
#include <thread>
#include <cublas_v2.h>

using std::string;
using std::stringstream;
using std::cout;
using std::endl;

extern __constant__ uint debugFlags;
extern uint hDebugFlags;
extern string gpuNames[];
extern string CuMatrixExceptionStrings[];

extern const long SECOND_MS;
extern const long MINUTE_S;
extern const long MINUTE_MS;
extern const long HOUR_S;
extern const long HOUR_MS;
extern const long DAY_S;
extern const long DAY_MS;
extern const float F_DAY_MS;
enum CuMatrixException {
	successEx,
	illegalArgumentEx,
	outOfBoundsEx,
	columnOutOfBoundsEx,
	rowOutOfBoundsEx,
	notRowVectorEx,
	notColumnVectorEx,
	notSyncedEx,
	notSyncedDevEx,
	notSyncedHostEx,
	cantSyncHostFromDeviceEx,
	notSquareEx,
	badDimensionsEx,
	matricesOfIncompatibleShapeEx,
	rowDimsDisagreeEx,
	columnDimsDisagreeEx,
	exceedsMaxBlockDimEx,
	spansMultipleTileEx,
	precisionErrorEx,
	notImplementedEx,
	nNeqPnotImplementedEx,
	singularMatrixEx,
	noDeviceBufferEx,
	noHostBufferEx,
	alreadyPointingDeviceEx,
	hostAllocationFromDeviceEx,
	hostReallocationEx,
	smemExceededEx,
	notResidentOnDeviceEx,
	timerNotStartedEx,
	timerAlreadyStartedEx,
	wrongStreamEx,
	insufficientGPUCountEx,
	nullPointerEx,
	multipleGpusEx,
};
extern  __device__ CuMatrixException lastEx;
typedef __device_builtin__ enum CuMatrixException CuMatrixException_t;

#define debugUseTimers 	1
#define debugExec 		(1 << 1)
#define debugMem 		(1 << 2)
#define debugCheckValid (1 << 3)
#define debugFtor  		(1 << 4)
#define debugCopy  		(1 << 5)
#define debugCopyDh  	(1 << 6)
#define debugFill  		(1 << 7)
#define debugMatProd  	(1 << 8)
#define debugCons  		(1 << 9)
#define debugDestr 		(1 << 10)
#define debugTxp 		(1 << 11)
#define debugRefcount  	(1 << 12)
#define debugVerbose  	(1 << 13)
#define debugNn  		(1 << 14)
#define debugCg  		(1 << 15)
#define debugMultGPU  	(1 << 16)
#define debugPm  		(1 << 17)
#define debugMaths (1 << 18)
#define debugAnomDet	(1 << 19)
#define debugStream  	(1 << 20)
#define debugRedux 		(1 << 21)
#define debugSpoofSetLastError (1 << 22)
#define debugMatProdBlockResize  	(1 << 23)
#define debugMatStats  	(1 << 24)
#define debugTiler  	(1 << 25)
#define debugUnaryOp 	(1 << 26)
#define debugPrecision 	(1 << 27)
#define debugMeans 		(1 << 28)
#define debugBinOp		(1 << 29)
#define debugFile		(1 << 30)

const auto allChoice = "all";
const auto anomChoice = "anom";
const auto memChoice = "mem";
const auto copyChoice = "copy";
const auto copyDhChoice = "copyDh";
const auto execChoice = "exec";
const auto fillChoice = "fill";
const auto ftorChoice = "ftor";
const auto matprodChoice = "matprod";
const auto matprodBlockResizeChoice = "mpbr";
const auto debugMatStatsChoice = "stats";

const auto consChoice = "cons";
const auto destrChoice = "dstr";
const auto refcountChoice = "ref";
const auto verboseChoice = "verb";
const auto syncChoice = "sync";
const auto nnChoice = "nn";
const auto cgChoice = "cg";
const auto txpChoice = "txp";
const auto pmChoice = "pack";
const auto smallBlkChoice = "sblk";
const auto medBlkChoice = "mblk";
const auto lrgBlkChoice = "lblk";
const auto debugMultGPUChoice = "gpu";
const auto debugMillisForMicrosChoice = "m4m";
const auto debugReduxChoice = "rdx";
const auto debugTilerChoice = "tlr";
const auto debugUnaryOpChoice = "uny";
const auto debugPrecisionChoice = "prec";
const auto debugMeansChoice = "mean";
const auto debugBinOpChoice = "bnop";
const auto debugCheckValidChoice = "dcv";

__host__ __device__ const char* __cudaGetErrorEnum(cudaError_t res);
__host__ __device__ const char *__cublasGetErrorEnum(cublasStatus_t error);

void cherr_(cudaError_t err, char* file, int line);
//#define cherr(exp) cherr_((exp))
#define chsuckor(exp,exp2) if((exp)!= cudaSuccess && (exp) != (exp2)) {printf( "%s : %d --> %s\n", __FILE__, __LINE__ , __cudaGetErrorEnum(exp));assert(0);}
//#define cherr(exp) if(exp != cudaSuccess) {printf( "%s.%s : %d --> %s\n", __FILE__, __PRETTY_FUNCTION__, __LINE__ , __cudaGetErrorEnum(exp));assert(0);}
//#define cherr(exp) if(exp != cudaSuccess) {printf( "%s --> %s\n", __FILE__, __cudaGetErrorEnum(exp));assert(0);}
//#define cherr(exp) if(exp != cudaSuccess) {printf( "%d --> %s\n", __LINE__ , __cudaGetErrorEnum(exp));assert(0);}
//#define cherr(exp) if(exp != cudaSuccess) {printf( "%s : %d\n", __FILE__, __LINE__ );assert(0);}
__host__ __device__ inline void cherrp(cudaError exp);
#define arrrg(exp) assert(exp == cudaSuccess)
#define maxAbs(exp) ((exp) < 0) ? -(exp) : (exp)
#define czeckerr(exp) do { \
			if(exp != cudaSuccess) { \
				printf( "%s : %d --> %s\n", __FILE__, __LINE__ , __cudaGetErrorEnum(exp)); \
				assert(0);	}\
		}while(0)


__host__ __device__ __forceinline__ void cherrf(cudaError_t exp) {
	if(exp != cudaSuccess) printf( "%s : %d --> %s\n", __FILE__, __LINE__ , __cudaGetErrorEnum(exp) );
}

#ifndef __CUDA_ARCH__
#define flprintf( format, ...) printf ( "%s:%d %s " format, __FILE__, __LINE__, __func__,  __VA_ARGS__)
#else
#define flprintf( format, ...) printf ( "[d]%s:%d %s " format, __FILE__, __LINE__, __func__,  __VA_ARGS__)
#endif

// can't print from within functors with above, not curious enough to identify cause
#define flnfprintf( format, ...) printf ( "%s:%d " format, __FILE__, __LINE__,  __VA_ARGS__)

#define ERR_EQ(X,Y) do { if ((X) == (Y)) { \
            fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
            exit(-1);}} while(0)

#define outln(exp) cout << __FILE__ << "("<< __LINE__ << "): " << exp << endl
#define FL  (" " + __FILE__ + ":" + __LINE__ + " " )
#define soutln(exp) s << __FILE__ << "("<< __LINE__ << "): " << exp << "\n\n"
#define sout(exp) s << __FILE__ << "("<< __LINE__ << "): " << exp
#define doutln(exp) printf("%s(%d): %s\n", __FILE__,  __LINE__ ,exp)
#define dout(exp) printf("%s(%d): %s", __FILE__,  __LINE__ ,exp)
#define tout(exp) cout << __FILE__ << "("<< __LINE__ << "): " << exp
#define ot(exp) cout << exp
#define _at() cout << "at " << __FILE__ << "("<< __LINE__ << ")" << endl
#define nicen(exp) b_util::unmangl( typeid( exp ).name())
#define prloc() 	printf(__FILE__ ": %d ", __LINE__)
// prlocf is a compound statement and must be commented out with {}
#ifndef __CUDA_ARCH__
#define prlocf(exp) 	printf(__FILE__ "(%d): " exp, __LINE__)
#else
#define prlocf(exp) 	printf( "[d]" __FILE__ "(%d): " exp, __LINE__)
#endif

string fromSeconds(double seconds);
string fromMillis(double millis);
string fromMicros(double micros);
string niceEpochMicros(long micros);

template<typename T> string niceVec(T* v) {
	stringstream ss;
	ss << v[0] << ", " << v[1] << ", " << v[2];
	return ss.str();
}

template <typename K, typename V> void printMap(string name, std::map<K,V>& theMap) {
	typedef typename map<K, V>::iterator iterator;
	iterator it = theMap.begin();
	cout << name.c_str() << endl;
	while(it != theMap.end()) {
		cout << "\t" << (*it).first << " -> " << (*it).second << endl;
		it++;
	}
}
template <typename K, typename V> void printMap(string name,CMap<K,V>& theMap) {
	typename CMap<K,V>::iterator it = theMap.begin();
	cout << name.c_str() << endl;
	while(it != theMap.end()) {
		cout << "\t" << (*it).first << " -> " << (*it).second << endl;
		it++;
	}
}

__host__ __device__ void printLongSizes();
template<typename T> void printObjSizes();

__host__ __device__ void setLastError(CuMatrixException lastEx);
__host__ __device__ CuMatrixException getLastError();


/*
 * checkDebug / set[All/Curr]GpuDebugFlags
 */
inline __host__ __device__ bool checkDebug(uint flags) {
#ifndef __CUDA_ARCH__
	return hDebugFlags & flags;
#else
	//#ifdef CuMatrix_DebugBuild
		return debugFlags & flags;
	//#else
	//	return false;
	//#endif
#endif
}

void setAllGpuDebugFlags(uint flags, bool orThem, bool andThem);
void setCurrGpuDebugFlags(uint flags, bool orThem, bool andThem,  cudaStream_t stream = 0);

#define VerbOn() setCurrGpuDebugFlags( debugVerbose,true,false)
#define VerbOff() setCurrGpuDebugFlags( ~debugVerbose,false,true)

template <typename T> void printAllDeviceGFlops();

float pctChg(float a, float b);
__host__ __device__ CuMatrixException getLastError();

class ostreamlike {
public:
	__host__ __device__ ostreamlike();
	__host__ __device__ ~ostreamlike();
	__host__ __device__ ostreamlike& write(int n);
	__host__ __device__  ostreamlike& write(char* n);
	__host__ __device__  ostreamlike& write(const char* n);
	__host__ __device__  ostreamlike& write(char c);
	__host__ ostreamlike& write(const string& s);
	__host__ __device__ ostreamlike& write(float n);
	__host__ __device__ ostreamlike& write(double n);
	__host__ __device__ ostreamlike& write(long n);
	__host__ __device__ ostreamlike& write(unsigned int n);
	__host__ __device__ ostreamlike& write(unsigned long n);
	__host__ __device__ ostreamlike& write(bool n);
	__host__ __device__ ostreamlike& write(const float* p);
	__host__ __device__ ostreamlike& write(const double* p);
	__host__ __device__ ostreamlike& write(const int* p);
	__host__ __device__ ostreamlike& write(const void* p);
	__host__ __device__ ostreamlike& write(const long* p);
};

// operator for types that is supported ostreamlike internally
template <typename type>
__host__ __device__ ostreamlike& operator<<(ostreamlike& stream, const type& data) {
  return stream.write(data);
}

