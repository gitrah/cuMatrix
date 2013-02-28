/*
 *
 *  Created on: Jul 23, 2012
 *      Author: reid
 */

#ifndef UTIL_H_
#define UTIL_H_
#include <map>
#include <vector>
#include <string>
#include <time.h>

#include <cuda_runtime_api.h>

#include "debug.h"
using namespace std;

#ifdef __GNUC__
    #define MAYBE_UNUSED __attribute__((used))
#elif defined _MSC_VER
    #pragma warning(disable: Cxxxxx)
    #define MAYBE_UNUSED
#else
    #define MAYBE_UNUSED
#endif

#define null NULL
#define enuff(step, target)   ((target + step  - 1)/step)
#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif
#ifndef Pi
#define Pi 3.141592653589793
#endif
#ifndef ONE_OVER_SQR_2PI
#define ONE_OVER_SQR_2PI 	0.3989422804014
#endif
#ifndef ONE_OVER_2PI
#define ONE_OVER_2PI 	0.159154943
#endif

#define Giga (1024*1024*1024)
#define tOrF(exp) (exp ? "true" : "false")
#define Mega (1024*1024)
#define Kila (1024)

#ifndef WARP_SIZE
#define WARP_SIZE 32
#define MAX_BLOCK_DIM   64
#define MAX_THREAD_DIM   256
#endif

// host-device modification status
enum Modification {
	mod_neither, mod_synced, mod_host, mod_device,
};

typedef unsigned int uint;
typedef pair<uint,uint> uintPair;
typedef unsigned long ulong;
typedef pair<ulong,ulong> ulongPair;

template<typename T, typename U> string pp(pair<T,U>& p);

struct b_util {
	template<typename T> inline static string pv1_3(T v) {
		char buff[5];
		sprintf(buff,"%1.3g", v);
		return string(buff);
	}


	static inline ulong twoDTo1D(dim3& idx3, dim3& basis) { return idx3.x + idx3.y * basis.x; }
	static inline ulong threeDTo1D(dim3& idx3, dim3& basis) { return idx3.x + idx3.y * basis.x + idx3.z * basis.x * basis.y; }
	static void to1D(dim3& d3);
	static vector<string> readLines(const char * );
	static vector<string> split(string);

	static const char * modStr(Modification lastMod);

	static double diffclock(clock_t clock1,clock_t clock2);
	template<typename T> static string pvec(vector<T>& v);
	static string pxd(const dim3& grid,const  dim3& block);
	static string pexec(const dim3& grid,const  dim3& block, uint smem);
	static string pd3(const dim3& d3);
	static string pd3(const dim3* d3);
	static string toStr(const uintPair& p);
	static void syncGpu(const char * msg = null);
	static uint nextPowerOf2(uint x);
	template<typename T> static inline T nextFactorable(T start, uint factor) {
		return static_cast<T>( ceilf(start*1./factor)*factor);
	}
	static bool isPow2(uint x);
	static uint largestMutualFactor(uint count, uint args[]);
	static void execContext(int threads, uint count, dim3& dBlocks,
			dim3& dThreads);
	static void execContext(int threads, uint count, uint spacePerThread,
			dim3& dBlocks, dim3& dThreads, uint& smem);
	static void execContextSmem(int threads, uint count, dim3& dBlocks,
			dim3& dThreads);
	static inline uint vol(dim3& d3) { return d3.x * d3.y *d3.z; }

	static inline uint enough(uint step, uint target) {
		return (target + step  - 1)/step;
	}

	static string expNotation(long val);
	static string stackString(string msg, int start, int depth = -1) ;
	static string stackString(int start = 0, int depth = -1);
	static string caller();
	static string caller2();
	static string callerN(int n);
	static void dumpStack(int start, int depth);
	static void dumpStack(int depth = -1);
	static void dumpStack(const char * msg,  int depth = -1) ;
	static void dumpStackIgnoreHere(int depth) ;
	static void dumpStackIgnoreHere(string msg, int depth = -1);
	static void waitEnter();

	static const char* lastErrStr();
	static void dumpAnyErr(string file, int line);
	static void dumpError(cudaError_t err);
	static double usedMemRatio();
	static void usedDmem();
	static void _checkCudaError(string file, int line, cudaError_t val);
	static void _printCudaError(string file, int line, cudaError_t val);

	static void announceTime();
	static void announceDuration(float exeTime);

	static void abortDumper(int level = -1);
	static void fpeDumper(int level = -1);
	static void segvDumper(int level = -1);
	static void handleSignals();
	template<typename T> static T getParameter(int argc, const char** argv, const char* parameterName, T defaultValue = 1);
	static int getCount(int argc, const char** argv,int defaultCount = 1);

	static time_t timeReps( void(*fn)(), int reps) ;

	static void randSequence(vector<uint>& ret, uint count, uint start = 0);
	template<typename T> static void toArray(vector<T>& v, T* arry, int start, int count);

};
#define anyErr()  b_util::dumpAnyErr(__FILE__,__LINE__)
#define checkCudaError(exp) b_util::_checkCudaError(__FILE__, __LINE__, exp)
#define printCudaError(exp) b_util::_printCudaError(__FILE__, __LINE__, exp)
#define dthrow(exp) { cout << "Exception" << endl << "\\/   at " << __FILE__ << "("<< __LINE__ << ")\n" << b_util::stackString(2) << endl; throw(exp);}
#define dassert(exp) { if(!exp) { cout << "assertion failed " << endl << "\\/   at " << __FILE__ << "("<< __LINE__ << ")\n" << b_util::stackString() << endl; assert(exp); exit(-1);}}

template<typename T> struct DMatrix;
template<typename T> struct Matrix;
template<typename T> struct Test;

template <typename T> struct util {
	static T minValue();
	static inline uint vol(dim3& d3) { return d3.x * d3.y *d3.z * sizeof(T); }
	static T maxValue();
	static bool almostEquals(T t1, T t2, T epsilon = 1e-6);
	static  int deleteMap( map<string, T*>&  m);
	static  int deleteVector( vector<T*>&  v);
	static  int cudaFreeVector( vector<T*>&  v, bool device = true);
	static  cudaError_t copyRange(T* targ, ulong targOff, T* src, ulong srcOff, ulong count);
	static  T sumCPU(T* vals, ulong count);
	static string pdm(const DMatrix<T>& md);
	static int release(std::map<std::string, Matrix<T>*>& map);
	static void parseDataLine(string line, T* elementData,
			unsigned int currRow, unsigned int rows, unsigned int cols,
			bool colMajor);
	static map<string, Matrix<T>*> parseOctaveDataFile(
			const char * path, bool colMajor, bool matrixOwnsBuffer = true);

	static T timeMain( int(*mainLike)(int, char**), const char* name, int argv, char** args, int* theResult ) ;
	static T timeReps( Matrix<T> (Matrix<T>::*unaryFn)(), const char* name, Matrix<T>* mPtr, int reps) ;
	static __device__ __host__ T epsilon();
	static string parry(int cnt, const T* arry);
};

struct IndexArray {
	uint * indices;
	uint count;
	bool owner;
	IndexArray();
	IndexArray(const IndexArray& o);
	IndexArray(uint* _indices, uint _count, bool _owner=true);
	IndexArray(uint idx1,uint idx2);
	string toString() const;
	uintPair toPair() const;
	~IndexArray();
	friend std::ostream& operator<<(std::ostream& os,  const IndexArray arry)  {
		return os << arry.toString().c_str();
	}
};

template <typename T> class Math {
public:
	static bool aboutEq(T x1, T x2, T epsilon = static_cast<T>( .0001));
};

template <typename T> struct SizedArray {
	T* vals;
	ulong count;
	SizedArray(T* vals, ulong count) {
		this->vals = vals; this->count = count;
	}
};

enum TimerStatus { ready, started };
class CuTimer {
private:
    cudaEvent_t evt_start, evt_stop;
    cudaStream_t stream;
    TimerStatus status;
public:
    CuTimer(cudaStream_t stream = 0);
    ~CuTimer();
    void start();
    float stop();
};
#endif /* UTIL_H_ */
