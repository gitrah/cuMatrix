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
#include <sstream>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "debug.h"
#include "CuDefs.h"

using namespace std;

#define DevSync checkCudaError(cudaDeviceSynchronize())

#define REGISTER_HEADROOM_FACTOR .87

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

// host-device modification status
enum Modification {
	mod_neither, mod_synced, mod_host, mod_device,
};

enum BuildType {
	btUnknown, debug, release,
};

struct IndexArray;

typedef unsigned int uint;
typedef pair<uint,uint> uintPair;
typedef unsigned long ulong;
typedef pair<ulong,ulong> ulongPair;

template<typename T, typename U> string pp(pair<T,U>& p);
string print_stacktrace(unsigned int max_frames = 63);

extern string parry(const int* arry, int cnt);
template<typename T> __host__ __device__ T sqrt_p(T);

struct b_util {

	static __device__ int sumRec(int s);
	static __host__ CUDART_DEVICE int countSpanrows( uint m, uint n, uint warpSize = WARP_SIZE);
	static __host__ __device__ bool spanrowQ( uint row, uint n, uint warpSize = WARP_SIZE);
	static __host__ void freeOnDevice(void * mem);
	static int currDevice();

	static inline ulong twoDTo1D(dim3& idx3, dim3& basis) { return idx3.x + idx3.y * basis.x; }
	static inline ulong threeDTo1D(dim3& idx3, dim3& basis) { return idx3.x + idx3.y * basis.x + idx3.z * basis.x * basis.y; }
	static void to1D(dim3& d3);

	static int countLines(const char* path, int headerLength = 1);
	static vector<string> readLines(const char * );
	static vector<string> split(string);

	static __host__ __device__ const char * modStr(Modification lastMod);
	static __device__ __host__ void printModStr(Modification lastMod);

	static double diffclock(clock_t clock1,clock_t clock2);
	template<typename T> static string pvec(vector<T>& v);
	static string pxd(const dim3& grid,const  dim3& block);
	static string sPtrAtts(const cudaPointerAttributes& atts);
	template<typename T> static __host__ __device__ void pPtrAtts(T * ptr);
	template<typename T> static __host__ __device__ void pFuncPtrAtts(T * ptr);
	static string pexec(const dim3& grid,const  dim3& block, uint smem);
	static __host__ CUDART_DEVICE bool validLaunchQ( void* pKernel, dim3 grid, dim3 block);
	static string pd3(const dim3& d3);
	static __host__ __device__ void prd3( const dim3& d3,const char *msg = null);
	static string pd3(const dim3* d3);
	static string pd3(const dim3* d3, const char* msg);
	static string pd3(const dim3* d3, string msg);
	static string toStr(const uintPair& p);
	static void syncGpu(const char * msg = null);
	__device__ __host__ static uint nextPowerOf2(uint x);
	template<typename T> static inline T nextFactorable(T start, uint factor) {
		return static_cast<T>( ceilf(start*1./factor)*factor);
	}
	static __host__ __device__ bool isPow2(uint x);
	static __host__ __device__ bool onGpuQ();
	static __host__ __device__ void execContext(uint threads, uint count, dim3& dBlocks,
			dim3& dThreads);
	static void execContext(uint threads, uint count, uint spacePerThread,
			dim3& dBlocks, dim3& dThreads, uint& smem);
	static inline uint vol(dim3& d3) { return d3.x * d3.y *d3.z; }

	inline static string plz(const char* s, int c){

		stringstream ss;

		ss << s;

		if(c>1) {
			ss << "s";
		}

		return (ss.str());
	}

	static string expNotation(long val);
	static string unmangl(const char* mangl);
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
	static __host__ __device__ void  _checkCudaError(const char* file, int line, cudaError_t val);

	static void _printCudaError(const char* file, int line, cudaError_t val);

	static void announceTime();
	static void announceDuration(float exeTime);
	static long nowMillis();

	static void abortDumper(int level = -1);
	static void fpeDumper(int level = -1);
	static void segvDumper(int level = -1);
	static void handleSignals();
	template<typename T> static T getParameter(int argc, const char** argv, const char* parameterName, T defaultValue = 1);
	static int getCount(int argc, const char** argv,int defaultCount = 1);
	static int getIntArg(int argc, const char** argv,const char* argName, int defaultVal = 1);
	static int getStart(int argc, const char** argv,int defaultStart = 1);
	static string getPath(int argc, const char** argv,const char* defaultPath);
	static time_t timeReps( void(*fn)(), int reps) ;

	static void randSequence(vector<uint>& ret, uint count, uint start = 0);
	static void randSequenceIA(IndexArray& ret, uint count, uint start = 0);
	template<typename T> static void toArray( T* arry, const vector<T>& v, int start, int count);
	static void intersects(int** indices, const float2* segments);
	static void intersects(int** indices, const double2* segments);
	static __device__ __forceinline__ void laneid(uint&  lid) {
		asm("{\n\tmov.u32 %0, %%laneid ;\n\t}" : "=r"(lid));
	}

	static int kernelOccupancy( void * kernel, int* maxBlocks, int blockSize);
};
#define anyErr()  b_util::dumpAnyErr(__FILE__,__LINE__)
#define checkCudaError(exp) b_util::_checkCudaError(__FILE__, __LINE__, exp)
#define printCudaError(exp) b_util::_printCudaError(__FILE__, __LINE__, exp)
#define dthrow(exp) { cout << "Exception" << endl << "\\/   at " << __FILE__ << "("<< __LINE__ << "):  " << exp << "\n" << b_util::stackString(2) << endl; throw(exp);}
#define dthrow2(exp) { cout << "Exception" << endl << "\\/   at " << __FILE__ << "("<< __LINE__ << "):  " << exp << "\n" << print_stacktrace(50) << endl; throw(exp);}
#define dthrowm(msg,exp) { cout << "Exception: " << msg << endl << "\\/   at " << __FILE__ << "("<< __LINE__ << ")\n" << b_util::stackString(2) << endl; throw(exp);}
#define dassert(exp) { if(!(exp)) { cout << "assertion failed " << endl << "\\/   at " << __FILE__ << "("<< __LINE__ << ")\n" << b_util::stackString() << endl; assert(exp); exit(-1);}}

template<typename T> struct DMatrix;
template<typename T> class CuMatrix;
template<typename T> struct Test;

template <typename T> struct util {
	static __host__ __device__ T minValue();
	static __host__ float vAddGflops(int device);
	static inline __host__ __device__ uint vol(dim3& d3) { return d3.x * d3.y *d3.z * sizeof(T); }
	static __host__ __device__ T maxValue();
	static bool almostEquals(T t1, T t2, T epsilon = 1e-6);
	template <typename K> static  int deletePtrMap( map<K, T*>&  m);
	template <typename K> static void deletePtrArray( K**  m, int size);
	template <typename K> static void deleteDevPtrArray( K**  m, int size);
	static  int deleteVector( vector<T*>&  v);
	static  int cudaFreeVector( vector<T*>&  v, bool device = true);
	static  cudaError_t copyRange(T* targ, ulong targOff, T* src, ulong srcOff, ulong count);
	static  T sumCPU(T* vals, ulong count);
	static string pdm(const DMatrix<T>& md);
	static __host__ __device__ void prdm(const DMatrix<T>& md);
	static __host__ __device__ void printDm( const DMatrix<T>& dm, const char* msg = null);
	static __host__ __device__ void printRow(const DMatrix<T>& dm, uint row = 0);

	static int release(std::map<std::string, CuMatrix<T>*>& map);
	static cudaError_t migrate(int dev, CuMatrix<T>& m, ...);
	static void parseDataLine(string line, T* elementData,
			unsigned int currRow, unsigned int rows, unsigned int cols,
			bool colMajor);
	static void parseCsvDataLine(const CuMatrix<T>* x, int currLine, string line, const char* sepChars);
	static map<string, CuMatrix<T>*> parseOctaveDataFile(
			const char * path, bool colMajor, bool matrixOwnsBuffer = true);
	static map<string, CuMatrix<T>*> parseCsvDataFile(
			const char * path, const char * sepChars, bool colMajor, bool matrixOwnsBuffer = true, bool hasXandY = false);

	static T timeMain( int(*mainLike)(int, char**), const char* name, int argv, char** args, int* theResult ) ;
	static T timeReps( CuMatrix<T> (CuMatrix<T>::*unaryFn)(), const char* name, CuMatrix<T>* mPtr, int reps) ;
	static __device__ __host__ T epsilon();
	static string parry(const T* arry, int cnt);
	static __host__ __device__ void pDarry(const T* arry, int cnt);

	inline static __host__ __device__ T dot(const T* u, const T* v) {
		return u[0] * v[0] + u[1] * v[1] + u[2]* v[2];
	}
	inline static __host__ __device__ T dot(T u1, T u2, T u3, T v1, T v2, T v3) {
		return u1 * v1 + u2 * v2 + u3 * v3;
	}
	inline static __host__ __device__ void cross3(T* out, const T* u, const T* v) {
		out[0] = v[2] * u[1] - u[2] * v[1];
		out[1] = u[2] * v[0] - u[0] * v[2];
		out[3] = u[0] * v[1] - u[1] * v[0];
	}
	inline static __host__ __device__ void cross3(T& out1, T& out2, T& out3, T u1, T u2, T u3, T v1, T v2, T v3) {
		out1 = v3 * u2 - u3 * v2;
		out2 = u3 * v1 - u1 * v3;
		out3 = u1 * v2 - u2 * v1;
	}
};

struct IndexArray {
	uint * indices;
	uint count;
	bool owner;
	__host__ __device__ IndexArray();
	__host__ __device__ IndexArray(const IndexArray& o);
	__host__ __device__ IndexArray(uint* _indices, uint _count, bool _owner=true);
	__host__ __device__ IndexArray(uint idx1,uint idx2);
	string toString(bool lf = true) const;
	uintPair toPair() const;
	__host__ __device__ ~IndexArray();
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
    float stop(); // in millis
};

#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)

  #include <windows.h>

  inline void delay( unsigned long ms )
    {
    Sleep( ms );
    }

#else  /* presume POSIX */

  #include <unistd.h>

  inline void delayMics( unsigned long us )
    {
    usleep( us );
    }
  inline void delayMillis( unsigned long ms )
    {
    usleep( ms * 1000 );
    }

#endif

template<typename T> class toT {
public:
	static T fromUint(void* elem) {
		uint val = *((uint*) elem);
		return (T) val;
	}
};



#endif /* UTIL_H_ */
