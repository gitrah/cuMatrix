/*
 *
 *  Created on: Jul 23, 2012
 *      Author: reid
 */

#pragma once

#include <map>
#include <vector>
#include <list>
#include <stack>
#include <string>
#include <time.h>
#include <functional>
#include <sstream>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "debug.h"
#include "CuDefs.h"

#ifdef CuMatrix_NVML
	#include <nvml.h>
#endif

using std::pair;
using std::string;
using std::stringstream;
using std::list;
using std::map;
using std::vector;
using std::function;
using std::ostream;
using std::istream;
using std::ifstream;
using std::back_inserter;
using std::cout;

#define REGISTER_HEADROOM_FACTOR (0.87)
extern __host__ __device__ const char *__cudaGetErrorEnum(cudaError_t error);

#define cherr(exp) if((exp)!= cudaSuccess) {printf( "\n\n%s : %d --> %s\n\n", __FILE__, __LINE__ , __cudaGetErrorEnum(exp));assert(0);}
#define chnerr(exp) if((exp)!= NVML_SUCCESS) {printf( "\n\n%s : %d --> %s\n\n", __FILE__, __LINE__ , getNvmlErrorEnum(exp));assert(0);}

#define chblerr(exp) if((exp)!= CUBLAS_STATUS_SUCCESS) {printf( "\n\n%s : %d --> %s\n\n", __FILE__, __LINE__ , __cublasGetErrorEnum(exp));assert(0);}

const char* getNvmlErrorEnum(int en);

template <typename T> inline __host__ __device__ void printColoArray(const T* array, int n, int direction = 1);
template <typename T> inline __host__ __device__ void prtColoArrayDiag(const T* array,const char*msg, int line,  int pitch, int n, int direction = 1, T notEqual = 0);
template <typename T> inline __host__ __device__ void cntColoArrayDiag(const T* array,const char*msg, int line,  int pitch, int n, int direction = 1, T notEqual = 0);
template <typename T> inline __host__ __device__ void prtColoArrayInterval(const T* array, const char* msg, long n, int sampleElemCount, int sampleCount);
#define printColoArrayDiag( array, pitch, n ) prtColoArrayDiag( array, __PRETTY_FUNCTION__, __LINE__, pitch,  n )
#define printColoArrayDiagNe( array, pitch, n, notEq) prtColoArrayDiag( array, __PRETTY_FUNCTION__, __LINE__, pitch,  n , 1, notEq)
#define countColoArrayDiagNe( array, pitch, n, notEq) cntColoArrayDiag( array, __PRETTY_FUNCTION__, __LINE__, pitch,  n , 1, notEq)
#define printColoArrayInterval( array, n, sampleElemCount, sampleCount) prtColoArrayInterval( array, __PRETTY_FUNCTION__,   n, sampleElemCount, sampleCount)

#ifndef __CUDA_ARCH__
#define usedDevMem() {cout << __PRETTY_FUNCTION__ << "[" << __FILE__ << "::" << __LINE__ << "]\n"; b_util::usedDmem(0);}
#else
#define usedDevMem() {}
#endif
#define usedCurrMem() { printf("%s [%s :: %d] %.2f %%\n",__PRETTY_FUNCTION__ , __FILE__ ,__LINE__ , b_util::currMemRatio());}

template <typename T> __host__ __device__ void printDevArray(const T* array, const char*, int line, int n, int direction = 1, T test = (T) 0);
#define printArray( array, n) printDevArray( array, __PRETTY_FUNCTION__, __LINE__,   n,1)
#define printArrayNe( array, n, ne) printDevArray( array, __PRETTY_FUNCTION__, __LINE__,   n, 1, ne)

template <typename T> __host__ __device__ void printDevArrayDiag(
		const T* array, const char*, int line, int pitch, int n, int direction =1, T test = (T) 0);
#define printArrayDiag( array, p, n) printDevArrayDiag( array, __PRETTY_FUNCTION__, __LINE__,  p, n)
#define printArrayDiagNeq( array, p, n, dir, test) printDevArrayDiag( array, __PRETTY_FUNCTION__, __LINE__,  p, n, dir, test)
template <typename T> __host__ __device__ void prtDevArrayDiagSpan(
		const T* array, const char*, int line, int pitch, int n, int span = 1, int direction =1);
#define printDevArrayDiagSpan( array, p, n, span) prtDevArrayDiagSpan( array, __PRETTY_FUNCTION__, __LINE__,  p, n, span)


// aka ceil(a/b)
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))


#define TFMT util<T>::type_format()
//using namespace std;

// host-device modification status
enum Modification {
	mod_neither, mod_synced, mod_host, mod_device,
};


enum BuildType {
	btUnknown, debug, release,
};

enum TileDirection {
	tdNeither = 1, tdRows = 2, tdCols = 4, tdBoth = tdRows | tdCols,
};

enum Padding {
	padWarp, padEven
};


extern const char* GpuCriteriaNames[];
enum GpuCriteria { gcCoolest, gcFreeest, gcFastest };
GpuCriteria operator++(GpuCriteria& x);
GpuCriteria operator*(GpuCriteria c);
GpuCriteria begin(GpuCriteria r);
GpuCriteria end(GpuCriteria r);

__host__ __device__ const char* tdStr(TileDirection td);

struct IndexArray;
template<typename T> struct DMatrix;
template<typename T> class CuMatrix;
template<typename T> struct Test;


typedef unsigned int uint;
typedef pair<int,int> intPair;
typedef unsigned long ulong;
typedef pair<long,long> longPair;

template<typename T, typename U> string pp(pair<T,U>& p);
string print_stacktrace(unsigned int max_frames = 63);

extern string parry(const int* arry, int cnt);
template<typename T> __host__ __device__ T sqrt_p(T);

// printf with prepended thread id
void tprintf(const char *__restrict __format, ...);

struct b_util {

	static inline __host__ __device__ void printBinary(char* out, int v, int widthBits) {
		for(int i = 0; i < widthBits; i++) {
			out[widthBits - 1 - i] = (v >> i & 1) ? '1' : '0';
		}
		out[widthBits]=0;
	}
	static __host__ __device__ void printBinByte(char* out, char c) {
		printBinary(out,c,8);
	}
	static __host__ __device__ void printBinShort(char* out, short c) {
		printBinary(out,c,16);
	}
	static __host__ __device__ void printBinInt(char* out, int c) {
		printBinary(out,c,32);
	}
	static __host__ __device__ void printBinLong(char* out, long c) {
		printBinary(out,c,64);
	}



	static __device__ int sumRec(int s);
	static __host__ CUDART_DEVICE int countSpanrows( int m, int n, uint warpSize = WARP_SIZE);
	static __host__ __device__ bool spanrowQ( int row, int n, uint warpSize = WARP_SIZE);
	static __host__ void freeOnDevice(void * mem);

	static __host__ void warmupL();

	/*
	 * headroom- nax per matrix allocation as fraction of total device gmem
	 */

	static inline ulong twoDTo1D(dim3& idx3, dim3& basis) { return idx3.x + idx3.y * basis.x; }
	static inline ulong threeDTo1D(dim3& idx3, dim3& basis) { return idx3.x + idx3.y * basis.x + idx3.z * basis.x * basis.y; }
	static void to1D(dim3& d3);

	static int countLines(const char* path, int headerLength = 1);
	static vector<string> readLines(const char * );
	static vector<string> split(string);

	static __host__ __device__ const char * modStr(Modification lastMod);
	static __device__ __host__ void printModStr(Modification lastMod);

	static void deCap(char* s);
	static string deCap(const string& s);
	static string cap(const string& s);
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
	static string toStr(const intPair& p);
	static void syncGpu(const char * msg = null);
	__device__ __host__ static inline uint maskShifts(uint x) {
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return x;
	}
	__device__ __host__ static uint nextPowerOf2(uint x);
	__device__ __host__ static uint prevPowerOf2(uint x);

	template<typename T> static inline T nextFactorable(T start, uint factor) {
		return static_cast<T>( ceilf(start*1./factor)*factor);
	}
	static __host__ __device__ bool isPow2(uint x);
	static __host__ __device__ bool onGpuQ();

	static __host__ __device__ void vectorExecContext(int threads, int count, dim3& dBlocks,
			dim3& dThreads);
	static void vectorExecContext(int threads, uint count, uint spacePerThread,
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
	static __host__ __device__ void expNotation(char* buff, long val);
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
	static __host__ CUDART_DEVICE double currMemRatio();
	static __host__ CUDART_DEVICE double usedMemRatio(bool allDevices = false);
	static __host__ CUDART_DEVICE void usedDmem(bool allDevices = false);
	static __host__ __device__ void  _checkCudaError(const char* file, int line, cudaError_t val);

	static void _printCudaError(const char* file, int line, cudaError_t val);

	static void announceTime();
	static void announceDuration(float exeTime);
	static long nowMillis();

	//static int allDevices(void (*ftor)());
	static uint getDeviceTemp(int dev);

	static int getDeviceThatIs( GpuCriteria );

	static void dumpGpuStats();

	template<typename T> static bool onCurrDeviceQ(T& m);
	template<typename T> static int deviceOfResidence(T& m);
	template<typename T> static bool onDeviceQ(int dev, T& m);
	template<typename T, typename... Args> static bool onCurrDeviceQ(T& first, Args... args) {
		//printf("onCurrDeviceQ argslen %d\n", sizeof...(args));
		return  onCurrDeviceQ(first ) && onCurrDeviceQ(args...);
	}

	template<typename T, typename... Args> static bool onDeviceQ(int dev, T& first, Args... args) {
		return  onDeviceQ(dev, first ) && onDeviceQ(dev, args...);
	}

	template<typename T, typename... Args> static bool colocatedQ(T& first, Args... args) {
		static int dev = deviceOfResidence(first);
	    return onDeviceQ(dev, args...);
	}

	template<typename T> static int devByMaxOccupancy( map<int,long>& devOx, T& m);

	template<typename T, typename... Args> static int devByMaxOccupancy(map<int,long>& devOx, T& first, Args... args) {
		printf("b_util_devByMaxOccupancy argslen %d\n", sizeof...(args));
		int devCount = -1;
		cudaGetDeviceCount(&devCount);
		if(devOx.size() == 0) {
			printf("b_util_ devOx.size() == 0\n");
			for( int i = 0; i < devCount;i++) {
				devOx[i] = 0;
			}
		}
		if(sizeof...(args) == 0) {
			int maxGpu;
			long maxOx = 0;
			for( const auto &p : devOx) {
				printf("b_util_ testing dev %d has ox %ld\n",p.first, p.second);
				if(p.second > maxOx) {
					printf("b_util_ dev %d has beggist so far: ox %ld\n",p.first, p.second);
					maxOx = p.second;
					maxGpu = p.first;
				}
			}
			printf("b_util_ maxOx %d on gpiux %d\n",maxOx, maxGpu);
			return maxGpu;
	    } else {
	    	 devByMaxOccupancy(devOx, first );
	    			 return devByMaxOccupancy( devOx, args...);
	    }
	}

	template<typename T> static __host__ int migrate(int dev, T& m) {
		outln("m " << m.toShortString());
		return m.getMgr().migrate(dev,m);
		//return 0;
	}

	template<typename T, typename... Args> static __host__ int migrate(int dev, T& first, Args ... args) {
		printf("util_migrate argslen %d\n", sizeof...(args));

		if(sizeof...(args) == 0) {
			printf("util_migrate dev to %d\n",dev);
			return 0;
	    } else {
	    	return migrate(dev, first ) +
	    	migrate( dev, args...);
	    }
	}


	static int allDevices(function<void()>ftor);
	static bool anyDevice(function<bool()>ftor);

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

	static void randSequence(vector<int>& ret, int count, int start = 0);
	static void randSequenceIA(IndexArray& ret, int count, int start = 0);
	template<typename T> static void toArray( T* arry, const vector<T>& v, int start, int count);
	static void intersects(int** indices, const float2* segments);
	static void intersects(int** indices, const double2* segments);
	static __device__ __forceinline__ void laneid(uint&  lid) {
		asm("{\n\tmov.u32 %0, %%laneid ;\n\t}" : "=r"(lid));
	}

	static int kernelOccupancy( void * kernel, int* maxBlocks, int blockSize);

	static __host__ __device__ bool adjustExpectations(dim3& grid, dim3& block, const cudaFuncAttributes& atts);

	static list<uint> rangeL( uint2 range , int step =1);
	static list<list<uint>> rangeLL( uint2* ranges, int count, int step = 1 );

	template <typename E> static void print(list<E> l);
	template <typename E> static string toString(list<E> l);
	template <typename E> static void printAll(list<list<E>> listOfLists, list<E> sol);
	template <typename E> static int countAll(list<list<E>> listOfLists, list<E> sol);

};
#define anyErr()  b_util::dumpAnyErr(__FILE__,__LINE__)
#define checkCudaError(exp) b_util::_checkCudaError(__FILE__, __LINE__, exp)
#define printCudaError(exp) b_util::_printCudaError(__FILE__, __LINE__, exp)
#define dthrow(exp) { cout << "Exception" << endl << "\\/   at " << __FILE__ << "("<< __LINE__ << "):  " << exp << "\n" << b_util::stackString(2) << endl; throw(exp);}
#define dthrow2(exp) { cout << "Exception" << endl << "\\/   at " << __FILE__ << "("<< __LINE__ << "):  " << exp << "\n" << print_stacktrace(50) << endl; throw(exp);}
#define dthrowm(msg,exp) { cout << "Exception: " << msg << endl << "\\/   at " << __FILE__ << "("<< __LINE__ << ")\n" << b_util::stackString(2) << endl; throw(exp);}
#define dassert(exp) { if(!(exp)) { cout << "assertion failed " << endl << "\\/   at " << __FILE__ << "("<< __LINE__ << ")\n" << b_util::stackString() << endl; assert(exp); exit(-1);}}

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

	static bool onDevice(T* ptr, int dev);
	static bool onCurrentDevice(T* ptr);
	static int getDevice(T* ptr);

	static __host__ __device__ void prdm(const char* msg, const DMatrix<T>& md);
	static __host__ __device__ void prdm( const DMatrix<T>& md) { prdm("",md);}
	static __host__ __device__ void prdmln(const char* msg, const DMatrix<T>& md);
	static __host__ __device__ void prdmln( const DMatrix<T>& md) { prdmln("",md);}
	static __host__ __device__ void printDm( const DMatrix<T>& dm, const char* msg = null);
	static __host__ __device__ void printRow(const DMatrix<T>& dm, int row = 0);

	static void setNDev(T* trg, T val, long n) {
		switch (sizeof(T)) {
		case 4: {

			char c1 = *(char*) &val;
			char c2 = *((char*) &val + 1);
			char c3 = *((char*) &val + 2);
			char c4 = *((char*) &val + 3);

			checkCudaErrors(cudaMemset2D((char* )trg, 4, c1, 1, n));
			checkCudaErrors(cudaMemset2D((char* )trg + 1, 4, c2, 1, n));
			checkCudaErrors(cudaMemset2D((char* )trg + 2, 4, c3, 1, n));
			checkCudaErrors(cudaMemset2D((char* )trg + 3, 4, c4, 1, n));
			break;
		}
		case 8: {
			char c1 = *(char*) &val;
			char c2 = *((char*) &val + 1);
			char c3 = *((char*) &val + 2);
			char c4 = *((char*) &val + 3);
			char c5 = *((char*) &val + 4);
			char c6 = *((char*) &val + 5);
			char c7 = *((char*) &val + 6);
			char c8 = *((char*) &val + 7);

			checkCudaErrors(cudaMemset2D((char* )trg + 0, 8, c1, 1, n));
			checkCudaErrors(cudaMemset2D((char* )trg + 1, 8, c2, 1, n));
			checkCudaErrors(cudaMemset2D((char* )trg + 2, 8, c3, 1, n));
			checkCudaErrors(cudaMemset2D((char* )trg + 3, 8, c4, 1, n));
			checkCudaErrors(cudaMemset2D((char* )trg + 4, 8, c5, 1, n));
			checkCudaErrors(cudaMemset2D((char* )trg + 5, 8, c6, 1, n));
			checkCudaErrors(cudaMemset2D((char* )trg + 6, 8, c7, 1, n));
			checkCudaErrors(cudaMemset2D((char* )trg + 7, 8, c8, 1, n));
		}
		}
	}
	static void setNH( T* trg, T val, long n) {
		switch(sizeof(T)) {
		case 4:
		{
			char c1 = * (char*) &val;
			char c2 = *  ((char*) &val + 1);
			char c3 = *  ((char*) &val + 2);
			char c4 = *  ((char*) &val + 3);

			checkCudaErrors(cudaMemset2D((char*)trg, 4, c1, 1, n));
			checkCudaErrors(cudaMemset2D((char*)trg + 1, 4, c2,1, n));
			checkCudaErrors(cudaMemset2D((char*)trg + 2, 4, c3,1, n));
			checkCudaErrors(cudaMemset2D((char*)trg + 3, 4, c4,1, n));
			break;
		}
		case 8:
		{
			char c1 = * (char*) &val;
			char c2 = *  ((char*) &val + 1);
			char c3 = *  ((char*) &val + 2);
			char c4 = *  ((char*) &val + 3);
			char c5 = *  ((char*) &val + 4);
			char c6 = *  ((char*) &val + 5);
			char c7 = *  ((char*) &val + 6);
			char c8 = *  ((char*) &val + 7);

			checkCudaErrors(cudaMemset2D((char*)trg + 0, 8,	c1,1, n));
			checkCudaErrors(cudaMemset2D((char*)trg + 1, 8, c2,1, n));
			checkCudaErrors(cudaMemset2D((char*)trg + 2, 8, c3,1, n));
			checkCudaErrors(cudaMemset2D((char*)trg + 3, 8, c4,1, n));
			checkCudaErrors(cudaMemset2D((char*)trg + 4, 8, c5,1, n));
			checkCudaErrors(cudaMemset2D((char*)trg + 5, 8, c6,1, n));
			checkCudaErrors(cudaMemset2D((char*)trg + 6, 8, c7,1, n));
			checkCudaErrors(cudaMemset2D((char*)trg + 7, 8, c8,1, n));
		}
		}
	}

	static void setNI( T* trg, T val, long n) {
		switch(sizeof(T)) {
			case 4: {
				checkCudaErrors(cudaMemset(trg, (int) val, n * sizeof(T)));
				break;
			}
			case 8: {
				int i1 = * (int*) &val;
				int i2 = *  ((int*) &val + 1);
				//int height = n/2;
				checkCudaErrors(cudaMemset2D((int*)trg + 0, 8, i1, 1, n));
				checkCudaErrors(cudaMemset2D((int*)trg + 1, 8, i2, 1, n));
			}
		}
	}

	static void setNI2( T* trg, T val, long n) {
		switch(sizeof(T)) {
			case 4: {
				checkCudaErrors(cudaMemset(trg, (int) val, n * sizeof(T)));
				break;
			}
			case 8: {
				cudaStream_t streams[2];

				cherr(cudaStreamCreateWithFlags(&streams[0],cudaStreamNonBlocking));
				cherr(cudaStreamCreateWithFlags(&streams[1],cudaStreamNonBlocking));
				int* pint = (int*) &val;
				int i1 = *pint++;
//extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0));

				int i2 = *pint;
				int* pitarg = (int*) trg;
				printf("pitarg %p \n", pitarg);
				printf("pitarg++ %p \n", pitarg+1);
				printf("pitarg++ %p \n", pitarg+1);
				printf("pitarg++ %p \n", pitarg+1);

//extern __host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
				checkCudaErrors(cudaMemset2DAsync(pitarg, 8, 	i1, 1, n, streams[0]));
				checkCudaErrors(cudaMemset2DAsync(pitarg + 	1, 	8, i2, 1, n, streams[1]));

				cherr(cudaStreamDestroy(streams[0]));
				cherr(cudaStreamDestroy(streams[1]));

			}
		}
	}

	static void fillNDev(T* trg, T val, long n);

	static T timeMain( int(*mainLike)(int, char**), const char* name, int argv, char** args, int* theResult ) ;
	static T timeReps( CuMatrix<T> (CuMatrix<T>::*unaryFn)(), const char* name, CuMatrix<T>* mPtr, int reps) ;
	static __device__ __host__ T epsilon();
	static inline T relDiff(T v1, T v2) {
		return (v2-v1)/(1.0*v1);
	}
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

	__host__ __device__ void colMajorCopy( T* dest, const T* srcInRowMajor, int m, int n ) {
		int len = m*n;
		for(int i =0; i < len; i++){
			int colIdx = i % n * m + i / n;
			dest[i] = srcInRowMajor[colIdx];
		}
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
	intPair toPair() const;
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
class CuTimer  {
protected:
    cudaEvent_t evt_start, evt_stop;
    cudaStream_t stream;
    TimerStatus status;
    int 	device;
public:
    CuTimer(cudaStream_t stream = 0);
    ~CuTimer();
    void start();
    float stop(); // in millis
};

class CuMethodTimer : public CuTimer {
private:
	union {
		void (*funcPtr)();
		const char* funcName;
	};
	int count;
	float total;
public:
    void summary();
    CuMethodTimer(void (*func)(), cudaStream_t stream = 0);
    CuMethodTimer(const char*name, cudaStream_t stream = 0);
    void enter();
    void exit();
    ~CuMethodTimer();
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
