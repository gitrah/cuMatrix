#include "debug.h"
#include "CuFunctor.h"
#include "CuMatrix.h"
__constant__ uint debugFlags;
uint hDebugFlags;

__device__ CuMatrixException lastEx;

string gpuNames[MAX_GPUS];
char cuScratch[ 8 ];
int* cuIntPtr = (int*) &cuScratch;
float* cuFltPtr = (float*) &cuScratch;

__host__ __device__ inline void cherrp(cudaError exp) {
	if(exp != cudaSuccess) {
		printf( "%s : %d --> %s\n", __FILE__, __LINE__ , __cudaGetErrorEnum(exp));
		assert(0);
	}
}

__host__ __device__ void setLastError(CuMatrixException lastException) {
	flprintf("setLastError called with %d\n", lastException);

#ifndef __CUDA_ARCH__
	dthrow(CuMatrixExceptionStrings[lastException]);
#else
	flprintf("changng lastEx from %d to %d\n", lastEx, lastException);
	lastEx= lastException;
#endif
	if(!checkDebug(debugSpoofSetLastError))assert(false);
}

__host__ __device__ CuMatrixException getLastError() {
#ifndef __CUDA_ARCH__
	return successEx;
#else
	flprintf("gsetLastError called, lastEx == %d\n", lastEx);
	return lastEx;
#endif
}

__host__ __device__ void printLongSizes() {
#ifndef __CUDA_ARCH__
	printf("host sizes\n");
#else
	printf("dev sizes\n");
#endif
	flprintf("\tsizeof(long double) %lu\n" , sizeof(long double));
	flprintf("\tsizeof(long long) %lu\n" , sizeof(long long));
}

template<typename T> void printObjSizes() {
	flprintf("\tsizeof(CuMatrix<T>) %lu\n" , sizeof(CuMatrix<T>));
	flprintf("\tsizeof(DMatrix<T>) %lu\n" , sizeof(DMatrix<T>));

	flprintf("\tsizeof(CuFunctor<T,1>) %lu\n" , sizeof(CuFunctor<T,1>));
	flprintf("\tsizeof(CuFunctor<T,2>) %lu\n" , sizeof(CuFunctor<T,2>));
	flprintf("\tsizeof(CuFunctor<T,3>) %lu\n" , sizeof(CuFunctor<T,3>));
	flprintf("\tsizeof(BinaryOpF<T,0>) %lu\n" , sizeof(BinaryOpF<T,0>));
	flprintf("\tsizeof(BinaryOpF<T,1v>) %lu\n" , sizeof(BinaryOpF<T,1>));
}
template void printObjSizes<double>();
template void printObjSizes<float>();
template void printObjSizes<long>();
template void printObjSizes<ulong>();
template void printObjSizes<int>();
template void printObjSizes<uint>();

void setAllGpuDebugFlags(uint flags, bool orThem, bool andThem ) {
	prlocf("setAllGpuDebugFlags entre...\n");
	int devCount, currDev;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	checkCudaErrors(cudaGetDevice(&currDev));
	flprintf("device count %d\n",devCount);
	flprintf("curr device %d\n",currDev);

	cudaStream_t *streams = (cudaStream_t *) malloc(
			devCount * sizeof(cudaStream_t));

	for(int i = 0; i < devCount;i++) {

		if(strstr(gpuNames[i].c_str(), "750 Ti")) {
			prlocf("not skipping sluggish 750 ti\n");
			//continue;
		}
		flprintf("setting DbugFlags for device %s %d\n",gpuNames[i].c_str(),i);

		ExecCaps_setDevice(i);
		flprintf("set device %d\n",i);
		checkCudaErrors(cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking));
		prlocf("create stream\n");
		setCurrGpuDebugFlags(flags,orThem,andThem, streams[i]);
		prlocf("set gpu dbg flags\n");
	}

	for(int i = 0; i < devCount; i++) {
		flprintf("synching stream for dev %d\n",i);
		checkCudaErrors(cudaStreamSynchronize(streams[i]));
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	}

	ExecCaps_setDevice(currDev);
}


void setCurrGpuDebugFlags(uint flags, bool orThem, bool andThem,  cudaStream_t stream ) {

	uint curr = flags;
	if(orThem) {
	//	outln( b_util::caller() << " copying DebugFlag fr device for or'n...");
		checkCudaErrors(cudaMemcpyFromSymbol(&curr, debugFlags,sizeof(uint)));
		curr |= flags;
	} else if(andThem) {
//		outln(b_util::caller() << " copying DebugFlag fr device fur and'n...");
		checkCudaErrors(cudaMemcpyFromSymbol(&curr, debugFlags,sizeof(uint)));
		curr &= flags;
	}
	hDebugFlags = curr;
//	outln("copying to device");
	checkCudaErrors(cudaMemcpyToSymbolAsync(debugFlags,&curr,sizeof(uint),0,  cudaMemcpyHostToDevice, stream));

	char buff[33];
    buff[32]=0;
    b_util::printBinInt(buff, hDebugFlags);

	outln("hDebugFlags bin str " << buff);

	flprintf("copied flags %s to device\n", dbgStr().c_str());
}

__host__ __device__ ostreamlike::ostreamlike() {}
__host__ __device__ ostreamlike::~ostreamlike() {}
__host__ __device__ ostreamlike& ostreamlike::write(int n) {
	//printf("int n\n");
	printf("%d", n);
	    return *this;
  }
__host__ __device__  ostreamlike& ostreamlike::write(char* n) {
	//printf("char* n\n");
    printf("%s",n);
    return *this;
}
__host__ __device__  ostreamlike& ostreamlike::write(const char* n) {
	//printf("const char* n\n");
    printf("%s",n);
    return *this;
}
__host__ __device__  ostreamlike& ostreamlike::write(char c) {
	//printf("char c\n");
    printf("%c", c);
    return *this;
}
__host__ ostreamlike& ostreamlike::write(const string& s) {
	//printf("const string& s\n");
	printf("%s", s.c_str());
	return *this;
}
__host__ __device__ ostreamlike& ostreamlike::write(float n) {
	//printf("float n\n");
	printf("%f", n);
	return *this;
}

__host__ __device__ ostreamlike& ostreamlike::write(double n) {
	//printf("double n\n");
	printf("%f", n);
	return *this;
}

__host__ __device__ ostreamlike& ostreamlike::write(long n) {
	//printf("long n\n");
	printf("%ld", n);
	return *this;
}

__host__ __device__ ostreamlike& ostreamlike::write(unsigned int n) {
	//printf("unsigned int n\n");
	printf("%u", n);
	return *this;
}

__host__ __device__ ostreamlike& ostreamlike::write(unsigned long n) {
	//printf("unsigned long n\n");
	printf("%lu ", n);
	return *this;
}

__host__ __device__ ostreamlike& ostreamlike::write(bool n) {
	//printf("bool n\n");
	printf("%s ", n ? "true" : "false");
	return *this;
}

__host__ __device__ ostreamlike& ostreamlike::write(const float* p) {
	if(p)
		printf("%p",  p);
	else
		printf("null");
	return *this;
}

__host__ __device__ ostreamlike& ostreamlike::write(const double* p) {
	if(p)
		printf("%p",  p);
	else
		printf("null");
	return *this;
}

__host__ __device__ ostreamlike& ostreamlike::write(const int* p) {
	if(p)
		printf("%p",  p);
	else
		printf("null");
	return *this;
}

__host__ __device__ ostreamlike& ostreamlike::write(const void* p) {
	if(p)
		printf("%p",  p);
	else
		printf("null");
	return *this;
}

__host__ __device__ ostreamlike& ostreamlike::write(const long* p) {
	if(p)
		printf("%p",  p);
	else
		printf("null");
	return *this;
}

void cherr_(cudaError_t err, char* file, int line) {
	if(err != cudaSuccess) {
		printf( "%s : %d --> %s\n", file, line, __cudaGetErrorEnum(err));
		assert(0);
	}
}
