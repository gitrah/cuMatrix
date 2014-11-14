#include "debug.h"

__constant__ uint debugFlags;
uint hDebugFlags;

__managed__ CuMatrixException lastEx;

string gpuNames[MAX_GPUS];


__host__ __device__ inline void cherrp(cudaError exp) {
	if(exp != cudaSuccess) {
		printf( "%s : %d --> %s\n", __FILE__, __LINE__ , __cudaGetErrorEnum(exp));
		assert(0);
	}
}

void __host__ __device__ setLastError(CuMatrixException lastException) {
#ifndef __CUDA_ARCH__
	dthrow(CuMatrixExceptionStrings[lastException]);
#else
	lastEx= lastException;
#endif
	if(!checkDebug(debugSpoofSetLastError))assert(false);
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

void setAllGpuDebugFlags(uint flags, bool orThem, bool andThem ) {
	prlocf("setAllGpuDebugFlags entre...\n");
	int devCount, currDev;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	checkCudaErrors(cudaGetDevice(&currDev));

	for(int i = 0; i < devCount;i++) {

		if(strstr(gpuNames[i].c_str(), "750 Ti")) {
			prlocf("not skipping sluggish 750 ti\n");
			//continue;
		}
		flprintf("setting DbugFlags for device %s %d\n",gpuNames[i].c_str(),i);

		checkCudaErrors(cudaSetDevice(i));
		prlocf("set device\n");
		setCurrGpuDebugFlags(flags,orThem,andThem);
	}
	checkCudaErrors(cudaSetDevice(currDev));
}


void setCurrGpuDebugFlags(uint flags, bool orThem, bool andThem ) {

	uint curr = flags;
	if(orThem) {
		prlocf("copying DebugFlag fr device for or'n...\n");
		checkCudaErrors(cudaMemcpyFromSymbol(&curr, debugFlags,sizeof(uint)));
		curr |= flags;
	} else if(andThem) {
		prlocf("copying DebugFlag fr device fur and'n...\n");
		checkCudaErrors(cudaMemcpyFromSymbol(&curr, debugFlags,sizeof(uint)));
		curr &= flags;
	}
	prlocf("copying DebugFlag to device...\n");
	checkCudaErrors(cudaMemcpyToSymbol(debugFlags,&curr,sizeof(uint)));
	prlocf("copied to device\n");
	hDebugFlags = curr;
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
