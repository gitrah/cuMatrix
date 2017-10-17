/*
 * Kernels.cu
 *
 *  Created on: Oct 19, 2013
 *      Author: reid
 */
#include "Kernels.h"
#include "util.h"

extern __host__ __device__ void printLongSizes();

dim3 gDefaultMatProdBlock = dim3(16,16);

__device__ clock_t global_now;

// todo redundant to util<T> ::eps
template<> __host__ __device__ float Epsilon() {
	return 1e-6;
}
template<> __host__ __device__ double Epsilon() {
	return 1e-10;
}
template<> __host__ __device__ ulong Epsilon() {
	return 1;
}

__global__ void warmup() {
#if __CUDA_ARCH__ == 610
	prlocf("\n\nwarmup<<<>>> on cc 6.1 device\n\n\n");
#elif __CUDA_ARCH__ == 520
	prlocf("\n\nwarmup<<<>>> on cc 5.2 device\n\n\n");
#elif __CUDA_ARCH__ == 500
	prlocf("\n\nwarmup<<<>>> on cc 5 device\n\n\n");
#elif __CUDA_ARCH__ == 350
	prlocf("\n\nwarmup<<<>>> on cc 3.5 device\n\n\n");
#elif __CUDA_ARCH__ == 300
	prlocf("\n\nwarmup<<<>>> on cc 3 device\n\n\n");
#else
	prlocf("\n\nwarmup<<<>>> on cc UNKNOWN device\n\n\n");
#endif

	printLongSizes();
}

__global__ void slep(long slepMs) {
	clock_t start = clock();
	clock_t now;
	for (;;) {
	  now = clock();
	  clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
	  if (cycles >= slepMs) {
	    break;
	  }
	}
	// Stored "now" in global memory here to prevent the
	// compiler from optimizing away the entire loop.
	global_now = now;
}

template <typename T> __host__ CUDART_DEVICE void setL(T* elements, int m, int n, int p, int row, int col, T val) {
	if(row > m) { printf("rowOutOfBounds()\n"); return;}
	if(col > n) { printf("columnOutOfBounds()\n"); return; }
	setKernel<<<1,1>>>(elements, row, col, p, row * p + col, val);
}
template void setL<float>(float*, int, int, int, int, int, float);
template void setL<double>(double*, int, int, int, int, int, double);
template void setL<long>(long*, int, int, int, int, int, long);
template void setL<ulong>(ulong*, int, int, int, int, int, ulong);
template void setL<int>(int*, int, int, int, int, int, int);
template void setL<uint>(uint*, int, int, int, int, int, uint);

template <typename T> __global__ void setKernel(T* elements,  int m, int n, int p,  long l, T val) {
	if(n == p) {
		elements[l] = val;
	} else {
		// todo simplify this
		uint div = l /n;
		uint idx = div * p;
		idx += l - div * n;
		//printf("offset l %u -> %u\n", l, idx);
		elements[idx ] = val;
	}
}

template <typename T> __host__ CUDART_DEVICE void setL(T* elements, int m, int n, int p, long l, T val) {
	if(l > m * p) {printf("outOfBounds()\n"); return;}
	setKernel<<<1,1>>>(elements, m, n, p, l,val);
}

template void setL<float>(float*, int, int, int, long, float);
template void setL<double>(double*, int, int, int, long, double);
template void setL<long>(long*, int, int, int, long, long);
template void setL<ulong>(ulong*, int, int, int, long, ulong);
template void setL<int>(int*, int, int, int, long, int);
template void setL<uint>(uint*, int, int, int, long, uint);
// filluers
/*
template __global__ void fillOpKernel<float, stepFiller<float> >(stepFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, constFiller<float> >(constFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, sinFiller<float> >(sinFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, cosFiller<float> >(cosFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, randFiller<float> >(randFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, sequenceFiller<float> >(sequenceFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, seqModFiller<float> >(seqModFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, diagonalFiller<float> >(diagonalFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, increasingColumnsFiller<float> >(increasingColumnsFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, increasingRowsFiller<float> >(increasingRowsFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, sequenceScaleFiller<float> >(sequenceScaleFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpKernel<float, spanFiller<float> >(spanFiller<float>, float*, int, int, int, bool);

template __global__ void fillOpKernel<double, stepFiller<double> >(stepFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, constFiller<double> >(constFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, sinFiller<double> >(sinFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, cosFiller<double> >(cosFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, randFiller<double> >(randFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, sequenceFiller<double> >(sequenceFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, seqModFiller<double> >(seqModFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, diagonalFiller<double> >(diagonalFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, increasingColumnsFiller<double> >(increasingColumnsFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, increasingRowsFiller<double> >(increasingRowsFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, sequenceScaleFiller<double> >(sequenceScaleFiller<double>, double*, int, int, int, bool);
template __global__ void fillOpKernel<double, spanFiller<double> >(spanFiller<double>, double*, int, int, int, bool);
*/

/*
template<typename T> __global__ void fill_Kernel(
		T* trg,
		int height,
		int width,
		int pitch,
		bool colMajor)
{
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    uint indexOut = colMajor ? xIndex * pitch + yIndex : yIndex * pitch + xIndex;
    if(xIndex < width && yIndex < height)
    	trg[indexOut] = indexOut;
}
*/

template <typename T> __global__ void fillKernel(T* elements,  T val, long n) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < n) {
		elements[idx] = val;
	}
}
template __global__ void fillKernel<float>(float*, float, long);
template __global__ void fillKernel<double>(double*, double, long);
template __global__ void fillKernel<int>(int*, int, long);
template __global__ void fillKernel<uint>(uint*, uint, long);
template __global__ void fillKernel<long>(long*, long, long);
template __global__ void fillKernel<ulong>(ulong*, ulong, long);

// Non-Square Block version, to amortize index calcs
template<typename T, typename FillOp> __global__ void fillOpNsbKernel(
		FillOp op,
		T* trg,
		int height,
		int width,
		int pitch,
		bool colMajor)
{
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.x + threadIdx.y;
    uint indexOut = colMajor ? xIndex * pitch + yIndex : yIndex * pitch + xIndex;
    uint ip;
	if( xIndex < width )
		for(int i = 0; i < blockDim.x; i+= blockDim.y) {
			if( i + yIndex < height) {
				ip = i * pitch;
				trg[ip + indexOut] = op(indexOut + ip);
			}
		}
}
template __global__ void fillOpNsbKernel<float, oneOverFiller<float> >(oneOverFiller<float>, float*, int, int, int, bool);
template __global__ void fillOpNsbKernel<double, oneOverFiller<double> >(oneOverFiller<double>, double*, int, int, int, bool);

__global__ void setup_kernel ( curandState * state, int width, int pitch )
{
    int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = xIdx + yIdx * pitch;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init (1234, offset, 0, & state [ offset ]) ;
}



template __global__ void generate_kernel ( curandState * state,
                                      float*  result, int height, int width );
template __global__ void generate_kernel ( curandState * state,
                                      double*  result, int height, int width );
template <typename T> __global__ void generate_kernel ( curandState * state,
                                      T*  result, int height, int width )
{
	int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = xIdx + yIdx * width;
    if(xIdx < width && yIdx < height) {
    	/* Copy state to local memory for efficiency */
    	curandState localState = state [ offset ];
    	/* Generate pseudo - random unsigned ints */
    	result[offset]= static_cast<T>(curand (&localState) / RAND_MAX);
        /* Copy state back to global memory */
    	state [ offset ] = localState ;
    }
}
template __global__ void generate_uniform_kernel ( curandState * state,
		float*  result, float epsilon, int height, int width, int );
template  __global__ void generate_uniform_kernel ( curandState * state,
		double*  result, double epsilon, int height, int width, int );
template  __global__ void generate_uniform_kernel ( curandState * state,
		long*  result, long epsilon, int height, int width, int );
template  __global__ void generate_uniform_kernel ( curandState * state,
		ulong*  result, ulong epsilon, int height, int width, int );
template  __global__ void generate_uniform_kernel ( curandState * state,
		uint*  result, uint epsilon, int height, int width , int);
template  __global__ void generate_uniform_kernel ( curandState * state,
		int*  result, int epsilon, int height, int width, int  );

template <typename T>  __global__ void generate_uniform_kernel ( curandState * state,
		T*  result, T epsilon, int height, int width, int pitch )
{
 	int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = xIdx + yIdx * pitch;
    if(xIdx < width && yIdx < height) {
    	/* Copy state to local memory for efficiency */
    	curandState localState = state [ offset ];
    	/* Generate pseudo - random uniforms */
    	result[offset] = (2 * curand_uniform (& localState ) - 1) * epsilon;
    	/* Copy state back to global memory */
    	state [ offset ] = localState ;
    }
}

template __global__ void generate_uniform_kernel_mod ( curandState * state,
		float*  result, float epsilon, int height, int width, int , int );
template  __global__ void generate_uniform_kernel_mod ( curandState * state,
		double*  result, double epsilon, int height, int width, int , int );
template  __global__ void generate_uniform_kernel_mod ( curandState * state,
		long*  result, long epsilon, int height, int width, int, int  );
template  __global__ void generate_uniform_kernel_mod ( curandState * state,
		ulong*  result, ulong epsilon, int height, int width, int , int );
template  __global__ void generate_uniform_kernel_mod ( curandState * state,
		uint*  result, uint epsilon, int height, int width , int, int );
template  __global__ void generate_uniform_kernel_mod ( curandState * state,
		int*  result, int epsilon, int height, int width, int, int   );

template <typename T>  __global__ void generate_uniform_kernel_mod ( curandState * state,
		T*  result, T epsilon, int height, int width, int pitch, int mod )
{
 	int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = xIdx + yIdx * pitch;
    if(xIdx < width && yIdx < height) {
    	/* Copy state to local memory for efficiency */
    	curandState localState = state [ offset ];
    	/* Generate pseudo - random uniforms */
    	result[offset] = (T) ( (int)((2 * curand_uniform (& localState ) - 1) * epsilon) % mod);
    	/* Copy state back to global memory */
    	state [ offset ] = localState ;
    }
}
