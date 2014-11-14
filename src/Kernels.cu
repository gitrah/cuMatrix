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
#if __CUDA_ARCH__ == 500
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


template <typename T> __host__ CUDART_DEVICE void set(T* elements, uint m, uint n, uint p, uint row, uint col, T val) {
	if(row > m) { printf("rowOutOfBounds()\n"); return;}
	if(col > n) {printf("columnOutOfBounds()\n"); return; }
	setKernel<<<1,1>>>(elements,row,col,p, row * p + col, val);
}
template void set<float>(float*, uint, uint, uint, uint, uint, float);
template void set<double>(double*, uint, uint, uint, uint, uint, double);
template void set<ulong>(ulong*, uint, uint, uint, uint, uint, ulong);
template void set<int>(int*, uint, uint, uint, uint, uint, int);
template void set<uint>(uint*, uint, uint, uint, uint, uint, uint);
//template void set<uint>(uint*, unsigned int, unsigned int, unsigned int, unsigned long, uint);

template <typename T> __global__ void setKernel(T* elements,  uint m, uint n, uint p,  ulong l, T val) {
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

template <typename T> __host__ CUDART_DEVICE void set(T* elements, uint m, uint n, uint p, ulong l, T val) {
	if(l > m * p) {printf("outOfBounds()\n"); return;}
	setKernel<<<1,1>>>(elements, m, n, p, l,val);
}

template void set<float>(float*, uint, uint, uint, ulong, float);
template void set<double>(double*, uint, uint, uint, ulong, double);
template void set<ulong>(ulong*, uint, uint, uint, ulong, ulong);
template void set<int>(int*, uint, uint, uint, ulong, int);
template void set<uint>(uint*, uint, uint, uint, ulong, uint);
// filluers
/*
template __global__ void fillOpKernel<float, stepFiller<float> >(stepFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, constFiller<float> >(constFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, sinFiller<float> >(sinFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, cosFiller<float> >(cosFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, randFiller<float> >(randFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, sequenceFiller<float> >(sequenceFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, seqModFiller<float> >(seqModFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, diagonalFiller<float> >(diagonalFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, increasingColumnsFiller<float> >(increasingColumnsFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, increasingRowsFiller<float> >(increasingRowsFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, sequenceScaleFiller<float> >(sequenceScaleFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<float, spanFiller<float> >(spanFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);

template __global__ void fillOpKernel<double, stepFiller<double> >(stepFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, constFiller<double> >(constFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, sinFiller<double> >(sinFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, cosFiller<double> >(cosFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, randFiller<double> >(randFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, sequenceFiller<double> >(sequenceFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, seqModFiller<double> >(seqModFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, diagonalFiller<double> >(diagonalFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, increasingColumnsFiller<double> >(increasingColumnsFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, increasingRowsFiller<double> >(increasingRowsFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, sequenceScaleFiller<double> >(sequenceScaleFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpKernel<double, spanFiller<double> >(spanFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);
*/

/*
template<typename T> __global__ void fill_Kernel(
		T* trg,
		uint height,
		uint width,
		uint pitch,
		bool colMajor)
{
    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    uint indexOut = colMajor ? xIndex * pitch + yIndex : yIndex * pitch + xIndex;
    if(xIndex < width && yIndex < height)
    	trg[indexOut] = indexOut;
}
*/

// Non-Square Block version, to amortize index calcs
template<typename T, typename FillOp> __global__ void fillOpNsbKernel(
		FillOp op,
		T* trg,
		uint height,
		uint width,
		uint pitch,
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
template __global__ void fillOpNsbKernel<float, oneOverFiller<float> >(oneOverFiller<float>, float*, unsigned int, unsigned int, unsigned int, bool);
template __global__ void fillOpNsbKernel<double, oneOverFiller<double> >(oneOverFiller<double>, double*, unsigned int, unsigned int, unsigned int, bool);

__global__ void setup_kernel ( curandState * state, int width )
{
    int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = xIdx + yIdx * width;
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
		float*  result, float epsilon, int height, int width );
template  __global__ void generate_uniform_kernel ( curandState * state,
		double*  result, double epsilon, int height, int width );
template  __global__ void generate_uniform_kernel ( curandState * state,
		ulong*  result, ulong epsilon, int height, int width );
template  __global__ void generate_uniform_kernel ( curandState * state,
		uint*  result, uint epsilon, int height, int width );
template  __global__ void generate_uniform_kernel ( curandState * state,
		int*  result, int epsilon, int height, int width );

template <typename T>  __global__ void generate_uniform_kernel ( curandState * state,
		T*  result, T epsilon, int height, int width )
{
	int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = xIdx + yIdx * width;
    if(xIdx < width && yIdx < height) {
    	/* Copy state to local memory for efficiency */
    	curandState localState = state [ offset ];
    	/* Generate pseudo - random uniforms */
    	result[offset] = (2 * curand_uniform (& localState ) - 1) * epsilon;
    	/* Copy state back to global memory */
    	state [ offset ] = localState ;
    }
}
