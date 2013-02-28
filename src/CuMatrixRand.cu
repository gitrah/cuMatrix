#include "CuMatrix.h"
#include "caps.h"

__global__ void setup_kernel ( curandState * state, int width )
{
    int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = xIdx + yIdx * width;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init (1234, offset, 0, & state [ offset ]) ;
}

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


template <typename T> void CuMatrix<T>::initRand(int height, int width) {
	checkCudaError( cudaMalloc (( void **) & devStates, height * width *  sizeof ( curandState ))) ;
	uint blockW = MIN(width, caps.maxBlock.x);
	uint blockH = MIN( caps.thrdPerBlock/blockW, MIN(height, caps.maxBlock.y));
	dim3 block(blockW, blockH);
	dim3 grid(b_util::enough(blockW, width), b_util::enough(blockH, height));
	setup_kernel<<<grid, block >>>( devStates, width );
}

template <typename T> void CuMatrix<T>::freeRand() {
	cudaFree(devStates);
	devStates = null;
}

template <typename T> cudaError_t CuMatrix<T>::randn(DMatrix<T>& d_ret, T epsilon) {
	if(devStates) {
		freeRand();
	}
	initRand(d_ret.m, d_ret.n);
	uint blockW = MIN(d_ret.n, caps.maxBlock.x);
	uint blockH = MIN( caps.thrdPerBlock/blockW, MIN(d_ret.m, caps.maxBlock.y));
	dim3 block(blockW, blockH);
	dim3 grid(b_util::enough(blockW, d_ret.n), b_util::enough(blockH, d_ret.m));
	generate_uniform_kernel<<<grid,block>>>(devStates, d_ret.elements, epsilon, d_ret.m, d_ret.n);
	freeRand();
	return cudaDeviceSynchronize();
}
//template cudaError_t CuMatrix<float>::randn(DMatrix<float>&, float);
//template cudaError_t CuMatrix<double>::randn(DMatrix<double>&, double);

#include "CuMatrixInster.cu"
