/*
 * Maths.cu
 *
 *  Created on: May 15, 2014
 *      Author: reid
 */

#include "Maths.h"
//#include "CuMatrix.h"
#include "Kernels.h"


template<> __device__  uint cube<uint>(uint x) {
	uint y;
	asm("{\n\t"
			// use braces for local scope
			".reg .u32 t1;\n\t"
			// temp reg t1,
			" mul.lo.u32 t1, %1, %1;\n\t"
			// t1 = x * x
			" mul.lo.u32 %0, t1, %1;\n\t"
			// y = t1 * x
			"}"
			: "=r"(y) : "r" (x));
	return y;
}

template __device__ uint cube<uint>(uint);

template <> __device__  int cube<int>( int v) {
	return cube<uint>(v);
}

template __device__ int cube<int>(int);

template<> __device__ long cube<long>(long x) {
	long y;
	asm("{\n\t"
			// use braces for local scope
			".reg .u64 t1;\n\t"
			// temp reg t1,
			" mul.lo.u64 t1, %1, %1;\n\t"
			// t1 = x * x
			" mul.lo.u64 %0, t1, %1;\n\t"
			// y = t1 * x
			"}"
			: "=l"(y) : "l" (x));
	return y;
}

template __device__ long cube<long>(long);

template<> __device__  float cube<float>(float x) {
	float y;
	asm("{\n\t"
			// use braces for local scope
			".reg .f32 t1;\n\t"
			// temp reg t1,
			" mul.f32 t1, %1, %1;\n\t"
			// t1 = x * x
			" mul.f32 %0, t1, %1;\n\t"
			// y = t1 * x
			"}"
			: "=f"(y) : "f" (x));
	return y;
}

template __device__ float cube<float>(float);

template <typename T> __host__ __device__ T sign(T x) ;
__device__ int signi(int x) {
	int result = 0;
	if(x < 0) {
		result = -1;
	} else {
		result = 1;
	}
	return result;
}


template <> __host__ __device__ float sign(float x) {
	float res = 0;
#ifdef __CUDA_ARCH__
	asm("{\n\t"
			// use braces for local scope
			" .reg .pred p;\n\t"
			" setp.lt.f32 p,%1,0F00000000;\n\t"  // start of loop: set predicate p if %rx < 0
			"@p bra Neg;\n\t"  //
			" mov.f32   %0, 0F3f800000;\n\t" 	// 1
			" bra End;"
			"Neg: \n\t"
			" mov.f32 	%0, 0Fbf800000;\n\t"   // -1
			"End:}"
			: "=f"(res) : "f" (x));
#else
	res = x < 0 ? -1 : 1;
#endif
	return res;

}

template <> __host__ __device__ int sign(int x) {
	int res = 0;
#ifdef __CUDA_ARCH__
	asm volatile("{\n\t"
			// use braces for local scope
			" .reg .pred p;\n\t"
			" setp.lt.s32 p,%1,0;\n\t"  // start of loop: set predicate p if %rx < 0
			"@p bra Neg;\n\t"  //
			" mov.b32   %0, 1;\n\t" 	// 1
			" bra End;\n\t"
			"Neg: \n\t"
			" mov.b32 	%0, -1;\n\t"   // -1
			"End:}"
			: "=r"(res) : "r" (x));
#else
	res = x < 0 ? -1 : 1;
#endif
	return res;

}

template <> __host__ __device__ long sign(long x) {
	long res = 0;
#ifdef __CUDA_ARCH__
	asm("{\n\t"
			// use braces for local scope
			" .reg .pred p;\n\t"
			" setp.lt.s64 p,%1,0;\n\t"  // start of loop: set predicate p if  < 0
			"@p bra Neg;\n\t"  //
			" mov.s64   %0, 1;\n\t" 	// 1
			" bra End;"
			"Neg: \n\t"
			" mov.s64 	%0, -1;\n\t"   // -1
			"End:}"
			: "=l"(res) : "l" (x));
#else
	res = x < 0 ? -1 : 1;
#endif
	return res;

}

template <> __host__ __device__ double sign(double x) {
	double res = 0;
#ifdef __CUDA_ARCH__
	asm("{\n\t"
			// use braces for local scope
			" .reg .pred p;\n\t"
			" setp.lt.f64 p,%1,0.0;\n\t"  // start of loop: set predicate p if %rx < 0
			"@p bra Neg;\n\t"  //
			" mov.f64   %0, 1.0;\n\t" 	// 1 for pos (or zero)
			" bra End;"
			"Neg: \n\t"
			" mov.f64 	%0, -1.0;\n\t"   // -1
			"End:}"
			: "=d"(res) : "d" (x));
#else
	res = x < 0 ? -1 : 1;
#endif
	return res;

}
template <> __host__ __device__ ulong sign(ulong  x) {
	return 1;
}

__device__ uint inclusiveSumAsm(uint s) {
	uint tot = 0;

	asm("{\n\t"
			// use braces for local scope
			" .reg.u32 %total, %curr;\n\t"
			" .reg .pred p;\n\t"
			" mov.u32 	%total, 0;\n\t"  // set %total to 0
			" mov.u32 	%curr, %1;\n\t"  // set %curr to arg s
			"sumLuexp: setp.eq.u32 p,%curr,0;\n\t"  // start of loop: set predicate p if %curr is 0
			"@p bra Leavus;\n\t"  // exit if p true
			" add.u32 %total, %total, %curr;\n\t" 	// add %curr to %total
			" sub.u32 %curr, %curr, 1;\n\t" 		// subtract 1 from %curr
			" bra sumLuexp;\n\t"
			"Leavus: \n\t"
			" mov.u32 	%0, %total;\n\t"   // set tot with %total
			"}"
			: "=r"(tot) : "r" (s));

	return tot;
}

__global__ void inclusiveSum(uint* d_res, uint fin) {
	if(threadIdx.x == 0 && threadIdx.y == 0 && d_res) {
		uint isum = inclusiveSumAsm(fin);
		flprintf("isum for %u is %u\n",fin, isum);
		*d_res = isum;
	}
}
template<typename T> __host__ CUDART_DEVICE void testem(BinaryOpF<T,1> ref)  {
#ifdef __CUDA_ARCH__
	flprintf("ref(6,7) %f\n", (float) ref(6,7));
#else
	prlocf("hosties");
#endif
}

__host__ CUDART_DEVICE  int largestFactor(uint v, bool smallest) {
	if(v < 4) {
		return v;
	}
	uint nP = 10000;
	uint threads;
	uint blocks;
	::getReductionExecContext(blocks,threads, nP);
	int* d_ints;
	cherr(cudaMalloc(&d_ints, blocks * sizeof(int)));
	DMatrix<int> d_Res(d_ints, blocks,1);
	DMatrix<int> d_M(First10KPrimes, 10000,1);

	maxBinaryOp<int> max = Functory<int,maxBinaryOp>::pinch();
	testem(max);
	minNotZeroBinaryOp<int> min  = Functory<int,minNotZeroBinaryOp>::pinch();
	testem(min);
	divisibleUnaryOp<int> div  = Functory<int,divisibleUnaryOp>::pinch((int)v);
	int total;
	if(smallest) {
		gloloReduceOpLauncher(&total, d_Res, nP, d_M, div, min, 0);
	}else {
		gloloReduceOpLauncher(&total, d_Res, nP, d_M, div, max, 0);
	}
	cherr(cudaDeviceSynchronize());
	cherr(cudaFree(d_ints));
	return total;

}

__host__ CUDART_DEVICE  int smallestFactor(uint v) {
	return largestFactor(v,true);
}

__host__ CUDART_DEVICE  int largestMutualFactor(uint v, uint w, bool smallest){
	if(v < 4 && w < 4  && v == w) {
		return v;
	}

	uint nP = 10000;
	uint threads;
	uint blocks;
	::getReductionExecContext(blocks,threads, nP);
	int* d_ints;
	cherr(cudaMalloc(&d_ints, blocks * sizeof(int)));
	DMatrix<int> d_Res(d_ints, blocks,1);
	DMatrix<int> d_M(First10KPrimes, 10000,1);
	maxBinaryOp<int> max = Functory<int,maxBinaryOp>::pinch();
	minNotZeroBinaryOp<int> min = Functory<int,minNotZeroBinaryOp>::pinch();
	mutuallyDivisibleUnaryOp<int> mutdiv = Functory<int,mutuallyDivisibleUnaryOp>::pinch(v,w);

	int total;
	if(smallest) {
		gloloReduceOpLauncher(&total, d_Res, nP, d_M, mutdiv, min, 0);
	}else {
		gloloReduceOpLauncher(&total, d_Res, nP, d_M, mutdiv, max, 0);
	}
	cherr(cudaDeviceSynchronize());
	cherr(cudaFree(d_ints));
	return total;
}

__host__ CUDART_DEVICE  int smallestMutualFactor(uint v, uint w) {
	return largestMutualFactor(v,w,true);
}

__host__ CUDART_DEVICE  bool primeQ(uint v) {
	return v == largestFactor(v);
}

__device__ double power( double x,uint N) {
	double r = 1.0;
	for(int i = 0 ; i < N ; i++)
		r *= x;
	return r;
}

/*
template <typename T, template <typename> class Function>  __global__ void bisduction(Function<T> function, T a, T b) {
	uint tidx = threadIdx.x + blockIdx.x * blockDim.x;


}
*
*/

template<typename T, template <typename> class Function>  __host__ __device__ inline T gradient(Function<T> function, T x, T epsilon) {
	return (function(x + epsilon) - function(x - epsilon))/(2 * epsilon);
}

#ifdef CuMatrix_Enable_Cdp
template<typename T, template <typename> class Function> __global__
void rooties(T* roots, uint* count, uint maxRoots,Function<T> fn, T a, T span, T interval, uint threads, T epsilon) {
	/*
	 *  one thread per interval
	 *  if sign of ordinate or gradient changes
	 *  	if interval width < epsilon, search  launch a new grid on interval
	 *
	 */
	uint intervalIdx = threadIdx.x + blockDim.x * blockIdx.x;
	T myA = a + span * intervalIdx / threads;
	T myB = myA + interval;
	T fa = fn(myA);
	T ga = gradient(fn,myA, epsilon);
	T fb = fn(myB);
	T gb = gradient(fn,myB, epsilon);

	if(sign(fa) != sign(fb)) {
		if(abs((float)(myB-myA)) > epsilon) {
			if(checkDebug(debugMaths))flprintf("[%f,%f] brackets a root, launching !\n", (float) myA, (float)myB);
			bisection(roots, count, maxRoots, fn, myA, myB, epsilon, 1024);
/*
			dim3 grid, block;
			block.x = 256;
			grid.x = threads/block.x;
			uint rthreads = grid.x * block.x;

			rooties<<<grid,block>>>(fn, myA, myB-myA, (myB-myA)/rthreads, rthreads, epsilon);
*/
		} else {
			if(checkDebug(debugMaths))flprintf("[%f,%f] brackets a root within epsilon!\n", (float) myA, (float)myB);
			uint idx = atomicInc(count,maxRoots);
			if(idx < maxRoots){
				roots[idx] = myA + epsilon/2;
			}else {
				if(checkDebug(debugMaths))flprintf("ignoring [%f,%f] because maxRoots exceeded!\n", (float)myA, (float)myB);
			}
		}
	}
	if(sign(ga) != sign(gb)) {
		if(checkDebug(debugMaths))flprintf("[%f,%f] -> [%f,%f] brackets a local max/minima!\n", (float) myA, (float)myB, (float) ga, (float)gb);
	}

}
#endif
/*
 *
 * at launch divides [a,b] into grid*block subintervals
 * 	every subinterval with sign change
 * 		if interval width < epsilon
 * 		launches a new kernel on that subint
 *
 */
template<typename T, template <typename> class Function> __host__ CUDART_DEVICE
T bisection(T* roots, uint* count, uint maxRoots,Function<T> function, T a, T b, T epsilon, uint slices) {
	T fa = function(a);
	T fb = function(b);
	if(checkDebug(debugMaths))if(sign(fa) == sign(fb) ) {
		flprintf("[%f, %f] may not bracket root[s]\n", (float) a, (float) b);
	} else {
		flprintf("[%f, %f] brackets at least one root\n", (float) a, (float) b);
	}

	if(checkDebug(debugMaths))if(sign(gradient(function, a, (T).1)) == sign(gradient(function, b, (T).1)) ) {
		flprintf("[%f, %f] may not bracket a local minima or maxima\n", (float) a, (float) b);
	} else {
		flprintf("[%f, %f] brackets at least one local minima or maxima\n", (float) a, (float) b);
	}

	dim3 grid, block;
	block.x = 256;
	grid.x = slices/block.x;
	if(checkDebug(debugMaths))flprintf("grid.x %u block.x %u\n", grid.x, block.x);
#ifdef CuMatrix_Enable_Cdp
	uint threads = grid.x * block.x;
	rooties<<<grid,block>>>(roots, count, maxRoots, function, a, b-a, (b-a)/threads, threads, epsilon);
#else
	prlocf("not implemented for non-cdp\n");
	assert(false);
#endif
	return 0;
}

template __host__ CUDART_DEVICE float bisection< float, deg2UnaryOp>(float*, uint*, uint, deg2UnaryOp<float>, float, float, float, uint);
template __host__ CUDART_DEVICE double bisection<double, deg2UnaryOp>(double*, uint*, uint, deg2UnaryOp<double>, double, double,double, uint);
template __host__ CUDART_DEVICE ulong bisection<ulong, deg2UnaryOp>(ulong*, uint*, uint, deg2UnaryOp<ulong>, ulong, ulong, ulong, uint);


template<typename T>  __host__ __device__ inline T gradient(typename func1<T>::inst function, T x, T epsilon) {
	return ( function(x + epsilon) - function(x - epsilon))/(2 * epsilon);
}

template<typename T> __global__
void rooties( T* roots, uint* count, uint maxRoots, typename func1<T>::inst function, T a, T span, T interval, uint threads, T epsilon) {
	/*
	 *  one thread per interval
	 *  if sign of ordinate or gradient changes
	 *  	if interval width < epsilon, search  launch a new grid on interval
	 *
	 */
	uint intervalIdx = threadIdx.x + blockDim.x * blockIdx.x;
	T myA = a + span * intervalIdx / threads;
	T myB = myA + interval;
	T fa = function(myA);
	T ga = gradient(function,myA, epsilon);
	T fb = function(myB);
	T gb = gradient(function,myB, epsilon);

	if(sign(fa) != sign(fb)) {
		if(abs((float)(myB-myA)) > epsilon) {
			//flprintf("[%f,%f] brackets a root, launching !\n", (float) myA, (float)myB);
			//bisection(function, myA, myB, epsilon, 1024);
			dim3 grid, block;
			block.x = 256;
			grid.x = threads/block.x;
#ifdef CuMatrix_Enable_Cdp
			uint rthreads = grid.x * block.x;
			rooties<<<grid,block>>>(roots, count, maxRoots, function, myA, myB-myA, (myB-myA)/rthreads, rthreads, epsilon);
#else
	prlocf("not implemented for non-cdp\n");
	assert(false);
#endif

		} else {
			//flprintf("[%f,%f] brackets a root within epsilon!\n", (float) myA, (float)myB);
			uint idx = atomicInc(count,maxRoots);
			if(idx < maxRoots){
				roots[idx] = myA + epsilon/2;
			}else {
				if(checkDebug(debugMaths))flprintf("ignoring [%f,%f] because maxRoots exceeded!\n", (float)myA, (float)myB);
			}
		}
	}
	if(sign(ga) != sign(gb)) {
		if(checkDebug(debugMaths))flprintf("[%f,%f] -> [%f,%f] brackets a local max/minima!\n", (float) myA, (float)myB,(float) ga,(float)gb);
	}
}

template<typename T> __host__ CUDART_DEVICE T bisection( T* roots, uint* count, uint maxRoots, typename func1<T>::inst function, T a, T b, T epsilon, uint slices) {
	if(checkDebug(debugMaths))prlocf("bisection enter\n");
	ulong ul = (ulong) function;
	if(checkDebug(debugMaths))flprintf("ul %lu, as ptr %p\n", ul, ul);
	dim3 grid, block;
	block.x = 256;
	grid.x = slices/block.x;
	uint threads = grid.x * block.x;
	if(checkDebug(debugMaths))flprintf("grid.x %u block.x %u\n", grid.x, block.x);
	rooties<<<grid,block>>>(roots, count, maxRoots, function, a, b-a, (b-a)/threads, threads, epsilon);
	cherr(cudaDeviceSynchronize());
	return 0;
}

template __host__ CUDART_DEVICE float bisection<float>(float*, uint*, uint,func1<float>::inst, float, float, float, uint);
template __host__ CUDART_DEVICE double bisection<double>(double*, uint*, uint,func1<double>::inst, double, double, double, unsigned int);
template __host__ CUDART_DEVICE ulong bisection<ulong>(ulong*, uint*, uint,func1<ulong>::inst, ulong, ulong, ulong, uint);

template<typename T> __host__ CUDART_DEVICE T bisection(T* roots, uint* count, uint maxRoots,  FuncNums idx, T a, T b, T epsilon, uint slices) {
	if(checkDebug(debugMaths))prlocf("bisection enter\n");
	typename func1<T>::inst function = 	(typename func1<T>::inst) funcPtres[idx];
	dim3 grid, block;
	block.x = 256;
	grid.x = slices/block.x;
	uint threads = grid.x * block.x;
	if(checkDebug(debugMaths))flprintf("grid.x %u block.x %u\n", grid.x, block.x);
	rooties<<<grid,block>>>(roots, count, maxRoots, function, a, b-a, (b-a)/threads, threads, epsilon);
	cherr(cudaDeviceSynchronize());
	return 0;
}

template __host__ CUDART_DEVICE float bisection<float>(float*, uint*, uint,FuncNums, float, float, float, uint);
template __host__ CUDART_DEVICE double bisection<double>(double*, uint*, uint,FuncNums, double, double, double, unsigned int);
template __host__ CUDART_DEVICE ulong bisection<ulong>(ulong*, uint*, uint,FuncNums, ulong, ulong, ulong, uint);


