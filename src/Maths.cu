/*
 * Maths.cu
 *
 *  Created on: May 15, 2014
 *      Author: reid
 */

#include "Maths.h"
//#include "CuMatrix.h"
#include "Kernels.h"

template<> __device__  uint qpow<uint>(uint x, int y) {
	uint z;
	asm("{\n\t"
			// use braces for local scope
			".reg.s32 pow;\n\t"
			".reg.pred p;\n\t"
			"mov.u32 %0,1;\n\t"// set res to 1 initially
			"mov.s32 pow,%2;\n\t"
			"qpowLoop: setp.eq.s32 p,pow,0;\n\t"  // start of loop: set predicate p if pow is 0
			"@p bra LeaveQpow;\n\t"
			" mul.lo.u32 %0, %0, %1;\n\t"
			" sub.s32 pow, pow, 1;\n\t"
			" bra qpowLoop;\n\t"
			"LeaveQpow: }"
			: "+r"(z) : "r" (x), "r" (y));
	return z;
}

template<> __device__  int qpow<int>(int x, int y) {
	int z;
	asm("{\n\t"
			// use braces for local scope
			".reg.s32 pow;\n\t"
			".reg.pred p;\n\t"
			"mov.s32 %0,1;\n\t"// set res to 1 initially
			"mov.s32 pow,%2;\n\t"
			"qpowLoop: setp.eq.s32 p,pow,0;\n\t"  // start of loop: set predicate p if pow is 0
			"@p bra Leave_qpow;\n\t"
			" mul.lo.s32 %0, %0, %1;\n\t"
			" sub.s32 pow, pow, 1;\n\t"
			" bra qpowLoop;\n\t"
			"Leave_qpow: \n\t"
			"}"
			: "+r"(z) : "r" (x), "r" (y));
	return z;
}

template<> __device__  ulong qpow<ulong>(ulong x, int y) {
	ulong z;
	asm("{\n\t"
			// use braces for local scope
			".reg.s32 pow;\n\t"
			".reg.pred p;\n\t"
			"mov.u64 %0,1;\n\t"// set res to 1 initially
			"mov.s32 pow,%2;\n\t"
			"qpowLoop: setp.eq.s32 p,pow,0;\n\t"  // start of loop: set predicate p if pow is 0
			"@p bra Leave_qpow;\n\t"
			" mul.lo.u64 %0, %0, %1;\n\t"
			" sub.s32 pow, pow, 1;\n\t"
			" bra qpowLoop;\n\t"
			"Leave_qpow: }"
			: "+l"(z) : "l" (x), "r" (y));
	return z;
}

template<> __device__  long qpow<long>(long x, int y) {
	long z;
	asm("{\n\t"
			// use braces for local scope
			".reg.s32 pow;\n\t"
			".reg.pred p;\n\t"
			"mov.s64 %0,1;\n\t"// set res to 1 initially
			"mov.s32 pow,%2;\n\t"
			"qpowLoop: setp.eq.s32 p,pow,0;\n\t"  // start of loop: set predicate p if pow is 0
			"@p bra Leave_qpow;\n\t"
			" mul.lo.s64 %0, %0, %1;\n\t"
			" sub.s32 pow, pow, 1;\n\t"
			" bra qpowLoop;\n\t"
			"Leave_qpow: }"
			: "+l"(z) : "l" (x), "r" (y));
	return z;
}

template<> __device__  double qpow<double>(double x, int y) {
	double z;
	asm("{\n\t"
			// use braces for local scope
			".reg.s32 pow;\n\t"
			".reg.pred p;\n\t"
			"mov.f64 %0,1.0;\n\t"// set res to 1 initially
			"mov.s32 pow,%2;\n\t"
			"qpowLoop: setp.eq.s32 p,pow,0;\n\t"  // start of loop: set predicate p if pow is 0
			"@p bra Leave_qpow;\n\t"
			" mul.f64 %0, %0, %1;\n\t"
			" sub.s32 pow, pow, 1;\n\t"
			" bra qpowLoop;\n\t"
			"Leave_qpow: }"
			: "+d"(z) : "d" (x), "r" (y));
	return z;
}

template<> __device__  float qpow<float>(float x, int y) {
	float z;
	asm("{\n\t"
			// use braces for local scope
			".reg.s32 pow;\n\t"
			".reg.pred p;\n\t"
			"mov.f32 %0,1.0;\n\t"// set res to 1 initially
			"mov.s32 pow,%2;\n\t"
			"qpowLoop: setp.eq.s32 p,pow,0;\n\t"  // start of loop: set predicate p if pow is 0
			"@p bra Leave_qpow;\n\t"
			" mul.f32 %0, %0, %1;\n\t"
			" sub.s32 pow, pow, 1;\n\t"
			" bra qpowLoop;\n\t"
			"Leave_qpow: }"
			: "+f"(z) : "f" (x), "r" (y));
	return z;
}

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

//template __device__ int cube<int>(int);

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

//template __device__ long cube<long>(long);

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
/*
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
	flprintf("tot %u\n", tot);
	return tot;
}

__global__ void inclusiveSum(uint* d_res, uint fin) {
	if(threadIdx.x == 0 && threadIdx.y == 0 && d_res) {
		uint isum = inclusiveSumAsm(fin);
		flprintf("isum for %u is %u\n",fin, isum);
		*d_res = isum;
	}
}*/
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
	long nP = 10000;
	int threads;
	int blocks;
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

__host__ CUDART_DEVICE  int largestCommonFactor(uint v, uint w, bool smallest){
	if(v < 4 && w < 4  && v == w) {
		return v;
	}

	long nP = 10000;
	int threads;
	int blocks;
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

__host__ CUDART_DEVICE  int smallestCommonFactor(uint v, uint w) {
	return largestCommonFactor(v,w,true);
}

__host__ CUDART_DEVICE  bool primeQ(uint v) {
	return v == largestFactor(v);
}

__device__ double power( double x,int n) {
	double r = 1.0;
	for(int i = 0 ; i < n ; i++)
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
void rooties(T* roots, uint* count, uint maxRoots,Function<T> fn, T a, T span, T interval, int threads, T epsilon) {
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
	int threads = grid.x * block.x;
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
void rooties( T* roots, uint* count, uint maxRoots, typename func1<T>::inst function, T a, T span, T interval, int threads, T epsilon) {
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
	int threads = grid.x * block.x;
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
	typename func1<T>::inst function;
#ifdef __CUDA_ARCH__
	function = 	(typename func1<T>::inst) funcPtres[idx];
#else
	cherr(cudaMemcpy(&function, &(funcPtres[idx]), sizeof(typename func1<T>::inst), cudaMemcpyDeviceToHost));
#endif
	dim3 grid, block;
	block.x = 256;
	grid.x = slices/block.x;
	int threads = grid.x * block.x;
	if(checkDebug(debugMaths))flprintf("grid.x %u block.x %u\n", grid.x, block.x);
	rooties<<<grid,block>>>(roots, count, maxRoots, function, a, b-a, (b-a)/threads, threads, epsilon);
	cherr(cudaDeviceSynchronize());
	return 0;
}

template __host__ CUDART_DEVICE float bisection<float>(float*, uint*, uint,FuncNums, float, float, float, uint);
template __host__ CUDART_DEVICE double bisection<double>(double*, uint*, uint,FuncNums, double, double, double, unsigned int);
template __host__ CUDART_DEVICE ulong bisection<ulong>(ulong*, uint*, uint,FuncNums, ulong, ulong, ulong, uint);

__host__ CUDART_DEVICE void biggestBin( int* whichBin, const int* binCounts, int nbins) {
	int maxIdx, maxCount = -1, currCount;
	for(int i =0; i < nbins; i++) {
		currCount = binCounts[i];
		if( currCount > maxCount) {
			maxCount = currCount;
			maxIdx = i;
		}
	}
	*whichBin = maxIdx;
}


template <> __device__  uint popc( int val) {
	uint ret;
	asm("{\n\t"
			" popc.b32 %0, %1;\n\t"
			"}"
			: "=r"(ret) : "r" (val));
	return ret;

}
template <> __device__ uint popc( uint val) {
	uint ret;
	asm("{\n\t"
			" popc.b32 %0, %1;\n\t"
			"}"
			: "=r"(ret) : "r" (val));
	return ret;

}
template <> __device__  uint popc( long val) {
	uint ret;
	asm("{\n\t"
			" popc.b64 %0, %1;\n\t"
			"}"
			: "=r"(ret) : "l" (val));
	return ret;

}
template <> __device__  uint popc( ulong val) {
	uint ret;
	asm("{\n\t"
			" popc.b64 %0, %1;\n\t"
			"}"
			: "=r"(ret) : "l" (val));
	return ret;

}

template<> __device__ float sigmoid(const float x, const int depth) {
	float ret;
	asm("{\n\t"
			".reg.f32 accum, ffact, currPow;\n\t"
			".reg.f32 signedX;\n\t"
			".reg.s32 fact, currentTermIdx;\n\t"
			".reg.pred %p;\n\t"
			"mov.s32 fact, 1;\n\t"
			"mov.f32 accum, 1.0;\n\t"
			"mov.f32 currPow, 1.0;\n\t"
			"mov.s32 currentTermIdx, 1;\n\t"
			"mov.f32 signedX, %1;\n\t"
			"neg.f32 signedX, signedX;\n\t"
			"loop:  setp.eq.s32 %p, currentTermIdx, %2;\n\t"
			"@%p bra leavus;\n\t"
			"mul.f32 currPow, currPow, signedX;\n\t"
			"mul.lo.s32 fact, fact, currentTermIdx;\n\t"
			"cvt.rn.f32.s32 ffact, fact;\n\t"
			"div.approx.f32 ffact, currPow, ffact;\n\t"
			"add.f32 accum, accum, ffact;\n\t"
			"add.s32 currentTermIdx, currentTermIdx, 1;\n\t"
			"bra loop;\n\t"
			"leavus:\n\t"
			"add.f32 accum, accum, 1.0;\n\t"
			"div.approx.f32 accum, 1.0, accum;\n\t"
			"mov.f32 %0, accum;\n\t"
			"}"
			: "=f"(ret) : "f" (x), "r" (depth));
	return ret;

}

template <> __device__  double sigmoid( const double x, const int depth) {
	double ret;

	asm("{\n\t"
			".reg.f64 accum, ffact, currPow;\n\t"
			".reg.f64 signedX;\n\t"
			".reg.s32 fact, currentTermIdx;\n\t"
			".reg.pred %p;\n\t"
			"mov.s32 fact, 1;\n\t" // set factorial dividend to initially be 1
			"mov.f64 accum, 1.0;\n\t" // set accumulated result initially to 1 (1 + x + x^2/2! + x^3/3! + ...
			"mov.f64 currPow, 1.0;\n\t" // currPow holds the power of x part of the element, initially at mult identity
			"mov.s32 currentTermIdx, 1;\n\t" // index of current term in taylor series
			"mov.f64 signedX, %1;\n\t" // negate x for sigmoid
			"neg.f64 signedX, signedX;\n\t"
			"loop:  setp.eq.s32 %p, currentTermIdx, %2;\n\t" // check if calculated to the <<depth>>-st term
			"@%p bra leavus;\n\t"
			"mul.f64 currPow, currPow, signedX;\n\t"  // multiply currPow by signedX to get this term's power of x
			"mul.lo.s32 fact, fact, currentTermIdx;\n\t" // mult fact by currentTermIdx to get this term's factorial divisor
			"cvt.rn.f64.s32 ffact, fact;\n\t"
			"div.f64 ffact, currPow, ffact;\n\t"  // divide currPow by fact
			"add.f64 accum, accum, ffact;\n\t"	 // and accumulate
			"add.s32 currentTermIdx, currentTermIdx, 1;\n\t"
			"bra loop;\n\t"
			"leavus:\n\t"  // e^-x calced, now calc rest of sigmoid
			"add.f64 accum, accum, 1.0;\n\t"  // 1 + e^-x
			"div.f64 accum, 1.0, accum;\n\t"  // 1/(1 + e^-x)
			"mov.f64 %0, accum;\n\t"
			"}"
			: "=d"(ret) : "d" (x), "r" (depth));

	return ret;

}


template <> __device__  ulong sigmoid( const ulong x, const int depth) {
	return 0;
}


