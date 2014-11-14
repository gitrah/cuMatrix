/*
 * PFunctor.cu
 *
 *  Created on: Sep 28, 2014
 *      Author: reid
 */



#include "CuFunctor.h"

#include <iostream>
#include <float.h>
#include "UnaryOpIndexF_Gen.h"
#include "UnaryOpF_Gen.h"
#include "BinaryOpF_Gen.h"
#include "debug.h"

bool SetupMbrFuncs[dtLast][MAX_GPUS];

void clearSetupMbrFlags() {
	for(auto i =0; i < dtLast; i++) {
		for (auto j = 0; j < MAX_GPUS; j++) {
			SetupMbrFuncs[i][j] = false;
		}
	}
}

__host__ __device__ float& CuFunctor<float,2>::operator[](ptrdiff_t ofs) {	assert(ofs >= 0 && ofs < 2); return ofs ? state.y : state.x;	}
__host__ __device__ const float& CuFunctor<float,2>::operator[](ptrdiff_t ofs) const {	assert(ofs >= 0 && ofs < 2); return ofs ? state.y : state.x;	}

__host__ __device__ double& CuFunctor<double,2>::operator[](ptrdiff_t ofs) {	assert(ofs >= 0 && ofs < 2); return ofs ? state.y : state.x;	}
__host__ __device__ const double& CuFunctor<double,2>::operator[](ptrdiff_t ofs) const {	assert(ofs >= 0 && ofs < 2); return ofs ? state.y : state.x;	}

__host__ __device__ int& CuFunctor<int,2>::operator[](ptrdiff_t ofs) {	assert(ofs >= 0 && ofs < 2); return ofs ? state.y : state.x;	}
__host__ __device__ const int& CuFunctor<int,2>::operator[](ptrdiff_t ofs) const {	assert(ofs >= 0 && ofs < 2); return ofs ? state.y : state.x;	}

__host__ __device__ uint& CuFunctor<uint,2>::operator[](ptrdiff_t ofs) {	assert(ofs >= 0 && ofs < 2); return ofs ? state.y : state.x;	}
__host__ __device__ const uint& CuFunctor<uint,2>::operator[](ptrdiff_t ofs) const {	assert(ofs >= 0 && ofs < 2); return ofs ? state.y : state.x;	}

__host__ __device__ ulong& CuFunctor<ulong,2>::operator[](ptrdiff_t ofs) {	assert(ofs >= 0 && ofs < 2); return ofs ? state.y : state.x;	}
__host__ __device__ const ulong& CuFunctor<ulong,2>::operator[](ptrdiff_t ofs) const {	assert(ofs >= 0 && ofs < 2); return ofs ? state.y : state.x;	}

__host__ __device__ float& CuFunctor<float,3>::operator[](ptrdiff_t ofs) {
	assert(ofs >= 0 && ofs < 3);
	switch (ofs) {
	case 0:
		return state.x;
	case 1:
		return state.y;
	case 2:
		return state.z;
	default:
		assert(false);
		return state.x;
	}
}

__host__ __device__ const float& CuFunctor<float,3>::operator[](ptrdiff_t ofs) const {
	assert(ofs >= 0 && ofs < 3);
	switch (ofs) {
	case 0:
		return state.x;
	case 1:
		return state.y;
	case 2:
		return state.z;
	default:
		assert(false);
		return state.x;
	}
}

__host__ __device__ double& CuFunctor<double,3>::operator[](ptrdiff_t ofs) {
	assert(ofs >= 0 && ofs < 3);
	switch (ofs) {
	case 0:
		return state.x;
	case 1:
		return state.y;
	case 2:
		return state.z;
	default:
		assert(false);
		return state.x;
	}
}

__host__ __device__ const double& CuFunctor<double,3>::operator[](ptrdiff_t ofs) const {
	assert(ofs >= 0 && ofs < 3);
	switch (ofs) {
	case 0:
		return state.x;
	case 1:
		return state.y;
	case 2:
		return state.z;
	default:
		assert(false);
		return state.x;
	}
}

__host__ __device__ int& CuFunctor<int,3>::operator[](ptrdiff_t ofs) {
	assert(ofs >= 0 && ofs < 3);
	switch (ofs) {
	case 0:
		return state.x;
	case 1:
		return state.y;
	case 2:
		return state.z;
	default:
		assert(false);
		return state.x;
	}
}

__host__ __device__ const int& CuFunctor<int,3>::operator[](ptrdiff_t ofs) const {
	assert(ofs >= 0 && ofs < 3);
	switch (ofs) {
	case 0:
		return state.x;
	case 1:
		return state.y;
	case 2:
		return state.z;
	default:
		assert(false);
		return state.x;
	}
}

__host__ __device__ uint& CuFunctor<uint,3>::operator[](ptrdiff_t ofs) {
	assert(ofs >= 0 && ofs < 3);
	switch (ofs) {
	case 0:
		return state.x;
	case 1:
		return state.y;
	case 2:
		return state.z;
	default:
		assert(false);
		return state.x;
	}
}

__host__ __device__ const uint& CuFunctor<uint,3>::operator[](ptrdiff_t ofs) const {
	assert(ofs >= 0 && ofs < 3);
	switch (ofs) {
	case 0:
		return state.x;
	case 1:
		return state.y;
	case 2:
		return state.z;
	default:
		assert(false);
		return state.x;
	}
}

__host__ __device__ ulong& CuFunctor<ulong,3>::operator[](ptrdiff_t ofs) {
	assert(ofs >= 0 && ofs < 3);
	switch (ofs) {
	case 0:
		return state.x;
	case 1:
		return state.y;
	case 2:
		return state.z;
	default:
		assert(false);
		return state.x;
	}
}

__host__ __device__ const ulong& CuFunctor<ulong,3>::operator[](ptrdiff_t ofs) const {
	assert(ofs >= 0 && ofs < 3);
	switch (ofs) {
	case 0:
		return state.x;
	case 1:
		return state.y;
	case 2:
		return state.z;
	default:
		assert(false);
		return state.x;
	}
}

template<> __host__ __device__ float epsilon<float>() {
	return 1e-6;
}
template<> __host__ __device__ double epsilon<double>() {
	return 1e-10;
}

template<typename T> __host__ __device__ T minValue() {
	return 0; // will be overridden by specializations
}

template<> __host__ __device__ float minValue<float>() {
	return FLT_MIN;
}
template<> __host__ __device__ double minValue<double>() {
	return DBL_MIN;
}
template<> __host__ __device__ int minValue<int>() {
	return INT_MIN;
}

template<> __host__ __device__ float maxValue<float>() {
	return FLT_MAX;
}
template<> __host__ __device__ double maxValue<double>() {

	return DBL_MAX;
}
template<> __host__ __device__ ulong maxValue<ulong>() {
	return 0xffffffff;
}
template<> __host__ __device__ int maxValue<int>() {
	return INT_MAX;
}
template<> __host__ __device__ uint maxValue<uint>() {
	return 0xFFFF;
}



/*
 *
 * test kernels
 *
 *
 */

#ifdef CuMatrix_Enable_KTS
template<typename T, template <typename> class IndexUnaryOp> __global__ void switchableIndexFunctorTest( IndexUnaryOp<T> ftr )
#else
template <typename T, int StateDim> __global__ void switchableIndexFunctorTest( UnaryOpIndexF<T,StateDim> ftr )
#endif
{
	uint idx = blockDim.x * threadIdx.y + threadIdx.x;

#ifdef CuMatrix_StatFunc
	flprintf("ftr.fn %p\n",ftr.fn);
#else
	#ifndef CuMatrix_Enable_KTS
		flprintf("ftr.operation %p\n",ftr.operation);
	#endif
#endif
	/*
	flprintf("(device-side) &UnaryOpIndexF<T,0>::operatorOneOver == %p\n", &UnaryOpIndexF<T,0>::operatorOneOver );
	uof.operation =  &UnaryOpIndexF<T,0>::operatorOneOver;
*/
//	flprintf("switchableIndexFunctorTest idx %u, t %f\n", idx, ftr(idx));
//	flprintf("switchableIndexFunctorTest idx %u, oif %f\n", idx, uof(idx));
}

#ifdef CuMatrix_Enable_KTS
template<typename T, template <typename> class IndexUnaryOp> __global__ void indexFunctorTest( IndexUnaryOp<T> ftr )
#else
template <typename T, int StateDim> __global__ void indexFunctorTest( UnaryOpIndexF<T,StateDim> ftr )
#endif
{
	uint idx = blockDim.x * threadIdx.y + threadIdx.x;
#ifdef CuMatrix_StatFunc
	flprintf("indexFunctorTest<float,%d> ftr.fn %p\n", StateDim, ftr.fn);
#else
	#ifndef CuMatrix_Enable_KTS
		flprintf("indexFunctorTest<float,%d> ftr.operation %p\n", StateDim, ftr.operation);
	#endif
#endif
#ifdef CuMatrix_Enable_KTS
	flprintf("indexFunctorTest<float> idx %u, t %f\n", idx, (float) ftr(idx));
#else
	flprintf("indexFunctorTest<float,%d> idx %u, t %f\n", StateDim, idx, (float) ftr(idx));
#endif
}

#ifdef CuMatrix_Enable_KTS
template <typename T, template <typename> class UnaryOp> __global__ void unaryOpTest( UnaryOp<T> uopf ) {
#else
template <typename T, int StateDim> __global__ void unaryOpTest( UnaryOpF<T,StateDim> uopf ) {
#endif
	T xi = static_cast<T>( -5 + 1.0 * threadIdx.x);
	flprintf("unaryOpTest xi %f, uopf(xi) %f\n", (float) xi, (float)uopf(xi));
}

#ifdef CuMatrix_Enable_KTS
template <typename T, template <typename> class BinaryOp> __global__ void binaryOpTest( BinaryOp<T> bopf ) {
#else
template <typename T, int StateDim> __global__ void binaryOpTest( BinaryOpF<T,StateDim> bopf ) {
#endif
	T xi1 = static_cast<T>( -1.5 + 1.0 * threadIdx.x);
	T xi2 = static_cast<T>( -1.5 + 1.0 * threadIdx.y);
	flprintf("binaryOpTest xi1 %f,xi2 %f, bopf(xi1,xi2) %f\n", (float) xi1, (float) xi2, (float)bopf(xi1,xi2));
}


void testGets() {
	CuFunctor<float,1> d;
	d.state = 5;
	std::cout << "d[0] " << d[0] << "\n";
	assert(d[0]==5);
	CuFunctor<ulong,1> du;
	du.state = 55u;
	std::cout << "du[0] " << du[0] << "\n";
	assert(du[0]==55u);
	CuFunctor<float,2> d2;
	float2 f2;
	f2.x = 5; f2.y = 6;
	d2.state = f2;
	std::cout << "d2[1] " << d2[1] << "\n";
	std::cout << "d2[0] " << d2[0] << "\n";
	assert(d2[1]==6);
	assert(d2[0]==5);
}

/* demonstrates using a switch to select templating by base functor (and using method
 * pointer to call the subclass's operator) or templating by functor directly
 */
template<typename T>void test0sFillers() {
	oneOverFiller<T> oof = Functory<T,oneOverFiller>::pinch();
	UnaryOpIndexF<T,0> uof(oof);
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		printf("uof.operatorOneOver(5) %f\n",(float)uof.operatorOneOver(uof,5));
	#else
		printf("uof.operatorOneOver(5) %f\n",(float)uof.operatorOneOver(5));
	#endif
#endif

	checkCudaErrors(cudaDeviceSynchronize());

#ifdef CuMatrix_Enable_KTS
	std::cout << "callin switchableIndexFunctorTest<<<1,3>>>(oof)\n";
	flprintf("test0sFillers host oof(5) %f\n", oof(5));
	switchableIndexFunctorTest<<<1,3>>>(oof);
#else
	flprintf("(host-side) &UnaryOpIndexF<T,0>::operatorOneOver == %p\n", &UnaryOpIndexF<T,0>::operatorOneOver );
	flprintf("test0sFillers host oof(5) %f\n", oof(5));
	std::cout << "callin switchableIndexFunctorTest with one over filler\n";
	std::cout << "callin switchableIndexFunctorTest<T,0><<<1,3>>>(oof)\n";
	switchableIndexFunctorTest<T,0><<<1,3>>>(oof);
#endif

	checkCudaErrors(cudaDeviceSynchronize());

#ifdef CuMatrix_Enable_KTS
	switchableIndexFunctorTest<<<1,3>>>(oof);
#else
	switchableIndexFunctorTest<T,0><<<1,3>>>(oof);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "callin indexFunctorTest with oof 1/ filler with value\n";
#ifdef CuMatrix_Enable_KTS
	indexFunctorTest<<<1,3>>>(oof);
#else
	indexFunctorTest<T,0><<<1,3>>>(oof);
#endif

	checkCudaErrors(cudaDeviceSynchronize());
}

template<typename T>void test0sUnaryOps() {
	sigmoidUnaryOp<T> z = Functory<T,sigmoidUnaryOp>::pinch();
	std::cout << "callin unaryOpTest with sigmoidUnaryOp\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,sigmoidUnaryOp><<<1,10>>>(z);
#else
	unaryOpTest<T,0><<<1,10>>>(z);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	sigmoidGradientUnaryOp<T> zg = Functory<T,sigmoidGradientUnaryOp>::pinch();

	std::cout << "callin unaryOpTest with sigmoidGradientUnaryOp\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,sigmoidGradientUnaryOp><<<1,10>>>(zg);
#else
	unaryOpTest<T,0><<<1,10>>>(zg);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	negateUnaryOp<T> neg = Functory<T,negateUnaryOp>::pinch();
	std::cout << "callin unaryOpTest with negateUnaryOp\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,negateUnaryOp><<<1,10>>>(neg);
#else
	unaryOpTest<T,0><<<1,10>>>(neg);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	logUnaryOp<T> lg = Functory<T,logUnaryOp>::pinch();
	std::cout << "callin unaryOpTest with logUnaryOp\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,logUnaryOp><<<1,10>>>(lg);
#else
	unaryOpTest<T,0><<<1,10>>>(lg);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	oneOverUnaryOp<T> oog = Functory<T,oneOverUnaryOp>::pinch();
	std::cout << "callin unaryOpTest with oneOverUnaryOp\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,oneOverUnaryOp><<<1,10>>>(oog);
#else
	unaryOpTest<T,0><<<1,10>>>(oog);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	sqrtUnaryOp<T> sqrtf = Functory<T,sqrtUnaryOp>::pinch();
	std::cout << "callin unaryOpTest with sqrtUnaryOp\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,sqrtUnaryOp><<<1,10>>>(sqrtf);
#else
	unaryOpTest<T,0><<<1,10>>>(sqrtf);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	sqrUnaryOp<T> sqrf = Functory<T,sqrUnaryOp>::pinch();
	std::cout << "callin unaryOpTest with sqrUnaryOp\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,sqrUnaryOp><<<1,10>>>(sqrf);
#else
	unaryOpTest<T,0><<<1,10>>>(sqrf);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	slowInvSqrtUnaryOp<T> sisf = Functory<T,slowInvSqrtUnaryOp>::pinch();
	std::cout << "callin unaryOpTest with slowInvSqrtUnaryOp\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,slowInvSqrtUnaryOp><<<1,10>>>(sisf);
#else
	unaryOpTest<T,0><<<1,10>>>(sisf);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	approxInvSqrtUnaryOp<T> aisf = Functory<T,approxInvSqrtUnaryOp>::pinch();
	std::cout << "callin unaryOpTest with approxInvSqrtUnaryOp\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,approxInvSqrtUnaryOp><<<1,10>>>(aisf);
#else
	unaryOpTest<T,0><<<1,10>>>(aisf);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	oneOrZeroUnaryOp<T> oozbuof = Functory<T,oneOrZeroUnaryOp>::pinch();
	std::cout << "callin unaryOpTest with oneOrZeroUnaryOp\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,oneOrZeroUnaryOp><<<1,10>>>(oozbuof);
#else
	unaryOpTest<T,0><<<1,10>>>(oozbuof);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

}

template<typename T>void test1sFillers() {
	constFiller<T> cf = Functory<T,constFiller>::pinch(6.5);
	UnaryOpIndexF<T,1> uof(cf);
#ifdef CuMatrix_StatFunc
	printf("uof.operatorConst(5) %f\n",(float)uof.operatorConst(uof,5));
#else
	#ifndef CuMatrix_Enable_KTS
		printf("uof.operatorConst(5) %f\n",(float)uof.operatorConst(5));
	#endif
#endif
	std::cout << "callin indexFunctorTest with uof const filler with value " << uof[0] << "\n";
#ifndef CuMatrix_Enable_KTS
	indexFunctorTest<T,1><<<1,3>>>(uof);
	checkCudaErrors(cudaDeviceSynchronize());
	std::cout << "callin indexFunctorTest with cf const filler with value " << cf[0] << "\n";
	indexFunctorTest<T,1><<<1,3>>>(uof);
	checkCudaErrors(cudaDeviceSynchronize());

	sequenceFiller<T> seqf = Functory<T,sequenceFiller>::pinch(21);
	std::cout << "callin indexFunctorTest with sequenceFiller filler with value " << seqf[0] << "\n";
	indexFunctorTest<T,1><<<1,3>>>(seqf);
	checkCudaErrors(cudaDeviceSynchronize());

	powFiller<T> powFlr = Functory<T,powFiller>::pinch(1.1);
	std::cout << "callin indexFunctorTest with powFiller filler with value " << powFlr[0] << "\n";
	indexFunctorTest<T,1><<<1,3>>>(powFlr);
	checkCudaErrors(cudaDeviceSynchronize());
#else
	indexFunctorTest<T,constFiller><<<1,3>>>(cf);
	checkCudaErrors(cudaDeviceSynchronize());
	std::cout << "callin indexFunctorTest with cf const filler with value " << cf[0] << "\n";
	//indexFunctorTest<T,UnaryOpIndexF><<<1,3>>>(uof);
	checkCudaErrors(cudaDeviceSynchronize());

	sequenceFiller<T> seqf = Functory<T,sequenceFiller>::pinch(21);
	std::cout << "callin indexFunctorTest with sequenceFiller filler with value " << seqf[0] << "\n";
	indexFunctorTest<T,sequenceFiller><<<1,3>>>(seqf);
	checkCudaErrors(cudaDeviceSynchronize());

	powFiller<T> powFlr = Functory<T,powFiller>::pinch(1.1);
	std::cout << "callin indexFunctorTest with powFiller filler with value " << powFlr[0] << "\n";
	indexFunctorTest<T,powFiller><<<1,3>>>(powFlr);
	checkCudaErrors(cudaDeviceSynchronize());
#endif
}

template<typename T>void test1sUnaryOps() {
	powUnaryOp<T> puo = Functory<T,powUnaryOp>::pinch(5);
	std::cout << "callin unaryOpTest<T,1> with powUnaryOp " << puo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,powUnaryOp><<<1,10>>>(puo);
#else
	unaryOpTest<T,1><<<1,10>>>(puo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	scaleUnaryOp<T> suo = Functory<T,scaleUnaryOp>::pinch(50);
	std::cout << "callin unaryOpTest<T,1> with scaleUnaryOp " << suo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,scaleUnaryOp><<<1,10>>>(suo);
#else
	unaryOpTest<T,1><<<1,10>>>(suo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	translationUnaryOp<T> tuo = Functory<T,translationUnaryOp>::pinch(-37.5);
	std::cout << "callin unaryOpTest<T,1> with translationUnaryOp " << tuo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,translationUnaryOp><<<1,10>>>(tuo);
#else
	unaryOpTest<T,1><<<1,10>>>(tuo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	subFromUnaryOp<T> sfuo = Functory<T,subFromUnaryOp>::pinch(101);
	std::cout << "callin unaryOpTest<T,1> with subFromUnaryOp " << sfuo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,subFromUnaryOp><<<1,10>>>(sfuo);
#else
	unaryOpTest<T,1><<<1,10>>>(sfuo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	ltUnaryOp<T> ltuo = Functory<T,ltUnaryOp>::pinch(2);
	std::cout << "callin unaryOpTest<T,1> with ltUnaryOp " << ltuo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,ltUnaryOp><<<1,10>>>(ltuo);
#else
	unaryOpTest<T,1><<<1,10>>>(ltuo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	lteUnaryOp<T> lteuo = Functory<T,lteUnaryOp>::pinch(2);
	std::cout << "callin unaryOpTest<T,1> with lteUnaryOp " << lteuo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,lteUnaryOp><<<1,10>>>(lteuo);
#else
	unaryOpTest<T,1><<<1,10>>>(lteuo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	gtUnaryOp<T> gtuo = Functory<T,gtUnaryOp>::pinch(2);
	std::cout << "callin unaryOpTest<T,1> with gtUnaryOp " << gtuo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,gtUnaryOp><<<1,10>>>(gtuo);
#else
	unaryOpTest<T,1><<<1,10>>>(gtuo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	gteUnaryOp<T> gteuo = Functory<T,gteUnaryOp>::pinch(2);
	std::cout << "callin unaryOpTest<T,1> with gteUnaryOp " << gteuo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,gteUnaryOp><<<1,10>>>(gteuo);
#else
	unaryOpTest<T,1><<<1,10>>>(gteuo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	eqUnaryOp<T> equo = Functory<T,eqUnaryOp>::pinch(2);
	std::cout << "callin unaryOpTest<T,1> with eqUnaryOp " << equo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,eqUnaryOp><<<1,10>>>(equo);
#else
	unaryOpTest<T,1><<<1,10>>>(equo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	neqUnaryOp<T> nequo = Functory<T,neqUnaryOp>::pinch(2);
	std::cout << "callin unaryOpTest<T,1> with neqUnaryOp " << nequo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,neqUnaryOp><<<1,10>>>(nequo);
#else
	unaryOpTest<T,1><<<1,10>>>(nequo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

}

template<typename T>void test1sBinaryOps() {
	multBinaryOp<T> mbo = Functory<T,multBinaryOp>::pinch();
	std::cout << "callin binaryOpTest<T,1> with multBinaryOp (identity == " << mbo[0] << ")\n";
#ifdef CuMatrix_Enable_KTS
	binaryOpTest<T,multBinaryOp><<<dim3(1),dim3(3,3)>>>(mbo);
#else
	binaryOpTest<T,1><<<dim3(1),dim3(3,3)>>>(mbo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	plusBinaryOp<T> pbo = Functory<T,plusBinaryOp>::pinch();
	std::cout << "callin binaryOpTest<T,1> with plusBinaryOp (identity == " << pbo.identity_ro() << ")\n";
#ifdef CuMatrix_Enable_KTS
	binaryOpTest<T,plusBinaryOp><<<dim3(1),dim3(3,3)>>>(pbo);
#else
	binaryOpTest<T,1><<<dim3(1),dim3(3,3)>>>(pbo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

}



template<typename T>void test2sFillers() {
	increasingColumnsFiller<T> icf = Functory<T,increasingColumnsFiller>::pinch(10,2);
	std::cout << "callin indexFunctorTest<T,2> with increasingColumnsFiller with start " << icf[0] << " and width " << icf[1] << " cols \n";
#ifndef CuMatrix_Enable_KTS
	indexFunctorTest<T,2><<<1,10>>>(icf); // 5 rows
#else
	indexFunctorTest<T,increasingColumnsFiller><<<1,10>>>(icf); // 5 rows
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	increasingRowsFiller<T> irf = Functory<T,increasingRowsFiller>::pinch(5, 5);
	std::cout << "callin indexFunctorTest<T,2> with increasingRowsFiller with start " << irf[0]  << " and height " << irf[1] << " rows \n";
#ifndef CuMatrix_Enable_KTS
	indexFunctorTest<T,2><<<1,10>>>(irf);
#else
	indexFunctorTest<T,increasingRowsFiller><<<1,10>>>(irf);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	seqModFiller<T> smf = Functory<T,seqModFiller>::pinch(5, 5);
	std::cout << "callin indexFunctorTest<T,2> with seqModFiller with phase " << smf[0]  << " and mod " << smf[1] << "\n";
#ifndef CuMatrix_Enable_KTS
	indexFunctorTest<T,2><<<1,10>>>(smf);
#else
	indexFunctorTest<T,seqModFiller><<<1,10>>>(smf);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	diagonalFiller<T> dgf = Functory<T,diagonalFiller>::pinch(5, 10);
	std::cout << "callin indexFunctorTest<T,2> with diagonalFiller with value " << dgf.value_ro()  << " and dim " << dgf[1] << "\n";
#ifndef CuMatrix_Enable_KTS
	indexFunctorTest<T,2><<<1,dim3(10,10,1)>>>(dgf);
#else
	indexFunctorTest<T,diagonalFiller><<<1,dim3(10,10,1)>>>(dgf);
#endif
	checkCudaErrors(cudaDeviceSynchronize());
}

template<typename T>void test2sUnaryOps() {
	almostEqUnaryOp<T> aeuo = Functory<T,almostEqUnaryOp>::pinch(static_cast<T>(0),static_cast<T>(2));
	std::cout << "callin unaryOpTest<T,2> with almostEqUnaryOp " << aeuo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,almostEqUnaryOp><<<1,10>>>(aeuo);
#else
	unaryOpTest<T,2><<<1,10>>>(aeuo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

	notAlmostEqUnaryOp<T> naeuo = Functory<T,notAlmostEqUnaryOp>::pinch(static_cast<T>(0),2);
	std::cout << "callin unaryOpTest<T,2> with notAlmostEqUnaryOp " << naeuo[0] << "\n";
#ifdef CuMatrix_Enable_KTS
	unaryOpTest<T,notAlmostEqUnaryOp><<<1,10>>>(naeuo);
#else
	unaryOpTest<T,2><<<1,10>>>(naeuo);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

}

template<typename T>void test3sFillers() {
	sinFiller<T> sf = Functory<T,sinFiller>::pinch(20,3,10);
	cosFiller<T> cf = Functory<T,cosFiller>::pinch(20,3,10);

	/*
	 or more readably
	sf.phase() = 10;
	sf.amplitude() = 20;
	sf.period() =3;
	 */
	std::cout << "sf.ampl " << sf[0] << "\n";
	std::cout << "callin indexFunctorTest<T,3> with sin filler\n";
#ifdef CuMatrix_Enable_KTS
	indexFunctorTest<T,sinFiller><<<1,3>>>(sf);
	checkCudaErrors(cudaDeviceSynchronize());
	std::cout << "cf.ampl " << cf.amplitude_ro() << "\n";
#else
	indexFunctorTest<T,3><<<1,3>>>(sf);
	checkCudaErrors(cudaDeviceSynchronize());
	std::cout << "cf.ampl " << cf.amplitude_ro() << "\n";
#endif
	std::cout << "cf[1] " << cf[1] << "\n";
	std::cout << "callin indexFunctorTest<T,3> with cos filler\n";
#ifdef CuMatrix_Enable_KTS
	indexFunctorTest<T,cosFiller><<<1,3>>>(cf);
#else
	indexFunctorTest<T,3><<<1,3>>>(cf);
#endif
	checkCudaErrors(cudaDeviceSynchronize());

}

int cuFunctorMain() {
    int device;
    checkCudaErrors(cudaGetDevice(&device));

#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		unaryOpIndexMbrs<float>::setupAllFunctionTables(device);
	#else
		unaryOpIndexMbrs<float>::setupAllMethodTables(device);
	#endif
#endif
	testGets();
	test0sFillers<float>();
	test1sFillers<float>();
	test2sFillers<float>();
	test2sUnaryOps<float>();
	test3sFillers<float>();
	test0sUnaryOps<float>();
	test1sUnaryOps<float>();
	test1sBinaryOps<float>();
	return 0;
}


