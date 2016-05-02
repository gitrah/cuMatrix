
/*
 * testMaths.cu
 *
 *  Created on: May 15, 2014
 *      Author: reid
 */




#include "tests.h"
#include "../util.h"
#include "../Maths.h"
#include "../CuMatrix.h"
#include "../Kernels.h"

template<typename T> __global__ void cubeKernel(T* res, T input) {
	*res = cube<T>(input);
}

template int testCubes<float>::operator()(int argc, const char **argv) const;
template int testCubes<double>::operator()(int argc, const char **argv) const;
template int testCubes<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testCubes<T>::operator()(int argc, const char **argv) const {

	uint xui = 3, *d_resui =null, h_resui =0;
	czeckerr(cudaMalloc(&d_resui,sizeof(uint)));
	cubeKernel<<<1,1>>>(d_resui, xui);
	checkCudaErrors(cudaDeviceSynchronize());
	czeckerr(cudaMemcpy(&h_resui, d_resui,sizeof(uint), cudaMemcpyDeviceToHost));
	outln("h_resui " << h_resui);
	assert(h_resui == 27);

	cubeKernel<<<1,1>>>(d_resui, 776u);
	checkCudaErrors(cudaDeviceSynchronize());
	czeckerr(cudaMemcpy(&h_resui, d_resui,sizeof(uint), cudaMemcpyDeviceToHost));
	outln("h_resui " << h_resui);
	assert(h_resui == 467288576u );


	czeckerr(cudaFree(d_resui));

	int xi = 3, *d_resi =null, h_resi =0;
	czeckerr(cudaMalloc(&d_resi,sizeof(int)));
	cubeKernel<<<1,1>>>(d_resi, xi);
	checkCudaErrors(cudaDeviceSynchronize());
	czeckerr(cudaMemcpy(&h_resi, d_resi,sizeof(int), cudaMemcpyDeviceToHost));
	outln("h_resi " << h_resi);
	assert(h_resi == 27);
	czeckerr(cudaFree(d_resi));

	long xl = 3, *d_resl =null, h_resl =0;
	czeckerr(cudaMalloc(&d_resl,sizeof(long)));
	cubeKernel<<<1,1>>>(d_resl, xl);
	checkCudaErrors(cudaGetLastError());
	czeckerr(cudaMemcpy(&h_resl, d_resl,sizeof(long), cudaMemcpyDeviceToHost));
	outln("h_resl " << h_resl);
	assert(h_resl == 27);
	czeckerr(cudaFree(d_resl));

	float xf = 3, *d_resf =null, h_resf =0;
	czeckerr(cudaMalloc(&d_resf,sizeof(float)));
	cubeKernel<<<1,1>>>(d_resf, xf);
	checkCudaErrors(cudaDeviceSynchronize());
	czeckerr(cudaMemcpy(&h_resf, d_resf,sizeof(float), cudaMemcpyDeviceToHost));
	outln("h_resf " << h_resf);
	assert(h_resf == 27);
	czeckerr(cudaFree(d_resf));

	return 0;
}


template int testNextPowerOf2<float>::operator()(int argc, const char **argv) const;
template int testNextPowerOf2<double>::operator()(int argc, const char **argv) const;
template int testNextPowerOf2<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testNextPowerOf2<T>::operator()(int argc, const char **argv) const {
	uint pow2 = 0;
	for(uint i = 0; i < 100; i++) {
		uint p2 = b_util::nextPowerOf2(i);
		if(p2 != pow2){
			outln( i << " np2: " << p2);
			pow2=p2;
		}
	}
	return 0;
}

__global__ void testLargestFactorKernel() {
#ifdef CuMatrix_Enable_Cdp
	prlocf("testLargestFactorKernel ent\n");
	flprintf("largestFactor(34513) %d\n", largestFactor(34513));
	flprintf("largestFactor(917) %d\n", largestFactor(917));
	flprintf("largestFactor(22255) %d\n", largestFactor(22255));
#else
	prlocf("not implemented for non-cdp\n");
	assert(false);
#endif
}
template int testLargestFactor<float>::operator()(int argc, const char **argv) const;
template int testLargestFactor<double>::operator()(int argc, const char **argv) const;
template int testLargestFactor<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testLargestFactor<T>::operator()(int argc, const char **argv) const {

	outln( "largestFactor(2415) " << largestFactor(2415));
	outln( "largestFactor(28) " << largestFactor(28));
	outln( "largestFactor(43543) " << largestFactor(43543));

	testLargestFactorKernel<<<1,1>>>();
	checkCudaErrors(cudaDeviceSynchronize());
	return 0;
}

__global__ void spanrowQkernel(uint* res, int row, int n) {
	if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z ==0) {
		*res = spanrowQ(row,n);
	}
}

bool spanrowKL(int row, int n) {
	uint* res;
	uint hres;
	cherr(cudaMalloc(&res, sizeof(uint)));
	spanrowQkernel<<<1,1>>>(res, row, n);
	cherr(cudaDeviceSynchronize());
	cherr(cudaMemcpy(&hres, res, sizeof(uint), cudaMemcpyDeviceToHost));
	cherr(cudaFree(res));
	return hres;
}

template int testCountSpanrows<float>::operator()(int argc, const char **argv) const;
template int testCountSpanrows<double>::operator()(int argc, const char **argv) const;
template int testCountSpanrows<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testCountSpanrows<T>::operator()(int argc, const char **argv) const {

	CuMatrix<T> n7 = CuMatrix<T>::ones(100,7);
	//  b_util::countSpanrows( int m, int n, uint warpSize )
	outln( "b_util::countSpanrows( 1, 7) " <<  b_util::countSpanrows( 1, 7));

	outln( "b_util::countSpanrows( 100, 7) " <<  b_util::countSpanrows( 100, 7));

	outln("n7\n"<< n7.syncBuffers());
	outln( "b_util::countSpanrows( 1, 18) " <<  b_util::countSpanrows( 1, 18));

	outln( "b_util::countSpanrows( 2, 18) " <<  b_util::countSpanrows( 2, 18));

	outln( "b_util::countSpanrows( 3, 18) " <<  b_util::countSpanrows( 3, 18));

	outln( "b_util::countSpanrows( 70, 18) " <<  b_util::countSpanrows( 70, 18));

	outln( "b_util::countSpanrows( 199, 18) " <<  b_util::countSpanrows( 199, 18));
	CuMatrix<T> n18 = CuMatrix<T>::ones(200,18);
	outln("n18\n"<< n18.syncBuffers());

	outln( "b_util::countSpanrows(  1, 33) " <<  b_util::countSpanrows( 1, 33));

	outln( "b_util::countSpanrows(  2, 33) " <<  b_util::countSpanrows( 2, 33));

	outln( "b_util::countSpanrows(  199, 33) " <<  b_util::countSpanrows( 199, 33));

	outln( "b_util::countSpanrows(  1, 34) " <<  b_util::countSpanrows( 1, 34));

	outln( "b_util::countSpanrows(  2, 34) " <<  b_util::countSpanrows( 2, 34));

	outln( "b_util::countSpanrows(  17, 34) " <<  b_util::countSpanrows( 17, 34));

	outln( "b_util::countSpanrows(  199, 34) " <<  b_util::countSpanrows( 199, 34));
	CuMatrix<T> n34 = CuMatrix<T>::ones(200,34);
	outln("n34\n"<< n34.syncBuffers());

	CuMatrix<T> n17 = CuMatrix<T>::ones(33,17);
	setCurrGpuDebugFlags( debugVerbose,true,false);
	outln("n17\n"<< n17.syncBuffers());
	setCurrGpuDebugFlags( ~debugVerbose,false,true);

	for(int i = 0; i < 33 ; i++) {
		if(b_util::spanrowQ(i,17) != spanrowKL(i,17)) outln( "b_util::spanrowQ(" << i << ",17) " << b_util::spanrowQ(i,17) << ", spanrowKL " << spanrowKL(i,17));
		bool truth = b_util::spanrowQ(i,17) == spanrowKL(i,17);
		assert(truth);
	}

	return 0;
}

__global__ void modK(uint* res, uint x, uint div) {
	if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z ==0) {
		*res = mod(x,div);
	}
}
uint modKL(uint x, uint div) {
	uint* res;
	uint hres;
	cherr(cudaMalloc(&res, sizeof(uint)));
	modK<<<1,1>>>(res, x, div);
	cherr(cudaDeviceSynchronize());
	cherr(cudaMemcpy(&hres, res, sizeof(uint), cudaMemcpyDeviceToHost));
	cherr(cudaFree(res));
	return hres;
}

template int testMod<float>::operator()(int argc, const char **argv) const;
template int testMod<double>::operator()(int argc, const char **argv) const;
template int testMod<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testMod<T>::operator()(int argc, const char **argv) const {

	outln("modKL(4,4) " << modKL(4,4));
	outln("modKL(4,5) " << modKL(4,5));
	outln("modKL(6,5) " << modKL(6,5));
	outln("modKL(0,5) " << modKL(0,5));
	outln("modKL(5,0) " << modKL(5,0));
	outln("modKL(9,2) " << modKL(9,2));
	outln("modKL(2,9) " << modKL(2,9));
	outln("modKL(1002,5) " << modKL(1002,5));
	assert( modKL(4,4) == 0);
	assert( modKL(4,5) == 4);
	assert( modKL(6,5) == 1);
	assert( modKL(0,5) == 0);
	assert( modKL(2,9) == 2);
	assert( modKL(9,2) == 1);
	assert( modKL(1002,5) == 2);
	return 0;
}

template int testMatRowMath<float>::operator()(int argc, const char **argv) const;
template int testMatRowMath<double>::operator()(int argc, const char **argv) const;
template int testMatRowMath<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testMatRowMath<T>::operator()(int argc, const char **argv) const {

	CuMatrix<T> src = CuMatrix<T>::increasingColumns(1,28,28);
	outln("src " << src.syncBuffers());
	CuMatrix<T> row = CuMatrix<T>::increasingColumns(1,1,28);
	CuMatrix<T> col = 2.5 * CuMatrix<T>::increasingColumns(.0001,28,1);

	CuMatrix<T> sum = src + src;
	CuMatrix<T> rowsum = row + row;
	CuMatrix<T> colsum = col + col;
	int blockSize=256;
	int maxBlocks;
#ifdef  CuMatrix_Enable_KTS
	b_util::kernelOccupancy((void*)binaryOpKernel1dNeqP<T, plusBinaryOp>, &maxBlocks, blockSize );
#else
	b_util::kernelOccupancy((void*)binaryOpKernel1dNeqP<T, 1>, &maxBlocks, blockSize );
#endif
	outln("sum\n" << sum.syncBuffers());
	outln("row\n" << row.syncBuffers());
	outln("col\n" << col.syncBuffers());
	outln("rowsum" << rowsum.syncBuffers());
	outln("colsum" << colsum.syncBuffers());
	CuMatrix<T> rsum = src + row;
	outln("rsum " << rsum.syncBuffers());
	CuMatrix<T> diff = src - row;
	outln("diff " << diff.syncBuffers());
	if(b_util::getParameter(argc,argv,"boom",0)) {
		CuMatrix<T> diffbad = row - src;
		outln("diffbad " << diffbad.syncBuffers());
	}


	return 0;
}
#include "tests.cc"
