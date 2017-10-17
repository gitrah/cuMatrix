/*
 * testamort.cc
 *
 *  Created on: Feb 16, 2014
 *      Author: reid
 */


#include "tests.h"
#include "../CuMatrix.h"
#include "../util.h"
#include "../Kernels.h"
#include <typeinfo>

template int testBinaryOpAmort<float>::operator()(int argc, const char **argv) const;
template int testBinaryOpAmort<double>::operator()(int argc, const char **argv) const;
template<typename T> int testBinaryOpAmort<T>::operator()(int argc, const char** argv) const {

	int m = 5000, n = 3000;
	int ma = 5*1024, na = 3*1024;
	CuMatrix<T> big2s = CuMatrix<T>::fill( m,n,(T)2);
	DMatrix<T> dbig2s = big2s.asDmatrix();
	CuMatrix<T> big2sa = CuMatrix<T>::fill(ma,na,(T)2);
	DMatrix<T> dbig2sa = big2sa.asDmatrix();
	CuMatrix<T> big3s = CuMatrix<T>::fill( m,n,(T)3);
	DMatrix<T> dbig3s = big3s.asDmatrix();
	CuMatrix<T> big3sa = CuMatrix<T>::fill(ma,na,(T)3);
	DMatrix<T> dbig3sa = big3sa.asDmatrix();
	CuMatrix<T> res = CuMatrix<T>::zeros(m, n);
	DMatrix<T> dres = res.asDmatrix();
	CuMatrix<T> resa = CuMatrix<T>::zeros(ma, na);
	DMatrix<T> dresa = resa.asDmatrix();

	//binaryOpDmL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOp op, int w2h, cudaStream_t stream)

	multBinaryOp<T> mult;
	//
	int count = b_util::getCount(argc,argv,1);
	int h2w[] = {0, 1, 2, 4, 8, 16};
	uint* misses;
	checkCudaError(cudaMalloc(&misses,sizeof(uint)));
	uint missCount;
	for(int h2wi = 0; h2wi < 6;h2wi++) {
		CuTimer timer;
		timer.start();
		for(int i =0; i < count;i++) {
			 //big2s.binaryOp(res, big3s,mult);
			if(h2wi == 0) {
				binaryOpL(dres,dbig2s,dbig3s,mult);
			}else {
				binaryOpL2(dres, dbig2s, dbig3s, mult, h2w[h2wi],misses);
				checkCudaError(cudaDeviceSynchronize());
				checkCudaError(cudaMemcpy(&missCount,misses,sizeof(uint), cudaMemcpyDeviceToHost));
				printf("misses %u\n",missCount);
				missCount = 0;
				checkCudaError(cudaMemcpy(misses,&missCount,sizeof(uint), cudaMemcpyHostToDevice));
			}
		}
		float binoptime = timer.stop();
		outln(count << " of " <<  b_util::unmangl(typeid(mult).name()) << " with w2h = " << h2w[h2wi] << " took " << binoptime );
		res.invalidateHost();
		outln("res " << res.syncBuffers());
		res.setAll(0);
	}
	for(int h2wi = 0; h2wi < 6;h2wi++) {
		CuTimer timer;
		timer.start();
		for(int i =0; i < count;i++) {
			 //big2s.binaryOp(res, big3s,mult);
			if(h2wi == 0) {
				binaryOpL(dresa,dbig2sa,dbig3sa,mult);
			}else {
				binaryOpL2(dresa, dbig2sa, dbig3sa, mult, h2w[h2wi],misses);
				checkCudaError(cudaDeviceSynchronize());
				checkCudaError(cudaMemcpy(&missCount,misses,sizeof(uint), cudaMemcpyDeviceToHost));
				printf("misses %u\n",missCount);
				missCount = 0;
				checkCudaError(cudaMemcpy(misses,&missCount,sizeof(uint), cudaMemcpyHostToDevice));
			}
		}
		float binoptime = timer.stop();
		outln(count << " of aligned " <<  b_util::unmangl(typeid(mult).name()) << " with w2h = " << h2w[h2wi] << " took " << binoptime );
		resa.invalidateHost();
		outln("resa " << resa.syncBuffers());
		resa.setAll(0);
	}
	checkCudaError(cudaFree(misses));

/*
	for(int i =0; i < count;i++) {
		 big2s.binaryOpDm(res, big3s,mult);
	}
*/

	return 0;
}
