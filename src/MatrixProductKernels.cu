/*
 * MatrixProductKernels.cu
 *
 *  Created on: Oct 20, 2013
 *      Author: reid
 */

#include "Kernels.h"
#include "util.h"
#include "caps.h"
#include <typeinfo>
//#include <sys/types.h>
//#include <typeinfo.h>

long biggestMult = 0;


template<typename T> __host__ CUDART_DEVICE void matrixProductL(DMatrix<T>& d_res,
		const DMatrix<T>& d_A, const DMatrix<T>& d_B, dim3* block,cudaStream_t stream) {
	cherr(cudaPeekAtLastError());
#ifndef __CUDA_ARCH__
	static CuMethodTimer mTimer((void(*)())matrixProductKernel4<T>);
#endif
	if(d_A.n != d_B.m) {
		flprintf("d_A.n %u != d_B.m %u\n",d_A.n, d_B.m);
		flprintf("d_A.m %u, d_B.n %u\n",d_A.m, d_B.n);
		flprintf("d_A.elm %p, d_B.elements %p\n",d_A.elements, d_B.elements);
		setLastError(matricesOfIncompatibleShapeEx);
	}
	if(d_res.m != d_A.m) {
		setLastError(rowDimsDisagreeEx);
	}
	if(d_res.n != d_B.n) {
		doutln(columnDimsDisagreeEx);
	}
	if (checkDebug(debugMatProd) && block) {
		prlocf("blockSize ");
		b_util::prd3(*block);
	}
	// block dim (bases?) should be warp-divisible
	// but constrained by smem
	if (block == null) {
#ifndef __CUDA_ARCH__
		block = &gDefaultMatProdBlock;
#else
		int currDev;
		cudaGetDevice(&currDev);
		dim3 bk(dgDefaultMatProdBlock);
		block = &bk;
#endif
	}
	int columnBlocks = DIV_UP( d_A.n, block->x); //(d_A.n + block->x - 1) /block->x;
	int rowBlocks = DIV_UP(d_B.m, block->y ); //d_B.m + block->y - 1) / block->y;

	const int stepsPerResultBlock = MAX(columnBlocks,rowBlocks);
	if (checkDebug(debugMatProd)) {
		flprintf("stepsPerResultBlock %d\n", stepsPerResultBlock);
	}

	ExecCaps* caps = ExecCaps::currCaps();
	if(checkDebug(debugMatProd)) {
		caps->printMaxDims(__FILE__ "::matrixProductL");
		flprintf("div_up(%d,%d) columnBlocks %d %d\n",  d_A.n, block->x, columnBlocks);
		flprintf("div_up(%d,%d) rowBlocks %d\n", d_B.m, block->y, rowBlocks);
		if(!caps->okBlock(*block)) {
			prlocf("block exceeds max ");
			b_util::prd3(caps->maxBlock);
		}
	}
	dim3 resultGrid(DIV_UP(d_res.n,block->x), DIV_UP(d_res.m, block->y));
	if (checkDebug(debugMatProd)) {
		prlocf("resultGrid ");
		b_util::prd3(resultGrid);
		if(!caps->okGrid(resultGrid)) {
			prlocf("resultGrid exceeds max ");
			b_util::prd3(caps->maxGrid);
		}
	}
	uint smem = block->x * block->y * sizeof(T) * 2;
	if (smem > ExecCaps::currCaps()->memSharedPerBlock) {
		setLastError(smemExceededEx);
	}
	if (checkDebug(debugMatProd)) {
		flprintf( "launching matPrdKrnl resultGrid  smem %d " , smem);
		b_util::prd3(resultGrid);
		prlocf("block ");
		b_util::prd3(*block);
	}

#ifndef __CUDA_ARCH__
	long nextBiggest = MAX(biggestMult,stepsPerResultBlock * block->x );
	if (checkDebug(debugMatProd) && nextBiggest > biggestMult) {
		flprintf("steps %d X blockDim.x %d--wide reduction == %d\n", stepsPerResultBlock, block->x, nextBiggest);
		outln(util<T>::pdm(d_A) << " X b " << util<T>::pdm(d_B));
		biggestMult = nextBiggest;
	}
#endif

#ifndef __CUDA_ARCH__
	mTimer.enter();
#endif
	matrixProductKernel4<<<resultGrid, *block, smem>>>(d_res, d_A, d_B , stepsPerResultBlock);
#ifndef __CUDA_ARCH__
		checkCudaError(cudaDeviceSynchronize());
		mTimer.exit();
#else
		__syncthreads();
#endif
	cherr(cudaPeekAtLastError());
}


template __host__ CUDART_DEVICE void matrixProductL<float>(DMatrix<float>&, DMatrix<float> const&, DMatrix<float> const&, dim3*, cudaStream_t);
template __host__ CUDART_DEVICE void matrixProductL<double>(DMatrix<double>&, DMatrix<double> const&, DMatrix<double> const&, dim3*, cudaStream_t);
template __host__ CUDART_DEVICE void matrixProductL<long>(DMatrix<long>&, DMatrix<long> const&, DMatrix<long> const&, dim3*, cudaStream_t);
template __host__ CUDART_DEVICE void matrixProductL<ulong>(DMatrix<ulong>&, DMatrix<ulong> const&, DMatrix<ulong> const&, dim3*, cudaStream_t);
template __host__ CUDART_DEVICE void matrixProductL<int>(DMatrix<int>&, DMatrix<int> const&, DMatrix<int> const&, dim3*, cudaStream_t);
template __host__ CUDART_DEVICE void matrixProductL<uint>(DMatrix<uint>&, DMatrix<uint> const&, DMatrix<uint> const&, dim3*, cudaStream_t);

template <typename T> __host__ __device__ const char* matProdKernelName(void (*matProdKptr) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int)){

/*
	flprintf("matProdKptr %p\n", matProdKptr);
	flprintf("matrixProductKernelTxdB<T> %p\n", matrixProductKernelTxdB<T>);
	flprintf("matrixProductKernelTxdB2<T> %p\n", matrixProductKernelTxdB2<T>);
	flprintf("matrixProductBandwidthKernel<T> %p\n", matrixProductBandwidthKernel<T>);
	flprintf("matrixProductKernel<T> %p\n", matrixProductKernel<T>);
    flprintf("matrixProductKernel2<T> %p\n", matrixProductKernel2<T>);
	flprintf("matrixProductReductionTxdBKernel<T> %p\n", matrixProductReductionTxdBKernel<T>);
*/
	cudaFuncAttributes fatts;
	cudaFuncGetAttributes(&fatts,matProdKptr);
	static char buff[1024];

	if( matProdKptr ==  (void (*)(DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int)) matrixProductKernelTxdB<T>) {
		sprintf(buff, "matrixProductKernelTxdB(regs=%d, shared %lu local %lu maxtpb %d )", fatts.numRegs,fatts.sharedSizeBytes, fatts.localSizeBytes, fatts.maxThreadsPerBlock);
		return buff;
	}else if(matProdKptr == (void (*)(DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int)) matrixProductKernelTxdB2<T>){
		sprintf(buff, "matrixProductKernelTxdB2(regs=%d, shared %lu local %lu maxtpb %d )", fatts.numRegs,fatts.sharedSizeBytes, fatts.localSizeBytes, fatts.maxThreadsPerBlock);
		return buff;
	}else if(matProdKptr == (void (*)(DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int)) matrixProductBandwidthKernel<T>){
		sprintf(buff, "matrixProductBandwidthKernel(regs=%d, shared %lu local %lu maxtpb %d )", fatts.numRegs,fatts.sharedSizeBytes, fatts.localSizeBytes, fatts.maxThreadsPerBlock);
		return buff;
	}else if(matProdKptr == (void (*)(DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int))matrixProductKernel<T>){
		sprintf(buff, "matrixProductKernel(regs=%d, shared %lu local %lu maxtpb %d )", fatts.numRegs,fatts.sharedSizeBytes, fatts.localSizeBytes, fatts.maxThreadsPerBlock);
		return buff;
	}else if(matProdKptr == (void (*)(DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int))matrixProductKernel2<T>){
		sprintf(buff, "matrixProductKernel2(regs=%d, shared %lu local %lu maxtpb %d )", fatts.numRegs,fatts.sharedSizeBytes, fatts.localSizeBytes, fatts.maxThreadsPerBlock);
		return buff;
	}else if(matProdKptr == (void (*)(DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int))matrixProductKernel3<T>){
		sprintf(buff, "matrixProductKernel3(regs=%d, shared %lu local %lu maxtpb %d )", fatts.numRegs,fatts.sharedSizeBytes, fatts.localSizeBytes, fatts.maxThreadsPerBlock);
		return buff;
	}
#ifdef CuMatrix_Enable_Cdp
	else if(matProdKptr == (void (*)(DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int))matrixProductReductionTxdBKernel<T>){
		sprintf(buff, "matrixProductReductionTxdBKernel(regs=%d, shared %lu local %lu maxtpb %d )", fatts.numRegs,fatts.sharedSizeBytes, fatts.localSizeBytes, fatts.maxThreadsPerBlock);
		return buff;
	}
#endif
	return "dunno";
}
template<typename T> __host__ CUDART_DEVICE void matrixProductKPtrL(DMatrix<T>& d_res,
		void (*matProdKptr) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int),
		 const DMatrix<T>& d_A, const DMatrix<T>& d_B,
		 dim3* block,cudaStream_t stream) {
	// (que& (que::*)())
	bool isTxdB = (
			matProdKptr == ( void (*)(DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int)) &matrixProductKernelTxdB<T>
					|| matProdKptr == (void (*)(DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int)) &matrixProductKernelTxdB2<T>);
	if (checkDebug(debugMatProd)  ) {
#ifndef __CUDA_ARCH__
		flprintf("matrixProductKPtrL(..) kernel %s\n" , matProdKernelName(matProdKptr));
#endif
		flprintf("matrixProductKPtrL(..) isTxdB %s\n" , tOrF(isTxdB));
	}
	if((isTxdB && (d_A.n != d_B.n)) || (!isTxdB && (d_A.n != d_B.m)) ) {
		if(isTxdB) {
			flprintf("matrixProductKPtrL(..) d_A.n %d != d_B.n %d\n" ,d_A.n, d_B.n);
		} else {
			flprintf("matrixProductKPtrL(..) txd B d_A.n %d != d_B.n %d\n" ,d_A.n, d_B.m);
		}
		setLastError(matricesOfIncompatibleShapeEx);
		return;
	}
	if(d_res.m != d_A.m) {
		setLastError(rowDimsDisagreeEx);
		return;
	}
	if((isTxdB && (d_res.n != d_B.m)) || (!isTxdB && (d_res.n != d_B.n))) {
		if(isTxdB) {
			flprintf("matrixProductKPtrL(..) d_res.n %d != d_B.m %d\n", d_res.n ,d_B.m);
		} else {
			flprintf("matrixProductKPtrL(..) d_res.n %d != d_B.n %d\n", d_res.n,d_B.n);
		}
		setLastError(columnDimsDisagreeEx);
		return;
	}

	if(block) {
		if (checkDebug(debugMatProd) ) {
			prlocf("matrixProductKPtrL(..) thread block ");
			b_util::prd3(*block);
		}

		if(checkDebug(debugMatProdBlockResize)) {
			cudaFuncAttributes fatts;
			cudaFuncGetAttributes(&fatts,matProdKptr);
			uint blockRegs = block->x* block->y * fatts.numRegs;
			ExecCaps* curr = ExecCaps::currCaps();
			uint capRegsBlock = curr->regsPerBlock;
			if(blockRegs > capRegsBlock) {
				flprintf("dev caps of %d regs per blocks exceeded by kernel's %d threads/block\n",capRegsBlock, blockRegs);
				if(block->x == 32 && block->y == 32) {
					block->x = block->y = 16;
					prlocf("matrixProductKPtrL(..) shrunk block to 16,16\n");
				} else {
					assert(false);
				}
			}
		}
	}
	// block dim (bases?) should be warp-divisible
	// but constrained by smem
	dim3 bk;
	if (block == null) {
#ifndef __CUDA_ARCH__
		block = &gDefaultMatProdBlock;
#else
		int currDev;
		bk = dgDefaultMatProdBlock;
		cudaGetDevice(&currDev);
		block = &bk;
#endif
	}
	int columnBlocks = DIV_UP( d_A.n, block->x); //(d_A.n + block->x - 1) /block->x;
	int rowBlocks = DIV_UP(isTxdB ? d_B.n : d_B.m, block->y ); //(transposed B)

	const int steps = MAX(columnBlocks,rowBlocks); // add extra step when A or B not of integral size
	if(checkDebug(debugMatProd)) {
		flprintf("matrixProductKPtrL(..) columnBlocks %d rowBlocks %d\n" ,columnBlocks ,rowBlocks);
	}

	dim3 resultGrid(DIV_UP(d_res.n,block->x), DIV_UP(d_res.m, block->y));
	if (checkDebug(debugMatProd)) {
		prlocf("matrixProductKPtrL(..) resultGrid ");
		b_util::prd3(resultGrid);
	}

	uint smem = block->x * block->y * sizeof(T) * 2;
	if (smem > ExecCaps::currCaps()->memSharedPerBlock) {
		setLastError(smemExceededEx);
		return;
	}

	if (checkDebug(debugMatProd) ) {
		prlocf("matrixProductKPtrL(..) A pre kernel is ");
		util<T>::prdmln(d_A);
	}

	if (checkDebug(debugMatProd) )  {
		prlocf("matrixProductKPtrL(..) B pre kernel is " );
		util<T>::prdmln(d_B);
	}

	if (checkDebug(debugMatProd)) {
		flprintf( "matrixProductKPtrL(..) launching kernel at %p on resultGrid ",(void*)matProdKptr );
		b_util::prd3(resultGrid);
		prlocf("block " );
		b_util::prd3(*block);
		flprintf("smem %d steps %d\n",smem,steps);
		prlocf("d_A " );
		util<T>::printRow(d_A);
		prlocf("d_B " );
		util<T>::printRow(d_B);

		prlocf("pre d_res " );
		util<T>::printRow(d_res);
	}

	assert(b_util::validLaunchQ((void*)matProdKptr,resultGrid,*block));

	// 'an indirect call with a launch configuration from a __host__ __device__ function("matrixProductKPtrL") is only allowed on the compute_35 architecture or above'
	(*matProdKptr)<<<resultGrid, *block, smem>>>(d_res, d_A, d_B , steps);

#ifndef __CUDA_ARCH__
		checkCudaError(cudaDeviceSynchronize());
#else
		__syncthreads();
#endif
	if (checkDebug(debugMatProd)) {
		prlocf("d_res " );
		util<T>::printRow(d_res);
	}
}
template void matrixProductKPtrL<float>(DMatrix<float>&, void (*)(DMatrix<float>, DMatrix<float>, DMatrix<float>, int), DMatrix<float> const&, DMatrix<float> const&, dim3*, cudaStream_t);
template void matrixProductKPtrL<double>(DMatrix<double>&, void (*)(DMatrix<double>, DMatrix<double>, DMatrix<double>, int), DMatrix<double> const&, DMatrix<double> const&, dim3*, cudaStream_t);
template void matrixProductKPtrL<long>(DMatrix<long>&, void (*)(DMatrix<long>, DMatrix<long>, DMatrix<long>, int), DMatrix<long> const&, DMatrix<long> const&, dim3*, cudaStream_t);
template void matrixProductKPtrL<ulong>(DMatrix<ulong>&, void (*)(DMatrix<ulong>, DMatrix<ulong>, DMatrix<ulong>, int), DMatrix<ulong> const&, DMatrix<ulong> const&, dim3*, cudaStream_t);
template void matrixProductKPtrL<uint>(DMatrix<uint>&, void (*)(DMatrix<uint>, DMatrix<uint>, DMatrix<uint>, int), DMatrix<uint> const&, DMatrix<uint> const&, dim3*, cudaStream_t);
template void matrixProductKPtrL<int>(DMatrix<int>&, void (*)(DMatrix<int>, DMatrix<int>, DMatrix<int>, int), DMatrix<int> const&, DMatrix<int> const&, dim3*, cudaStream_t);
