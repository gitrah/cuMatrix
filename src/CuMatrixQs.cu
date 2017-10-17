/*
 * CuMatrixQs.cu
 *
 *      Author: reid
 */
#include "CuMatrix.h"



template<typename T> void __global__ indexOf2DKernel(DMatrix<T> dm, int2* idx, bool* found, almostEqUnaryOp<T> op) {
	int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
	int colIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if(dm.m > rowIdx && dm.n > colIdx && !(*found)) {
		if(op( dm.elements[ rowIdx * dm.p + colIdx])) {
			idx->x = colIdx;
			idx->y = rowIdx;
		}
	}
}

template<typename T> int2 CuMatrix<T>::indexOf2D(T val, cudaStream_t stream) {
	int2 idx;
	idx.x =-1;
	idx.y =-1;
	if( !zeroDimsQ() && ! scalarQ()) {
		//outln("indexOf2D !zeroDims");

		int2 *pdIdx;
		cherr(cudaMalloc(&pdIdx, sizeof(int2)));
		bool  hFound = false;
		bool* pFound;
		cherr(cudaMalloc(&pFound, sizeof(bool)));
		cherr(cudaMemcpy(pFound,&hFound, sizeof(bool), cudaMemcpyHostToDevice));

		almostEqUnaryOp<T> op = Functory<T,almostEqUnaryOp>::pinch(val, util<T>::epsilon());

		dim3 block( MIN(32,n), MIN(32,m) );

		dim3 grid( DIV_UP( m, block.y), DIV_UP(n, block.x));

		DMatrix<T> d_A;

		if(tiler.tileSize >= tiler.m_size) {

			tile0(d_A,lastMod==mod_host,stream);
			if (checkDebug(debugBinOp)) prlocf("this tile0\n");

			bool valid = b_util::validLaunchQ((void*)indexOf2DKernel<T>,grid, block);
			//assert(valid);
			indexOf2DKernel<<<grid,block,0, stream>>>(d_A,pdIdx, pFound, op);
			if(stream) {
				cherr(cudaStreamSynchronize(stream));
			}
			else {
				cherr(cudaDeviceSynchronize());
			}
			cudaMemcpy(&idx,pdIdx, sizeof(int2), cudaMemcpyDeviceToHost);
		}
	}
	return idx;
}

template<typename T> int CuMatrix<T>::indexOfGlolo(T val) {
	if( rowVectorQ() ) {
		//outln("indexOfGlolo !zeroDims");
		// this would work if
		//idx1DAlmostEqUnaryOp<T> op = Functory<T,idx1DAlmostEqUnaryOp>::pinch(val, util<T>::epsilon());
		bool powOf2Q = b_util::isPow2(n);
		int blocks, threads;
		getReductionExecContext(blocks, threads, n);
		idx1DblockAlmostEqUnaryOp<T> op = Functory<T,idx1DblockAlmostEqUnaryOp>::pinch(val, util<T>::epsilon(), threads);
		plusBinaryOp<T> plusOp =  Functory<T,plusBinaryOp>::pinch();
		return gloloReduce(op, plusOp, 0);
	}
	return true;
}

template<typename T> bool CuMatrix<T>::zeroQ(T epsilon) {
	if( !zeroDimsQ() ) {
		//outln("zeroQ !zeroDims");
		almostEqUnaryOp<T> op = Functory<T,almostEqUnaryOp>::pinch((T)0, epsilon);
		andBinaryOp<T> andOp =  Functory<T,andBinaryOp>::pinch();
		return gloloReduce(op, andOp, true);
	}
	return true;
}

template<typename T> __host__ __device__  bool CuMatrix<T>::biasedQ() const {
	//CuMatrix<T> col1;
	//submatrix(col1, m, 1, 0,0);
	if(checkDebug(debugCheckValid))flprintf("this sum() %d\n ", sum() );
	T* dptr = currBuffer();
	CuMatrix<T> col1 = columnMatrix(0);
	col1.syncBuffers();
#ifndef __CUDA_ARCH__
//	if(checkDebug(debugCheckValid))outln("col1  " << col1 );
#endif
	if(checkDebug(debugCheckValid))flprintf("col1 sum %d\n ",col1.sum() );
	almostEqUnaryOp<T> eqOp = Functory<T,almostEqUnaryOp>::pinch((T)1, util<T>::epsilon());
	const bool  ret = col1.all(eqOp);
#ifndef __CUDA_ARCH__
//	if(checkDebug(debugCheckValid))outln(toShortString() << " is biased " << ret);
#else
	if(checkDebug(debugCheckValid))flprintf("mat %dx%dx%d with dbuff %p is biased %d\n", m,n,p,currBuffer(),ret);
#endif
	return ret;
}


template<typename T> bool CuMatrix<T>::isBinaryCategoryMatrix() const {
#ifdef  CuMatrix_Enable_KTS
	return gloloReduce( oneOrZeroUnaryOp<T>(), andBinaryOp<T>(), true);
#else
	return gloloReduce( Functory<T,oneOrZeroUnaryOp>::pinch(), Functory<T,andBinaryOp>::pinch(), true);
#endif
}

#include "CuMatrixInster.cu"
