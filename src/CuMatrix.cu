#include "CuMatrix.h"
#include "caps.h"
#include "debug.h"

template <typename T> int CuMatrix<T>::MaxThreads = 512;
template <typename T> int CuMatrix<T>::MaxBlocks = 128;


template<typename T> void CuMatrix<T>::getReductionExecContext(int &blocks, int &threads, ulong nP,
		int maxBlocks, int maxThreads) {

	threads =  nP == 2 ? 1 : (nP < (ulong) maxThreads * 2) ? b_util::nextPowerOf2((nP + 1) / 2) : maxThreads;
	blocks = (nP + (threads * 2 - 1)) / (threads * 2);

	blocks = MIN(maxBlocks, blocks);
	if (debugExec) {
			char buff[5];
			T blockOcc = threads* 1. / caps.thrdPerBlock;
			sprintf(buff,"%1.3g",blockOcc);
			ot("nP " << nP << ", threads " << threads << "(" << buff << ")");
			T globOcc = threads*blocks *1./ (caps.totalThreads());
			sprintf(buff,"%1.3g",globOcc);
			outln(", blocks " << blocks  << "(" << buff << ")" );
	}
}

template<typename T, typename UnaryOp> __global__ void unaryOp1dKernel(
		T* trg, const T* src, UnaryOp op, ulong len) {
	ulong i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		trg[i] = op(src[i]);
	}
}

template<typename T, typename UnaryOp> __global__ void unaryOpDmKernel(
		DMatrix<T> trg, const DMatrix<T> src, UnaryOp op ) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.x + threadIdx.y;
	uint srcOff = y * src.p + x;
	uint trgOff = y * trg.p + x;
	for(int i = 0; i < blockDim.x; i+=blockDim.y) {
		if(x < src.n && y + i < src.m) {
			trg.elements[trgOff + i * trg.p] = op(src.elements[srcOff + i * src.p]);
		}
	}
}

template<typename T> template<typename UnaryOp> void CuMatrix<T>::unaryOpL(DMatrix<T>& trg, const DMatrix<T>& src, UnaryOp op) {
	uint threads = 256;
	uint len = src.m * src.n;
	dim3 dBlocks, dThreads;
	b_util::execContext(threads, len, dBlocks, dThreads);
	unaryOp1dKernel<<<dBlocks,dThreads>>>(trg.elements, src.elements, op, len);
	cudaError_t err = syncHappy ? cudaDeviceSynchronize() : cudaSuccess;
	if(err != cudaSuccess) {
		outln("failed with threads " << threads << " len " << len << " dBlocks " <<
				b_util::pd3(dBlocks).c_str() << " dThreads " << b_util::pd3(dThreads) << " src " << (util<T>::pdm(src)).c_str()  << ", trg " << (util<T>::pdm(trg)).c_str());
	}
	checkCudaError(err);
}
template void CuMatrix<float>::unaryOpL<expUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, expUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<expUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, expUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<translationUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, translationUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<translationUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, translationUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<scaleUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, scaleUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<scaleUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, scaleUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<subFromUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, subFromUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<subFromUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, subFromUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<negateUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, negateUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<negateUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, negateUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<sigmoidUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, sigmoidUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<sigmoidUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, sigmoidUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<sigmoidGradientUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, sigmoidGradientUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<sigmoidGradientUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, sigmoidGradientUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<logUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, logUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<logUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, logUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<oneOverUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, oneOverUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<oneOverUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, oneOverUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<sqrtUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, sqrtUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<sqrtUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, sqrtUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<sqrUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, sqrUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<sqrUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, sqrUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<powUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, powUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<powUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, powUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<divSqrtUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, divSqrtUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<divSqrtUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, divSqrtUnaryOp<double>);

template void CuMatrix<float>::unaryOpL<ltUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, ltUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<ltUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, ltUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<lteUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, lteUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<lteUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, lteUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<gtUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, gtUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<gtUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, gtUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<gteUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, gteUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<gteUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, gteUnaryOp<double>);
template void CuMatrix<float>::unaryOpL<eqUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, eqUnaryOp<float>);
template void CuMatrix<double>::unaryOpL<eqUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, eqUnaryOp<double>);

#define UNOP_BLOCK_SIZE 	32
#define UNOP_X2Y			4

template<typename T> template<typename UnaryOp> void CuMatrix<T>::unaryOpDmL(DMatrix<T>& trg, const DMatrix<T>& src, UnaryOp op, int w2h) {
	dassert( trg.m >= src.m && trg.n >= src.n );
	int blockW = UNOP_BLOCK_SIZE;
	dim3 block(blockW,blockW/w2h);
    dim3 grid(b_util::enough(blockW,src.n), b_util::enough(blockW, src.m));
    if(debugExec)outln("unaryOpDmL grid " << b_util::pd3(grid) << " of block " << b_util::pd3(block));
    unaryOpDmKernel<<<grid,block>>>(trg, src, op);
}
template void CuMatrix<float>::unaryOpDmL<expUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, expUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<expUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, expUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<translationUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, translationUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<translationUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, translationUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<scaleUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, scaleUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<scaleUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, scaleUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<subFromUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, subFromUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<subFromUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, subFromUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<negateUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, negateUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<negateUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, negateUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<sigmoidUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, sigmoidUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<sigmoidUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, sigmoidUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<sigmoidGradientUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, sigmoidGradientUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<sigmoidGradientUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, sigmoidGradientUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<logUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, logUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<logUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, logUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<oneOverUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, oneOverUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<oneOverUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, oneOverUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<sqrtUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, sqrtUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<sqrtUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, sqrtUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<sqrUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, sqrUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<sqrUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, sqrUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<powUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, powUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<powUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, powUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<divSqrtUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, divSqrtUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<divSqrtUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, divSqrtUnaryOp<double>,int);

template void CuMatrix<float>::unaryOpDmL<ltUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, ltUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<ltUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, ltUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<lteUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, lteUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<lteUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, lteUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<gtUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, gtUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<gtUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, gtUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<gteUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, gteUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<gteUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, gteUnaryOp<double>,int);
template void CuMatrix<float>::unaryOpDmL<eqUnaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, eqUnaryOp<float>,int);
template void CuMatrix<double>::unaryOpDmL<eqUnaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, eqUnaryOp<double>,int);


template<typename T> bool CuMatrix<T>::equalDims(const CuMatrix<T>& other) const {
	return m == other.m && n == other.n;
}


template<typename T, typename BinaryOp> __global__ void binaryOpKernel(
		T* trg, const T* src1, const T* src2, BinaryOp op,
		ulong len) {
	ulong i = blockIdx.x * blockDim.x + threadIdx.x + threadIdx.y * blockDim.x*blockDim.y;
	if (i < len) {
		trg[i] = op(src1[i], src2[i]);
	}
}
template<typename T, typename BinaryOp> __global__ void binaryOpDmKernel(
		DMatrix<T> trg, const DMatrix<T> src1, const DMatrix<T> src2, BinaryOp op ) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.x + threadIdx.y;
	uint src1Off = y * src1.p + x;
	uint src2Off = y * src2.p + x;
	uint trgOff = y * trg.p + x;
	for(int i = 0; i < blockDim.x; i+=blockDim.y) {
		if(x < src1.n && y + i < src1.m && x < src2.n && y + i < src2.m) {
			trg.elements[trgOff + i * trg.p] = op(src1.elements[src1Off + i * src1.p],src2.elements[src2Off + i * src2.p]);
		}
	}
}

template<typename T> template<typename BinaryOp> void CuMatrix<T>::binaryOpL(
		DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOp op) {
	uint threads = 256;
	uint len = src1.m * src2.n;
	dim3 dBlocks, dThreads;
	b_util::execContext(threads, len, dBlocks, dThreads);
	binaryOpKernel<<<dBlocks,dThreads>>>(trg.elements, src1.elements, src2.elements, op, len);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}
template void CuMatrix<float>::binaryOpL<minusBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, const DMatrix<float>&, minusBinaryOp<float>);
template void CuMatrix<double>::binaryOpL<minusBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, const DMatrix<double>&, minusBinaryOp<double>);
template void CuMatrix<float>::binaryOpL<plusBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, const DMatrix<float>&, plusBinaryOp<float>);
template void CuMatrix<double>::binaryOpL<plusBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, const DMatrix<double>&, plusBinaryOp<double>);
template void CuMatrix<float>::binaryOpL<multBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, const DMatrix<float>&, multBinaryOp<float>);
template void CuMatrix<double>::binaryOpL<multBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, const DMatrix<double>&, multBinaryOp<double>);
template void CuMatrix<float>::binaryOpL<quotientBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, const DMatrix<float>&, quotientBinaryOp<float>);
template void CuMatrix<double>::binaryOpL<quotientBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, const DMatrix<double>&, quotientBinaryOp<double>);
template void CuMatrix<float>::binaryOpL<andBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, const DMatrix<float>&, andBinaryOp<float>);
template void CuMatrix<double>::binaryOpL<andBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, const DMatrix<double>&, andBinaryOp<double>);

template<typename T> template<typename BinaryOp> void CuMatrix<T>::binaryOpDmL(
		DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2, BinaryOp op, int w2h) {
	dassert( trg.m >= MIN( src1.m,src2.m) && trg.n >= MIN(src1.n,src2.n) );
	int blockW = UNOP_BLOCK_SIZE;
	dim3 block(blockW,blockW/w2h);
    dim3 grid(b_util::enough(blockW,MIN(src1.n,src2.n)), b_util::enough(blockW, MIN(src1.m,src2.m)));
    outln("grid " << b_util::pd3(grid) << " of block " << b_util::pd3(block));
	binaryOpDmKernel<<<grid,block>>>(trg, src1, src2, op);
}

template void CuMatrix<float>::binaryOpDmL<minusBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, const DMatrix<float>&, minusBinaryOp<float>,int);
template void CuMatrix<double>::binaryOpDmL<minusBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, const DMatrix<double>&, minusBinaryOp<double>,int);
template void CuMatrix<float>::binaryOpDmL<plusBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, const DMatrix<float>&, plusBinaryOp<float>,int);
template void CuMatrix<double>::binaryOpDmL<plusBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, const DMatrix<double>&, plusBinaryOp<double>,int);
template void CuMatrix<float>::binaryOpDmL<multBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, const DMatrix<float>&, multBinaryOp<float>,int);
template void CuMatrix<double>::binaryOpDmL<multBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, const DMatrix<double>&, multBinaryOp<double>,int);
template void CuMatrix<float>::binaryOpDmL<quotientBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, const DMatrix<float>&, quotientBinaryOp<float>,int);
template void CuMatrix<double>::binaryOpDmL<quotientBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, const DMatrix<double>&, quotientBinaryOp<double>,int);
template void CuMatrix<float>::binaryOpDmL<andBinaryOp<float> >(DMatrix<float>&, const DMatrix<float>&, const DMatrix<float>&, andBinaryOp<float>,int);
template void CuMatrix<double>::binaryOpDmL<andBinaryOp<double> >(DMatrix<double>&, const DMatrix<double>&, const DMatrix<double>&, andBinaryOp<double>,int);

// TODO implement column and row-wise reductions and replace these
template<typename T> __global__ void varianceMusKernel(DMatrix<T> sigmas, const DMatrix<T> x,
		const DMatrix<T> mus) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	T curr;
	T sqrDiff = 0;
	if (i < x.n) {
		const T currMu = mus.elements[i];
		//printf("i %d -> avg %f\n", i, currMu);
		for (uint row = 0; row < x.m; row++) {
			curr = x.elements[row * x.p + i] - currMu;
			sqrDiff += curr * curr;
		}
		sigmas.elements[i] = sqrDiff / x.m;
	}
}

template<typename T> __global__ void varianceKernel(DMatrix<T> sigmas, DMatrix<T> mus, const DMatrix<T> x ) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	T avgSum = 0;
	T sqrDiff = 0;
	T curr;
	uint row;
	// compute column (feature) averages
	if (i < x.n) {
		avgSum = x.elements[i];
		for (row = 1; row < x.m; row++) {
			curr = x.elements[row * x.p + i];
			avgSum += curr;
		}
		avgSum /= x.m; // now column avg
		mus.elements[i] = avgSum;
		//printf("i %d -> avg %f\n", i, avgSum);
		for (row = 0; row < x.m; row++) {
			curr = x.elements[row * x.p + i] - avgSum;
			sqrDiff += curr * curr;
		}
		sigmas.elements[i] = sqrDiff / x.m;
	}
}


template<typename T> __global__ void columnMatrixKernel(DMatrix<T> column, const DMatrix<T> x,
		int col) {
	uint row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < x.m) {
		column.elements[row] = x.elements[row * x.p + col];
	}
}

template<typename T> void CuMatrix<T>::columnMatrixL(DMatrix<T>& d_column, const DMatrix<T>& d_x,
		 int col) {
	dim3 block(MIN(512, b_util::nextPowerOf2(d_x.m)));
	dim3 grid(b_util::enough(block.x, d_x.m));
	columnMatrixKernel<<<grid,block>>>(d_column, d_x, col );
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}

template<typename T> void CuMatrix<T>::varianceL( DMatrix<T>& d_Sigmas, const DMatrix<T>& d_X,
		const DMatrix<T>& d_Mus) {
	dim3 block(MIN(512, b_util::nextPowerOf2(d_X.n)));
	dim3 grid(b_util::enough(block.x, d_X.n));
	outln("varianceL(&,const&,const&) for " << util<T>::pdm(d_X) << " with mus " << util<T>::pdm(d_Mus) << " have exctx grid " << b_util::pd3(grid) << " or blk " <<  b_util::pd3(block));
	varianceMusKernel<<<grid,block>>>(d_Sigmas,d_X,d_Mus);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}

template<typename T> void CuMatrix<T>::varianceAndMeanL(DMatrix<T>& d_Sigmas,  DMatrix<T>& d_Mus, const DMatrix<T>& d_X) {
	dim3 block(MIN(512, b_util::nextPowerOf2(d_X.n)));
	dim3 grid(b_util::enough(block.x, d_X.n));
	outln("varianceL(&,&,const&) for " << util<T>::pdm(d_X) << " have exctx grid " << b_util::pd3(grid) << " or blk " <<  b_util::pd3(block));
	varianceKernel<<<grid,block>>>(d_Sigmas, d_Mus, d_X);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}

/*
 * TODO re-implement
 *
 */

template<typename T> __global__ void featureAvgKernel(DMatrix<T> means, const DMatrix<T> x) {
	uint tid = threadIdx.x; // index into smem for block
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	T* sdata = SharedMemory<T>();
	// have to zero sm first?
	if (i < x.n) {
		sdata[tid] = x.elements[i];
	}

	if (i < x.n) {
		for (uint row = 1; row < x.m; row++) {
				sdata[tid] += x.elements[row * x.p + i];
		}
	}
	//__syncthreads();
	if (tid < x.n) {
		sdata[tid] /= static_cast<T>(x.m);
	}
	//__syncthreads();
	if (i < x.n) {
		means.elements[i] = sdata[tid];
	}
}

template<typename T> __global__ void featureAvgKernelLv(DMatrix<T> means, const DMatrix<T> x) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	// have to zero sm first?
	T t = (i < x.n) ? x.elements[i] : 0;

	for (uint row = 1; row < x.m; row++) {
		if (i < x.n) {
			t += x.elements[row * x.p + i];
		}
	}
	if (i < x.n) {
		means.elements[i] = t / x.m;
	}
}

/*
 * TODO re-implement with column major mats that launch reducers for each column vector
 * 	(1 launch of nCols cores vs  (sum  (i 1-x) (x : m*n/2^i > 0)  cores over i launches)
 * 	need a way of dynamically selecting not just exec context but also among different algorithm -- ie learn which algorithm
 * 	to apply
 */


template<typename T> void CuMatrix<T>::featureAvgKernelL(DMatrix<T>& d_means, const DMatrix<T>& d_x, bool localVar) {
	uint threads = 512;
	dim3 dBlocks, dThreads;
	uint smem = d_x.n * sizeof(T);
	if (smem > caps.memSharedPerBlock) {
		dthrow (  new smemExceeded);
	}
	b_util::execContext(threads, d_x.n, dBlocks, dThreads);
	//outln("for " << nRows << "*" << nCols << ", have " << dBlocks.x << " blocks of " << dThreads.x << " threads with " << smem << " smem");
	//outln("rows " << d_x.m);
	if(syncHappy)checkCudaError(cudaGetLastError());
	if (localVar) {
		featureAvgKernelLv<<<dBlocks, dThreads>>>(d_means, d_x);
	} else {
		featureAvgKernel<<<dBlocks, dThreads, smem>>>(d_means, d_x);
	}
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}

/*
 * expects a nonsqr block and a grid of (sqr) blocks to cover x (and pdens)
 * x is input sampleCount*featureCount
 * pdens is sampleCount*featureProbability
 */
template<typename T> __global__ void multivariateGaussianFeaturesKernel(
		DMatrix<T> d_pdens, const DMatrix<T> d_x, const DMatrix<T> d_sigmaSquared, const DMatrix<T> d_mu) {
	uint col = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	uint rowStart = blockIdx.y * blockDim.x + threadIdx.y; //
	uint rowsRemaining = MIN(d_x.m - rowStart, blockDim.x);
	uint xRowOff,pdensRowOff;
	T x_n, sigmaSquared_n, mu_n;
	if(col < d_x.n && rowStart < d_x.m) {
		sigmaSquared_n = d_sigmaSquared.elements[col];
		mu_n = d_mu.elements[col];
		xRowOff = rowStart * d_x.p;
		pdensRowOff = rowStart * d_pdens.p;
		for (uint row = 0; row < rowsRemaining; row += blockDim.y) {
			x_n = d_x.elements[xRowOff + row * d_x.p + col];
			d_pdens.elements[pdensRowOff + row * d_pdens.p + col] =
					(ONE_OVER_SQR_2PI / sqrt(sigmaSquared_n)) / exp(  ( (x_n - mu_n )*( x_n - mu_n ) / (2.0 * sigmaSquared_n)));
		}
	}
}


template<typename T>  void CuMatrix<T>::multivariateGaussianFeatures( DMatrix<T>& d_pden, const DMatrix<T>& d_x, const DMatrix<T>& d_sqrdSigmas, const DMatrix<T>& d_mu) {
	dim3 block(32,8);
	dim3 grid(b_util::enough(block.x, d_x.n), b_util::enough(block.x, d_x.m));
	outln("multivariateGaussianFeatures on d_x " << util<T>::pdm(d_x) );
	outln("multivariateGaussianFeatures with grid " << b_util::pd3(grid) << " block " << b_util::pd3(block));
	//outln("rows " << d_x.m);
	if(syncHappy)checkCudaError(cudaGetLastError());
	multivariateGaussianFeaturesKernel<<<grid,block>>>(d_pden, d_x, d_sqrdSigmas, d_mu);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}


/*
 * converts probability density matrix to probability vector
 * assumes block as wide as pdens.n
 */
template<typename T> __global__ void mvGaussianVectorFromFeaturesKernel(
		DMatrix<T> d_pvec, const DMatrix<T> d_pdens) {
	uint col = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	uint row = blockIdx.y * blockDim.y + threadIdx.y; //
	uint tileIdx = threadIdx.x + threadIdx.y * blockDim.x;
	// load smem
	T* tile = SharedMemory<T>();
	T currP;
	if(col < d_pdens.n && row < d_pdens.m) {
		// fill tile with block of pdens
		tile[tileIdx] = d_pdens.elements[col + row * d_pdens.p];
		__syncthreads();
		if(threadIdx.x == 0) {
			currP = tile[tileIdx];
			for(uint icol = 1; icol < d_pdens.n; icol++) {
				currP *= tile[tileIdx +icol];
			}
			d_pvec.elements[row* d_pvec.p] = currP;
		}
	}
}

template<typename T>  void CuMatrix<T>::mvGaussianVectorFromFeatures( DMatrix<T>& d_pvec, const DMatrix<T>& d_pdens) {
	//uint blockX =MAX(32, d_pdens.n);
	//dim3 block(blockX,caps.maxTsPerBlock<T>()/blockX);
	//uint smem = block.x * block.y * sizeof(T);
	uint blockX = d_pdens.n;
	uint blockY = MIN(d_pdens.m, caps.maxTsPerBlock<T>()/blockX);
	outln("blockY by max smem " << blockY);
	if(blockY < 1) {
		dthrow(notEnoughSmem());
	}
	blockY = MIN(blockY, caps.thrdPerBlock/blockX);
	outln("blockY by max thread/block" << blockY);
	uint smem = blockX * blockY * sizeof(T);
	dim3 block(blockX,blockY);
	dim3 grid(1, b_util::enough(block.y, d_pdens.m));
	//dim3 grid(b_util::enough(block.x, d_pdens.n), b_util::enough(block.y, d_pdens.m));
	outln("mvGaussianVectorFromFeatures with grid " << b_util::pd3(grid) << " block " << b_util::pd3(block) << ", smem " << smem);
	//outln("rows " << d_x.m);
	if(syncHappy)checkCudaError(cudaGetLastError());
	mvGaussianVectorFromFeaturesKernel<<<grid,block,smem>>>(d_pvec, d_pdens);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}

/*
 *
 * d_x.n must < max block dim otherwise two-step it
 * rows is how many rows of d_x.n cols will fit in smem
 *
 */
template<typename T> __global__ void multivariateGaussianVectorKernel(
		DMatrix<T> d_pvec, const DMatrix<T> d_x, const DMatrix<T> d_sigmaSquared, const DMatrix<T> d_mu) {
	uint col = threadIdx.x; // index into column
	uint row = blockIdx.y * blockDim.y + threadIdx.y; //
	uint xRowOff;
	uint tileIdx = threadIdx.x + threadIdx.y * blockDim.x;
	T x_n, sigmaSquared_n, mu_n;
	T* tile = SharedMemory<T>();
	if(col < d_x.n && row < d_x.m) {
		sigmaSquared_n = d_sigmaSquared.elements[col];
		mu_n = d_mu.elements[col];
		xRowOff = row * d_x.p;
		// fill as many rows as will fit in smem with feautures probs
		x_n = d_x.elements[xRowOff + col];
		tile[tileIdx] =
				(ONE_OVER_SQR_2PI / sqrt(sigmaSquared_n)) / exp(  ( (x_n - mu_n )*( x_n - mu_n ) / (2.0 * sigmaSquared_n)));
		__syncthreads();
		// now find the sample prob (== product of feature probs)
		if(threadIdx.x == 0) {
			x_n = tile[tileIdx] ;
			for(uint icol = 1; icol < blockDim.x; icol++) {
				x_n *= tile[tileIdx + icol];
			}
			d_pvec.elements[row * d_pvec.p] = x_n;
		}
	}
}


template<typename T>  void CuMatrix<T>::multivariateGaussianVector(  DMatrix<T>& d_pvec, const DMatrix<T>& d_x, const DMatrix<T>& d_sqrdSigmas, const DMatrix<T>& d_mu) {
	uint blockX = d_x.n;
	uint blockY = MIN(d_x.m, caps.maxTsPerBlock<T>()/blockX);
	if(blockY < 1) {
		dthrow(notEnoughSmem());
	}
	uint smem = blockX * blockY * sizeof(T);
	dim3 block(blockX,blockY);
	dim3 grid(1, b_util::enough(block.y, d_x.m));
	outln("multivariateGaussianVector on d_x " << util<T>::pdm(d_x));
	outln("multivariateGaussianVector with grid " << b_util::pd3(grid) << " of blocks " << b_util::pd3(block) << " with smem " << smem);
	//outln("rows " << d_x.m);
	if(syncHappy)checkCudaError(cudaGetLastError());
	multivariateGaussianVectorKernel<<<grid,block,smem>>>(d_pvec, d_x, d_sqrdSigmas,d_mu);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}

template<typename T> __global__ void meanSubKernel(DMatrix<T> res, const DMatrix<T> x,
		const DMatrix<T> means) {
//	uint tid = threadIdx.x; // index into smem for block
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	T mu;
	// have to zero sm first?
	if (i < x.n) {
		mu = means.elements[i];
		uint idx;
		for (uint row = 0; row < x.m; row++) {
			idx = row * x.n + i;
			res.elements[idx] = x.elements[idx] - mu;
		}
	}
}

template<typename T> __global__ void meanSubSqrKernel(DMatrix<T> res, const DMatrix<T> x,
		const DMatrix<T> means) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	T mu;
	T curr;
	// have to zero sm first?
	if (i < x.n) {
		mu = means.elements[i];
		uint idx;
		for (uint row = 0; row < x.m; row++) {
			idx = row * x.n + i;
			curr = x.elements[idx] - mu;
			res.elements[idx] = curr * curr;
		}
	}
}

template<typename T> void CuMatrix<T>::meanSubL(DMatrix<T>& d_res, const DMatrix<T>& d_x, const DMatrix<T>& d_means) {
	uint threads = 512;
	dim3 dBlocks, dThreads;
	uint smem = d_x.n * sizeof(T);
	if (smem > caps.memSharedPerBlock) {
		throw new smemExceeded;
	}
	b_util::execContext(threads, d_x.n, dBlocks, dThreads);
	outln(
			"meanSubL for " << d_x.m << "*" << d_x.n << ", have " << dBlocks.x << " blocks of " << dThreads.x << " threads with " << smem << " smem");
	//outln("rows " << d_x.m);
	if(syncHappy)checkCudaError(cudaGetLastError());
	meanSubKernel<<<dBlocks, dThreads, smem>>>(d_res, d_x, d_means);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}

template<typename T> void CuMatrix<T>::meanSubSqrL(DMatrix<T>& d_res, const DMatrix<T>& d_x, const DMatrix<T>& d_means) {
	uint threads = 512;
	dim3 dBlocks, dThreads;
	uint smem = d_x.n * sizeof(T);
	if (smem > caps.memSharedPerBlock) {
		throw new smemExceeded;
	}
	b_util::execContext(threads, d_x.n, dBlocks, dThreads);
	outln("meanSubL for " << d_x.m << "*" << d_x.n << ", have " << dBlocks.x << " blocks of " << dThreads.x << " threads with " << smem << " smem");
	//outln("rows " << d_x.m);
	if(syncHappy)checkCudaError(cudaGetLastError());
	meanSubSqrKernel<<<dBlocks, dThreads, smem>>>( d_res, d_x, d_means);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}

template<typename T> __global__ void columnProdKernel(DMatrix<T> prod, const DMatrix<T> x) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x; // index into column
	T product = 1;
	if(i < x.n) {
		for (uint row = 0; row < x.m; row++) {
			product *= x.elements[i + row * x.p];
		}
		prod.elements[i] = product;
	}
}

template<typename T> void CuMatrix<T>::columnProduct(DMatrix<T>& d_prod, const DMatrix<T>& d_x) {
	dim3 block(32), grid(b_util::enough(block.x, d_x.n));
	//outln("rows " << d_x.m);
	if(syncHappy)checkCudaError(cudaGetLastError());
	columnProdKernel<<<grid, block>>>(d_prod, d_x);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}

// block is x.n wide, deep as smem allows
template<typename T> __global__ void rowSumKernel(DMatrix<T> d_rowSums, const DMatrix<T> d_x) {
	uint col = threadIdx.x; // index into column
	uint row = blockIdx.y * blockDim.y + threadIdx.y; //
	uint tileIdx = threadIdx.x + threadIdx.y * blockDim.x;
	T* tile = SharedMemory<T>();
	T currRowSum = 0;
	if(threadIdx.x < d_x.n && row < d_x.m) {
		tile[tileIdx] = d_x.elements[row * d_x.p + col];
		__syncthreads();
		if(threadIdx.x == 0) {
			for(uint icol = 0; icol< d_x.n;icol++) {
				currRowSum += tile[tileIdx + icol];
			}
			d_rowSums.elements[row] = currRowSum;
		}
	}
}

template<typename T> void CuMatrix<T>::rowSum(DMatrix<T>& d_rowSums, const DMatrix<T>& d_x) {
	uint blockX = MAX(WARP_SIZE,d_x.n);
	if(blockX > caps.maxBlock.x) {
		dthrow(exceedsMaxBlockDim());
	}
	uint blockY = MIN( d_x.m, MIN( caps.maxBlock.y, MIN( caps.thrdPerBlock/blockX, caps.maxTsPerBlock<T>() )));
	dim3 block(blockX,blockY);
	uint smem = util<T>::vol(block);
	if (smem > caps.memSharedPerBlock) {
		dthrow (  new smemExceeded);
	}
	dim3 grid(1,b_util::enough(block.y, d_x.m));
	if(debugExec)outln("rowSum on " << util<T>::pdm(d_x) << " with " << b_util::pexec(grid,block, smem));
	if(syncHappy)checkCudaError(cudaGetLastError());
	rowSumKernel<<<grid, block, smem>>>(d_rowSums, d_x);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
}


template<typename T> string CuMatrix<T>::dimsString() const {
	stringstream ssout;
	ssout << m << "x" << n << "x" << p << "-" << size;
	return ssout.str();
}

template<typename T> string CuMatrix<T>::toShortString() const {
	stringstream ssout;
	ssout << "[[";
	ssout << this;
	ssout << " ";
	ssout << m;
	ssout << "x";
	ssout << n;
	ssout << "x";
	ssout << p;
	if(ownsBuffers) ssout << " owns";
	ssout << " (sz " << size << ") [";
	ssout << b_util::modStr(lastMod);
	ssout << (colMajor ? "] ColMajor" : "]");
	ssout << " h: ";
	if(elements)
		ssout << elements;
	else
		ssout << "null";
	ssout << ", d: ";
	if(d_elements)
		ssout << d_elements;
	else
		ssout << "null";
	ssout << "]]";
	return ssout.str();
}


template<typename T> template<typename CostFunction> void CuMatrix<T>::gradientApprox(
		CostFunction costFn, DMatrix<T> theta, DMatrix<T> perturb, DMatrix<T> gradApprox, T epsilon) {
	const uint l = theta.m * theta.n;
	ulong i = 0;
	constFiller<T> filler;
	filler.value = 0;
	fillFn(filler, perturb);
	fillFn(filler, gradApprox);
	T jMinus = 0, jPlus = 0;
	while (i < l) {
		//perturb.set(i, epsilon);
		CuMatrix<T>::set(perturb.elements, perturb.m, perturb.n, perturb.p, i, static_cast<T>(epsilon));
		jMinus = costFn(theta - perturb);
		jPlus = costFn(theta + perturb);
		//gradApprox.set(i, (jPlus - jMinus) / (2. * epsilon));
		CuMatrix<T>::set(gradApprox.elements, gradApprox.m, gradApprox.n, gradApprox.p, i, static_cast<T>((jPlus - jMinus) / (2. * epsilon)));
		//perturb.set(i, 0);
		CuMatrix<T>::set(perturb.elements, perturb.m, perturb.n, perturb.p, i, static_cast<T>(0));
		i += 1;
	}
}



#include "CuMatrixInster.cu"

