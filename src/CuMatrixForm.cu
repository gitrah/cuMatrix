#include "CuMatrix.h"
#include "util.h"
#include "debug.h"
#include "caps.h"
#include "MatrixExceptions.h"
#include <set>

template <typename T> __global__ void binaryCategoryKernel(const T* sElements, T* tElements, uint width, uint height, bool oneBased)
{
	T* tile = SharedMemory<T>();

    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    uint idxOut = xIndex + yIndex * width;
    if(blockDim.x == threadIdx.x == 0) {
    	tile[threadIdx.y] = sElements[yIndex];
    }
    __syncthreads();
    if(xIndex < width && yIndex < height) {
    	tElements[idxOut] = tile[threadIdx.y] == xIndex + (oneBased ? 1 : 0);
    }
}

template <typename T> void CuMatrix<T>::binaryCategoryKernelL(DMatrix<T>& t, const DMatrix<T>& s, bool oneBased)  {
	uint blockW = b_util::nextPowerOf2(t.n);
	uint blockH = maxH<T>(*ExecCaps::currCaps(),blockW);
	dim3 block(blockW, blockH);
	if(checkDebug(debugExec))outln("binCat blockW " << blockW << ", blockH " << blockH << " block " << b_util::pd3(block));
    dim3 grid(DIV_UP( t.n, block.x), DIV_UP(t.m, block.y));
	if(checkDebug(syncHappy))b_util::syncGpu("before knl " );
	int smem = block.x * block.y * sizeof(T);
	binaryCategoryKernel<<<grid, block, smem>>>(s.elements, t.elements, t.n, t.m, oneBased);
	if(checkDebug(debugExec))outln("binCatKernel with grid " << b_util::pd3(grid).c_str() << " of block " << b_util::pd3(block).c_str() << " smem " << smem);
	if(checkDebug(syncHappy))b_util::syncGpu("after");
	checkCudaError(cudaDeviceSynchronize());
}

template<typename T> CuMatrix<T> CuMatrix<T>::toBinaryCategoryMatrix() const {
	const uint len = m * n;
	if(!elements) dthrow(noHostBuffer());
	::set<T> s(elements, elements + len); 	// TODO make a kernel for this (self-reduction until steady state?)
	bool oneBased = (s.find(0) == s.end());
	//outln("oneBased " << tOrF(oneBased));
	uint newCols = s.size();
	CuMatrix<T> res(m, newCols,false,true);
	DMatrix<T> d_res = res.asDmatrix(false);
	DMatrix<T> d_src = asDmatrix();
	//outln("binCat found " << newCols << " distinct values");
	binaryCategoryKernelL(d_res, d_src, oneBased);
	res.lastMod=mod_device;
	return res;
}


// TODO add oldP and tests for p != n
template<typename T> CuMatrix<T> CuMatrix<T>::poseAsRow() {
	oldM = m;
	m = 1;
	n *= oldM;
	p = n;
	posed = true;
	return *this;
}

template<typename T> CuMatrix<T> CuMatrix<T>::poseAsCol() {
	oldN = n;
	n = 1;
	p = n;
	m *= oldN;
	posed = true;
	return *this;
}

template<typename T> CuMatrix<T> CuMatrix<T>::unPose() {
	if (posed && oldM != 0) {
		m = oldM;
		n /= oldM;
		p = n;
		oldM = 0;
	} else if (posed && oldN != 0) {
		n = oldN;
		p = n;
		m /= oldN;
		oldN = 0;
	}
	posed = false;
	return *this;
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::reshape(CuMatrix<T>& target, uint rows, uint cols, ulong offsetInTs) {
	if(d_elements == null ) setLastError(noDeviceBufferEx); else if(checkDebug(debugMem)) prlocf("reshape have nz d_elements");
	if(target.d_elements == null ) setLastError(noDeviceBufferEx); else if(checkDebug(debugMem)) prlocf("reshape have nz ret.d_elements");
	uint l = rows * cols;
	if(contiguousQ()) {
		if(gpuReadyQ()) {
#ifndef __CUDA_ARCH__
			cherr(
				cudaMemcpy(target.d_elements, d_elements + offsetInTs, l * sizeof(T), cudaMemcpyDeviceToDevice));
			DDCopied++;
			MemDdCopied += l *sizeof(T);
#else
	#ifdef CuMatrix_Enable_Cdp
				cherr(
					cudaMemcpyAsync(target.d_elements, d_elements + offsetInTs, l * sizeof(T), cudaMemcpyDeviceToDevice));
	#endif
#endif
			target.lastMod = mod_device;
		}else {
#ifndef __CUDA_ARCH__
			cherr(
				cudaMemcpy(target.elements, elements + offsetInTs, l * sizeof(T), cudaMemcpyHostToHost));
			HHCopied++;
			MemHhCopied += l *sizeof(T);
			target.lastMod = mod_host;
#endif
		}
	} else {
		DMatrix<T> src, trg;
		asDmatrix(src);
		target.asDmatrix(trg);
		if(gpuReadyQ()) {
			src.elements += offsetInTs;
#ifndef __CUDA_ARCH__
			MemDdCopied += l *sizeof(T);
#endif
			target.lastMod = mod_device;
		}else {
			src.elements = elements + offsetInTs;
			trg.elements = target.elements + offsetInTs;
#ifndef __CUDA_ARCH__
			HHCopied++;
#endif
			target.lastMod = mod_host;
		}
		copyUintDvrg(trg,src,0,0);
	}
	if(checkDebug(debugSync )) {
		flprintf("CuMatrix (%p)::reshaped( %u * %u, off %u ) -> %p setLastMod %s\n",this,rows,cols,offsetInTs,&target, b_util::modStr(lastMod));
	}
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::reshape(uint rows, uint cols,
		ulong offsetInTs) {
	CuMatrix<T> res(rows, cols,false,true);
	reshape(res, rows, cols, offsetInTs);
	return res;
}

template<typename T> CuMatrix<T> CuMatrix<T>::redimension(
		pair<uint, uint>& dims, uint offset) {
	return reshape(dims.first, dims.second, offset);
}

template<typename T> void CuMatrix<T>::unconcat(CuMatrix<T>& v, uint rows, uint cols, uint pitch, uint offset) const {
	if(!vectorQ()){
		dthrow(notVector());
	}
	if(offset + rows * cols > m * n) {
		outln("invalid submatrix ( > this)");
		dthrow(badDimensions());
	}
	v.elements =  elements ? elements + offset : null ;
	v.d_elements =  d_elements ? d_elements + offset : null;
	v.m = rows;
	v.n = cols;
	v.p = pitch;
	v.size = v.m * v.n * sizeof(T);
	v.lastMod = CuMatrix<T>::lastMod;
	v.ownsBuffers = false;

	if(checkDebug(debugFill)) outln("of " << toShortString() << " i am " << v.toShortString());
}

template<typename T> void CuMatrix<T>::submatrix(CuMatrix<T>& v, uint rows, uint cols, uint roff, uint coff) const {
	if(roff + rows > m || coff + cols > n) {
		outln("invalid submatrix ( > this)");
		dthrow(badDimensions());
	}
	uint offset = roff * p + coff;
	v.elements =  elements ? elements + offset : null ;
	v.d_elements =  d_elements ? d_elements + offset : null;
	v.m = rows;
	v.n = cols;
	v.p = p;
	v.size = v.m * v.p * sizeof(T);
	v.lastMod = lastMod;
	v.ownsBuffers = false;

	if(checkDebug(debugFill)) outln("of " << toShortString() << " i am " << v.toShortString());
}
// crap
template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::columnMatrix(uint col) const {
	CuMatrix<T> column(m, 1, false, true);
	DMatrix<T> d_X, d_Col;
	asDmatrix(d_X);
	column.asDmatrix(d_Col, false);
	columnMatrixL(d_Col, d_X, col);
	return column;
}

template<typename T> CuMatrix<T> CuMatrix<T>::rowMatrix(uint row) const {
	CuMatrix<T> rowm(1,n, false, true);
	if(colMajor) {
		DMatrix<T> d_X, d_row;
		asDmatrix(d_X);
		rowm.asDmatrix(d_row, false);
		rowMatrixCmL(d_row, d_X, row);
	} else {
		checkCudaError(cudaMemcpy(rowm.d_elements, elements + row*p, n*sizeof(T), cudaMemcpyDeviceToDevice));
		DDCopied++;
		MemDdCopied += (n - 1) * sizeof(T);
	}
	return rowm;
}

template<typename T> CuMatrix<T> CuMatrix<T>::dropFirst(bool copy) const {
	if(lastMod == mod_host) dthrow(notSynced());

	CuMatrix<T> res(m, n - 1, false, copy);
	if(copy){
		uint i = 0;
		while (i < m) {
			checkCudaError(
					cudaMemcpy(res.d_elements + i * (n - 1), d_elements + i * n + 1, (n - 1) * sizeof(T), cudaMemcpyDeviceToDevice));
			i++;
			DDCopied++;
			MemDdCopied += (n - 1) * sizeof(T);
			res.lastMod = mod_device;
		}
	} else {
		submatrix(res, m, n -1, 0, 1);
	}
	return res;
}

template<typename T> CuMatrix<T> CuMatrix<T>::dropLast(bool copy) const {
	if(lastMod == mod_host) dthrow(notSynced());

	CuMatrix<T> res(m, n - 1, false, copy);
	if(copy){
		uint i = 0;
		while (i < m) {
			checkCudaError(
					cudaMemcpy(res.d_elements + i * (n - 1), d_elements + i * n, (n - 1) * sizeof(T), cudaMemcpyDeviceToDevice));
			i++;
			DDCopied++;
			MemDdCopied += (n - 1) * sizeof(T);
			res.lastMod = mod_device;
		}
	} else {
		submatrix(res, m, n -1, 0, 0);
	}
	return res;
}

template<typename T> CuMatrix<T> CuMatrix<T>::vectorToDiagonal() const {
	if (!vectorQ()) {
		dthrow (  notVector());
	}
	if(!elements) {
		dthrow(noHostBuffer());
	}
	if(lastMod == mod_device) {
		dthrow(notSyncedHost());
	}
	uint dim = longAxis();
	CuMatrix<T> ret = CuMatrix<T>::zeros(dim, dim);
	ret.syncBuffers();
	for (uint i = 0; i < dim; i++) {
		ret.elements[i * dim + i] = elements[i];
	}
	ret.lastMod = mod_host;
	return ret;
}


template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::columnVector(uint col) const {
	return columnMatrix(col);
}

template<typename T> CuMatrix<T> CuMatrix<T>::rowVector(uint row) const {
	return rowMatrix(row);
}

template<typename T> CuMatrix<T> CuMatrix<T>::toRowVector() const {
	return CuMatrix<T>(elements, 1, m * n, true);
}

template<typename T> CuMatrix<T> CuMatrix<T>::toColumnVector() const {
	return CuMatrix<T>(elements, m * n, 1, true);
}

template<typename T> T CuMatrix<T>::toScalar() const {
	dassert(scalarQ());
	if(elements && (lastMod == mod_synced || lastMod == mod_neither) ) {
		return elements[0];
	}
	return get(0);
}

template<typename T> CuMatrix<T> CuMatrix<T>::toDiagonalsVector() const {
	dassert(squareQ());
	CuMatrix<T> ret (n,1,true,true);
	uint i = 0;
	while (i < n) {
		ret.elements[i] = elements[i * n + i];
		i++;
	}
	ret.lastMod = mod_host;
	return ret;
}


template<typename T> __global__ void columnMatrixKernel(DMatrix<T> column, const DMatrix<T> x,
		uint col) {
	uint row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < x.m) {
		column.elements[row] = x.elements[row * x.p + col];
	}
}

template<typename T> __global__ void rowMatrixCMKernel(DMatrix<T> d_row, const DMatrix<T> x,
		uint row) {
	uint col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < x.n) {
		d_row.elements[col] = x.elements[col * x.p + row];
	}
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::columnMatrixL(DMatrix<T>& d_column, const DMatrix<T>& d_x,
		 uint col) {
	dim3 block(MIN(512, b_util::nextPowerOf2(d_x.m)));
	dim3 grid(DIV_UP(d_x.m,block.x));
	columnMatrixKernel<<<grid,block,0>>>(d_column, d_x, col );
#ifndef __CUDA_ARCH__
	if(checkDebug(syncHappy))checkCudaError(cudaDeviceSynchronize());
#endif
}

template<typename T> void CuMatrix<T>::rowMatrixCmL(DMatrix<T>& d_row, const DMatrix<T>& d_x,
		 uint row) {
	dim3 block(MIN(512, b_util::nextPowerOf2(d_x.m)));
	dim3 grid(DIV_UP(d_x.m,block.x));
	rowMatrixCMKernel<<<grid,block,0>>>(d_row, d_x, row);
#ifndef __CUDA_ARCH__
	if(checkDebug(syncHappy))checkCudaError(cudaDeviceSynchronize());
#endif
}

#include "CuMatrixInster.cu"

