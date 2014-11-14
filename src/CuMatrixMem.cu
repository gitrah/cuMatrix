#include "CuMatrix.h"
#include "util.h"
#include "Kernels.h"
#include "debug.h"
#include "caps.h"
#include "MatrixExceptions.h"
#include <typeinfo>

template<typename T> __host__ __device__ CuMatrix<T>& CuMatrix<T>::syncBuffers(bool copy ) {
	if(checkDebug(debugCopy)){
		printf("syncBuffers(%s) on %dX%d d_elements %p\n", tOrF(copy), m,n,d_elements);
#ifndef __CUDA_ARCH__
		outln("[caller " << b_util::caller() << "]");
		if(checkDebug(debugCopyDh)) {
			outln("[["<< b_util::unmangl(typeid(this).name())<< "]] from syncBuffers\n" << print_stacktrace());
		}
#else
		prlocf("syncBuffers called from device");
#endif
	}
	//dassert(CuMatrix<T>::d_elements && elements);
	if(lastMod != mod_synced) {
		if (lastMod == mod_device) {
			//outln*
#ifndef __CUDA_ARCH__
			if(!elements) {
				if(checkDebug(debugMem)) outln("syncBuffers !elements");
				cherr(cudaPeekAtLastError());
				getMgr().allocHost(*this);
			}
			if(n != p) {
				if (checkDebug(debugCopy | debugSync))
					outln( "syncBuffers() n != p so doing line by line copy");
				for(int i = 0; i < m; i++ ){
					cherr(cudaMemcpy(elements + i * n, d_elements + i * p, n*sizeof(T), cudaMemcpyDeviceToHost));
				}
			}else {
				cherr(cudaMemcpy(elements, d_elements, size, cudaMemcpyDeviceToHost));
			}
			//err = cudaMemcpy(elements, d_elements, size, cudaMemcpyDeviceToHost);
			lastMod = mod_synced;
			DHCopied++;
			MemDhCopied += size;
			if (checkDebug(debugCopy | debugSync))
				outln( "syncBuffers() mat " << this << " copied " << size << " from d " << d_elements << " to  h " << elements);
#else
			printShortString("WARN syncBuffers can't update host from device");
			setLastError(cantSyncHostFromDeviceEx);
#endif
		} else {// if (lastMod == mod_host) {
			if(d_elements == null) {
				if(checkDebug(debugSync))printf("creating device buffer\n");
#ifndef __CUDA_ARCH__
				getMgr().allocDevice(*this);
#else
				d_elements = (T*) malloc(size);
#endif
			}
#ifndef __CUDA_ARCH__
			if(!elements) {
				if(lastMod == mod_neither) {
					if(checkDebug(debugSync)) printf("creating host buffer\n");
					getMgr().allocHost(*this);
				} else {
					dthrow(noHostBuffer());
				}
			}
#endif
			if(lastMod != mod_neither && copy) {
#ifndef __CUDA_ARCH__
				cherr(cudaMemcpy(d_elements, elements, size, cudaMemcpyHostToDevice));
				HDCopied++;
				MemHdCopied += size;
				if (checkDebug(debugCopy| debugSync))
					outln("syncBuffers() mat " << this << " copied h " << elements << " to  d " << d_elements);
				lastMod = mod_synced;
#endif
			}
		}
	}
	return *this;
}

template<typename T> __host__ __device__  CuMatrix<T> CuMatrix<T>::syncHost() {
	invalidateHost();
	return syncBuffers();
}

template<typename T> __host__ __device__  CuMatrix<T> CuMatrix<T>::syncDevice() {
	invalidateDevice();
	return syncBuffers();
}

template<typename T> __host__ __device__ void CuMatrix<T>::invalidateHost() {
	if(!d_elements) {
		setLastError(noDeviceBufferEx);
	}
	if(checkDebug(debugSync) && lastMod != mod_device) {
#ifndef __CUDA_ARCH__
		outln("matrix " << this << " invalHost clr " << b_util::callerN(3));
#else
		printf("matrix %p invalHost clr\n",this);
#endif
	}
	lastMod = mod_device;
	freeTxp();
}

template<typename T> __host__ __device__ void CuMatrix<T>::invalidateDevice() {
	if(!elements) {
		setLastError(noHostBufferEx);
	}
	if(checkDebug(debugSync) && lastMod != mod_host) {
#ifndef __CUDA_ARCH__
		outln("matrix " << this << " invalidateDevice caller " << b_util::callerN(3));
#else
		printf("matrix %p invalidateDevice clr\n",this);
#endif
	}
	lastMod = mod_host;
	freeTxp();
}

template<typename T> __host__ __device__ cudaError_t CuMatrix<T>::asDmatrix(DMatrix<T>& md,
		bool copy, bool force) const {

	if(lastMod == mod_device) {
		if(checkDebug(debugSync | debugMem)) {
			printShortString(" asDmatrix: lastMod == device; not copying host-dev");
		}
		copy = false;
	}
	md.m = m;
	md.n = n;
	md.p = p;
	bool needForce = false;

	if (d_elements != null) {
		needForce = true;
#ifndef __CUDA_ARCH__
		if(checkDebug(debugMem)) MemMgr<T>::checkValid(d_elements);
#endif
		if (md.elements == null) {
			md.elements = d_elements;
		}
	} else {
#ifndef __CUDA_ARCH__
		dthrow(noDeviceBuffer());
#else
		setLastError(noDeviceBufferEx);
#endif
	}

	if (lastMod == mod_host || (copy && (!needForce || (needForce && force)))) {
		if (checkDebug(debugCopy) || (lastMod == mod_host && checkDebug(debugSync))) {
			printf(" asDmatrix %p h-d copying %u * %u - %u from %p to %p\n",this, m, n, size,elements, md.elements);
			printf("lastMod == mod_host %s\n", tOrF(lastMod == mod_host));
#ifndef __CUDA_ARCH__
			outln("callerN " <<  b_util::callerN(3) );
#endif
		}
#ifndef __CUDA_ARCH__
		if (checkDebug(debugCopy))printf("asDmatrix cudaMemcpy");
		cherr(
				cudaMemcpy(d_elements, elements, size, cudaMemcpyHostToDevice));
#else
		if (checkDebug(debugCopy))printf("asDmatrix memcpy");
		memcpy(d_elements, elements, size);
#endif
#ifndef __CUDA_ARCH__
		HDCopied++;
		MemHdCopied += size;
#endif
	}
	if (checkDebug(debugMem))
		printShortString("asDmatrix(DMatrix<T>&,bool,bool) exit");
	return cudaSuccess;
}

template<typename T> __host__ __device__  DMatrix<T> CuMatrix<T>::asDmatrix(  bool copy) const {
	DMatrix<T> ret;
	asDmatrix(ret,copy,false);
	return ret;
}


template <typename T> __global__ void
copyKernel(T* tElements, const T* sElements, uint tWidth, uint tHeight, uint tPitch, uint sWidth, uint sHeight, uint sPitch, uint xOff, uint yOff)
{
    ulong x = blockIdx.x * blockDim.x + threadIdx.x;
    ulong y = blockIdx.y * blockDim.y + threadIdx.y;
    ulong tx = x + xOff;
    ulong ty = y + yOff;
    ulong sIdx = y * sPitch + x;
    ulong tIdx = ty * tPitch + tx;
    if(threadIdx.x == 0 && threadIdx.y == 0) {
    	if(checkDebug(debugCopy))flprintf("block %u,%u tx %lu ty %lu sIdx %lu tIdx %lu\n", blockIdx.x, blockIdx.y, tx,ty,sIdx,tIdx);
    }
    if(x < sWidth && y < sHeight && tx < tWidth && ty < tHeight) {
    	tElements[tIdx] = sElements[sIdx];
    }
}

template <typename T> __global__ void
copyDmKernelUlong(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) {
    ulong x = blockIdx.x * blockDim.x + threadIdx.x;
    ulong y = blockIdx.y * blockDim.x + threadIdx.y;
    ulong tx = x + tcoff;
    ulong ty = y + troff;
    ulong sIdx = y * src.p + x;
    ulong tIdx = ty * trg.p + tx;
    for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
		if(x < src.n && y + i < src.m && tx < trg.n && ty + i < trg.m) {
			trg.elements[tIdx + i * trg.p] = src.elements[sIdx + i * src.p];
		}
    }
}

template <typename T> __global__ void
copyDmKernelUint(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.x + threadIdx.y;
	uint tx = x + tcoff;
	uint ty = y + troff;
	uint sIdx = y * src.p + x;
	uint tIdx = ty * trg.p + tx;
    for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
		if(x < src.n && y + i < src.m && tx < trg.n && ty + i < trg.m) {
			trg.elements[tIdx + i * trg.p] = src.elements[sIdx + i * src.p];
		}
    }
}

template <typename T> __global__ void
copyDmKernelUlongDvrg(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) {
    ulong x = blockIdx.x * blockDim.x + threadIdx.x;
    ulong y = blockIdx.y * blockDim.x + threadIdx.y;
    ulong tx = x + tcoff;
    ulong ty = y + troff;
    ulong sIdx = y * src.p + x;
    ulong tIdx = ty * trg.p + tx;
    if(x<src.n && tx < trg.n) {
    	uint sdiff = src.m - y;
    	uint tdiff = trg.m - y;
		for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
			if ( i < sdiff && i < tdiff) {
				trg.elements[tIdx + i * trg.p] = src.elements[sIdx + i * src.p];
			}
		}
    }
}

template <typename T> __global__ void
copyDmKernelUintDvrg(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.x + threadIdx.y;
    uint tx = x + tcoff;
    uint ty = y + troff;
    uint sIdx = y * src.p + x;
    uint tIdx = ty * trg.p + tx;
    if(x<src.n && tx < trg.n) {
    	uint sdiff = src.m - y;
    	uint tdiff = trg.m - y;
		for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
			if ( i < sdiff && i < tdiff) {
				trg.elements[tIdx + i * trg.p] = src.elements[sIdx + i * src.p];
			}
		}
    }
}

template <typename T> __global__ void
copyDmKernelIntDvrg(DMatrix<T> trg, const DMatrix<T> src, int troff, int tcoff) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.x + threadIdx.y;
	int tx = x + tcoff;
	int ty = y + troff;
	int sIdx = y * src.p + x;
	int tIdx = ty * trg.p + tx;
    if(x<src.n && tx < trg.n) {
    	int sdiff = src.m - y;
    	int tdiff = trg.m - y;
		for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
			if ( i < sdiff && i < tdiff) {
				trg.elements[tIdx + i * trg.p] = src.elements[sIdx + i * src.p];
			}
		}
    }
}

// indices.length === trg.m
template <typename T> __global__ void
copyDmRowShuffleKernel(DMatrix<T> trg, const DMatrix<T> src, uint* indices) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.x + threadIdx.y;
	uint tIdx = y * trg.p + x;
    if(x < trg.n && y < trg.m) {
    	int tdiff = trg.m - y;
		for(int i = 0; i < blockDim.x ; i+=blockDim.y ) {
			if ( i < tdiff) {
				trg.elements[tIdx + i * trg.p] = src.elements[ indices[y + i] * src.p + x];
			} else {
		    	if(blockIdx.x == 0 && threadIdx.x == 0) {
		    		if (checkDebug(debugMem))flprintf("i (%d) !< tdiff (%d), y = %d\n", i, tdiff, y);
		    	}
			}
		}
    } else {
    	if(blockIdx.x == 0 && threadIdx.x == 0) {
    		if (checkDebug(debugMem))flprintf("%d !< trg.n or %d !< trg.m\n", x, y);
    	}
    }
}

template <typename T> __global__ void
copyKernel(T* tElements, const T* sElements, uint amountInTs, uint offsetInTs, ulong lengthInTs)
{
    ulong id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < amountInTs && id + offsetInTs < lengthInTs) {
    	tElements[id + offsetInTs] = sElements[id];
    }
}



template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::rightConcatenateL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2) {
	assert(src1.m == src2.m);
	ExecCaps* pcaps = ExecCaps::currCaps();

	dim3 block;
	if(src1.m > src1.n) {
		block.x = MIN(MAX(src1.n, src2.n), CAT_BLOCK_SIZE);
		block.y = MIN(MIN(src1.m, pcaps->thrdPerBlock/block.x), pcaps->maxBlock.y);
	} else {
		block.y = MIN(src1.m, CAT_BLOCK_SIZE);
		block.x = MIN(MIN(src1.n, pcaps->thrdPerBlock/block.y), pcaps->maxBlock.x);
	}
	dim3 grid1(DIV_UP(src1.n,block.x), DIV_UP(src1.m,block.y),1);
	dim3 grid2(DIV_UP(src2.n,block.x), grid1.y,1);

	dim3 launches1 ( grid1.x < pcaps->maxGrid.x ? 1 : DIV_UP(pcaps->maxGrid.x, grid1.x), grid1.y < pcaps->maxGrid.y ? 1 : DIV_UP(pcaps->maxGrid.y, grid1.y));
	dim3 launches2 (grid2.x < pcaps->maxGrid.x ? 1 : DIV_UP(pcaps->maxGrid.x, grid2.x),  grid2.y < pcaps->maxGrid.y ? 1 : DIV_UP(pcaps->maxGrid.y, grid2.y));

	if(checkDebug(debugCopy)) {
		flprintf("block %u,%u\n", block.y, block.x);
		flprintf("grid1 %u,%u\n", grid1.y, grid1.x);
		flprintf("grid2 %u,%u\n", grid2.y, grid2.x);
		flprintf("launches1 %d,%d, launches2 %d,%d\n", launches1.y, launches1.x, launches2.y, launches2.x);
	}

	//copyKernel<<<gridAc,block>>>(d_C, d_A, 0, 0);
	for(int y = 0; y < launches1.y || y < launches2.y; y++) {
		for(int x =0; x < launches1.x || x < launches2.x; x++) {
			if(y < launches1.y && x< launches1.x) {
				dim3 grid(grid1.x/launches1.x,grid1.y/launches1.y);
				uint xoff = src1.n / launches1.x * x;
				uint yoff = src1.m / launches1.y * y;
				copyKernel<<<grid1,block>>>(trg.elements, src1.elements, trg.n, trg.m, trg.p, src1.n, src1.m, src1.p, xoff, yoff);
			}
			if(y < launches2.y && x< launches2.x) {
				dim3 grid(grid2.x/launches1.x,grid2.y/launches1.y);
				uint xoff = src2.n / launches2.x * x;
				uint yoff = src2.m / launches2.y * y;
				copyKernel<<<grid2,block>>>(trg.elements, src2.elements, trg.n, trg.m, trg.p, src2.n, src2.m, src2.p, src1.n + xoff, yoff);
			}
		}
	}
	cherr(cudaDeviceSynchronize());
}

template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::bottomConcatenateL(DMatrix<T>& trg, const DMatrix<T>& src1, const DMatrix<T>& src2)  {
	dim3 block(CAT_BLOCK_SIZE,CAT_BLOCK_SIZE,1);
	dim3 dmBlock(CAT_BLOCK_SIZE,CAT_BLOCK_SIZE/8,1);
	dim3 gridAc(DIV_UP(src1.n,CAT_BLOCK_SIZE), DIV_UP(src1.m,CAT_BLOCK_SIZE),1);
	dim3 gridBc(DIV_UP(src2.n,CAT_BLOCK_SIZE), DIV_UP(src2.m,CAT_BLOCK_SIZE),1);
	if(checkDebug(debugExec)){
#ifndef __CUDA_ARCH__
		outln("gridAc " << b_util::pd3(gridAc).c_str() << " for " << util<T>::pdm(src1));
		outln("gridBc " << b_util::pd3(gridBc).c_str() << " for " << util<T>::pdm(src2));
#else
		printf("gridAc " );
		b_util::prd3(gridAc);
		//c_str() << " for " << util<T>::pdm(src1));

	//	printf("gridBc " << b_util::pd3(gridBc).c_str() << " for " << util<T>::pdm(src2));
#endif
	}
	if(src1.n == src1.p) {
		copyKernel<<<gridAc,block>>>(trg.elements, src1.elements,trg.n, trg.m, trg.p, src1.n, src1.m, src1.p, 0, 0);
	} else {
#ifndef __CUDA_ARCH__
		dthrow(notImplemented());
#else
		setLastError(notImplementedEx);
#endif
	}
	copyKernel<<<gridBc,block>>>(trg.elements, src2.elements, trg.n, trg.m, trg.p, src2.n, src2.m, src2.p, 0, src1.m);
}


template <typename T> void CuMatrix<T>::copy1D(T* trg, const T* src, uint amountInTs, uint offsetInTs, uint lengthInTs, cudaStream_t stream) {

	dim3 block(32);
	dim3 grid(DIV_UP(lengthInTs, block.x));
	if(stream)
		copyKernel<<<grid,block,0,stream>>>(trg,src, amountInTs, offsetInTs, lengthInTs);
	else
		copyKernel<<<grid,block>>>(trg,src, amountInTs, offsetInTs, lengthInTs);
}

template <typename T> void CuMatrix<T>::copyK(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	copyUint(trg, src, troff, tcoff);
}

template <typename T> void CuMatrix<T>::copy(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
	T* dst = trg.elements + troff*trg.p + tcoff;
	const size_t tSize= sizeof(T);
	checkCudaError(cudaMemcpy2D(dst, trg.p * tSize, src.elements, src.p* tSize, src.n* tSize, src.m,cudaMemcpyDeviceToDevice));
}

template <typename T> void CuMatrix<T>::copyAsync(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
	T* dst = trg.elements + troff*trg.p + tcoff;
	const size_t tSize= sizeof(T);
	checkCudaError(cudaMemcpy2DAsync(dst, trg.p * tSize, src.elements, src.p* tSize, src.n* tSize, src.m,cudaMemcpyDeviceToDevice));
}

template <typename T> void CuMatrix<T>::copyUlong(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(DIV_UP(src.n,block.x), DIV_UP(src.m,block.x));
	copyDmKernelUlong<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::copyUint(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(DIV_UP(src.n,block.x), DIV_UP(src.m,block.x));
	copyDmKernelUint<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::copyUlongDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(DIV_UP(src.n,block.x), DIV_UP(src.m,block.x));
	copyDmKernelUlongDvrg<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> __host__ CUDART_DEVICE void CuMatrix<T>::copyUintDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(DIV_UP(src.n,block.x), DIV_UP(src.m,block.x));
	copyDmKernelUintDvrg<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::copyIntDvrg(DMatrix<T>& trg, const DMatrix<T>& src, int troff, int tcoff) {
	dim3 block(32);
	dim3 grid(DIV_UP(src.n,block.x), DIV_UP(src.m,block.x));
	copyDmKernelIntDvrg<<<grid,block>>>(trg,src, troff,tcoff);
}

template <typename T> void CuMatrix<T>::shuffleCopyRows(DMatrix<T>& trg, const DMatrix<T>& src, uint* rowIndices) {
	dim3 block(32,8);
	dim3 grid(DIV_UP(trg.n,block.x), DIV_UP(trg.m,block.x ));
	outln("grid " << b_util::pd3(grid));
	copyDmRowShuffleKernel<<<grid,block>>>(trg,src, rowIndices);
}

template<typename T> __host__ __device__ CuMatrix<T> CuMatrix<T>::copy(bool copyDeviceMem) const {
	CuMatrix<T> ret(m, n, elements, d_elements);
#ifndef __CUDA_ARCH__
	if (elements) {
		cherr(
				cudaMemcpy(ret.elements, elements, size, cudaMemcpyHostToHost));
		HHCopied++;
		MemHhCopied += size;
	}
#endif
	if (d_elements && copyDeviceMem) {
		ret.asDmatrix();
#ifndef __CUDA_ARCH__
		cherr(
				cudaMemcpy(ret.d_elements, d_elements, size, cudaMemcpyDeviceToDevice));
		DDCopied++;
		MemDdCopied += size;
#else
		memcpy(ret.d_elements, d_elements, size);

#endif
	}
	ret.lastMod  = lastMod;
	if(checkDebug(debugSync) && ret.lastMod == mod_host) {
		printf("CuMatrix (%p::copy(%s) -> %p set lastMod of host\n",this, tOrF(copyDeviceMem),&ret );
	}
	ret.posed = posed;
	ret.colMajor = colMajor;
	ret.oldM = oldM;
	ret.oldN = oldN;
	ret.p = p;
	ret.size = size;
	if(txp && ownsTxp) {
		if(checkDebug(debugSync))printf("copy() recreating txp\n");
		ret.txp = new CuMatrix<T>(n,m, true, true);
		ret.ownsTxp = true;
		if (txp->elements) {
			if(checkDebug(debugSync))printf("copy() copying txp->elements\n");
#ifndef __CUDA_ARCH__
			cherr(
					cudaMemcpy(ret.txp->elements, txp->elements, size, cudaMemcpyHostToHost));
			HHCopied++;
			MemHhCopied += size;
#endif
		}
		if (txp->d_elements && copyDeviceMem) {
			if(checkDebug(debugSync))printf("copy() copying txp->d_elements\n");
			ret.txp->asDmatrix();
#ifndef __CUDA_ARCH__
			cherr(
					cudaMemcpy(ret.txp->d_elements, txp->d_elements, size, cudaMemcpyDeviceToDevice));
			DDCopied++;
			MemDdCopied += size;
#else
			memcpy(ret.txp->d_elements, txp->d_elements, size);
#endif
		}
		ret.txp->lastMod  = txp->lastMod;
		ret.txp->posed = txp->posed;
		ret.txp->colMajor = txp->colMajor;
		ret.txp->oldM =txp->oldM;
		ret.txp->oldN = txp->oldN;
		ret.txp->p = txp->p;
		ret.txp->size = txp->size;
	}
	return ret;
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::columnSubset( const uint* indices,
		uint count) const {
	uint i = 0;
	CuMatrix<T> res = CuMatrix<T>::zeros(0, 0);
	while (i < count) {
		CuMatrix<T> cVec = columnVector(indices[i]);
		if (res.m == 0 && res.n == 0) {
			res = cVec;
		} else {
			res |= cVec;
		}
		i++;
	}
	res.printShortString("columnSubset ");
	//res.lastMod = mod_device;
	return res;
}

template<typename T> CuMatrix<T> CuMatrix<T>::clippedRowSubset( const int *r, uint count,
		pair<uint, uint> colRange) const {
	if(colMajor) {
		dthrow(notImplemented())
	}
	if(!elements) {
		dthrow(noHostBuffer())
	}
	if(lastMod == mod_device) {
		dthrow(notSyncedHost());
	}
	//outln("clippedRowSubset *this " << *this);
	printf("clippedRowSubset colRange %d-%d, n %d\n", colRange.first, colRange.second,n);
	assert((colRange.first < colRange.second && colRange.second < n));
	uint width = colRange.second - colRange.first + 1;
	uint newM = count;
	CuMatrix<T> res = zeros(newM,width).syncBuffers();
	//res.printShortString("clippedRowSubset res " );
	//printShortString("clippedRowSubset this ");
	uint i = 0;
	while (i < newM) {
		//outln("i " << i << " r[i] * p " << (r[i]*p) << ",  i * res.p " << ( i * res.p) );
		memcpy(res.elements + i * res.p, elements + r[i] * p, width * sizeof(T));
		i++;
	}
	//outln("res after " << res);
	res.lastMod = mod_host;
	//res.syncBuffers();
	//outln("clippedRowSubset res " << res);
	return res;
}

template<typename T> CuMatrix<T> CuMatrix<T>::addBiasColumn() const {
	CuMatrix<T> bias = CuMatrix<T>::ones(m, 1);
	return bias.rightConcatenate(*this);
}

template<typename T> CuMatrix<T> CuMatrix<T>::replicateTiled(uint mcopies, uint ncopies) const {
	if(!mcopies || ! ncopies) {
		dthrow(illegalArgument());
	}
	CuMatrix<T> tiled(mcopies*m, ncopies *n,false,true);
	DMatrix<T> dTiled = tiled.asDmatrix();
	DMatrix<T> dSrc  = asDmatrix();
	for(int row = 0; row < mcopies; row++) {
		for(int col = 0; col < ncopies; col++) {
			CuMatrix<T>::copyAsync(dTiled, dSrc, row*m, col*n);
		}
	}
	tiled.invalidateHost();
	checkCudaError(cudaDeviceSynchronize());
	return tiled;
}

template<typename T> void CuMatrix<T>::copy(CuMatrix<T>& res, int roff, int coff, bool onlyDevice) const {
	if(roff + m  >  res.m || coff + n > res.n ) {
		outln("roff " << roff + " +  m " << m << " > res.m" << res.m << ", or coff " << coff << " + n " << n << " > res.n " << res.n);
		dthrow(outOfBounds());
	}

	if(contiguousQ() && res.contiguousQ() && roff == 0 && coff == 0) {
		if(!onlyDevice && (!res.elements && elements)) {
			dthrow(noHostBuffer());
		}
		if(!res.d_elements && d_elements) {
			dthrow(noDeviceBuffer());
		}

		if(elements && !onlyDevice) {
			cherr( cudaMemcpy(res.elements, elements, size, cudaMemcpyHostToHost));
			HHCopied++;
			MemHhCopied +=size;
			if(checkDebug(debugCopy)) outln("host copied " << toShortString() << " to " << res.toShortString());
		}
		if(d_elements) {
			cherr( cudaMemcpy(res.d_elements, d_elements, size, cudaMemcpyDeviceToDevice));
			DDCopied++;
			MemDdCopied += size;
			if(checkDebug(debugCopy)) outln("dev copied " << toShortString() << " to " << res.toShortString());
		}
	} else {
		DMatrix<T> d_res, d_M;
		asDmatrix(d_M);
		res.asDmatrix(d_res);
		copy(d_res, d_M, roff, coff);
		res.lastMod = mod_device;
	}
}

// tiles l-to-r
template<typename T> void CuMatrix<T>::concat(CuMatrix<T>& canvas, int components, const CuMatrix<T>** parts) {
	if(checkDebug(debugMem))outln("concat with canvas " << canvas.toShortString());
    ulong canvasSize = 0;
    int dcount =0, hcount=0;
    for(int i = 0; i < components; i++) {
    	const CuMatrix<T>* c = parts[i];
    	switch(c->lastMod) {
			case mod_host:
				hcount++;
				break;
			case mod_device:
				dcount++;
				break;
    	}
    	canvasSize += c->size;
    }
    if(checkDebug(debugMem))outln("concat canvasSize " << canvasSize);
    if(checkDebug(debugMem))outln("concat dcount " << dcount << ", hcount " << hcount << (hcount == 0 ? ";  only copying dmem":""));
	uint n =  canvasSize/sizeof(T);
	//CuMatrix<T> canvas(1, n, n, false, true);
	if(canvas.d_elements){
		if(checkDebug(debugMem))outln("canvas had d_el " << canvas.d_elements);
		if(canvas.size != canvasSize) {
			if(checkDebug(debugMem))outln("canvas " << canvas.toShortString() << " size != " << canvasSize << " freeing old d mem " << canvas.d_elements);
			canvas.getMgr().freeDevice(canvas);
			if(canvas.elements) {
				outln("\talso freeing h mem " << canvas.elements);
				canvas.getMgr().freeHost(canvas);
			}
			canvas.elements=canvas.d_elements = null;
		}
	}
	canvas.size = canvasSize;
	canvas.n = n;
	canvas.m = 1;
	canvas.p = n;
	if(!canvas.d_elements) {
		canvas.getMgr().allocDevice(canvas);
	}

	if(checkDebug(debugMem))outln("concat having canvas.m " << canvas.m << ", n " << canvas.n << ", size " << canvas.size);
	DMatrix<T> dret;
	canvas.asDmatrix(dret,false);
	int streamCount = 2 * components;
	cudaEvent_t cycleDone[streamCount];
	cudaStream_t stream[streamCount];
	for(int i = 0; i < streamCount; i++) {
		cherr(cudaStreamCreate(&stream[i]));
        cherr(cudaEventCreate(&cycleDone[i]));
	}
	int next_stream = 0;
	uint offset = 0;
	uint len = 0;
	for(int i = 0; i < components; i++) {
		const CuMatrix<T>* currMat = parts[i];
		len = currMat->m * currMat->n;
		if( hcount != 0) {
			if(checkDebug(debugCheckValid))MemMgr<T>::checkValid(canvas.elements);
			if(checkDebug(debugCheckValid))MemMgr<T>::checkValid(canvas.elements + offset);
			if(checkDebug(debugMem))outln("concat copying h2h " << currMat->toShortString() << " using cudaMemcpyAsync\n\t\tcopying " << len << " host Ts from " << currMat->elements << " to " << (canvas.elements + offset));
			cherr(cudaMemcpyAsync(
							  canvas.elements + offset,
							  currMat->elements,
							  len * sizeof(T),
							  cudaMemcpyHostToHost,
							  stream[next_stream]));

			HHCopied++;
			MemHhCopied += len * sizeof(T);
			cherr(cudaEventRecord(
								cycleDone[next_stream],
								stream[next_stream]));
			next_stream +=1;
		} else {
			if(checkDebug(debugMem))outln("concat skipping host copy (hcount == 0)");
		}
		if(checkDebug(debugCheckValid))MemMgr<T>::checkValid(canvas.d_elements);
		if(checkDebug(debugCheckValid))MemMgr<T>::checkValid(canvas.d_elements + offset);
		if(checkDebug(debugMem))outln("concat copying d2d " << currMat->toShortString() <<
				" using cudaMemcpyAsync\n\t\tcopying " << len << " dev Ts from " << currMat->d_elements << " to " << (canvas.d_elements + offset)) <<
				" ie " << canvas.d_elements << " plus offset " << offset << endl ;
		if(checkDebug(debugMem))outln("&canvas.d_elements[len] " << &canvas.d_elements[len]);

		cherr(cudaMemcpyAsync(
							  canvas.d_elements + offset,
							  currMat->d_elements,
							  len * sizeof(T),
							  cudaMemcpyDeviceToDevice,
							  stream[next_stream]));
		DDCopied++;
		MemDdCopied += len * sizeof(T);

		cherr(cudaEventRecord(
							cycleDone[next_stream],
							stream[next_stream]));
		next_stream +=1;
    	offset += len;

	}
	canvas.lastMod = hcount==0 ? mod_device : mod_synced;
	if(checkDebug(debugFill))outln("concat made canvas " << canvas.toShortString() << "\n\n");
	for(int i = 0; i < streamCount; i++) {
		cherr(cudaStreamDestroy(stream[i]));
	}
}

template<typename T> __host__ __device__ cudaError_t CuMatrix<T>::rowCopy(CuMatrix<T>& trg, uint tRow, uint sRow) const {
	IndexArray tary = trg.rowIndices(tRow);
	IndexArray sary = rowIndices(sRow);
/*
	outln("tRow " << tRow << " tary " << tary);
	outln("sRow " << sRow << " sary " << sary);
*/
	return copyIndexed(trg, tary, *this, sary);
}

/*
 * uses
 */
template<typename T> __host__ __device__ CuMatrix<T> CuMatrix<T>::derefIndices(const IndexArray& indices) const {
	CuMatrix<T> deref = CuMatrix<T>::zeros(indices.count, n);
	for(int i = 0; i < indices.count; i++) {
		rowCopy(deref, i, indices.indices[i]);
	}
	return deref;
}

template<typename T> cudaError_t CuMatrix<T>::copyIndexed(CuMatrix<T>& trg, const IndexArray& tary,  const CuMatrix<T>& src, const IndexArray& sary) {
	if(!src.d_elements || !trg.d_elements) {
		dthrow(noDeviceBuffer());
	}

	if(tary.count == sary.count && sary.count == 2) {
		// rowMajr to rowMaj
		//cudaMemcpy(targ.elements + tary.indices[0], src.elements + sary.indices[0], (tary.indices[1]-tary.indices[0])* sizeof(T), cudaMemcpyHostToHost);
		//flprintf("trg.d_elements  %p tary.indices[0] %d src.d_elements %p sary.indices[0] %d\n", trg.d_elements, tary.indices[0], src.elements , sary.indices[0] );
		cherr(cudaMemcpy(trg.d_elements + tary.indices[0], src.d_elements + sary.indices[0], (1 + tary.indices[1]-tary.indices[0])* sizeof(T),cudaMemcpyDeviceToDevice));
		DDCopied++;
		MemDdCopied += (tary.indices[1]-tary.indices[0])* sizeof(T);
		trg.invalidateHost();
		return cudaSuccess;
	} else if(tary.count == 2 && sary.count > 2) {
		uint start = tary.indices[0];
		uint tlen = tary.indices[1] - tary.indices[0];
		dassert(tlen == sary.count);
		for(uint i = 0; i < sary.count; i++) {
			trg.elements[start + i] = src.elements[ sary.indices[i]];
		}
	} else if(sary.count == 2 && tary.count > 2) {
		uint start = sary.indices[0];
		uint slen = sary.indices[1] - sary.indices[0];
		dassert(slen == tary.count);
		for(uint i = 0; i < tary.count; i++) {
			trg.elements[tary.indices[i]] = src.elements[start + i];
		}
	} else {
		outln("error, bad source indexarray " << sary.toString().c_str() << " or bad targ array " << tary.toString().c_str());
		return cudaErrorInvalidConfiguration;
	}
	trg.invalidateDevice();
	return cudaSuccess;
}

/*
 *  todo implement randSequence as a kernel on column or row matrix
 * 		whose re-arrangment when sorted (idx0 -> idx0sorted ...) is applied to the original sequence
 */
template<typename T> void CuMatrix<T>::shuffle(CuMatrix<T>& trg, CuMatrix<T>& leftovers, T fraction, vector<uint>& vIndices ) const {
	if( !(fraction >= 0. && fraction <= 1.)) {
		dthrow(outOfBounds());
	}
	if(d_elements == null){
		dthrow(noDeviceBuffer());
	}
	if(lastMod == mod_host) {
		dthrow(notSynceCUDART_DEVICE());
	}

	uint rows;

	if(integralTypeQ())  {
		rows = round(m *  fraction/100.);
	} else {
		rows = round(m * (double)fraction);
	}

	trg.m = rows;
	trg.n = trg.p = n;
	trg.size = trg.m * trg.p * sizeof(T);
	trg.getMgr().allocDevice(trg);
	if(rows == m) {
		leftovers = ZeroMatrix;
	} else {
		leftovers.m = m - rows;
		leftovers.n = leftovers.p = n;
		leftovers.size = leftovers.m *  leftovers.p * sizeof(T);
		leftovers.getMgr().allocDevice(leftovers);
	}

	// re-use passed-in index buffer, to keep multple sample matrices in sync
	if(vIndices.size() == 0 ) {
		b_util::randSequence(vIndices, m, 0);
	} else if (vIndices.size() != m) {
		outln("shuffle passed a row index vector, but it was the wrong size (" << vIndices.size() << " <> " <<  m << ")");
		dthrow(badDimensions());
	}

	if(checkDebug(debugFill))outln("vIndices\n" << b_util::pvec(vIndices));
	uint* indices, *d_indices;
	uint indexSize = m * sizeof(uint);
	cherr( cudaHostAlloc( (void**)&indices, indexSize, 0));
	b_util::toArray(indices, vIndices, 0, rows);
	cherr( cudaMalloc( (void**)&d_indices, indexSize));
	cherr(cudaMemcpy(d_indices, indices,indexSize, cudaMemcpyHostToDevice));
	HDCopied++;
	MemHdCopied += indexSize;
	DMatrix<T> s, t, l;
	asDmatrix(s,false,false);
	trg.asDmatrix(t,false,false);
	shuffleCopyRows(t,s, d_indices);

	trg.lastMod = mod_device;

	cherr(cudaDeviceSynchronize());

	if( !leftovers.zeroDimsQ()) {
		indexSize = leftovers.m * sizeof(uint);
		if(leftovers.m > rows) {
			// need a bigger index buffer
			cherr( cudaFreeHost(indices));
			cherr( cudaHostAlloc( (void**)&indices, indexSize, 0));
			cherr( cudaFree( d_indices));
			cherr( cudaMalloc( (void**)&d_indices, indexSize));
		}
		b_util::toArray( indices, vIndices, rows, leftovers.m);
		cherr(cudaMemcpy(d_indices, indices, indexSize, cudaMemcpyHostToDevice));
		HDCopied++;
		MemHdCopied += indexSize;
		leftovers.asDmatrix(l,false,false);
		shuffleCopyRows(l,s, d_indices);
		leftovers.lastMod = mod_device;
	}

	cherr(cudaDeviceSynchronize());
	cherr(cudaFreeHost(indices));
	cherr( cudaFree( d_indices));

}

template<typename T> void CuMatrix<T>::toDevice(int dev) {
	getMgr().migrate(dev, *this);
}

template<typename T> __host__ __device__ void CuMatrix<T>::zero() {
	if(!d_elements) {
		setLastError(noDeviceBufferEx);
	}
	if(!lastMod == mod_host) {
		setLastError(notSynceCUDART_DEVICEEx);
	}
#ifndef __CUDA_ARCH__
	checkCudaError(cudaMemset(d_elements, 0, size));
#else
	memset(d_elements, 0, size);
#endif
	lastMod = mod_device;
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::rightConcatenate(
		 const CuMatrix<T>& other) const {
	if(other.zeroDimsQ()) {
		return *this;
	}
	if(zeroDimsQ()) {
		return other;
	}
	if(other.m != m) {
		setLastError(matricesOfIncompatibleShapeEx);
	}
	//outln("this " << toShortString());
	//outln("other " << other.toShortString());
	if(! gpuReadyQ() ) {
		setLastError(notSyncedEx);
	}
	if(! other.gpuReadyQ() ) {
		setLastError(notSyncedEx);
	}

	uint newCols = n + other.n;
	CuMatrix<T> ret(m, newCols,false, true);
	if (colMajor){
		setLastError (  notImplementedEx);
	} else {
		DMatrix<T> d_A, d_B, d_Res;
		asDmatrix(d_A);
		other.asDmatrix(d_B);
		ret.asDmatrix(d_Res,false);
		rightConcatenateL(d_Res, d_A,d_B);
	}
	ret.lastMod = mod_device;
	cherr(cudaPeekAtLastError());
	//outln("returning rightCat res " << ret.toShortString());
	return ret;
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::bottomConcatenate(
		 const CuMatrix<T>& other) const {
	if(other.zeroDimsQ()) {
		return *this;
	}
	if(zeroDimsQ()) {
		return other;
	}
	if (other.n != n) {
#ifndef __CUDA_ARCH__
		dthrow (  matricesOfIncompatibleShape());
#else
		setLastError(matricesOfIncompatibleShapeEx);
#endif
	}
	uint newRows = m + other.m;
	CuMatrix<T> ret(newRows, n,false,true);
	if (colMajor) {
#ifndef __CUDA_ARCH__
		dthrow ( notImplemented() );
#else
		setLastError(notImplementedEx);
#endif
	} else {
		DMatrix<T> d_A, d_B, d_Res;
		asDmatrix(d_A);
		other.asDmatrix(d_B);
		ret.asDmatrix(d_Res,false);
		bottomConcatenateL(d_Res, d_A,d_B);
	}
	ret.lastMod = mod_device;
	return ret;
}

template <typename T> __global__ void setKernel(T* elements, uint p, uint row, uint col, T val) {
	elements[row * p + col] = val;
}


template<typename T> __host__ __device__ void CuMatrix<T>::set(uint r, uint c, T val) {
	if (r >= m || c >= n)
		setLastError(outOfBoundsEx);
	if(!d_elements) {
		setLastError(noDeviceBufferEx);
	}
	uint idx = colMajor ? c * p + r : r*p + c;
	set(idx, val);
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::set(uint l, T val) {
	if (l >= size / sizeof(T))
		setLastError(outOfBoundsEx);
	if(!d_elements) {
		setLastError(noDeviceBufferEx);
	}
	::set(d_elements, m, n, p, l, val);
	invalidateHost();
}

template<typename T> __host__ __device__ T CuMatrix<T>::get(uint l) const {
	if (l >= size / sizeof(T))
		setLastError(outOfBoundsEx);
	if(!d_elements) {
		setLastError(noDeviceBufferEx);
	}
#ifndef __CUDA_ARCH__
	if(lastMod == mod_synced || lastMod == mod_host) {
		return elements[l];
	}
	T res;
	cherr(cudaMemcpy(&res, d_elements + l, sizeof(T), cudaMemcpyDeviceToHost));
	if(checkDebug(debugCopyDh))outln("debugCopyDh " << "CuMatrix<T>::get");
	DHCopied++;
	MemDhCopied += sizeof(T);
	return res;
#else
	if(lastMod == mod_synced || lastMod == mod_device) {
		return d_elements[l];
	} else {
		setLastError(notSynceCUDART_DEVICEEx);
		return -1;
	}
#endif
}

template<typename T> __host__ __device__ T CuMatrix<T>::get(uint r, uint c) const {
	if (r >= CuMatrix<T>::m || c >= CuMatrix<T>::n)
		setLastError(outOfBoundsEx);
	return get(r * p + c);
}

template <typename T> void CuMatrix<T>::linterp(CuMatrix<T>& result, const CuMatrix<T>& src, const CuMatrix<T>& dest, T factor) {
	result = src + factor * (dest-src);
}


#include "CuMatrixInster.cu"

