#include "CuMatrix.h"
#include "util.h"
#include "debug.h"
#include "caps.h"
#include "MatrixExceptions.h"
#include <set>

template <typename T> __global__ void binaryCategoryKernel(const T* sElements, T* tElements, int width, int height, bool oneBased)
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
	//if(checkDebug(debugNn))outln("binCat blockW " << blockW << ", blockH " << blockH << " block " << b_util::pd3(block));
    dim3 grid(DIV_UP( t.n, block.x), DIV_UP(t.m, block.y));
	int smem = block.x * block.y * sizeof(T);
	binaryCategoryKernel<<<grid, block, smem>>>(s.elements, t.elements, t.n, t.m, oneBased);
	//if(checkDebug(debugNn))outln("binCatKernel with grid " << b_util::pd3(grid).c_str() << " of block " << b_util::pd3(block).c_str() << " smem " << smem);
	checkCudaError(cudaDeviceSynchronize());
}

/*
 * takes a column vector of classes whose each row is the 'max' category (often the category/class with the highest probability),
 * converts that to a rect matrix whose width is the number of categories and whose value is
 * zero for each column except for the category of the row in the original column vector
 * (inverse operation is toMaxColumnIndexVector)
 */
template<typename T> CuMatrix<T> CuMatrix<T>::toBinaryCategoryMatrix() const {
	const uint len = m * n;
	if(!elements ) dthrow(noHostBuffer());

	if(lastMod == mod_device) {
		outln("toBinCat mod_device " << toShortString());
		dthrow(notSyncedHost());
	}

	std::set<T> s(elements, elements + len); 	// TODO make a kernel for this (self-reduction until steady state?)
	bool oneBased = (s.find(0) == s.end());

	uint newCols = s.size();
	CuMatrix<T> res(m, newCols,false,true);
	DMatrix<T> d_res = res.asDmatrix(false);
	DMatrix<T> d_src = asDmatrix();

	binaryCategoryKernelL(d_res, d_src, oneBased);
	res.lastMod=mod_device;
	if(checkDebug(debugNn))outln("res " << res.toShortString());
	return res;
}

template<typename T> ulong CuMatrix<T>::distinctCount() const {

	const ulong len = m * n;
	if(lastMod == mod_device) {
		dthrow(notSyncedHost());
	}

	std::set<T> s(elements, elements + (uint)len); 	// TODO make a kernel for this (self-reduction until steady state?)

	return (ulong) s.size();
}

template<typename T> std::set<T> CuMatrix<T>::distinct() const {

	const ulong len = m * n;
	if(lastMod == mod_device) dthrow(notSyncedHost());

	std::set<T> s(elements, elements + (uint)len); 	// TODO make a kernel for this (self-reduction until steady state?)

	return s;
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

template<typename T> CuMatrix<T>& CuMatrix<T>::unPose() {
	outln("CuMatrix<T>::unPose() entre; posed && oldM != 0:  " << (posed && oldM != 0));
	cherr(cudaPeekAtLastError());
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
	cherr(cudaPeekAtLastError());
	posed = false;
	outln("CuMatrix<T>::unPose() entre; posed && oldM != 0:  " << (posed && oldM != 0));
	return *this;
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::reshape(CuMatrix<T>& target, int rows, int cols, ulong offsetInTs) {
	if(!tiler.hasDmemQ()) setLastError(noDeviceBufferEx); else if(checkDebug(debugMem)) prlocf("reshape have nz d buff");
	if(!target.tiler.hasDmemQ() ) setLastError(noDeviceBufferEx); else if(checkDebug(debugMem)) prlocf("reshape have nz ret dbuff");
	uint l = rows * cols;
	assert(tiler.tileSize == tiler.m_size);
	T* sd_elements = tiler.currBuffer();
	T* td_elements = target.tiler.currBuffer();
	if(contiguousQ()) {
		if(gpuReadyQ()) {
#ifndef __CUDA_ARCH__
			cherr(
				cudaMemcpy(td_elements, sd_elements + offsetInTs, l * sizeof(T), cudaMemcpyDeviceToDevice));
			DDCopied++;
			MemDdCopied += l *sizeof(T);
#else
	#ifdef CuMatrix_Enable_Cdp
				cherr(
					cudaMemcpyAsync(td_elements, sd_elements + offsetInTs, l * sizeof(T), cudaMemcpyDeviceToDevice));
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
		tiler.tile0(src,true);
		target.tile0(trg,false);
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
	if(checkDebug(debugMem)) {
		flprintf("CuMatrix (%p)::reshaped( %u * %u, off %u ) -> %p setLastMod %s\n",this,rows,cols,offsetInTs,&target, b_util::modStr(lastMod));
	}
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::reshape(int rows, int cols,
		long offsetInTs) {
	CuMatrix<T> res(rows, cols,false,true);
	reshape(res, rows, cols, offsetInTs);
	return res;
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::recreate(int rows, int cols, int pitch, bool allocH,bool allocD) {
	assert(m == 0 && n == 0);
	m = rows;
	n = cols;
	p = pitch;
	updateSize();
	//size = p * m * sizeof(T);
#ifndef __CUDA_ARCH__
	if(allocH) mgr->allocHost(*this);
#endif
	if(allocD) {
		tiler.set(*this);
		tiler.allocTiles();
	}
}

template<typename T> CuMatrix<T> CuMatrix<T>::redimension(
		pair<int, int>& dims, int offset) {
	return reshape(dims.first, dims.second, offset);
}

// tiles l-to-r
// assumes p = n
template<typename T> void CuMatrix<T>::concat(CuMatrix<T>& canvas,
		int components, const CuMatrix<T>** parts, bool colMajor) {
	if(checkDebug(debugMem))outln("concat with canvas " << canvas.toShortString());
    ulong canvasSize = 0;
    int dcount =0, hcount=0;
    for(int i = 0; i < components; i++) {
    	const CuMatrix<T>* c = parts[i];
    	if(checkDebug(debugMem))outln("concat with c[" << i << "] " << parts[i]->toShortString());
    	switch(c->lastMod) {
			case mod_host:
				hcount++;
				break;
			case mod_device:
				dcount++;
				break;
			case mod_synced:
				dcount++;
				hcount++;
				break;
    	}
    	canvasSize += c->size;
    }
    if(checkDebug(debugMem))outln("concat canvasSize " << canvasSize);
    if(checkDebug(debugMem))outln("concat dcount " << dcount << ", hcount " << hcount << (hcount == 0 ? ";  only copying dmem":""));
	int n =  canvasSize/sizeof(T);
	//CuMatrix<T> canvas(1, n, n, false, true);
	if(! canvas.tiler.hasDmemQ() || ( canvas.tiler.hasDmemQ() && canvas.size != canvasSize) ){
		if(canvas.tiler.hasDmemQ()) {
			if(checkDebug(debugMem))outln("canvas " << canvas.toShortString() << " size != " << canvasSize << " freeing old d mem " << canvas.tiler.currBuffer());
			canvas.getMgr().freeTiles(canvas);
			if(canvas.elements) {
				outln("\talso freeing h mem " << canvas.elements);
				canvas.getMgr().freeHost(canvas);
			}
		}
		canvas.elements = null;
		canvas.tiler.buffers = {0,0,0,0};
		canvas.size = canvasSize;
		canvas.m = n;
		canvas.n = 1;
		canvas.p = 1;
		canvas.tiler.m_size = canvas.tiler.tileSize = canvasSize;
		canvas.tiler.m_m = n;
		canvas.tiler.m_n = 1;
		canvas.tiler.m_p = 1;
		canvas.tiler.allocTiles();
		if(checkDebug(debugCheckValid)) flprintf("%dx%dx%d -:> %p\n", n, 1, 1, canvas.tiler.buff());
		canvas.getMgr().addTiles(&canvas.tiler);
	}

	if(!canvas.elements) {
		canvas.getMgr().allocHost(canvas);
	}

	if(checkDebug(debugMem))outln("concat having canvas.m " << canvas.m << ", n " << canvas.n << ", size " << canvas.size);
	DMatrix<T> dret;
	canvas.tile0(dret,false);
	int streamCount = 2 * components;
	if(checkDebug(debugMem))outln("concat streamCount " << streamCount);
	cudaEvent_t cycleDone[streamCount];
	cudaStream_t stream[streamCount];
	for(int i = 0; i < streamCount; i++) {
		cherr(cudaStreamCreate(&stream[i]));
		if(checkDebug(debugMem))outln("concat created stream " << stream[i]);
        cherr(cudaEventCreate(&cycleDone[i]));
	}
	int next_stream = 0;
	uint offset = 0;
	uint len = 0;
	for(int i = 0; i < components; i++) {
		const CuMatrix<T>* currMat = parts[i];
		assert(currMat->n == currMat->p);  // TODO array2d
		len = currMat->m * currMat->n;
		if( hcount != 0) {
			if(checkDebug(debugCheckValid))MemMgr<T>::checkValid(canvas.elements);
			if(checkDebug(debugCheckValid))MemMgr<T>::checkValid(canvas.elements + offset);
			if (checkDebug(debugMem)) {
				outln("concat copying h2h " << currMat->toShortString()
						<< " using cudaMemcpyAsync\n\t\tcopying " << len <<
						" host Ts from " << currMat->elements <<
						" to " << (canvas.elements + offset));

			}
			if (checkDebug(debugMem))
				flprintf("void *dst %p, const void *src %p, size_t count %d\n",
						canvas.elements + offset, currMat->elements,
						len * sizeof(T));

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
		if(checkDebug(debugMem))outln("concat copying d2d " << currMat->toShortString() <<
				" using cudaMemcpyAsync\n\t\tcopying " << len << " dev Ts from " << currMat->tiler.currBuffer() << " to " << (canvas.tiler.currBuffer() + offset)) <<
				" ie " << canvas.tiler.currBuffer() << " plus offset " << offset << endl ;
		MemMgr<T>::checkValid(canvas.tiler.currBuffer() + offset,"canvas.tiler.currBuffer() + offset");
		MemMgr<T>::checkValid(canvas.tiler.currBuffer() + offset + len - 1, "canvas.tiler.currBuffer() + offset + len");
		MemMgr<T>::checkValid( currMat->tiler.currBuffer(), " currMat->tiler.currBuffer()");
		MemMgr<T>::checkValid( currMat->tiler.currBuffer() + len - 1, " currMat->tiler.currBuffer() + len");
		if(checkDebug(debugMem))outln("&canvas.tiler.currBuffer()[len] " << &canvas.tiler.currBuffer()[len]);
		if(checkDebug(debugMem))outln("next_stream " << next_stream << ", stream[next_stream] " << stream[next_stream] );

		if(canvas.tiler.tileSize == canvas.tiler.m_size) {
			cherr(cudaMemcpyAsync(
								  canvas.tiler.currBuffer() + offset,
								  currMat->tiler.currBuffer(),
								  len * sizeof(T),
								  cudaMemcpyDeviceToDevice,
								  stream[next_stream]));
			DDCopied++;
			MemDdCopied += len * sizeof(T);
		}else {
			assert(false);
		}
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

template<typename T> void CuMatrix<T>::concat(CuMatrix<T>& canvas,
		vector< CuMatrix<T> >parts, bool colMajor) {
	if(checkDebug(debugMem))outln("concat with canvas " << canvas.toShortString());
    ulong canvasSize = 0;
    int dcount =0, hcount=0;
    const int components = parts.size();
    for(int i = 0; i < components; i++) {
    	const CuMatrix<T> c = parts.at(i);
    	if(checkDebug(debugMem)) outln("concat with c[" << i << "] " << c.toShortString());
    	switch(c.lastMod) {
			case mod_host:
				hcount++;
				break;
			case mod_device:
				dcount++;
				break;
			case mod_synced:
				dcount++;
				hcount++;
				break;
    	}
    	canvasSize += c.size;
    }
    if(checkDebug(debugMem))outln("concat canvasSize " << canvasSize);
    if(checkDebug(debugMem))outln("concat dcount " << dcount << ", hcount " << hcount << (hcount == 0 ? ";  only copying dmem":""));
	int n =  canvasSize/sizeof(T);
	//CuMatrix<T> canvas(1, n, n, false, true);
	if(! canvas.tiler.hasDmemQ() || ( canvas.tiler.hasDmemQ() && canvas.size != canvasSize) ){
		if(canvas.tiler.hasDmemQ()) {
			if(checkDebug(debugMem))outln("canvas " << canvas.toShortString() << " size != " << canvasSize << " freeing old d mem " << canvas.tiler.currBuffer());
			canvas.getMgr().freeTiles(canvas);
			if(canvas.elements) {
				if(checkDebug(debugMem)) outln("\talso freeing h mem " << canvas.elements);
				canvas.getMgr().freeHost(canvas);
			}
		}
		canvas.elements = null;
		canvas.tiler.buffers = {0,0,0,0};
		canvas.size = canvasSize;
		canvas.m = n;
		canvas.n = 1;
		canvas.p = 1;
		canvas.tiler.m_size = canvas.tiler.tileSize = canvasSize;
		canvas.tiler.m_m = n;
		canvas.tiler.m_n = 1;
		canvas.tiler.m_p = 1;
		canvas.tiler.allocTiles();
		if(checkDebug(debugCheckValid)) flprintf("%dx%dx%d -:> %p\n", n, 1, 1, canvas.tiler.buff());
		canvas.getMgr().addTiles(&canvas.tiler);
	}

	if(!canvas.elements) {
		canvas.getMgr().allocHost(canvas);
	}

	if(checkDebug(debugMem))outln("concat having canvas.m " << canvas.m << ", n " << canvas.n << ", size " << canvas.size);
	DMatrix<T> dret;
	canvas.tile0(dret,false);
	int streamCount = 2 * components;
	if(checkDebug(debugMem))outln("concat streamCount " << streamCount);
	cudaEvent_t cycleDone[streamCount];
	cudaStream_t stream[streamCount];
	for(int i = 0; i < streamCount; i++) {
		cherr(cudaStreamCreate(&stream[i]));
		if(checkDebug(debugMem))outln("concat created stream " << stream[i]);
        cherr(cudaEventCreate(&cycleDone[i]));
	}
	int next_stream = 0;
	uint offset = 0;
	uint len = 0;
	for(int i = 0; i < components; i++) {
		CuMatrix<T> currMat = parts.at(i);
		assert(currMat.n == currMat.p);  // TODO array2d
		len = currMat.m * currMat.n;
		if( hcount != 0) {
			if(checkDebug(debugCheckValid))MemMgr<T>::checkValid(canvas.elements);
			if(checkDebug(debugCheckValid))MemMgr<T>::checkValid(canvas.elements + offset);
			if (checkDebug(debugMem)) {
				outln("concat copying h2h " << currMat.toShortString()
						<< " using cudaMemcpyAsync\n\t\tcopying " << len <<
						" host Ts from " << currMat.elements <<
						" to " << (canvas.elements + offset));

			}
			if (checkDebug(debugMem))
				flprintf("void *dst %p, const void *src %p, size_t count %d\n",
						canvas.elements + offset, currMat.elements,
						len * sizeof(T));

			cherr(cudaMemcpyAsync(
							  canvas.elements + offset,
							  currMat.elements,
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
		if(checkDebug(debugMem))outln("concat copying d2d " << currMat.toShortString() <<
				" using cudaMemcpyAsync\n\t\tcopying " << len << " dev Ts from " << currMat.tiler.currBuffer() << " to " << (canvas.tiler.currBuffer() + offset)) <<
				" ie " << canvas.tiler.currBuffer() << " plus offset " << offset << endl ;
		MemMgr<T>::checkValid(canvas.tiler.currBuffer() + offset,"canvas.tiler.currBuffer() + offset");
		MemMgr<T>::checkValid(canvas.tiler.currBuffer() + offset + len - 1, "canvas.tiler.currBuffer() + offset + len");
		MemMgr<T>::checkValid( currMat.tiler.currBuffer(), " currMat.tiler.currBuffer()");
		MemMgr<T>::checkValid( currMat.tiler.currBuffer() + len - 1, " currMat.tiler.currBuffer() + len");
		if(checkDebug(debugMem))outln("&canvas.tiler.currBuffer()[len] " << &canvas.tiler.currBuffer()[len]);
		if(checkDebug(debugMem))outln("next_stream " << next_stream << ", stream[next_stream] " << stream[next_stream] );

		if(canvas.tiler.tileSize == canvas.tiler.m_size) {
			cherr(cudaMemcpyAsync(
								  canvas.tiler.currBuffer() + offset,
								  currMat.tiler.currBuffer(),
								  len * sizeof(T),
								  cudaMemcpyDeviceToDevice,
								  stream[next_stream]));
			DDCopied++;
			MemDdCopied += len * sizeof(T);
		}else {
			assert(false);
		}
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

/*
 * assumes source is a vector, not array (so no pitch concerns)
 * 	offset in Ts, not bytes
 */
template<typename T> void CuMatrix<T>::unconcat(CuMatrix<T>& v, int rows, int cols, int pitch, int offset, bool colMajor) const {
	if(!vectorQ()){
		dthrow(notVector());
	}
	assert( tiler.tileSize == tiler.m_size );
	if(offset + rows * cols > m * n) {
		if(checkDebug(debugCheckValid)) outln("invalid submatrix (off " << offset << " rows " << rows << " X cols " << cols << " ==  " << (offset + rows * cols )<< " > m " << m << " X n " << n << " == " << m * n);
		dthrow(badDimensions());
	}

	if(v.m == 0 && v.n == 0 ) {
		assert(cols == pitch);// only because reshape assumes?
		v.recreate(rows,cols,pitch,false,true);
		v.invalidateHost();
	}

	if(checkDebug(debugCheckValid)) flprintf("elements %p\n", elements);
	if(checkDebug(debugCheckValid)) outln("v now " << v.toShortString());
	assert(v.m == rows && v.n == cols && v.p == pitch);

	if(lastMod == mod_device ||
			lastMod == mod_synced ) {
		// cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
		if(checkDebug(debugCheckValid))outln("v.tiler.currBuffer()  "<< v.tiler.currBuffer() << ", v.p * sizeof(T) " <<  v.p * sizeof(T) <<
				"tiler.currBuffer() " << tiler.currBuffer() << ", offest " << offset << ", p* sizeof(T) "<<p* sizeof(T) << ", rows " << rows <<
				", cols * sizeof(T) " << cols * sizeof(T));
		cherr(cudaMemcpy(v.tiler.currBuffer(), tiler.currBuffer() + offset, rows * cols * sizeof(T), cudaMemcpyDeviceToDevice));
		v.lastMod = mod_device;
	} else {
		if(true || checkDebug(debugCheckValid))outln("v.elements  "<< v.elements << ", v.p * sizeof(T) " <<  v.p * sizeof(T) <<


				", elements  " << elements  << ", offset " << offset  << offset << ", p* sizeof(T) "<<p* sizeof(T) << ", rows " << rows <<
				", cols * sizeof(T) " << cols * sizeof(T));

		cherr(cudaMemcpy(v.elements, elements + offset, rows * cols * sizeof(T), cudaMemcpyHostToHost));
		v.lastMod = mod_host;
	}
	if(!v.tiler.m_size) {
		  outln("unconcat m_size == 0  v " << v.toShortString());
	}
	v.syncBuffers();
	if(colMajor) {
		T* buff;
		int len = v.m * v.n;
		cherr(cudaHostAlloc(&buff, (size_t) v.size,cudaHostAllocMapped));
		for(int i =0; i < len; i++){
			int colIdx = i % n * m + i / n;
			buff[i] = v.elements[colIdx];
		}
		cherr(cudaMemcpy(v.elements, buff, v.size, cudaMemcpyHostToHost));
		MemMgr<T>::checkValid(buff);
		outln("freeing host " << buff);

		cherr(cudaFreeHost(buff));
		v.invalidateDevice();
		v.syncBuffers();
	}

}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::submatrix(CuMatrix<T>& v, int rows, int cols, int roff, int coff) const {
	if(roff + rows > m || coff + cols > n) {
		prlocf("invalid submatrix ( > this)");
		setLastError(badDimensionsEx);
	}
	assert(tiler.tileSize == tiler.m_size);
	uint offset = roff * p + coff;
	v.elements =  elements ? elements + offset : null ;
	int currDev = ExecCaps::currDev();
	v.tiler.setBuffer(currDev, tiler.buffer(currDev) ? tiler.buffer(currDev) + offset : 0);
	v.m = rows;
	v.n = cols;
	v.p = p;
	v.tiler.m_size = v.tiler.tileSize = v.size = v.m * v.p * sizeof(T);
	v.tiler.m_m = v.m;
	v.tiler.m_n = v.n;
	v.tiler.m_p = v.p;

	v.lastMod = lastMod;
	v.ownsBuffers = false;

}
// crap
template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::columnMatrix(int col) const {

#ifndef __CUDA_ARCH__
	if(checkDebug(debugTiler)) outln(toShortString());
#endif

	assert(tiler.tileSize == tiler.m_size);

	CuMatrix<T> column(m, 1, 1, false, true);
	DMatrix<T> dsrc, dtrg;
	dsrc.elements = currBuffer();
	dsrc.m = m;
	dsrc.n = 1;
	dsrc.p = p;
	dtrg.elements = column.currBuffer();
	dtrg.m = m;
	dtrg.n = 1;
	dtrg.p = 1;
	CuMatrix<T>::copy(dtrg, dsrc, 0, 0);

	/*column.elements = elements + col;
	assert(column.size == m*p*sizeof(T) );
	column.ownsBuffers = false;
	if(checkDebug(debugTiler)) flprintf("tiler.currBuffer() %p\n", tiler.currBuffer());
	column.tiler.setCurrBuffer( tiler.currBuffer() +  col);
	column.tiler.reset(column);
*/
	column.invalidateHost();
	column.syncBuffers();

	return column;
}

template<typename T> CuMatrix<T> CuMatrix<T>::rowMatrix(int row) const {
	CuMatrix<T> rowm(1,n, false, true);
	assert(tiler.tileSize == tiler.m_size);
	if(colMajor) {
		DMatrix<T> d_X, d_row;
		tile0(d_X, lastMod == mod_host);
		rowm.tile0(d_row, false);
		rowMatrixCmL(d_row, d_X, row);
	} else {
		checkCudaError(cudaMemcpy(rowm.tiler.currBuffer(), tiler.currBuffer() + row*p, n*sizeof(T), cudaMemcpyDeviceToDevice));
		DDCopied++;
		MemDdCopied += (n - 1) * sizeof(T);
	}
	return rowm;
}

template<typename T> CuMatrix<T> CuMatrix<T>::dropFirst(bool copy) const {
	if(lastMod == mod_host) dthrow(notSynced());

	CuMatrix<T> res(m, n - 1, false, copy);
	assert(tiler.tileSize == tiler.m_size);
	if(copy){
		uint i = 0;
		while (i < m) {
			checkCudaError(
					cudaMemcpy(res.tiler.currBuffer() + i * (n - 1), tiler.currBuffer() + i * n + 1, (n - 1) * sizeof(T), cudaMemcpyDeviceToDevice));
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
	assert(tiler.tileSize == tiler.m_size);
	if(copy){
		uint i = 0;
		while (i < m) {
			checkCudaError(
					cudaMemcpy(res.tiler.currBuffer() + i * (n - 1), tiler.currBuffer() + i * n, (n - 1) * sizeof(T), cudaMemcpyDeviceToDevice));
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
	for (uint i = 0; i < dim; i++) {
		ret.elements[i * dim + i] = elements[i];
	}
	ret.lastMod = mod_host;
	return ret;
}


template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::columnVector(int col) const {
	return columnMatrix(col);
}

template<typename T> CuMatrix<T> CuMatrix<T>::rowVector(int row) const {
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
		int col) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < x.m) {
		column.elements[row] = x.elements[row * x.p + col];
	}
}

template<typename T> __global__ void rowMatrixCMKernel(DMatrix<T> d_row, const DMatrix<T> x,
		int row) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < x.n) {
		d_row.elements[col] = x.elements[col * x.p + row];
	}
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::columnMatrixL(DMatrix<T>& d_column, const DMatrix<T>& d_x,
		 int col) {
	dim3 block(MIN(512, b_util::nextPowerOf2(d_x.m)));
	dim3 grid(DIV_UP(d_x.m,block.x));
	columnMatrixKernel<<<grid,block,0>>>(d_column, d_x, col );
}

template<typename T> void CuMatrix<T>::rowMatrixCmL(DMatrix<T>& d_row, const DMatrix<T>& d_x,
		 int row) {
	dim3 block(MIN(512, b_util::nextPowerOf2(d_x.m)));
	dim3 grid(DIV_UP(d_x.m,block.x));
	rowMatrixCMKernel<<<grid,block,0>>>(d_row, d_x, row);
}

#include "CuMatrixInster.cu"

