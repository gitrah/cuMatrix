#include "tests.h"
#include "../CuMatrix.h"
#include "../util.h"
#include "../caps.h"

template int testCudaMemcpy<float>::operator()(int argc, const char **argv) const;
template int testCudaMemcpy<double>::operator()(int argc, const char **argv) const;
template <typename T>  int testCudaMemcpy<T>::operator()(int argc, const char **argv) const {
	size_t tsize = sizeof(T);
	const int size = 2 * ExecCaps::currCaps()->alignment;
	const int len = size/sizeof(T);
	outln("size " << size);
	outln("size/sizeof(T) " << len );
	T* vmem1 = null;
	T* vmem2 = null;
	T* hmem1 = null;
	T* hmem2 = null;
	T* vmem1off = null;
	checkCudaError(cudaMalloc((void**)&vmem1, size));
	outln("cudaMallocd " << size << " to vmem1 " << vmem1);
	outln("vmem1 + 1 " << (vmem1 + 1));
	outln("vmem1 + 2 " << (vmem1 + 2));
	outln("cudaMallocd " << size << " to vmem1 " << vmem1);
	checkCudaError(cudaMalloc((void**)&vmem2, size));
	outln("cudaMallocd " << size << " to vmem2 " << vmem2);
	checkCudaError(cudaHostAlloc((void**)&hmem1, size,0));
	outln("cudaHostAlloc " << size << " to hmem1 " << hmem1);
	checkCudaError(cudaHostAlloc((void**)&hmem2, size,0));
	outln("cudaHostAlloc " << size << " to hmem2 " << hmem2);
	for(int i = 0; i < len; i++) {
		hmem1[i] = i;
		hmem2[i] = 1./(i + 1);
	}
	outln("hmem1 " << util<T>::parry(hmem1,len));
	outln("hmem2 " << util<T>::parry(hmem2,len));

	checkCudaError(cudaMemcpy(vmem1, hmem1, len, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(vmem2, hmem2, len, cudaMemcpyHostToDevice));
	outln("copied h to d buffers");

	for(int i = 0; i < len - 5; i++) {
		vmem1off = vmem1 + i;
		outln(i << " copying h to d offset by " << i * tsize);
		checkCudaError(cudaMemcpy(vmem1off, hmem1,  2, cudaMemcpyHostToDevice));
	}

	outln("2 retrying legal copy after...");
	checkCudaError(cudaMemcpy(vmem1, hmem1, len, cudaMemcpyHostToDevice));

	vmem1off = vmem1 + ExecCaps::currCaps()->alignment/sizeof(T);
	outln("3 copying h to d offset by " << ExecCaps::currCaps()->alignment/sizeof(T));
	printCudaError(cudaMemcpy(vmem1off, hmem1, 2, cudaMemcpyHostToDevice));

	outln("4 retrying legal copy after...");
	checkCudaError(cudaMemcpy(vmem1, hmem1, len, cudaMemcpyHostToDevice));

	vmem1off = vmem1 + 2;
	outln("5 copying vmem2 " << vmem2 << " to vmem1off " << vmem1off << " offset by " << 2 * tsize);
	printCudaError(cudaMemcpy(vmem1off, vmem2, 2, cudaMemcpyDeviceToDevice));

	outln("6 retrying legal copy after...");
	checkCudaError(cudaMemcpy(vmem1, hmem1, len, cudaMemcpyHostToDevice));

	outln("7 copying d to offset h by " << tsize);
	printCudaError(cudaMemcpy(hmem1 + 1, vmem2, 2  , cudaMemcpyDeviceToHost));

	outln("8 retrying legal d->h copy after...");
	checkCudaError(cudaMemcpy(hmem1, vmem1, len, cudaMemcpyDeviceToHost));

	outln("9 copying offset d by " << tsize << " to offset h by " << tsize);
	printCudaError(cudaMemcpy(hmem1 + tsize, vmem2 + 1, 2 , cudaMemcpyDeviceToHost));

	checkCudaError(cudaFree(vmem1));
	checkCudaError(cudaFree(vmem2));
	checkCudaError(cudaFreeHost(hmem1));
	checkCudaError(cudaFreeHost(hmem2));

	return 0;
}



template int testMemUsage<float>::operator()(int argc, const char **argv) const;
template int testMemUsage<double>::operator()(int argc, const char **argv) const;
template <typename T> int testMemUsage<T>::operator()(int argc, const char **argv) const {
	outln("testMemUsage start");
    int total = b_util::getCount(argc,argv,1000);
/*
	T s = 0;
	for(int i = 0; i < total; i++ ){
		CuMatrix<T> m = CuMatrix<T>::sin(500,500);
		CuMatrix<T> mc =CuMatrix<T>::cos(500,500);
		s += ((m * mc) * mc / 2.).sum();
		outln(i << "th iter; sum " << s << " usage: " << usedMemRatio() << "%");
	}

*/
	CuMatrix<T> col = CuMatrix<T>::ones(500,1);
	CuMatrix<T> blarb;
	for(int i = 0; i < total; i++) {
		blarb = blarb |= (col * (static_cast<T>( i)));
		outln(i << "th iter; usage: " << b_util::usedMemRatio() << "%");
		//outln(blarb);
	}
	outln("blarb is " << blarb.toShortString() << ", sum " << blarb.sum());
	outln(blarb.syncBuffers());
	return 0;
}


template int testCudaMemcpyVsCopyKernelVsmMemcpy<float>::operator()(int argc, const char **argv) const;
template int testCudaMemcpyVsCopyKernelVsmMemcpy<double>::operator()(int argc, const char **argv) const;
template <typename T> int testCudaMemcpyVsCopyKernelVsmMemcpy<T>::operator()(int argc, const char **argv) const {

	CuMatrix<T>::setMaxColsDisplayed(5);
	CuMatrix<T>::setMaxRowsDisplayed(5);
	CuMatrix<T> src = CuMatrix<T>::sin(5000,1000,1./10, 2*Pi,1);
	const T period = Pi/4;
	outln("period " << period);
	CuMatrix<T> src2 = CuMatrix<T>::sin(5000,1000,5.,period, 1);
	//src.syncBuffers(); // to create d mem
	//CuMatrix<T> src2 = 2 * src + 4;
	float exeTime;
	const float sizeG= 1. * src.size / Giga;
	const uint lengthInTs = src.size/sizeof(T);
	float memFlowIter, delta;
	CuMatrix<T> trg = CuMatrix<T>::zeros(5000,1000);
	trg.syncBuffers();
	outln("created sizeG " << sizeG << " src.size " << src.size) ;
	outln("created src " << src.syncBuffers());
	outln("created src2 " << src2.syncBuffers());

    CuTimer timer;

	DMatrix<T> src_d, src2_d, trg_d;
	src.syncBuffers(); // to create d mem
	src2.syncBuffers(); // to create d mem
	T srcSum = src.sum();
	T src2Sum = src2.sum();
	src.tile0(src_d, true);
	src2.tile0(src2_d,  true);
	trg.tile0(trg_d, false);
	outln("dmatrix src " << src.toShortString());
	outln("dmatrix trg.ss " << trg.toShortString());

	const int count = b_util::getCount(argc,argv,1000);
	const uint xfer = count * sizeG;

	// cudmemcpy first
	outln("count " << count);
	timer.start();
	clock_t lastTime = clock();
	for(int i = 0; i < count; i++) {
		checkCudaError(cudaMemcpy(trg.tiler.currBuffer(), src.tiler.currBuffer(), src.size, cudaMemcpyDeviceToDevice));
		checkCudaError(cudaDeviceSynchronize());
	}
	exeTime = timer.stop();
	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("cudaMemcpy d2d " << count << " took " << delta << " secs");
    outln("exeTime " << exeTime);
    memFlowIter = xfer * 1000/exeTime;
    trg.syncHost();
    outln("trg sum " << trg.sum());
    dassert(util<T>::almostEquals(trg.sum(), srcSum));
    outln("cudaMemcpy trg.tiler.currBuffer(), src.tiler.currBuffer() N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
	outln("after s-t trg.tiler.currBuffer(), src.tiler.currBuffer() trg " << trg);

	// cudmemcpy h2d
    timer.start();
	lastTime = clock();
	for(int i = 0; i < count; i++) {
		checkCudaError(cudaMemcpy(trg.tiler.currBuffer(), src2.elements, src.size, cudaMemcpyHostToDevice));
		checkCudaError(cudaDeviceSynchronize());
	}
	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("cudaMemcpy h2d " << count << " took " << delta << " secs");
	exeTime = timer.stop();
	outln("exeTime " << exeTime);
    memFlowIter = xfer * 1000/exeTime;
    outln("cudaMemcpy trg.tiler.currBuffer(), src2.elements N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
    outln("trg sum " << trg.sum());
    dassert(util<T>::almostEquals(trg.sum(), src2Sum));
    trg.invalidateHost();
    trg.syncBuffers();
    outln("after trg.tiler.currBuffer(), src2.elements trg " << trg.toShortString() << "\n" << trg.toString());

	// cudmemcpy d2h
    timer.start();
	lastTime = clock();
	outln("trg " << trg);
	for(int i = 0; i < count; i++) {
		checkCudaError(cudaMemcpy(trg.elements, src.tiler.currBuffer(), src.size, cudaMemcpyDeviceToHost));
		checkCudaError(cudaDeviceSynchronize());
	}
	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("cudaMemcpy d2h " << count << " took " << delta << " secs");
	exeTime = timer.stop();
    outln("exeTime " << exeTime);
    memFlowIter = xfer * 1000 / exeTime;
    trg.invalidateDevice();
    trg.syncBuffers();
    outln("trg sum " << trg.sum());
    dassert(util<T>::almostEquals(trg.sum(), srcSum));
    outln("cudaMemcpy trg.elements, src.tiler.currBuffer() N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
	outln("after s-t trg.elements, src.tiler.currBuffer() trg " << trg);

	// copy kernel d2d
    lastTime = clock();

    timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::copy1D(trg.tiler.currBuffer(), src2.tiler.currBuffer(), lengthInTs, 0, lengthInTs);
		checkCudaError(cudaDeviceSynchronize());
	}
    exeTime = timer.stop();

	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("copyKernel trg.tiler.currBuffer(), src2.tiler.currBuffer() " << count << " took " << delta << " secs");

    outln("exeTime " << exeTime);
    memFlowIter = xfer * 1000/exeTime;
	outln("copyKernel trg.tiler.currBuffer(), src2.tiler.currBuffer() N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
    trg.invalidateHost();
	trg.syncBuffers();
    outln("trg sum " << trg.sum());
    dassert(util<T>::almostEquals(trg.sum(), src2Sum));
	outln("after s-t trg.tiler.currBuffer(), src2.tiler.currBuffer() trg " << trg);

	// copy kerne h2h
    lastTime = clock();
    timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::copy1D(trg.elements, src.elements, lengthInTs, 0, lengthInTs);
		checkCudaError(cudaDeviceSynchronize());
	}
    exeTime = timer.stop();

	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("copyKernel trg.elements, src.elements " << count << " took " << delta << " secs");

    memFlowIter = xfer * 1000/exeTime;
    trg.invalidateDevice();
    trg.syncBuffers();
    outln("trg sum " << trg.sum());
    dassert(util<T>::almostEquals(trg.sum(), srcSum));
	outln("copyKernel trg.elements, src.elements N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");

	// copy kernel h2d myem
    lastTime = clock();
    timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::copy1D(trg.tiler.currBuffer(), src2.elements, lengthInTs, 0, lengthInTs);
		checkCudaError(cudaDeviceSynchronize());
	}
    exeTime = timer.stop();

	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("copyKernel trg.tiler.currBuffer(), src2.elements " << count << " took " << delta << " secs");

    memFlowIter = xfer * 1000/exeTime;
	outln("copyKernel trg.tiler.currBuffer(), src2.elements N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
	trg.syncBuffers();
    outln("trg sum " << trg.sum());
    dassert(util<T>::almostEquals(trg.sum(), src2Sum));
	outln("after s-t trg.tiler.currBuffer(), src2.elements trg " << trg);

	// copy kernel d2h myem
    lastTime = clock();
    timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::copy1D(trg.elements, src.tiler.currBuffer(), lengthInTs, 0, lengthInTs);
		checkCudaError(cudaDeviceSynchronize());
	}
    exeTime = timer.stop();

	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("copyKernel trg.elements, src.tiler.currBuffer() " << count << " took " << delta << " secs");

    memFlowIter = xfer * 1000/exeTime;
	outln("copyKernel trg.elements, src.tiler.currBuffer() N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
	outln("after s-t trg.elements, src.tiler.currBuffer() trg " << trg);
	trg.syncDevice();
    outln("trg sum " << trg.sum());
    dassert(util<T>::almostEquals(trg.sum(), srcSum));

	// plain memcpy
	lastTime = clock();
    timer.start();
    trg.syncBuffers();
    outln("trg.elements " << trg.elements << ", src.elements " << src.elements << ", src.size " << src.size);
	for(int i = 0; i < count; i++) {
		memcpy(trg.elements, src2.elements, src.size);
	}
    exeTime = timer.stop();

	delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("plain old memcpy trg.elements, src2.elements " << count << " took " << delta << " secs");

    memFlowIter = xfer * 1000/exeTime;
	outln("plain old memcpy trg.elements, src2.elements N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
    trg.invalidateDevice();
    trg.syncBuffers();
    outln("trg sum " << trg.sum());
    dassert(util<T>::almostEquals(trg.sum(), src2Sum));
	outln("after s-t trg.elements, src2.elements trg " << trg);

	return 0;
}


template int testCopyKernels<float>::operator()(int argc, const char **argv) const;
template int testCopyKernels<double>::operator()(int argc, const char **argv) const;
template <typename T>  int testCopyKernels<T>::operator()(int argc, const char **argv) const {
	const int count = b_util::getCount(argc,argv,1000);
	float exeTime;
	CuMatrix<T> src = CuMatrix<T>::sequence(0,1000,900);
	CuMatrix<T> trg = CuMatrix<T>::zeros(1000,900);
	CuMatrix<T> ssub, tsub;
	src.submatrix(ssub,900,800,50,50);
	trg.submatrix(tsub,950,850,25,25);
	outln("ssub " << ssub.sum());
	const float sizeG= 1. * src.size / Giga;
	const uint xfer = count * sizeG;
	//const uint lengthInTs = src.size/sizeof(T);
	float memFlowIter = 0;
	CuTimer timer;

	DMatrix<T> ds, dt;
	ssub.tile0(ds,true);
	tsub.tile0(dt,false);

	CuMatrix<T>::copyUlong(dt,ds, 25,25);

	timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::copyUlong(dt,ds, 25,25);
	}
    exeTime = timer.stop();


    memFlowIter = xfer * 1000/exeTime;
	outln("CuMatrix<T>::copyUlong N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
	outln("tsub " << tsub.sum());

	timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::copyUint(dt,ds, 25,25);
	}
    exeTime = timer.stop();


    memFlowIter = xfer * 1000/exeTime;
	outln("CuMatrix<T>::copyUint N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
	outln("tsub " << tsub.sum());

	timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::copyUlongDvrg(dt,ds, 25,25);
	}
    exeTime = timer.stop();


    memFlowIter = xfer * 1000/exeTime;
	outln("CuMatrix<T>::copyUlongDvrg N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
	outln("tsub " << tsub.sum());

	timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::copyUintDvrg(dt,ds, 25,25);
	}
    exeTime = timer.stop();


    memFlowIter = xfer * 1000/exeTime;
	outln("CuMatrix<T>::copyUintDvrg N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
	outln("tsub " << tsub.sum());

	timer.start();
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::copyIntDvrg(dt,ds, 25,25);
	}
    exeTime = timer.stop();


    memFlowIter = xfer * 1000/exeTime;
	outln("CuMatrix<T>::copyIntDvrg N " << count << " took exeTime " << (exeTime /1000) << "s or flow (r + w) of " << memFlowIter << "GB/s");
	outln("tsub " << tsub.sum());



	return 0;
}

template int testCudaMemcpyArray<float>::operator()(int argc, const char **argv) const;
template int testCudaMemcpyArray<double>::operator()(int argc, const char **argv) const;
template <typename T>  int testCudaMemcpyArray<T>::operator()(int argc, const char **argv) const {

	return 0;
}

template int testCudaMemcpy2D<float>::operator()(int argc, const char **argv) const;
template int testCudaMemcpy2D<double>::operator()(int argc, const char **argv) const;
template int testCudaMemcpy2D<ulong>::operator()(int argc, const char **argv) const;
template <typename T>  int testCudaMemcpy2D<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> ones = CuMatrix<T>::ones(16,16);
	outln("ones " << ones.syncBuffers());
	//const T* array, const char* msg, int line, int n,int direction
	printDevArray<T>(ones.tiler.buff(), "ones.tlr.buff", __LINE__, 16*16);
	CuMatrix<T> zed = CuMatrix<T>::zeros(4,4);
	outln("zed " << zed.syncBuffers());
	CuMatrix<T> tinyOnes = CuMatrix<T>::ones(4,4);
	//cherr( cudaMemcpy2D(m_elems +offset(roff, coff), m_p* sizeof(T), dm.elements, dm.p* sizeof(T), dm.n* sizeof(T), dm.m, cudaMemcpyDeviceToHost));
	DMatrix<T> dmZed;
	zed.tile0(dmZed,true);
	flprintf("after tile0, dmZed.m %u dmZed n %u, dmZed.elems %p\n", dmZed.m, dmZed.n, dmZed.elements);
	printDevArray<T>(dmZed.elements, "dmZed.elems", __LINE__, 4*4);

	cherr( cudaMemcpy2D(ones.elements, ones.p * sizeof(T), dmZed.elements, dmZed.p* sizeof(T), dmZed.n* sizeof(T), dmZed.m, cudaMemcpyDeviceToHost));
	cherr( cudaMemcpy2D(ones.elements + ones.tiler.offset(4,4), ones.p * sizeof(T), dmZed.elements, dmZed.p* sizeof(T), dmZed.n* sizeof(T), dmZed.m, cudaMemcpyDeviceToHost));
	cherr( cudaMemcpy2D(ones.elements+ ones.tiler.offset(8,8), ones.p * sizeof(T), dmZed.elements, dmZed.p* sizeof(T), dmZed.n* sizeof(T), dmZed.m, cudaMemcpyDeviceToHost));
	cherr( cudaMemcpy2D(ones.elements+ ones.tiler.offset(12,12), ones.p * sizeof(T), dmZed.elements, dmZed.p* sizeof(T), dmZed.n* sizeof(T), dmZed.m, cudaMemcpyDeviceToHost));

	outln("ones post " << ones); // should set 16x16 ones with 4 4x4 'holes' in it

	assert(ones.sum() == 16 * 16 - 4* 4* 4);

	return 0;
}


template int testRowCopy<float>::operator()(int argc, const char **argv) const;
template int testRowCopy<double>::operator()(int argc, const char **argv) const;
template <typename T> int testRowCopy<T>::operator()(int argc, const char **argv) const {
	int rows = 5, cols = 10;
	CuMatrix<T> mOnes_rm = CuMatrix<T>::ones(rows, cols);
	CuMatrix<T> mZeros_cm = CuMatrix<T>::zeros(rows,cols,true);
	IndexArray mOnes_rm_col4 = mOnes_rm.columnIndices(4);
	IndexArray mZeros_c_col4 = mZeros_cm.columnIndices(4);
	IndexArray mOnes_rm_row2 = mOnes_rm.rowIndices(2);
	IndexArray mZeros_c_row2 = mZeros_cm.rowIndices(2);
	//dassert( (sum(mOnes_rm_row2.indices, mOnes_rm_row2.count) == ));
	outln(mOnes_rm.toString());
	outln("mOnes_rm_row2 " << mOnes_rm_row2.toString().c_str());
	outln("mZeros_c_row2 " << mZeros_c_row2.toString().c_str());
	outln("mOnes_rm_col4 " << mOnes_rm_col4.toString().c_str());
	outln("mZeros_c_col4 " << mZeros_c_col4.toString().c_str());

	return 0;
}


template int testCopyVsCopyK<float>::operator()(int argc, const char **argv) const;
template int testCopyVsCopyK<double>::operator()(int argc, const char **argv) const;
template int testCopyVsCopyK<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testCopyVsCopyK<T>::operator()(int argc, const char **argv) const {

	int rows = 10, cols = 10;
	CuMatrix<T> mOnes = CuMatrix<T>::ones(rows, cols);
	CuMatrix<T> mZeros = CuMatrix<T>::zeros(3,3);

	outln("mOnes " << mOnes.syncBuffers());
	outln("mZeros " << mZeros.syncBuffers());

	mZeros.copy(mOnes,5,5,true);

	CuMatrix<T> tiled = mOnes.replicateTiled(30,5);

	outln("mOnes now " << mOnes.syncBuffers());
	outln("tiled " << tiled.syncBuffers());
	return 0;
}


template int testClippedRowSubset<float>::operator()(int argc, const char **argv) const;
template int testClippedRowSubset<double>::operator()(int argc, const char **argv) const;
template int testClippedRowSubset<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testClippedRowSubset<T>::operator()(int argc, const char **argv) const {
	int rows = 5, cols = 10;
	CuMatrix<T> seq = CuMatrix<T>::sequence(-1,10,5);
	CuMatrix<T> seq2 = CuMatrix<T>::sequence(-1,5,10);
	outln("seq " << seq.syncBuffers());
	outln("seq2 " << seq2.syncBuffers());
	//seq.printShortString("seq");
	CuMatrix<T> seqCm = CuMatrix<T>::sequence(-1,10,5,true);
	CuMatrix<T> mOnes_rm = CuMatrix<T>::ones(rows, cols);
	outln("mOnes_rm " << mOnes_rm.syncBuffers());

	CuMatrix<T> mZeros_cm = CuMatrix<T>::zeros(rows,cols,true);
	IndexArray mOnes_rm_col4 = mOnes_rm.columnIndices(4);
	IndexArray mZeros_c_col4 = mZeros_cm.columnIndices(4);
	IndexArray mOnes_rm_row2 = mOnes_rm.rowIndices(2);
	IndexArray mZeros_c_row2 = mZeros_cm.rowIndices(2);
	//dassert( (sum(mOnes_rm_row2.indices, mOnes_rm_row2.count) == ));
	outln("mOnes_rm_row2 " << mOnes_rm_row2);
	outln("mZeros_c_row2 " << mZeros_c_row2);
	outln("mOnes_rm_col4 " << mOnes_rm_col4);
	outln("mZeros_c_col4 " << mZeros_c_col4);

	return 0;
}


#include "tests.cc"
