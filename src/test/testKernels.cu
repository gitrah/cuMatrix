/*
 * testKernels.cu
 *
 *  Created on: Dec 18, 2013
 *      Author: reid
 */

#include "testKernels.h"
#include "tests.h"
#include "../FuncPtr.h"
#include "../CuDefs.h"
#include "../CuMatrix.h"
#include "../Kernels.h"
#include "../Maths.h"
template<typename T> __global__ void testShuffleKernel(T* d_res, T* ary, int len) {
	ostreamlike s;

	//cudaStream_t stream;
	//cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

}

template<typename T> void launchTestShuffleKernel() {
	const int len = 32;
	//size_t size = len * sizeof(T);
	T ary[len];
	T total = 0;
	for(int i = 0; i < len; i++) {
		ary[i] = 2 * i;
		total += 2 * i;
	}
	T* dary;
	T* dres;
	T res;
	plusBinaryOp<T> plus = Functory<T,plusBinaryOp>::pinch();

	checkCudaError(cudaMalloc(&dary, len * sizeof(T)));
	checkCudaError(cudaMalloc(&dres, sizeof(T)));
	checkCudaError(cudaMemcpy(dary,ary,len * sizeof(T),cudaMemcpyHostToDevice));
	outln("launchTestShuffleKernel total " << total );
	shuffle<<<1,len>>>(dres,(const T*)dary ,len,plus);
	checkCudaError(cudaDeviceSynchronize());
	checkCudaError(cudaMemcpy(&res,dres,sizeof(T),cudaMemcpyDeviceToHost));
	assert(res == total);
	checkCudaError(cudaFree(dary));
	checkCudaError(cudaFree(dres));
}
template void launchTestShuffleKernel<float>();
template void launchTestShuffleKernel<double>();
template void launchTestShuffleKernel<ulong>();

__global__ void testSetLastErrorKernel(CuMatrixException ex) {
	flprintf("ex %d\n",ex);
	setLastError(ex);
}

template<typename T> __global__ void testCdpKernel(T* res, CuMatrix<T> mat) {
	printf("in testCdpKernel\n");

#ifdef CuMatrix_Enable_Cdp
	*res = mat.sum();
#else
	assert(0);
#endif
}

template<typename T> void launchTestCdpKernel(T* res, CuMatrix<T> mat) {
	mat.printShortString("in launchTestCdpKernel");

	cudaFuncAttributes fatts;
	cudaFuncGetAttributes(&fatts, testCdpKernel<T>);
	cudaError_t err;
#ifdef CuMatrix_DebugBuild
	// default size of 1024 results in 'Lane User Stack Overflow'
	size_t currVal = 2048;
	err = cudaDeviceSetLimit(cudaLimitStackSize,currVal);
#endif
	flprintf( "atts for testCdpKernel (regs=%d, shared %lu local %lu maxtpb %d )", fatts.numRegs,fatts.sharedSizeBytes, fatts.localSizeBytes, fatts.maxThreadsPerBlock);
	//err = cudaDeviceGetLimit(&currVal,cudaLimitStackSize);
	testCdpKernel<T><<<1,1>>>(res,mat);
	cherr(cudaDeviceSynchronize());
	if(err != cudaSuccess) {
		outln("err " << err);
	}

	outln("invoked testCdpKernel");
	outln("sanched");
/*
	dim3 grid(1);
	dim3 block(1);
	testCdpKernel<T><<<grid,block>>>(res,mat);
*/
}

template void launchTestCdpKernel<float>(float*,CuMatrix<float>);
template void launchTestCdpKernel<double>(double*,CuMatrix<double>);
template void launchTestCdpKernel<ulong>(ulong*,CuMatrix<ulong>);


template<typename T> void launchRedux(T* res, CuMatrix<T> mat) {
	*res = mat.sum();
}

template<typename T> __global__ void launchReduxD(T* res, CuMatrix<T> mat) {
	*res = mat.sum();
}

template<typename T, typename BinaryOp> __global__ void kFunctor(BinaryOp op){
	printf("in kFunctor ");
}

template<typename T, typename BinaryOp> void launchKFunctor(BinaryOp op){
	printf("in launchKFunctor ");
	kFunctor<T><<<1,1>>>(op);
}
template void launchKFunctor<float, almostEqualsBinaryOp<float> >(almostEqualsBinaryOp<float>);
template void launchKFunctor<double, almostEqualsBinaryOp<double> >(almostEqualsBinaryOp<double>);

__global__ void kFoo(kfoo foo) {
	printf("kFoo %p and %d\n", foo.pointierre, foo.somint);
}
void launchKfoo(kfoo foo) {
	kFoo<<<1,1>>>(foo);
}
template<typename T> __global__ void kBar(kbar<T> bar) {
	printf("kBar %p and %f and %f\n", bar.pointierre, bar.somt, bar(50));
}
template<typename T>void launchKbar(kbar<T> bar) {
	kBar<T><<<1,1>>>(bar);
}
template void launchKbar<float>(kbar<float>);
template void launchKbar<double>(kbar<double>);

template<typename T> __global__ void testSmemcpy(T* out, const T* in, int count) {
	T* sdata = SharedMemory<T>();
 	memcpy(sdata,in,count * sizeof(T));
	memcpy(out,sdata,count * sizeof(T));
}

template int testMemcpyShared<float>::operator()(int argc, const char **argv) const;
template int testMemcpyShared<double>::operator()(int argc, const char **argv) const;
template <typename T> int testMemcpyShared<T>::operator()(int argc, const char **argv) const{
	CuMatrix<T> ones = CuMatrix<T>::ones(1024,1);
	CuMatrix<T> res = CuMatrix<T>::zeros(1024,1);
	//auto z = 5;
	testSmemcpy<<<1, 1, ones.size>>>( res.tiler.currBuffer(), ones.tiler.currBuffer(), ones.size );
	res.invalidateHost();
	checkCudaError(cudaDeviceSynchronize());
	outln("res " << res.syncBuffers());

	return 0;
}

__global__ void testSignKernel() {
	if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z ==0) {
		flprintf("sign(.5f) %f\n", sign(0.5f));
		flprintf("sign(-.5f) %f\n", sign(-0.5f));
		flprintf("sign(-.5 ) %f\n", sign(-0.5));
		flprintf("sign(.5 ) %f\n", sign(0.5));
		flprintf("sign(5) %d\n", sign(5));
		flprintf("sign(-5) %d\n", sign(-5));
		flprintf("sign(5l) %d\n", sign(5l));
		flprintf("sign(-1l) %d\n", sign(-1l));
		flprintf("sign(0f) %f\n", sign(0.0f));
		flprintf("sign(55f) %f\n", sign(55.0f));
	}
}

template int testSign<float>::operator()(int argc, const char **argv) const;
template int testSign<double>::operator()(int argc, const char **argv) const;
template int testSign<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testSign<T>::operator()(int argc, const char **argv) const{
	testSignKernel<<<1,1>>>();
	return 0;
}


template <typename T> __global__ void gpoly2_1(void* fptr, T x) {
	prlocf("gpoly2_1 enter\n");
	void * p = poly2_1<T>;
	void * ptr = &poly2_1<T>;
	void * ptrf = poly2_1<float>;
	void * ptrd = &poly2_1<double>;
	typename func1<T>::inst fn = poly2_1<T>;
	flprintf("gpoly2_1 p %p ptr %p ptrf %p ptrd %p\n", p, ptr, ptrf, ptrd);
	flprintf("gpoly2_1 fn %p\n", fn);
	T res = fn(5);
	if(fptr != null) {
		*((ulong*)fptr) = (ulong)fn;
	}
	flprintf("fn(5) %f\n", (T)res);
	//.void* ptr =
}

//template <typename T> __global__ void getFunction(typename func1<T>::inst fn&, )

template int testBisection<float>::operator()(int argc, const char **argv) const;
template int testBisection<double>::operator()(int argc, const char **argv) const;
template int testBisection<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testBisection<T>::operator()(int argc, const char **argv) const{
	outln("testBisection start " );
	//bisection<T,deg2>(fn, -10, 10, util<T>::epsilon(), 1024);
	typename func1<T>::inst fn = null;
	typedef	 T (*inst)(T x);
	void* fptr;
	checkCudaError(cudaMalloc(&fptr, sizeof(typename func1<T>::inst)));
	ulong res;
	gpoly2_1<<<1,1>>>(fptr, (T)1.0);
	checkCudaError(cudaMemcpy(&fn, fptr, sizeof(ulong), cudaMemcpyDeviceToHost));
	flprintf("fn %p\n", fn);
	//checkCudaError(cudaMemcpyFromSymbol(&fn, poly2_1<T>, sizeof(typename func1<T>::inst)));

	outln("sizeof(typename func1<T>::inst) " << sizeof(typename func1<T>::inst));

	setDFarray<T><<<1,1>>>();
	checkCudaErrors(cudaDeviceSynchronize());

	/*void * p = (void*) poly2_1<T>;
	flprintf("p %p\n", p);
	*/
	inst pfunc =  poly2_1<T>;
	flprintf("fn %p\n", fn);
	flprintf("pfunc %p\n", pfunc);
	checkCudaErrors(cudaDeviceSynchronize());

	//checkCudaErrors(cudaMalloc(&fn, sizeof(typename func1<T>::inst*)));
	//checkCudaErrors(cudaMemcpy( fn, poly2_1<T>, sizeof(typename func1<T>::inst*), cudaMemcpyHostToDevice));

	outln("testBisection fn "  << fn);
	flprintf("testBisection fn %p\n", fn);
	outln("testBisection pfunc "  << pfunc);
	flprintf("testBisection pfunc %p\n", pfunc);
	int count = b_util::getCount(argc,argv,5);
	uint maxRoots = 5;
	uint* rootCount;
	T* roots;
	checkCudaErrors(cudaMallocManaged(&roots, maxRoots*sizeof(T)));
	checkCudaErrors(cudaMallocManaged(&rootCount, sizeof(uint)));
	CuTimer timer;
	timer.start();
	for(int i = 0; i < count; i++) {
		*rootCount = 0;
		bisection<T>(roots, rootCount, maxRoots, fn, (T)-10000, (T)10000, util<T>::epsilon(), 1024);
	}
	outln( count << " of 1st bisection took " << timer.stop() << " μs");
	for(uint i = 0; i < *rootCount; i++) {
		flprintf("root %u = %f\n", i, (float) roots[i]);
	}
	outln("2nd bisection idx "  << ePoly1_2);
	timer.start();
	for(int i = 0; i < count; i++) {
		*rootCount = 0;
		bisection<T>( roots, rootCount, maxRoots,  ePoly1_2, (T)-10000, (T)10000, util<T>::epsilon(), 1024);
	}
	outln( count << " of 2nd bisection took " << timer.stop() << " μs");
	for(uint i = 0; i < *rootCount; i++) {
		flprintf("root %u = %f\n", i, (float) roots[i]);
	}

	checkCudaErrors(cudaGetLastError());
	//ftor1Ops<T> h1ftors;

	typename func1<T>::inst f1 = negateFn<T>;
	almostEqUnaryOp<T> almEq;

/*
	ftor1Ops<T> h1ftors;
	ftor1Ops<T> d1fors;
	checkCudaError(cudaMalloc(&d1fors.ops, sizeof(typename ftor1<T>::opPtr)));
*/

	//buildUnaryOpFtorArray<<<1,1>>>( typename ::ops  array);
	return 0;
}


__global__ void dCreateDmem(ulong* trg) {
	uint* dmem;
#ifdef CuMatrix_Enable_Cdp
	cudaMalloc(&dmem, sizeof(uint));
#else
	prlocf("not implemented for non-cdp\n");
	assert(false);
#endif
	flprintf("dmem after cudaMalloc %p\n", dmem);
	*trg = (ulong)dmem;
}
__global__ void dFreeDmem(uint* trg) {
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaFree(trg));
#else
	prlocf("not implemented for non-cdp\n");
	assert(false);
#endif
}

template int testHfreeDalloc<float>::operator()(int argc, const char **argv) const;
template int testHfreeDalloc<double>::operator()(int argc, const char **argv) const;
template int testHfreeDalloc<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testHfreeDalloc<T>::operator()(int argc, const char **argv) const{
	ulong* trg;
	uint *hdmem;
	checkCudaError(cudaMalloc(&trg, sizeof(ulong)));
	dCreateDmem<<<1,1>>>(trg);
	checkCudaError(cudaMemcpy(&hdmem, trg, sizeof(ulong), cudaMemcpyDeviceToHost));

	checkCudaError(cudaFree(trg));
	outln("hdem  " << hdmem);
	// should cause error

	if(b_util::getParameter(argc,argv,"hfreed",0)) {
		checkCudaError(cudaFree(hdmem));
	} else{
		dFreeDmem<<<1,1>>>(hdmem);
	}
	return 0;
}

__global__ void inc(float* data, int count) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < count) {
		data[tid] = data[tid] + 1;
	}
}

__global__ void uinc(float* data, uint count) {
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < count) {
		data[tid] = data[tid] + 1;
	}
}

__global__ void inc2(float2* data, int count) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < count) {
/*
		if(tid == count -1 ) {
			flprintf("incing last elem[%d] at %p\n", tid, data + tid);
			flprintf("last float @ %p\n", &(data[tid].y));
		}
*/
		float2 f = data[tid];
		f.x += 1;
		f.y += 1;
		data[tid] = f;
	}
}

__global__ void inc3(float3* data, int count) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < count) {
		float3 f = data[tid];
		f.x += 1;
		f.y += 1;
		f.z += 1;
		data[tid] = f;
	}
}

__global__ void inc4(float4* data, int count) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < count) {
		float4 f = data[tid];
		f.x += 1;
		f.y += 1;
		f.z += 1;
		f.w += 1;
		data[tid] = f;
	}
}

template int testInc<float>::operator()(int argc, const char **argv) const;
template int testInc<double>::operator()(int argc, const char **argv) const;
template int testInc<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testInc<T>::operator()(int argc, const char **argv) const{
	CuTimer timer;
	int count = b_util::getCount(argc,argv,5);
	ulong size0 =  64 * Mega + 5;
	outln("size0 " << size0);
	int size[] = {size0, DIV_UP(size0, 2), DIV_UP(size0, 3), DIV_UP(size0, 4)};
	int block = 1024;
	int bf[] =  {DIV_UP(size[0], block),DIV_UP(size[1], block), DIV_UP(size[2], block), DIV_UP(size[3], block)};
	const char* szs[]= {"float","float2","float3", "float4"};

	CuMatrix<float> m = CuMatrix<float>::zeros(size0,1);
	notAlmostEqUnaryOp<float> naEq = Functory<float,notAlmostEqUnaryOp>::pinch((float)0, util<float>::epsilon());
	outln("m.count(naEq) " << m.count(naEq));
	outln("m.sum " << m.sum());
	outln("m " << m.syncBuffers());
	outln("m.kahanSum " << m.kahanSum());
	neqUnaryOp<float> ne  = Functory<float,neqUnaryOp>::pinch((float)0);
	int outlierCount = 20;
	uint idxs[outlierCount];
	memset(idxs,0,20 * sizeof(uint));
	IndexArray outliers(idxs,outlierCount,false);
	//outln("pre outliers " <<outliers);
	//int outlier = pvec.count(ne);
	m.findFirstN(outliers,ne);
	outln("1st " << outlierCount <<" outliers " << outliers);

	checkCudaErrors(cudaGetLastError());
	ExecCaps* pcaps = ExecCaps::currCaps();
	//warmup<<<1,1>>>();
	for(int sz = 0; sz < 4; sz++) {
		int slices = DIV_UP( bf[sz], pcaps->maxGrid.y);
		int sliceSize = DIV_UP( size[sz], slices);
		outln("slices " << slices << ", sliceSize " << sliceSize << ", size - (slices * sliceSize) " << (size[sz] - slices * sliceSize));
		int offset, thisBlock;
		int lastIdx = -1;
		outln(slices << " x bf["<< sz << "]/slices ( " << (bf[sz]/slices )<< "), block " << block);
		timer.start();
		for(int i = 0;i < count; i++ ) {
			thisBlock = DIV_UP(bf[sz],slices);
			for(int currSlice = 0; currSlice < slices; currSlice++) {
				offset = currSlice * sliceSize;
				if(lastIdx > -1) {
					//outln("offset - lastIdx " << (offset - lastIdx));
				}
				if(currSlice == slices -1) {
					//outln("last slice");
					sliceSize = size[sz] - (slices - 1 ) * sliceSize;
					thisBlock = DIV_UP(bf[sz],slices);
				}
				int currDev = ExecCaps::currDev();
				//outln( "thisBlock " << thisBlock << ", offset " << offset << ", sliceSize " << sliceSize);
				switch(sz) {
				case 0:
					inc<<< thisBlock, block>>>(m.tiler.buffer(currDev) + offset, sliceSize);
					break;
				case 1:
					inc2<<< thisBlock, block>>>( ((float2*)m.tiler.buffer(currDev) )+ offset, sliceSize);
					break;
				case 2:
					inc3<<< thisBlock, block>>>( ((float3*)m.tiler.buffer(currDev) )+ offset, sliceSize);
					break;
				case 3:
					inc4<<< thisBlock, block>>>( ((float4*)m.tiler.buffer(currDev) )+ offset, sliceSize);
					break;
				}
				lastIdx = offset;
			}
			checkCudaErrors(cudaDeviceSynchronize());
		}
		outln(count << " " << szs[sz] << " took " << timer.stop());
		m.invalidateHost();
		outln("m " << m.syncBuffers());
		outln("m.sum " << m.sum());
	}

	/*
	ne.target = count;
	outln("m.count(ne) with target != 1: " << m.count(ne));
	ne.target = 0;
	outln("m.count(ne) with target != 0: " << m.count(ne));
	ne.target = count;
	m.findFirstN(outliers,ne);
	outln("1st " << outlierCount <<" outliers after inc of " << count <<": " << outliers);

*/
	//checkCudaErrors(cudaFree(buff));

	return 0;
}

template<typename T> __global__ void shufflet() {
	T tid = (T) threadIdx.x;
	uint tidy = threadIdx.y;
	T t_1 = shfl<T>(tid, tid+1);
	T t_16 =shfl<T>(tid, tid+16);
	flprintf("tidy %u tid %f, shfl<T>(tid, tid+2) %f\n", tidy, tid, t_1);
	flprintf("tidy %u tid %f, shfl<T>(tid, tid+16) %f\n", tidy, tid, t_16);
}

template<typename T> __device__ T shuffplus(const T* ary, int len, int width) {
	if(len <= WARP_SIZE && len <= width) {
		int currLen =  b_util::nextPowerOf2(len)/2;
		T val = ary[threadIdx.x];
		while(currLen > 0 ) {
			int lane = threadIdx.x + currLen;
			if(lane < len) {
				val += shfl<T>(ary[threadIdx.x], lane, width);
			}
			currLen >>= 1;
		}
	}
}

template<typename T> __global__ void gShufPlus(T* out, T* arry) {
	uint tidx = threadIdx.x;
	uint tidy = threadIdx.y;
	T val = arry[tidy * 2 + tidx];
	__syncthreads();
	int lane = tidx + tidy * blockDim.x + 1;
	flprintf("%u,%u (l %d)-> %f\n", tidy,tidx, (lane -1), val);
	T val2 = val + shfl<T>(val,  lane);
	// equiv:  T val2 = val + shflDown<T>(val,  1);
	flprintf("%u,%u l %d-> %f\n", tidy,tidx, lane, val2);
	out[tidy + tidx * 5] = val2;
}

template int testShufflet<float>::operator()(int argc, const char **argv) const;
template int testShufflet<double>::operator()(int argc, const char **argv) const;
template int testShufflet<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testShufflet<T>::operator()(int argc, const char **argv) const{
	//shufflet<T><<<31,2,0>>>();
	T harry[] = {	1,2,
					3,4,
					5,6,
					7,8,
					9,10,
					11,12};
	T* arry, *out;
	cudaMalloc(&arry, 12 * sizeof(T));
	cudaMalloc(&out, 10 * sizeof(T));
	cudaMemcpy(arry, harry, 12 * sizeof(T), cudaMemcpyHostToDevice);
	dim3 threads(2,5);
	gShufPlus<T><<<1,threads>>>(out,arry);
	cudaDeviceSynchronize();
	util<T>::pDarry(out,10);
	cudaFree(out);
	cudaFree(arry);
	return 0;
}

template<typename T, int StateDim> __device__ void locConst(UnaryOpIndexF<T,StateDim> fill) {
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		flprintf("locConst fill.fn %p\n", fill.fn);
	#else
		flprintf("locConst fill.operation %p\n", fill.operation);
	#endif
#endif
	flprintf("locConst fill(13) %f\n", fill(13));
}

template<typename T, int StateDim> __global__ void constFillKrnle(UnaryOpIndexF<T,StateDim> fill) {
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		flprintf("constFillKrnle fill.fn %p\n", fill.fn);
	#else
		flprintf("constFillKrnle fill.operation %p\n", fill.operation);
	#endif
#endif
	constFiller<T> cf = Functory<T, constFiller>::pinch(16);
	locConst(cf);
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		flprintf("cf.fn == fill.fn %d\n", (cf.fn == fill.fn));
		fill.fn = cf.fn;
	#else
		flprintf("cf.operation == fill.operation %d\n", (cf.operation == fill.operation));
		fill.operation = cf.operation;
	#endif
#endif
	flprintf("constFillKrnle fill(12) %f\n", fill(12));
}

template<typename T>void constFillKrnleL( ){
	constFiller<T> cf = Functory<T, constFiller>::pinch(15);
#ifndef CuMatrix_Enable_KTS
	#ifdef CuMatrix_StatFunc
		outln("cf.fn "  << cf.fn << "\n");
	#else
		outln("cf.operation "  << cf.operation << "\n");
	#endif
#endif
	outln("cf[0] "  << cf[0] << "\n");
	outln("cf(0) "  << cf(0) << "\n");
	constFillKrnle<T,1><<<1,1>>>(cf);
	checkCudaErrors(cudaDeviceSynchronize());
}
template void constFillKrnleL<float>();


template int testMemset<float>::operator()(int argc, const char **argv) const;
template int testMemset<ulong>::operator()(int argc, const char **argv) const;
template int testMemset<double>::operator()(int argc, const char **argv) const;
template <typename T> int testMemset<T>::operator()(int argc, const char **argv) const {
	outln("testMemset start");

    int total = b_util::getCount(argc,argv,1000);

    CuTimer timer;

    T* buffer;

	cherr(cudaMalloc(&buffer,1000*sizeof(T)));

	T val = 42;

	timer.start();
	for(int i =0; i < total; i++) {
		util<T>::setNDev(buffer, val, 1000);
	}
	float ndevTime=timer.stop();

	outln("ndevTime " << ndevTime/1000);

	printArray(buffer, 1000);

	int threads = 512;
	dim3 dBlocks, dThreads;
	b_util::vectorExecContext(threads, 1000, dBlocks, dThreads);

	val = 43;
	timer.start();
	for(int i =0; i < total; i++) {
		fillKernel<<<dBlocks,dThreads>>>(buffer, val, 1000);
		cherr(cudaDeviceSynchronize());
	}
	float fillkTime=timer.stop();

	outln("fillkTime " << fillkTime/1000);

	printArray(buffer, 1000);

	cherr(cudaFree(buffer));
	return 0;
}

