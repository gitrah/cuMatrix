#include "tests.h"
#include "../CuMatrix.h"
#include "../Kernels.h"
#include "../caps.h"
#include "testKernels.h"

__device__ char * pBuffer = NULL;
__device__ char * pBuffer2 = NULL;
__device__ ExecCaps ** ppCaps = NULL;
__device__ ExecCaps * paCaps[2];

char* gbuff;

__device__ void testBid() {
	uint bid = blockIdx.x;
	printf("testBid bid %d\n", bid);
}

template<typename T> __global__ void kernelInner() {
	printf("in kernelInner\n");
	// testBid(); blows up in cuda-memchk

	printf("pBuffer %s\n",pBuffer);
	printf("pBuffer2 %s\n",pBuffer2);
	printf("kernelInner blockIdx.x %u\n",blockIdx.x);

	CuMatrix<T> m1 = CuMatrix<T>::ones(100,100);
	printf("m1 " );
	m1.printShortString();
	CuMatrix<T> m2 = CuMatrix<T>::ones(100,100) * 2;
	printf("m2 " );
	m2.printShortString();
	CuMatrix<T> m3 = m1 + m2;
	printf("m3 " );
	m3.printShortString();
	CuMatrix<T> m4 = 2*m1 + m2/7;
	printf("m4 " );
	m4.printShortString();

	m2.print("m2 full\n" );

	//m2.printShortString();
	m3.print("m3 full\n" );
	//m3.printShortString();
	m4.print("m4 full\n" );
	//m4.printShortString();
}

template<typename T> __global__ void kernelOuter() {
	printf("in kernelOuter, launching kernelInner\n");
	//testBid(); blows up in cuda-memchk
	const char* stuff = "some stuff\n\n";
	printf(stuff);
	printf("pBuffer %s\n",pBuffer);
	printf("pBuffer2 %s\n",pBuffer2);
	printf("kernelOuter blockIdx.x %u\n",blockIdx.x);
	printf("pBuffer3[0]->smCount %u\n",paCaps[0]->smCount);

	pBuffer[0]='z';
	pBuffer2[0]='g';

#ifdef CuMatrix_Enable_Cdp
	kernelInner<T><<<1,1>>>();
#else
	prlocf("not implemented for non-cdp\n");
	assert(false);
#endif
}


__global__ void freePpCaps() {
#ifdef CuMatrix_Enable_Cdp
	cherr(cudaFree(ppCaps));
#else
	prlocf("not implemented for non-cdp\n");
	assert(false);
#endif
}

template int testRecCuMatAddition<float>::operator()(int argc, char const ** args) const;
template int testRecCuMatAddition<double>::operator()(int argc, char const ** args) const;
template <typename T> int testRecCuMatAddition<T>::operator()(int argc, const char** args) const {

	const char* b1= "hell";
	const char* b2 = "ouch";

	char* buff;
	checkCudaError(cudaMalloc(&buff, 8));
	checkCudaError(cudaMemcpy(buff,b1,5,cudaMemcpyHostToDevice));
	checkCudaError(cudaMalloc(&gbuff, 8));
	checkCudaError(cudaMemcpy(gbuff,b2,5,cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpyToSymbol(pBuffer, &buff, sizeof(char*)));
	checkCudaError(cudaMemcpyToSymbol(pBuffer2, &gbuff, sizeof(char*)));

	ExecCaps* pCurrCaps = ExecCaps::currCaps();
	char* cbCaps;
	// create and fill ec buff with an ec
	checkCudaError(cudaMalloc(&cbCaps, sizeof(ExecCaps)));
	flprintf("testRecCuMatAddition allocated %d to %p\n", sizeof(ExecCaps),&cbCaps);
	checkCudaError(cudaMemcpy(cbCaps,pCurrCaps,sizeof(ExecCaps),cudaMemcpyHostToDevice));

	ExecCaps* apCurrCaps[2];
	apCurrCaps[0] = (ExecCaps*)cbCaps;
	//outln("apCurrCaps[0]->smCount " << apCurrCaps[0]->smCount);
	checkCudaError(cudaMemcpyToSymbol(paCaps, &apCurrCaps, sizeof(apCurrCaps)));
	outln("sizeof(paCaps) " <<sizeof(paCaps));
	checkCudaError(cudaMalloc(&ppCaps, sizeof(paCaps))); // LEAK!!!!
	flprintf("testRecCuMatAddition allocated %d to %p\n", sizeof(paCaps),&ppCaps);
	//checkCudaError(cudaMemcpyToSymbol(ppCaps,&apCurrCaps,sizeof(apCurrCaps)));

	checkCudaError(cudaGetLastError());

	outln("launching kernelOuter");
	kernelOuter<T><<<1,1>>>();
	checkCudaError(cudaGetLastError());

	checkCudaError(cudaFree(buff));
	checkCudaError(cudaFree(gbuff));
	checkCudaError(cudaFree(cbCaps));
	freePpCaps<<<1,1>>>();

	return 0;
}

template int testCdpSum<float>::operator()(int argc, char const ** args) const;
template int testCdpSum<double>::operator()(int argc, char const ** args) const;
template <typename T> int testCdpSum<T>::operator()(int argc, const char** args) const {
	CuMatrix<T> m = CuMatrix<T>::ones(5000,1000);
	T sum = m.sum();
	outln("m.sum " << sum);
	m.printShortString("in testCdpSum");

	T* devSum;
	T sum2;
	cudaMalloc(&devSum, sizeof(T));
	launchTestCdpKernel(devSum, m);
	cudaMemcpy(&sum2,devSum,sizeof(T),cudaMemcpyDeviceToHost);
	outln("sum2 " << sum2);

	assert(sum2 == sum);

	int count = b_util::getCount(argc,args,500);

	return 0;
}

float slepstr() {
	cudaStream_t stream[2];
	for(int i = 0; i < 2; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
	CuTimer watch;

	cudaEvent_t start_event, stop_event;

	checkCudaErrors(cudaEventCreateWithFlags(&start_event, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&stop_event, cudaEventDisableTiming));

    checkCudaErrors(cudaEventRecord(start_event, 0));
    watch.start();
    slep<<<1,1, 0,stream[0]>>>(1000000);
    slep<<<1,1,0, stream[1]>>>(1000000);
    float tm = watch.stop();
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));   // block until the event is actually recorded
	//outln("slep took " << watch.stop());
	for(int i = 0; i < 2; i++) {
		checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
    checkCudaErrors(cudaEventDestroy(start_event));
	checkCudaErrors(cudaEventDestroy(stop_event));
	return tm;
}

template<typename T> float sigmoidstr(DMatrix<T>& trg1, DMatrix<T>& trg2, const DMatrix<T>& src1, const DMatrix<T>& src2) {
	cudaStream_t stream[2];
	for(int i = 0; i < 2; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
	CuTimer watch;

/*
	cudaEvent_t start_event, stop_event;

	checkCudaErrors(cudaEventCreateWithFlags(&start_event, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&stop_event, cudaEventDisableTiming));

    checkCudaErrors(cudaEventRecord(start_event, 0));
*/
    watch.start();

    unaryOpL(trg1, src1, sigmoidUnaryOp<T>(), stream[0]);
    unaryOpL(trg2, src2, sigmoidUnaryOp<T>(), stream[1]);

/*
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));   // block until the event is actually recorded
*/
	//outln("slep took " << watch.stop());
	for(int i = 0; i < 2; i++) {
		checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
 /*   checkCudaErrors(cudaEventDestroy(start_event));
	checkCudaErrors(cudaEventDestroy(stop_event));
*/	return watch.stop();
}

float slepdbl() {
	CuTimer watch;

    watch.start();
    slep<<<1,1>>>(1000000);
    slep<<<1,1>>>(1000000);
    return watch.stop();
	//outln("slep took " << watch.stop());
}
template<typename T> float sigmoiddbl(DMatrix<T>& trg1, DMatrix<T>& trg2, const DMatrix<T>& src1, const DMatrix<T>& src2) {
	CuTimer watch;

    watch.start();
    unaryOpL(trg1, src1, sigmoidUnaryOp<T>());
    unaryOpL(trg2, src2, sigmoidUnaryOp<T>());
    return watch.stop();
}

template int testStrmOrNot<float>::operator()(int argc, char const ** args) const;
template int testStrmOrNot<double>::operator()(int argc, char const ** args) const;
template <typename T> int testStrmOrNot<T>::operator()(int argc, const char** args) const {
	int count = b_util::getCount(argc,args,500);
	checkCudaError(cudaGetLastError());
    CuTimer timer;
    float exeTime,exeTime2;
    float delTimes = 0,delTimes2 = 0;

    timer.start();
	for(int i = 0; i < count; i++) {
		delTimes += slepstr();
	}
	exeTime = timer.stop();
    outln(count << " slepstr took " << exeTime);
	checkCudaError(cudaGetLastError());

	timer.start();
	for(int i = 0; i < count; i++) {
		delTimes2 += slepdbl();
	}
	exeTime2 = timer.stop();
    outln(count << " slepdbl took " << exeTime2);
    outln(count << " delTimes took " << delTimes);
    outln(count << " delTimes2 took " << delTimes2);
	checkCudaError(cudaGetLastError());


	CuMatrix<T> seq = CuMatrix<T>::sequence(-5, 5000, 5000);
	CuMatrix<T> trg1 = CuMatrix<T>::zeros(5000, 5000);
	DMatrix<T> d_seq1= seq.asDmatrix();
	DMatrix<T> d_trg1= trg1.asDmatrix();
	CuMatrix<T> seq2 = CuMatrix<T>::sequence(-5, 5000, 5000);
	CuMatrix<T> trg2 = CuMatrix<T>::zeros(5000, 5000);
	DMatrix<T> d_seq2 = seq2.asDmatrix();
	DMatrix<T> d_trg2= trg2.asDmatrix();

    float delSigTimes = 0,delSigTimes2 = 0;

	timer.start();
	for (int i = 0; i < count; i++) {
		delSigTimes += sigmoidstr<T>(d_trg1, d_trg2, d_seq1, d_seq1);
	}
	exeTime = timer.stop();
	outln(count << " sigmoidstr took " << exeTime);

	timer.start();
	for(int i = 0; i < count; i++) {
		delSigTimes2 += sigmoiddbl<T>(d_trg1, d_trg2, d_seq1, d_seq1);
	}
	exeTime2 = timer.stop();
    outln(count << " sigmoiddbl took " << exeTime2);
    outln(count << " delSigTimes took " << delSigTimes);
    outln(count << " delSigTimes2 took " << delSigTimes2);

	return 0;
}

template int testBounds<float>::operator()(int argc, char const ** args) const;
template int testBounds<double>::operator()(int argc, char const ** args) const;
template <typename T> int testBounds<T>::operator()(int argc, const char** args) const {
	int count = b_util::getCount(argc,args,10);
	checkCudaError(cudaGetLastError());
	CuMatrix<T> seq = -1 * CuMatrix<T>::sequence(-5, 5000, 5000);
	CuMatrix<T> tiny = -1 * CuMatrix<T>::sequence(-5, 32, 32);
	outln("seq " << seq.syncBuffers());
	outln("tiny " << tiny.syncBuffers());
	checkCudaError(cudaGetLastError());
	T *min, *max;
	checkCudaError(cudaMallocHost(&min,sizeof(T)));
	checkCudaError(cudaMallocHost(&max,sizeof(T)));
    CuTimer timer;
    float boundsTime,sequentialMinMaxTime;

	uint nP = seq.m * seq.n;
	uint tinyP= tiny.m * tiny.n;
	outln("tinyP " << tinyP);
	outln("nP " << nP);

	uint threads;
	uint blocks;
	uint tthreads;
	uint tblocks;
	getReductionExecContext(blocks, threads, nP);
	getReductionExecContext(tblocks, tthreads, tinyP);
	outln("testBounds reduce blocks " << blocks);
	outln("testBounds reduce threads " << threads);
	outln("testBounds reduce tblocks " << tblocks);
	outln("testBounds reduce tthreads " << tthreads);
	CuMatrix<T> minBuffer(blocks, 1, true, true);
	minBuffer.syncBuffers();
	CuMatrix<T> maxBuffer(blocks, 1, true, true);
	maxBuffer.syncBuffers();

	DMatrix<T> d_minBuffer;
	minBuffer.asDmatrix(d_minBuffer, false);
	DMatrix<T> d_maxBuffer;
	maxBuffer.asDmatrix(d_maxBuffer, false);
	DMatrix<T> d_seq;
	DMatrix<T> d_tiny;
	seq.asDmatrix(d_seq, false);
	tiny.asDmatrix(d_tiny, false);
	maxBinaryOp<T> maxOp = Functory<T,maxBinaryOp>::pinch();
	minBinaryOp<T> minOp = Functory<T,minBinaryOp>::pinch();
    timer.start();
    outln("started timer");
    *min = *max = 0;
	checkCudaError(cudaGetLastError());
    *min = *max = 0;
	for(int i = 0; i < count; i++) {
		CuMatrix<T>::bounds( min,max, d_minBuffer, d_maxBuffer,d_seq, blocks, threads, nP);
	}
	boundsTime = timer.stop();
    outln(count << " bounds (" << *min << ", " << *max << ") took " << boundsTime);
	checkCudaError(cudaGetLastError());

	timer.start();
	for(int i = 0; i < count; i++) {
		*max = seq.max();
		*min = seq.min();
	}
	sequentialMinMaxTime = timer.stop();
    outln(count << " min/max (" << *min << ", " << *max << ") took " << sequentialMinMaxTime);
  	T seqSum = seq.sum();
	outln("seqSum sum " << seqSum);
	checkCudaError(cudaFreeHost(min));
	checkCudaError(cudaFreeHost(max));
	checkCudaError(cudaGetLastError());

	return 0;
}

template<typename T> __global__ void saxy(T* out, const T* in, T x,T y, uint len){
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len) {
		out[tid] = x * in[tid] + y;
	}
}
template<typename T> void hostSaxy(T* out, const T* in, T x,T y, uint len){
	for(uint tid = 0; tid < len; tid++ ) {
		out[tid] = x * in[tid] + y;
	}
}

template<typename T> T hostSum(const T* in, uint len){
	T totl=0;
	for(uint tid = 0; tid < len; tid++ ) {
		totl += in[tid];
	}
	return totl;
}

template int testLastError<float>::operator()(int argc, char const ** args) const;
template int testLastError<double>::operator()(int argc, char const ** args) const;
template <typename T> int testLastError<T>::operator()(int argc, const char** args) const {
	setCurrGpuDebugFlags( debugSpoofSetLastError,true,false);
	testSetLastErrorKernel<<<1,1>>>(columnOutOfBoundsEx);
	assert(columnOutOfBoundsEx == getLastError());
	return 0;
}

__global__ void child_launch(int *data) {
	data[threadIdx.x] = data[threadIdx.x] + 1;
}
__global__ void parent_launch(int *data) {
	data[threadIdx.x] = threadIdx.x;
	if(threadIdx.x < 255) data[threadIdx.x] += data[threadIdx.x + 1];
	//__syncthreads();
	if (threadIdx.x == 0) {
#ifdef CuMatrix_Enable_Cdp
		child_launch<<< 1, 256 >>>(data);
#else
	prlocf("not implemented for non-cdp\n");
	assert(false);
#endif
		//cudaDeviceSynchronize();
	}
	//__syncthreads();
}
void host_launch(int *data) {
}

template int testCdpSync1<float>::operator()(int argc, char const ** args) const;
template int testCdpSync1<double>::operator()(int argc, char const ** args) const;
template <typename T> int testCdpSync1<T>::operator()(int argc, const char** args) const {
	int *data, *h_data;
	int count = 256;
	int size = count * sizeof(int);
	cudaMalloc(&data, size);
	cudaMallocHost(&h_data, size);
	parent_launch<<< 1, count >>>(data);
	//cudaDeviceSynchronize();
	cudaMemcpy(h_data,data, size, cudaMemcpyDeviceToHost);
	outln("testCdpSync1 h_data results found\n " << ::parry(h_data,count));
	return 0;
}

template int testCdpRedux<float>::operator()(int argc, char const ** args) const;
template int testCdpRedux<double>::operator()(int argc, char const ** args) const;
template <typename T> int testCdpRedux<T>::operator()(int argc, const char** args) const {
	CuMatrix<T> big1s=CuMatrix<T>::ones(5000,8000);
	ulong nP = big1s.m * big1s.n;
	T *res, *hRes;
	cudaMalloc(&res,  sizeof(T));
	cudaMallocHost(&hRes,  sizeof(T));
	plusBinaryOp<T> plusOp = Functory<T,plusBinaryOp>::pinch();
	int count = b_util::getCount(argc,args,10);
	CuTimer timer;

	float exeTime, dexeTime;
	timer.start();
	T sum;
	uint threads, blocks;
	getReductionExecContext(blocks,threads,nP);
	outln("testCdpRedux nP " << nP << ", blocks " << blocks << ", threads " << threads);
	CuMatrix<T>buffer = CuMatrix<T>::reductionBuffer(blocks);
	DMatrix<T> buff = buffer.asDmatrix();
	DMatrix<T>src = big1s.asDmatrix();
	for(int i = 0; i < count; i++) {
		reduceLauncher(&sum, buff, nP, src, plusOp, (T)0,1,0);
	}
	exeTime = timer.stop();
	outln("sum " << sum);
	timer.start();
	//(T* result, DMatrix<T> buff, ulong nP, const DMatrix<T> src, BinaryOp op, T start, int count)
#ifdef CuMatrix_Enable_Cdp
	reduceLauncherCount<<<1,1>>>(res, buff, nP, src, plusOp,(T)0, count);
#else
	assert(0);
#endif
	cudaMemcpy(&sum,res,sizeof(T),cudaMemcpyDeviceToHost);
	outln("sum2  " << sum);
	dexeTime = timer.stop();

	outln("exeTime " << exeTime);
	outln("dexeTime " << dexeTime);
	return 0;
}

