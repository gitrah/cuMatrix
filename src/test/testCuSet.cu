/*
 * testCuSet.cu
 *
 *      Author: reid
 */
#include "tests.h"
#include "../CuSet.h"
#include "../CuMatrix.h"
/*
#include "../util.h"
#include "../MatrixExceptions.h"
#include "../Maths.h"
#include "testKernels.h"
*/

const char* DATA_CSV_FILE = "ex9data.csv.txt";


template int testCuSet1D<float>::operator()(int argc, const char **argv) const;
template int testCuSet1D<double>::operator()(int argc, const char **argv) const;
template int testCuSet1D<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testCuSet1D<T>::operator()(int argc, const char **argv) const {

	int len = 4096;
	ExecCaps* caps = ExecCaps::currCaps();
	int blockShmem = caps->memSharedPerBlock;

	int maxThreads = blockShmem/ (2 * sizeof(T));
	int threads = MIN(len, maxThreads);
	outln("testCuSet1D creatinmats");
	CuMatrix<T> sorted = CuMatrix<T>::zeros(1,len);
	CuMatrix<T> deduped = CuMatrix<T>::zeros(1,len);
	outln("testCuSet1D creatinspanmod");
	// ( int m, int n, T start, T end, int steps,  int gpuMask = Tiler<T>::gpuMaskFromCurrGpu(), TileDirection tileD = tdRows, bool colMajor = false, cudaStream_t stream = 0);
	CuMatrix<T> source = CuMatrix<T>::spanMod(1,1200, (T)0,(T)1200,len);
	outln("testCuSet1D createdpanmod");
	source.syncBuffers();
	sorted.syncBuffers();
	outln("testCuSet1D source " << source);

	vector<int> vIndices;

	CuMatrix<T> training, cv;

	outln("source.sum " << source.sum());
	source.transpose().shuffle(&training, &cv, (T) .75, vIndices);
	outln("shuffled ");
	training.syncBuffers();
	cv.syncBuffers();
	outln("training " << training);
	outln("cv " << cv);

	dim3 grid;
	dim3 block;
	b_util::vectorExecContext(threads, len, grid, block, (void*)setMakerIterativeKernel<T>);
	outln("testCuSet grid " << b_util::pd3(grid));
	outln("testCuSet block " << b_util::pd3(block));
	int* d_plen = nullptr;
	cherr(cudaMalloc(&d_plen, grid.x * sizeof(int)));
	int* h_pNewLen = (int* )malloc(grid.x * sizeof(int));
	int* d_psetlen = nullptr;
	cherr(cudaMalloc(&d_psetlen, grid.x * sizeof(int)));

	int smem = block.x * (sizeof(T) + sizeof(int));
	outln("smem " << smem);

	T* ptrdev_Sorted = sorted.currBuffer();
	T* ptrdev_Training = training.currBuffer();
	outln("ptrdev_Sorted " << ptrdev_Sorted);
	outln("ptrdev_Training " << ptrdev_Training);
	outln("current device " << ExecCaps::currDev());

	// T* setPtr, int* len, const int pset,	const T* atts, const int patts,const long n)
	//checkCudaErrors(cudaSetDevice(1));
	setMakerIterativeKernel<T><<<grid,block, smem>>>(
			ptrdev_Sorted, d_plen, 1, ptrdev_Training, 1, (long) source.n);
	outln("launched kernel, about to sync");
	cherr(cudaDeviceSynchronize());
	sorted.invalidateHost();
	sorted.syncBuffers();
	outln("sorted " << sorted);
	cherr(cudaMemcpy(h_pNewLen, d_plen, grid.x * sizeof(int),  cudaMemcpyDeviceToHost));
	int sets =  grid.x;
	int setOff = 0;
	int srcOff = 0;

	// merging
	maxThreads = blockShmem/ (3 * sizeof(T));
	threads = MIN(len, maxThreads);
	dim3 mgrid;
	dim3 mblock;//( 1 + MAX(h_pNewLen[0], h_pNewLen[1]));
	b_util::vectorExecContext(threads, len, mgrid, mblock, (void*)mergeSetsKernel<T>);

	int mshmem = mblock.x * 3 * sizeof(T);
	outln("mshmem " << mshmem);
	outln("deduped.currBuffer() " << deduped.currBuffer());
	outln("sorted.currBuffer() " << sorted.currBuffer());
	outln("sorted.currBuffer() + h_pNewLen[0] " << (sorted.currBuffer() + h_pNewLen[0]));
	outln("d_psetlen " << d_psetlen);
	outln("h_pNewLen[0] " << h_pNewLen[0]);
	outln("h_pNewLen[1] " << h_pNewLen[1]);
	// p ((@global double*) sorted.tiler.buffers.x)[0]@1024
	// p ((@global double*) deduped.tiler.buffers.x)[0]@1024
	mergeSetsKernel<T><<<mgrid,mblock,mshmem>>>( deduped.currBuffer(), d_psetlen, 1, sorted.currBuffer(), sorted.currBuffer() + h_pNewLen[0] + 1, 1, 1, h_pNewLen[0] + 1, h_pNewLen[1] + 1);
	cherr(cudaDeviceSynchronize());

	deduped.invalidateHost();
	deduped.syncBuffers();
	outln("deduped " << deduped);

	return 0;
}


template int testCuSet<float>::operator()(int argc, const char **argv) const;
template int testCuSet<double>::operator()(int argc, const char **argv) const;
template int testCuSet<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testCuSet<T>::operator()(int argc, const char **argv) const {

	cherr(cudaPeekAtLastError());
	int len = 512;
	ExecCaps* caps = ExecCaps::currCaps();
	int blockShmem = caps->memSharedPerBlock;

	int maxThreads = blockShmem/ (2 * sizeof(T));
	int threads = MIN(len, maxThreads);
	outln("testCuSet creatinmats");
	CuMatrix<T> sorted = CuMatrix<T>::zeros(1,len);
	CuMatrix<T> deduped = CuMatrix<T>::zeros(1,len);
	outln("testCuSet creatinspanmod");
	//CuMatrix<T> source = CuMatrix<T>::spanMod(1200,1,0,1200,len);
	//static CuMatrix<T> randmod(int rows, int cols, T epsilon = static_cast<T>(  1), int mod, bool colMajor = false);
	CuMatrix<T> source = CuMatrix<T>::randmod(1,len,(T)100, 20);
	outln("testCuSet createdpanmod");
	source.syncBuffers();
	sorted.syncBuffers();
	outln("source " << source);

	dim3 grid;
	dim3 block;
	b_util::vectorExecContext(threads, len, grid, block, (void*)setMakerIterativeKernel<T>);
	outln("testCuSet grid " << b_util::pd3(grid));
	outln("testCuSet block " << b_util::pd3(block));
	int* d_plen = nullptr;
	cherr(cudaMalloc(&d_plen, grid.x * sizeof(int)));
	int* h_pNewLen = (int* )malloc(grid.x * sizeof(int));
	int* d_psetlen = nullptr;
	cherr(cudaMalloc(&d_psetlen, grid.x * sizeof(int)));

	int smem = block.x * (sizeof(T) + sizeof(int));
	outln("grid-block " << b_util::pxd(grid, block )<< ", smem " << smem);

	T* ptrdev_Sorted = sorted.currBuffer();
	outln("ptrdev_Sorted " << ptrdev_Sorted);
	outln("sorted.currBuffer() " << sorted.currBuffer());
	// T* setPtr, int* len, const int pset,	const T* atts, const int patts,const long n)
	cherr(cudaDeviceSynchronize());
	cherr(cudaPeekAtLastError());
	setMakerIterativeKernel<T><<<grid, block, smem>>>(
			ptrdev_Sorted, d_plen, 1, source.currBuffer(), 1, (long) source.n);
	cherr(cudaDeviceSynchronize());
	sorted.invalidateHost();
	sorted.syncBuffers();
	outln("sorted " << sorted);
	cherr(cudaMemcpy(h_pNewLen, d_plen, grid.x * sizeof(int),  cudaMemcpyDeviceToHost));
	int sets =  grid.x;
	int setOff = 0;
	int srcOff = 0;


	// merging
	maxThreads = blockShmem/ (3 * sizeof(T));
	threads = MIN(len, maxThreads);
	dim3 mgrid;
	dim3 mblock;//( 1 + MAX(h_pNewLen[0], h_pNewLen[1]));
	b_util::vectorExecContext(threads, len, mgrid, mblock, (void*)mergeSetsKernel<T>);

	int mshmem = mblock.x * 3 * sizeof(T);
	outln("mshmem " << mshmem);
	outln("deduped.currBuffer() " << deduped.currBuffer());
	outln("sorted.currBuffer() " << sorted.currBuffer());
	outln("sorted.currBuffer() + h_pNewLen[0] " << (sorted.currBuffer() + h_pNewLen[0]));
	outln("d_psetlen " << d_psetlen);
	outln("h_pNewLen[0] " << h_pNewLen[0]);
	outln("h_pNewLen[1] " << h_pNewLen[1]);
	// p ((@global double*) sorted.tiler.buffers.x)[0]@1024
	// p ((@global double*) deduped.tiler.buffers.x)[0]@1024
	mergeSetsKernel<T><<<mgrid,mblock,mshmem>>>( deduped.currBuffer(), d_psetlen, 1, sorted.currBuffer(), sorted.currBuffer() + h_pNewLen[0] + 1, 1, 1, h_pNewLen[0] + 1, h_pNewLen[1] + 1);
	cherr(cudaDeviceSynchronize());

	deduped.invalidateHost();
	deduped.syncBuffers();
	outln("deduped " << deduped);

	return 0;
}

template int testDedeup<float>::operator()(int argc, const char **argv) const;
template int testDedeup<double>::operator()(int argc, const char **argv) const;
template int testDedeup<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testDedeup<T>::operator()(int argc, const char **argv) const {
	outln("in testDedeeuup");
	T a[] = {8,6,7,5,3,0,9,4,3,3,6,3,5,3,10,1};

	int aSize =  T_ARRAY_SIZE(a);

	util<T>::pDarry(a, aSize);
	CuSet<T> setter;

	setter.quicksort(a,0, aSize-1);
	util<T>::pDarry(a,  aSize);
	//verify sortosity
	for(int i =0; i < aSize; i++) {
		if(i < aSize-1) {
			assert( a[i] <= a[i+1]);
		}
	}

	int newAsize = 0;
	setter.dedup(a, &newAsize, aSize);
	util<T>::pDarry(a,  aSize);

	assert(newAsize < aSize);
	assert(newAsize == 10);

	//verify presiervashun d'sortosity
	for(int i =0; i < newAsize; i++) {
		if(i < newAsize-1) {
			assert( a[i] <= a[i+1]);
		}
	}

	return 0;
}

template int testJaccard<float>::operator()(int argc, const char **argv) const;
template int testJaccard<double>::operator()(int argc, const char **argv) const;
template int testJaccard<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testJaccard<T>::operator()(int argc, const char **argv) const {
	outln("in testJaccard");
	T a[] = {8,6,7,5,3,0,9,4,3,3,6,3,5,3,10,1};
	T b[] = {4,22,7,5,3,0,14,23,3,6,3,15,3,10,2};


	int aSize =  T_ARRAY_SIZE(a);
	int bSize =  T_ARRAY_SIZE(b);

	std::cout << "a:  ";
	for (int i = 0; i < aSize; i++)
		std::cout << ' ' << a[i];
	std::cout << '\n';

	std:cout << "b:  ";
	for (int i = 0; i < bSize; i++)
		std::cout << ' ' << b[i];
	std::cout << '\n';

	typename std::vector<T> vecT(MAX(aSize,bSize));                      // 0  0  0  0  0  0  0  0  0  0
	typename std::vector<T>::iterator itT;

	CuSet<T>::quicksort(a,0,aSize-1);
	CuSet<T>::quicksort(b,0,bSize-1);
	std::set<T> sa(a, a+aSize);
	std::set<T> sb(b, b+bSize);
/*

	itT = std::set_intersection(a,a+aSize,b,b+bSize,vecT.begin());
	vecT.resize(itT - vecT.begin());                      // 10 20
	std::cout << "The a-b intersection has " << (vecT.size()) << " elements:\n";
	for (itT = vecT.begin(); itT != vecT.end(); ++itT)
		std::cout << ' ' << *itT;
	std::cout << '\n';

	std::set<T> xs, un;
	typename std::set<T>::iterator iXs;
	typename std::set<T>::iterator iUn;
	std::set_intersection(sa.begin(),sa.end(),sb.begin(),sb.end(), std::inserter(xs, xs.end()));
	std::set_union(sa.begin(),sa.end(),sb.begin(),sb.end(), std::inserter(un, un.end()));


	//std::inserter(the_intersection, the_intersection.end())
	std::cout << "The a-b intersection has " << (xs.size()) << " elements:\n";
	for (iXs = xs.begin(); iXs != xs.end(); ++iXs)
		std::cout << ' ' << *iXs;
	std::cout << '\n';
	std::cout << "The a-b union has " << (un.size()) << " elements:\n";
	for (iUn = un.begin(); iUn != un.end(); ++iUn)
		std::cout << ' ' << *iUn;
	std::cout << '\n';*/


	std::cout << "The a-b jaccard index is  " << CuSet<T>::jaccard(sa,sb) << "\n";


	int first[] = { 5, 10, 15, 20, 25 };
	int second[] = { 50, 40, 30, 20, 10 };
	std::vector<int> v(10);                      // 0  0  0  0  0  0  0  0  0  0
	std::vector<int>::iterator it;

	CuSet<int>::quicksort(first, 0, 4);
	CuSet<int>::quicksort(second, 0, 4);

	it = std::set_intersection(first, first + 5, second, second + 5, v.begin());
	// 10 20 0  0  0  0  0  0  0  0
	v.resize(it - v.begin());                      // 10 20

	  std::cout << "The first-second intersection has " << (v.size()) << " elements:\n";
	  for (it=v.begin(); it!=v.end(); ++it)
	    std::cout << ' ' << *it;
	  std::cout << '\n';

//	std::set_intersection(a,a+aSize,b,b+bSize,xs);

	return 0;
}

template int testQpow<float>::operator()(int argc, const char **argv) const;
template int testQpow<double>::operator()(int argc, const char **argv) const;
template int testQpow<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testQpow<T>::operator()(int argc, const char **argv) const {
    CuTimer timer;

    {
        outln(  "testMergeSorted() enter");
        int len = 10000;
      	CuMatrix<T> vec1 = CuMatrix<T>::increasingRows(len,1, 1);
       	CuMatrix<T> vec2 = CuMatrix<T>::increasingRows( len,1, 500);
        outln(  "increasings");
      	CuMatrix<T> vecres = CuMatrix<T>::zeros(len, 1);
       	CuMatrix<T> vec3 = CuMatrix<T>::zeros(2000, 1);
        outln(  "zrs");
        const int trials = 5000;
        timer.start();
        outln(  "preloop");
        for(int i =0; i < trials; i++ ){
        	vecres = vec2.qpow(2);
        }
        outln( trials << " qpow(2) sum " << vecres.sum() << " took "<< timer.stop()/ 1000.0 << "s");
        timer.start();
        for(int i =0; i < trials; i++ ){
        	vecres = vec2.qpow(3);
        }
        outln( trials << " qpow(3) sum " << vecres.sum() << " took "<< timer.stop()/ 1000.0 << "s");
        timer.start();
        for(int i =0; i < trials; i++ ){
        	vecres = vec2.pow(2);
        }
        outln( trials << " pow(2) sum " << vecres.sum() << " took "<< timer.stop()/ 1000.0 << "s");
        timer.start();
        for(int i =0; i < trials; i++ ){
        	vecres = vec2.pow(3);
        }
        outln( trials << " pow(3) sum " << vecres.sum() << " took "<< timer.stop()/ 1000.0 << "s");

        return 0;
   }
}
template int testMergeSorted<float>::operator()(int argc, const char **argv) const;
template int testMergeSorted<double>::operator()(int argc, const char **argv) const;
template int testMergeSorted<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testMergeSorted<T>::operator()(int argc, const char **argv) const {
    CuTimer timer;

    {
        outln(  "testMergeSorted() enter");
        int len = 10000;
      	CuMatrix<T> vec1 = CuMatrix<T>::increasingRows(len,1, 1);
       	CuMatrix<T> vec2 = CuMatrix<T>::increasingRows( len,1, 500);
        outln(  "increasings");
      	CuMatrix<T> vecres = CuMatrix<T>::zeros(len, 1);
       	CuMatrix<T> vec3 = CuMatrix<T>::zeros(2000, 1);
        outln(  "zrs");
        const int trials = 5000;
        timer.start();
        enum cudaMemcpyKind  vecs13K = b_util::copyKind(vec3.elements, vec1.elements),  vecs23K = b_util::copyKind(vec3.elements, vec2.elements);

        //cherr( cudaMemcpy2D(vec3.elements, 8, vec1.elements, 8,8, ))
        for(int i =0; i < trials; i++ ){
        	CuSet<T>::mergeSorted(vec3.elements, vec1.elements, vec2.elements, vec1.length(),vec2.length(),vecs13K,vecs23K);
        }
        outln( trials << " mergeSorted took " << timer.stop()/ 1000.0 << "s");

        timer.start();
        for(int i =0; i < trials; i++ ){
        	CuSet<T>::mergeSorted2(vec3.elements, vec1.elements, vec2.elements, vec1.length(),vec2.length());
        }
        outln( trials << " mergeSorted2 took " << timer.stop()/ 1000.0 << "s");

      	CuMatrix<T> vec1od = CuMatrix<T>::sequence(len,1,1,2);
       	CuMatrix<T> vec2ev = CuMatrix<T>::sequence(len,1,2,2);
        timer.start();
        for(int i =0; i < trials; i++ ){
        	CuSet<T>::mergeSorted(vec3.elements, vec1od.elements, vec2ev.elements, vec1od.length(),vec2ev.length());
        }
        outln( trials << " mergeSorted ev/od overlap took " << timer.stop()/ 1000.0 << "s");

        timer.start();
        for(int i =0; i < trials; i++ ){
        	CuSet<T>::mergeSorted2(vec3.elements, vec1od.elements, vec2ev.elements, vec1od.length(),vec2ev.length());
        }
        outln( trials << " mergeSorted2 ev/od overlap took " << timer.stop()/ 1000.0 << "s");

    	CuMatrix<T> steps = CuMatrix<T>::increasingRows( 1000,1000,1);
		outln("steps  "<<  steps.syncBuffers());
        timer.start();
    	CuMatrix<T> muSteps = CuMatrix<T>::zeros(1,1000);
    	steps.featureMeans(muSteps,  true);
		outln("muSteps  took "<< timer.stop() / 1000.0 << "\n" << muSteps.syncBuffers());
        timer.start();
    	CuMatrix<T> stepsT = steps.transpose();
		outln("stepsT  took "<< timer.stop() / 1000.0 << "\n" << stepsT.syncBuffers());
        timer.start();
        CuMatrix<T> muStepsT = CuMatrix<T>::zeros(1,1000);
        stepsT.featureMeansTx(muStepsT);
		outln("muStepsT  took "<< timer.stop() / 1000.0 << "\n" << muStepsT.syncBuffers());
    }

	{
		// put in blocks to recover h/d memz
		timer.start();
		CuMatrix<T> bigun = CuMatrix<T>::odds(1000, 1000, 1);
		int2 idx_105011 = bigun.indexOf2D(105011);
		outln("idx_105011 (" << idx_105011.x << ", " << idx_105011.y << ") took "<< timer.stop() / 1000.0);
	}

	{
	   timer.start();
		CuMatrix<T> bigv = CuMatrix<T>::odds(1, 1000000);
		int bigv_1 = bigv.indexOfGlolo(1);
		outln("bigv_1 " << bigv_1 << " took "<< timer.stop() / 1000.0);

		timer.start();
		int bigv_3 = bigv.indexOfGlolo(3);
		outln("bigv_3 " << bigv_3 << " took "<< timer.stop() / 1000.0);

		timer.start();
		int bigv_end = bigv.indexOfGlolo(bigv.get(0,999999));
		outln("bigv_end " << bigv_end << " took "<< timer.stop() / 1000.0);
	}

	T* sortedA = nullptr, *sortedB = nullptr;
	int lenA, lenB;

	   {
		CuMatrix<T> out1 = CuMatrix<T>::zeros(1,5).syncBuffers();
		CuMatrix<T> out2 = CuMatrix<T>::zeros(1,5).syncBuffers();
		CuMatrix<T> out3 = CuMatrix<T>::zeros(1,5).syncBuffers();
		CuMatrix<T> out4 = CuMatrix<T>::zeros(1,5).syncBuffers();
		CuMatrix<T> out5 = CuMatrix<T>::zeros(1,5).syncBuffers();
	    	CuMatrix<T> m0 = CuMatrix<T>::fill(0,1,1).syncBuffers();
	    	CuMatrix<T> m1 = CuMatrix<T>::fill(1,1,1).syncBuffers();
	    	CuMatrix<T> m2 = CuMatrix<T>::fill(2,1,1).syncBuffers();
	    	CuMatrix<T> m3 = CuMatrix<T>::fill(3,1,1).syncBuffers();
	    	CuMatrix<T> m4 = CuMatrix<T>::fill(4,1,1).syncBuffers();
	    	CuMatrix<T> m5 = CuMatrix<T>::fill(5,1,1).syncBuffers();
	    	sortedA = m3.elements;
	    	sortedB = m4.elements;
	    	int newSize = CuSet<T>::mergeSorted(out1.elements, sortedA, sortedB, 1, 1);
			outln("newSize " << newSize);
	    	sortedB = m0.elements;
	    	newSize = CuSet<T>::mergeSorted(out2.elements, out1.elements, sortedB, newSize, 1);
	    	sortedB = m2.elements;
	    	newSize = CuSet<T>::mergeSorted(out3.elements, out2.elements, sortedB, newSize, 1);
	    	sortedB = m1.elements;
	    	newSize = CuSet<T>::mergeSorted(out4.elements, out3.elements, sortedB, newSize, 1);
	    	sortedB = m5.elements;
	    	newSize = CuSet<T>::mergeSorted(out5.elements, out4.elements, sortedB, newSize, 1);
	    }

	CuMatrix<T> m_1by10_1_odd = CuMatrix<T>::odds(1, 10, 1).syncBuffers();
	CuMatrix<T> m_1by10_2_odd = CuMatrix<T>::odds(1, 10, m_1by10_1_odd.get(0,4)+2).syncBuffers() ;
	CuMatrix<T> m_1by10_3_odd = CuMatrix<T>::odds(1, 10, m_1by10_2_odd.get(0,4)+2).syncBuffers() ;
	outln("m_1by10_1_odd " << m_1by10_1_odd);
	outln("m_1by10_2_odd " << m_1by10_2_odd);
	outln("m_1by10_3_odd " << m_1by10_3_odd);

	sortedA = m_1by10_1_odd.elements;
	lenA =  m_1by10_1_odd.size/sizeof(T);

	CuMatrix<T> m_1by10_1_even = CuMatrix<T>::evens(1, 10, 2).syncBuffers();
	CuMatrix<T> m_1by10_2_even = CuMatrix<T>::evens(1, 10, m_1by10_1_even.get(0,4) +2 ).syncBuffers() ;
	CuMatrix<T> m_1by10_3_even = CuMatrix<T>::evens(1, 10, m_1by10_2_even.get(0,4) +2 ).syncBuffers();
	lenB =  m_1by10_3_even.size/sizeof(T);
	sortedB = m_1by10_3_even.elements;

	outln("m_1by10_1_even " << m_1by10_1_even);
	outln("m_1by10_2_even " << m_1by10_2_even);
	outln("m_1by10_3_even " << m_1by10_3_even);

	// test bpSearch (before entire list
	assert( CuSet<T>::bpSearch(m_1by10_1_odd.elements, 10, 0) == 0);
	// test bpSearch eq 1st ele
	assert( CuSet<T>::bpSearch(m_1by10_1_odd.elements, 10, 1) == 0);
	// test bpSearch between 1st and 2nd
	assert( CuSet<T>::bpSearch(m_1by10_1_odd.elements, 10, 2) == 1);
	// test bpSearch eq midpoint -1
	assert( CuSet<T>::bpSearch(m_1by10_1_odd.elements, 10, 9) == 4);
	// test bpSearch ed midpoint
	assert( CuSet<T>::bpSearch(m_1by10_1_odd.elements, 10, 11) == 5);
	// test bpSearch ed midpoint + 1
	assert( CuSet<T>::bpSearch(m_1by10_1_odd.elements, 10, 18) == 9);
	// test bpSearch bet last and 2nd last
	assert( CuSet<T>::bpSearch(m_1by10_1_odd.elements, 10, 20) == 10);
	// test bpSearch eq last ele
	// test bpSearch aft last

	Packeur2 p2;
	p2.pack(sortedA[0],sortedA[lenA - 1],  sortedB[0], sortedB[lenB - 1]);
	double4 d4;
	p2.render(d4);

	Packeur3<T> p3;
	p3.pack(sortedA[0],sortedA[lenA - 1],  sortedB[0], sortedB[lenB - 1]);

	T firstA = sortedA[0];
	T firstB = sortedB[0];
	T lastA = sortedA[lenA - 1];
	T lastB = sortedB[lenB - 1];

	//OrderedIntersectionType p3sr = CuSet<T>::getOrderedIntersectionType( p3.render());
	//outln("ordXsectTypeStr a,b " <<  CuSet<T>::ordXsectTypeStr( p3sr ));
	CuMatrix<T> out = CuMatrix<T>::zeros(1,lenA + lenB).syncBuffers();
	int newSize = CuSet<T>::mergeSorted(out.elements, sortedA, sortedB, lenA, lenB);
	out.invalidateDevice();
	outln("out " << out);
	assert(out.sum() == m_1by10_1_odd.sum()+m_1by10_3_even.sum());

	sortedB = m_1by10_2_even.elements;
	firstB = sortedB[0];
	lastB = sortedB[lenB - 1];
	out.zero();
	out.syncBuffers();
	//Counter<T> outCtr = new
	//outln("ordXsectTypeStr a,b " <<  CuSet<T>::ordXsectTypeStr( CuSet<T>::getOrderedIntersectionType( firstA, lastA, firstB, lastB)));
	CuSet<T>::mergeSorted(out.elements, sortedA, sortedB, lenA, lenB);
	out.invalidateDevice();
	outln("out " << out);
	assert(out.sum() == m_1by10_1_odd.sum()+m_1by10_2_even.sum());


	lenB =  m_1by10_2_even.size/sizeof(T);
	sortedB = m_1by10_2_even.elements;
	out.zero();

/*
	CuMatrix<T> a2 = CuMatrix<T>::odds(1, 10, 5);
	CuMatrix<T> a3 = z5 |= CuMatrix<T>::odds(1, 5, 5);

	CuMatrix<T> b1 = CuMatrix<T>::evens(1, 10, 2).syncBuffers();
	CuMatrix<T> b2 = CuMatrix<T>::evens(1, 10, 6);
	CuMatrix<T> b3 = z5 |= CuMatrix<T>::evens(1, 5, 5);
*/

	return 0;
}


template int testOrderedCountedSet<float>::operator()(int argc, const char **argv) const;
template int testOrderedCountedSet<double>::operator()(int argc, const char **argv) const;
template int testOrderedCountedSet<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testOrderedCountedSet<T>::operator()(int argc, const char **argv) const {

	T a [] = {2,5,18,3,2,5,7,9,11,3,20,15,11, 7,7,7,7, 15,19};
	T b [] = {18, 19, 15, 10, 11, 10, 8, 5, 5, 7, 11, 12, 14, 19, 7, 17, 8, 6, 11, 16};
	int alen = sizeof(a)/sizeof(T);
	int blen = sizeof(b)/sizeof(T);
	outln("alen " <<alen);
	outln("blen " <<blen);
	CuMatrix<T> ma(a, alen,1,true);
	OrderedCountedSet<T> ocs(alen,CudaHostAlloced,CudaHostAlloced);
	outln("deduping" );
	CuSet<T>::dedupOCS(ocs,  a, alen);
	CuSet<T>::quicksort(a, 0, alen-1);
	CuSet<T>::quicksort(ocs);

	// 		const char * path, const char* sepChars, bool colMajor, 	bool matrixOwnsBuffer, bool hasXandY
/*
	map<string, CuMatrix<T>*> results = CuMatrix<T>::parseCsvDataFile(DATA_CSV_FILE,",",false, true,false);

	CuMatrix<T>& x = *results["x"];
	// 32:   Object { field: "partyonlineservicejoindate_diff", values: Array[381] }

	outln("x " << x);

	CuMatrix<T> col0 = x.columnMatrix(0);

	outln("col 0 " << col0);
	T pmax= col0.max();
	outln("pmax " << pmax);

	CuMatrix<T> ownership_typ = x.columnMatrix(7);
	outln("ownership_typ " << ownership_typ);
	pmax= ownership_typ.max();
	outln("ownership_typ pmax " << pmax);
	CuMatrix<T> employee_relation_typ = x.columnMatrix(6);
	outln("employee_relation_typ " << employee_relation_typ);
	pmax= employee_relation_typ.max();
	outln("employee_relation_typ pmax " << pmax);
*/

	return 0;
}

