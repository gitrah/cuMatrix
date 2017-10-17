/*
 * Kmeans.cu
 *
 *  Created on: Apr 24, 2014
 *      Author: reid
 */

#include "Kmeans.h"
#include "Maths.h"
/*
 *  block must be wide enough to cover features (x.n)
 * shared contents;
 * 	copy of the centroids
 * 		?? feature deltas for current row in x and current centroid (if not shuffle-reducing; x.n > WARP_SIZE
 */
template <typename T> __global__ void findClosestKernel(uint* indices, const DMatrix<T> centroids, const DMatrix<T> x) {
	assert(centroids.n == x.n);

	// dim centroids.m * centroids.n
	T* s_centroids = SharedMemory<T>();

	// dim blockDim.y * centroids.n to hold blockDim.y rows of feature deltas
	T* delta = s_centroids + centroids.m * centroids.n;

	// dim blockDim.y x 1  to hold blockDim.y row totals
	T* total= delta + centroids.n * blockDim.y;

	// dim blockDim.y x 1 to hold blockDim.y curr min deltas
	T* minDelta= total + blockDim.y;

	uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
	uint tidy = threadIdx.y + blockIdx.y * blockDim.y;
	uint tib = threadIdx.x + threadIdx.y * blockDim.x;
	T del;

	if ( tidx == 0 ) {
		minDelta[threadIdx.y] = util<T>::maxValue();
	}

	// copy centroids into shared memory, one thread per centroid element
	if(threadIdx.y < centroids.m && threadIdx.x < centroids.n) {
		s_centroids [threadIdx.y * centroids.n + threadIdx.x] = centroids.elements[threadIdx.y *centroids.p + threadIdx.x];
	}
	__syncthreads();

	// thread per row-feature
	if( tidy < x.m && tidx < x.n) {
		T* currXrow = x.elements + tidy * x.p;
		T* currCentRow = s_centroids;
		uint deltaOff = threadIdx.y * x.n + threadIdx.x;
		uint laneid;
		b_util::laneid(laneid);
		for(int crow = 0; crow < centroids.m; crow++) {
			// need test that fails for each column of a spanrow
			if(x.n <= WARP_SIZE  && !spanrowQ(crow, x.n)) { // todo check this for x.n > 2 !!!
				del = currXrow[threadIdx.x] - currCentRow[threadIdx.x];
				del *= del;  // each thread holds square dist bet current feature and that feature of current centroid
				int currLen = b_util::nextPowerOf2(x.n)/2;
				while(currLen > 0 ) {
					T ndel = shflDown<T>(del, currLen);  // not tib
					if(checkDebug(debugMeans)) {
						flprintf("crow %d tib %u del %f ndel %f\n", crow, tib, del, ndel);
					}
					if(threadIdx.x + currLen < x.n)
						del += ndel;
					currLen >>= 1;
				}
				if(tidx == 0) {
					if(del < minDelta[threadIdx.y]) {
						minDelta[threadIdx.y]= del;
						indices[tidy] = crow;
					}
				}
			} else {
				delta[deltaOff] = currXrow[threadIdx.x] - currCentRow[threadIdx.x];
				delta[deltaOff] *= delta[deltaOff];
				__syncthreads();
				if(tidx == 0) {
					if(checkDebug(debugMeans)){
						switch(x.n) {
						case 2:
							flprintf("row %d crow %d: %f %f\n",tidy,crow,delta[threadIdx.y * x.n ],delta[threadIdx.y * x.n + 1] );
							break;
						case 3:
							flprintf("row %d crow %d: %f %f %f\n",tidy,crow,delta[threadIdx.y * x.n ],delta[threadIdx.y * x.n + 1],delta[threadIdx.y * x.n + 2] );
							break;
						case 4:
							flprintf("row %d crow %d: %f %f %f %f\n",tidy,crow,delta[threadIdx.y * x.n ],delta[threadIdx.y * x.n + 1],delta[threadIdx.y * x.n + 2],delta[threadIdx.y * x.n + 3] );
							break;
						}
					}
					total[threadIdx.y] = 0;
					for(int col = 0; col < centroids.n; col++) {
						total[threadIdx.y] += delta[threadIdx.y * x.n + col];
					}
					if(checkDebug(debugMeans))flprintf("xrow %u crow %d total %f\n", tidy, crow, total[threadIdx.y]);
					if(total[threadIdx.y] < minDelta[threadIdx.y]) {
						if(checkDebug(debugMeans))flprintf("ty %d tx %d found new minDelta %f @ %d\n",tidy, tidx, total[threadIdx.y], crow);
						minDelta[threadIdx.y]= total[threadIdx.y];
						indices[tidy] = crow;
					}
				}
			}
			currCentRow += centroids.n;
		}
	} else {
		if(checkDebug(debugMeans))flprintf("tidy %u !< x.m %u || tidx %u < x.n %u\n",tidy , x.m , tidx , x.n);
	}
}

template <typename T> __host__ CUDART_DEVICE void Kmeans<T>::findClosest(IndexArray& indices, const CuMatrix<T>& centroids, const CuMatrix<T>& x) {
	assert(indices.count == x.m);
	assert(centroids.n == x.n);
	assert( x.tiler.tileSize == x.tiler.m_size); // todo tile impl
	uint threadX, threadY;
	threadX = x.n;
	ExecCaps* pcaps = ExecCaps::currCaps();
	assert(threadX < pcaps->maxBlock.x ) ;
	uint centroidsSize = sizeof(T) * (centroids.m * centroids.n );
	uint perRowDeltaSize = sizeof(T) * centroids.n;
	uint perRowTotalAndMinSize = 2 * sizeof(T);
	uint perRowSmemSize =  perRowDeltaSize + perRowTotalAndMinSize;
	if(checkDebug(debugMeans))flprintf("centroidsSize %u, perRowDeltaSize %u, perRowTotalAndMinSize %u, perRowSmemSize %u\n",centroidsSize,perRowDeltaSize,perRowTotalAndMinSize,perRowSmemSize);
	assert(perRowSmemSize <= pcaps->memSharedPerBlock);
	threadY = MIN(x.m, MIN( (pcaps->thrdPerBlock-1)/threadX,  MIN( (pcaps->memSharedPerBlock - centroidsSize)/ perRowSmemSize, pcaps->maxBlock.y)));
	if(checkDebug(debugMeans))flprintf("threadY %u\n",threadY);
	assert(threadX * threadY < pcaps->thrdPerBlock);

	struct cudaFuncAttributes funcAttrib;
    cherr(cudaFuncGetAttributes(&funcAttrib, findClosestKernel<T>));
    flprintf("findClosestKernel numRegs=%d maxThreadsPerBlock=%d localSizeBytes=%d, sharedSizeBytes =%d \n",funcAttrib.numRegs, funcAttrib.maxThreadsPerBlock, funcAttrib.localSizeBytes, funcAttrib.sharedSizeBytes);
	uint gridY = DIV_UP(x.m,threadY);
	uint gridX = 1;
    dim3 block(threadX, threadY);
	dim3 grid(gridX,gridY);

	b_util::adjustExpectations(grid,block,funcAttrib);
	uint smemSize = centroidsSize + perRowSmemSize * threadY;
	if(checkDebug(debugMeans)){
		flprintf("findClosest x: %uX%u, (%p-%p), means; %uX%u, (%p-%p)\n", x.m,x.n, x.tiler.currBuffer(), x.tiler.currBuffer() + (x.m -1) * x.p + x.n - 1,
				centroids.m,centroids.n, centroids.tiler.currBuffer(), centroids.tiler.currBuffer()+ centroids.m * centroids.n -1);
		flprintf("findClosest smemSize %u, block %uX%u, grid %uX%u\n",smemSize, threadY, threadX, gridY, 1);
	}
	uint *d_indx;
#ifndef __CUDA_ARCH__
	cherr(cudaMalloc(&d_indx, x.m*sizeof(uint)));
#else
	d_indx = indices.indices;
#endif

	flprintf("smemSize %d, grid ", smemSize);
	b_util::prd3(grid);
	prlocf("block ");
	b_util::prd3(block);

	findClosestKernel<<<grid,block, smemSize>>>(d_indx,centroids.asDmatrix(), x.asDmatrix());
	cherr(cudaDeviceSynchronize());
#ifndef __CUDA_ARCH__
	CuTimer timer;
	timer.start();
	cherr(cudaMemcpy(indices.indices,d_indx, x.m*sizeof(uint), cudaMemcpyDeviceToHost));
	//CuMatrix<T>::incDhCopy("Kmeans<T>::findClosest",x.m*sizeof(uint),timer.stop());
	if(checkDebug(debugMeans)) prlocf("copied dev indices to host\n");
	cherr(cudaFree(d_indx));
#endif
}
template  __host__ CUDART_DEVICE void Kmeans<float>::findClosest(IndexArray& indices, const CuMatrix<float>& means, const CuMatrix<float>& x);
template  __host__ CUDART_DEVICE void Kmeans<double>::findClosest(IndexArray& indices, const CuMatrix<double>& means, const CuMatrix<double>& x);
template  __host__ CUDART_DEVICE void Kmeans<ulong>::findClosest(IndexArray& indices, const CuMatrix<ulong>& means, const CuMatrix<ulong>& x);


// 1 thread per column
template <typename T> __global__ void calcMeansColThreadKernel(DMatrix<T> nCentroids, const uint* indices, const uint* counts, const DMatrix<T> x) {
	T* s_centroids = SharedMemory<T>(); // dim centroids.m * centroids.n
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	T* centRow;
	T* xRow;
	for(int i = 0; i < x.m; i++) {
		centRow =  nCentroids.elements + indices[i] * nCentroids.n;
		xRow = x.elements + i * x.p ;
		centRow[col] += xRow[col];
	}
	__syncthreads();
	if(threadIdx.x ==0) {
		if(checkDebug(debugMeans)){
			T xT = 0,yT = 0,x,y;
			for(int i = 0; i < nCentroids.m; i++) {
				x=nCentroids.elements[ i * nCentroids.p];
				y=nCentroids.elements[ i * nCentroids.p + 1];
				xT += x; yT += y;
				flprintf("cent %d (%f, %f)\n", i, x,y);
			}
			flprintf("total %f,%f\n", xT, yT);
		}
	}
	for(int i = 0; i < nCentroids.m; i++) {
		centRow =  nCentroids.elements + i * nCentroids.n;
		centRow[col] /= counts[i];
	}
}


template<int K>__global__ void countIndices(uint* counts, const uint* indices, int n) {
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < n) {
		atomicAdd(counts + indices[tid],1);
	}
}


template <typename T> __host__ CUDART_DEVICE void Kmeans<T>::calcMeansColThread(IndexArray& indices, CuMatrix<T>& centroids, const CuMatrix<T>& x){
	if(checkDebug(debugRedux))flprintf("enter %d\n",0);
	assert(indices.count == x.m);
	assert(centroids.n == x.n);
	uint threadX;
	threadX = x.n;
	ExecCaps* pcaps = ExecCaps::currCaps();
	assert(threadX < pcaps->maxBlock.x ) ;

	DMatrix<T> d_centroids = centroids.asDmatrix();
	DMatrix<T> d_x= x.asDmatrix();

	uint*  counts= new uint[centroids.m];
	memset(counts,0,centroids.m * sizeof(uint));

	uint *d_indx;
	uint *d_counts;
#ifndef __CUDA_ARCH__
	cherr(cudaMalloc(&d_indx, x.m*sizeof(uint)));
	cherr(cudaMalloc(&d_counts, centroids.m * sizeof(uint)));
	cherr(cudaMemcpy(d_indx, indices.indices, x.m*sizeof(uint), cudaMemcpyHostToDevice));
	cherr(cudaMemcpy(d_counts, counts, centroids.m*sizeof(uint), cudaMemcpyHostToDevice));
#else
	d_indx = indices.indices;
	d_counts = counts;
#endif
	uint blockX = MAX(1, DIV_UP(indices.count,threadX));
	if(checkDebug(debugRedux)) {
		flprintf("counting indices block %u threads %u\nbefore call: ",blockX,threadX);
		util<uint>::pDarry(d_counts,centroids.m);
	}
	switch(centroids.m) {
	case 5:
		countIndices<5><<<blockX,threadX>>>(d_counts, d_indx,indices.count);
		break;
	case 4:
		countIndices<4><<<blockX,threadX>>>(d_counts, d_indx,indices.count);
		break;
	case 3:
		countIndices<3><<<blockX,threadX>>>(d_counts, d_indx,indices.count);
		break;
	case 2:
		countIndices<2><<<blockX,threadX>>>(d_counts, d_indx,indices.count);
		break;
	default:
		setLastError(notImplementedEx);
	}

	cherr(cudaDeviceSynchronize());
	//if(checkDebug(debugRedux)) {
		prlocf("after call:  ");
		util<uint>::pDarry(d_counts,centroids.m);
	//}

	blockX = MAX(1, DIV_UP(x.n, threadX));
	if(checkDebug(debugRedux))flprintf("calling calcMeansColThreadKernel<<<%u,%u>>\n",blockX,threadX);
	//calcMeansColThreadKernel(DMatrix<T> nCentroids, const uint* indices, const uint* counts, const DMatrix<T> x)
	calcMeansColThreadKernel<<<blockX, threadX>>>(d_centroids, d_indx, d_counts, d_x);
	cherr(cudaDeviceSynchronize());
	centroids.invalidateHost();

#ifndef __CUDA_ARCH__
	cherr(cudaFree(d_indx));
	CuTimer timer;
	timer.start();
	cherr(cudaMemcpy(counts, d_counts, centroids.m*sizeof(uint), cudaMemcpyDeviceToHost));
	//CuMatrix<T>::incDhCopy("Kmeans<T>::calcMeansColThread",centroids.m*sizeof(uint),timer.stop());
	cherr(cudaFree(d_counts));
#endif
	delete[] counts;
}
template  __host__ CUDART_DEVICE void Kmeans<float>::calcMeansColThread(IndexArray& indices, CuMatrix<float>& means, const CuMatrix<float>& x);
template  __host__ CUDART_DEVICE void Kmeans<double>::calcMeansColThread(IndexArray& indices, CuMatrix<double>& means, const CuMatrix<double>& x);
template  __host__ CUDART_DEVICE void Kmeans<ulong>::calcMeansColThread(IndexArray& indices, CuMatrix<ulong>& means, const CuMatrix<ulong>& x);

/*
 * alg
 * 	1st reduction
 * 		for regular reduction, there is a block-sized vector to hold intermediate results;
 * 		we need (feature count 'n' * centroid count 'k') of those vectors
 * 		each sample row derefs its centroid from the IndexArray,
 * 		each column derefs its feature -> particular feature-centroid block
 *
 * 		will this all fit in smem, or will I need to do a launch per centroid etc
 *
 */
template <typename T> __host__ CUDART_DEVICE void Kmeans<T>::calcMeans(IndexArray& indices, CuMatrix<T>& centroids, const CuMatrix<T>& x){
	assert(indices.count == x.m);
	assert(centroids.n == x.n);
	uint threadX, threadY;
	threadX = x.n;
	ExecCaps* pcaps = ExecCaps::currCaps();
	assert(threadX < pcaps->maxBlock.x ) ;
	if(checkDebug(debugMeans)) flprintf("x.m %u maxBlock.y %u\n", x.m, pcaps->maxBlock.x);
	threadY = MIN(x.m, pcaps->maxBlock.y); // max block y
	if(checkDebug(debugMeans)) flprintf("MIN(x.m, pcaps->maxBlock.y) %u\n", threadY);
	if(checkDebug(debugMeans)) flprintf("pcaps->thrdPerBlock/threadX %u\n", pcaps->thrdPerBlock/threadX);
	threadY = MIN(threadY, pcaps->thrdPerBlock/threadX); // max sm threads
	if(checkDebug(debugMeans)) flprintf("MIN(threadY, pcaps->thrdPerBlock/threadX) %u\n",threadY);
	uint szPerCentroidSet = sizeof(T) * (centroids.m * centroids.n );
	uint maxCentCopiesPerSM = pcaps->memSharedPerBlock/szPerCentroidSet; // == gridY = x.m/threadY -> threadY =  x.m/gridY == x.m / maxCentCopies
	if(checkDebug(debugMeans)) flprintf("maxCentCopiesPerSM %u\n", maxCentCopiesPerSM);
	if(maxCentCopiesPerSM < x.m)
		threadY = MIN( x.m/maxCentCopiesPerSM, threadY);
	if(checkDebug(debugMeans)) flprintf("MIN( x.m/maxCentCopiesPerSM, threadY)  %u\n",threadY);

	uint gridY =  threadY < x.m ? DIV_UP(x.m, threadY) : 1;
	uint smemSize = gridY * szPerCentroidSet;
	if(checkDebug(debugMeans)) flprintf("gridY %u smemSize %u\n",gridY,smemSize);

	assert(threadX * threadY < pcaps->thrdPerBlock);
	dim3 block(threadX, threadY);
	assert(gridY < pcaps->maxGrid.y);
	dim3 grid(1,gridY);

	uint*  counts= new uint[centroids.m];
	memset(counts,0,centroids.m * sizeof(uint));

	uint *d_indx = null;
	uint *d_counts = null;
#ifndef __CUDA_ARCH__
	cherr(cudaMalloc(&d_indx, x.m*sizeof(uint)));
	cherr(cudaMalloc(&d_counts, centroids.m * sizeof(uint)));
	cherr(cudaMemcpy(d_indx, indices.indices, x.m*sizeof(uint), cudaMemcpyHostToDevice));
	cherr(cudaMemcpy(d_counts, counts, centroids.m*sizeof(uint), cudaMemcpyHostToDevice));
#else
	d_indx = indices.indices;
	d_counts = counts;
#endif
	prlocf("before launch, counts\n");
	util<uint>::pDarry(d_counts,centroids.m);

	//assert(0);
	calcMeansColThreadKernel<T><<<grid,block, smemSize>>>(centroids.asDmatrix(), d_indx, d_counts, x.asDmatrix());
	//<<<>>>
	util<uint>::pDarry(d_counts,centroids.m);
	cherr(cudaDeviceSynchronize());
#ifndef __CUDA_ARCH__
	cherr(cudaFree(d_indx));
	CuTimer timer;
	timer.start();
	cherr(cudaMemcpy(counts, d_counts, centroids.m*sizeof(uint), cudaMemcpyDeviceToHost));
	//CuMatrix<T>::incDhCopy("Kmeans<T>::calcMeans" ,centroids.m*sizeof(uint),timer.stop());
	cherr(cudaFree(d_counts));
#endif
	delete[] counts;

}
template  __host__ CUDART_DEVICE void Kmeans<float>::calcMeans(IndexArray& indices, CuMatrix<float>& means, const CuMatrix<float>& x);
template  __host__ CUDART_DEVICE void Kmeans<double>::calcMeans(IndexArray& indices, CuMatrix<double>& means, const CuMatrix<double>& x);
template  __host__ CUDART_DEVICE void Kmeans<ulong>::calcMeans(IndexArray& indices, CuMatrix<ulong>& means, const CuMatrix<ulong>& x);


template <typename T> __host__ CUDART_DEVICE T Kmeans<T>::distortion(IndexArray& indices, CuMatrix<T>& centroids, const CuMatrix<T>& x) {
	CuMatrix<T> centMap = centroids.derefIndices(indices);
#ifndef __CUDA_ARCH__
	outln("centMap\n" << centMap.syncBuffers());
#endif
	return x.sumSqrDiff(centMap)/x.m;

}
template  __host__ CUDART_DEVICE float Kmeans<float>::distortion(IndexArray& indices, CuMatrix<float>& means, const CuMatrix<float>& x);
template  __host__ CUDART_DEVICE double Kmeans<double>::distortion(IndexArray& indices, CuMatrix<double>& means, const CuMatrix<double>& x);
template  __host__ CUDART_DEVICE ulong Kmeans<ulong>::distortion(IndexArray& indices, CuMatrix<ulong>& means, const CuMatrix<ulong>& x);

