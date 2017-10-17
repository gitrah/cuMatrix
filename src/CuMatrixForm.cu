#include "CuMatrix.h"
#include "util.h"
#include "debug.h"
#include "caps.h"
#include "MatrixExceptions.h"
#include "Kernels.h"
#include <set>

template <typename T> struct LinkedAttribute {
	T value;
	int count;
	LinkedAttribute<T>* previous;
	LinkedAttribute<T>* next;

	static void insertBefore(LinkedAttribute<T>* targ, LinkedAttribute<T>* item ) {
		item->next = targ;
		LinkedAttribute<T>* tmp = targ->previous;
		if(tmp != nullptr) {
			tmp->next = item;
			item->previous = tmp;
		}
		targ->previous = item;
	}
	static void insertAfter(LinkedAttribute<T>* targ, LinkedAttribute<T>* item ) {
		item->previous = targ;
		LinkedAttribute<T>* tmp = targ->next;
		if(tmp != nullptr) {
			tmp->previous = item;
			item->next = tmp;
		}
		targ->next = item;
	}
};

template <typename T> __global__ void binaryCategoryKernel(const T* sElements, T* tElements, int tpitch, int spitch, int width, int height, bool oneBased)
{
	T* tile = SharedMemory<T>();

    uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    uint idxOut = xIndex + yIndex * tpitch;
    if(blockDim.x == threadIdx.x == 0) {
    	tile[threadIdx.y] = sElements[yIndex * spitch];
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
	binaryCategoryKernel<<<grid, block, smem>>>(s.elements, t.elements, t.p, s.p, t.n, t.m, oneBased);
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

template<typename T> __device__  std::set<T> distinct(DMatrix<T> dm) {

}
/*
 * attribute frequency matrix
 * thread per feature and per bank of samples
 *   shared memory has tuples
 *
 *     	f1		f2		f3		f4		...		fn
 *		tx1    	tx2		tx3		tx4		...		tx5
 *		va111
 *		cnt11
 *		val12
 *		cnt12
 *
 *	m of this matrix is the 2 * max(i: 1-n  attributeCount(feature_i) )
 *	(less mem to have two arrays, type T for atts and int for counts?
 *
 *
 *	depth is #of samples
 *	countList is m(i,distinct) * n, where n is # of attributes, and m(i,distinct) is the number of distinct values for attribute i
 *	distance is
 *
 *  each val is check by walking down the distinct attribute list (set of dist. atts rep'd as list)
 *  i.e., r from 0 until m
 *  	if r is past last distinct attribute (whose index is deepest-1),
 *  		know it's a new
 *  			so add it to end of att list, set the counter at that index to 1
 *  			add r to total 'distance' walked down this attribute's distinct value list
 *  	else if val is approx equal to attribute value at
 *  			increase count of that attr valu
 *  			add r to total 'distance' walked down this attribute's distinct value list
 *
 */

template<typename T> __device__ void checkEntry(T val, T* attList,
		int* countList, int& deepest, int& distance, int attListPitch, int countListPitch,
		int depth) {
	T* attPtr = attList;
	int* countPtr = countList;
	int r = 0;
	while( r < depth) {
		if(r == deepest) {
			*attPtr = val;
			*countPtr = 1;
			deepest++;
			distance += r;
			return;
		}
		if(util<T>::almostEquals( val, *attPtr)) {
			(*countPtr) ++;
			distance += r;
			return;
		}
		attPtr += attListPitch;
		countPtr += countListPitch;
		r++;
	}
}

template<typename T> __device__ void checkEntryStl(T val, list<T>& attList,
		list<T>& countList, int& deepest, int& distance, int attListPitch, int countListPitch,
		int depth) {
	//T* attPtr = attList;
	auto ait = attList.begin();
	auto cit = countList.begin();
	int r = 0;
	while( r < depth) {
		if(r == deepest) {
			attList.push_back(val);
			countList.push_back(1);
			deepest++;
			distance += r;
			return;
		}
/*
		if(util<T>::almostEquals( val, *attPtr)) {
			(*countPtr) ++;
			distance += r;
			return;
		}
		attPtr += attListPitch;
		countPtr += countListPitch;
*/
		r++;
	}
}

/*
 * 1 thread per col
 * d_atts and d_x are m*n <T>
 * d_counts is m * n <int>
*  d_distances is 1 * n <int>
 */
template<typename T> __global__ void attributeCountsKernel(DMatrix<T> d_atts,
		DMatrix<int> d_counts, DMatrix<int> d_depths, DMatrix<int> d_distances,
		const DMatrix<T> d_x) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if(col < d_x.n) {
		T curr;
		d_depths.elements[col] = 0;
		for(int row = 0; row < d_x.m; row++) {
			curr = d_x.elements[row * d_x.p + col];
			 //checkEntry(T val, T* freqMap, T*spillMap, int& lastDepth, int freqMapPitch, int spillMapPitch, int depth, int spillDepth );
			checkEntry(curr, d_atts.elements + col, d_counts.elements + col,
					d_depths.elements[col], d_distances.elements[col], d_atts.p,
					d_counts.p, d_x.m);
		}
	}
}

template<typename T> __inline__ __device__ void toShared(
		T* shx,
		T* shatts,
		int* shcounts,
		const DMatrix<T>& d_atts,
		const DMatrix<int>& d_counts,
		const DMatrix<T>& d_x,
		int loff,
		int2 g_idx ) {
	shatts[loff] = d_atts.elements[g_idx.x * d_atts.p + g_idx.y];
	shx[loff] = d_x.elements[g_idx.x * d_x.p + g_idx.y];
	shcounts[loff] = d_counts.elements[g_idx.x * d_counts.p + g_idx.y];
	__syncthreads();
}
/*
 * block (m, dn,1)  shmem holds enough for dn cols * m rows of
 * 		X  		m * dn * szT     or dn ( m*(2szT + szI) + 2 szI)
 * 		atts  	m * dn * szT
 * 		counts	m * dn * szInt
 * 		depths	dn * szInt
 * 		dists	dn * szInt
 */
template<typename T> __global__ void attributeCountsKernelShared(DMatrix<T> d_atts,
		DMatrix<int> d_counts, DMatrix<int> d_depths, DMatrix<int> d_distances,
		const DMatrix<T> d_x/*, int stride*/) {

	T* shatts =  SharedMemory<T>(); // local copy of attribute sets
#ifdef CuMatrix_DebugBuild
	int blockRows = blockDim.x, blockCols = blockDim.y;
	int threadRow= threadIdx.x, threadCol = threadIdx.y;
#endif
	int d_x_size = d_x.m * d_x.n;
	T* shx = shatts + d_x_size; // local copy of samples
	int* shcounts = (int*) (shx + d_x_size); // local copy of attr counts
	int* shdepths = (shcounts + d_x_size); // local copy of sizes of attr sets
	int* shdists = (shdepths + d_x.n ); // local copy of sizes of attr sets

	// global x (rows),y (cols) indices
	int2 g_idx{(int)(blockDim.x * blockIdx.x + threadIdx.x),(int) (blockDim.y * blockIdx.y + threadIdx.y)};

	// local offset
	int loff = threadIdx.x * blockDim.y + threadIdx.y;

	// shcopy to shmem
	if(g_idx.x < d_x.m && g_idx.y < d_x.n) {
		toShared(
			shx,
			shatts,
			shcounts,
			d_atts,
			d_counts,
			d_x,
			loff,
			g_idx);
	}

	int row = 0;
	// if on first row and in column range
	if(g_idx.y < d_x.n && threadIdx.x == 0 ) {
		T curr;
		shdepths[threadIdx.y] = 0;
		while(row < d_x.m) {
			curr = shx[row * blockDim.y  + threadIdx.y];
			 //checkEntry(T val, T* attList,int* countList, int& deepest, int& distance, int attListPitch, int countListPitch,	int depth)
			checkEntry(curr, shatts + threadIdx.y, shcounts + threadIdx.y,
					shdepths[threadIdx.y], shdists[threadIdx.y], blockDim.y,
					blockDim.y, d_x.m);
			row++;
		}
	}

	__syncthreads();
	if(g_idx.x < d_x.m && g_idx.y < d_x.n) {
		d_atts.elements[g_idx.x * d_atts.p + g_idx.y] = shatts[loff];
		d_counts.elements[g_idx.x * d_counts.p + g_idx.y] = shcounts[loff] ;
	}

	if(threadIdx.y == 0) {
		shdepths[threadIdx.y] = d_depths.elements[g_idx.y];
	//	shdists[threadIdx.x] = d_distances.elements[g_idx.x];
	}
}
template __global__ void attributeCountsKernelShared(DMatrix<float> d_atts,
		DMatrix<int> d_counts, DMatrix<int> d_depths, DMatrix<int> d_distances,
		const DMatrix<float> d_x);
template __global__ void attributeCountsKernelShared(DMatrix<double> d_atts,
		DMatrix<int> d_counts, DMatrix<int> d_depths, DMatrix<int> d_distances,
		const DMatrix<double> d_x/*, int stride*/);
template __global__ void attributeCountsKernelShared(DMatrix<ulong> d_atts,
		DMatrix<int> d_counts, DMatrix<int> d_depths, DMatrix<int> d_distances,
		const DMatrix<ulong> d_x/*, int stride*/);


template<typename T> __global__ void attributeFreqsKernel(DMatrix<T> d_freqs,
		DMatrix<T> d_entropies, DMatrix<int> d_counts, DMatrix<int> d_depths) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if(col < d_counts.n) {
		T* currFreq = nullptr;
		T* currEntr= nullptr;
		int colDepth = d_depths.elements[col];
		for(int row = 0; row < colDepth; row++) {
			currFreq = d_freqs.elements + row * d_freqs.p + col;
			currEntr = d_entropies.elements + row * d_entropies.p + col;
			*currFreq = (T) (1.0 * d_counts.elements[row * d_counts.p + col] / d_counts.m);
			*currEntr  = - (*currFreq) * log2( (float)*currFreq );
		}
	}
}


template<typename T> __global__ void attributeFreqsKernelShared(DMatrix<T> d_freqs,
		DMatrix<T> d_entropies, DMatrix<int> d_counts, DMatrix<int> d_depths) {

	// each block gets as big a chunk of an attribute columnn as will fit in shmem
	T* shfreqs =  SharedMemory<T>(); // local copy of frequency sets
	T* shentropies = shfreqs + blockDim.x * d_freqs.m;
	int* shcounts = (int*) (shentropies + blockDim.x * d_freqs.m);
	int* shdepths = (shcounts + blockDim.x );

	// copy globtoshmem

	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if(col < d_counts.n) {
		T* currFreq = nullptr;
		T* currEntr= nullptr;
		int colDepth = d_depths.elements[col];
		for(int row = 0; row < colDepth; row++) {
			currFreq = d_freqs.elements + row * d_freqs.p + col;
			currEntr = d_entropies.elements + row * d_entropies.p + col;
			*currFreq = (T) (1.0 * d_counts.elements[row * d_counts.p + col] / d_counts.m);
			*currEntr  = - (*currFreq) * log2( (float)*currFreq );
		}
	}
}

// dn ( m*(2szT + szI) + 2 szI)
template <typename T> int shmemCounts(int dn, int m) {
	return dn * ( m * (2 * sizeof(T) + sizeof(int)) + 2 * sizeof(int));
}

template <typename T> int dnAttFreqs( ExecCaps* pc, int m) {
	return pc->memSharedPerBlock / ( m * ( 2*  sizeof(T) + sizeof(int)) + 2 * sizeof(int));
}

template <typename T> int dnStepsAttFreqs( ExecCaps* pc, int m, int dn) {
	return DIV_UP((dn*  m * ( 2*  sizeof(T) + sizeof(int)) + 2 * sizeof(int)), pc->memSharedPerBlock) ;
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::attributeFreqsKernelL(
		DMatrix<T>& d_freqs, DMatrix<T>& d_atts, DMatrix<T>& d_entropies,
		DMatrix<int>& d_counts, DMatrix<int>& d_depths,
		DMatrix<int>& d_distances, const DMatrix<T>& d_x,
		cudaStream_t stream) const {
	int threads = 0;
	dim3 dBlocks, dThreads;
	ExecCaps* pcaps = ExecCaps::currCaps();
	DMatrix<T> t_freqs, t_atts, t_entropies;
	DMatrix<T> t_x;
	DMatrix<int> t_counts, t_depths, t_distances;
	int rowSteps = 1;
	int tileM = m;
	int tileN = dnAttFreqs<T>(pcaps, m);

	// matrix too large for even one column & support data to fit in shmem
	if( tileN == 0) {
		tileN = 1;
		rowSteps = dnStepsAttFreqs<T>( pcaps, m, tileN);
		tileM = m/rowSteps;
		if(tileM > pcaps->thrdPerBlock ) {
			tileM = pcaps->thrdPerBlock;
			rowSteps = DIV_UP(m, tileM);
		}
		flprintf( "tileM %d, rowSteps %d\n" ,tileM ,rowSteps  );
	} else {
		tileN = MIN(n, tileN);
	}

	int colSteps = n / tileN;
	flprintf( "colSteps %d\n", colSteps  );

	int shmemSize = shmemCounts<T>(tileN, tileM);
	flprintf( "shmemSize %d\n" ,shmemSize  );
	DTiler<T> dt_x(d_x, rowSteps, colSteps);
	DTiler<int> dt_counts(d_counts, rowSteps, colSteps);
	for(int cstep = 0; cstep < colSteps; cstep++) {
		dt_x.peekNextTile1D(t_freqs, d_freqs);
		dt_x.peekNextTile1D(t_atts, d_atts);
		dt_counts.peekNextTile1D(t_counts);
		dt_counts.peekNextTile1D(t_depths,d_depths);
		dt_counts.peekNextTile1D(t_distances,d_distances);
		for(int rstep = 0; rstep < rowSteps; rstep++) {
			dt_x.peekNextTile2D(t_x);
			dt_x.peekNextTile1D(t_entropies, d_entropies);
	#ifndef __CUDA_ARCH__
			outln("cstep " << cstep << "/" << colSteps<<"\nt_x: " << util<T>::pdm(t_x));
			outln("rstep " << rstep << "/" << rowSteps<<"\nt_x: " << util<T>::pdm(t_x));
			outln("d_freqs: " << util<T>::pdm(d_freqs));
			outln("t_freqs: " << util<T>::pdm(t_freqs));
			outln("d_atts: " << util<T>::pdm(d_atts));
			outln("t_atts: " << util<T>::pdm(t_atts));
			outln("d_entropies: " << util<T>::pdm(d_entropies));
			outln("t_entropies: " << util<T>::pdm(t_entropies));
			outln("t_counts: " << util<int>::pdm(t_counts));
			outln("t_depths: " << util<int>::pdm(t_depths));
			outln("t_distances: " << util<int>::pdm(t_distances));
	#endif

			dThreads.x = tileM;
			dThreads.y = tileN;
			dBlocks.x = rowSteps;
			dBlocks.y = 1;
#ifndef __CUDA_ARCH__
			outln( "blocks " << b_util::pd3(dBlocks) << " threads " << b_util::pd3(dThreads) );
			outln( "shmemSize " << shmemSize << " stream " << stream );
#endif
			attributeCountsKernelShared<<<dBlocks, dThreads, shmemSize, stream>>>(t_atts, t_counts, t_depths, t_distances, t_x);
			//attributeCountsKernel<<<dBlocks, dThreads, 0, stream>>>(t_atts, t_counts, t_depths, t_distances, t_x);
			if(stream) {
				attributeFreqsKernel<<<dBlocks, dThreads, 0, stream>>>(t_freqs, t_entropies, t_counts, t_depths );
				cherr(cudaStreamSynchronize(stream));
			} else {
				cherr(cudaDeviceSynchronize());
				attributeFreqsKernel<<<dBlocks, dThreads, 0, stream>>>(t_freqs, t_entropies, t_counts, t_depths);
				cherr(cudaDeviceSynchronize());
			}
			dt_x.advance2D(tdRows);
		}
		dt_x.advance2D(tdCols);

	}

/*
	b_util::vectorExecContext(threads, d_x.n, dBlocks, dThreads);
	outln( "blocks " << b_util::pd3(dBlocks) << " threads " << b_util::pd3(dThreads) );
	attributeCountsKernel<<<dBlocks, dThreads, 0, stream>>>(d_atts, d_counts, d_depths, d_distances, d_x);
	if(stream) {
		attributeFreqsKernel<<<dBlocks, dThreads, 0, stream>>>(d_freqs, d_entropies, d_counts, d_depths );
		cherr(cudaStreamSynchronize(stream));
	} else {
		cherr(cudaDeviceSynchronize());
		attributeFreqsKernel<<<dBlocks, dThreads, 0, stream>>>(d_freqs, d_entropies, d_counts, d_depths);
		cherr(cudaDeviceSynchronize());
	}
*/

}

/*
 * for each feature (column),
 * 		get distinct values set
 	 	  	for each distinct value,
  				get frequency (count(val) / m)
 * 	this: m * n (m samples * n features)
 * 	freqs:  [1-m] * n unique feature-values
 * 	atts: mi * n ( mi <= m, mi = number of unique values for attribute ni )
 */
template<typename T> void CuMatrix<T>::attributeFrequencies(CuMatrix<T>& freqs,
		CuMatrix<T>& atts, CuMatrix<T>& entropies, CuMatrix<int>& counts,
		CuMatrix<int>& distances, CuMatrix<int>& depths,
		cudaStream_t stream) const {
	DMatrix<T> d_X, d_atts, d_freqs, d_entropies;
	DMatrix<int> d_counts,d_depths, d_distances;
	int tileM = 0, tileN=0, tileP=0;
	int roff = 0, coff = 0 ;
	int depthRoff =0;
	dim3 threadDim(256);
/*
	outln("this->toss() " << this->toss());
	outln("freqs.toss() " << freqs.toss());
	outln("atts.toss() " << atts.toss());
	outln("entropies.toss() " << entropies.toss());
	outln("atts.toss() " << atts.toss());
*/
	tiler.tileDims(tileM, tileN, tileP, tdCols);
	assert(tileM == m);
	int tileCount = DIV_UP(m,tileM);
#ifndef __CUDA_ARCH__
	outln("tileM " << tileM <<", tileN " <<  tileN << ", tileP " << tileP << ", tileCount " << tileCount);
#endif
	for(int i = 0; i < tileCount; i++) {
#ifndef __CUDA_ARCH__
		outln( "tiler " << tiler << "\n\t roff" << roff << ",coff " << coff << ", tileM " << tileM << ", tileN " << tileN << "  i " << i);
#endif
		tiler.tileLike(d_X, roff,coff, tileM, tileN, tileP, i, tdCols, true);
		if(checkDebug(debugTiler))prlocf("means tiling");

		atts.tiler.tileLike(d_atts, roff,coff, tileM, tileN, tileP, i, tdCols, true);
		freqs.tiler.tileLike(d_freqs, roff,coff, tileM, tileN, tileP, i, tdCols, true);
		freqs.tiler.tileLike(d_freqs, roff,coff, tileM, tileN, tileP, i, tdCols, true);
		entropies.tiler.tileLike(d_entropies, roff,coff, tileM, tileN, tileP, i, tdCols, true);

		counts.tiler.tileLike(d_counts, roff,coff, tileM, tileN, tileP, i, tdCols, true);
		depths.tiler.tileLike(d_depths, depthRoff,coff, 1, tileN, tileP, i, tdCols, true);
		distances.tiler.tileLike(d_distances, depthRoff,coff, 1, tileN, tileP, i, tdCols, true);

		attributeFreqsKernelL(d_freqs, d_atts,d_entropies, d_counts, d_depths,d_distances, d_X, stream);

		atts.tiler.syncTile(d_atts, roff, coff);
		entropies.tiler.syncTile(d_entropies, roff, coff);
		freqs.tiler.syncTile(d_freqs, roff, coff);
		counts.tiler.syncTile(d_counts, roff, coff);
		depths.tiler.syncTile(d_depths, depthRoff, coff);
	}

	int maxDepth = depths.max();
	//outln("maxDepth " << maxDepth);

}
/*
template void CuMatrix<unsigned long>::attributeFrequencies(CuMatrix<unsigned long>&, CuMatrix<unsigned long>&, CuMatrix<int>&, CuMatrix<int>&, CuMatrix<int>&, CUstream_st*) const;
template void CuMatrix<double>::attributeFrequencies(CuMatrix<double>&, CuMatrix<double>&, CuMatrix<int>&, CuMatrix<int>&, CuMatrix<int>&,  CUstream_st*) const;
template void CuMatrix<float>::attributeFrequencies(CuMatrix<float>&, CuMatrix<float>&, CuMatrix<int>&, CuMatrix<int>&, CuMatrix<int>&, CUstream_st*) const;
*/

template<typename T>  CuMatrix<T> CuMatrix<T>::attributeFrequencies(cudaStream_t stream) const {
	CuMatrix<T> freqs = CuMatrix<T>::zeros(m, n, tiler.gpuMask, tdCols);
	CuMatrix<T> atts = CuMatrix<T>::zeros(m, n, tiler.gpuMask, tdCols);
	CuMatrix<T> entropies = CuMatrix<T>::zeros(m, n, tiler.gpuMask, tdCols);
	CuMatrix<int> counts = CuMatrix<int>::zeros(m, n, tiler.gpuMask, tdCols);
	CuMatrix<int> depths = CuMatrix<int>::zeros(1,n,tiler.gpuMask, tdCols);
	CuMatrix<int> distances = CuMatrix<int>::zeros(1,n,tiler.gpuMask, tdCols);

	//cherr(cudaDeviceSynchronize());
	attributeFrequencies(freqs,atts,entropies, counts,distances, depths,stream);

	return freqs;
}
// TODO add oldP and tests for p != n
template<typename T> CuMatrix<T> CuMatrix<T>::poseAsRow() {
	assert(n == p);
	oldM = m;
	m = 1;
	n *= oldM;
	p = n;
	posed = true;
	tiler.reset(*this);
	return *this;
}

template<typename T> CuMatrix<T> CuMatrix<T>::poseAsCol() {
	assert(n == p);
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
		tiler.reset(*this);
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
    ulong canvasElems = 0;
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
    	canvasElems += c->m * c->n;
    }
    if(checkDebug(debugMem))outln("concat canvasElems " << canvasElems);
    if(checkDebug(debugMem))outln("concat dcount " << dcount << ", hcount " << hcount << (hcount == 0 ? ";  only copying dmem":""));
	int n =  canvasElems;
	//CuMatrix<T> canvas(1, n, n, false, true);
	if(! canvas.tiler.hasDmemQ() || ( canvas.tiler.hasDmemQ() && !(canvas.tiler.tileSize >= canvas.size)  ) ){
		if(canvas.tiler.hasDmemQ()) {
			if(checkDebug(debugMem))outln("canvas " << canvas.toShortString() << " size ! <= " << canvas.tiler.tileSize << " freeing old d mem " << canvas.tiler.currBuffer());
			canvas.getMgr().freeTiles(canvas);
			if(canvas.elements) {
				outln("\talso freeing h mem " << canvas.elements);
				canvas.getMgr().freeHost(canvas);
			}
		}
		canvas.elements = null;
		canvas.tiler.buffers = {0,0,0,0};
		canvas.size = canvasElems * sizeof(T);
		canvas.m = 1;
		canvas.n = n;
		canvas.p = n;
		canvas.tiler.m_size = canvas.size;
		canvas.tiler.m_m = 1;
		canvas.tiler.m_n = n;
		canvas.tiler.m_p = n;
		canvas.tiler.allocTiles(canvas._tileM, canvas._tileN, canvas._tileP);
		if(checkDebug(debugCheckValid)) flprintf("%dx%dx%d -:> %p\n", 1,n,n,  canvas.tiler.buff());
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
			// cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));

			if(checkDebug(debugCopy))flprintf("dst %p + %d == %p dpitch %d src %p spitch %d widthbytes %d height %d\n",
					canvas.elements,offset , canvas.elements + offset,canvas.p *sizeof(T),
					currMat->elements, currMat->p * sizeof(T), currMat->n * sizeof(T), currMat->m);


			cherr(cudaMemcpy2DAsync(
							  canvas.elements + offset,
							  currMat->n * sizeof(T),
							  currMat->elements, currMat->p * sizeof(T),
							  currMat->n * sizeof(T),
							  currMat->m,
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
		if(checkDebug(debugCopy))outln("concat copying d2d " << currMat->toShortString() <<
				" using cudaMemcpy2DAsync\n\t\tcopying " << len << " dev Ts from " << currMat->tiler.currBuffer() << " to " << (canvas.tiler.currBuffer() + offset)) <<
				" ie " << canvas.toShortString() << " plus offset " << offset << endl ;
		MemMgr<T>::checkValid(canvas.tiler.currBuffer() + offset,"canvas.tiler.currBuffer() + offset");
		MemMgr<T>::checkValid(canvas.tiler.currBuffer() + offset + len - 1, "canvas.tiler.currBuffer() + offset + len");
		MemMgr<T>::checkValid( currMat->tiler.currBuffer(), " currMat->tiler.currBuffer()");
		if(checkDebug(debugCopy))outln("&canvas.tiler.currBuffer()[len] " << &canvas.tiler.currBuffer()[len]);
		if(checkDebug(debugCopy))outln("next_stream " << next_stream << ", stream[next_stream] " << stream[next_stream] );

		if(checkDebug(debugCopy)) outln("canvas " << canvas.toShortString());
		if(checkDebug(debugCopy)) outln("currMat " << currMat->toShortString());
		// cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
		if(canvas.tiler.tileSize >= canvas.tiler.m_size) {
			if(checkDebug(debugCopy))flprintf("dev %d canvas.tiler.currBuffer() + offset %p tpitch %d\n "\
					" currMat->tiler.currBuffer() %p srcP %d colBytes %d height %d ",
					ExecCaps::currDev(),
					  canvas.tiler.currBuffer() + offset,
					  currMat->n * sizeof(T),
					  currMat->tiler.currBuffer(),
					  currMat->_tileP * sizeof(T),
					  currMat->n * sizeof(T),
					  currMat->m
					);
			cherr(cudaMemcpy2DAsync(
								  canvas.tiler.currBuffer() + offset,
								  currMat->n * sizeof(T),
								  currMat->tiler.currBuffer(),
								  currMat->_tileP * sizeof(T),
								  currMat->n * sizeof(T),
								  currMat->m,
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
		canvas.m = 1;
		canvas.n = n;
		canvas.p = n;
		canvas.tiler.m_size = canvas.tiler.tileSize = canvasSize;
		canvas.tiler.m_m = 1;
		canvas.tiler.m_n = n;
		canvas.tiler.m_p = n;
		canvas.tiler.allocTiles(canvas._tileM,canvas._tileN, canvas._tileP);
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

			cherr(cudaMemcpy2DAsync(
							  canvas.elements + offset,
							  currMat.n * sizeof(T),
							  currMat.elements, currMat.p * sizeof(T),
							  currMat.n * sizeof(T),
							  currMat.m,
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

		if(canvas.tiler.tileSize >= canvas.tiler.m_size) {
			cherr(cudaMemcpy2DAsync(
								  canvas.tiler.currBuffer() + offset,
								  currMat.n * sizeof(T),
								  currMat.tiler.currBuffer(),
								  currMat._tileP * sizeof(T),
								  currMat.n * sizeof(T),
								  currMat.m,
								  cudaMemcpyDeviceToDevice,
								  stream[next_stream]));
			DDCopied++;
			MemDdCopied += len * sizeof(T);
		}else {
			dthrow(notImplemented());
//			assert(false); // er, n
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
	assert( tiler.tileSize >= tiler.m_size );
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
		if(checkDebug(debugCopy))outln("currDev " << ExecCaps::currDev() << ", dest " << v.toShortString() << ", src " << toShortString());

		if(checkDebug(debugCopy))outln( "dpitch Ts " << v._tileP << ", spitch Ts " << _tileP << ", cols " << cols);
		if(v.p != v._tileP) {
			T* temp;
			if(checkDebug(debugCopy)) flprintf("v.m %d, v.n %d\n", v.m, v.n);
			cherr(cudaMalloc(&temp, v.m*v.n*sizeof(T) ));
			if(checkDebug(debugCopy)) flprintf("temp %p, tiler.currBuffer %p, offset %d, cols %d rows %d\n",
					temp, tiler.currBuffer(), offset, cols, rows);
			cherr(
					cudaMemcpy(temp, tiler.currBuffer() + offset, cols * rows * sizeof(T),
							  cudaMemcpyDeviceToDevice));
			cherr( 	cudaMemcpy2D(v.tiler.currBuffer(), v._tileP * sizeof(T), temp, cols * sizeof(T), cols* sizeof(T), rows,
					 cudaMemcpyDeviceToDevice));
			cherr(cudaFree(temp));
		}
		cherr(
				cudaMemcpy(v.tiler.currBuffer(), tiler.currBuffer() + offset, cols * rows * sizeof(T),
						  cudaMemcpyDeviceToDevice));
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
	v._tileP = _tileP;

	v.lastMod = lastMod;
	v.ownsDBuffers = v.ownsHBuffers = false;
}

// crap
template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::columnMatrix(int col) const {


	if(checkDebug(debugTiler)) flprintf("tile.tileSize %ld, tiler.m_size %ld\n", tiler.tileSize , tiler.m_size);
	if(! (tiler.tileSize >= tiler.m_size)) {
		assert(false);
	}

	CuMatrix<T> column(m, 1, 1, true, true);
#ifndef __CUDA_ARCH__
	if(checkDebug(debugCopy)) outln("curr dev " << ExecCaps::currDev() << ": colmatrix(" << col << ") on " << toShortString());
	if(checkDebug(debugCopy)) outln("column " << column.toShortString());
#endif
	DMatrix<T> dsrc, dtrg;
	dsrc.elements = currBuffer() + col;
	dsrc.m = m;
	dsrc.n = 1;
	dsrc.p = _tileP;
	dtrg.elements = column.currBuffer();
	dtrg.m = m;
	dtrg.n = 1;
	dtrg.p = column._tileP;
	if(checkDebug(debugCopy))flprintf("dtrg.el %p pitch %d  dsrc.el %p pitch %d\n", dtrg.elements, dtrg.p, dsrc.elements, dsrc.p);
	CuMatrix<T>::copy(dtrg, dsrc, 0, 0);

	column.invalidateHost();
	column.syncBuffers();

	return column;
}

template<typename T> CuMatrix<T> CuMatrix<T>::rowMatrix(int row) const {
	CuMatrix<T> rowm(1,n, false, true);
	assert(tiler.tileSize >= tiler.m_size);
	if(colMajor) {
		DMatrix<T> d_X, d_row;
		tile0(d_X, lastMod == mod_host);
		rowm.tile0(d_row, false);
		rowMatrixCmL(d_row, d_X, row);
	} else {
		checkCudaError(cudaMemcpy2D(rowm.tiler.currBuffer(), rowm._tileP * sizeof(T),
				tiler.currBuffer() + row* _tileP, _tileP * sizeof(T),  n*sizeof(T), 1, cudaMemcpyDeviceToDevice));
		DDCopied++;
		MemDdCopied += (n - 1) * sizeof(T);
	}
	return rowm;
}

template<typename T> CuMatrix<T> CuMatrix<T>::dropFirst(bool copy,cudaStream_t stream ) const {
	if(lastMod == mod_host) dthrow(notSynced());

	CuMatrix<T> res(m, n - 1, false, copy);
	assert(tiler.tileSize >= tiler.m_size);
	if(copy){
		if(stream) {
			checkCudaError(
					cudaMemcpy2DAsync(res.tiler.currBuffer(), res._tileP * sizeof(T),
							tiler.currBuffer() + 1, _tileP * sizeof(T), (n-1) * sizeof(T), m, cudaMemcpyDeviceToDevice, stream));

		} else {
			checkCudaError(
				cudaMemcpy2D(res.tiler.currBuffer(), res._tileP * sizeof(T),
						tiler.currBuffer() + 1, _tileP * sizeof(T), (n-1) * sizeof(T), m, cudaMemcpyDeviceToDevice));
		}
		DDCopied++;
		MemDdCopied += (n - 1) * sizeof(T);
		res.lastMod = mod_device;
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

