/*
 * DMatrix.h
 *
 *  Created on: Oct 13, 2013
 *
 */

#pragma once
#include "util.h"
template<typename T> struct DMatrix {
public:
	T* elements;
	int m, n, p;

	__host__ __device__ DMatrix() :
		elements(nullptr), m(0), n(0), p(0) {
	}
	__host__ __device__ DMatrix(int m, int n) :
		elements(nullptr), m(m), n(n), p(n) {
	}
	__host__ __device__ DMatrix(T* elements, int m, int n) :
		elements(elements), m(m), n(n), p(n) {
	}
	__host__ __device__ DMatrix(T* elements, int m, int n, int p) :
		elements(elements), m(m), n(n), p(p) {
	}

	__host__ __device__ bool zeroMatrixQ() const {
		return m == 0 && n == 0 && p == 0 && elements == nullptr;
	}
	__host__ __device__ void zero() {
		m = 0; n = 0; p = 0; elements = nullptr;
	}

	static DMatrix<T> ZeroMatrix;
};

template<class T>
struct SharedMemory
{
	__device__ inline operator T *() {
		extern __shared__ int __smem[];
		return (T*) __smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T*) __smem;
	}
};

template<typename T> extern __inline__ __device__ T get(const DMatrix<T>& dm, uint l) {
	return dm.elements[ (l / dm.n)* dm.p + l % dm.n ];
}

template<typename T> struct MatProd {
	typedef void (*MatProdKptr) (DMatrix<T>,const DMatrix<T>,const DMatrix<T>,int);
};

/*
  Tiles device matrices for kernels that can't operate on whole mats
  the tile is always a subset  of src's memory, not a separate buffer
 */
template<typename T> class DTiler {
	const DMatrix<T>& src;
	TileDirection tDir;
	int rowSteps;
	int currRowStep;
	int colSteps;
	int currColStep;
	int tileM,tileN;

public:
	__host__ __device__ DTiler(const DMatrix<T>& src, TileDirection tDir, int steps) :
			src(src), tDir(tDir),
			rowSteps(0), currRowStep(0) ,
			colSteps(0), currColStep(0), tileM(0), tileN(0) {
		if(tDir == tdCols) {
			colSteps = steps;
			tileN = DIV_UP(src.n,colSteps);
			tileM = src.m;
			rowSteps = 1;
		} else if(tDir == tdRows) {
			colSteps = 1;
			rowSteps = steps;
			tileM = DIV_UP(src.m,rowSteps);
			tileN = src.n;
		} else {
			assert(false);
		}
	}

	__host__ __device__ DTiler(const DMatrix<T>& src, int rowSteps, int colSteps) :
			src(src), tDir(tdBoth),
			rowSteps(rowSteps), currRowStep(0),
			colSteps(colSteps), currColStep(0), tileM(DIV_UP(src.m,rowSteps)), tileN(DIV_UP(src.n,colSteps))  {}

	__host__ __device__ int offset1D(int currStep, int steps) {
		if(tDir == tdCols) {
			return tileN * currStep;
		} else if(tDir == tdRows) {
			return tileM * currStep + src.p;
		}
		//outln("tDir " << b_util::tileDir(tDir) );
		assert(false);
		return -1;
	}

	__host__ __device__ int offset1D(int currStep, int steps,TileDirection td) {
		assert(tDir == tdBoth);
		if(td == tdCols) {
			return tileN * currStep;
		} else if(td == tdRows) {
			return tileM * currStep + src.p;
		}
		//outln("td " << b_util::tileDir(td) );
		assert(false);
		return -1;
	}

	__host__ __device__ int offset2D( int tileM, int tileN, int currRstep,int currCstep) {
		return tileM * currRstep * src.p + tileN * currCstep;
	}

	__host__ __device__ void clip1D(DMatrix<T>& trg, int currStep, int steps) {
		if(tDir == tdCols) {
			trg.n = src.n - trg.n * currStep;
		} else if(tDir == tdRows) {
			trg.m = src.m - trg.m  * currStep ;
		}
	}

	__host__ __device__ void clip2D(DMatrix<T>& trg) {
		if(currColStep == colSteps -1 && trg.n == DIV_UP(src.n, colSteps))
			trg.n = src.n - trg.n * currColStep;
		if(currRowStep == rowSteps -1 && trg.m == DIV_UP(src.m, rowSteps))
			trg.m = src.m - trg.m * currRowStep;
	}

	__host__ __device__ void advance1D() {
		int& steps = tDir == tdRows ? rowSteps :  colSteps;
		int& currStep = tDir == tdRows ? currRowStep :  currColStep;
		if(currStep < steps)
			currStep++;
	}

	__host__ __device__ void advance2D(TileDirection td) {
		if( (td & tdRows) && currRowStep < rowSteps)
				currRowStep++;
		if( (td & tdCols) && currColStep < colSteps)
				currColStep++;
	}

	__host__ __device__ void peekNextTile1D(DMatrix<T>& trg) {
		trg.zero();
		if(tDir == tdBoth) {
			if(currRowStep < rowSteps && currColStep < colSteps) {
				trg.m = tileM;
				trg.n = tileN;
				trg.p = src.p;
				trg.elements = src.elements + offset2D(tileM, tileN, currRowStep,  currColStep);
				clip2D(trg);
			}
		}else {
			int& steps = tDir == tdRows ? rowSteps :  colSteps;
			int& currStep = tDir == tdRows ? currRowStep :  currColStep;
			if(currStep < steps) {
				trg.m = tileM;
				trg.n = tileN;
				trg.p = src.p;
				trg.elements = src.elements + offset1D(currStep, steps);
				if(checkDebug(debugTiler)) flprintf("clipTile now %d.%d.%d\n", trg.m, trg.n, trg.p);
				if(! (currStep < steps -1)) {
					clip1D(trg,currStep, steps);
				}
			}
		}
	}

	__host__ __device__ void peekNextTile2D(DMatrix<T>& trg) {
		int coff = 0, roff = 0;
		trg.zero();
		if(currRowStep < rowSteps && currColStep < colSteps) {
			trg.m = tileM;
			trg.n = tileN;
			trg.p = src.p;
			trg.elements = src.elements + offset2D(trg.m, trg.n, currRowStep, currColStep);
			if(! (currRowStep < rowSteps -1) || !(currColStep<  colSteps -1)) {
				clip2D(trg);
			}
		}
	}

	__host__ __device__ void nextTile1D(DMatrix<T>& trg) {
		peekNextTile1D(trg);
		advance1D();
	}

	__host__ __device__ void peekNextTile1D(DMatrix<T>& trg, const DMatrix<T>& oSrc) {
		trg.zero();
		if(tDir == tdBoth) {
			if(currRowStep < rowSteps && currColStep < colSteps) {
				trg.m = tileM;
				trg.n = tileN;
				trg.p = src.p;
				trg.elements = oSrc.elements + offset2D(tileM, tileN, currRowStep,  currColStep);
				clip2D(trg);
			}
		}else {
			int& steps = tDir == tdRows ? rowSteps :  colSteps;
			int& currStep = tDir == tdRows ? currRowStep :  currColStep;
			if(currStep < steps) {
				trg.m = tileM;
				trg.n = tileN;
				trg.p = src.p;
				trg.elements = oSrc.elements + offset1D(currStep, steps);
				if(! (currStep < steps -1)) {
					clip1D(trg,currStep, steps);
				}
			}
		}
	}

	__host__ __device__ void nextTile1D(DMatrix<T>& trg, const DMatrix<T>& oSrc) {
		peekNextTile1D(trg,oSrc);
		advance1D();
	}
};


