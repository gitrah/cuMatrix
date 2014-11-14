/*
 * CuMatrixMath.cu
 *
 *  Created on: Mar 3, 2013
 *      Author: reid
 */
#include "CuMatrix.h"


template<typename T> CuMatrix<T> CuMatrix<T>::subMeans( const CuMatrix<T>& means) const {
	CuMatrix<T> res = zeros(m, n);
	subMeans( res, means);
	return res;
}

template<typename T> __host__ CUDART_DEVICE void CuMatrix<T>::subMeans( CuMatrix<T>& res,
		 const CuMatrix<T>& means) const {
	printf("means %p with elements %p and dims %d X %d\n",&means, means.elements, means.m , means.n);
	DMatrix<T> d_Means, d_X, d_Res;
	asDmatrix(d_X);
	means.asDmatrix(d_Means);
	res.asDmatrix(d_Res, false);
	meanSubL(d_Res, d_X, d_Means);
}


template<typename T> cudaError_t CuMatrix<T>::sqrSubMeans( CuMatrix<T>& res, const CuMatrix<T>& mus) const {
	DMatrix<T> d_Means, d_X, d_Res;
	asDmatrix(d_X);
	mus.asDmatrix(d_Means);
	res.asDmatrix(d_Res, false);
	meanSubSqrL(d_Res, d_X, d_Means);
	return cudaGetLastError();
}

template<typename T> CuMatrix<T> CuMatrix<T>::sqrSubMeans( const CuMatrix<T>& mus) const {
	CuMatrix<T> res(m, n,false, true);
	checkCudaError(sqrSubMeans(res, mus));
	return res;
}

template<typename T> CuMatrix<T> CuMatrix<T>::normalize() const {
	CuMatrix<T> mus = featureMeans(true);
	CuMatrix<T> subm = subMeans(mus);
	uint l = m * n;
#ifdef  CuMatrix_Enable_KTS
	T sqrSum = subm.reduce(sqrPlusBinaryOp<T>(), 0);
#else
	T sqrSum = subm.reduce(Functory<T,sqrPlusBinaryOp>::pinch(), 0);
#endif
	T sum = subm.sum();
	T avg = sum / l;
	T stdDev = ::sqrt(sqrSum / l - (avg * avg));
	return subm / stdDev;
}

/*
 * m*n -> m*1; each row is sum of all row features
 */
template<typename T> __host__ CUDART_DEVICE void  CuMatrix<T>::rowSum(CuMatrix<T>& rowSumM) const {
	if(rowSumM.m != m || rowSumM.n != 1) {
		setLastError(matricesOfIncompatibleShapeEx);
	}
	DMatrix<T> d_rowSum, d_x;
	asDmatrix(d_x);
	rowSumM.asDmatrix(d_rowSum, false);

	reduceRows(d_rowSum,  d_x, Functory<T, plusBinaryOp>::pinch());

	//rowSum(d_rowSum, d_x);
	rowSumM.invalidateHost();
}

template<typename T> __host__ CUDART_DEVICE CuMatrix<T> CuMatrix<T>::rowSum() const {
	CuMatrix<T> rowSumM(m, 1, false,true);
	rowSum(rowSumM);
	return rowSumM;
}


template<typename T> int CuMatrix<T>::sgn(uint row, uint col) const {
	validIndicesQ(row, col);
	int l = -1;
	uint k = 0;
	uint total = row + col;
	while (k <= total) {
		l *= -1;
		k++;
	}
	return l;
}

template <typename T> __global__ void
matrixMinorKernel(DMatrix<T> trg, const DMatrix<T> src, uint row , uint col) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.x + threadIdx.y;
    uint sidx = x + y * src.p;
    uint tidx = x + y * trg.p;
    if(x < src.n && y < src.m && x != col) {
    	for(int i = 0; i < blockDim.x; i+= blockDim.y) {
    		if( i + y < src.m && i +y  != row) {
    			if(x < col) {
    				if(y < row)
    					trg.elements[tidx + i * trg.p] = src.elements[sidx + i * src.p];
    				else
    					trg.elements[tidx + (i - 1) * trg.p]  = src.elements[sidx + i * src.p];
    			} else {
    				if(y < row)
    					trg.elements[tidx + i * trg.p - 1] = src.elements[sidx + i * src.p];
    				else
    					trg.elements[tidx + (i - 1) * trg.p - 1]  = src.elements[sidx + i * src.p];
    			}
    		}
    	}
    }
}

template<typename T> void CuMatrix<T>::matrixMinorM(CuMatrix<T>& trg, uint row, uint col) const {
	validIndicesQ(row, col);
	DMatrix<T> d_r, d_this = asDmatrix();
	trg.asDmatrix(d_r,false);
	dim3 block(DEFAULT_BLOCK_X,DEFAULT_BLOCK_Y);
	dim3 grid(DIV_UP( n, block.x), DIV_UP(m,block.x));
	matrixMinorKernel<<<grid,block>>>(d_r,d_this,row,col);
	trg.invalidateHost();
}

template<typename T> CuMatrix<T> CuMatrix<T>::matrixMinorM(uint row, uint col) const {
	validIndicesQ(row, col);
	CuMatrix<T> ret(m - 1, n - 1, false, true);
	matrixMinorM(ret, row,col);
	checkCudaError(cudaDeviceSynchronize());
	ret.syncBuffers();
	return ret;
}

template<typename T> T CuMatrix<T>::matrixMinor(uint row, uint col) const {
	return (matrixMinorM(row, col).determinant());
}

template<typename T> T CuMatrix<T>::cofactor(uint row, uint col) const {
	return (matrixMinor(row, col) * sgn(row, col));
}

template<typename T> CuMatrix<T> CuMatrix<T>::cofactorM() const {
	CuMatrix<T> ret(m, n,true, true);

	T* c = ret.elements;
	uint row = 0;
	uint col = 0;
	uint i = 0;
	while (row < m) {
		col = 0;
		while (col < n) {
			c[i] = cofactor(row, col);
			col++;
			i++;
		}
		row++;
	}
	ret.lastMod = mod_host;
	return (ret);
}
long lctr=0;
template<typename T> T CuMatrix<T>::determinant() const {
	if(!d_elements) {
		dthrow(noDeviceBuffer());
	}
	//printf("m %d, n %d \n",m,n);
	printf("%c", '0' + (lctr++ % 10));
	//outln(toShortString());

	dassert((n == m));
	switch (n) {
	case 1:
		T ret;
		checkCudaError(cudaMemcpy(&ret, d_elements, sizeof(T),cudaMemcpyDeviceToHost));
		if(checkDebug(debugCopyDh))outln("debugCopyDh " << "CuMatrix<T>::determinant1");
		DHCopied++;
		MemDhCopied +=sizeof(T);
		return ret;
	case 2:
		if(sizeof(T) == 8) {
			double4 ret;
			checkCudaError(cudaMemcpy(&ret, d_elements, 4*sizeof(T),cudaMemcpyDeviceToHost));
			if(checkDebug(debugCopyDh))outln("debugCopyDh " << "CuMatrix<T>::determinant2");
			DHCopied++;
			MemDhCopied +=4*sizeof(T);
			return ret.x * ret.w - ret.y*ret.z;
		} else if(sizeof(T) == 4) {
			float4 ret;
			checkCudaError(cudaMemcpy(&ret, d_elements, 4*sizeof(T),cudaMemcpyDeviceToHost));
			if(checkDebug(debugCopyDh))outln("debugCopyDh " << "CuMatrix<T>::determinant3");
			DHCopied++;
			MemDhCopied +=4*sizeof(T);
			return ret.x * ret.w - ret.y*ret.z;
		} else {
			dthrow(notImplemented());
		}
		break;
	default:
		// cofactor expansion along the first row or column
		T sum = 0;

		if(colMajor) {
			uint col = 0;
			while (col < n) {
				sum += elements[col * m] * cofactor(0, col);
				col++;
			}
		} else {
			uint row = 0;
			while (row < m) {
				sum += elements[row * n] * cofactor(row, 0);
				row++;
			}
		}
		return (sum);
	}
}


template<typename T> CuMatrix<T> CuMatrix<T>::inverse() const {
	dassert(n == m);
	T d = determinant();
	dassert(d != 0);
	// linearly independent
	CuMatrix<T> mT = cofactorM().syncBuffers().transpose();
	return (mT / d);
}
template<typename T> CuMatrix<T> CuMatrix<T>::inverse(T determinant) const {
	dassert(n == m);
	dassert(determinant != 0);
	// linearly independent
	CuMatrix<T> mT = cofactorM().syncBuffers().transpose();
	return (mT / determinant);
}

template<typename T> CuMatrix<T> CuMatrix<T>::subFrom(T o) const {
	subFromUnaryOp<T> subff = Functory<T,subFromUnaryOp>::pinch(o);
	return unaryOp(subff);
}

template<typename T> void CuMatrix<T>::fitGaussians(CuMatrix<T>& sqrdSigmas, CuMatrix<T>& mus) const {
	DMatrix<T> d_Sigmas, d_X, d_Mus;
	sqrdSigmas.poseAsRow();
	sqrdSigmas.asDmatrix(d_Sigmas, false);
	sqrdSigmas.unPose();
	asDmatrix(d_X);
	mus.asDmatrix(d_Mus);
	varianceAndMeanL(d_Sigmas, d_Mus, d_X );
	sqrdSigmas.invalidateHost();
	mus.invalidateHost();
}

template<typename T> void CuMatrix<T>::variance(CuMatrix<T>& sqrdSigmas, const CuMatrix<T>& mus) const {
	DMatrix<T> d_Sigmas, d_X, d_Mus;
	sqrdSigmas.poseAsRow();
	sqrdSigmas.asDmatrix(d_Sigmas, false);
	sqrdSigmas.unPose();
	asDmatrix(d_X);
	mus.asDmatrix(d_Mus);
	varianceL(d_Sigmas, d_X, d_Mus);
	sqrdSigmas.invalidateHost();
}

template<typename T> void CuMatrix<T>::toCovariance(CuMatrix<T>& covmat) const {
	if(!vectorQ()) {
		dthrow(notVector());
	}
	if(!covmat.squareQ() || covmat.n != longAxis()) {
		dthrow(badDimensions());
	}
	if(covmat.lastMod == mod_device) {
		dthrow(notSyncedHost());
	}
	for(uint diag = 0; diag < covmat.n; diag++) {
		covmat.set(diag,diag, get(diag));
	}
	covmat.invalidateHost();
}

template<typename T> CuMatrix<T> CuMatrix<T>::toCovariance() const {
	if(!vectorQ()) {
		dthrow(notVector());
	}
	CuMatrix<T> covmat = zeros(longAxis(), longAxis()).syncBuffers();
	toCovariance(covmat);
	return covmat;
}

template<typename T> void CuMatrix<T>::multivariateGaussianFeatures( CuMatrix<T>& pden, const CuMatrix<T>& sqrdSigmas, const CuMatrix<T>& mu) {
	DMatrix<T> d_sqrdSigmas, d_x, d_mu,d_pden;
	sqrdSigmas.asDmatrix(d_sqrdSigmas);
	asDmatrix(d_x);
	mu.asDmatrix(d_mu);
	pden.asDmatrix(d_pden,false);
	multivariateGaussianFeatures(d_pden,d_x, d_sqrdSigmas, d_mu);
	pden.invalidateHost();
}

template<typename T> void CuMatrix<T>::mvGaussianVectorFromFeatures( CuMatrix<T>& pvec){
	DMatrix<T> d_pvec,d_pdens;
	asDmatrix(d_pdens);
	pvec.asDmatrix(d_pvec,false);
	mvGaussianVectorFromFeatures(d_pvec,d_pdens);
	pvec.invalidateHost();
}

template<typename T> void CuMatrix<T>::multivariateGaussianVector( CuMatrix<T>& pvec, const CuMatrix<T>& sqrdSigmas, const CuMatrix<T>& mu) {
	DMatrix<T> d_sqrdSigmas, d_x, d_mu,d_pvec;
	sqrdSigmas.asDmatrix(d_sqrdSigmas);
	asDmatrix(d_x);
	mu.asDmatrix(d_mu);
	pvec.asDmatrix(d_pvec, false);
	multivariateGaussianVector(d_pvec,d_x, d_sqrdSigmas, d_mu);
	pvec.invalidateHost();
}
long detCount = 0;
template<typename T> CuMatrix<T> CuMatrix<T>::multivariateGaussianVectorM( const CuMatrix<T>& sqrdSigmas, const CuMatrix<T>& mu) {
	CuMatrix<T> covariance = sqrdSigmas.squareQ() ? sqrdSigmas : sqrdSigmas.toCovariance();
	covariance.printShortString("CuMatrix<T>::multivariateGaussianVectorM covariance:");
	CuMatrix<T> coi =  covariance.inverse();
	outln("coi " << coi.syncBuffers());
	CuMatrix<T> xnorm = subMeans(mu);
	outln("xnorm " << xnorm.syncBuffers());
	return (::pow(ONE_OVER_2PI, xnorm.n/2.0) / ::sqrt(covariance.determinant())) /
			(((xnorm * coi) % xnorm).rowSum() * 0.5).exp();
}

template<typename T> __host__ CUDART_DEVICE  CuMatrix<T> CuMatrix<T>::mapFeature(CuMatrix<T> m1, CuMatrix<T> m2, int degree) {
	CuMatrix<T> res = CuMatrix<T>::ones(m1.m, 1);
	for(int i = 1; i <= degree; i++) {
		for(int j = 0; j <= i; j++ ) {
			res = res.rightConcatenate( ( m1 ^ ((T)(i - j))).hadamardProduct(m2 ^ ((T)j)));
		}
	}
	return res;
}



#include "CuMatrixInster.cu"
