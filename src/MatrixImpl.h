#include "CuMatrix.h"
#include "MatrixExceptions.h"
#include "Matrix.h"
#include "util.h"
#include "debug.h"
#include "caps.h"
#include <cuda_runtime.h>
#include <set>
#include <vector>
#include <cstdarg>

//#define MATTMPLT

//#ifdef MATTMPLT
//#endif


/////////////////////////////////////////////////////////////////////////
//
// Matrix constructors and destructors
//
/////////////////////////////////////////////////////////////////////////

template<typename T> MemMgr<T>& Matrix<T>::getMgr() {
	return *mgr;
}

template<typename T> void Matrix<T>::initMembers() {
	// ah, templates; do they not make the code beautiful?
	// yes, they do not
	CuMatrix<T>::elements = null;
	CuMatrix<T>::d_elements = null;
	CuMatrix<T>::m = 0;
	CuMatrix<T>::n = 0;
	CuMatrix<T>::p = 0;
	CuMatrix<T>::oldM = 0;
	CuMatrix<T>::oldN = 0;
	CuMatrix<T>::posed = false;
	CuMatrix<T>::colMajor = false;
	CuMatrix<T>::lastMod = mod_neither;
	CuMatrix<T>::ownsBuffers = true;
	txp = null;
	ownsTxp = false;
	Constructed++;
}
template<typename T> void Matrix<T>::freeTxp() {
	if(txp ) {
		if(ownsTxp) {
			//if(debugTxp)outln(this->toShortString() <<" deleting txp " << txp->toShortString());
			delete txp;
		} else {
			//if(debugTxp)outln(this->toShortString() <<" reseting txp " << txp->toShortString());
		}
		txp = null;
	}
}

// ctors

template<typename T> Matrix<T>::Matrix() {
	initMembers();
	if (debugCons || debugLife) {
		outln( "default constructor Matrix() -> " << CuMatrix<T>::toShortString());
		b_util::dumpStack();
	}
}

template<typename T> Matrix<T>::Matrix(const Matrix<T>& o) {
	CuMatrix<T>::elements = o.elements;
	CuMatrix<T>::d_elements = o.d_elements;
	CuMatrix<T>::m =  o.m;
	CuMatrix<T>::n =  o.n;
	CuMatrix<T>::p =  o.p;
	CuMatrix<T>::size = o.size;
	CuMatrix<T>::posed = o.posed;
	CuMatrix<T>::colMajor = o.colMajor;
	CuMatrix<T>::lastMod = o.lastMod;
	CuMatrix<T>::ownsBuffers = o.ownsBuffers;
	if (CuMatrix<T>::elements)
		Matrix<T>::getMgr().addHost(*this);
	if (CuMatrix<T>::d_elements)
		Matrix<T>::getMgr().addDevice(*this);
	txp = o.txp;
	ownsTxp = o.ownsTxp;
	Constructed++;

	if (debugCons || debugLife) {
		outln("default copy cons Matrix(src <<" << o.toShortString() << ">>) -> trg " << this->toShortString());
		b_util::dumpStack();
	}

}

template<typename T> Matrix<T>::Matrix(const Matrix<T>& o, bool alloc) {
	CuMatrix<T>::m =  o.m;
	CuMatrix<T>::n =  o.n;
	CuMatrix<T>::p =  o.p;
	CuMatrix<T>::size = o.size;
	CuMatrix<T>::posed = o.posed;
	CuMatrix<T>::colMajor = o.colMajor;
	CuMatrix<T>::lastMod = o.lastMod;
	CuMatrix<T>::ownsBuffers = o.ownsBuffers;

	if(alloc) {
		if(o.elements) {
			getMgr().allocHost(*this);
		}
		if(o.d_elements) {
			getMgr().allocDevice(*this);
		}
		o.copy(*this,0,0);
	} else {
		CuMatrix<T>::elements = o.elements;
		CuMatrix<T>::d_elements = o.d_elements;
		if (CuMatrix<T>::elements)
			Matrix<T>::getMgr().addHost(*this);
		if (CuMatrix<T>::d_elements)
			Matrix<T>::getMgr().addDevice(*this);
	}
	txp = o.txp;
	ownsTxp = false;
	Constructed++;

	if (debugCons || debugLife) {
		outln("copy cons Matrix(src <<" << o.toShortString() << ">>, " << tOrF(alloc)<< ") -> trg " << this->toShortString());
		b_util::dumpStack();
	}
}

template<typename T> Matrix<T>::Matrix(const Matrix<T>& o, uint rows, uint cols, uint pitch, uint offset) {
	CuMatrix<T>::elements = o.elements + offset;
	CuMatrix<T>::d_elements = o.d_elements + offset;
	CuMatrix<T>::m =  rows;
	CuMatrix<T>::n =  cols;
	CuMatrix<T>::p =  pitch;
	CuMatrix<T>::size = o.size;
	CuMatrix<T>::posed = o.posed;
	CuMatrix<T>::colMajor = o.colMajor;
	CuMatrix<T>::lastMod = o.lastMod;
	//txp = o.txp;
	//ownsTxp = false;
	Constructed++;

	if (debugCons || debugLife) {
		outln("cons (as subm) Matrix(<<" << o.toShortString() << ">>, " << rows << ", " << cols << ", " << pitch << ", " << offset << ") -> " << this->toShortString());
		b_util::dumpStack();
	}
	Matrix<T>::getMgr().addSubmatrix(*this,o);
}

template<typename T> Matrix<T>::Matrix( T* h_data, uint m, uint n, bool allocateD )  {
	initMembers();
	CuMatrix<T>::elements = h_data;
	this->m = m;
	this->n = n;
	CuMatrix<T>::p = n;
	CuMatrix<T>::size = m * n * sizeof(T);
	getMgr().addHost(*this);
	if(allocateD) {
		getMgr().allocDevice(*this);
		this->lastMod = mod_host;
		syncBuffers();
	}
	if (debugCons || debugLife) {
		outln("cons Matrix(" << h_data << ", " <<  m << ", " << n << ", " << tOrF(allocateD) << ") -> " << this->toShortString());
		b_util::dumpStack();
	}
}

template<typename T> Matrix<T>::Matrix( T* h_data, uint m, uint n, uint p, bool allocateD) {
	initMembers();
	CuMatrix<T>::elements = h_data;
	this->m = m;
	this->n = n;
	this->p = p;
	CuMatrix<T>::size = m * n * sizeof(T);
	getMgr().addHost(*this);
	if(allocateD) {
		getMgr().allocDevice(*this);
		this->lastMod = mod_host;
		syncBuffers();
	}
	if (debugCons || debugLife) {
		outln("cons Matrix(" << h_data << ", " <<  m << ", " << n << ", " << ", " << p << ", " << tOrF(allocateD) << ") -> " << this->toShortString());
	}
}

template<typename T> Matrix<T>::Matrix(uint m, uint n, uint p, bool allocate, bool allocateD) {
	initMembers();
	this->m = m;
	this->n = n;
	this->p = p;
	CuMatrix<T>::size = m * n * sizeof(T);
	if (this->size) {
		if (allocate) {
			getMgr().allocHost(*this);
		}
		if(allocateD) {
			getMgr().allocDevice(*this);
		}
	}
	if (debugCons || debugLife) {
		outln("cons Matrix(" << m << ", " << n << ", " << ", " << p << ", " << tOrF(allocate) << ", " << tOrF(allocateD) << ") -> " << this->toShortString());
		b_util::dumpStack();
	}
}

template<typename T> Matrix<T>::Matrix(uint m, uint n, bool allocate, bool allocateD) {
	initMembers();
	this->m = m;
	this->n = n;
	this->p = n;
	this->size = m * n * sizeof(T);
	if (this->size) {
		if(allocate) {
			getMgr().allocHost(*this);
		}
		if(allocateD) {
			getMgr().allocDevice(*this);
		}
	}
	if (debugCons || debugLife) {
		outln("cons Matrix(" << m << ", " << n << ", " << tOrF(allocate) << ", " << tOrF(allocateD) << ") -> " << this->toShortString());
		b_util::dumpStack();
	}
}

template<typename T> Matrix<T>::Matrix(const DMatrix<T>& o, bool allocate_h, bool copy)  {
	initMembers();
	this->m = o.m;
	this->n = o.n;
	this->p = o.n;
	this->d_elements=o.elements;
	this->size = this->m * this->n * sizeof(T);
	if (allocate_h) {
		getMgr().allocHost(*this);
	}
	getMgr().addDevice(*this);
	if (copy) {
		if (debugCopy)
			outln(
					this << " h-h copying " << CuMatrix<T>::size << " from " << o.elements << " to " << CuMatrix<T>::elements);
		checkCudaError(
				cudaMemcpy(CuMatrix<T>::elements, o.elements, CuMatrix<T>::size, cudaMemcpyHostToHost));
		HHCopied++;
		MemHhCopied += CuMatrix<T>::size;
	}
	if (debugCons || debugLife) {
		outln("cons Matrix(" << util<T>::pdm(o) << ", " << tOrF(allocate_h) << ", " << tOrF(copy) << ") -> " << this->toShortString());
		b_util::dumpStack();
	}
}

template<typename T> Matrix<T>::~Matrix() {
	stringstream disposition;
	int refcount;
	disposition << "~Matrix() " << this->toShortString();
	T* e_ptr = CuMatrix<T>::elements;
	if (e_ptr != null )  {
		if(CuMatrix<T>::ownsBuffers) {
			refcount = getMgr().freeHost(*this);
			if(refcount) {
				disposition << "\n\thost " << this->elements << " still has " << refcount << " refs";
			} else {
				disposition << "\n\thost " << e_ptr << " freed";
			}

		} else {
			disposition << "\nwas submatrix, no disposition of these host/dev buffers";
		}
		CuMatrix<T>::elements = null;
	}
	e_ptr = CuMatrix<T>::d_elements;
	if (e_ptr != null) {
		if(CuMatrix<T>::ownsBuffers) {
			refcount = getMgr().freeDevice(*this);
			if(refcount) {
				disposition << "\n\tdevice " << this->d_elements << " still has " << refcount << " refs";
			} else {
				disposition << "\n\tdevice " << e_ptr << " freed";
			}
		}
		CuMatrix<T>::d_elements = null;
	}
	freeTxp();
	Destructed++;
	if (debugCons || debugLife) {
		outln(disposition.str());
		if(debugLife) {
			b_util::dumpStack();
		}
	}
}


/////////////////////////////////////////////////////////////////////////
//
// matrix accessor/mutator/migrator
//
/////////////////////////////////////////////////////////////////////////

template<typename T> void Matrix<T>::set(uint r, uint c, T val) {
	if (r >= CuMatrix<T>::m || c >= CuMatrix<T>::n)
		dthrow(outOfBounds());
	uint idx = this->colMajor ? c * this->p + r : r*this->p + c;
	this->elements[idx] = val;
	invalidateDevice();
}

template<typename T> void Matrix<T>::set(uint l, T val) {
	if (l >= CuMatrix<T>::size / sizeof(T))
		dthrow(outOfBounds());
	if(this->n == this->p) {
		CuMatrix<T>::elements[l] = val;
	} else {
		uint div = l /this->n;
		uint idx = div * this->p;
		idx += l - div * this->n;
		if(debugMem)outln("offset l " << l << " -> " << idx);
		CuMatrix<T>::elements[idx ] = val;
	}
	this->lastMod = mod_host;
}

template<typename T> T Matrix<T>::get(uint r, uint c) const {
	if (r >= CuMatrix<T>::m || c >= CuMatrix<T>::n)
		dthrow(outOfBounds());
	ulong idx = this->colMajor ? c * this->p + r : r*this->p + c;
	return this->elements[idx];
}

template<typename T> T Matrix<T>::get(uint l) const {
	if (l >= CuMatrix<T>::size / sizeof(T))
		dthrow(outOfBounds());
	if(this->n == this->p) {
		return (CuMatrix<T>::elements[l]);
	}
	uint div = l /this->n;
	uint idx = div * this->p;
	idx += l - div * this->n;
	if(debugMem)outln("offset l " << l << " -> " << idx);
	return CuMatrix<T>::elements[idx ];
}

template<typename T> Matrix<T> Matrix<T>::copy(bool copyDeviceMem) const {
	Matrix<T> ret(CuMatrix<T>::m, CuMatrix<T>::n, CuMatrix<T>::elements, CuMatrix<T>::d_elements);
	if (CuMatrix<T>::elements) {
		checkCudaError(
				cudaMemcpy(ret.elements, CuMatrix<T>::elements, CuMatrix<T>::size, cudaMemcpyHostToHost));
		HHCopied++;
		MemHhCopied += CuMatrix<T>::size;
	}
	if (CuMatrix<T>::d_elements && copyDeviceMem) {
		ret.asDmatrix();
		checkCudaError(
				cudaMemcpy(ret.d_elements, CuMatrix<T>::d_elements, CuMatrix<T>::size, cudaMemcpyDeviceToDevice));
		DDCopied++;
		MemDdCopied += CuMatrix<T>::size;
	}
	ret.lastMod  = CuMatrix<T>::lastMod;
	if(debugSync && ret.lastMod == mod_host) {
		outln("Matrix (" << this << " )::copy(" << tOrF(copyDeviceMem) << ") -> " << &ret << " set lastMod of host");
	}
	ret.posed = CuMatrix<T>::posed;
	ret.colMajor = CuMatrix<T>::colMajor;
	ret.oldM = CuMatrix<T>::oldM;
	ret.oldN = CuMatrix<T>::oldN;
	ret.p = CuMatrix<T>::p;
	ret.size = CuMatrix<T>::size;
	if(txp && ownsTxp) {
		outln("copy() recreating txp");
		ret.txp = new Matrix<T>(CuMatrix<T>::n,CuMatrix<T>::m, true);
		ret.ownsTxp = true;
		if (txp->elements) {
			outln("copy() copying txp->elements");
			checkCudaError(
					cudaMemcpy(ret.txp->elements, txp->elements, CuMatrix<T>::size, cudaMemcpyHostToHost));
			HHCopied++;
			MemHhCopied += CuMatrix<T>::size;
		}
		if (txp->d_elements && copyDeviceMem) {
			outln("copy() copying txp->d_elements");
			ret.txp->asDmatrix();
			checkCudaError(
					cudaMemcpy(ret.txp->d_elements, txp->d_elements, CuMatrix<T>::size, cudaMemcpyDeviceToDevice));
			DDCopied++;
			MemDdCopied += CuMatrix<T>::size;
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

template<typename T> void Matrix<T>::invalidateHost() {
	if(debugSync && CuMatrix<T>::lastMod != mod_device) {
		outln("matrix " << this << " invalHost clr " << b_util::callerN(3));
	}
	CuMatrix<T>::lastMod = mod_device;
	freeTxp();
}

template<typename T> void Matrix<T>::invalidateDevice() {
	if(debugSync && CuMatrix<T>::lastMod != mod_host) {
		outln("matrix " << this << " invalidateDevice caller " << b_util::callerN(3));
	}
	CuMatrix<T>::lastMod = mod_host;
	freeTxp();
}

template<typename T> __host__ cudaError_t Matrix<T>::asDmatrix(DMatrix<T>& md,
		bool copy, bool force) const {
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());

	if(Matrix<T>::lastMod == mod_device) {
		if(debugSync || debugMem) outln(CuMatrix<T>::toShortString() << " asDmatrix: lastMod == device; not copying host-dev");
		copy = false;
	}
	md.m = this->m;
	md.n = this->n;
	md.p = this->p;
	bool needForce = false;

	if (CuMatrix<T>::d_elements != null) {
		needForce = true;
		if(debugMem) MemMgr<T>::checkValid(CuMatrix<T>::d_elements);
		if (md.elements == null) {
			md.elements = CuMatrix<T>::d_elements;
		}
	} else {
		dthrow(noDeviceBuffer());
	}

	if (CuMatrix<T>::lastMod == mod_host || (copy && (!needForce || (needForce && force)))) {
		if (debugCopy || (CuMatrix<T>::lastMod == mod_host && debugSync)) {
			outln(" asDmatrix " << this << " h-d copying " << CuMatrix<T>::m << "*" << CuMatrix<T>::n << "-" << CuMatrix<T>::size << " from " << CuMatrix<T>::elements << " to " << md.elements);
			outln("CuMatrix<T>::lastMod == mod_host " << tOrF(CuMatrix<T>::lastMod == mod_host));
			outln("callerN " <<  b_util::callerN(3) );

		}
		checkCudaError(
				cudaMemcpy(CuMatrix<T>::d_elements, CuMatrix<T>::elements, CuMatrix<T>::size, cudaMemcpyHostToDevice));
		HDCopied++;
		MemHdCopied += CuMatrix<T>::size;
	}
	if (debugMem)
		outln("asDmatrix(DMatrix<T>&,bool,bool) exit, this " << this->toShortString());
	return cudaSuccess;
}

template<typename T> __host__ DMatrix<T> Matrix<T>::asDmatrix(  bool copy) const {
	DMatrix<T> ret;
	asDmatrix(ret,copy,false);
	return ret;
}
// lazy, doesn't actually copy to host
template<typename T> __host__ cudaError_t Matrix<T>::fromDevice(
		const DMatrix<T>& md, bool copy) {
	CuMatrix<T>::n = CuMatrix<T>::p = md.n;
	CuMatrix<T>::m = md.m;
	CuMatrix<T>::size = CuMatrix<T>::m * CuMatrix<T>::n * sizeof(T);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
	if (copy) {
		if (debugCopy)
			outln(
					"setFrom " << this << " d-h copying " << CuMatrix<T>::m << "*" << CuMatrix<T>::n << "-sz" << CuMatrix<T>::size << " from " << md.elements << " to " << CuMatrix<T>::elements);
		checkCudaError(
				cudaMemcpy( CuMatrix<T>::elements, md.elements, this->size, cudaMemcpyDeviceToHost));
		DHCopied++;
		MemDhCopied += this->size;
		CuMatrix<T>::lastMod = mod_synced;
		freeTxp();
	} else {
		invalidateHost();
	}
	CuMatrix<T>::d_elements = md.elements;
	getMgr().addDevice(*this);
	return cudaSuccess;
}

template<typename T> __host__ Matrix<T>& Matrix<T>::syncBuffers(bool copy ) {
	if(debugCopy)outln("syncBuffers on " << this->toShortString() << " [caller " << b_util::caller() << "]");
	//dassert(CuMatrix<T>::d_elements && CuMatrix<T>::elements);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
	cudaError_t err = cudaSuccess;
	if(this->lastMod != mod_synced) {
		if (this->lastMod == mod_device) {
			if(!CuMatrix<T>::elements) {
				getMgr().allocHost(*this);
			}
			err = cudaMemcpy(CuMatrix<T>::elements, CuMatrix<T>::d_elements, CuMatrix<T>::size, cudaMemcpyDeviceToHost);
			checkCudaError(err);
			DHCopied++;
			MemDhCopied += CuMatrix<T>::size;
			if (debugCopy || debugSync)
				outln( "syncBuffers() mat " << this << " copied " << CuMatrix<T>::size << " from d " << CuMatrix<T>::d_elements << " to  h " << CuMatrix<T>::elements);
		} else {// if (this->lastMod == mod_host) {
			if(CuMatrix<T>::d_elements == null) {
				if(debugSync) outln("creating device buffer");
				getMgr().allocDevice(*this);
			}
			if(!CuMatrix<T>::elements) {
				if(CuMatrix<T>::lastMod == mod_neither) {
					if(debugSync) outln("creating host buffer");
					getMgr().allocHost(*this);
				} else {
					dthrow(noHostBuffer());
				}
			}
			if(this->lastMod != mod_neither && copy) {
				err = cudaMemcpy(CuMatrix<T>::d_elements, CuMatrix<T>::elements, CuMatrix<T>::size, cudaMemcpyHostToDevice);
				HDCopied++;
				MemHdCopied += CuMatrix<T>::size;
				checkCudaError(err);
				if (debugCopy|| debugSync)
					outln("syncBuffers() mat " << this << " copied h " << CuMatrix<T>::elements << " to  d " << CuMatrix<T>::d_elements);
			}
		}
		this->lastMod = mod_synced;
	}
	return *this;
}

template<typename T> __host__ Matrix<T> Matrix<T>::syncHost() {
	invalidateHost();
	return syncBuffers();
}

template<typename T> __host__ Matrix<T> Matrix<T>::syncDevice() {
	invalidateDevice();
	return syncBuffers();
}

template<typename T> __host__ cudaError_t Matrix<T>::toStream( std::ofstream& ofs ) const {
	ofs.write((char *)this, sizeof(*this));
	if(CuMatrix<T>::elements) {
		uint l = CuMatrix<T>::m * CuMatrix<T>::n;
		ofs.write((char*)CuMatrix<T>::elements, l*sizeof(T));
		outln("wrote " << l << " elements");
	}
	return cudaSuccess;
}

template<typename T> Matrix<T> Matrix<T>::toBinaryCategoryMatrixCPU() const {
	outln("toBinCat " << CuMatrix<T>::toShortString());
	const uint len = CuMatrix<T>::m * CuMatrix<T>::n;
	std::set<T> s(CuMatrix<T>::elements, CuMatrix<T>::elements + len);
	uint newCols = s.size();
	Matrix<T> bmat = Matrix<T>::zeros(CuMatrix<T>::m, newCols);
	uint i = 0;
	uint off = 0;
	while (i < CuMatrix<T>::m) {
		off = get(i, 0) - 1;
		bmat.elements[i * newCols + off] = 1;
		i += 1;
	}
	bmat.lastMod=mod_host;
	return bmat;
}

template<typename T> Matrix<T> Matrix<T>::toBinaryCategoryMatrix() const {
	const uint len = CuMatrix<T>::m * CuMatrix<T>::n;

	std::set<T> s(CuMatrix<T>::elements, CuMatrix<T>::elements + len); 	// TODO make a kernel for this (self-reduction until steady state?)
	bool oneBased = (s.find(0) == s.end());
	//outln("oneBased " << tOrF(oneBased));
	uint newCols = s.size();
	Matrix<T> res(CuMatrix<T>::m, newCols,false,true);
	DMatrix<T> d_res = res.asDmatrix(false);
	DMatrix<T> d_src = asDmatrix();
	//outln("binCat found " << newCols << " distinct values");
	CuMatrix<T>::binaryCategoryKernelL(d_res, d_src, oneBased);
	res.lastMod=mod_device;
	return res;
}

template<typename T> Matrix<T> Matrix<T>::transposeCpu() const {
	Matrix<T> res(CuMatrix<T>::n, CuMatrix<T>::m, true);
	uint l = CuMatrix<T>::m * CuMatrix<T>::n - 1;
	T t = CuMatrix<T>::elements[0];
	res.elements[0] = t;
	res.elements[l] = CuMatrix<T>::elements[l];
	uint i = 1;
	while (i < l) {
		res.elements[i * CuMatrix<T>::m % l] = CuMatrix<T>::elements[i];
		i++;
	}
	res.lastMod=mod_host;
	return res;
}

/*
 *  todo implement randSequence as a kernel on column or row matrix
 * 		whose re-arrangment when sorted (idx0 -> idx0sorted ...) is applied to the original sequence
 */
template<typename T> void Matrix<T>::shuffle(Matrix<T>& trg, Matrix<T>& leftovers, T fraction, vector<uint>& vIndices ) const {
	if( !(fraction >= 0. && fraction <= 1.)) {
		dthrow(outOfBounds());
	}
	if(this->d_elements == null){
		dthrow(noDeviceBuffer());
	}
	if(this->lastMod == mod_host) {
		dthrow(notSyncedDevice());
	}

	uint rows = round(this->m * fraction);
	trg.m = rows;
	trg.n = trg.p = this->n;
	trg.size = trg.m * trg.p * sizeof(T);
	trg.getMgr().allocDevice(trg);
	if(rows == this->m) {
		leftovers = Matrix<T>::ZeroMatrix;
	} else {
		leftovers.m = this->m - rows;
		leftovers.n = leftovers.p = this->n;
		leftovers.size = leftovers.m *  leftovers.p * sizeof(T);
		leftovers.getMgr().allocDevice(leftovers);
	}

	// re-use passed-in index buffer, to keep multple sample matrices in sync
	if(vIndices.size() == 0 ) {
		b_util::randSequence(vIndices, this->m, 0);
	} else if (vIndices.size() != this->m) {
		outln("shuffle passed a row index vector, but it was the wrong size (" << vIndices.size() << " <> " << this-> m << ")");
		dthrow(badDimensions());
	}

	if(debugFill)outln("vIndices\n" << b_util::pvec(vIndices));
	uint* indices, *d_indices;
	uint indexSize = this->m * sizeof(uint);
	checkCudaError( cudaHostAlloc( (void**)&indices, indexSize, 0));
	b_util::toArray(vIndices, indices, 0, rows);
	checkCudaError( cudaMalloc( (void**)&d_indices, indexSize));
	checkCudaError(cudaMemcpy(d_indices, indices,indexSize, cudaMemcpyHostToDevice));

	DMatrix<T> s, t, l;
	this->asDmatrix(s,false,false);
	trg.asDmatrix(t,false,false);
	CuMatrix<T>::shuffleCopyRows(t,s, d_indices);

	trg.lastMod = mod_device;

	checkCudaError(cudaDeviceSynchronize());

	if( !leftovers.zeroDimsQ()) {
		indexSize = leftovers.m * sizeof(uint);
		if(leftovers.m > rows) {
			// need a bigger index buffer
			checkCudaError( cudaFreeHost(indices));
			checkCudaError( cudaHostAlloc( (void**)&indices, indexSize, 0));
			checkCudaError( cudaFree( d_indices));
			checkCudaError( cudaMalloc( (void**)&d_indices, indexSize));
		}
		b_util::toArray(vIndices, indices, rows, leftovers.m);
		checkCudaError(cudaMemcpy(d_indices, indices, indexSize, cudaMemcpyHostToDevice));
		leftovers.asDmatrix(l,false,false);
		CuMatrix<T>::shuffleCopyRows(l,s, d_indices);
		leftovers.lastMod = mod_device;
	}

	checkCudaError(cudaDeviceSynchronize());
	checkCudaError(cudaFreeHost(indices));
	checkCudaError( cudaFree( d_indices));

}

// TODO add oldP and tests for p != n
template<typename T> Matrix<T> Matrix<T>::poseAsRow() {
	CuMatrix<T>::oldM = CuMatrix<T>::m;
	CuMatrix<T>::m = 1;
	CuMatrix<T>::n *= CuMatrix<T>::oldM;
	CuMatrix<T>::p = CuMatrix<T>::n;
	CuMatrix<T>::posed = true;
	return *this;
}

template<typename T> Matrix<T> Matrix<T>::poseAsCol() {
	CuMatrix<T>::oldN = CuMatrix<T>::n;
	CuMatrix<T>::n = 1;
	CuMatrix<T>::p = CuMatrix<T>::n;
	CuMatrix<T>::m *= CuMatrix<T>::oldN;
	CuMatrix<T>::posed = true;
	return *this;
}

template<typename T> Matrix<T> Matrix<T>::unPose() {
	if (CuMatrix<T>::posed && CuMatrix<T>::oldM != 0) {
		CuMatrix<T>::m = CuMatrix<T>::oldM;
		CuMatrix<T>::n /= CuMatrix<T>::oldM;
		CuMatrix<T>::p = CuMatrix<T>::n;
		CuMatrix<T>::oldM = 0;
	} else if (CuMatrix<T>::posed && CuMatrix<T>::oldN != 0) {
		CuMatrix<T>::n = CuMatrix<T>::oldN;
		CuMatrix<T>::p = CuMatrix<T>::n;
		CuMatrix<T>::m /= CuMatrix<T>::oldN;
		CuMatrix<T>::oldN = 0;
	}
	CuMatrix<T>::posed = false;
	return *this;
}

template<typename T> void Matrix<T>::reshape(Matrix<T>& target, uint rows, uint cols, ulong offsetInTs) {
	if(this->d_elements == null ) dthrow(noDeviceBuffer()) else if(debugMem) outln("reshape have nz this->d_elements");
	if(target.d_elements == null ) dthrow(noDeviceBuffer())  else if(debugMem) outln("reshape have nz ret.d_elements");
	uint l = rows * cols;
	if(contiguousQ()) {
		if(gpuReadyQ()) {
			checkCudaError(
				cudaMemcpy(target.d_elements, CuMatrix<T>::d_elements + offsetInTs, l * sizeof(T), cudaMemcpyDeviceToDevice));
			DDCopied++;
			MemDdCopied += l *sizeof(T);
			target.lastMod = mod_device;
		}else {
			checkCudaError(
				cudaMemcpy(target.elements, CuMatrix<T>::elements + offsetInTs, l * sizeof(T), cudaMemcpyHostToHost));
			HHCopied++;
			target.lastMod = mod_host;
		}
	} else {
		DMatrix<T> src, trg;
		this->asDmatrix(src);
		target.asDmatrix(trg);
		if(gpuReadyQ()) {
			src.elements += offsetInTs;
			MemDdCopied += l *sizeof(T);
			target.lastMod = mod_device;
		}else {
			src.elements = this->elements + offsetInTs;
			trg.elements = target.elements + offsetInTs;
			HHCopied++;
			target.lastMod = mod_host;
		}
		CuMatrix<T>::copyUintDvrg(trg,src,0,0);
	}
	if(debugSync ) {
		outln("Matrix (" << this << " )::reshaped(" << rows << " * " << cols << ", " << offsetInTs << ") -> " << &target << " set lastMod " << b_util::modStr(this->lastMod));
	}
}

template<typename T> Matrix<T> Matrix<T>::reshape(uint rows, uint cols,
		ulong offsetInTs) {
	Matrix<T> res(rows, cols,false,true);
	reshape(res, rows, cols,offsetInTs);
	return res;
}

template<typename T> Matrix<T> Matrix<T>::redimension(
		std::pair<uint, uint>& dims, uint offset) {
	return reshape(dims.first, dims.second, offset);
}

template<typename T> void Matrix<T>::unconcat(Matrix<T>& v, uint rows, uint cols, uint pitch, uint offset) const {
	if(!vectorQ()){
		dthrow(notVector());
	}
	if(offset + rows * cols > this->m * this->n) {
		outln("invalid submatrix ( > this)");
		dthrow(badDimensions());
	}
	v.elements =  this->elements ? this->elements + offset : null ;
	v.d_elements =  this->d_elements ? this->d_elements + offset : null;
	v.m = rows;
	v.n = cols;
	v.p = pitch;
	v.size = v.m * v.n * sizeof(T);
	v.lastMod = CuMatrix<T>::lastMod;
	v.ownsBuffers = false;

	if(debugFill) outln("of " << this->toShortString() << " i am " << v.toShortString());
}

template<typename T> void Matrix<T>::submatrix(Matrix<T>& v, uint rows, uint cols, uint roff, uint coff) const {
	if(roff + rows > this->m || coff + cols > this->n) {
		outln("invalid submatrix ( > this)");
		dthrow(badDimensions());
	}
	uint offset = roff * this->p + coff;
	v.elements =  this->elements ? this->elements + offset : null ;
	v.d_elements =  this->d_elements ? this->d_elements + offset : null;
	v.m = rows;
	v.n = cols;
	v.p = this->p;
	v.size = v.m * v.p * sizeof(T);
	v.lastMod = CuMatrix<T>::lastMod;
	v.ownsBuffers = false;

	if(debugFill) outln("of " << this->toShortString() << " i am " << v.toShortString());
}
// crap
template<typename T> Matrix<T> Matrix<T>::columnMatrix(int col) const {
	Matrix<T> column(CuMatrix<T>::m, 1, false);
	DMatrix<T> d_X, d_Col;
	asDmatrix(d_X);
	column.asDmatrix(d_Col, false);
	CuMatrix<T>::columnMatrixL(d_Col, d_X, col);
	return column;
}

template<typename T> Matrix<T> Matrix<T>::dropFirst(bool copy) const {
	if(this->lastMod == mod_host) dthrow(notSynced());

	Matrix<T> res(CuMatrix<T>::m, CuMatrix<T>::n - 1, false, copy);
	if(copy){
		uint i = 0;
		while (i < CuMatrix<T>::m) {
			checkCudaError(
					cudaMemcpy(res.d_elements + i * (CuMatrix<T>::n - 1), CuMatrix<T>::d_elements + i * CuMatrix<T>::n + 1, (CuMatrix<T>::n - 1) * sizeof(T), cudaMemcpyDeviceToDevice));
			i++;
			DDCopied++;
			MemDdCopied += (CuMatrix<T>::n - 1) * sizeof(T);
			res.lastMod = mod_device;
		}
	} else {
		submatrix(res, this->m, this->n -1, 0, 1);
	}
	return res;
}

template<typename T> Matrix<T> Matrix<T>::vectorToDiagonal() const {
	if (!vectorQ()) {
		dthrow (  notVector());
	}
	if(!this->elements) {
		dthrow(noHostBuffer());
	}
	if(this->lastMod == mod_device) {
		dthrow(notSyncedHost());
	}
	uint dim = longAxis();
	Matrix<T> ret = Matrix<T>::zeros(dim, dim);
	ret.syncBuffers();
	for (uint i = 0; i < dim; i++) {
		ret.elements[i * dim + i] = CuMatrix<T>::elements[i];
	}
	ret.lastMod = mod_host;
	return ret;
}

template<typename T> Matrix<T> Matrix<T>::columnVector(int col) const {
	Matrix<T> res(CuMatrix<T>::m, 1, true);

	for (uint i = 0; i < CuMatrix<T>::m; i++) {
		res.elements[i] = CuMatrix<T>::elements[i * CuMatrix<T>::n + col];
	}
	res.lastMod = mod_host;
	return res;
}

template<typename T> Matrix<T> Matrix<T>::rowVector(int row) const {
	Matrix<T> res(1, CuMatrix<T>::n,true);
	for (uint i = 0; i < CuMatrix<T>::n; i++) {
		res.elements[i] = CuMatrix<T>::elements[row * CuMatrix<T>::n + i];
	}
	res.lastMod = mod_host;
	return res;
}

template<typename T> Matrix<T> Matrix<T>::toRowVector() const {
	return Matrix<T>(CuMatrix<T>::elements, 1, CuMatrix<T>::m * CuMatrix<T>::n, true);
}

template<typename T> Matrix<T> Matrix<T>::toColumnVector() const {
	return Matrix<T>(CuMatrix<T>::elements, CuMatrix<T>::m * CuMatrix<T>::n, 1, true);
}

template<typename T> T Matrix<T>::toScalar() const {
	dassert(scalarQ());

	return CuMatrix<T>::elements[0];
}

template<typename T> Matrix<T> Matrix<T>::toDiagonalsVector() const {
	dassert(squareQ());
	Matrix<T> ret (CuMatrix<T>::n,1,true);
	uint i = 0;
	while (i < CuMatrix<T>::n) {
		ret.elements[i] = CuMatrix<T>::elements[i * CuMatrix<T>::n + i];
		i++;
	}
	ret.lastMod = mod_host;
	return ret;
}

template<typename T> Matrix<T> Matrix<T>::addBiasColumn() const {
	Matrix<T> bias = Matrix<T>::ones(CuMatrix<T>::m, 1);
	return bias.rightConcatenate(*this);
}

template<typename T> inline IndexArray Matrix<T>::rowIndices(uint row) const {
	dassert( validRowQ(row));
	if (CuMatrix<T>::colMajor) {
		uint* ary = new uint[this->n];
		for(uint i =0; i< this->n; i++) {
			ary[i] = i + row * this->p;
		}
		return IndexArray(ary, this->n);

	} else  {
		uint start = row * this->n;
		return IndexArray(start, start + this->n - 1);
	}
}

template<typename T> inline IndexArray Matrix<T>::columnIndices(uint col) const {
	dassert( validColQ(col));
	if (this->colMajor) {
		uint start = col * this->m;
		return IndexArray(start, start + this->m - 1);
	} else {
		uint* c = new uint[this->m];
		for(uint i = 0; i < this->m; i++) {
			c[i] = i * this->m + col;
		}
		return IndexArray(c, this->m);
	}
}

template<typename T> cudaError_t Matrix<T>::rowCopy(Matrix<T>& targ, uint tRow, uint sRow) const {
	IndexArray tary = targ.rowIndices(tRow);
	IndexArray sary = rowIndices(sRow);
	return copyIndexed(targ, tary, *this, sary);
}

template<typename T> void Matrix<T>::copy(Matrix<T>& res, int roff, int coff) const {
	if(roff + res.m > this->m || coff + res.n > this->n) {
		dthrow(outOfBounds());
	}

	if(this->contiguousQ() && res.contiguousQ() && roff == 0 && coff == 0) {
		if(!res.elements || !this->elements) {
			dthrow(noHostBuffer());
		}
		if(!res.d_elements || !this->d_elements) {
			dthrow(noDeviceBuffer());
		}

		if(this->elements) {
			checkCudaErrors( cudaMemcpy(res.elements, this->elements, this->size, cudaMemcpyHostToHost));
			if(debugCopy) outln("host copied " << this->toShortString() << " to " << res.toShortString());
		}
		if(this->d_elements) {
			checkCudaErrors( cudaMemcpy(res.d_elements, this->d_elements, this->size, cudaMemcpyDeviceToDevice));
			if(debugCopy) outln("dev copied " << this->toShortString() << " to " << res.toShortString());
		}
	} else {
		DMatrix<T> d_res, d_M;
		asDmatrix(d_M);
		res.asDmatrix(d_res);
		CuMatrix<T>::copy(d_res, d_M, roff, coff);
		res.lastMod = mod_device;
	}
}

template<typename T> Matrix<T> Matrix<T>::rightConcatenate(
		 const Matrix<T>& other) const {
	if(other.zeroDimsQ()) {
		return *this;
	}
	if(zeroDimsQ()) {
		return other;
	}
	if(other.m != CuMatrix<T>::m) {
		dthrow(matricesOfIncompatibleShape());
	}
	//outln("this " << this->toShortString());
	//outln("other " << other.toShortString());
	if(! gpuReadyQ() ) {
		dthrow(notSynced());
	}
	if(! other.gpuReadyQ() ) {
		dthrow(notSynced());
	}

	uint newCols = CuMatrix<T>::n + other.n;
	Matrix<T> ret(CuMatrix<T>::m, newCols,false, true);
	if (CuMatrix<T>::colMajor){
		dthrow (  notImplemented());
	} else {
		DMatrix<T> d_A, d_B, d_Res;
		asDmatrix(d_A);
		other.asDmatrix(d_B);
		ret.asDmatrix(d_Res,false);
		CuMatrix<T>::rightConcatenateL(d_Res, d_A,d_B);
	}
	ret.lastMod = mod_device;
	return (ret);
}

template<typename T> Matrix<T> Matrix<T>::bottomConcatenate(
		 const Matrix<T>& other) const {
	if(other.zeroDimsQ()) {
		return *this;
	}
	if(zeroDimsQ()) {
		return other;
	}
	if (other.n != CuMatrix<T>::n)
		dthrow (  matricesOfIncompatibleShape());
	uint newRows = CuMatrix<T>::m + other.m;
	Matrix<T> ret(newRows, CuMatrix<T>::n,false);
	ret.syncBuffers();
	if (CuMatrix<T>::colMajor) {
		dthrow ( notImplemented() );
	} else {
		DMatrix<T> d_A, d_B, d_Res;
		asDmatrix(d_A);
		other.asDmatrix(d_B);
		ret.asDmatrix(d_Res,false);
		CuMatrix<T>::bottomConcatenateL(d_Res, d_A,d_B);
	}
	ret.lastMod = mod_device;
	return ret;
}

template<typename T> Matrix<T> Matrix<T>::prependColumnNew( T* col,
		uint count) const {
	dassert(count == CuMatrix<T>::m);
	Matrix<T> res(col, count, false, true);
	return res.rightConcatenate(*this);
}

template<typename T> Matrix<T> Matrix<T>::appendColumnNew( T* col,
		uint count) const {
	dassert(count == CuMatrix<T>::m);
	return rightConcatenate(Matrix<T>(col, count, 1, true));
}

template<typename T> Matrix<T> Matrix<T>::prependRowNew( T* row,
		uint count) const {
	dassert(count == CuMatrix<T>::n);
	return Matrix<T>(row, 1, CuMatrix<T>::n, true).bottomConcatenate(*this);
}

template<typename T> Matrix<T> Matrix<T>::appendRowNew( T* row,
		uint count) const {
	dassert(count == CuMatrix<T>::n);
	return bottomConcatenate(Matrix<T>(row, 1, CuMatrix<T>::n, true));
}

template<typename T> void Matrix<T>::flipud() {
	dthrow(notImplemented());
}

template<typename T> Matrix<T> Matrix<T>::columnSubset( const uint* indices,
		uint count) const {
	uint i = 0;
	Matrix<T> res = Matrix<T>::zeros(0, 0);
	while (i < count) {
		Matrix<T> cVec = columnVector(indices[i]);
		if (res.m == 0 && res.n == 0) {
			res = cVec;
		} else {
			res |= cVec;
		}
		i++;
	}
	outln("columnSubset " << res);
	//res.lastMod = mod_device;
	return res;
}

template<typename T> Matrix<T> Matrix<T>::clippedRowSubset( const int *r, uint count,
		std::pair<uint, uint> colRange) const {
	if(this->colMajor) {
		dthrow(notImplemented())
	}
	if(!this->elements) {
		dthrow(noHostBuffer())
	}
	if(this->lastMod == mod_device) {
		dthrow(notSyncedHost());
	}
	outln("colRange " << pp(colRange) << ", n " << CuMatrix<T>::n);
	dassert((colRange.first < colRange.second && colRange.second < this->n));
	uint newM = count;
	Matrix<T> res = Matrix<T>::zeros(newM,
			colRange.second - colRange.first + 1).syncBuffers();
	outln("clippedRowSubset res " << res.toShortString());
	outln("clippedRowSubset this " << this->toShortString());
	T* b = res.elements;
	uint i = 0;
	uint j = 0;
	uint boff = 0;
	uint eoff = 0;
	while (i < newM) {
		j = colRange.first;
		boff = i * res.n - colRange.first;
		eoff = r[i] * CuMatrix<T>::n;
		while (j <= colRange.second) {
			b[boff + j] = CuMatrix<T>::elements[eoff + j];
			j++;
		}
		i++;
	}
	res.lastMod = mod_host;
	return res;
}

template<typename T> Matrix<T> Matrix<T>::extrude(uint depth) const {
	if (CuMatrix<T>::m != 1) {
		dthrow (notRowVector());
	} else {
		Matrix<T> ret(depth, CuMatrix<T>::n,true);
		for (uint r = 0; r < depth; r++) {
			for (uint c = 0; c < CuMatrix<T>::n; c++) {
				ret.elements[r * CuMatrix<T>::n + c] = CuMatrix<T>::elements[c];
			}
		}
		ret.lastMod = mod_host;
		return (ret);
	}
}

template<typename T> Matrix<T> Matrix<T>::transposeKernelPtr(void (*kernel)(const T*  sElements,  T* tElements, uint width, uint height)) {
/*
	if(scalarQ()) {
		return *this;
	}
	if(!txp) {
		txp = new Matrix<T>(n, m);
		ownsTxp = true;
		DMatrix<T> retD;
		txp->asDmatrix(retD, false);
		transposeL(asDmatrix(), retD);
		txp->lastMod = device;
		txp->txp = this;
		txp->ownsTxp = false;
		if(debugTxp)outln("created txp for " << this->toShortString() << " ( " << txp->toShortString() << ")");
	} else {
		if(debugTxp)outln("reusing txp for " << this->toShortString() << " ( " << txp->toShortString() << ")");
	}
	return *txp;
*/
	if(vectorQ()) {
		outln("degenerate tx");
		syncHost();

		Matrix<T> ret = copy(true);
		ret.m = this->n;
		ret.n = this->m;
		return ret;
	}
	Matrix<T> ret(CuMatrix<T>::n,CuMatrix<T>::m,false,true);
	outln("tx from " << this->toShortString() << " to " << ret.toShortString() );
	DMatrix<T> retD;
	ret.asDmatrix(retD, false);
	transposeKernelPtrL(retD, kernel, asDmatrix());
	ret.invalidateHost();
	return ret;
}

template<typename T> Matrix<T> Matrix<T>::transpose() const {
/*
	if(scalarQ()) {
		return *this;
	}
	if(!txp) {
		txp = new Matrix<T>(n, m);
		ownsTxp = true;
		DMatrix<T> retD;
		txp->asDmatrix(retD, false);
		transposeL(asDmatrix(), retD);
		txp->lastMod = device;
		txp->txp = this;
		txp->ownsTxp = false;
		if(debugTxp)outln("created txp for " << this->toShortString() << " ( " << txp->toShortString() << ")");
	} else {
		if(debugTxp)outln("reusing txp for " << this->toShortString() << " ( " << txp->toShortString() << ")");
	}
	return *txp;
*/
	if(vectorQ() && this->n == this->p)
	{

		Matrix<T> ret;
		ret.m = this->n;
		ret.n = this->m;
		ret.p = this->m;
		ret.ownsBuffers = this->ownsBuffers;
		ret.elements = this->elements;
		ret.d_elements = this->d_elements;
		if(ret.d_elements) ret.getMgr().addDevice(ret);
		if(ret.elements) ret.getMgr().addHost(ret);
		if(debugExec)outln("spoofing transpose for column/row matrix " << this->toShortString());
		return ret;
	}
	Matrix<T> ret(CuMatrix<T>::n,CuMatrix<T>::m, false,true);
	//outln("tx from " << this->toShortString() << " to " << ret.toShortString() );
	DMatrix<T> retD;
	ret.asDmatrix(retD, false);
	transposeL(retD, asDmatrix());
	ret.invalidateHost();
	return ret;
}
// tiles l-to-r
template<typename T> void Matrix<T>::concat(Matrix<T>& canvas, int components, const Matrix<T>** parts) {
	if(debugMem)outln("concat with canvas " << canvas.toShortString());
    ulong canvasSize = 0;
    int dcount =0, hcount=0;
    for(int i = 0; i < components; i++) {
    	const Matrix<T>* c = parts[i];
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
    if(debugMem)outln("concat canvasSize " << canvasSize);
    if(debugMem)outln("concat dcount " << dcount << ", hcount " << hcount << (hcount == 0 ? ";  only copying dmem":""));
	uint n =  canvasSize/sizeof(T);
	//Matrix<T> canvas(1, n, n, false, true);
	if(canvas.d_elements){
		if(debugMem)outln("canvas had d_el " << canvas.d_elements);
		if(canvas.size != canvasSize) {
			if(debugMem)outln("canvas " << canvas.toShortString() << " size != " << canvasSize << " freeing old d mem " << canvas.d_elements);
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

	if(debugMem)outln("concat having canvas.m " << canvas.m << ", n " << canvas.n << ", size " << canvas.size);
	DMatrix<T> dret;
	canvas.asDmatrix(dret,false);
	int streamCount = 2 * components;
	cudaEvent_t cycleDone[streamCount];
	cudaStream_t stream[streamCount];
	for(int i = 0; i < streamCount; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
        checkCudaErrors(cudaEventCreate(&cycleDone[i]));
	}
	int next_stream = 0;
	uint offset = 0;
	uint len = 0;
	for(int i = 0; i < components; i++) {
		const Matrix<T>* currMat = parts[i];
		len = currMat->m * currMat->n;
		if( hcount != 0) {
			if(debugCheckValid)MemMgr<T>::checkValid(canvas.elements);
			if(debugCheckValid)MemMgr<T>::checkValid(canvas.elements + offset);
			if(debugMem)outln("concat copying h2h " << currMat->toShortString() << " using cudaMemcpyAsync\n\t\tcopying " << len << " host Ts from " << currMat->elements << " to " << (canvas.elements + offset));
			checkCudaErrors(cudaMemcpyAsync(
							  canvas.elements + offset,
							  currMat->elements,
							  len * sizeof(T),
							  cudaMemcpyHostToHost,
							  stream[next_stream]));

			HHCopied++;
			MemHhCopied += len * sizeof(T);
			checkCudaErrors(cudaEventRecord(
								cycleDone[next_stream],
								stream[next_stream]));
			next_stream +=1;
		} else {
			if(debugMem)outln("concat skipping host copy (hcount == 0)");
		}
		if(debugCheckValid)MemMgr<T>::checkValid(canvas.d_elements);
		if(debugCheckValid)MemMgr<T>::checkValid(canvas.d_elements + offset);
		if(debugMem)outln("concat copying d2d " << currMat->toShortString() <<
				" using cudaMemcpyAsync\n\t\tcopying " << len << " dev Ts from " << currMat->d_elements << " to " << (canvas.d_elements + offset)) <<
				" ie " << canvas.d_elements << " plus offset " << offset << endl ;
		if(debugMem)outln("&canvas.d_elements[len] " << &canvas.d_elements[len]);

		checkCudaErrors(cudaMemcpyAsync(
							  canvas.d_elements + offset,
							  currMat->d_elements,
							  len * sizeof(T),
							  cudaMemcpyDeviceToDevice,
							  stream[next_stream]));
		DDCopied++;
		MemDdCopied += len * sizeof(T);

		checkCudaErrors(cudaEventRecord(
							cycleDone[next_stream],
							stream[next_stream]));
		next_stream +=1;
    	offset += len;

	}
	canvas.lastMod = hcount==0 ? mod_device : mod_synced;
	if(debugFill)outln("concat made canvas " << canvas.toShortString() << "\n\n");
	for(int i = 0; i < streamCount; i++) {
		checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
}

template<typename T> void Matrix<T>::transpose(DMatrix<T>& retD) {
	transposeL(retD, asDmatrix());
	invalidateHost();
}

template<typename T> void Matrix<T>::transposeKernelPtr(DMatrix<T>& retD, void (*kernel)(const T*,T*,uint,uint)) {
	transposeKernelPtrL(retD, kernel, asDmatrix());
	invalidateHost();
}

/////////////////////////////////////////////////////////////////////////
//
//  Qs
//
/////////////////////////////////////////////////////////////////////////

template<typename T> inline bool Matrix<T>::biLocatedQ() const {
	return (this->elements && this->d_elements);
}

template<typename T> inline bool Matrix<T>::gpuReadyQ() const {
	return this->d_elements != null &&
			(this->lastMod == mod_synced || this->lastMod == mod_device || this->lastMod == mod_neither);
}

template<typename T> inline bool Matrix<T>::vectorQ() const {
	return (CuMatrix<T>::m == 1 || CuMatrix<T>::n == 1);
}

template<typename T> inline bool Matrix<T>::squareQ() const {
	return (CuMatrix<T>::m == CuMatrix<T>::n);
}

template<typename T> inline bool Matrix<T>::rowVectorQ() const {
	return CuMatrix<T>::m == 1;
}

template<typename T> inline bool Matrix<T>::columnVectorQ() const {
	return CuMatrix<T>::n == 1;
}

template<typename T> inline bool Matrix<T>::scalarQ() const {
	return CuMatrix<T>::m == 1 && CuMatrix<T>::n == 1;
}

template<typename T> inline bool Matrix<T>::validDimsQ() const {
	return CuMatrix<T>::m > 0 && CuMatrix<T>::n > 0;
}

template<typename T> inline bool Matrix<T>::validColQ(uint col) const {
	return (col < CuMatrix<T>::n);
}

template<typename T> inline bool Matrix<T>::validRowQ(uint row) const {
	return (row < CuMatrix<T>::m);
}

template<typename T> inline bool Matrix<T>::validIndicesQ(uint row, uint col) const {
	return (col < CuMatrix<T>::n && row < CuMatrix<T>::m);
}

template<typename T> uintPair Matrix<T>::dims() const {
	return uintPair(CuMatrix<T>::m,CuMatrix<T>::n);
}

template<typename T> uint Matrix<T>::longAxis() const {
	return MAX(CuMatrix<T>::m,CuMatrix<T>::n);
}

template<typename T> T Matrix<T>::vectorLength() const {
	return ::sqrt(autoDot());
}

template<typename T> bool Matrix<T>::zeroDimsQ() const {
	return CuMatrix<T>::m == 0 && CuMatrix<T>::n == 0 && CuMatrix<T>::size == 0;
}

template<typename T> bool Matrix<T>::zeroQ(T epsilon) {
	if( !zeroDimsQ() ) {
		outln("zeroQ !zeroDims");
		almostEqualsBoolUnaryOp<T> op;
		op.epsilon = epsilon;
		op.target = 0;
		andBinaryOp<T> andOp;
		return gloloReduce(op, andOp, true);
	}
	return true;
}

template<typename T> bool Matrix<T>::hasBiasColumn() {
	Matrix<T> col1;
	submatrix(col1, this->m, 1, 0,0);
	almostEqualsBoolUnaryOp<T> eqOp;
	eqOp.epsilon = util<T>::epsilon();
	eqOp.target = 1;
	return col1.all(eqOp);
}

template<typename T> template<typename BoolUnaryOp> bool Matrix<T>::all(
		BoolUnaryOp fn) const {
	return gloloReduce(fn, andBinaryOp<T>(), true);
}

template<typename T> template<typename BoolUnaryOp> bool Matrix<T>::any(
		BoolUnaryOp fn) const {
	return gloloReduce(fn, orBinaryOp<T>(), false);
}

template<typename T> template<typename BoolUnaryOp> bool Matrix<T>::none(
		BoolUnaryOp fn) const {
	return !any(fn);
}

/////////////////////////////////////////////////////////////////////////
//
// matrix math/features
//
/////////////////////////////////////////////////////////////////////////


template<typename T> Matrix<T> Matrix<T>::inverse() const {
	dassert(CuMatrix<T>::n == CuMatrix<T>::m);
	T d = determinant();
	dassert(d != 0);
	// linearly independent
	Matrix<T> mT = cofactorM().syncBuffers().transpose();
	return (mT / d);
}

// todo cudatize me
template<typename T> T Matrix<T>::determinant() const {
	dassert((CuMatrix<T>::n == CuMatrix<T>::m));
	switch (CuMatrix<T>::n) {
	case 1:
		return (CuMatrix<T>::elements[0]);
	case 2:
		return (CuMatrix<T>::elements[0] * CuMatrix<T>::elements[3] - CuMatrix<T>::elements[1] * CuMatrix<T>::elements[2]);
	default:
		// cofactor expansion along the first row or column
		T sum = 0;

		if(CuMatrix<T>::colMajor) {
			uint col = 0;
			while (col < CuMatrix<T>::n) {
				sum += CuMatrix<T>::elements[col * CuMatrix<T>::m] * cofactor(0, col);
				col++;
			}
		} else {
			uint row = 0;
			while (row < CuMatrix<T>::m) {
				sum += CuMatrix<T>::elements[row * CuMatrix<T>::n] * cofactor(row, 0);
				row++;
			}
		}
		return (sum);
	}
}

template<typename T> Matrix<T> Matrix<T>::cofactorM() const {
	Matrix<T> ret(CuMatrix<T>::m, CuMatrix<T>::n,true, true);

	T* c = ret.elements;
	uint row = 0;
	uint col = 0;
	uint i = 0;
	while (row < CuMatrix<T>::m) {
		col = 0;
		while (col < CuMatrix<T>::n) {
			c[i] = cofactor(row, col);
			col++;
			i++;
		}
		row++;
	}
	ret.lastMod = mod_host;
	return (ret);
}

template<typename T> int Matrix<T>::sgn(uint row, uint col) const {
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

template<typename T> Matrix<T> Matrix<T>::minorM(uint row, uint col) const {
	validIndicesQ(row, col);
	Matrix<T> ret(CuMatrix<T>::m - 1, CuMatrix<T>::n - 1, true);
	uint i = 0;
	long j = 0;
	long len = CuMatrix<T>::m * CuMatrix<T>::n;
	uintPair range = CuMatrix<T>::colMajor ? columnIndices(col).toPair() : rowIndices(row).toPair();
	//outln( (CuMatrix<T>::colMajor ? "col ":  "row ") << (CuMatrix<T>::colMajor ? col :  row )  << " indices " << pp(range));
	while (j < len) {
		if ((j < range.first || j > range.second)
				&&  (CuMatrix<T>::colMajor  ? ( j % CuMatrix<T>::m != row) : (j % CuMatrix<T>::n != col) )) {
			ret.elements[i] = CuMatrix<T>::elements[j];
			i++;
		}
		j++;
	}
	//outln(toString() << "\nminor " << row << ", " << col << " " << ret.toString());
	return ret;
}

template<typename T> T Matrix<T>::minor(uint row, uint col) const {
	return (minorM(row, col).determinant());
}

template<typename T> T Matrix<T>::cofactor(uint row, uint col) const {
	return (minor(row, col) * sgn(row, col));
}

template<typename T> Matrix<T> Matrix<T>::matrixProduct( const Matrix<T>& b, dim3* block) const {
	// todo convert mats into big warp div matrix and one < blocksize matrix
	if(b.scalarQ()) {
		return this->operator *(b.get(0));
	} else if(scalarQ()) {
		return b.operator *(get(0));
	} else if(vectorQ() && b.vectorQ()) {
		// better as a reduction
		if( !(rowVectorQ() && b.rowVectorQ()) && !(columnVectorQ() && b.columnVectorQ())) {
			if(this->n == 1) {
				if(debugMatProd) outln(this->toShortString() << " posing as row");
			} else {
				if(debugMatProd) outln(b.toShortString() << " posing as row");
			}
		}

		if(debugMatProd) outln("cuplavects");
		T ret = matrixReduce(multBinaryOp<T>(), plusBinaryOp<T>(), b, 0);
		if(debugMatProd) outln("matProd expecting a vector " << ret);
		return Matrix<T>::fromScalar(ret);
	}

	Matrix<T> res(CuMatrix<T>::m, b.n,false, true);
	if(debugMatProd) outln("matrix product this " << this << ", res " << &res << ", b " << &b);
	if(debugMatProd) outln("matrix product dims " << Matrix<T>::toShortString() << ", res.dims " << res.toShortString() << ", b.po " << b.toShortString());
	if(debugMatProd) outln("matrix product this.lastMod " << CuMatrix<T>::lastMod << ", o.lastMod " << b.lastMod );
	DMatrix<T> d_A, d_B, d_C;
	asDmatrix(d_A);
	b.asDmatrix(d_B);
	res.asDmatrix(d_C, false);
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
	if(debugMatProd) outln("d_B.n " << d_B.n);
	if(debugMatProd) outln("d_A.m " << d_A.m);
	CuMatrix<T>::matrixProductL(d_C, d_A, d_B,  block);
	res.invalidateHost();

	if(debugMatProd) outln("matrixProduct updated res " << &res << " to LastUpdate " << mod_device);
	return res;
}

// DMatrix<T> multiplication kernel called by MatMul()

template<typename T> Matrix<T> Matrix<T>::toMaxColumnIndexVector() const {
	Matrix<T> ret(CuMatrix<T>::m,1,false, true);
	DMatrix<T> d_A, d_res;
	asDmatrix(d_A);
	ret.asDmatrix(d_res, false);
	CuMatrix<T>::toMaxColumnIndexVectorL(d_res,d_A);
	ret.invalidateHost();
	return ret;
}

template<typename T> Matrix<T> Matrix<T>::subFrom(T o) const {
	subFromUnaryOp<T> subff;
	subff.source = o;
	return unaryOp(subff);
}

/////////////////////////////////////////////////////////////////////////
//
// unary and binary element-wise and reduction functor launchers
//
/////////////////////////////////////////////////////////////////////////

template<typename T> template<typename UnaryOp> Matrix<T> Matrix<T>::unaryOp(
		UnaryOp op) const {
	Matrix<T> res(CuMatrix<T>::m, CuMatrix<T>::n, false, true);
	if(debugExec) outln("unary op src " << this->toShortString() << " targ " << res.toShortString());
	unaryOp(res, op);
	checkCudaError(cudaGetLastError());
	return res;
}

template<typename T> template<typename UnaryOp> void Matrix<T>::unaryOp(
		Matrix<T>& res, UnaryOp op) const {
	if(debugExec) outln("unaryOp this->d_element " << this->d_elements << ", " << res.d_elements);
	//outln("this " << this->toShortString() << ", res " << res.toShortString());
	DMatrix<T> d_A, d_res;
	asDmatrix(d_A);
	res.asDmatrix(d_res, false);
	if(this->p == this->n) {
		CuMatrix<T>::unaryOpL( d_res, d_A, op);
	} else {
		if(debugExec)outln("invoking DMatrix version of unaryOp");
		CuMatrix<T>::unaryOpDmL(d_res, d_A, op);
	}
	res.invalidateHost();
}

template<typename T> template<typename BinaryOp> void Matrix<T>::binaryOp(
		const Matrix<T>& o, Matrix<T>& res, BinaryOp op) const {
	DMatrix<T> d_A, d_B, d_res;
	asDmatrix(d_A);
	o.asDmatrix(d_B);
	res.asDmatrix(d_res, false);
	if(this->n == this->p) {
		CuMatrix<T>::binaryOpL( d_res, d_A, d_B,op);
	} else {
		CuMatrix<T>::binaryOpDmL( d_res, d_A, d_B,op);
	}
	res.invalidateHost();
}

template<typename T> template<typename BinaryOp> Matrix<T> Matrix<T>::binaryOp(
		const Matrix<T>& o, BinaryOp op) const {
	if(!equalDims(o)) {
		outln(this->toShortString() << " can't be bin-opd with " << o.toShortString());
		dthrow(matricesOfIncompatibleShape());
	}
	Matrix<T> res(CuMatrix<T>::m, CuMatrix<T>::n, false, true);
	binaryOp(o, res, op);
	return res;
}

template<typename T> Matrix<T> Matrix<T>::hadamardProduct(const Matrix<T> o) const {
	return binaryOp(o, multBinaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::hadamardQuotient(const Matrix<T> o) const {
	return binaryOp(o, quotientBinaryOp<T>());
}

template<typename T> template<typename BinaryOp> __host__ T Matrix<T>::reduce(
		const DMatrix<T>& d_M, BinaryOp op, T start, cudaStream_t stream  ) {
	uint nP = d_M.m * d_M.n;
	int threads;
	int blocks;
	CuMatrix<T>::getReductionExecContext(blocks, threads, nP);
	if(debugExec) outln("reduce blocks " << blocks);
	Matrix<T> res(blocks, 1, true, true);
	DMatrix<T> d_Res;
	res.asDmatrix(d_Res, false);
	T total;
	if(d_M.p != d_M.n) {
		total = CuMatrix<T>::reduceLauncherDm(d_Res, d_M, nP, op, start, stream);
	} else {
	 total = CuMatrix<T>::reduceLauncher(res.d_elements, d_M.elements, nP, op, start, stream);
	}
	if(syncHappy)checkCudaError(cudaDeviceSynchronize());
	return total;
}

template<typename T> template<typename BinaryOp> T Matrix<T>::reduce(BinaryOp op, T start, cudaStream_t stream ) const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
	T res = reduce(d_A, op, start, stream);
	return res;
}

template<typename T> template<typename IndexBoolUnaryOp,typename BinaryOp> __host__ T Matrix<T>::indexedReduceL(
		const DMatrix<T>& d_M, IndexBoolUnaryOp idxOp, BinaryOp op, T start) const {
	unsigned int nP = d_M.m * d_M.n;
	int threads;
	int blocks;
	CuMatrix<T>::getReductionExecContext(blocks,threads, nP);
	Matrix<T> res(blocks, 1, false, true);
	DMatrix<T> d_Res;
	res.asDmatrix(d_Res, false);
	T total = CuMatrix<T>::indexedReduceLauncher( res.d_elements, d_M.elements, nP, idxOp, op, start);
	if(syncHappy)cudaDeviceSynchronize();
	return total;
}

template<typename T> template<typename IndexBoolUnaryOp, typename BinaryOp> T Matrix<T>::indexedReduce(
		IndexBoolUnaryOp idxOp,BinaryOp op, T start) const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
	T res = indexedReduceL(d_A, idxOp, op, start);
	return res;
}

template<typename T> template<typename UnaryOp, typename BinaryOp> __host__ T Matrix<
		T>::gloloReduceL(const DMatrix<T>& d_M, UnaryOp gop, BinaryOp lop,
		T start) const {
	uint nP = d_M.m * d_M.n;
	int threads;
	int blocks;
	CuMatrix<T>::getReductionExecContext(blocks,threads, nP);
	Matrix<T> res(blocks, 1,true, true);
	DMatrix<T> d_Res;
	res.asDmatrix(d_Res, false);
	T total = gloloReduceOpLauncher(res.d_elements, d_M.elements, nP, gop, lop, start);
	if(syncHappy)cudaDeviceSynchronize();
	return total;
}

template<typename T> template<typename UnaryOp, typename BinaryOp> T Matrix<T>::gloloReduce(
		UnaryOp gop, BinaryOp lop, T start) const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
	T res = gloloReduceL(d_A, gop, lop, start);
	return res;
}

template<typename T> template<typename IndexBoolUnaryOp, typename UnaryOp, typename BinaryOp> __host__ T Matrix<
		T>::indexedGloloReduceL(const DMatrix<T>& d_M, IndexBoolUnaryOp idxOp, UnaryOp gop, BinaryOp lop,
		T start) const {
	uint nP = d_M.m * d_M.n;
	int threads;
	int blocks;
	CuMatrix<T>::getReductionExecContext(blocks,threads, nP);
	Matrix<T> res(blocks, 1,true, true);
	DMatrix<T> d_Res;
	res.asDmatrix(d_Res, false);
	T total = indexedGloloReduceOpLauncher( res.d_elements, d_M.elements, nP, idxOp, gop, lop, start);
	if(syncHappy)cudaDeviceSynchronize();
	return total;
}


template<typename T> template<typename IndexBoolUnaryOp, typename UnaryOp, typename BinaryOp> T Matrix<T>::indexedGloloReduce(
		IndexBoolUnaryOp idxOp, UnaryOp gop, BinaryOp lop, T start) const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
	T res = indexedGloloReduceL(d_A, idxOp, gop, lop, start);
	return res;
}

template<typename T> template<typename MatBinaryOp, typename BinaryOp> __host__ T Matrix<
		T>::matrixReduceL(const DMatrix<T>& d_M1, const DMatrix<T>& d_M2,
		MatBinaryOp mop, BinaryOp op, T start) const {
	uint nP = d_M1.m * d_M1.n;
	int threads;
	int blocks;
	CuMatrix<T>::getReductionExecContext(blocks,threads, nP);
	Matrix<T> res(blocks, 1,true,true);
	DMatrix<T> d_Res;
	res.asDmatrix(d_Res, false);
	T total = CuMatrix<T>::matrixReduceOpLauncher(res.d_elements, d_M1.elements, d_M2.elements, nP, mop, op, start);
	if(syncHappy)cudaDeviceSynchronize();
	return total;
}

template<typename T> template<typename MatBinaryOp, typename BinaryOp> __host__ T Matrix<
		T>::matrixReduceL(const DMatrix<T>& d_M1, const DMatrix<T>& d_M2,
		const Matrix<T>& temp, MatBinaryOp mop, BinaryOp op, T start) const {
	uint nP = d_M1.m * d_M1.n;
	T total = CuMatrix<T>::matrixReduceOpLauncher(temp.d_elements, d_M1.elements, d_M2.elements, nP, mop, op, start);
	if(syncHappy)cudaDeviceSynchronize();
	return total;
}

template<typename T> template<typename MatBinaryOp, typename BinaryOp> T Matrix<
		T>::matrixReduce(MatBinaryOp mop, BinaryOp op, const Matrix<T>& o, T start) const {
	DMatrix<T> d_A, d_B;
	asDmatrix(d_A);
	o.asDmatrix(d_B);
	T res = matrixReduceL(d_A, d_B, mop, op, start);
	return res;
}

template<typename T> template<typename MatBinaryOp, typename BinaryOp> T Matrix<
		T>::matrixReduce(MatBinaryOp mop, BinaryOp op, const Matrix<T>& o,
		Matrix<T>& buffer, T start) const {
	DMatrix<T> d_A, d_B;
	asDmatrix(d_A);
	o.asDmatrix(d_B);
	T res = matrixReduceL(d_A, d_B, buffer, mop, op, start);
	return res;
}

template<typename T> T Matrix<T>::autoDot() const {
	return gloloReduce(sqrUnaryOp<T>(), plusBinaryOp<T>(), 0);
}

template<typename T> template<typename BinaryOp> T Matrix<T>::columnReduce(
		BinaryOp op, uint column, T start ) const {
	if(!validColQ(column)) {
		throw(columnOutOfBounds());
	}
	isColumnUnaryOp idxOp;
	idxOp.column = column;
	idxOp.pitch = CuMatrix<T>::p;
	return indexedReduce(idxOp, op, 0);
}


template<typename T> T Matrix<T>::columnSum(uint column) const {
	return columnReduce( plusBinaryOp<T>(), column, 0);
}

template<typename T> template<typename BinaryOp> T Matrix<T>::rowReduce(
		BinaryOp op, uint row, T start ) const {
	if(!validRowQ(row)) {
		throw(rowOutOfBounds());
	}
	isRowUnaryOp idxOp;
	idxOp.row = row;
	idxOp.pitch = CuMatrix<T>::p;
	return indexedReduce(idxOp, op, 0);
}

template<typename T> T Matrix<T>::rowSum(uint row) const {
	return rowReduce(plusBinaryOp<T>(), row, 0);
}

template<typename T> T Matrix<T>::sum(cudaStream_t stream ) const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
	if(syncHappy)b_util::syncGpu();
	T res = reduce(d_A, plusBinaryOp<T>(), 0 );
	return res;
}

template<typename T> T Matrix<T>::prod( cudaStream_t stream ) const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
	if(syncHappy)b_util::syncGpu();
	T res = reduce(d_A, multBinaryOp<T>(), 1.0);
	return res;
}

template<typename T> T Matrix<T>::min(  cudaStream_t stream ) const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
	if(syncHappy)b_util::syncGpu();
	T res = reduce(d_A, minBinaryOp<T>(), util<T>::maxValue() );
	return res;
}

template<typename T> T Matrix<T>::max( cudaStream_t stream ) const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
	if(syncHappy)b_util::syncGpu();
	T res = reduce(d_A, maxBinaryOp<T>(), util<T>::minValue());
	return res;
}

template<typename T> pair<T,T> Matrix<T>::bounds() const {
	DMatrix<T> d_A;
	asDmatrix(d_A);
	if(syncHappy)b_util::syncGpu();
	cudaStream_t stream[2];
	for(int i = 0; i < 2; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
	CuTimer watch;
	watch.start();
	T min = reduce(d_A, minBinaryOp<T>(), util<T>::maxValue(), stream[0]);
	outln("min took " << watch.stop());
	T max = reduce(d_A, maxBinaryOp<T>(), util<T>::minValue(), stream[1]);
	watch.start();
	outln("max took " << watch.stop());
	b_util::syncGpu();
	for(int i = 0; i < 2; i++) {
		checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
	return pair<T,T>(min,max);
}

template<typename T> T Matrix<T>::sumSqrDiff( const Matrix<T>& o) const {
	return matrixReduce(diffSquaredBinaryOp<T>(), plusBinaryOp<T>(), o, 0);
}

template<typename T> T Matrix<T>::sumSqrDiff( Matrix<T>& reductionBuffer, const Matrix<T>& o) const {
	return matrixReduce(diffSquaredBinaryOp<T>(), plusBinaryOp<T>(), o,
			reductionBuffer, 0);
}

template<typename T> T Matrix<T>::accuracy( const Matrix<T>& o) const {
	return matrixReduce(equalsBinaryOp<T>(), plusBinaryOp<T>(), o, 0)/CuMatrix<T>::m;
}

template<typename T> T Matrix<T>::accuracy( Matrix<T>& reductionBuffer, const Matrix<T>& o) const {
	return matrixReduce(equalsBinaryOp<T>(), plusBinaryOp<T>(), o,
			reductionBuffer, 0)/CuMatrix<T>::m;
}

template<typename T> bool Matrix<T>::isBinaryCategoryMatrix() const {
	return gloloReduce(oneOrZeroBoolUnaryOp<T>(), andBinaryOp<T>(), true);
}

template<typename T> Matrix<T> Matrix<T>::negate() const {
	return unaryOp(negateUnaryOp<T>());
}

/////////////////////////////////////////////////////////////////////////
//
// operators
//
/////////////////////////////////////////////////////////////////////////

template<typename T> Matrix<T> Matrix<T>::operator=(const Matrix<T> o) {
	if(debugMem)outln("operator= setting " << this->toShortString() << " from " << o.toShortString() );
	anyErr();
	if (this == &o) {
		return *this;
	}
	this->m = o.m;
	this->n = o.n;
	this->p = o.p;
	this->size = o.size;
	this->lastMod = o.lastMod;
	if(this->elements) {
		if(debugMem) outln( this << " operator=(const Matrix<T> o) freeing h " << this->elements );
		getMgr().freeHost(*this);
	}
	if(this->d_elements) {
		if(debugMem) outln( this << " operator=(const Matrix<T> o) freeing d " << this->d_elements );
		getMgr().freeDevice(*this);
	}
	if(o.elements) {
		this->elements=o.elements;
		getMgr().addHost(*this);
	}
	if(o.d_elements) {
		this->d_elements=o.d_elements;
		getMgr().addDevice(*this);
	}
	freeTxp();
	return *this;
}

template<typename T> Matrix<T> Matrix<T>::operator^(T o) const {
	return pow(o);
}

template<typename T> Matrix<T> Matrix<T>::operator^(int o) const {
	return pow( static_cast<T>(  o));
}

template<typename T> Matrix<T> Matrix<T>::operator<(T o) const {
	ltUnaryOp<T> ltf;
	ltf.comp = o;
	return unaryOp(ltf);
}

template<typename T> Matrix<T> Matrix<T>::operator<=(T o) const {
	lteUnaryOp<T> ltef;
	ltef.comp = o;
	return unaryOp(ltef);
}

template<typename T> Matrix<T> Matrix<T>::operator>(T o) const {
	gtUnaryOp<T> gtf;
	gtf.comp = o;
	return unaryOp(gtf);
}

template<typename T> Matrix<T> Matrix<T>::operator>=(T o) const {
	gteUnaryOp<T> gtef;
	gtef.comp = o;
	return unaryOp(gtef);
}

template<typename T> Matrix<T> Matrix<T>::operator==(T o) const {
	eqUnaryOp<T> eqf;
	eqf.comp = o;
	return unaryOp(eqf);
}

template<typename T> Matrix<T> Matrix<T>::operator+( const Matrix<T> o) const {
	return binaryOp(o, plusBinaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::operator+(T o) const {
	translationUnaryOp<T> addf;
	addf.addend = o;
	return unaryOp(addf);
}

template<typename T> Matrix<T> Matrix<T>::operator-( const Matrix<T> o) const {
	return binaryOp(o, minusBinaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::operator-(T o) const {
	translationUnaryOp<T> addf;
	addf.addend = -o;
	return unaryOp(addf);
}

template<typename T> Matrix<T> Matrix<T>::operator*(Matrix<T> o)  const {
	return matrixProduct(o);
}

template<typename T> Matrix<T> Matrix<T>::operator%(Matrix<T> o) const {
	return hadamardProduct(o);
}

template<typename T> Matrix<T> Matrix<T>::operator&&(Matrix<T> o) const {
	return binaryOp(o, andBinaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::operator*(T o) const {
	scaleUnaryOp<T> multf;
	multf.multiplicand = o;
	return unaryOp(multf);
}

template<typename T> Matrix<T> Matrix<T>::operator/(T o) const {
	scaleUnaryOp<T> multf;
	multf.multiplicand = static_cast<T>( 1. / o);
	return unaryOp(multf);
}

template<typename T> Matrix<T> Matrix<T>::operator|=( const Matrix<T> b) const {
	return rightConcatenate(b);
}

template<typename T> Matrix<T> Matrix<T>::operator/=( const Matrix<T> b) const {
	return bottomConcatenate(b);
}

template<typename T> bool Matrix<T>::operator==( const Matrix<T> o) const {
	bool thisZero = CuMatrix<T>::size == 0;
	bool oZero = o.size == 0;
	if(this == &o || ( thisZero && oZero)) {
		outln("both are zero");
		b_util::dumpStack();
		return true;
	}
	if( oZero || thisZero ) {
		return false;
	}
	return matrixReduce(equalsBinaryOp<T>(), andBinaryOp<T>(), o, true);
}

template<typename T> bool Matrix<T>::operator!=( const Matrix<T> o) const {
	return !matrixReduce(equalsBinaryOp<T>(), andBinaryOp<T>(), o, true);
}

template<typename T> bool Matrix<T>::almostEq( const Matrix<T>& o, T epsilon) const {
	almostEqualsBinaryOp<T> op;
	op.epsilon = epsilon;
	return matrixReduce(op, andBinaryOp<T>(), o, true);
}

/////////////////////////////////////////////////////////////////////////
//
// ml functions
//
/////////////////////////////////////////////////////////////////////////


template<typename T> Matrix<T> Matrix<T>::sigmoid() const {
	return unaryOp(sigmoidUnaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::sigmoidGradient() const {
	return unaryOp(sigmoidGradientUnaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::log() const {
	return unaryOp(logUnaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::oneOver() const {
	return unaryOp(oneOverUnaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::exp() const {
	return unaryOp(expUnaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::sqrt() const {
	return unaryOp(sqrtUnaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::sqr() const {
	return unaryOp(sqrUnaryOp<T>());
}

template<typename T> Matrix<T> Matrix<T>::pow(T o) const {
	powUnaryOp<T> pf;
	pf.power = o;
	return unaryOp(pf);
}

template<typename T> Matrix<T> Matrix<T>::divSqrt(T divisor) const {
	divSqrtUnaryOp<T> dsf;
	dsf.divisor = divisor;
	return unaryOp(dsf);
}

template<typename T> void Matrix<T>::fitGaussians(Matrix<T>& sqrdSigmas, Matrix<T>& mus) const {
	DMatrix<T> d_Sigmas, d_X, d_Mus;
	sqrdSigmas.poseAsRow();
	sqrdSigmas.asDmatrix(d_Sigmas, false);
	sqrdSigmas.unPose();
	asDmatrix(d_X);
	mus.asDmatrix(d_Mus);
	CuMatrix<T>::varianceAndMeanL(d_Sigmas, d_Mus, d_X );
	sqrdSigmas.invalidateHost();
	mus.invalidateHost();
}

template<typename T> void Matrix<T>::variance(Matrix<T>& sqrdSigmas, const Matrix<T>& mus) const {
	DMatrix<T> d_Sigmas, d_X, d_Mus;
	sqrdSigmas.poseAsRow();
	sqrdSigmas.asDmatrix(d_Sigmas, false);
	sqrdSigmas.unPose();
	asDmatrix(d_X);
	mus.asDmatrix(d_Mus);
	CuMatrix<T>::varianceL(d_Sigmas, d_X, d_Mus);
	sqrdSigmas.invalidateHost();
}

template<typename T> void Matrix<T>::toCovariance(Matrix<T>& covmat) const {
	if(!vectorQ()) {
		dthrow(notVector());
	}
	if(!covmat.squareQ() || covmat.n != this->longAxis()) {
		dthrow(badDimensions());
	}
	if(covmat.lastMod == mod_device) {
		dthrow(notSyncedHost());
	}
	for(uint diag = 0; diag < covmat.n; diag++) {
		covmat.set(diag,diag, get(diag));
	}
	covmat.invalidateDevice();
}

template<typename T> Matrix<T> Matrix<T>::toCovariance() const {
	if(!vectorQ()) {
		dthrow(notVector());
	}
	Matrix<T> covmat = Matrix<T>::zeros(this->longAxis(), this->longAxis());
	covmat.syncBuffers();
	toCovariance(covmat);
	return covmat;
}

template<typename T> void Matrix<T>::multivariateGaussianFeatures( Matrix<T>& pden, const Matrix<T>& sqrdSigmas, const Matrix<T>& mu) {
	DMatrix<T> d_sqrdSigmas, d_x, d_mu,d_pden;
	sqrdSigmas.asDmatrix(d_sqrdSigmas);
	asDmatrix(d_x);
	mu.asDmatrix(d_mu);
	pden.asDmatrix(d_pden,false);
	CuMatrix<T>::multivariateGaussianFeatures(d_pden,d_x, d_sqrdSigmas, d_mu);
	pden.invalidateHost();
}

template<typename T> void Matrix<T>::mvGaussianVectorFromFeatures( Matrix<T>& pvec){
	DMatrix<T> d_pvec,d_pdens;
	asDmatrix(d_pdens);
	pvec.asDmatrix(d_pvec,false);
	CuMatrix<T>::mvGaussianVectorFromFeatures(d_pvec,d_pdens);
	pvec.invalidateHost();
}

template<typename T> void Matrix<T>::multivariateGaussianVector( Matrix<T>& pvec, const Matrix<T>& sqrdSigmas, const Matrix<T>& mu) {
	DMatrix<T> d_sqrdSigmas, d_x, d_mu,d_pvec;
	sqrdSigmas.asDmatrix(d_sqrdSigmas);
	asDmatrix(d_x);
	mu.asDmatrix(d_mu);
	pvec.asDmatrix(d_pvec, false);
	CuMatrix<T>::multivariateGaussianVector(d_pvec,d_x, d_sqrdSigmas, d_mu);
	pvec.invalidateHost();
}

template<typename T> Matrix<T> Matrix<T>::multivariateGaussianVectorM( const Matrix<T>& sqrdSigmas, const Matrix<T>& mu) {
	Matrix<T> covariance = sqrdSigmas.squareQ() ? sqrdSigmas : sqrdSigmas.toCovariance();
	Matrix<T> coi =  covariance.inverse();
	Matrix<T> xnorm = subMeans(mu);
	coi.syncBuffers();
	return (::pow(ONE_OVER_2PI, xnorm.n/2.0) / ::sqrt(covariance.determinant())) /
			(((xnorm * coi) % xnorm).rowSum() * 0.5).exp();
}

template<typename T> Matrix<T> Matrix<T>::normalize() const {
	Matrix<T> mus = featureMeans(true);
	Matrix<T> subm = subMeans(mus);
	uint l = CuMatrix<T>::m * CuMatrix<T>::n;
	T sqrSum = subm.reduce(sqrPlusBinaryOp<T>(), 0);
	T sum = subm.sum();
	T avg = sum / l;
	T stdDev = ::sqrt(sqrSum / l - (avg * avg));
	return subm / stdDev;
}

template<typename T> void Matrix<T>::featureMeans( Matrix<T>& means, bool lv) const {
	DMatrix<T> d_Means, d_X;
	asDmatrix(d_X);
	means.asDmatrix(d_Means);
	CuMatrix<T>::featureAvgKernelL(d_Means, d_X, lv);
	means.invalidateHost();
}

template<typename T> void Matrix<T>::subMeans( Matrix<T>& res,
		 const Matrix<T>& means) const {
	outln("means " << &means << " with elements " << means.elements << " and dims " << means.m << "*" << means.n);
	DMatrix<T> d_Means, d_X, d_Res;
	asDmatrix(d_X);
	means.asDmatrix(d_Means);
	res.asDmatrix(d_Res, false);
	CuMatrix<T>::meanSubL(d_Res, d_X, d_Means);
}

template<typename T> Matrix<T> Matrix<T>::featureMeans(bool lv) const {
	Matrix<T> means = Matrix<T>::zeros(CuMatrix<T>::n, 1);
	featureMeans(means, lv);
	return means;
}

template<typename T> Matrix<T> Matrix<T>::subMeans( const Matrix<T>& means) const {
	Matrix<T> res = Matrix<T>::zeros(CuMatrix<T>::m, CuMatrix<T>::n);
	subMeans( res, means);
	return res;
}

template<typename T> cudaError_t Matrix<T>::sqrSubMeans( Matrix<T>& res, const Matrix<T>& mus) const {
	DMatrix<T> d_Means, d_X, d_Res;
	asDmatrix(d_X);
	mus.asDmatrix(d_Means);
	res.asDmatrix(d_Res, false);
	CuMatrix<T>::meanSubSqrL(d_Res, d_X, d_Means);
	return cudaGetLastError();
}

/*
 * m*n -> m*1; each row is product of all row features
 */
template<typename T> void  Matrix<T>::rowProductTx(Matrix<T>& res) const {
	if(res.m != this->m || res.n != 1) {
		dthrow(matricesOfIncompatibleShape());
	}
	res.poseAsRow();
	DMatrix<T> d_prod, d_x;
	Matrix<T> tx = transpose();
	tx.asDmatrix(d_x);
	res.asDmatrix(d_prod, false);
	res.unPose();
	CuMatrix<T>::columnProduct(d_prod, d_x);
	res.invalidateHost();
}

/*
 * m*n -> m*1; each row is sum of all row features
 */
template<typename T> void  Matrix<T>::rowSum(Matrix<T>& rowSumM) const {
	if(rowSumM.m != this->m || rowSumM.n != 1) {
		dthrow(matricesOfIncompatibleShape());
	}
	DMatrix<T> d_rowSum, d_x;
	asDmatrix(d_x);
	rowSumM.asDmatrix(d_rowSum, false);
	CuMatrix<T>::rowSum(d_rowSum, d_x);
	rowSumM.invalidateHost();
}

template<typename T> Matrix<T> Matrix<T>::rowSum() const {
	Matrix<T> rowSumM(this->m, 1, false,true);
	rowSum(rowSumM);
	return rowSumM;
}

template<typename T> Matrix<T> Matrix<T>::sqrSubMeans( const Matrix<T>& mus) const {
	Matrix<T> res(CuMatrix<T>::m, CuMatrix<T>::n,false, true);
	checkCudaError(sqrSubMeans(res, mus));
	return res;
}

//
// statics
//

template <typename T> Matrix<T> Matrix<T>::fromFile(const char* fileName) {
	std::string in;
	outln(" Matrix<T>::fromFile(const char* fileName)");
	ifstream ifs(fileName, ios::binary);
	Matrix<T> tmp;
	ifs.read((char *)&tmp, sizeof(tmp));
	outln("read in temp " << tmp.toShortString());
	if(tmp.elements ) {
		outln("erasing stale pointer " << tmp.elements);
		tmp.elements = null;
	}
	if(tmp.d_elements ) {
		outln("erasing stale pointer " << tmp.d_elements );
		tmp.d_elements = null;
	}
	Matrix<T> res(tmp.m, tmp.n,true);
	uint l = res.m * res.n;
	ifs.read((char *)res.elements, l * sizeof(T));
	outln("read " << l << " elements");
	ifs.close();
	res.invalidateDevice();
	return res;
}

template <typename T> std::vector< Matrix<T> > Matrix<T>::fromFileN(const char* fileName) {
	std::string in;
	outln(" Matrix<T>::fromFileN(const char* fileName)");
	ifstream ifs(fileName, ios::binary);
	Matrix<T> tmp;
	std::vector< Matrix<T> > ret;
	do{
		ifs.read((char *)&tmp, sizeof(tmp));
		Matrix<T> res(tmp.m, tmp.n, true);
		uint l = res.m * res.n;
		ifs.read((char *)res.elements, l * sizeof(T));
		outln("read " << l << " elements");
		ret.push_back(res);
		outln("ret now has " << ret.size());

	} while(ifs.peek() != EOF);
	ifs.close();
	return ret;
}

string theTypeStr;
template <typename T> string Matrix<T>::typeStr() {
	if(theTypeStr.length() == 0) {
		theTypeStr = string( typeid(T).name());
	}
	return theTypeStr;
}

template <typename T> void Matrix<T>::init(int maxThreads, int maxBlocks) {
	outln(" Matrix<" << Matrix<T>::typeStr() << ">::init");
	outln(" Matrix<" << Matrix<T>::typeStr() << ">::init creating MemMgr");
	mgr = new MemMgr<T>();
	outln(" Matrix<" << Matrix<T>::typeStr() << ">::init created MemMgr " << mgr);
	mgr->enable();
	outln(" Matrix<" << Matrix<T>::typeStr() << ">::init with MemMgr " << mgr);
	CuMatrix<T>::MaxThreads = maxThreads;
	CuMatrix<T>::MaxBlocks = maxBlocks;
}

template <typename T> void Matrix<T>::cleanup() {
	if(mgr) {
		outln(" Matrix<T>::cleanup mgr " << mgr);
		delete mgr;
		mgr = null;
	}
	cudaDeviceReset();
	outln("\nConstructed " << Constructed << "\nDestructed " << Destructed);

	outln("HDCopied " << HDCopied << ", mem " << b_util::expNotation(MemHdCopied));
	outln("DDCopied " << DDCopied << ", mem " << b_util::expNotation(MemDdCopied));
	outln("DHCopied " << DHCopied << ", mem " << b_util::expNotation(MemDhCopied));
	outln("HHCopied " << HHCopied << ", mem " << b_util::expNotation(MemHhCopied));
}

template<typename T> template<typename FillOp> void Matrix<T>::fillFn(
		FillOp op, Matrix<T> & ret) {

	if(debugFill)outln("fillFn [caller " << b_util::caller() << "]");
	if(ret.d_elements == null) {
		dthrow(noDeviceBuffer());
	}
	CuMatrix<T>::fillFn(op, ret);
	ret.lastMod = mod_device;
	if(debugFill) {
		outln("ret " << ret.toShortString());
	}
}

template<typename T> template<typename FillOp> void Matrix<T>::fillFnCPU(
		FillOp op, Matrix<T>& ret) {

	if(debugFill) outln("fillFnCPU [caller " << b_util::caller() << "]");

	if(ret.elements == null) {
		dthrow(noHostBuffer());
	}
	int i = 0;
	if(!ret.colMajor) {
		if(debugFill) outln("fillFnCPU is rowMajr");
		for(uint row = 0; row < ret.m; row++) {
			if(debugFill) cout << "fillFnCPU i: " << i << " => " << op(i) << endl ;
			for(uint col = 0; col < ret.n; col++) {
				ret.set(row,col, op(i));
				i++;
			}
		}
	}else {
		for(uint col = 0; col < ret.n; col++) {
			for(uint row = 0; row < ret.m ; row++) {
				ret.set(row,col, op(i));
				i++;
			}
		}
	}
	if(debugFill) cout << endl;
	ret.lastMod = mod_host;
}

template <typename T> Matrix<T> Matrix<T>::fill(T t, uint nRows, uint nCols, bool colMajor) {
	constFiller<T> filler;
	filler.value = t;
	Matrix<T> mat(nRows,nCols,false,true);
	mat.colMajor=colMajor;
	fillFn(filler, mat);
	return mat;
}

template <typename T> Matrix<T> Matrix<T>::increasingColumns(T start, int rows, int cols, bool colMajor) {
/*
 * works but o the buffrin
	Matrix<T> b = Matrix<T>::fill(start, rows,1);
	for(int i = start+1; i < start + cols; i++) {
		b = b |= (Matrix<T>::ones(rows,1) * i);
	}
	return b;
*/
	increasingColumnsFiller<T> filler;
	filler.start = start;
	filler.cols = cols;
	Matrix<T> mat(rows,cols,false,true);
	mat.colMajor=colMajor;
	fillFn(filler, mat);
	return mat;

}

template <typename T> Matrix<T> Matrix<T>::increasingRows(T start, int rows, int cols, bool colMajor) {
/*
	Matrix<T> b = Matrix<T>::fill(start, 1,cols);
	for(int i = start+1; i < start + rows; i++) {
		b = b /= (Matrix<T>::ones(1, cols) * i);
	}
	return b;
*/
	increasingRowsFiller<T> filler;
	filler.start = start;
	filler.cols = cols;
	Matrix<T> mat(rows,cols,false,true);
	mat.colMajor=colMajor;
	fillFn(filler, mat);
	return mat;
}

template <typename T> Matrix<T> Matrix<T>::freeform(int cols, const T* vals, ulong count ) {
	int rows =count/cols;
	Matrix<T> mat(rows,cols,true);
	for(ulong i = 0; i < count; i++) {
		mat.set(i, vals[i]);
	}
	return mat;
}

template <typename T> Matrix<T> Matrix<T>::fromScalar(T t, bool colMajor) {
	Matrix<T> mat(1,1,true);
	mat.colMajor = colMajor;
	mat.set(0, t);
	return mat;
}

template <typename T> Matrix<T> Matrix<T>::fill(T t, uintPair dims, bool colMajor) {
	return fill(t, dims.first, dims.second,colMajor);
}

template <typename T> Matrix<T> Matrix<T>::zeros(uint nRows, uint nCols, bool colMajor) {
	return fill(0,nRows,nCols, colMajor);
}

template <typename T> Matrix<T> Matrix<T>::zeros(uintPair dims, bool colMajor) {
	return fill(0,dims.first, dims.second);
}

template <typename T> Matrix<T> Matrix<T>::ones(uint nRows, uint nCols, bool colMajor) {
	return fill(1,nRows,nCols);
}

template <typename T> Matrix<T> Matrix<T>::ones(uintPair dims, bool colMajor) {
	return fill(1, dims.first, dims.second);
}

template <typename T> Matrix<T> Matrix<T>::sin(uint m, uint n, T amplitude, T period, T phase, bool colMajor) {
	sinFiller<T> filler;
	filler.amplitude = amplitude;
	filler.period = period ;
	filler.phase = phase;
	Matrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
	fillFn(filler, mat);
	return mat;
}

template <typename T> Matrix<T> Matrix<T>::cos(uint m, uint n, T amplitude, T period, T phase, bool colMajor) {
	cosFiller<T> filler;
	filler.amplitude = amplitude;
	filler.period = period ;
	filler.phase = phase;
	Matrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
	fillFn(filler, mat);
	return mat;
}

template <typename T> Matrix<T> Matrix<T>::sin(uintPair dims, T amplitude, T period, T phase, bool colMajor) {
	return sin(dims.first, dims.second,amplitude,period,phase,colMajor);
}
template <typename T> Matrix<T> Matrix<T>::cos(uintPair dims, T amplitude, T period, T phase, bool colMajor) {
	return cos(dims.first, dims.second,amplitude,period,phase,colMajor);
}

template <typename T> Matrix<T> Matrix<T>::diagonal(uint dim, T val, bool colMajor) {
	if(dim > MaxDim) {
		dthrow(badDimensions());
	}
	dassert((dim <= MaxDim));
	diagonalFiller<T> filler;
	filler.value = val;
	filler.dim = dim;
	Matrix<T> mat(dim,dim,false,true);
	mat.colMajor= colMajor;
	fillFn(filler, mat);
	return mat;
}

template <typename T> Matrix<T> Matrix<T>::diagonal(uint dim, const T* val, bool colMajor) {
	return diagonal(dim, *val,colMajor);
}

template <typename T> Matrix<T> Matrix<T>::identity(uint dim, bool colMajor) {
	return diagonal(dim, static_cast<T>( 1), colMajor);
}

template <typename T> Matrix<T> Matrix<T>::randn(uint rows, uint cols, T epsilon, bool colMajor) {
	if(colMajor) {
		dthrow(notImplemented());
	}
	Matrix<T> ret = Matrix<T>::zeros(rows,cols);
	ret.syncBuffers();
	DMatrix<T> d_ret;
	ret.asDmatrix(d_ret,false);
	CuMatrix<T>::randn(d_ret, epsilon);
	ret.lastMod = mod_device;
	return ret;
}

template <typename T> Matrix<T> Matrix<T>::randn( const uintPair& dims, float epsilon, bool colMajor) {
	return (randn(dims.first, dims.second, epsilon, colMajor));
}
template <typename T> Matrix<T> Matrix<T>::randn( const uintPair& dims, bool colMajor) {
	return (randn(dims.first, dims.second, colMajor));
}
template <typename T> Matrix<T> Matrix<T>::sequence(T start, uint m, uint n, bool colMajor) {
	sequenceFiller<T> filler;
	filler.phase = start;
	Matrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
	fillFn(filler, mat);
	return mat;
}

template <typename T> Matrix<T> Matrix<T>::seqMod(T start, T mod, uint m, uint n, bool colMajor) {
	seqModFiller<T> filler;
	filler.phase = start;
	filler.mod = mod;
	Matrix<T> mat(m,n,false,true);
	mat.colMajor= colMajor;
	fillFn(filler, mat);
	return mat;
}

template <typename T> Matrix<T> Matrix<T>::fromDmatrix(const DMatrix<T>& mf, bool allocate, bool copy) {
	Matrix<T> ret(mf.m, mf.n, mf.p, allocate);
	ret.d_elements = mf.elements;
	if (allocate && copy) {
		if(debugCopy)outln("matrxD " << &mf << " d-h copying " <<  mf.m * mf.n * sizeof(T) << " from " << mf.elements << " to mat " << &ret << " at " << ret.elements);
		checkCudaError(
				cudaMemcpy(ret.elements, mf.elements, mf.m * mf.n * sizeof(T), cudaMemcpyDeviceToHost));
		DHCopied++;
		MemDhCopied += mf.m * mf.n * sizeof(T);
	}
	ret.lastMod = mod_synced;
	return ret;
}

template <typename T> Matrix<T> Matrix<T>::mapFeature(Matrix<T> m1, Matrix<T> m2, int degree) {
	Matrix<T> res = Matrix<T>::ones(m1.m, 1);
	for (int i = 1; i <= degree; i++) {
		for(int j = 0; j <= i; j++) {
			res = res.rightConcatenate( (m1 ^ (i-j)).hadamardProduct(m2 ^ j));
		}
	}
	return res;
}

template <typename T> Matrix<T> Matrix<T>::reductionBuffer(uint rows) {
	Matrix<T> res = Matrix<T>::zeros(rows,1);
	return res;
}


/////////////////////////////////////////////////////////////////////////
//
// printing
//
/////////////////////////////////////////////////////////////////////////

template<typename T> string Matrix<T>::toString() const {

	stringstream ss1,ss2,ss3,sstrout;
	char value[200];

	if(!this->elements) {
		dthrow(noHostBuffer());
	}

	sstrout << "(" ;
	ss1 << CuMatrix<T>::m;
	sstrout << ss1.str();
	sstrout << "*" ;
	ss2<< CuMatrix<T>::n ;
	sstrout <<  ss2.str();
	sstrout << "*" ;
	ss3 <<  CuMatrix<T>::p;
	sstrout << ss3.str();
	sstrout << ")<" << CuMatrix<T>::size << "> [" << (CuMatrix<T>::colMajor ? "cm]" : "rm]");

	sstrout << " matrix at ";
	sprintf(value, "%p", this);
	sstrout << value;
	sstrout << " h ";
	sprintf(value, "%p", CuMatrix<T>::elements);
	sstrout <<  value;
	sstrout << " d ";
	sprintf(value, "%p", CuMatrix<T>::d_elements);
	sstrout << value;
	sstrout << " {" << b_util::modStr(CuMatrix<T>::lastMod) << "}";
	sstrout << "\n";
	bool header = false;
	if (Matrix<T>::verbose || (CuMatrix<T>::m < MaxRowsDisplayed && CuMatrix<T>::n < MaxColsDisplayed)) {
		for (uint i1 = 0; i1 < CuMatrix<T>::m; i1++) {
			if(!header) {
				sstrout << "-";
				for (uint j1 = 0; j1 < CuMatrix<T>::n; j1++) {
					if(j1 % 10 == 0) {
						sstrout << " " << j1/10;
					}else {
						sstrout << "  ";
					}
					sstrout << " ";
				}
				sstrout << endl;
				header = true;
			}
			sstrout << "[";
			for (uint j1 = 0; j1 < CuMatrix<T>::n; j1++) {

				if(sizeof(T) == 4)
					sprintf(value, "% 2.10g", get(i1,j1) );
				else
					sprintf(value, "% 2.16g", get(i1,j1) );
						//CuMatrix<T>::elements[i1 * CuMatrix<T>::p + j1]);
				sstrout <<  value;
				if (j1 < CuMatrix<T>::n - 1) {
					sstrout <<  " ";
				}
			}
			sstrout << "] ";
			if(i1 % 10 == 0) {
				sstrout  << i1;
			}

			sstrout << "\n";
		}
		if(header) {
			sstrout << "+";
			for (uint j1 = 0; j1 < CuMatrix<T>::n; j1++) {
				if(j1 % 10 == 0) {
					sstrout << " " << j1/10;
				}else {
					sstrout << "  ";
				}
				sstrout << " ";
			}
			sstrout << endl;
			header = false;
		}

	} else {
		for (uint i2 = 0; i2 < MaxRowsDisplayed + 1 && i2 < CuMatrix<T>::m; i2++) {
			if (i2 == MaxRowsDisplayed) {
				sstrout <<  ".\n.\n.\n";
				continue;
			}
			for (uint j2 = 0; j2 < MaxColsDisplayed + 1 && j2 < CuMatrix<T>::n; j2++) {
				if (j2 == MaxColsDisplayed) {
					sstrout << "...";
					continue;
				}
				if(sizeof(T) == 4)
					sprintf(value, "% 2.10g", get(i2,j2));
				else
					sprintf(value, "% 2.16g", get(i2,j2));
						//CuMatrix<T>::elements[i2 * CuMatrix<T>::p + j2]);
				sstrout <<  value;
				if (j2 < CuMatrix<T>::n - 1) {
					sstrout <<  " ";
				}
			}
			sstrout <<  "\n";
		}
		if (CuMatrix<T>::m > MaxRowsDisplayed) {
			for (uint i3 =CuMatrix<T>::m - MaxRowsDisplayed; i3 < CuMatrix<T>::m; i3++) {
				if (CuMatrix<T>::n > MaxColsDisplayed) {
					for (uint j3 = CuMatrix<T>::n - MaxColsDisplayed; j3 < CuMatrix<T>::n; j3++) {
						if (j3 == CuMatrix<T>::n - MaxColsDisplayed) {
							sstrout << "...";
							continue;
						}
						if(sizeof(T) == 4)
							sprintf(value, "% 2.10g", get(i3, j3));
						else
							sprintf(value, "% 2.16g", get(i3,j3));
								//CuMatrix<T>::elements[i3 * CuMatrix<T>::p + j3]);
						sstrout <<  value;
						if (j3 < CuMatrix<T>::n - 1) {
							sstrout << " ";
						}
					}
				} else {
					for (uint j4 = 0; j4 < CuMatrix<T>::n; j4++) {
						if(sizeof(T) == 4)
							sprintf(value, "% 2.10g", get(i3,j4));
						else
							sprintf(value, "% 2.16g", get(i3,j4));
								//CuMatrix<T>::elements[i3 * CuMatrix<T>::p + j4]);
						sstrout << value;

						if (j4 < CuMatrix<T>::n - 1) {
							sstrout << " ";
						}
					}

				}
				sstrout <<  "\n";
			}
		} else { //if(m > 10) -> n > 10
			for (uint i5 = 0; i5 < MaxRowsDisplayed + 1 && i5 < CuMatrix<T>::m; i5++) {

				if (CuMatrix<T>::n > MaxColsDisplayed) {
					for (uint j5 = CuMatrix<T>::n - MaxColsDisplayed; j5 < CuMatrix<T>::n; j5++) {
						if (j5 == CuMatrix<T>::n - MaxColsDisplayed) {
							sstrout << "...";
							continue;
						}
						T t = get(i5,j5);

						if(sizeof(T) == 4)
							sprintf(value, "% 2.10g", t);
						else
							sprintf(value, "% 2.16g", t);
						sstrout << value;
						if (j5 < CuMatrix<T>::n - 1) {
							sstrout <<  " ";
						}
					}
				} else {
					for (uint j4 = 0; j4 < CuMatrix<T>::n; j4++) {
						if(sizeof(T) == 4)
							sprintf(value, "% 2.10g", get(i5,j4));
						else
							sprintf(value, "% 2.16g", get(i5,j4));
						sstrout << value;

						if (j4 < CuMatrix<T>::n - 1) {
							sstrout << " ";
						}
					}
				}

				sstrout << "\n";
			}

		}
	}
	return sstrout.str();
}

template<typename T> string Matrix<T>::pAsRow() {
	poseAsRow();
	string ret = toString();
	unPose();
	return ret;
}
