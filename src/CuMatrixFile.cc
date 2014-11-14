/*
 * CuCuMatrixFile.cc
 *
 *  Created on: Dec 27, 2013
 *      Author: reid
 */

#include "CuMatrix.h"

template<typename T> __host__ void CuMatrix<T>::toFile( const char* fileName ) const {
	toFile(fileName, *this);
}
template void CuMatrix<float>::toFile(char const*) const;
template void CuMatrix<double>::toFile(char const*) const;

template<typename T> void CuMatrix<T>::toFile(const char* fileName, const CuMatrix<T>& o )  {
	ofstream ofs(fileName, ios::binary);
	const CuMatrix<T>* ptr = &o;
	ofs.write((char *)ptr, sizeof(o));
	if(o.elements) {
		uint l = o.m * o.n;
		ofs.write((char*)o.elements, l*sizeof(T));
		outln("wrote " << l << " elements");
	}
}


template <typename T> CuMatrix<T> CuMatrix<T>::fromFile(const char* fileName) {
	string in;
	outln(" CuMatrix<T>::fromFile(const char* fileName)");
	ifstream ifs(fileName, ios::binary);
	CuMatrix<T> tmp;
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
	CuMatrix<T> res(tmp.m, tmp.n,true,false);
	uint l = res.m * res.n;
	ifs.read((char *)res.elements, l * sizeof(T));
	outln("read " << l << " elements");
	ifs.close();
	res.invalidateDevice();
	return res;
}
template CuMatrix<float> CuMatrix<float>::fromFile(char const*);
template CuMatrix<double> CuMatrix<double>::fromFile(char const*);

template <typename T> vector< CuMatrix<T> > CuMatrix<T>::fromFileN(const char* fileName) {
	string in;
	outln(" CuMatrix<T>::fromFileN(const char* fileName)");
	ifstream ifs(fileName, ios::binary);
	CuMatrix<T> tmp;
	vector< CuMatrix<T> > ret;
	do{
		ifs.read((char *)&tmp, sizeof(tmp));
		CuMatrix<T> res(tmp.m, tmp.n, true,false);
		uint l = res.m * res.n;
		ifs.read((char *)res.elements, l * sizeof(T));
		outln("read " << l << " elements");
		ret.push_back(res);
		outln("ret now has " << ret.size());

	} while(ifs.peek() != EOF);
	ifs.close();
	return ret;
}

template vector< CuMatrix<float> > CuMatrix<float>::fromFileN(char const*);
template vector< CuMatrix<double> > CuMatrix<double>::fromFileN(char const*);
