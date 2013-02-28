/*
 * testmatrix.cc
 *
 *  Created on: Sep 17, 2012
 *      Author: reid
 */

#include "../Matrix.h"
#include "../MatrixExceptions.h"
#include "../LuDecomposition.h"
#include "../debug.h"
#include "../util.h"
#include <cstdio>
#include <helper_string.h>
#include "tests.h"

template int testPrint<float>::operator()(int argc, char const ** args) const;
template int testPrint<double>::operator()(int argc, char const ** args) const;
template <typename T> int testPrint<T>::operator()(int argc, const char** args) const {
	outln("testPrint start");
	Matrix<T> seq = Matrix<T>::sequence(1, 100, 1).syncBuffers();
	outln("seq\n" << seq);

	Matrix<T> seqEx;
	Matrix<T> tr = seq.transpose();

	try {
		seqEx = tr.extrude(2).syncBuffers();
		outln("seqEx " << seqEx.toShortString());
		outln(seqEx.toString());
	} catch (const MatrixException& err) {
		outln("got ex " << typeid(err).name());
	}

	outln("seq\n"<<tr.toString());
	Matrix<T> trEx = seqEx.transpose();
	outln("trEx " << trEx.toShortString());
	outln(trEx.syncBuffers().toString());
	outln("testPrint finish");
	return 0;
 }


template int testOps<float>::operator()(int argc, char const ** args) const;
template int testOps<double>::operator()(int argc, char const ** args) const;
template <typename T> int testOps<T>::operator()(int argc, const char** args) const {
	//float els[]= {1., 2, 3, 0, 5, 6, 3, 8, 9};
	outln("testOps start");
	T els[] = { 1, 2, 3., 0, 1, 5, 6, 3, 2, 8, 9, 4.3, 9, 2, .3, 4 };
	Matrix<T> m(els, 4, 4, true);
	outln("have m " << m.toString());
	T det = m.determinant();
	outln("has det " << det);
	assert(abs( abs(det) -50.37) < .001);
	Matrix<T> inv = m.inverse();
	outln("have inv ");
	outln(inv.syncBuffers().toString());

	Matrix<T> mp5 = m + 5;
	T smp5 = mp5.sum();
	outln("have mp5 sum " << smp5 << "\n" << mp5.syncBuffers().toString());
	assert(util<T>::almostEquals(smp5,139.6));
	Matrix<T> mp5t5 = mp5 * 5;
	T smp5t5 = mp5t5.sum();
	outln("have mp5t5 " << smp5t5 << "\n" << mp5t5.syncBuffers().toString());
	assert(util<T>::almostEquals(smp5t5,698));
	Matrix<T> mp5t5t = mp5 * mp5t5;
	outln("have mp5t5t " << mp5t5t.sum() << "\n" << mp5t5t.syncBuffers().toString());

	Matrix<T> prod1 = inv * m;
	outln("after prod1 " << prod1.syncBuffers());
	Matrix<T> prod2 = m * inv;
	outln("after prod2 " << prod2.syncBuffers());
	T s1 = prod1.sum();
	outln("have prod1.sum() " << s1);
	outln(prod1.toString());
	T s2 = prod2.sum();
	outln("have prod2.sum() " << s2);
	outln(prod2.toString());
	assert( 4 - abs(s1 )<.001);
	assert( 4 - abs(s2 )<.001);
	outln("testOps finish");
	return 0;
}

template int testSumSqrDiffsLoop<float>::operator()(int argc, char const ** args) const;
template int testSumSqrDiffsLoop<double>::operator()(int argc, char const ** args) const;
template <typename T> int testSumSqrDiffsLoop<T>::operator()(int argc, const char** args) const {
	outln("testSumSqrDiffsLoop start");
	Matrix<T> m1 = Matrix<T>::ones(1000, 1000);
	outln("m1 " << &m1 << "\n");
	Matrix<T> m2 = Matrix<T>::ones(1000, 1000) * 2;
	outln("made mats m1 " << &m1 << " << m2 " << &m2 << "\n\n");
	int blocks;
	int threads;
	uint n = m1.m * m1.n;
	m1.getReductionExecContext(blocks, threads, n);
	outln("blocks " << blocks << "\n");
	Matrix<T> buffer = Matrix<T>::reductionBuffer(blocks);
	outln("m1 " << m1.toShortString());
	outln("m2 " << m2.toShortString());
	outln("buffer" << buffer.toShortString());
	outln("m1.sumSqrDiff(m2) " << m1.sumSqrDiff(m2));
	T s = 0;
	clock_t lastTime = clock();
	CuTimer timer;
	timer.start();
	int count = b_util::getCount(argc,args,10000);
	for (int i = 0; i < count; i++) {
		s += m1.sumSqrDiff(buffer, m2 );
	}
	float exeTime = timer.stop();
	double delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("s " << s << " took " << delta << " secs");
	outln("flow of " << m1.flow(count, 2, exeTime) << "GB/s");
	outln("m1.d " << m1.d_elements);
	outln("m2.d " << m2.d_elements);
	outln("buff.d " << buffer.d_elements);
	outln("testSumSqrDiffsLoop finish");
	return 0;
}

template int testBinaryOps<float>::operator()(int argc, char const ** args) const;
template int testBinaryOps<double>::operator()(int argc, char const ** args) const;
template <typename T> int testBinaryOps<T>::operator()(int argc, const char** args) const {
	outln("testBinaryOps start");
	Matrix<T>::init(256, 64);
	Matrix<T> m1 = Matrix<T>::ones(1000, 1000) * 3;
	outln("m1 " << &m1 << "\n");
	Matrix<T> m2 = Matrix<T>::ones(1000, 1000) * 2;
	outln("made mats m1 " << &m1 << " << m2 " << &m2 << "\n\n");
	Matrix<T> m3 = m1 % m2;
	outln("m3 = hadamard sum " << m3.sum());

	Matrix<T> eye1000 = Matrix<T>::identity(1000);
	Matrix<T> mpi = m1 + 5 * eye1000;
	outln("mpi.sum " << mpi.sum());

	Matrix<T> v1 = Matrix<T>::sequence(0, 10250, 1);
	Matrix<T> v2 = Matrix<T>::ones(10250, 1);
	Matrix<T> v3  = v1 + v2;
	T sv1 = v1.sum(), sv2 = v2.sum(), sv3 = v3.sum();
	outln("sv1 " << sv1 << ", sv2 " << sv2 << ", sv3 " << sv3);
	assert(util<T>::almostEquals(sv3, sv1 + sv2));
	outln("v3 " << v3.toShortString() << "; " << v3.sum());

	Matrix<T> vb1 = Matrix<T>::sequence(0, 1, 10250);
	Matrix<T> vb2 = Matrix<T>::ones(1,10250);
	Matrix<T> vb3  = vb1 + vb2;
	T sbv1 = vb1.sum(), sbv2 = vb2.sum(), sbv3 = vb3.sum();
	outln("sbv1 " << sbv1 << ", sbv2 " << sbv2 << ", sbv3 " << sbv3);
	assert(util<T>::almostEquals(sbv3, sbv1 + sbv2));
	outln("vb3 " << vb3.toShortString() << "; " << vb3.sum());
	return 0;
}


template int testReassign<float>::operator()(int argc, char const ** args) const;
template int testReassign<double>::operator()(int argc, char const ** args) const;
template <typename T> int testReassign<T>::operator()(int argc, const char** args) const {
	Matrix<T> s = Matrix<T>::ones(5, 1).syncBuffers();
	outln(s.toString());
	s = Matrix<T>::zeros(5, 1).syncBuffers();
	outln(s.toString());

	return 0;
}

template int testLUdecomp<float>::operator()(int argc, char const ** args) const;
template int testLUdecomp<double>::operator()(int argc, char const ** args) const;
template <typename T> int testLUdecomp<T>::operator()(int argc, const char** args) const {
	// 5 0 2 3 4 ; 5 4 0 2 4 ; 2 3 4 0 5; 1 2 -4 3 5; 2 0 2 0 4
	T vals[]= {5, 0, 2, 3., 4, 5, 4, 0, 2, 4, 2, 3, 4, 0, 5, 1, 2, -4, 3, 5, 2, 0, 2, 0, 4};
	outln("count " << Matrix<T>::sizeOfArray( vals));
	Matrix<T> mf = Matrix<T>::freeform(5,vals, Matrix<T>::sizeOfArray( vals)).syncBuffers();
	outln("mf " << mf << "\ndet " << mf.determinant());
	Matrix<T>imf = mf.inverse().syncBuffers();
	outln("imf " << imf);
	outln("imf * mf " << (imf * mf).syncBuffers());

	Matrix<T> i5 = Matrix<T>::identity(5).syncBuffers();
	LUDecomposition<T> lu5(mf);
	Matrix<T> inv5 = lu5.solve(i5);
	outln("inv5 " << inv5);
	outln("inv5 * mf " << (inv5 * mf).syncBuffers());


	Matrix<T> m400 = Matrix<T>::randn(400, 400).syncBuffers();
	outln("m400 " << m400);
	Matrix<T> i400 = Matrix<T>::identity(400).syncBuffers();
	outln("i400 " << i400);
	Matrix<T> iprod = i400 * m400;
	Matrix<T> prodi = m400 * i400;
	assert( ! (m400 - iprod).sum());
	assert( (prodi - iprod).zeroQ());
	assert( m400.almostEq(iprod ));
	//outln("i400 " << i400);
	LUDecomposition<T> lu400(m400);
	Matrix<T> inv400 = lu400.solve(i400).syncBuffers();
	outln("inv " << inv400.toShortString());
	outln(inv400.toString());
	Matrix<T> i400p = inv400 * m400;
	outln(i400p.syncBuffers().toString());
	assert(util<T>::almostEquals(i400p.sum(), 400));
	return 0;
}

template int testFileIO<float>::operator()(int argc, char const ** args) const;
template int testFileIO<double>::operator()(int argc, char const ** args) const;
template <typename T> int testFileIO<T>::operator()(int argc, const char** args) const {
	const char* fileName = "m400.mat";
	const char* fileNameN = "thetas.mat";
	FILE * pFile;
	pFile = fopen(fileName, "r");
	if (pFile) {
		fclose(pFile);
		outln("removing old " << fileName);
		if (remove(fileName) != 0)
			perror("Error deleting file ");
		else
			outln( fileName << " successfully deleted");

	}
	pFile = fopen(fileNameN, "r");
	if (pFile) {
		fclose(pFile);
		outln("removing old " << fileNameN);
		if (remove(fileNameN) != 0)
			perror("Error deleting file");
		else
			outln( fileNameN << " successfully deleted");
	}
	Matrix<T> m400 = Matrix<T>::sin(400, 400).syncBuffers();
	std::ofstream ofs(fileName, ios::binary);
	m400.toStream(ofs);
	ofs.close();
	pFile = fopen(fileName, "r");
	assert(pFile);
	outln("wrote " << fileName);
	fclose(pFile);
	Matrix<T> m400prime = Matrix<T>::fromFile("m400.mat");
	Matrix<T> m40 = Matrix<T>::sin(40, 40).syncBuffers();
	outln("last element of m40 " << m40.get(39,39));
	Matrix<T> m20 = Matrix<T>::sin(20, 20).syncBuffers();
	outln("last element of m20 " << m20.get(19,19));
	Matrix<T> m60 = Matrix<T>::sin(60, 60).syncBuffers();
	outln("last element of m60 " << m60.get(59,59));
	ofstream ofsN(fileNameN, ios::binary);
	m40.toStream(ofsN);
	m20.toStream(ofsN);
	m60.toStream(ofsN);
	ofsN.close();
	pFile = fopen(fileNameN, "r");
	assert(pFile);
	outln("wrote " << fileNameN);
	fclose(pFile);
	std::vector<Matrix<T> > list = Matrix<T>::fromFileN(fileNameN);
	//list.clear();
	remove(fileName);
	remove(fileNameN);
	return 0;
}



template int testAccuracy<float>::operator()(int argc, char const ** args) const;
template int testAccuracy<double>::operator()(int argc, char const ** args) const;
template <typename T> int testAccuracy<T>::operator()(int argc, const char** args) const {
	Matrix<T> a = Matrix<T>::ones(1000,1);
	outln("a " << a.toShortString());
	Matrix<T> b = Matrix<T>::ones(1000,1);
	Matrix<T> c = Matrix<T>::ones(1000,1);
	outln("c " << c.toShortString());
	c.syncBuffers();
	outln("c " << c.toShortString());
	MemMgr<T>::checkValid(c.elements);
	MemMgr<T>::checkValid(c.d_elements);
	for(int i = 0; i < 1000; i++) {
		if((1. * rand()) / (RAND_MAX - 1) > .7){
			c.set(i, 0);
		}
	}
	c.syncBuffers();
	outln("accuracy a-b " << a.accuracy(b));
	outln("accuracy a-c " << a.accuracy(c));
	return 0;
}


template int testSuite<float>::operator()(int argc, char const ** args) const;
template int testSuite<double>::operator()(int argc, char const ** args) const;
template <typename T> int testSuite<T>::operator()(int argc, char const ** args) const {
	outln("in testSuite");
	int ret,status;
	tests<T>::timeTest(testAccuracy<T>(), argc, args, &ret);
    status = ret;
    tests<T>::timeTest(testFileIO<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testLUdecomp<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testNeural2l<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testSubmatrices<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testAutodot<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testCat<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testEqualsEtc<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testOps<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testPrint<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testProductShapes<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testReassign<T>(),argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testReshape<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testTranspose<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testTransposeLoop<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testMaxColIdxs<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testMemUsage<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testMultLoop<T>(), argc, args, &ret);
    status += ret;
    tests<T>::timeTest(testSumSqrDiffsLoop<T>(), argc, args, &ret);
    status += ret;
	return ret;
}

#include "tests.cc"
