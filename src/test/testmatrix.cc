/*
 * testmatrix.cc
 *
 *  Created on: Sep 17, 2012
 *      Author: reid
 */

#include "../CuMatrix.h"
#include "../MatrixExceptions.h"
#include "../LuDecomposition.h"
#include "../debug.h"
#include "../util.h"
#include "../caps.h"
#include <cstdio>
#include <helper_string.h>
#include "tests.h"

template int testPrint<float>::operator()(int argc, const char **argv) const;
template int testPrint<double>::operator()(int argc, const char **argv) const;
template <typename T> int testPrint<T>::operator()(int argc, const char** argv) const {
	outln("testPrint start");
	CuMatrix<T> seq = CuMatrix<T>::sequence(1, 100, 1).syncBuffers();
	outln("seq\n" << seq);

	CuMatrix<T> seqEx;
	CuMatrix<T> tr = seq.transpose();

	try {
		seqEx = tr.extrude(2).syncBuffers();
		outln("seqEx " << seqEx.toShortString());
		outln(seqEx.toString());
	} catch (const MatrixException& err) {
		outln("got ex " << typeid(err).name());
	}

	outln("seq\n"<<tr.toString());
	CuMatrix<T> trEx = seqEx.transpose();
	outln("trEx " << trEx.toShortString());
	outln(trEx.syncBuffers().toString());
	outln("testPrint finish");
	return 0;
 }


template <typename T> int testOps<T>::operator()(int argc, const char **argv) const {
	//float els[]= {1., 2, 3, 0, 5, 6, 3, 8, 9};
	outln("testOps start");
	T els[] = { 1, 2, 3., 0, 1, 5, 6, 3, 2, 8, 9, 4.3, 9, 2, .3, 4 };
	CuMatrix<T> m(els, 4, 4, true);
	outln("have m " << m.toString());
	T det = m.determinant();
	outln("has det " << det);
	assert(abs( abs(det) -50.37) < .001);
	CuMatrix<T> inv = m.inverse();
	outln("have inv ");
	outln(inv.syncBuffers().toString());

	CuMatrix<T> mp5 = m + 5;
	T smp5 = mp5.sum();
	outln("have mp5 sum " << smp5 << "\n" << mp5.syncBuffers().toString());
	assert(util<T>::almostEquals(smp5,139.6));
	CuMatrix<T> mp5t5 = mp5 * 5;
	T smp5t5 = mp5t5.sum();
	outln("have mp5t5 " << smp5t5 << "\n" << mp5t5.syncBuffers().toString());
	assert(util<T>::almostEquals(smp5t5,698));
	CuMatrix<T> mp5t5t = mp5 * mp5t5;
	outln("have mp5t5t " << mp5t5t.sum() << "\n" << mp5t5t.syncBuffers().toString());

	CuMatrix<T> prod1 = inv * m;
	outln("after prod1 " << prod1.syncBuffers());
	CuMatrix<T> prod2 = m * inv;
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
template int testOps<float>::operator()(int argc, const char **argv) const;
template int testOps<double>::operator()(int argc, const char **argv) const;

template int testSumSqrDiffsLoop<float>::operator()(int argc, const char **argv) const;
template int testSumSqrDiffsLoop<double>::operator()(int argc, const char **argv) const;
template <typename T> int testSumSqrDiffsLoop<T>::operator()(int argc, const char **argv) const {
	outln("testSumSqrDiffsLoop start");
	CuMatrix<T> m1 = CuMatrix<T>::ones(1000, 1000);
	outln("m1 " << &m1 << "\n");
	CuMatrix<T> m2 = CuMatrix<T>::ones(1000, 1000) * 2;
	outln("made mats m1 " << &m1 << " << m2 " << &m2 << "\n\n");
	int blocks;
	int threads;
	int n = m1.m * m1.n;
	getReductionExecContext(blocks, threads, n);
	outln("blocks " << blocks << "\n");
	CuMatrix<T> buffer = CuMatrix<T>::reductionBuffer(blocks);
	outln("m1 " << m1.toShortString());
	outln("m2 " << m2.toShortString());
	outln("buffer" << buffer.toShortString());
	outln("m1.sumSqrDiff(m2) " << m1.sumSqrDiff(m2));
	T s = 0;
	clock_t lastTime = clock();
	CuTimer timer;
	timer.start();
	int count = b_util::getCount(argc,argv,10000);
	for (int i = 0; i < count; i++) {
		s += m1.sumSqrDiff(buffer, m2 );
	}
	float exeTime = timer.stop();
	double delta = b_util::diffclock(clock(), lastTime) / 1000;
	outln("s " << s << " took " << exeTime << " ms");
	outln("delta " << delta);
	outln("flow of " << m1.flow(count, 2, exeTime) << "GB/s");
	outln("m1.d " << m1.tiler.currBuffer());
	outln("m2.d " << m2.tiler.currBuffer());
	outln("buff.d " << buffer.tiler.currBuffer());
	outln("testSumSqrDiffsLoop finish");
	return 0;
}

template int testBinaryOps<float>::operator()(int argc, const char **argv) const;
template int testBinaryOps<double>::operator()(int argc, const char **argv) const;
template <typename T> int testBinaryOps<T>::operator()(int argc, const char **argv) const {
	outln("testBinaryOps start");
	CuMatrix<T>::initMemMgrForType(256, 64);
	CuMatrix<T> m1 = CuMatrix<T>::ones(1000, 1000) * 3;
	outln("m1 " << &m1 << "\n");
	CuMatrix<T> m2 = CuMatrix<T>::ones(1000, 1000) * 2;
	outln("made mats m1 " << &m1 << " << m2 " << &m2 << "\n\n");
	CuMatrix<T> m3 = m1 % m2;
	outln("m3 = hadamard sum " << m3.sum());

	CuMatrix<T> eye1000 = CuMatrix<T>::identity(1000);
	CuMatrix<T> mpi = m1 + 5 * eye1000;
	outln("mpi.sum " << mpi.sum());

	CuMatrix<T> v1 = CuMatrix<T>::sequence(0, 10250, 1);
	CuMatrix<T> v2 = CuMatrix<T>::ones(10250, 1);
	CuMatrix<T> v3  = v1 + v2;
	T sv1 = v1.sum(), sv2 = v2.sum(), sv3 = v3.sum();
	outln("sv1 " << sv1 << ", sv2 " << sv2 << ", sv3 " << sv3);
	assert(util<T>::almostEquals(sv3, sv1 + sv2));
	outln("v3 " << v3.toShortString() << "; " << v3.sum());

	CuMatrix<T> vb1 = CuMatrix<T>::sequence(0, 1, 10250);
	CuMatrix<T> vb2 = CuMatrix<T>::ones(1,10250);
	CuMatrix<T> vb3  = vb1 + vb2;
	T sbv1 = vb1.sum(), sbv2 = vb2.sum(), sbv3 = vb3.sum();
	outln("sbv1 " << sbv1 << ", sbv2 " << sbv2 << ", sbv3 " << sbv3);
	assert(util<T>::almostEquals(sbv3, sbv1 + sbv2));
	outln("vb3 " << vb3.toShortString() << "; " << vb3.sum());
	return 0;
}


template int testLUdecomp<float>::operator()(int argc, const char **argv) const;
template int testLUdecomp<double>::operator()(int argc, const char **argv) const;
template int testLUdecomp<ulong>::operator()(int argc, const char **argv) const;
template <typename T> int testLUdecomp<T>::operator()(int argc, const char **argv) const {
	// 5 0 2 3 4 ; 5 4 0 2 4 ; 2 3 4 0 5; 1 2 -4 3 5; 2 0 2 0 4
	T vals[]= { 5, 0, 2, (T)3., 4, 5, 4, 0, 2, 4, 2, 3, 4, 0, 5, 1, 2, (T)-4, 3, 5, 2, 0, 2, 0, 4};
	//outln("count " << CuMatrix<T>::sizeOfArray( vals));
	dassert(CuMatrix<T>::sizeOfArray( vals) == 25);
	CuMatrix<T> mf = CuMatrix<T>::freeform(5,vals, CuMatrix<T>::sizeOfArray( vals)).syncBuffers();
	dassert(mf.determinant() == 1050);
	//outln("mf " << mf << "\ndet " << mf.determinant());

	CuMatrix<T> i5 = CuMatrix<T>::identity(5).syncBuffers();
	CuMatrix<T>imf = mf.inverse().syncBuffers();
	//outln("imf " << imf);
	//outln("imf * mf " << (imf * mf).syncBuffers());
	outln("i5.sumSqrDiff(imf * mf) " << i5.sumSqrDiff(imf * mf));
	outln(" util<T>::epsilon() " <<  util<T>::epsilon());
	outln("i5.sumSqrDiff(imf * mf) < util<T>::epsilon() " << (i5.sumSqrDiff(imf * mf) < util<T>::epsilon()));
	dassert(  i5.sumSqrDiff(imf * mf) < util<T>::epsilon());

	LUDecomposition<T> lu5(mf);
	CuMatrix<T> luInv5 = lu5.solve(i5);
	dassert( (luInv5 * mf).sumSqrDiff(i5) < util<T>::epsilon());


	CuMatrix<T> m400 = CuMatrix<T>::randn(400, 400).syncBuffers();
	//outln("m400 " << m400);
	CuMatrix<T> i400 = CuMatrix<T>::identity(400).syncBuffers();
	//outln("i400 " << i400);
	CuMatrix<T> iprod = i400 * m400;
	CuMatrix<T> prodi = m400 * i400;
	dassert( ! (m400 - iprod).sum());
	dassert( (prodi - iprod).zeroQ());
	dassert( m400.almostEq(iprod ));
	//outln("i400 " << i400);
	LUDecomposition<T> lu400(m400);
	CuMatrix<T> luInv400 = lu400.solve(i400).syncBuffers();
	//outln("inv " << inv400.toShortString());
	//outln(luInv400.toString());
	CuMatrix<T> i400p = luInv400 * m400;
	outln(i400p.syncBuffers().toString());
	assert(util<T>::almostEquals(i400p.sum(), 400));
	return 0;
}

template int testFileIO<float>::operator()(int argc, const char **argv) const;
template int testFileIO<double>::operator()(int argc, const char **argv) const;
template <typename T> int testFileIO<T>::operator()(int argc, const char **argv) const {
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
	CuMatrix<T> m400 = CuMatrix<T>::sin(400, 400).syncBuffers();
	m400.toFile(fileName);
	pFile = fopen(fileName, "r");
	assert(pFile);
	outln("wrote " << fileName);
	fclose(pFile);
	CuMatrix<T> m400prime = CuMatrix<T>::fromFile("m400.mat");
	CuMatrix<T> m40 = CuMatrix<T>::sin(40, 40).syncBuffers();
	outln("last element of m40 " << m40.get(39,39));
	CuMatrix<T> m20 = CuMatrix<T>::sin(20, 20).syncBuffers();
	outln("last element of m20 " << m20.get(19,19));
	CuMatrix<T> m60 = CuMatrix<T>::sin(60, 60).syncBuffers();
	outln("last element of m60 " << m60.get(59,59));
	m40.toFile(fileNameN);
	m20.toFile(fileNameN);
	m60.toFile(fileNameN);
	pFile = fopen(fileNameN, "r");
	assert(pFile);
	outln("wrote " << fileNameN);
	fclose(pFile);
	std::vector<CuMatrix<T> > list = CuMatrix<T>::fromFileN(fileNameN);
	//list.clear();
	//util<T>::cudaFreeVector(list);
	remove(fileName);
	remove(fileNameN);
	return 0;
}



template int testAccuracy<float>::operator()(int argc, const char **argv) const;
template int testAccuracy<double>::operator()(int argc, const char **argv) const;
template <typename T> int testAccuracy<T>::operator()(int argc, const char **argv) const {
	CuMatrix<T> a = CuMatrix<T>::ones(1000,1);
	outln("a " << a.toShortString());
	CuMatrix<T> b = CuMatrix<T>::ones(1000,1);
	CuMatrix<T> c = CuMatrix<T>::ones(1000,1);
	outln("c " << c.toShortString());
	c.syncBuffers();
	outln("c " << c.toShortString());
	MemMgr<T>::checkValid(c.elements);
	MemMgr<T>::checkValid(c.tiler.currBuffer());
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


template int testSuite<float>::operator()(int argc, const char **argv) const;
template int testSuite<double>::operator()(int argc, const char **argv) const;
template <typename T> int testSuite<T>::operator()(int argc, const char **argv) const {
	outln("in testSuite");
	int ret,status;
	tests<T>::timeTest(testAccuracy<T>(), argc, argv, &ret);
    status = ret;
    tests<T>::timeTest(testFileIO<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testLUdecomp<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testNeural2l<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testSubmatrices<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testAutodot<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testCat<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testEqualsEtc<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testOps<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testPrint<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testProductShapes<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testReshape<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testTranspose<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testTransposeLoop<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testMaxColIdxs<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testMemUsage<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testMultLoop<T>(), argc, argv, &ret);
    status += ret;
    tests<T>::timeTest(testSumSqrDiffsLoop<T>(), argc, argv, &ret);
    status += ret;
	return ret;
}

#include "tests.cc"
