#ifndef TESTS_H_
#define TESTS_H_
#include "../Matrix.h"
template <typename T> struct Test {
	 virtual int operator()(int argc, const char** args) const {
		 outln("base test");
		return 0;
	}
};


template <typename T> struct testParseOctave : public Test<T> {	int operator()(int argc, const char** args)const; };
template <typename T> struct testCostFunctionNoReg0 : public Test<T> {	int operator()(int argc, const char** args)const; };
template <typename T> struct testCostFunction : public Test<T> {	int operator()(int argc, const char** args)const; };
template <typename T> struct testCudaMemcpy : public Test<T> {	int operator()(int argc, const char** args)const; };
template <typename T> struct testCudaMemcpyVsCopyKernelVsmMemcpy : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testCopyKernels : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testCudaMemcpyArray : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testMemUsage : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testRowCopy : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testTranspose : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testFillNsb : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testAutodot : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testMultLoop : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testProductShapes : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testProductShapesLoop : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testFillers : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testCat : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testBinCat : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testFillXvsY : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testReshape : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testTransposeNneqP : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testTransposeLoop : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testSubmatrices : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testSubmatrices2 : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testDropFirstAlts : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testSigmoidNneqP : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testPrint : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testOps : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testBinaryOps : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testSumSqrDiffsLoop : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testReassign : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testNeural2l : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testNeural3l : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testLUdecomp : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testFileIO : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testMaxColIdxs : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testAccuracy : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testEqualsEtc : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testColumnAndRowSum : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testSuite : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testCheckNNGradients : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testRedux : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testSumLoop : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testNneqP : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testAnP : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testAnomDet : public Test<T> {	int operator()(int argc, const char** args)const;};

template <typename T> struct testRandSequence : public Test<T> {	int operator()(int argc, const char** args)const;};
template <typename T> struct testShuffleCopyRows : public Test<T> {	int operator()(int argc, const char** args)const;};

template <typename T> struct tests {
	static int runTest(int argc, const char** argv);
	static T timeTest( const Test<T>& test, int argv, const char** args, int* theResult );
};
#endif
