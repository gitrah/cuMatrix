#pragma once

#include "../debug.h"


extern const char* ANOMDET_SAMPLES_FILE;
extern const char* ANOMDET_SAMPLES2_FILE;
extern const char* REDWINE_SAMPLES_FILE;
extern const char* KMEANS_FILE;
extern const char* SAMPLES_FILE;
extern const char* WEIGHTS_FILE;
extern const char* REDWINE_CSV_FILE;
extern const char* ADULT_FILE;
extern const char*  YEAR_PRED_FILE;
extern const char* NN_DIM_RESULTS_FILE;

extern std::thread::id main_thread_id;

template <typename T> struct Test {
	int testNo = 0;
	virtual int operator()(int argc, const char **argv) const {
		 outln("base test");
		return 0;
	}
	virtual ~Test() {}
};

// io
template <typename T> struct testParseOctave : public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testDim3Octave : public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testPrint : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testLogRegCostFunctionNoRegMapFeature : public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testLinRegCostFunctionNoReg : public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testLinRegCostFunction : public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testCostFunction : public Test<T> {	int operator()(int argc, const char **argv)const; };

// math
template <typename T> struct testCubes : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNextPowerOf2 : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testLargestFactor : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testCountSpanrows : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testMod : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testSign : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testBisection : public Test<T> {	int operator()(int argc, const char **argv)const;};

template <typename T> struct testMatRowMath : public Test<T> {	int operator()(int argc, const char **argv)const;};

template <typename T> struct testAutodot : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testMultLoop : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testSqrMatsMultSmall : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testSqrMatsMult : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testProductKernel3 : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testProductShapes : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testProductShapesTxB : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testProductShapesKernelPtr : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testProductShapesLoop : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testLargeMatProds : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testHugeMatProds : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testHugeMatProds2 : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testHugeMatProds3 : public Test<T> {	int operator()(int argc, const char **argv)const;};

// fill

template <typename T> struct testFillVsLoadRnd: public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testFillNsb : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testFillers : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testFillXvsY : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testXRFill : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testTinyXRFill : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testMemsetFill: public Test<T> {	int operator()(int argc, const char **argv)const;};

// form

template <typename T> struct testTiler : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testDTiler : public Test<T> {	int operator()(int argc, const char **argv)const;};

template <typename T> struct testAnonMatrices : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testTranspose : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testTransposeNneqP : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testTransposeLoop : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testTransposeHuge : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testCat : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testBinCat : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testReshape : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testSubmatrices : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testSubmatrices2 : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testDropFirstAlts : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testAttFreqsSmall : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testAttFreqsMed : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testAttFreqsLarge : public Test<T> {	int operator()(int argc, const char **argv)const;};
//template <typename T> struct testAttFreqs : public Test<T> {	int operator()(int argc, const char **argv)const;};

template <typename T> struct testSigmoidNneqP : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct intestOps : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testBinaryOps : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testSumSqrDiffsLoop : public Test<T> {	int operator()(int argc, const char **argv)const;};

template <typename T> struct testNormalize : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNeural : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNeural2l : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNeural2lCsv : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNeural2lAdult : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNeural2lYrPred : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNeural2lYrPredMulti : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNeural2Loop : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNeural3l : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testLUdecomp : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testFileIO : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testMaxColIdxs : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testAccuracy : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testEqualsEtc : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testSuite : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testCheckNNGradients : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testOps : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNeuralMnistHw : public Test<T> {	int operator()(int argc, const char **argv)const;};

// mem
template <typename T> struct testCudaMemcpy : public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testCudaMemcpyVsCopyKernelVsmMemcpy : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testCopyKernels : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testCudaMemcpyArray : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testCudaMemcpy2D : public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testMemUsage : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testRowCopy : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testCopyVsCopyK : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testClippedRowSubset : public Test<T> { int operator()(int argc, const char **argv) const;};

template <typename T> struct testLastError : public Test<T> { int operator()(int argc, const char **argv) const; };

template <typename T> struct testMemcpyShared: public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testInc: public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testHfreeDalloc: public Test<T> {	int operator()(int argc, const char **argv)const; };

template <typename T> struct testMemset: public Test<T> {	int operator()(int argc, const char **argv)const; };

template <typename T> struct testPackedMat: public Test<T> {	int operator()(int argc, const char **argv)const; };

// reduction
template <typename T> struct testStrmOrNot: public Test<T> { int operator()(int argc, const char **argv)const; };
template <typename T> struct testCount : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testColumnAndRowSum : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testBounds : public Test<T> { int operator()(int argc, const char **argv)const;};

template <typename T> struct testRedux : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testColumnRedux : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testReduceRows : public Test<T> { int operator()(int argc, const char **argv)const;};

template <typename T> struct testCuSet1D : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testCuSet : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testDedeup : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testJaccard : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testQpow : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testMergeSorted : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testOrderedCountedSet : public Test<T> { int operator()(int argc, const char **argv)const;};

template <typename T> struct testShuffle : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testShufflet : public Test<T> {	int operator()(int argc, const char **argv)const;};

template <typename T> struct testSumLoop : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testNneqP : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testAnP : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testMeansFile : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testFeatureMeans : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testMeansLoop : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testAnomDet : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testMVAnomDet : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testKmeans : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testMeansPitch : public Test<T> { int operator()(int argc, const char **argv)const;};


template <typename T> struct testEtoX : public Test<T> { int operator()(int argc, const char **argv)const;};

template <typename T> struct testRedWineScSv : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testCsv : public Test<T> {	int operator()(int argc, const char **argv)const;};

template <typename T> struct testRandSequence : public Test<T> { int operator()(int argc, const char **argv)const;};
template <typename T> struct testRandomizingCopyRows : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testRandAsFnOfSize : public Test<T> {	int operator()(int argc, const char **argv)const;};
template <typename T> struct testMontePi: public Test<T> {	int operator()(int argc, const char **argv)const;};

// multi gpu
template <typename T> struct testMultiGPUMemcpy : public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testMultiGPUMath: public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testMultiGPUTiling: public Test<T> {	int operator()(int argc, const char **argv)const; };
// multi thread / omp
template <typename T> struct testCMap: public Test<T> {	int operator()(int argc, const char **argv)const; };


// opengl
#ifdef  CuMatrix_Enable_Ogl
template <typename T> struct testOglHelloworld: public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testOglAnim0: public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testOglPointAnim: public Test<T> {	int operator()(int argc, const char **argv)const; };

template <typename T> struct testFrag7Csv: public Test<T> {	int operator()(int argc, const char **argv)const;};
#endif // CuMatrix_Enable_Ogl

// ftors
template <typename T> struct testFastInvSqrt: public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testCopyFtor: public Test<T> {	int operator()(int argc, const char **argv)const; };


// recurnels
template <typename T> struct testStrassen: public Test<T> {	int operator()(int argc, const char **argv)const; };

template <typename T> struct testRecCuMatAddition: public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testCdpSum: public Test<T> { int operator()(int argc, const char **argv)const; };
template <typename T> struct testCdpSync1: public Test<T> {	int operator()(int argc, const char **argv)const; };
template <typename T> struct testCdpRedux: public Test<T> {	int operator()(int argc, const char **argv)const; };


template <typename T> struct testPermus: public Test<T> {	int operator()(int argc, const char **argv)const; };
// umem

// amortization
template <typename T> struct testBinaryOpAmort: public Test<T> {	int operator()(int argc, const char **argv)const; };

template <typename T> struct tests {
	static list<string> argsl;
	static int runTest(int argc, const char** argv);
	static T timeTest( const Test<T>& test, int argv, const char** args, int* theResult );
};

