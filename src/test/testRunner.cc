/*
 * testRunner.cc
 *
 *  Created on: Sep 18, 2012
 *      Author: reid
 */

#include "../CuMatrix.h"
#include "../Kernels.h"
#include "../caps.h"
#include <execinfo.h>
#include <sys/wait.h>
#include "tests.h"
#include "../ogl/AccelEvents.h"
#include <pthread.h>
#include "../CuFunctor.h"

extern template class CuMatrix<float> ;
extern template class CuMatrix<double> ;
extern template class CuMatrix<long> ;
extern template class CuMatrix<ulong> ;
extern template class CuMatrix<int> ;
extern template class CuMatrix<uint> ;

namespace test {
	list<string> argsl;
}

template<typename T> T tests<T>::timeTest(const Test<T>& test, int argc,
		const char** argv, int* theResult) {
	int dev = ExecCaps::currDev();
	outln("freest device " << b_util::getDeviceThatIs(gcFreeest));
	outln("coolest device " << b_util::getDeviceThatIs(gcCoolest));
	CuTimer timer;
	sigmoidUnaryOp<T> sig;
	outln(
			"tests<T>::timeTest on device " << dev << " starting argc " << argc <<" argv " << argv);
	//auto fnctrPtr = &sigmoidUnaryOp<T>::operator();
	typedef T (sigmoidUnaryOp<T>::*memfunc)(T) const;

	memfunc fnctrPtr = &sigmoidUnaryOp<T>::operator();
	auto sigPtr = &sig;
	outln("awaaay &sigmoidUnaryOp<T>::operator() type name " << b_util::unmangl(typeid(fnctrPtr).name()));
	outln("awaaay &sig.operator() " << b_util::unmangl(typeid(sigPtr).name()));
	double (sigmoidUnaryOp<double>::*fncPtrExplicit)(double) const = null;
	T (sigmoidUnaryOp<T>::*fncPtrExplicitT)(
			T) const =&sigmoidUnaryOp<T>::operator();
	//fncPtrExplicitT = fncPtrExplicit;
	//T (unaryOpF<T>::*fncPtrExplicitTBase)(T const&) const  = &sigmoidUnaryOp<T>::operator();
	//fncPtrExplicitTBase = fncPtrExplicit;
	outln("(sig.*fncPtrExplicitT)(5) "<< (sig.*fncPtrExplicitT)(5));
	stringstream ss;
	ss << b_util::cardinalToOrdinal(test.testNo) << " test:  ";
	ss << b_util::unmangl(typeid(test).name()) ;
	string terst = ss.str();
	outln(
			"\n\n\n\n\nawaaay we geaux starting " << terst << "\n\n\n\n\n");
	timer.start();
	int res = test(argc, argv);
	dassert(res == 0);
	if (theResult) {
		*theResult = res;
	}
	checkCudaErrors(cudaSetDevice(dev));
	float dur = timer.stop();
	string type = sizeof(T) == 8 ? "<double>" : "<float>";
	//outln( typeid(tst).name() << type.c_str() << " took " << dur << "s");
	CuMatrix<T>::ZeroMatrix.getMgr().dumpLeftovers();
	outln(
				"\n\n\n\n\nbaaaack we come, " << terst << " taking " << debug::fromMillis((double)dur) << "\n\n\n\n\n");
	return dur;
}
template<typename T> void constFillKrnleL();

std::thread::id main_thread_id = std::this_thread::get_id();

void remarg(list<string>& targsl, const char* arg) {
	auto argi = targsl.begin();
	string argstr = arg[0]=='-' ? string(arg) : string("-").append(arg);
	while( argi != targsl.end() ) {
		if(( *argi).find(argstr) != string::npos)
			targsl.erase(argi++);
		else
			argi++;
	}
}


template<typename T> uint parseArgs(
		int&idev,
		int& stopAt,
		char *&deviceChoice,
		char *&stopAtChoice,
		char *&debugChoice,
		char *&testChoice,
		char *&kernelChoice,
		char *&coolChoice,
		int argc, const char** argv) {
	static bool printedSizes = false;
	if (!printedSizes) {
		printObjSizes<T>();
		printedSizes = true;
	}
	outln("tests<T>::runTest starting argc " << argc <<" argv " << argv);
	if (argv)
		for (int i = 0; i < argc; i++) {
			outln("tests<T>::runTest arg " <<i << ": " << argv[i]);
		}

	// CUDA events
	getCmdLineArgumentString(argc, (const char **) argv, "dbg", &debugChoice);
	remarg(test::argsl,"dbg");

	getCmdLineArgumentString(argc, (const char **) argv, "test", &testChoice);
	remarg(test::argsl,"test");
	getCmdLineArgumentString(argc, (const char **) argv, "dev", &deviceChoice);
	remarg(test::argsl,"dev");

	getCmdLineArgumentString(argc, (const char **) argv, "stopAt", &stopAtChoice);
	remarg(test::argsl,"stopAt");

	getCmdLineArgumentString(argc, (const char **) argv, "krn", &kernelChoice);
	remarg(test::argsl,"krn");

	getCmdLineArgumentString(argc, (const char **) argv, "cool", &coolChoice);
	remarg(test::argsl,"cool");

	outln("starting argc " << argc);

#ifdef CuMatrix_UseCublas
	g_useCublas = checkCmdLineFlag(argc,argv, "blas");
	remarg(test::argsl,"blas");

	outln("g_useCublas  "<< tOrF(g_useCublas ));
	if(g_useCublas) {
		chblerr( cublasCreate(&g_handle));// todo determine time/space effects of making this optional
		//flprintf("cublasCreate success");
	}
#endif

	if( test::argsl.size() > 0) {
		for( auto arg0 : test::argsl) {
			outln("unrecognized option " + arg0);
		}
		return (uint) -1;
	}

	// check for options of debug output per function
	uint localDbgFlags = 0;
	if (debugChoice) {
		string debug(debugChoice);
		cout << "\ndebug choices: ";
		if (debug.find(allChoice) != string::npos) {
			cout << " ALL";
			localDbgFlags |= debugMem;
			localDbgFlags |= debugFtor;
			localDbgFlags |= debugCopy;
			localDbgFlags |= debugCopyDh;
			localDbgFlags |= debugMatProd;
			localDbgFlags |= debugCons;
			localDbgFlags |= debugDestr;
			localDbgFlags |= debugRefcount;
			localDbgFlags |= debugVerbose;
			localDbgFlags |= debugNn;
			localDbgFlags |= debugCg;
			localDbgFlags |= debugTxp;
			localDbgFlags |= debugExec;
			localDbgFlags |= debugMultGPU;
			localDbgFlags |= debugFill;
			localDbgFlags |= debugAnomDet;
			localDbgFlags |= debugRedux;
		} else {
			if (debug.find(anomChoice) != string::npos) {
				cout << " " << anomChoice;
				localDbgFlags |= debugAnomDet;
			}
			if (debug.find(memChoice) != string::npos) {
				cout << " " << memChoice;
				localDbgFlags |= debugMem;
			}
			if (debug.find(debugCheckValidChoice) != string::npos) {
				cout << " " << debugCheckValidChoice;
				localDbgFlags |= debugCheckValid;
			}
			if (debug.find(ftorChoice) != string::npos) {
				cout << " " << ftorChoice;
				localDbgFlags |= debugFtor;
			}
			if (debug.find(copyChoice) != string::npos) {
				cout << " " << copyChoice;
				localDbgFlags |= debugCopy;
			}
			if (debug.find(execChoice) != string::npos) {
				cout << " " << execChoice;
				localDbgFlags |= debugExec;
			}
			if (debug.find(fillChoice) != string::npos) {
				cout << " " << fillChoice;
				localDbgFlags |= debugFill;
			}
			if (debug.find(matprodChoice) != string::npos) {
				cout << " " << matprodChoice;
				localDbgFlags |= debugMatProd;
			}
			if (debug.find(debugMatStatsChoice) != string::npos) {
				cout << " " << debugMatStatsChoice;
				localDbgFlags |= debugMatStats;
			}
			if (debug.find(nnChoice) != string::npos) {
				cout << " " << nnChoice;
				localDbgFlags |= debugNn;
			}
			if (debug.find(cgChoice) != string::npos) {
				cout << " " << cgChoice;
				localDbgFlags |= debugCg;
			}
			if (debug.find(debugBinOpChoice) != string::npos) {
				cout << " " << debugBinOpChoice;
				localDbgFlags |= debugBinOp;
			}
			if (debug.find(consChoice) != string::npos) {
				cout << " " << consChoice;
				localDbgFlags |= debugCons;
			}

			if (debug.find(refcountChoice) != string::npos) {
				cout << " " << refcountChoice;
				localDbgFlags |= debugRefcount;
			}
			if (debug.find(verboseChoice) != string::npos) {
				cout << " " << verboseChoice;
				localDbgFlags |= debugVerbose;
			}
			if (debug.find(pmChoice) != string::npos) {
				cout << " packed matrix ";
				localDbgFlags |= debugPm;
			}
			if (debug.find(txpChoice) != string::npos) {
				cout << " " << txpChoice;
				localDbgFlags |= debugTxp;
			}
			if (debug.find(debugReduxChoice) != string::npos) {
				cout << " " << debugReduxChoice;
				localDbgFlags |= debugRedux;
			}
			if (debug.find(debugTimerChoice) != string::npos) {
				cout << " " << debugTimerChoice;
				localDbgFlags |= debugTimer;
			}
			if (debug.find(debugTilerChoice) != string::npos) {
				cout << " " << debugTilerChoice;
				localDbgFlags |= debugTiler;
			}

			if (debug.find(debugUnaryOpChoice) != string::npos) {
				cout << " " << debugUnaryOpChoice;
				localDbgFlags |= debugUnaryOp;
			}
			if (debug.find(debugPrecisionChoice) != string::npos) {
				cout << " " << debugPrecisionChoice;
				localDbgFlags |= debugPrecision;
			}
			if (debug.find(debugMeansChoice) != string::npos) {
				cout << " " << debugMeansChoice;
				localDbgFlags |= debugMeans;
			}
			if (debug.find(smallBlkChoice) != string::npos) {
				cout << " " << smallBlkChoice;
				CuMatrix<T>::DefaultMatProdBlock = dim3(8, 8);
			}
			if (debug.find(medBlkChoice) != string::npos) {
				cout << " " << medBlkChoice;
				CuMatrix<T>::DefaultMatProdBlock = dim3(16, 16);
			}
			if (debug.find(lrgBlkChoice) != string::npos) {
				cout << " " << lrgBlkChoice;
				CuMatrix<T>::DefaultMatProdBlock = dim3(32, 32);
			}
			if (debug.find(debugMultGPUChoice) != string::npos) {
				cout << " " << debugMultGPUChoice;
				localDbgFlags |= debugMultGPU;
			}
			if (debug.find(debugMillisForMicrosChoice) != string::npos) {
				cout << " " << debugMillisForMicrosChoice;
				AccelEvents<T>::DelayMillisForMicros = true;
			}
		}
		cout << "\n\n";
	}
	return localDbgFlags;

}
template<typename T> int tests<T>::runTest(int argc, const char** argv) {


	int status = 0;

	b_util::announceTime();

	if (sizeof(T) == 4)
		cout.precision(10);
	else
		cout.precision(16);
	int idev = 0;
	int stopAt = -1;
	char *deviceChoice = nullptr;
	char *stopAtChoice = nullptr;
	char *debugChoice = nullptr;
	char *testChoice = nullptr;
	char *kernelChoice = nullptr;
	char *coolChoice = nullptr;

	uint localDbgFlags = parseArgs<T>(
			idev,
			stopAt,
			deviceChoice,
			stopAtChoice,
			debugChoice,
			testChoice,
			kernelChoice,
			coolChoice,
			argc, argv);
	outln("calling b_util::usedDmem()...");
//	b_util::usedDmem();

	outln("calling 	ExecCaps::initDevCaps()");
	ExecCaps::initDevCaps();

	if(true || !coolChoice) {
		b_util::allDevices([]() {
			b_util::warmupL();
			b_util::dumpGpuStats(ExecCaps::currDev());
		});
	}
	outln("localDbgFlags " << localDbgFlags);
	//setAllGpuDebugFlags(localDbgFlags,false,false);
	if(localDbgFlags != (uint)-1)
		b_util::allDevices(
				[localDbgFlags]() {setCurrGpuDebugFlags(localDbgFlags,false,false,0 );});
	outln("set debug flags ");

	if (checkDebug(debugCg))
		outln("dbg cg");
	if (deviceChoice) {
		idev = atoi(deviceChoice);
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		if (idev >= deviceCount || idev < 0) {
			fprintf(stderr,
					"Device number %d is invalid, will use default CUDA device 0.\n",
					idev);
			idev = 0;
		}
	}

	if(stopAtChoice) {
		stopAt = atoi(stopAtChoice);
		outln("will stop after  " << stopAt);
	}

	if (kernelChoice) {
		string kernel(kernelChoice);
		const auto twochoice = "2";
		if (kernel.find(twochoice) != string::npos) {
			cout << " using alt kernel";
			CuMatrix<T>::g_matrix_product_kernel = matrixProductKernel2;
		} else {
			CuMatrix<T>::g_matrix_product_kernel = matrixProductKernel;
		}
	}

	int specTest = -1;
	if (testChoice) {
		outln("testChoice " << testChoice);
		specTest = atoi(testChoice);
		outln("want specific test " << specTest);
	}

	outln("setting device to " << idev);
	checkCudaErrors(cudaSetDevice(idev));
	//cudaDeviceReset(); // cleans up the dev
	outln("set device");
	cudaError_t lastErr = cudaGetLastError();
	checkCudaErrors(lastErr);

	int priority_low;
	int priority_hi;
	checkCudaErrors(cudaDeviceGetStreamPriorityRange(&priority_low,
	  &priority_hi));
	outln("priority_low " << priority_low << ", priority_hi " <<  priority_hi);


	outln(
			"set device " << idev << "(" << ExecCaps::currCaps(idev)->deviceProp.name << "), now calling CuMatrix<T>::init(256, 64)");

	outln("calling b_util::usedDmem()...");
	b_util::usedDmem();
	outln("initing device...");
	if(coolChoice) outln("coolChoice..." <<coolChoice);


	CuMatrix<float>::initMemMgrForType(256, 64);
	CuMatrix<double>::initMemMgrForType(256, 64);
	CuMatrix<int>::initMemMgrForType(256, 64);

	outln("clearing member function setup flags...");
	clearSetupMbrFlags();
	outln("...cleared member function setup flags");

	//unaryOpIndexMbrs<float>::setupAllMethodTables();
//	unaryOpIndexMbrs<double>::setupAllMethodTables();

	//constFillKrnleL<float>();
	//unaryOpIndexMbrs<double>::setupAllMethodTables();
	//cuFunctorMain();
	b_util::usedDmem(true);

	//if(!coolChoice)
		printAllDeviceGFlops<T>();

	if (checkDebug(debugVerbose)) {
		flprintf("set debugFlags to %u\n", localDbgFlags);
	}
	// checkCudaErrors(cudaFuncSetCacheConfig((void*)matrixProductKernel<T>,cudaFuncCachePreferL1));
	// initialize events
	{
		outln("tests<T>::crunTest starting argc " << argc <<" argv " << argv);
		if (argv)
			for (int i = 0; i < argc; i++) {
				outln("tests<T>::crunTest arg " <<i << ": " << argv[i]);
			}

		CuTimer timer;
		int currDev = ExecCaps::currDev();
		outln("currDev: " << currDev);
		timer.start();
		vector<Test<T>*> vTests;

		shufftestLauncher<T>();
		shuffleSortTestLauncher<T>();

		vTests.push_back(new testTranspose<T>());
		vTests.push_back(new testCuSet1D<T>());
		vTests.push_back(new testCuSet<T>());

		vTests.push_back(new testSubmatrices2<T>());

		vTests.push_back(new testDropFirstAlts<T>());

		vTests.push_back(new testNeural<T>());

		vTests.push_back(new testOrderedCountedSet<T>());

		vTests.push_back(new testMergeSorted<T>());

		vTests.push_back(new testFillVsLoadRnd<T>());
		vTests.push_back(new testAttFreqsLarge<T>());

		vTests.push_back(new testJaccard<T>());

		vTests.push_back(new testDedeup<T>());


		//vTests.push_back(new testFillers<T>());

		vTests.push_back(new testTranspose<T>());

		vTests.push_back(new testDTiler<T>());
		vTests.push_back(new testAttFreqsSmall<T>());
		vTests.push_back(new testAttFreqsMed<T>());

		vTests.push_back(new testTransposeLoop<T>());

		vTests.push_back(new testTiler<T>());


		vTests.push_back(new testAnomDet<T>());
		vTests.push_back(new testCat<T>());
		vTests.push_back(new testXRFill<T>());

		vTests.push_back(new testProductShapesKernelPtr<T>());

		vTests.push_back(new testRedux<T>());
		//vTests.push_back(new testAttFreqs<T>());
		vTests.push_back(new testFillNsb<T>());
		vTests.push_back(new testColumnRedux<T>());
		vTests.push_back(new testTransposeHuge<T>());
		vTests.push_back(new testMeansPitch<T>());
		vTests.push_back(new testPermus<T>());

		/*
		 *
		(boom order:)
		vTests.push_back(new testCat<T>());
		vTests.push_back(new testXRFill<T>());
			vTests.push_back(new testNeural<T>());
	vTests.push_back(new testProductShapesKernelPtr<T>());


		vTests.push_back(new testRedux<T>());
		//vTests.push_back(new testAttFreqs<T>());
		vTests.push_back(new testAnomDet<T>());
		vTests.push_back(new testTranspose<T>());
		vTests.push_back(new testFillNsb<T>());

	work order
			vTests.push_back(new testAnomDet<T>());

		vTests.push_back(new testNeural<T>());
		vTests.push_back(new testCat<T>());
		vTests.push_back(new testXRFill<T>());
	vTests.push_back(new testProductShapesKernelPtr<T>());


		vTests.push_back(new testRedux<T>());
		//vTests.push_back(new testAttFreqs<T>());
		vTests.push_back(new testTranspose<T>());
		vTests.push_back(new testFillNsb<T>());

		 *
		 */
		/*
		 //		vTests.push_back(new testColumnRedux<T>());
		 //vTests.push_back(new testTransposeHuge<T>());

		 //vTests.push_back(new testNeuralMnistHw<T>());


		 //vTests.push_back(new testTranspose<T>());

		 //vTests.push_back(new testNeural<T>());

		 //vTests.push_back(new testDim3Octave<T>());

		 //vTests.push_back(new testNeural2Loop<T>());

		 //vTests.push_back(new testMeansPitch<T>());
		 //vTests.push_back(new testPermus<T>());

		 //vTests.push_back(new testMemsetFill<T>());

		 //vTests.push_back(new testRandAsFnOfSize<T>());

		 //vTests.push_back(new testAnomDet<T>());

		 //vTests.push_back(new testNeural2l<T>());
		 //vTests.push_back(new testPackedMat<T>());
		 //vTests.push_back(new testBounds<T>());
		 //vTests.push_back(new testNeural2lAdult<T>());
		 //vTests.push_back(new testNeural2l<T>());
		 vTests.push_back(new testNeural2lYrPred<T>());
		 //vTests.push_back(new testMemset<T>());
		 //vTests.push_back(new testProductShapesLoop<T>());
		 vTests.push_back(new testHugeMatProds<T>());
		 //vTests.push_back(new testCat<T>());
		 //vTests.push_back(new testKmeans<T>());
		 //vTests.push_back(new testSubmatrices<T>());
		 //vTests.push_back(new testLUdecomp<T>());
		 vTests.push_back(new testHugeMatProds2<T>());
		 vTests.push_back(new testProductShapes<T>());
		 vTests.push_back(new testLargeMatProds<T>());

		 vTests.push_back(new testCat<T>());
		 vTests.push_back(new testKmeans<T>());
		 vTests.push_back(new testSubmatrices<T>());
		 vTests.push_back(new testTranspose<T>());
		 vTests.push_back(new testLUdecomp<T>());

		 vTests.push_back(new testCat<T>());
		 vTests.push_back(new testKmeans<T>());
		 vTests.push_back(new testSubmatrices<T>());
		 vTests.push_back(new testTranspose<T>());

		 vTests.push_back(new testTinyXRFill<T>());
		 vTests.push_back(new testProductShapes<T>());
		 vTests.push_back(new testHugeMatProds<T>());
		 vTests.push_back(new testHugeMatProds2<T>());
		 */
		//vTests.push_back(new testCudaMemcpy2D<T>());
		//vTests.push_back(new testProductShapesKernelPtr<T>());
		//vTests.push_back(new testNormalize<T>());
		//memset(vTests, 0, count * sizeof(Test<T>*));
		/*
		 vTests.push_back(new testNeural<T>());
		 vTests.push_back(new testCMap<T>());
		 vTests.push_back(new testCMap<T>());
		 vTests.push_back(new testCMap<T>());
		 vTests.push_back(new testCMap<T>());
		 vTests.push_back(new testCMap<T>());
		 vTests.push_back(new testCMap<T>());
		 vTests.push_back(new testCMap<T>());
		 vTests.push_back(new testCMap<T>());

		 vTests.push_back(new testKmeans<T>());

		 vTests.push_back(new testLUdecomp<T>());

		 */
		/*
		 vTests.push_back(new testLastError<T>());
		 vTests.push_back(new testBounds<T>());

		 vTests.push_back(new testSign<T>());
		 vTests.push_back(new testCubes<T>());
		 vTests.push_back(new testReduceRows<T>());


		 vTests.push_back(new testClippedRowSubset<T>());
		 vTests.push_back(new testCdpSum<T>());
		 vTests.push_back(new testMeansFile<T>());
		 vTests.push_back(new testFeatureMeans<T>());

		 vTests.push_back(new testStrmOrNot<T>());


		 vTests.push_back(new testBinCat<T>());
		 vTests.push_back(new testMatRowMath<T>());

		 vTests.push_back(new testNeural2l<T>());

		 //vTests.push_back(new testLogRegCostFunctionNoRegMapFeature<T>());

		 vTests.push_back(new testAnonMatrices<T>());
		 vTests.push_back(new testTranspose<T>());
		 vTests.push_back(new testKmeans<T>());

		 vTests.push_back(new testCopyFtor<T>());
		 //vTests.push_back(new testSign<T>());

		 vTests.push_back(new testSqrMatsMultSmall<T>());


		 vTests.push_back(new testProductShapesKernelPtr<T>());

		 vTests.push_back(new testCount<T>());

		 vTests.push_back(new testCopyVsCopyK<T>());

		 vTests.push_back(new testInc<T>());

		 vTests.push_back(new testCostFunction<T>());

		 vTests.push_back(new testHfreeDalloc<T>());
		 vTests.push_back(new testMod<T>());
		 vTests.push_back(new testMatRowMath<T>());
		 vTests.push_back(new testCountSpanrows<T>());

		 vTests.push_back(new testFillers<T>());

		 //vTests.push_back(new testCubes<T>());
		 vTests.push_back(new testNextPowerOf2<T>());
		 vTests.push_back(new testLargestFactor<T>());

		 vTests.push_back(new testBisection<T>());

		 //vTests.push_back(new testReduceRows<T>());

		 vTests.push_back(new testLargeMatProds<T>());

		 vTests.push_back(new testCMap<T>());
		 vTests.push_back(new testAnomDet<T>());

		 //vTests.push_back(new testMVAnomDet<T>());

		 vTests.push_back(new testEtoX<T>());

		 vTests.push_back(new testColumnRedux<T>());

		 vTests.push_back(new testCMap<T>());
		 vTests.push_back(new testShufflet<T>());
		 vTests.push_back(new testShuffle<T>());

		 //vTests.push_back(new testRandAsFnOfSize<T>());
		 vTests.push_back(new testCdpRedux<T>());
		 vTests.push_back(new testRedux<T>());


		 //vTests.push_back(new testLastError<T>());
		 //vTests.push_back(new testRecCuMatAddition<T>());
		 vTests.push_back(new testBounds<T>());


		 vTests.push_back(new testClippedRowSubset<T>());
		 vTests.push_back(new testCdpSum<T>());
		 vTests.push_back(new testMeansFile<T>());
		 vTests.push_back(new testFeatureMeans<T>());

		 vTests.push_back(new testStrmOrNot<T>());

		 vTests.push_back(new testAutodot<T>());
		 vTests.push_back(new testMultLoop<T>());
		 vTests.push_back(new testProductKernel3<T>());
		 vTests.push_back(new testProductShapes<T>());
		 vTests.push_back(new testProductShapesLoop<T>());

		 vTests.push_back(new testProductShapesTxB<T>());

		 vTests.push_back(new testSqrMatsMult<T>());
		 vTests.push_back(new testSqrMatsMultSmall<T>());

		 //vTests.push_back(new testNeural2l<T>());
		 //vTests.push_back(new testNeural2Loop<T>());
		 //vTests.push_back(new testSumSqrDiffsLoop<T>());
		 //vTests.push_back(new testRedWineCsv<T>());
		 //vTests.push_back(new testCsv<T>());
		 //vTests.push_back(new testMeansLoop<T>());
		 //vTests.push_back(new testDynamicParallelism<T>());

		 //vTests.push_back(new testOglHelloworld<T>());
		 //vTests.push_back(new testOglAnim0<T>());
		 //vTests.push_back(new testFrag7Csv<T>());
		 //vTests.push_back(new testFastInvSqrt<T>());
		 //vTests.push_back(new testLUdecomp<T>());

		 //vTests.push_back(new testMemcpyShared<T>());
		 //vTests.push_back(new testBinaryOpAmort<T>());

		 //vTests.push_back(new testFillNsb<T>());
		 //vTests.push_back(new testParseOctave<T>());
		 //vTests.push_back(new testLogRegCostFunctionNoRegMapFeature<T>());
		 vTests.push_back(new testProductShapesLoop<T>());
		 */
		//testLogRegSuite(vTests,idx));
		//vTests.push_back(new testLinRegCostFunctionNoReg<T>());
		/*
		 * testFnArgCount testShuffle testNeural2l testNeural2Loop testProductShapes testProductShapesTxB testSqrMatsMult
		 testDynamicParallelism testMultiGPUMath testFileIO testProductShapes testMultiGPUMemcpy testSuite testAnomDet testProductShapesLoop testNeural2l testRandomizingCopyRows
		 testMeans testMeansLoop
		 */
		outln("added " << vTests.size() << " vTests");
		int currIdx = 1;
		for(auto t : vTests) {
			t->testNo = currIdx++;
		}
		outln(
				"using mat prod block of dims " << b_util::pd3(CuMatrix<T>::DefaultMatProdBlock));
		b_util::usedDmem();
		outln("tests<T>::2timeTest starting argc " << argc <<" argv " << argv);
		if (argv)
			for (int i = 0; i < argc; i++) {
				outln("tests<T>::2timeTest arg " <<i << ": " << argv[i]);
			}
		// specific test

		Tiler<T> gpuSwitcher(CuMatrix<T>::ZeroMatrix);
		int devCount;
		currIdx = -1;
		checkCudaErrors(cudaGetDeviceCount(&devCount));
#ifdef CuMatrix_NVML
		uint gpuTemps[devCount];
		nvmlDevice_t device;
		nvmlGpuOperationMode_t mode;
		//nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending)
		nvmlGpuOperationMode_t current, pending;
		uint unitCount = 0;
		chnerr(nvmlUnitGetCount(&unitCount));
		outln("unitCount " << unitCount);
#endif
		gpuSwitcher.setGpuMask(Tiler<T>::gpuMaskFromCount(devCount));
		outln("gpuSwitch mask");
		Tiler<T>::dumpMask(gpuSwitcher.gpuMask);
		if (specTest > 0) {
			if (specTest < vTests.size()) {
				tests<T>::timeTest(*(vTests[specTest]), argc, argv, &status);
			} else {
				outln("test must be [0," << vTests.size() << "); was " << specTest);
			}
		} else {
			outln("want selected vTests");
			auto itr = vTests.begin();
			auto testNo = 0;
			while (itr != vTests.end() && ( stopAt == -1 || stopAt > currIdx)) {
				if(stopAt != -1) {
					if (stopAt == currIdx + 1) {
						outln("stopping after next test");
					} else {
						outln("stopping in " << stopAt - currIdx << " tests");
					}
				}
				//
#ifdef CuMatrix_NVML
				// give coldest gpu next test
				uint minTemp = util<uint>::maxValue();

				for(uint i = 0; i < devCount; i++) {
					nvmlDevice_t device;
					chnerr(nvmlDeviceGetHandleByIndex(i, &device));
					chnerr(nvmlDeviceGetTemperature(device,NVML_TEMPERATURE_GPU, &gpuTemps[i] ));
					if(gpuTemps[i] < minTemp) {
						minTemp = gpuTemps[i];
						currDev = i;
					}
				}

				uint unitCount = 0;
				chnerr(nvmlUnitGetCount(&unitCount));
				outln("unitCount " << unitCount);
				nvmlUnit_t unit;
				nvmlUnitInfo_t info;
				for(uint i =0; i < unitCount; i++) {
					chnerr(nvmlUnitGetHandleByIndex(i, &unit));
					outln("unit " << unit);
					chnerr(nvmlUnitGetUnitInfo(unit, &info));
					outln("info.name " << info.name);
					outln("info.id " << info.id);
					outln("info.serial " << info.serial);
					outln("info.firmwareVersion " << info.firmwareVersion);
				}

#else
				currDev = gpuSwitcher.nextGpu(currIdx-1);
#endif
				currIdx++;
				//b_util::dumpGpuStats(ExecCaps::currDev());
				checkCudaError(cudaSetDevice(currDev));
				outln("currDev " << currDev);
				outln(
						"launching next test on device " << currDev << "\n\t\t" << gpuNames[currDev]);
				tests<T>::timeTest(*(*itr), argc, argv, &status);
				itr++;
			}
			if(stopAt != -1) {
				if(stopAt == 0)
					outln("chickened out after first test");
				else
					outln("chickened out after " << stopAt << " test" << (stopAt > 1 ? "s": "") );
			}
		}
		outln("currDev: " << currDev);
		checkCudaErrors(cudaSetDevice(currDev));
		outln("before stopping main timer, curr dev: " << ExecCaps::currDev());
		float exeTime = timer.stop();
		b_util::announceDuration(exeTime);
	}

	CuMatrix<T>::cleanup();
	outln("CuDestructs " << CuDestructs);
	outln("all done, bye! returning status " << status);
	return status;
}

int dmain(int argc, const char** argv);

void printUsage() {
	printf("\n");
	printf(UsageStrPreamble.c_str() );
	printf("\n");
	for(int i =0; i < 7; i++) {
		printf(UsagesStr[i].c_str());
		printf("\n");
	}
	printf("\n");
	printf(DebugOptionsPreamble.c_str());
	printf("\n");
	for(int i =0; i < 31; i++) {
		printf(DebugOptionsStr[i].c_str());printf("\n");
	}

}

int main(int argc, const char** argv) {
	outln("starting...");
	int devID;

	char* dummy= nullptr;
	if(getCmdLineArgumentString(argc, (const char **) argv, "usage", &dummy)) {
		printUsage();
		return 0;
	}

	if (argv)
		for (int i = 0; i < argc; i++) {
			flprintf("arg %d: %s\n", i, argv[i]);
			if(argv[i] && '-' == argv[i][0]) {
				test::argsl.push_front(argv[i]);
			}
		}

	//cudaError_t error;
	cudaDeviceProp deviceProp;

	// This will pick the best possible CUDA capable device.
	devID = findCudaDevice(argc, (const char **) argv);
	outln("devID "<< devID);

#ifdef CuMatrix_NVML
	nvmlReturn_t nvmlRet = nvmlInit();
#endif

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	outln("deviceProp.memoryClockRate "<< deviceProp.memoryClockRate);

	size_t dheap =0;
	checkCudaErrors(cudaDeviceGetLimit(&dheap,cudaLimitMallocHeapSize));

	outln("cudaLimitMallocHeapSize dheap " << dheap);

	outln("calling b_util::usedDmem()...");
	// b_util::usedDmem(true);
	//dmain(argc,argv);
	outln("starting argc " << argc <<" argv " << argv);
	pthread_t currThread = pthread_self();
	outln("currThread " << currThread);
	if (argv)
		for (int i = 0; i < argc; i++) {
			outln("arg " <<i << ": " << argv[i]);
		}
	srand(time(NULL));
	b_util::handleSignals();
	char* typeChoice = null;

	getCmdLineArgumentString(argc, (const char **) argv, "type", &typeChoice);
	if (typeChoice) {
		remarg(test::argsl,"type");
		string typeS(typeChoice);
		if (typeS.find("4") != string::npos)
			return tests<float>::runTest(argc, argv);
	}
	//if(typeS.find("8") != string::npos)
	return tests<double>::runTest(argc, argv);

}

template<typename T> void testLogRegSuite(vector<Test<T>*> vTests, int& idx) {
	//vTests.push_back(new testParseOctave<T>();

	vTests.push_back(new testLinRegCostFunctionNoReg<T>());
	//vTests.push_back(new testCostFunction<T>());
	//vTests.push_back(new testCostFunction<T>());
	//vTests.push_back(new testLogRegCostFunctionNoRegMapFeature<T>());

}

#include "tests.cc"
