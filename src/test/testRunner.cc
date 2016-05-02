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
extern template class CuMatrix<float>;
extern template class CuMatrix<double>;
extern template class CuMatrix<long>;
extern template class CuMatrix<ulong>;
extern template class CuMatrix<int>;
extern template class CuMatrix<uint>;

int funquenstein(int argc, char *argv<::>)
<%
    if (argc > 1 and argv<:1:> not_eq '\0') <%
        std::cout << "Hello " << argv<:1:> << '\n';
    %>
%>

struct A
{
	A(int) {}
	operator int() const { return 0; }
};

struct B
{
	explicit B(int) {}
	explicit operator int() const { return 0; }
};

template<typename T> T tests<T>::timeTest( const Test<T>& test, int argc, const char** argv, int* theResult )
{
	int dev = ExecCaps::currDev();
	outln("freest device " << b_util::getDeviceThatIs(gcFreeest));
	outln("coolest device " << b_util::getDeviceThatIs(gcCoolest));
	CuTimer timer;
	sigmoidUnaryOp<T> sig;
	outln("tests<T>::timeTest on device " << dev << " starting argc " << argc <<" argv " << argv);
	if(argv) for(int i =0; i < argc; i++) {
		outln("tests<T>::timeTest arg " <<i << ": " << argv[i]);
	}

	//auto fnctrPtr = &sigmoidUnaryOp<T>::operator();
	typedef T (sigmoidUnaryOp<T>::*memfunc)(T) const;

	memfunc fnctrPtr = &sigmoidUnaryOp<T>::operator();
	auto sigPtr = &sig;
	outln("awaaay &sigmoidUnaryOp<T>::operator() type name " << b_util::unmangl(typeid(fnctrPtr).name()));
	outln("awaaay &sig.operator() " << b_util::unmangl(typeid(sigPtr).name()));
	double (sigmoidUnaryOp<double>::*fncPtrExplicit)(double ) const  = null;
	T (sigmoidUnaryOp<T>::*fncPtrExplicitT)(T ) const  =&sigmoidUnaryOp<T>::operator();
	//fncPtrExplicitT = fncPtrExplicit;
	//T (unaryOpF<T>::*fncPtrExplicitTBase)(T const&) const  = &sigmoidUnaryOp<T>::operator();
	//fncPtrExplicitTBase = fncPtrExplicit;
	outln("(sig.*fncPtrExplicitT)(5) "<< (sig.*fncPtrExplicitT)(5));
	outln("\n\n\n\n\nawaaay we geaux starting " << b_util::unmangl(typeid(test).name()) << "\n\n\n\n\n");
	timer.start();
	int res = test(argc,argv);
	dassert(res ==  0);
	if(theResult) {
		*theResult = res;
	}
	checkCudaErrors(cudaSetDevice(dev));
	float dur = timer.stop();
	string type = sizeof(T) == 8 ? "<double>" : "<float>";
	//outln( typeid(tst).name() << type.c_str() << " took " << dur << "s");
	CuMatrix<T>::ZeroMatrix.getMgr().dumpLeftovers();

	return dur;
}
template<typename T>void constFillKrnleL( );


std::thread::id main_thread_id = std::this_thread::get_id();


template <typename T> int tests<T>::runTest(int argc, const char** argv) {

	static bool printedSizes = false;
	if(!printedSizes) {
		printObjSizes<T>();
		printedSizes  = true;
	}
	outln("tests<T>::runTest starting argc " << argc <<" argv " << argv);
	if(argv) for(int i =0; i < argc; i++) {
		outln("tests<T>::runTest arg " <<i << ": " << argv[i]);
	}

	// CUDA events
	int idev= 0;
	char *device = null;
	char *debugChoice = null;
	char *testChoice = null;
	char *kernelChoice = null;

    int status = -1;

	b_util::announceTime();

	if(sizeof(T) == 4)
		cout.precision(10);
	else
		cout.precision(16);

	//testCheckValid();
	getCmdLineArgumentString(argc, (const char **) argv, "dbg", &debugChoice);
	getCmdLineArgumentString(argc, (const char **) argv, "test", &testChoice);
	getCmdLineArgumentString(argc, (const char **) argv, "dev", &device);
	getCmdLineArgumentString(argc, (const char **) argv, "krn", &kernelChoice);

	flprintf("starting argc %d\n", argc);

	if(argv) for(int i =0; i < argc; i++) {
		flprintf("arg %d: %s\n",i,argv[i]);
	}

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
			if (debug.find(matprodBlockResizeChoice) != string::npos) {
				cout << " " << matprodBlockResizeChoice;
				localDbgFlags |= debugMatProdBlockResize;
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
/*
			if (debug.find(destrChoice) != string::npos) {
				cout << " " << destrChoice;
				localDbgFlags |= debugDestr;
			}
*/
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
				CuMatrix<T>::DefaultMatProdBlock = dim3(8,8);
			}
			if (debug.find(medBlkChoice) != string::npos) {
				cout << " " << medBlkChoice;
				CuMatrix<T>::DefaultMatProdBlock = dim3(16,16);
			}
			if (debug.find(lrgBlkChoice) != string::npos) {
				cout << " " << lrgBlkChoice;
				CuMatrix<T>::DefaultMatProdBlock = dim3(32,32);
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

    outln("calling b_util::usedDmem()...");
    b_util::usedDmem();

    outln("calling 	ExecCaps::initDevCaps()");
	ExecCaps::initDevCaps();


	outln("localDbgFlags " << localDbgFlags);
	//setAllGpuDebugFlags(localDbgFlags,false,false);
	b_util::allDevices([localDbgFlags]() { setCurrGpuDebugFlags(localDbgFlags,false,false,0 );});
	outln("set debug flags ");

	if(checkDebug(debugCg))  outln("dbg cg");
	if(device) {
		idev = atoi(device);
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		if(idev >= deviceCount || idev < 0) {
			fprintf(stderr, "Device number %d is invalid, will use default CUDA device 0.\n", idev);
            idev = 0;
		}
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
	if(testChoice ) {
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

    outln("set device " << idev << "(" << ExecCaps::currCaps(idev)->deviceProp.name << "), now calling CuMatrix<T>::init(256, 64)");

    outln("calling b_util::usedDmem()...");
    b_util::usedDmem();
	outln("initing device...");
	b_util::warmupL();


#ifdef CuMatrix_UseCublas

	g_useCublas = checkCmdLineFlag(argc,argv, "blas");
	outln("g_useCublas  "<< tOrF(g_useCublas ));
	if(g_useCublas) {
		chblerr( cublasCreate(&g_handle));// todo determine time/space effects of making this optional
		//flprintf("cublasCreate success");
	}
#endif

	CuMatrix<float>::initMemMgrForType(256, 64);
	CuMatrix<double>::initMemMgrForType(256, 64);

	outln("clearing member function setup flags...");
	clearSetupMbrFlags();
	outln("...cleared member function setup flags");
	
	//unaryOpIndexMbrs<float>::setupAllMethodTables();
//	unaryOpIndexMbrs<double>::setupAllMethodTables();

	//constFillKrnleL<float>();
	//unaryOpIndexMbrs<double>::setupAllMethodTables();
	//cuFunctorMain();
	outln("pritn gerflops device");

	b_util::usedDmem(true);

	printAllDeviceGFlops<T>();

	if(checkDebug(debugVerbose)) { flprintf("set debugFlags to %u\n",localDbgFlags); }
    // checkCudaErrors(cudaFuncSetCacheConfig((void*)matrixProductKernel<T>,cudaFuncCachePreferL1));
    // initialize events
    {
    	outln("tests<T>::crunTest starting argc " << argc <<" argv " << argv);
    	if(argv) for(int i =0; i < argc; i++) {
    		outln("tests<T>::crunTest arg " <<i << ": " << argv[i]);
    	}

    	CuTimer timer;
    	int currDev = ExecCaps::currDev();
    	outln("currDev: " << currDev);
		timer.start();
		int idx=0;
		vector<Test<T>*> vTests;
		vTests.push_back(new testNeural<T>());

		//vTests.push_back(new testDim3Octave<T>());

		//vTests.push_back(new testNeural2Loop<T>());

		//vTests.push_back(new testMeansPitch<T>());
		//vTests.push_back(new testPermus<T>());

		//vTests.push_back(new testMemsetFill<T>());

		//vTests.push_back(new testRandAsFnOfSize<T>());

		//vTests.push_back(new testAnomDet<T>());
	//	vTests.push_back(new testTranspose<T>());

		//vTests.push_back(new testNeural2l<T>());
		//vTests.push_back(new testPackedMat<T>());
		//vTests.push_back(new testBounds<T>());
 	//vTests.push_back(new testNeural2lAdult<T>());
		//vTests.push_back(new testNeural2l<T>());
	//	vTests.push_back(new testNeural2lYrPred<T>());
		//vTests.push_back(new testMemset<T>());
		//vTests.push_back(new testProductShapesLoop<T>());
		//vTests.push_back(new testHugeMatProds<T>());
		//vTests.push_back(new testCat<T>());
		//vTests.push_back(new testKmeans<T>());
		//vTests.push_back(new testSubmatrices<T>());
		//vTests.push_back(new testLUdecomp<T>());
		/*
		vTests.push_back(new testProductShapes<T>());
		vTests.push_back(new testLargeMatProds<T>());
		vTests.push_back(new testHugeMatProds2<T>());

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
		outln("using mat prod block of dims " << b_util::pd3(CuMatrix<T>::DefaultMatProdBlock));
		b_util::usedDmem();
		outln("tests<T>::2timeTest starting argc " << argc <<" argv " << argv);
		if(argv) for(int i =0; i < argc; i++) {
			outln("tests<T>::2timeTest arg " <<i << ": " << argv[i]);
		}
		// specific test

		Tiler<T> gpuSwitcher(CuMatrix<T>::ZeroMatrix);
		int devCount;
		int currIdx =0;
		checkCudaErrors(cudaGetDeviceCount(&devCount));
#ifdef CuMatrix_NVML
		uint gpuTemps[devCount];
#endif
		gpuSwitcher.setGpuMask( Tiler<T>::gpuMaskFromCount( devCount ));
		outln("gpuSwitch mask");
		Tiler<T>::dumpMask(gpuSwitcher.gpuMask);
		if(specTest > 0 ) {
			if(specTest < idx) {
				tests<T>::timeTest(*(vTests[specTest]), argc, argv, &status);
			} else {
				outln("test must be [0," << idx << "); was " << specTest);
			}
		} else {
			outln("want selected vTests");
			auto itr = vTests.begin();
			int dev;
			while(itr != vTests.end()){
				//
#ifdef CuMatrix_NVML
				// give coldest gpu next test
				uint minTemp = util<uint>::maxValue();
				for(uint i = 0; i < devCount; i++) {
					nvmlDevice_t device;
					chnerr(nvmlDeviceGetHandleByIndex(i,  &device));
					chnerr(nvmlDeviceGetTemperature(device,NVML_TEMPERATURE_GPU, &gpuTemps[i] ));
					if(gpuTemps[i] < minTemp) {
						minTemp = gpuTemps[i];
						currDev = i;
					}
					outln("gpu " << i << " temp " << gpuTemps[i]);
				}
#else
				currDev = gpuSwitcher.nextGpu(currIdx );
				currIdx ++;
#endif
				b_util::dumpGpuStats();
				checkCudaError(cudaSetDevice(currDev));
				outln("launching next test on device " << currDev << "\n\t\t" <<  gpuNames[dev]);
				tests<T>::timeTest(*(*itr), argc, argv, &status);
				itr++;
			}
		}
    	outln("currDev: " << currDev);
		checkCudaErrors(cudaSetDevice(currDev));
    	outln("before stopping main timer, curr dev: " << ExecCaps::currDev());
		float exeTime = timer.stop();
		b_util::announceDuration(exeTime);
		//b_util::waitEnter();
	}

    CuMatrix<T>::cleanup();
    outln("CuDestructs " << CuDestructs);
    outln("all done, bye!");
	return status;
}

int dmain(int argc, const char** argv);

int main(int argc, const char** argv) {

    int devID;

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

	b_util::allDevices([]() { b_util::warmupL(); });


	outln("calling b_util::usedDmem()...");
   // b_util::usedDmem(true);
    //dmain(argc,argv);
	outln("starting argc " << argc <<" argv " << argv);
	pthread_t currThread = pthread_self ();
	outln("currThread " << currThread);
	if(argv) for(int i =0; i < argc; i++) {
		outln("arg " <<i << ": " << argv[i]);
	}
	srand(time(NULL));
	b_util::handleSignals();
	char* typeChoice = null;

	getCmdLineArgumentString(argc, (const char **) argv, "type", &typeChoice);
	if(typeChoice) {
		string typeS(typeChoice);
		if(typeS.find("4") != string::npos)
			return tests<float>::runTest(argc,argv);
	}
	//if(typeS.find("8") != string::npos)
	return tests<double>::runTest(argc,argv);

}

template <typename T> void testLogRegSuite(vector<Test<T>*> vTests, int& idx) {
	//vTests.push_back(new testParseOctave<T>();

	vTests.push_back(new testLinRegCostFunctionNoReg<T>());
	//vTests.push_back(new testCostFunction<T>());
	//vTests.push_back(new testCostFunction<T>());
	//vTests.push_back(new testLogRegCostFunctionNoRegMapFeature<T>());

}

#include "tests.cc"
