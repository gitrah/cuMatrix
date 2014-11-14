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
const auto allChoice = "all";
const auto anomChoice = "anom";
const auto memChoice = "mem";
const auto copyChoice = "copy";
const auto copyDhChoice = "copyDh";
const auto execChoice = "exec";
const auto fillChoice = "fill";
const auto lifeChoice = "life";
const auto matprodChoice = "matprod";
const auto matprodBlockResizeChoice = "mpbr";
const auto debugMatStatsChoice = "stats";

const auto consChoice = "cons";
const auto stackChoice = "stack";
const auto verboseChoice = "verb";
const auto syncChoice = "sync";
const auto nnChoice = "nn";
const auto cgChoice = "cg";
const auto txpChoice = "txp";
const auto syncHappyChoice = "shappy";
const auto smallBlkChoice = "sblk";
const auto medBlkChoice = "mblk";
const auto lrgBlkChoice = "lblk";
const auto debugMultGPUChoice = "gpu";
const auto debugMillisForMicrosChoice = "m4m";
const auto debugReduxChoice = "rdx";
const auto debugNoReduxChoice = "ndx";
const auto debugUnaryOpChoice = "uny";
const auto debugPrecisionChoice = "prec";
const auto debugMeansChoice = "mean";
const auto debugBinOpChoice = "bnop";

extern template class CuMatrix<float>;
extern template class CuMatrix<double>;

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

template <typename T> int tests<T>::runTest(int argc, const char** argv) {

	outln("tests<T>::runTest starting argc " << argc <<" argv " << argv);
	if(argv) for(int i =0; i < argc; i++) {
		outln("tests<T>::runTest arg " <<i << ": " << argv[i]);
	}

	// CUDA events
	int idev= 0;
	char *device = null;
	char *debugChoice = null;
	char *kernelChoice = null;

    int status = -1;

	b_util::announceTime();

	if(sizeof(T) == 4)
		cout.precision(10);
	else
		cout.precision(16);

	//testCheckValid();
	getCmdLineArgumentString(argc, (const char **) argv, "dbg", &debugChoice);
	getCmdLineArgumentString(argc, (const char **) argv, "dev", &device);
	getCmdLineArgumentString(argc, (const char **) argv, "krn", &kernelChoice);
	outln("tests<T>::brunTest starting argc " << argc <<" argv " << argv);
	if(argv) for(int i =0; i < argc; i++) {
		outln("tests<T>::brunTest arg " <<i << ": " << argv[i]);
	}

	uint localDbgFlags = 0;
	if (debugChoice) {
		string debug(debugChoice);
		cout << "debug choices: ";
		if (debug.find(allChoice) != string::npos) {
			cout << " ALL";
			localDbgFlags |= debugMem;
			localDbgFlags |= debugLife;
			localDbgFlags |= debugCopy;
			localDbgFlags |= debugCopyDh;
			localDbgFlags |= debugMatProd;
			localDbgFlags |= debugCons;
			localDbgFlags |= debugStack;
			localDbgFlags |= debugVerbose;
			localDbgFlags |= debugNn;
			localDbgFlags |= debugCg;
			localDbgFlags |= debugTxp;
			localDbgFlags |= debugExec;
			localDbgFlags |= debugSync;
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
			if (debug.find(lifeChoice) != string::npos) {
				cout << " " << lifeChoice;
				localDbgFlags |= debugLife;
			}
			if (debug.find(copyChoice) != string::npos) {
				cout << " " << copyChoice;
				localDbgFlags |= debugCopy;
			}
			if (debug.find(copyDhChoice) != string::npos) {
				cout << " " << copyDhChoice;
				localDbgFlags |= debugCopyDh;
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
			if (debug.find(stackChoice) != string::npos) {
				cout << " " << stackChoice;
				localDbgFlags |= debugStack;
			}
			if (debug.find(verboseChoice) != string::npos) {
				cout << " " << verboseChoice;
				localDbgFlags |= debugVerbose;
			}
			if (debug.find(syncChoice) != string::npos) {
				cout << " " << syncChoice;
				localDbgFlags |= debugSync;
			}
			if (debug.find(syncHappyChoice) != string::npos) {
				cout << " s(ync)happy";
				localDbgFlags |= syncHappy;
			}
			if (debug.find(txpChoice) != string::npos) {
				cout << " " << txpChoice;
				localDbgFlags |= debugTxp;
			}
			if (debug.find(debugReduxChoice) != string::npos) {
				cout << " " << debugReduxChoice;
				localDbgFlags |= debugRedux;
			}
			if (debug.find(debugNoReduxChoice) != string::npos) {
				cout << " " << debugNoReduxChoice;
				localDbgFlags |= debugNoRedux;
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
		cout << endl;
	}

	ExecCaps::findDevCaps();

	outln("localDbgFlags " << localDbgFlags);
	setAllGpuDebugFlags(localDbgFlags,false,false);
	outln("set debug flags ");

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

	outln("setting device...");
	checkCudaErrors(cudaSetDevice(idev));
	//cudaDeviceReset(); // cleans up the dev

	outln("set device");
	cudaError_t lastErr = cudaGetLastError();
	checkCudaErrors(lastErr);

    outln("set device " << idev << "(" << ExecCaps::currCaps(idev)->deviceProp.name << "), now calling CuMatrix<T>::init(256, 64)");
    b_util::usedDmem();
	outln("initing device...");
	CuMatrix<float>::init(256, 64);
	CuMatrix<double>::init(256, 64);

	outln("clearing member function setup flags...");
	clearSetupMbrFlags();
	outln("...cleared member function setup flags");

	//unaryOpIndexMbrs<float>::setupAllMethodTables();
//	unaryOpIndexMbrs<double>::setupAllMethodTables();

	//constFillKrnleL<float>();
	//unaryOpIndexMbrs<double>::setupAllMethodTables();
	//cuFunctorMain();
	outln("init device");

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
		int count=10;
		int idx=0;
		Test<T>* actual[count];
		memset(actual, 0, count * sizeof(Test<T>*));

		//actual[idx++] = new testCopyFtor<T>();
		//actual[idx++] = new testSign<T>();
		//actual[idx++] = new testFillFPtr<T>();

		//actual[idx++] = new testProductShapesLoop<T>();
		//actual[idx++] = new testCount<T>();

		//actual[idx++] = new testCopyVsCopyK<T>();

		//actual[idx++] = new testInc<T>();

		//actual[idx++] = new testCostFunction<T>();

		//actual[idx++] = new testHfreeDalloc<T>();
		//actual[idx++] = new testMod<T>();
		//actual[idx++] = new testMatRowMath<T>();
		//actual[idx++] = new testCountSpanrows<T>();

		//actual[idx++] = new testFillers<T>();

		//actual[idx++] = new testCubes<T>();
		//actual[idx++] = new testNextPowerOf2<T>();
		//actual[idx++] = new testLargestFactor<T>();

		//actual[idx++] = new testBisection<T>();

		//actual[idx++] = new testReduceRows<T>();

		actual[idx++] = new testKmeans<T>();

		actual[idx++] = new testCMap<T>();
		actual[idx++] = new testAnomDet<T>();
				//actual[idx++] = new testMVAnomDet<T>();
		//actual[idx++] = new testBinCat<T>();

		//actual[idx++] = new testEtoX<T>();
		//actual[idx++] = new testFuncPtr<T>();

		//actual[idx++] = new testRedux<T>();
		//actual[idx++] = new testColumnRedux<T>();

		//actual[idx++] = new testCdpSync1<T>();
		//actual[idx++] = new testShufflet<T>();
		//actual[idx++] = new testShuffle<T>();

		//actual[idx++] = new testRandAsFnOfSize<T>();
		//actual[idx++] = new testCdpRedux<T>();


		//actual[idx++] = new testLastError<T>();
		//actual[idx++] = new testRecCuMatAddition<T>();
		//actual[idx++] = new testBounds<T>();


		//actual[idx++] = new testClippedRowSubset<T>();
		//actual[idx++] = new testCdpSum<T>();
		//actual[idx++] = new testMeansFile<T>();
		//actual[idx++] = new testFeatureMeans<T>();

		//actual[idx++] = new testStrmOrNot<T>();
		//actual[idx++] = new testUnimemThrupt1<T>();
/*
		actual[idx++] = new testAutodot<T>();
		actual[idx++] = new testMultLoop<T>();
		actual[idx++] = new testProductKernel3<T>();
		actual[idx++] = new testProductShapes<T>();
		actual[idx++] = new testProductShapesLoop<T>();

		actual[idx++] = new testProductShapesTxB<T>();

		actual[idx++] = new testSqrMatsMult<T>();
		actual[idx++] = new testSqrMatsMultSmall<T>();
*/
		//actual[idx++] = new testNeural2l<T>();
		//actual[idx++] = new testNeural2Loop<T>();
		//actual[idx++] = new testSumSqrDiffsLoop<T>();
		//actual[idx++] = new testRedWineCsv<T>();
		//actual[idx++] = new testCsv<T>();
		//actual[idx++] = new testMeansLoop<T>();
		//actual[idx++] = new testDynamicParallelism<T>();

		//actual[idx++] = new testOglHelloworld<T>();
		//actual[idx++] = new testOglAnim0<T>();
		//actual[idx++] = new testFrag7Csv<T>();
		//actual[idx++] = new testFastInvSqrt<T>();
		//actual[idx++] = new testLUdecomp<T>();

		//actual[idx++] = new testMemcpyShared<T>();
		//actual[idx++] = new testBinaryOpAmort<T>();

		//actual[idx++] = new testFillNsb<T>();
		//actual[idx++] = new testParseOctave<T>();
		//actual[idx++] = new testLogRegCostFunctionNoRegMapFeature<T>();
		//testLogRegSuite(actual,idx);

		//actual[idx++] = new testLinRegCostFunctionNoReg<T>();

		/*
		 * testFnArgCount testShuffle testNeural2l testNeural2Loop testProductShapes testProductShapesTxB testSqrMatsMult
		 testDynamicParallelism testMultiGPUMath testFileIO testProductShapes testMultiGPUMemcpy testSuite testAnomDet testProductShapesLoop testNeural2l testRandomizingCopyRows
		 testMeans testMeansLoop
		 */
		outln("created " << actual);
		outln("using mat prod block of dims " << b_util::pd3(CuMatrix<T>::DefaultMatProdBlock));
		b_util::usedDmem();
		outln("launching " << actual);
		outln("tests<T>::2timeTest starting argc " << argc <<" argv " << argv);
		if(argv) for(int i =0; i < argc; i++) {
			outln("tests<T>::2timeTest arg " <<i << ": " << argv[i]);
		}
		for(int i = 0; i < count; i++ ){
			if(actual[i]) {
				tests<T>::timeTest(*(actual[i]), argc, argv, &status);
			}
		}
		checkCudaErrors(cudaSetDevice(currDev));
    	outln("before stopping main timer, curr dev: " << ExecCaps::currDev());
		float exeTime = timer.stop();

		b_util::announceDuration(exeTime);
		//b_util::waitEnter();
	}
    CuMatrix<T>::cleanup();
    outln("all done, bye!");
	return status;
}

int main(int argc, const char** argv) {
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
	return tests<float>::runTest(argc,argv);

}

template <typename T> void testLogRegSuite(Test<T>* actual[], int& idx) {
	//actual[idx++] = new testParseOctave<T>();

	actual[idx++] = new testLinRegCostFunctionNoReg<T>();
	//actual[idx++] = new testCostFunction<T>();
	//actual[idx++] = new testCostFunction<T>();
	//actual[idx++] = new testLogRegCostFunctionNoRegMapFeature<T>();

}

#include "tests.cc"
