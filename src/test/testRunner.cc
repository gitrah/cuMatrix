/*
 * testRunner.cc
 *
 *  Created on: Sep 18, 2012
 *      Author: reid
 */

#include "../Matrix.h"
#include "../caps.h"
#include <cstring>
#include <ctime>
#include <execinfo.h>
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include "tests.h"

bool debugExec = false;
bool debugMem = false;
bool debugCheckValid = true;
bool debugLife = false;
bool debugCopy = false;
bool debugFill = false;
bool debugMatProd = false;
bool debugSync = false;
bool debugCons = false;
bool debugTxp = false;
bool debugStack = false;
bool debugVerbose = false;
bool debugNn = false;
bool debugCg = false;
bool syncHappy = false;

string const allChoice = "all";
string const memChoice = "mem";
string const copyChoice = "copy";
string const execChoice = "exec";
string const fillChoice = "fill";
string const lifeChoice = "life";
string const matprodChoice = "matprod";
string const consChoice = "cons";
string const stackChoice = "stack";
string const verboseChoice = "verb";
string const syncChoice = "sync";
string const nnChoice = "nn";
string const cgChoice = "cg";
string const txpChoice = "txp";
string const syncHappyChoice = "shappy";
string const smallBlkChoice = "sblk";
string const medBlkChoice = "mblk";
string const lrgBlkChoice = "lblk";

extern ExecCaps caps;
extern cudaDeviceProp deviceProp;

extern template class Matrix<float>;
extern template class Matrix<double>;

template<typename T> T tests<T>::timeTest( const Test<T>& test, int argv, const char** args, int* theResult )
{
	CuTimer timer;
	outln("awaaay we geaux starting " << typeid(test).name());
	timer.start();
	int res = test(argv,args);
	dassert(res == 0);
	if(theResult) {
		*theResult = res;
	}
	float dur = timer.stop();
	string type = sizeof(T) == 8 ? "<double>" : "<float>";
	//outln( typeid(tst).name() << type.c_str() << " took " << dur << "s");
	Matrix<T>::ZeroMatrix.getMgr().dumpLeftovers();

	return dur;
}

template <typename T> int tests<T>::runTest(int argc, const char** argv) {

	// CUDA events
	int idev= 0;
	char *device = null;
	char *debugChoice = null;
    int status = -1;

	b_util::announceTime();
	ExecCaps::getExecCaps(caps);

	if(sizeof(T) == 4)
		cout.precision(10);
	else
		cout.precision(16);

	Matrix<T>::init(256, 64);
	outln("card(s) capability: " << deviceProp.major << "." << deviceProp.minor);
	outln("dynamicPism " << tOrF(caps.dynamicPism));
	//testCheckValid();
	getCmdLineArgumentString(argc, (const char **) argv, "dbg", &debugChoice);
	getCmdLineArgumentString(argc, (const char **)argv, "dev", &device);
	if (debugChoice) {
		string debug(debugChoice);
		cout << "debug choices: ";
		if (debug.find(allChoice) != string::npos) {
			cout << " ALL";
			debugMem = true;
			debugLife = true;
			debugCopy = true;
			debugMatProd =true;
			debugMatProd = true;
			debugCons = true;
			debugStack = true;
			debugVerbose = true;
			debugNn = true;
			debugCg = true;
			debugTxp = true;
			debugExec = true;
			debugSync = true;
			debugFill = true;
		} else {
			if (debug.find(memChoice) != string::npos) {
				cout << " " << memChoice;
				debugMem = true;
			}
			if (debug.find(lifeChoice) != string::npos) {
				cout << " " << lifeChoice;
				debugLife = true;
			}
			if (debug.find(copyChoice) != string::npos) {
				cout << " " << copyChoice;
				debugCopy = true;
			}
			if (debug.find(execChoice) != string::npos) {
				cout << " " << execChoice;
				debugExec= true;
			}
			if (debug.find(fillChoice) != string::npos) {
				cout << " " << fillChoice;
				debugFill = true;
			}
			if (debug.find(matprodChoice) != string::npos) {
				cout << " " << matprodChoice;
				debugMatProd = true;
			}
			if (debug.find(nnChoice) != string::npos) {
				cout << " " << nnChoice;
				debugNn = true;
			}
			if (debug.find(cgChoice) != string::npos) {
				cout << " " << cgChoice;
				debugCg = true;
			}
			if (debug.find(consChoice) != string::npos) {
				cout << " " << consChoice;
				debugCons = true;
			}
			if (debug.find(stackChoice) != string::npos) {
				cout << " " << stackChoice;
				debugStack = true;
			}
			if (debug.find(verboseChoice) != string::npos) {
				cout << " " << verboseChoice;
				debugVerbose = true;
			}
			if (debug.find(syncChoice) != string::npos) {
				cout << " " << syncChoice;
				debugSync = true;
			}
			if (debug.find(syncHappyChoice) != string::npos) {
				cout << " s(ync)happy";
				syncHappy = true;
			}
			if (debug.find(txpChoice) != string::npos) {
				cout << " " << txpChoice;
				debugTxp = true;
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


		}
		cout << endl;
	}
	if(device) {
		idev = atoi(device);
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		if(idev >= deviceCount || idev < 0) {
			fprintf(stderr, "Device number %d is invalid, will use default CUDA device 0.\n", idev);
            idev = 0;
		}
	}
    checkCudaErrors(cudaSetDevice(idev));
    // initialize events
    outln(debugMem << debugLife << debugCopy);
    {
    	CuTimer timer;
		timer.start();
		Test<T>* actual = new testNeural2l<T>(); // testProductShapes testSuite testAnomDet testProductShapesLoop testNeural2l testShuffleCopyRows
		outln("created " << actual);
		outln("using mat prod block of dims " << b_util::pd3(CuMatrix<T>::DefaultMatProdBlock));
		b_util::usedDmem();
		outln("launching " << actual);
		tests<T>::timeTest(*actual, argc, argv, &status);
		float exeTime = timer.stop();

		b_util::announceDuration(exeTime);
		//b_util::waitEnter();
	}
    Matrix<T>::cleanup();
	return status;
}

int main(int argc, const char** argv) {
	srand(time(NULL));
	b_util::handleSignals();
	return tests<double>::runTest(argc,argv);
}

#include "tests.cc"
