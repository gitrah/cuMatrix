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
#include <sys/types.h>
#include <sys/wait.h>
#include "tests.h"


bool debugExec = false;
bool debugMem = false;
bool debugLife = false;
bool debugCopy = false;
bool debugMatProd = false;
bool debugSync = false;
bool debugCons = false;
bool debugTxp = false;
bool debugStack = false;
bool debugVerbose = false;
bool syncHappy = false;

string const allChoice = "all";
string const memChoice = "mem";
string const copyChoice = "copy";
string const lifeChoice = "life";
string const matprodChoice = "matprod";
string const consChoice = "cons";
string const stackChoice = "stack";
string const verboseChoice = "verb";
string const syncChoice = "sync";
string const txpChoice = "txp";
string const syncHappyChoice = "shappy";
extern ExecCaps caps;
extern cudaDeviceProp deviceProp;

int testCheckValid() {
	Matrix<float> m = Matrix<float>::zeros(5, 5);
	DMatrix<float> md;
	m.asDmatrix(md, true);
	return 0;
}

template <typename T> int runTest(int argc, char** argv) {
	// CUDA events
	CuTimer timer;
	int idev= 0;
	char *device = null;
	char *debugChoice = null;
    int status = -1;

    b_util::announceTime();
    ExecCaps::getExecCaps(caps);

	Matrix<T>::init(256, 64);
	outln("card(s) capability: " << deviceProp.major << "." << deviceProp.minor);
	b_util::initTimer();
	//testCheckValid();
	getCmdLineArgumentString(argc, (const char **) argv, "dbg", &debugChoice);
	getCmdLineArgumentString(argc, (const char **)argv, "dev", &device);
	if (debugChoice) {
		string debug(debugChoice);
		cout << "debug choices: ";
		if (debug.find(allChoice) != string::npos) {
			cout << " ALL";
			debugMem = debugLife = debugCopy = debugMatProd = debugCons = debugStack = debugVerbose = debugTxp= true;
		} else {
			if (debug.find(memChoice) != string::npos) {
				cout << " mem";
				debugMem = true;
			}
			if (debug.find(lifeChoice) != string::npos) {
				cout << " life";
				debugLife = true;
			}
			if (debug.find(copyChoice) != string::npos) {
				cout << " copy";
				debugCopy = true;
			}
			if (debug.find(matprodChoice) != string::npos) {
				cout << " verbose";
				debugMatProd = true;
			}
			if (debug.find(consChoice) != string::npos) {
				cout << " cons";
				debugCons = true;
			}
			if (debug.find(stackChoice) != string::npos) {
				cout << " stack";
				debugStack = true;
			}
			if (debug.find(stackChoice) != string::npos) {
				cout << " verbose";
				debugVerbose = true;
			}
			if (debug.find(syncChoice) != string::npos) {
				cout << " sync";
				debugSync = true;
			}
			if (debug.find(syncHappyChoice) != string::npos) {
				cout << " s(ync)happy";
				syncHappy = true;
			}
			if (debug.find(txpChoice) != string::npos) {
				cout << " txp";
				debugTxp = true;
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

    timer.start();
    int(*mainLike)(int, char**) = tests<T>::testMatrixSuite;
    util<T>::timeThis(mainLike, "??", argc, argv, &status);

    float exeTime = timer.stop();
	b_util::announceDuration(exeTime);
    b_util::waitEnter();
	Matrix<T>::cleanup();
	return status;
}

int main(int argc, char** argv) {
	b_util::handleSignals();
	return runTest<float>(argc,argv);
}

