/*
 * debug.h
 *
 *  Created on: Jul 24, 2012
 *      Author: reid
 */

#ifndef DEBUG_H_
#define DEBUG_H_

//#define TESTTMPLT

#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include "util.h"

using namespace std;
extern bool debugExec;
extern bool debugMem;
extern bool debugCheckValid;
extern bool debugNn;
extern bool debugCg;
extern bool debugLife;
extern bool debugCopy;
extern bool debugFill;
extern bool debugMatProd;
extern bool debugSync;
extern bool debugCons;
extern bool debugTxp;
extern bool debugStack;
extern bool debugVerbose;
extern bool pauseBetweenTests;
extern bool syncHappy;
#define outln(exp) cout << __FILE__ << "("<< __LINE__ << "): " << exp << endl
#define tout(exp) cout << __FILE__ << "("<< __LINE__ << "): " << exp
#define ot(exp) cout << exp
#define _at() cout << "at " << __FILE__ << "("<< __LINE__ << ")" << endl

#endif /* DEBUG_H_ */
