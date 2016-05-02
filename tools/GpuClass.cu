/*
 * GpuClass.cu
 *
 */


#include <helper_cuda.h>
#include <assert.h>

#include <iostream>
#include <sstream>
#include <set>

using namespace std;

int main(int argc, const char** argv) {

	int devCount;
    cudaDeviceProp deviceProp;

    checkCudaErrors(cudaGetDeviceCount(&devCount));

    set<string> strs;

    stringstream ss;
    for(int devID = 0; devID < devCount; devID++) {
		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
		ss << deviceProp.major << "." << deviceProp.minor;
		strs.insert( ss.str() );
		ss.str("");
    }
    typedef set<string>::iterator strit;
    int distinctCount = strs.size();
    int currGpu = 0;
    for( strit i = strs.begin(); i != strs.end(); i++ ){
    	cout << *i;
    	currGpu++;
    	if(currGpu < distinctCount -1) {
    		cout << " ";
    	}
    }
	return 0;
}
