
#include <string>
#include <time.h>
#include <sstream>
#include <assert.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

using namespace std;
#define MAX_STACK_DEPTH 100
#define MAX_FUNC_NAME 512
#define ADDRESS_STRING_LEN 20


string dmg(const char* mangl) {
	unsigned long unmangleBufferLength = MAX_FUNC_NAME;
	char unmangleBuffer[MAX_FUNC_NAME];
	stringstream ss;
	int status;
	char* raw = abi::__cxa_demangle(mangl, unmangleBuffer,
			&unmangleBufferLength, &status);
	if (status == 0) {
		ss << raw;
	} else {
		ss << mangl;
	}
	return ss.str();
}

static const char * preamble = "nvlink error   : Undefined reference to '";
int main(int argc, char** argv) {
	static int lpreamble = strlen(preamble);
	if(argc < 2) {
		cout << "usage: " << argv[0] << " <manglorsh>" << endl;
		return 0;
	}
	for(int i = 1; i < argc; i++ ){
		if(strlen(argv[i]) > 1 && argv[i][0]== '_' && argv[i][1] == 'Z')  {
			cout << "template " << dmg(argv[i]) << ";"<<endl ;
		}
	}
	return 0;
}
