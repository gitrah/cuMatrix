/*
 * util.cc
 *
 *  Created on: Jul 23, 2012
 *      Author: reid
 */
#include <iostream>
#include <string>
#include "debug.h"
#include "util.h"
#include "caps.h"
#include "Matrix.h"
#include <sstream>
#include <algorithm>
#include <iterator>
#include <assert.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <limits>
#include <csignal>
#include <ctime>

#define DEBUG = true
//#define UTLTYTMPLT

extern ulong totalThreads;
extern ulong totalElements;
extern ExecCaps caps;

namespace mods {
	static const char * host = "host";
	static const char * device = "device";
	static const char * synced = "synced";
	static const char * neither = "neither";
};

template<typename T, typename U> string pp(pair<T, U>& p) {
	stringstream s1, s2, res;
	s1 << p.first;
	s2 << p.second;
	res << "( ";
	res << s1.str();
	res << ", ";
	res << s2.str();
	res << " )";
	return res.str();
}
template string pp<uint, uint>(	pair<uint, uint>&);
template string pp<float, float>(	pair<float, float>&);
template string pp<double, double>(	pair<double, double>&);

template<typename T> string b_util::pvec(vector<T>& v) {
	stringstream ss;
	typedef typename vector<T>::iterator iterator;

	ss << "{";
	for(iterator i = v.begin(); i < v.end(); i++) {
		ss << *i;
		if(i < v.end() -1) {
			ss << ", ";
		}
	}
	ss << "}";
	return ss.str();
}
template string b_util::pvec(vector<uint>& );

string b_util::pd3(const dim3& d3) {
	stringstream sx, sy, sz, res;
	sx << d3.x;
	sy << d3.y;
	sz << d3.z;
	res << "( ";
	res << sx.str();
	res << ", ";
	res << sy.str();
	res << ", ";
	res << sz.str();
	res << " )";
	return res.str();
}

string b_util::pd3(const dim3* d3) {
	stringstream sx, sy, sz, res;
	sx << d3->x;
	sy << d3->y;
	sz << d3->z;
	res << "( ";
	res << sx.str();
	res << ", ";
	res << sy.str();
	res << ", ";
	res << sz.str();
	res << " )";
	return res.str();
}

string b_util::pxd(const dim3& grid,const  dim3& block) {
	stringstream ss;
	ss << "grid " << pd3(grid) << " of block " << pd3(block);
	return ss.str();
}
string b_util::pexec(const dim3& grid,const  dim3& block, uint smem) {
	stringstream ss;
	ss << pxd(grid,block) << " with smem " << expNotation(smem);
	return ss.str();
}

const char * b_util::modStr(Modification lastMod) {
	switch ( lastMod) {
	case mod_host:
		return mods::host;
		break;
	case mod_device:
		return mods::device;
		break;
	case mod_synced:
		return mods::synced;
		break;
	case mod_neither:
		return mods::neither;
		break;
	default:
		break;
	}
	return mods::neither;
}

inline string trim_right_copy(const string& s,
		const string& delimiters = " \f\n\r\t\v") {
	return s.substr(0, s.find_last_not_of(delimiters) + 1);
}

inline string trim_left_copy(const string& s,
		const string& delimiters = " \f\n\r\t\v") {
	return s.substr(s.find_first_not_of(delimiters));
}

inline string trim_copy(const string& s,
		const string& delimiters = " \f\n\r\t\v") {
	return trim_left_copy(trim_right_copy(s, delimiters), delimiters);
}

vector<string> b_util::readLines(const char * path) {
	ifstream f;
	istream& fi = f;
	string currLine;
	ifstream::pos_type size;
	vector<string> lines;

	f.open(path);
	if (!f.is_open()) {
		outln( "no file '" << path << "'");
		return (lines);
	}
	while (!f.eof()) {
		getline(fi, currLine );
		if (currLine.length() > 0) {
			lines.push_back(currLine);
		}
	}
	return (lines);
}

void print(vector<string> lines) {
	vector<string>::iterator it = lines.begin();
	string currStr;
	while (it != lines.end()) {
		//currStr = *it;
		outln((*it).c_str());
		it++;
	}
}

vector<string> b_util::split(string s) {
	vector<string> res;
	istringstream iss(s);
	copy(istream_iterator<string>(iss),
			istream_iterator<string>(),
			back_inserter<vector<string> >(res));
	return (res);
}

template <typename T> T util<T>::minValue() {
	return numeric_limits<T>::min();
}

template <typename T> T util<T>::maxValue() {
	return numeric_limits<T>::max();
}

template <typename T> bool util<T>::almostEquals(T t1, T t2, T epsilon) {
	return ::abs(t2-t1) < epsilon;
}


template<typename T> string util<T>::pdm(const DMatrix<T>& md) {
	stringstream sadr, sm, sn, sz, sp, res;
	sadr << (md.elements ? md.elements : 0);
	sm << md.m;
	sn << md.n;
	sp << md.p;
	sz << md.m * md.p;
	res << "DMatrix(at ";
	res << sadr.str();
	res << " ";
	res << sm.str();
	res << "x";
	res << sn.str();
	res << "x";
	res << sp.str();
	res << " sz";
	res << sz.str();
	res << ")";
	return res.str();
}


typedef enum _OctDataType {
	unknown, matrix, scalar
} OctaveType;
const string otUnknown("unknown");
const string otMatrix("matrix");
const string otScalar("scalar");

string ot2str(OctaveType t) {
	switch (t) {
	case matrix:
		return (otMatrix);
	case scalar:
		return (otScalar);
	default:
		return (otUnknown);
	}
}

OctaveType str2OctaveType(string str) {
	if ("matrix" == str) {
		return (matrix);
	} else if ("scalar" == str) {
		return (scalar);
	}
	return (unknown);
}

template<typename T> int util<T>::release(map<string, Matrix<T>*>& theMap) {
//	map
	outln("map release");
	int cnt = 0;
	typedef typename map<string, Matrix<T>*>::iterator iterator;
	iterator it = theMap.begin();
	while (it != theMap.end()) {
		Matrix<T>* m = (*it).second;
		delete m;
		it++;
		cnt++;
	}
	return (cnt);
}

template<typename T> void addOctaveObject(map<string, Matrix<T>*>& theMap,
		typename map<string, Matrix<T>* >::iterator& it,
		string name, OctaveType elementType, Matrix<T>* currMat, int rows,
		int cols, bool matrixOwnsBuffer) {
	switch (elementType) {
		case scalar:
		case matrix: {
			outln( "adding " << ot2str(elementType).c_str() << " '" << name.c_str() << "' of dims " << rows << ", " << cols);
			outln( "first elem " << currMat->get(0,0));
			//outln( "created matrix of dims " << m.getRows() << ", " << m.getCols());
			theMap.insert(it, pair<string, Matrix<T>*>(name, currMat));
			break;
		}
		default: {
			outln( "received unknown type for " << name.c_str());
			break;
		}
	}
}

template<typename T> void util<T>::parseDataLine(string line, T* elementData,
		uint currRow, uint rows, uint cols,
		bool colMajor) {
	//outln(line);
	//outln("elementData " << elementData << ", startIdx " << startIdx);
	vector<string> tokens = b_util::split(line);
	int len = tokens.size();
	//outln(currRow <<" has " << len << " tokens");

	int idx = 0;
	const uint l = cols * rows - 1;
	T next;
	const uint startIdx = colMajor ? currRow * rows : currRow * cols;
	while (idx < len) {
		stringstream(tokens[idx]) >> next;
		//outln("token " << idx << ": " << next);
		if (colMajor) {
			elementData[(idx + startIdx) * rows % l] = next;
		} else {
			elementData[startIdx + idx] = next;
		}
		//outln("input " << spl[idx]);
		//outln("output " << elementData[startIdx + idx]);
		idx++;
	}
}

template <typename T> string util<T>::parry(int cnt, const T* arry) {
	stringstream ss;
	if(cnt == 1) {
		ss <<  b_util::pv1_3(arry[0]);
		return ss.str();
	}
	ss << "{";
	for(int i = 0; i < cnt;i++) {
		ss << b_util::pv1_3(arry[i]);
		if(i < cnt -1) {
			ss << ", ";
		}
	}
	ss << "}";
	return ss.str();
}

void b_util::announceTime() {
	time_t nowt = time(0);
	outln("\n\n" << ctime(&nowt) << "\n\n");
}
extern long clock(void);
void b_util::announceDuration(float exeTime) {
	clock_t dur = clock();
	outln("Sys: " << (((float)dur)/CLOCKS_PER_SEC) << "s");
	outln("Gpu: " << exeTime/1000. << "s");
}

void b_util::abortDumper(int level) {
	dumpStack("SIGABORT");
	exit(-1);
}
void b_util::fpeDumper(int level) {
	dumpStack("SIGFPE");
	exit(-1);
}
void b_util::segvDumper(int level) {
	dumpStack("SIGSEGV");
	exit(-1);
}

void b_util::handleSignals() {
	struct sigaction abortAction = { };
	struct sigaction fpeAction = { };
	struct sigaction segvAction = { };
	abortAction.sa_handler = abortDumper;
	fpeAction.sa_handler = fpeDumper;
	segvAction.sa_handler = segvDumper;

	if (sigaction(SIGABRT, &abortAction, NULL) < 0) {
		perror("sigaction(SIGABRT)");
	}
	if (sigaction(SIGFPE, &fpeAction, NULL) < 0) {
		perror("sigaction(FPE)");
	}
	if (sigaction(SIGSEGV, &segvAction, NULL) < 0) {
		perror("sigaction(SEGV)");
	}

}

int b_util::getCount(int argc, const char** argv,int defaultCount) {
	int count= 1;
	char *countStr= null;
	getCmdLineArgumentString(argc, (const char **)argv, "count", &countStr);
	if(countStr) {
		count = atoi(countStr);
		if(count  < 0) {
			fprintf(stderr, "count %d is invalid, will use default count of %d.\n", count,defaultCount);
			count = defaultCount;
		}
	} else {
		count = defaultCount;
	}
	return count;
}

template<typename T> T  b_util::getParameter(int argc, const char** argv, const char* parameterName, T defaultValue) {
	T value = defaultValue;
	char *valueStr= null;
	getCmdLineArgumentString(argc, (const char **)argv, parameterName, &valueStr);
	if(valueStr) {
		value = (T) atof(valueStr);
	}
	return value;
}

template int b_util::getParameter<int>(int, const char**, const char*, int);

time_t b_util::timeReps( void(*fn)(), int reps) {
	time_t nowt = time(0);
	for(int i = 0; i < reps; i++ ){
		 (*fn)();
	}
	return time(0)-nowt;
}

void b_util::randSequence(vector<uint>& ret, uint count, uint start) {
	uint* arry = new uint[count];
	for(uint i = 0; i < count; i++ ) {
		arry[i] = start + i;
	}
	vector<uint> st(arry, arry+count);
	uint idx;
	while(!st.empty()) {
		idx = rand() % st.size();
		ret.insert(ret.end(), st[idx]);
		st.erase(st.begin() + idx);
	}
	delete [] arry;
}

template<typename T> void b_util::toArray(vector<T>& v, T* arry, int start, int count) {
	outln("toArray start " << start << ", count " << count);
	typedef typename vector<T>::iterator iterator;
	int idx = 0;
	for(iterator i = v.begin() + start; i < v.end() && idx < count ; i++) {
		outln("idx " << idx << ", *i " << *i);
		arry[idx++] = *i;
	}
}

template void b_util::toArray(vector<uint>& v, uint* arry, int, int);

double b_util::diffclock(clock_t clock1, clock_t clock2) {
	double diffticks = clock1 - clock2;
	double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
	return diffms;
}

const string Name("name:");
const string Type("type:");
const string Rows("rows:");
const string Columns("columns:");

template<typename T> map<string, Matrix<T>* > util<T>::parseOctaveDataFile(
		const char * path, bool colMajor, bool matrixOwnsBuffer) {
	map<string, Matrix<T>*> dataMap;
	typedef typename map<string, Matrix<T>*>::iterator iterator;
	iterator it = dataMap.begin();
	OctaveType elementType = unknown;
	unsigned long idx = string::npos;
	uint rows = 0, cols = 0, currRow = 0;
	clock_t lastChunk = 0, lastTime = 0;
	string name = "";
	double deltaMs = 0;
	Matrix<T>* currMat = null;
	bool colsSet = false, rowsSet = false;
	vector<string> lines = b_util::readLines(path);
	vector<string>::iterator itLines = lines.begin();
	while (itLines != lines.end()) {
		string line = trim_copy(*itLines);
		if (line.find("#") == 0) { // start of var declaration
			idx = line.find(Name);
			if (idx != string::npos) {
				if (name.length() > 0 && elementType != unknown) {
					// add previous object
					addOctaveObject(dataMap, it, name, elementType, currMat,
							rows, cols, matrixOwnsBuffer);
					outln(
							"adding " << name.c_str() << " took " << b_util::diffclock(clock(),lastTime) << "s");
					outln("now " << dataMap.size() << " elements in dataMap");
					currRow = 0;
				}
				name = trim_copy(line.substr(idx + Name.size()));
				outln( "found " << name.c_str());
				lastTime = clock();
				rows = cols = 0;
				colsSet = rowsSet = false;
				elementType = unknown;
				currMat = null;
			} else {
				idx = line.find(Type);
				if (idx != string::npos) {
					elementType = str2OctaveType(
							trim_copy(line.substr(idx + Type.size())));
					outln(name.c_str() << " has type " << ot2str(elementType).c_str());
					if (elementType == scalar) {
						currMat = new Matrix<T>(1, 1, matrixOwnsBuffer);
						currMat->invalidateDevice();
						rows = cols = 1;
					}
				} else {
					idx = line.find(Rows);
					if (idx != string::npos) {
						stringstream(
								trim_copy(line.substr(idx + Rows.size())))
								>> rows;
						outln(name.c_str() << " has rows " << rows);
						rowsSet = true;
					} else {
						idx = line.find(Columns);
						if (idx != string::npos) {
							stringstream(
									trim_copy(
											line.substr(idx + Columns.size())))
									>> cols;
							outln(name.c_str() << " has cols " << cols);
							colsSet = true;
						}
					}
					if (rowsSet && colsSet) {
						outln(
								"creating buffer for " << name.c_str() << " of size " << (rows * cols));
						currMat = new Matrix<T>(rows, cols, matrixOwnsBuffer);
						currMat->invalidateDevice();
						outln("got currMat " << currMat->toShortString());
						outln("got buffer " << currMat->elements);
					}
				}
			}
		} else if (!line.empty()) {
			//outln("expected data line");
			switch (elementType) {
			case matrix: {
				assert(
						!name.empty() && elementType != unknown && rowsSet && colsSet);
				// it's a data line (row)
				util::parseDataLine(line, currMat->elements, currRow, rows, cols,
						colMajor);
				currRow++;
				if ((currRow % 100) == 0) {
					clock_t now = clock();
					if (lastChunk != 0) {
						deltaMs = b_util::diffclock(now, lastChunk);
						outln(
								"on " << currRow << "/" << rows << " at " << (deltaMs / 100) << " ms/row");
					} else {
						outln("on " << currRow << "/" << rows);
					}
					lastChunk = now;
				}
				break;
			}
			case scalar: {
				stringstream(trim_copy(line)) >> currMat->elements[0];
				break;
			}

			default:
				outln("unknown type " << elementType);
				break;
			}
		}
		//outln("next");
		itLines++;
	}

	// add last obj
	if (!name.empty() && elementType != unknown) {
		addOctaveObject(dataMap, it, name, elementType, currMat, rows, cols,
				matrixOwnsBuffer);
	}

	return dataMap;
}

/*
template<typename T> void util<T>::copyRange(T* src, T* targ, uint tarOff,
		uint srcOff, size_t count) {
	memcpy(targ + tarOff, src + srcOff, count);
}
*/

string b_util::toStr(const uintPair& p) {
	stringstream s1, s2;
	s1 << p.first;
	s2 << p.second;
	stringstream res;
	res << "(";
	res << s1.str();
	res << ", ";
	res << s2.str();
	res << ")";
	return res.str();
}

//////////////////////////
//
// IndexArray
//
//////////////////////////
//

IndexArray::IndexArray() :
		indices(null), count(0), owner(true) {
}

IndexArray::IndexArray(const IndexArray & o) :
		indices(o.indices), count(o.count), owner(false) {
}

IndexArray::IndexArray(uint* _indices, uint _count, bool _owner) :
		indices(_indices), count(_count), owner(_owner) {
}

IndexArray::IndexArray(uint idx1, uint idx2) :
		count(2), owner(true){
	indices = new uint[2];
	indices[0] = idx1;
	indices[1] = idx2;
}

uintPair IndexArray::toPair() const {
	assert(count==2);
	return uintPair(indices[0], indices[1]);
}

IndexArray::~IndexArray() {
	if (indices != null && owner) {
		delete[] indices;
	}
}

string IndexArray::toString() const {
	stringstream ssout;
	ssout << "IdxArr" << (owner ? "+" : "-") << "(" << count << ")[ ";

	for (uint i = 0; i < count; i++) {
		ssout << indices[i];
		if(i < count -1)
			ssout << ", ";
	}
	ssout << " ]";
	return ssout.str();

}

template<typename T> bool Math<T>::aboutEq(T x1, T x2, T epsilon) {
	return (abs(x2 - x1) <= epsilon);
}
template bool Math<float>::aboutEq(float x1, float x2, float epsilon);
template bool Math<double>::aboutEq(double x1, double x2, double epsilon);


string b_util::expNotation(long val) {
	char buff[256];
	double factor=1.;
	string units = "";
	if(val >= Giga) {
		factor = 1./Giga;
		units = "G";
	} else if(val >= Mega) {
		factor = 1./Mega;
		units = "M";
	} else if(val >= Kila) {
		factor = 1./Kila;
		units = "K";
	} else {
		units = " bytes";
	}
	sprintf(buff, "%2.3g", val * factor);
	stringstream ss;
	ss << buff << units;
	return ss.str();
}

#define MAX_STACK_DEPTH 100
#define MAX_FUNC_NAME 512
#define ADDRESS_STRING_LEN 20

string b_util::stackString(string msg, int start, int depth) {
	//do_backtrace();
	void* addrlist[MAX_STACK_DEPTH];
	char address[ADDRESS_STRING_LEN + 1];
	char* addressStrStart = null, *addressStrEnd = null;
	ulong unmangleBufferLength = MAX_FUNC_NAME;
	char unmangleBuffer[MAX_FUNC_NAME];
	char* currentUnmangled = unmangleBuffer;
	depth = depth < 0 || depth > MAX_STACK_DEPTH ? MAX_STACK_DEPTH : depth;
	stringstream ss;

	int stElementCount = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

	if (stElementCount == 0) {
		ss << "??? no stack trace available";
		return ss.str();
	}

	char** stElements = backtrace_symbols(addrlist, stElementCount);
	int endIndex = start + depth;
	endIndex = endIndex > stElementCount ? stElementCount : endIndex;
	int firstElement = stElementCount >= start ? start : 0;
	for (int i = firstElement; i < endIndex; i++) {
		char *niceFnName = null, *addressOffset = null, *addressOffsetEnd = null;
		//cout << niceFnName << endl;
		// find parentheses and +address offset surrounding the mangled name:
		// ./module(function+0x15c) [0x8048a6d]
		//cout << "mangled:  " << stElements[i] << endl;
		for (char *stElementChar = stElements[i]; *stElementChar; ++stElementChar) {
			if (*stElementChar == '(')
				niceFnName = stElementChar;
			else if (*stElementChar == '+')
				addressOffset = stElementChar;
			else if (*stElementChar == ')' && addressOffset) {
				addressOffsetEnd = stElementChar;
			} else if(*stElementChar == '[') {
				addressStrStart = stElementChar + 1;
			} else if(*stElementChar == ']') {
				addressStrEnd = stElementChar - 1;
				if(addressStrEnd && addressStrStart) {
					int len = addressStrEnd - addressStrStart;
					len = len < ADDRESS_STRING_LEN ? len : ADDRESS_STRING_LEN;
					memcpy(address, addressStrStart, len);
					address[len+1] = 0;
				}
				break;
			}
		}

		if (niceFnName && addressOffset && addressOffsetEnd && niceFnName < addressOffset) {
			*niceFnName = *addressOffset = *addressOffsetEnd = '\0';
			niceFnName++;
			addressOffset++;
			int status;
			char* raw = abi::__cxa_demangle(niceFnName, unmangleBuffer, &unmangleBufferLength,
					&status);
			if (status == 0) {
				currentUnmangled = raw;
				ss << (i == firstElement ? msg.c_str()  : "  at ") << currentUnmangled << "   " << addressOffset;
				if(depth > 1) {
					ss << endl;
				}
			} else {
				ss << "  " << stElements[i] << " : " << niceFnName << "()+" <<	addressOffset << endl;
			}
		} else {
			ss << "  " << stElements[i] << endl;
		}
	}

	free(stElements);
	if(unmangleBuffer != currentUnmangled)
		free(currentUnmangled);
	return ss.str();
}

string b_util::stackString(int start, int depth) {
    stringstream ss;
     ss << "";
     return stackString(ss.str(), start, depth);
}

string b_util::caller() {
	return stackString(4, 1);
}

string b_util::caller2() {
	return stackString(4, 2);
}

string b_util::callerN(int n) {
	return stackString(4, n < 1 ? 1 : n);
}

void b_util::dumpStack(int start, int depth) {
	cout << stackString( start, depth) << endl;
}
void b_util::dumpStack(  int depth) {
	cout << stackString( 4, depth) << endl;
}
void b_util::dumpStack(const char * msg,  int depth) {
	cout << msg << endl << stackString( 4, depth) << endl;
}
void b_util::dumpStackIgnoreHere(int depth) {
	cout << stackString(5, depth) << endl;
}

void b_util::dumpStackIgnoreHere(string msg, int depth) {
	cout << msg << endl<< stackString(1,depth) << endl;
}

void b_util::waitEnter() {
	char cr;
	cout << "Enter to continue:";
    cin.get(cr);//
}

uint b_util::nextPowerOf2(uint x) {
	if(x < 2) {
		return 2;
	}
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

bool b_util::isPow2(uint x) {
    return ((x&(x-1))==0);
}

//template <> double ConjugateGradient<double>::MinPositiveValue = 4.9E-324;
//template <> float ConjugateGradient<float>::MinPositiveValue = 1.4E-45;


uint b_util::largestMutualFactor(uint count, uint args[]) {
	//va_list arguments;
	//va_start( arguments, count );
	//uint args[count];
	uint ret = 2;
	bool fin = false;
	uint currarg;
	//for(uint i = 0; i < count; i++ ) args[i] = va_arg(arguments, uint);
	do{
		for(uint i = 0; i < count; i++ ) {
			currarg = args[i];
			cout << i << ", arg " << currarg << endl;
			if( currarg % ret != 0 ) {
				cout << "% " << ret << " !=0" << endl;
				fin = true;
				break;
			}
		}
		if(!fin)ret *= 2;
	} while (!fin);

	return ret/2;
}


void b_util::execContext(int threads, uint count, dim3& dBlocks,
		dim3& dThreads) {
	if (threads % WARP_SIZE != 0) {
		outln(
				"WARN: " << threads << " is not a multiple of the warp size (32)");
	}
	uint blocks = (count + threads - 1) / threads;
	dBlocks.y = dBlocks.z = 1;
	dBlocks.x = blocks;
	dThreads.y = dThreads.z = 1;
	dThreads.x = threads;
	totalThreads += blocks * threads;
	totalElements += count;
	if (debugExec)
		outln(
				"contxt of " << blocks << " blks of " << threads << " threads for count of " << count);
}

void b_util::execContext(int threads, uint count, uint spacePerThread,
		dim3& dBlocks, dim3& dThreads, uint& smem) {
	if (threads % WARP_SIZE != 0) {
		outln(
				"WARN: " << threads << " is not a multiple of the warp size (32)");
	}
	uint limitSmemThreads = caps.memSharedPerBlock / spacePerThread;
	threads = (int) MIN((uint)threads, limitSmemThreads);
	smem = MIN(caps.memSharedPerBlock, threads * spacePerThread);
	uint blocks = (count + threads - 1) / threads;

	dBlocks.y = dBlocks.z = 1;
	dBlocks.x = blocks;
	dThreads.y = dThreads.z = 1;
	dThreads.x = threads;
	totalThreads += blocks * threads;
	totalElements += count;
	if(debugExec)outln(
			"for " << spacePerThread << " bytes per thread, smem " << smem << " bytes, grid of " << blocks << " block(s) of " << threads << " threads");
}

void b_util::execContextSmem(int threads, uint count, dim3& dBlocks,
		dim3& dThreads) {
	if (threads % WARP_SIZE != 0) {
		outln(
				"WARN: " << threads << " is not a multiple of the warp size (32)");
	}
	uint blocks = (count + threads - 1) / threads;
	dBlocks.y = dBlocks.z = 1;
	dBlocks.x = blocks;
	dThreads.y = dThreads.z = 1;
	dThreads.x = threads;
	totalThreads += blocks * threads;
	totalElements += count;
	if (debugExec)
		outln("contxt of " << blocks << " blks of " << threads << " threads");
}

const char* b_util::lastErrStr() {
	return _cudaGetErrorEnum(cudaGetLastError());
}

void b_util::dumpAnyErr(string file, int line) {
	if(cudaDeviceSynchronize() != cudaSuccess) {
		cout << _cudaGetErrorEnum(cudaGetLastError()) << " at " << file.c_str() << ":"<< line << endl;
	}  else if(debugVerbose) {
		cout << "OK at " << file.c_str() << ":"<< line << endl;
	}
}

void b_util::dumpError(cudaError_t err) {
	if(err != cudaSuccess) {
		cout << _cudaGetErrorEnum(err)  << endl;
	}
}
double b_util::usedMemRatio() {
	size_t freeMemory, totalMemory;
	cudaMemGetInfo(&freeMemory,&totalMemory);
	if(debugMem) outln("freeMemory " << freeMemory << ", total " << totalMemory);
	return 100 * (1 - freeMemory * 1. / totalMemory);
}

void b_util::usedDmem() {
	cout << "Memory " << usedMemRatio() << "% used\n";
}

void b_util::_checkCudaError(string file, int line, cudaError_t val) {
	if(val != cudaSuccess) {
		stringstream ss;
		ss << "CuError:  ("<< val<< ") " << _cudaGetErrorEnum(val) << " at " << file.c_str() << ":" << line << endl;
		cout << ss.str() << endl;
		assert(false);
	}
}
void b_util::_printCudaError(string file, int line, cudaError_t val) {
	if(val != cudaSuccess) {
		stringstream ss;
		ss << "CuError:  ("<< val<< ") " << _cudaGetErrorEnum(val) << " at " << file.c_str() << ":" << line << endl;
		b_util::dumpStackIgnoreHere(ss.str());
	}
}

template int util<Matrix<float> >::deleteMap( map<string, Matrix<float>* >&  m);
template int util<Matrix<double> >::deleteMap( map<string, Matrix<double>* >&  m);
template <typename T> int util<T>::deleteMap( map<string, T*>&  m) {
	typedef typename map<string, T*>::iterator iterator;
	iterator it = m.begin();
	while(it != m.end()) {
		delete (*it).second;
		it++;
	}
	return m.size();
}

template int util<float>::deleteVector( vector<float* >& );
template int util<double>::deleteVector( vector<double* >&);
template <typename T> int util<T>::deleteVector( vector<T*>&  v) {
	typedef typename vector<T*>::iterator iterator;
	iterator it = v.begin();
	while(it != v.end()) {
		delete (*it);
		it++;
	}
	return v.size();
}

template <typename T> int util<T>::cudaFreeVector( vector< T* >&  v, bool device) {
	typedef typename vector<T*>::iterator iterator;
	iterator it = v.begin();
	while(it != v.end()) {
		checkCudaError( device ? cudaFree (*it) : cudaFreeHost(*it));
		it++;
	}
	return v.size();
}


void b_util::syncGpu(const char * msg) {
	if(msg)cout << msg << ": ";
	checkCudaError(cudaDeviceSynchronize());
}

template <typename T> cudaError_t util<T>::copyRange(T* targ, ulong targOff, T* src, ulong srcOff, ulong count) {
	cudaError_t err = cudaMemcpy(targ + targOff, src + srcOff, count, cudaMemcpyHostToHost);
	Matrix<T>::HHCopied++;
	checkCudaError(err);
	return err;
}

template <typename T> T  util<T>::sumCPU(T* vals, ulong count) {
	T sum =0;
	while(count-- != 0) {
		sum += vals[count-1];
	}
	return sum;
}

CuTimer::CuTimer(cudaStream_t stream) {
    checkCudaError(cudaEventCreate(&evt_start));
    checkCudaError(cudaEventCreate(&evt_stop));
    status = ready;
    this->stream = stream;
    outln("created CuTimer " << this);
}

CuTimer::~CuTimer() {
    outln("~CuTimer " << this);
	checkCudaError(cudaEventDestroy(evt_start));
	checkCudaError(cudaEventDestroy(evt_stop));
}

void CuTimer::start() {
    outln("starting CuTimer " << this);
	if(status != ready) {
		dthrow(TimerAlreadyStarted());
	}
	status = started;
    checkCudaError(cudaEventRecord(evt_start, stream));
}

float CuTimer::stop() {
    outln("stopping CuTimer " << this);
	if(status != started) {
		dthrow(TimerNotStarted());
	}
	status = ready;
    checkCudaErrors(cudaEventRecord(evt_stop, stream));
    checkCudaErrors(cudaEventSynchronize(evt_stop));
    float exeTime;
    checkCudaErrors(cudaEventElapsedTime(&exeTime, evt_start, evt_stop));
    return exeTime;
}

template <> __device__ __host__ float util<float>::epsilon() { return 1e-6;}
template <> __device__ __host__ double util<double>::epsilon() { return 1e-10;}

template struct util<float>;
template struct util<double>;
