/*
 * util.cc
 *
 *  Created on: Jul 23, 2012
 *      Author: reid
 */

#include <list>
#include <algorithm>
#include <iostream>

#include <iostream>
#include <string>
#include "debug.h"
#include "util.h"
#include "caps.h"
#include "CuMatrix.h"
#include <algorithm>
#include <iterator>
#include <assert.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <csignal>
#include <ctime>
#include <sys/time.h>

#define DEBUG = true
//#define UTLTYTMPLT

extern ulong totalThreads;
extern ulong totalElements;


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
template string pp<uint, uint>(pair<uint, uint>&);
template string pp<float, float>(pair<float, float>&);
template string pp<double, double>(pair<double, double>&);
template string pp<ulong, ulong>(pair<ulong, ulong>&);


template<typename T> string b_util::pvec(vector<T>& v) {
	stringstream ss;
	typedef typename vector<T>::iterator iterator;

	ss << "{";
	for (iterator i = v.begin(); i < v.end(); i++) {
		ss << *i;
		if (i < v.end() - 1) {
			ss << ", ";
		}
	}
	ss << "}";
	return ss.str();
}
template string b_util::pvec(vector<uint>&);


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

string b_util::pd3(const dim3* d3, const char* msg) {
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
	res << " ) " << msg;
	return res.str();
}

string b_util::pd3(const dim3* d3, string msg) {
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
	res << " ) " << msg;
	return res.str();
}

string b_util::pxd(const dim3& grid, const dim3& block) {
	stringstream ss;
	ss << "grid " << pd3(grid) << " of block " << pd3(block);
	return ss.str();
}

string b_util::sPtrAtts(const cudaPointerAttributes& atts) {
	stringstream ss;
	ss << "gpu " << atts.device;
	if (atts.devicePointer != null) {
		ss << " dev ptr " << atts.devicePointer;
	} else {
		ss << " host ptr " << atts.hostPointer;
	}

	ss
			<< (atts.memoryType == cudaMemoryTypeHost ?
					" host type" : " device type");
	ss << "\n";
	return ss.str();
}

string b_util::pexec(const dim3& grid, const dim3& block, uint smem) {
	stringstream ss;
	ss << pxd(grid, block) << " with smem " << expNotation(smem);
	return ss.str();
}


inline string trim_right_copy(const string& s, const string& delimiters =
		" \f\n\r\t\v") {
	return s.substr(0, s.find_last_not_of(delimiters) + 1);
}

inline string trim_left_copy(const string& s, const string& delimiters =
		" \f\n\r\t\v") {
	return s.substr(s.find_first_not_of(delimiters));
}

inline string trim_copy(const string& s, const string& delimiters =
		" \f\n\r\t\v") {
	return trim_left_copy(trim_right_copy(s, delimiters), delimiters);
}

int b_util::currDevice() {
	int dev;
	checkCudaErrors(cudaGetDevice(&dev));
	return dev;
}

int b_util::countLines(const char* path, int headerLength) {
	ifstream f;
	string currLine;
	ifstream::pos_type size;
	int lines = 0;

	f.open(path);
	if (!f.is_open()) {
		outln("no file '" << path << "'");
		return (lines);
	}
	while (!f.eof()) {
		getline(f, currLine);
		if (currLine.length() > 0) {
			lines++;
		}
	}
	f.close();
	return (lines);
}

vector<string> b_util::readLines(const char * path) {
	ifstream f;
	istream& fi = f;
	string currLine;
	ifstream::pos_type size;
	vector<string> lines;

	f.open(path);
	if (!f.is_open()) {
		outln("no file '" << path << "'");
		return (lines);
	}
	int lc = 0;
	int dg = 0;
	while (!f.eof()) {
		getline(fi, currLine);
		lc++;
		if (lc % 1000 == 0) {
			cout << dg;
			dg = (dg + 1) % 10;
		}
		if (currLine.length() > 0) {
			lines.push_back(currLine);
		}
	}
	cout << "\n";
	f.close();
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
	copy(istream_iterator<string>(iss), istream_iterator<string>(),
			back_inserter<vector<string> >(res));
	return (res);
}


template<typename T> bool util<T>::almostEquals(T t1, T t2, T epsilon) {
	return ::abs(t2 - t1) < epsilon;
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

enum OctaveType {
	unknown, matrix, scalar
};
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

template<typename T> int util<T>::release(map<string, CuMatrix<T>*>& theMap) {
//	map
	outln("map release");
	int cnt = 0;
	typedef typename map<string, CuMatrix<T>*>::iterator iterator;
	iterator it = theMap.begin();
	while (it != theMap.end()) {
		CuMatrix<T>* m = (*it).second;
		delete m;
		it++;
		cnt++;
	}
	return (cnt);
}

template<typename T> void addOctaveObject(map<string, CuMatrix<T>*>& theMap,
		typename map<string, CuMatrix<T>*>::iterator& it, string name,
		OctaveType elementType, CuMatrix<T>* currMat, bool matrixOwnsBuffer) {
	switch (elementType) {
	case scalar:
	case matrix: {
		currMat->invalidateDevice();
		currMat->syncBuffers();
		outln(
				"adding " << ot2str(elementType).c_str() << " '" << name.c_str() << "' of dims " << currMat->m << ", " << currMat->n);
		outln("first elem " << currMat->get(0,0));
		outln("first h elem " << currMat->elements[0]);
		outln("last,0 elem " << currMat->get(currMat->m-1,0));
		//outln( "created matrix of dims " << m.getRows() << ", " << m.getCols());
		theMap.insert(it, pair<string, CuMatrix<T>*>(name, currMat));
		break;
	}
	default: {
		outln("received unknown type for " << name.c_str());
		break;
	}
	}
}

template<typename T> void util<T>::parseDataLine(string line, T* elementData,
		uint currRow, uint rows, uint cols, bool colMajor) {
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

#define MAX_COLS 4096
double lastRow[MAX_COLS];
bool warnedAlready[MAX_COLS];
template<typename T> void util<T>::parseCsvDataLine(const CuMatrix<T>* x,
		int currLine, string line, const char* sepChars) {
	//outln("parseCsvDataLine " << line);
	if(currLine ==0) {
		memset(lastRow,0,4096*sizeof(double));
		memset(warnedAlready,0,4096*sizeof(bool));
	}
	const char* cline = line.c_str();
	char * pch;
	int idx = 0;
	double dblNext;
	T next;
	char* copy = (char*) malloc(line.size() + 1);
	char* lcopy = copy;
	strcpy(lcopy, cline);
	pch = strtok(lcopy, sepChars);
	while (pch != null) {
		dblNext = atof(pch);
		if(currLine > 0 && idx < MAX_COLS && sizeof(T) < 8) {
			double diff = fabs(lastRow[idx] - dblNext);
			next = dblNext;
			T tdiff = (T) lastRow[idx] - next;
			if(diff != 0 && !warnedAlready[idx] &&
					( ::isinf(diff/tdiff) ||  fabs( 1- fabs(diff/tdiff)) > .05 )) {
				outln("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n" <<
						"WARNING (once per column) current type parameter results in loss of precision at " <<
						currLine << ", " << idx << ": " << (diff/tdiff) <<
						"\n\n!!!!!!!!!!!!!!!!!!!!!!!!\n\n") ;
				warnedAlready[idx] = true;
			}
		} else {
			next=atof(pch);
		}
		lastRow[idx] = dblNext;
		x->elements[currLine * x->p + idx++] = next;
		pch = strtok(null, sepChars);
	}
	free(copy);
}

template<typename T> inline string pv1_3(T v) {
	char buff[5];
	sprintf(buff,"%1.3g", (double)v);
	return string(buff);
}
template<> inline string pv1_3<ulong>(ulong v) {
	char buff[5];
	sprintf(buff,"%1.3lu", v);
	return string(buff);
}

template<typename T> string util<T>::parry(const T* arry, int cnt) {
	stringstream ss;
	if (cnt == 1) {
		ss << pv1_3<T>(arry[0]);
		return ss.str();
	}
	ss << "{";
	for (int i = 0; i < cnt; i++) {
		ss << pv1_3<T>(arry[i]);
		if (i < cnt - 1) {
			ss << ", ";
		}
	}
	ss << "}";
	return ss.str();
}

string parry(const int* arry, int cnt) {
	stringstream ss;
	if (cnt == 1) {
		ss << arry[0];
		return ss.str();
	}
	ss << "{";
	for (int i = 0; i < cnt; i++) {
		ss << arry[i];
		if (i < cnt - 1) {
			ss << ", ";
		}
	}
	ss << "}";
	return ss.str();
}


long b_util::nowMillis() {
	struct timeval tv;
	gettimeofday(&tv, null);
	return (tv.tv_sec * 1000000 + tv.tv_usec) / 1000;
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

int b_util::getCount(int argc, const char** argv, int defaultCount) {
	return getParameter<int>(argc, argv, "count", defaultCount);
}

int b_util::getStart(int argc, const char** argv, int defaultStart) {
	return getParameter<int>(argc, argv, "start", defaultStart);
}

string b_util::getPath(int argc, const char** argv, const char* defaultPath) {
	char *pathStr = null;
	string path("");
	getCmdLineArgumentString(argc, (const char **) argv, "path", &pathStr);
	if (pathStr) {
		return string(pathStr);
	}
	return path;
}

template<typename T> T b_util::getParameter(int argc, const char** argv,
		const char* parameterName, T defaultValue) {
	T value = defaultValue;
	char *valueStr = null;
	getCmdLineArgumentString(argc, (const char **) argv, parameterName,
			&valueStr);
	if (valueStr) {
		value = (T) atof(valueStr);
	}
	return value;
}

template int b_util::getParameter<int>(int, const char**, const char*, int);

time_t b_util::timeReps(void (*fn)(), int reps) {
	time_t nowt = time(0);
	for (int i = 0; i < reps; i++) {
		(*fn)();
	}
	return time(0) - nowt;
}

void b_util::randSequence(vector<uint>& ret, uint count, uint start) {
	uint* arry = new uint[count];
	for (uint i = 0; i < count; i++) {
		arry[i] = start + i;
	}
	vector<uint> st(arry, arry + count);
	uint idx;
	while (!st.empty()) {
		idx = rand() % st.size();
		ret.insert(ret.end(), st[idx]);
		st.erase(st.begin() + idx);
	}
	delete[] arry;
}

template<typename T> void b_util::toArray(T* arry, const vector<T>& v, int start,
		int count) {
	outln("toArray start " << start << ", count " << count);
	typedef typename vector<T>::const_iterator iterator;
	int idx = 0;
	for (iterator i = v.begin() + start; i < v.end() && idx < count; i++) {
		//outln("idx " << idx << ", *i " << *i);
		arry[idx++] = *i;
	}
}

template void b_util::toArray(uint* arry, const vector<uint>& v, int, int);

double b_util::diffclock(clock_t clock1, clock_t clock2) {
	double diffticks = clock1 - clock2;
	double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
	return diffms;
}

const string Name("name:");
const string Type("type:");
const string Rows("rows:");
const string Columns("columns:");

template<typename T> map<string, CuMatrix<T>*> util<T>::parseOctaveDataFile(
		const char * path, bool colMajor, bool matrixOwnsBuffer) {
	map<string, CuMatrix<T>*> dataMap;
	typedef typename map<string, CuMatrix<T>*>::iterator iterator;
	iterator it = dataMap.begin();
	OctaveType elementType = unknown;
	unsigned long idx = string::npos;
	uint rows = 0, cols = 0, currRow = 0;
	clock_t lastChunk = 0, lastTime = 0;
	string name = "";
	double deltaMs = 0;
	CuMatrix<T>* currMat = null;
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
							matrixOwnsBuffer);
					outln(
							"adding " << name.c_str() << " took " << b_util::diffclock(clock(),lastTime) << "s");
					outln("now " << dataMap.size() << " elements in dataMap");
					currRow = 0;
				}
				name = trim_copy(line.substr(idx + Name.size()));
				outln("found " << name.c_str());
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
					outln(
							name.c_str() << " has type " << ot2str(elementType).c_str());
					if (elementType == scalar) {
						currMat = new CuMatrix<T>(1, 1, matrixOwnsBuffer,true);
						currMat->invalidateDevice();
						rows = cols = 1;
					}
				} else {
					idx = line.find(Rows);
					if (idx != string::npos) {
						stringstream(trim_copy(line.substr(idx + Rows.size())))
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
						currMat = new CuMatrix<T>(rows, cols, matrixOwnsBuffer,true);
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
						!name.empty() && elementType != unknown && rowsSet
								&& colsSet);
				// it's a data line (row)
				util::parseDataLine(line, currMat->elements, currRow, rows,
						cols, colMajor);
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
		addOctaveObject(dataMap, it, name, elementType, currMat,
				matrixOwnsBuffer);
	}

	return dataMap;
}

template<typename T> map<string, CuMatrix<T>*> util<T>::parseCsvDataFile(
		const char * path, const char* sepChars, bool colMajor,
		bool matrixOwnsBuffer, bool hasXandY) {
	CuTimer timer;
	timer.start();
	map<string, CuMatrix<T>*> dataMap;
	typedef typename map<string, CuMatrix<T>*>::iterator iterator;
	iterator it = dataMap.begin();
	OctaveType elementType = unknown;
	//unsigned long idx = string::npos;
	int currLine = 0;
	uint cols = 0;//uint rows = 0, cols = 0, currRow = 0;

	//clock_t lastChunk = 0, lastTime = 0;
	string name = "";
	//double deltaMs = 0;
	//bool colsSet = false, rowsSet = false;
	outln("reading " << path);
	vector<string> lines = b_util::readLines(path);
	int lineCount = lines.size();
	outln(
			"reading " << lineCount << " lines from " << path << " took " << timer.stop()/1000 << "s");

	vector<string>::iterator itLines = lines.begin();
	// skip header

	string line = trim_copy(*itLines);
	const char* cline = line.c_str();
	char * pch;
	char lcopy[line.size() + 10];
	strcpy(lcopy, cline);
	printf("lcopy %s\n", lcopy);
	pch = strtok(lcopy, sepChars);
	while (pch != null) {
		printf("%s\n", pch);
		pch = strtok(null, sepChars);
		cols++;
	}
	elementType = matrix;
	itLines++;
	CuMatrix<T>* mat = new CuMatrix<T>(lineCount - 1, cols, matrixOwnsBuffer, true);
	outln("mat " << mat->toShortString());
	while (itLines != lines.end()) {

		string line = trim_copy(*itLines);
		//const char* cline = line.c_str();
		if (!line.empty()) {
			//outln("expected data line");
			switch (elementType) {
			case matrix: {
				// it's a data line (row)
				util::parseCsvDataLine(mat, currLine, line, sepChars);
				lineCount++;
				break;
			}
			default:
				outln("unknown type " << elementType);
				break;
			}
		}
		//outln("next");
		itLines++;
		currLine++;
	}
	if (hasXandY) {
		addOctaveObject(dataMap, it, "xAndY", elementType, mat,
				matrixOwnsBuffer);
	} else {
		addOctaveObject(dataMap, it, "x", elementType, mat, matrixOwnsBuffer);
	}

	return dataMap;
}

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


template<typename T> bool Math<T>::aboutEq(T x1, T x2, T epsilon) {
	return (abs(x2 - x1) <= epsilon);
}
template bool Math<float>::aboutEq(float x1, float x2, float epsilon);
template bool Math<double>::aboutEq(double x1, double x2, double epsilon);
template bool Math<ulong>::aboutEq(ulong x1, ulong x2, ulong epsilon);

string b_util::expNotation(long val) {
	char buff[256];
	double factor = 1.;
	string units = "";
	if (val >= Giga) {
		factor = 1. / Giga;
		units = "G";
	} else if (val >= Mega) {
		factor = 1. / Mega;
		units = "M";
	} else if (val >= Kilo) {
		factor = 1. / Kilo;
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

string b_util::unmangl(const char* mangl) {
	ulong unmangleBufferLength = MAX_FUNC_NAME;
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
	//printf("stackString enter\n");
	if (stElementCount == 0) {
		ss << "??? no stack trace available";
		return ss.str();
	}

	char** stElements = backtrace_symbols(addrlist, stElementCount);
	int endIndex = start + depth;
	endIndex = endIndex > stElementCount ? stElementCount : endIndex;
	int firstElement = stElementCount >= start ? start : 0;
	for (int i = firstElement; i < endIndex; i++) {
		char *mangldFnName = null, *addressOffset = null, *addressOffsetEnd =
				null;
		// find parentheses and +address offset surrounding the mangled name:
		// ./module(function+0x15c) [0x8048a6d]
		//cout << "mangled " << i << ":  " << stElements[i] << endl;
		//cout << "iterating:";
		for (char *stElementChar = stElements[i]; *stElementChar;
				++stElementChar) {
			//cout << *stElementChar;
			if (*stElementChar == '(')
				mangldFnName = stElementChar;
			else if (*stElementChar == '+')
				addressOffset = stElementChar;
			else if (*stElementChar == ')' && addressOffset) {
				addressOffsetEnd = stElementChar;
			} else if (*stElementChar == '[') {
				addressStrStart = stElementChar + 1;
			} else if (*stElementChar == ']') {
				addressStrEnd = stElementChar - 1;
				if (addressStrEnd && addressStrStart) {
					int len = addressStrEnd - addressStrStart;
					len = len < ADDRESS_STRING_LEN ? len : ADDRESS_STRING_LEN;
					memcpy(address, addressStrStart, len);
					address[len + 1] = 0;
				}
				break;
			}
		}
		//cout << endl;
		//outln("mangldFnName " << (void*)mangldFnName);
		//outln("addressOffset " << (void*)addressOffset);
		//outln("addressOffsetEnd " << (void*)addressOffsetEnd);
		if (mangldFnName && addressOffset && addressOffsetEnd
				&& mangldFnName < addressOffset) {
			*mangldFnName = *addressOffset = *addressOffsetEnd = '\0';
			mangldFnName++;
			addressOffset++;
			int status;
			char* raw = abi::__cxa_demangle(mangldFnName, unmangleBuffer,
					&unmangleBufferLength, &status);
			if (status == 0) {
				currentUnmangled = raw;
				ss << (i == firstElement ? msg.c_str() : "  at ")
						<< currentUnmangled << "   " << addressOffset;
				if (depth > 1) {
					ss << endl;
				}
			} else {
				ss << "  " << stElements[i] << " : " << mangldFnName << "()+"
						<< addressOffset << endl;
			}
		} else {
			ss << " 0-0 " << stElements[i] << "[" << mangldFnName << "]"
					<< endl;
		}
	}

	free(stElements);
	if (unmangleBuffer != currentUnmangled)
		free(currentUnmangled);
	return ss.str();
}

string print_stacktrace(unsigned int max_frames) {
	stringstream ss;
	char buff[1024];
	//fprintf(out, "stack trace:\n");
	ss << "print_stacktrace:\n";
	// storage array for stack trace address data
	void* addrlist[max_frames + 1];

	// retrieve current stack addresses
	int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

	if (addrlen == 0) {
		//fprintf(out, "  <empty, possibly corrupt>\n");
		ss << "  <empty, possibly corrupt>\n";
		return ss.str();
	}

	// resolve addresses into strings containing "filename(function+address)",
	// this array must be free()-ed
	char** symbollist = backtrace_symbols(addrlist, addrlen);

	// allocate string which will be filled with the demangled function name
	size_t funcnamesize = 256;
	char* funcname = (char*) malloc(funcnamesize);

	// iterate over the returned symbol lines. skip the first, it is the
	// address of this function.
	for (int i = 1; i < addrlen; i++) {
		char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

		// find parentheses and +address offset surrounding the mangled name:
		// ./module(function+0x15c) [0x8048a6d]
		//outln("symbollist[" << i <<"] " << symbollist[i]);
		for (char *p = symbollist[i]; *p; ++p) {
			if (*p == '(')
				begin_name = p;
			else if (*p == '+')
				begin_offset = p;
			else if (*p == ')' && begin_offset) {
				end_offset = p;
				break;
			}
		}

		if (begin_name && begin_offset && end_offset
				&& begin_name < begin_offset) {
			*begin_name++ = '\0';
			*begin_offset++ = '\0';
			*end_offset = '\0';


			// mangled name is now in [begin_name, begin_offset) and caller
			// offset in [begin_offset, end_offset). now apply
			// __cxa_demangle():

			int status;
			char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize,
					&status);
			if (status == 0) {
				funcname = ret; // use possibly realloc()-ed string
						/*
						 fprintf(out, "  %s : %s+%s\n",
						 symbollist[i], funcname, begin_offset);
						 */
				sprintf(buff, "  %s : %s+%s\n", symbollist[i], funcname,
						begin_offset);
				//outln("buff " << buff);
				ss << buff;
			} else {
				// demangling failed. Output function name as a C function with
				// no arguments.
				sprintf(buff, "  %s : %s()+%s\n", symbollist[i], begin_name,
						begin_offset);
				ss << buff;
			}
		} else {
			// couldn't parse the line? print the whole line.
			sprintf(buff, "  %s\n", symbollist[i]);
			ss << buff;
		}
	}

	free(funcname);
	free(symbollist);
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
	cout << stackString(start, depth) << endl;
}
void b_util::dumpStack(int depth) {
	cout << stackString(4, depth) << endl;
}
void b_util::dumpStack(const char * msg, int depth) {
	cout << msg << endl << stackString(4, depth) << endl;
}
void b_util::dumpStackIgnoreHere(int depth) {
	cout << stackString(5, depth) << endl;
}

void b_util::dumpStackIgnoreHere(string msg, int depth) {
	cout << msg << endl << stackString(1, depth) << endl;
}

void b_util::waitEnter() {
	char cr;
	cout << "Enter to continue:";
	cin.get(cr);		//
}

//template <> double ConjugateGradient<double>::MinPositiveValue = 4.9E-324;
//template <> float ConjugateGradient<float>::MinPositiveValue = 1.4E-45;


void b_util::execContext(uint threads, uint count, uint spacePerThread,
		dim3& dBlocks, dim3& dThreads, uint& smem) {
	if (threads % WARP_SIZE != 0) {
		outln(
				"WARN: " << threads << " is not a multiple of the warp size (32)");
	}
	ExecCaps* currCaps;
	checkCudaError(ExecCaps::currCaps(&currCaps));
	uint limitSmemThreads = currCaps->memSharedPerBlock / spacePerThread;
	threads = (int) MIN((uint)threads, limitSmemThreads);
	smem = MIN(currCaps->memSharedPerBlock, threads * spacePerThread);
	uint blocks = (count + threads - 1) / threads;

	dBlocks.y = dBlocks.z = 1;
	dBlocks.x = blocks;
	dThreads.y = dThreads.z = 1;
	dThreads.x = threads;
	totalThreads += blocks * threads;
	totalElements += count;
	if (debugExec)
		outln(
				"for " << spacePerThread << " bytes per thread, smem " << smem << " bytes, grid of " << blocks << " block(s) of " << threads << " threads");
}

const char* b_util::lastErrStr() {
	return __cudaGetErrorEnum(cudaGetLastError());
}

void b_util::dumpAnyErr(string file, int line) {
	if (cudaDeviceSynchronize() != cudaSuccess) {
		cout << __cudaGetErrorEnum(cudaGetLastError()) << " at " << file.c_str()
				<< ":" << line << endl;
	} else if (debugVerbose) {
		cout << "OK at " << file.c_str() << ":" << line << endl;
	}
}

void b_util::dumpError(cudaError_t err) {
	if (err != cudaSuccess) {
		cout << __cudaGetErrorEnum(err) << endl;
	}
}
double b_util::usedMemRatio() {
	size_t freeMemory, totalMemory;
	cudaMemGetInfo(&freeMemory, &totalMemory);
	if (debugMem)
		outln("freeMemory " << freeMemory << ", total " << totalMemory);
	return 100 * (1 - freeMemory * 1. / totalMemory);
}

void b_util::usedDmem() {
	cout << "Memory " << usedMemRatio() << "% used\n";
}


void b_util::_printCudaError(const char* file, int line, cudaError_t val) {
	if (val != cudaSuccess) {
		stringstream ss;
		ss << "CuError:  (" << val << ") " << __cudaGetErrorEnum(val) << " at "
				<< file << ":" << line << endl;
		b_util::dumpStackIgnoreHere(ss.str());
	}
}


template int util<CuMatrix<float> >::deletePtrMap<string>(
		map<string, CuMatrix<float>*>& m);
template int util<CuMatrix<double> >::deletePtrMap<string>(
		map<string, CuMatrix<double>*>& m);
template int util<CuMatrix<ulong> >::deletePtrMap<string>(
		map<string, CuMatrix<ulong>*>& m);
template int util<ExecCaps>::deletePtrMap<int>(map<int, ExecCaps*>& m);
template<typename T> template<typename K> int util<T>::deletePtrMap(
		map<K, T*>& m) {
	outln("deletePtrMap enter");
	typedef typename map<K, T*>::iterator iterator;
	iterator it = m.begin();
	while (it != m.end()) {
		delete (*it).second;
		it++;
	}
	return m.size();
}

template void util<float>::deletePtrArray<ExecCaps>(ExecCaps**,int);
template void util<double>::deletePtrArray<ExecCaps>(ExecCaps**,int);
template void util<ulong>::deletePtrArray<ExecCaps>(ExecCaps**,int);
template void util<uint>::deletePtrArray<ExecCaps>(ExecCaps**,int);
template void util<int>::deletePtrArray<ExecCaps>(ExecCaps**,int);
template<typename T> template<typename K> void util<T>::deletePtrArray(
		 K**  m, int size) {
	for (int i = 0 ; i < size; i++) {
		delete m[i];
	}
}

template void util<float>::deleteDevPtrArray<ExecCaps>(ExecCaps**,int);
template void util<double>::deleteDevPtrArray<ExecCaps>(ExecCaps**,int);
template void util<ulong>::deleteDevPtrArray<ExecCaps>(ExecCaps**,int);
template void util<uint>::deleteDevPtrArray<ExecCaps>(ExecCaps**,int);
template void util<int>::deleteDevPtrArray<ExecCaps>(ExecCaps**,int);
template<typename T> template<typename K> void util<T>::deleteDevPtrArray(
		 K**  m, int size) {
	for (int i = 0 ; i < size; i++) {
		cudaFree( m[i]);
	}
	cudaFree(m);
}



template int util<float>::deleteVector(vector<float*>&);
template int util<double>::deleteVector(vector<double*>&);
template int util<ulong>::deleteVector(vector<ulong*>&);
template int util<uint>::deleteVector(vector<uint*>&);
template int util<int>::deleteVector(vector<int*>&);
template<typename T> int util<T>::deleteVector(vector<T*>& v) {
	typedef typename vector<T*>::iterator iterator;
	iterator it = v.begin();
	while (it != v.end()) {
		delete (*it);
		it++;
	}
	return v.size();
}

template<typename T> int util<T>::cudaFreeVector(vector<T*>& v, bool device) {
	typedef typename vector<T*>::iterator iterator;
	iterator it = v.begin();
	while (it != v.end()) {
		checkCudaError(device ? cudaFree(*it) : cudaFreeHost(*it));
		it++;
	}
	return v.size();
}

void b_util::syncGpu(const char * msg) {
	if (msg)
		cout << msg << ": ";
	checkCudaError(cudaDeviceSynchronize());
}

template<typename T> cudaError_t util<T>::copyRange(T* targ, ulong targOff,
		T* src, ulong srcOff, ulong count) {
	cudaError_t err = cudaMemcpy(targ + targOff, src + srcOff,
			count * sizeof(T), cudaMemcpyHostToHost);
	CuMatrix<T>::HHCopied++;
	CuMatrix<T>::MemHhCopied += count * sizeof(T);

	checkCudaError(err);
	return err;
}

template<typename T> T util<T>::sumCPU(T* vals, ulong count) {
	T sum = 0;
	while (count-- != 0) {
		sum += vals[count - 1];
	}
	return sum;
}

CuTimer::CuTimer(cudaStream_t stream) : evt_start(0), evt_stop(0), stream(0), status(ready) {
	checkCudaError(cudaEventCreate(&evt_start));
	checkCudaError(cudaEventCreate(&evt_stop));
	this->stream = stream;
	if(checkDebug(debugExec))outln("created CuTimer " << this << " (stream " << stream << ", evt_start " << evt_start << ", evt_stop " << evt_stop << ")");
}

CuTimer::~CuTimer() {
	//outln("~CuTimer " << this);
	checkCudaError(cudaEventDestroy(evt_start));
	checkCudaError(cudaEventDestroy(evt_stop));
}

void CuTimer::start() {
	//outln("starting CuTimer " << this << " (stream " << stream << ")");
	if (status != ready) {
		dthrow(timerAlreadyStarted());
	}
	status = started;
	checkCudaError(cudaEventRecord(evt_start, stream));
}

float CuTimer::stop() {
	if(checkDebug(debugExec))outln("stopping CuTimer " << this << " stream " << stream << "; currDev " << ExecCaps::currDev());
	if(checkDebug(debugExec))outln("stopping CuTimer evt_stop " << evt_stop);
	if (status != started) {
		dthrow(timerNotStarted());
	}
	status = ready;
	checkCudaError(cudaEventRecord(evt_stop, stream));
	//checkCudaError(cudaGetLastError());
	checkCudaError(cudaEventSynchronize(evt_stop));
	float exeTimeMs;
	checkCudaError(cudaEventElapsedTime(&exeTimeMs, evt_start, evt_stop));
	return exeTimeMs;
}

template struct util<float> ;
template struct util<double> ;
template struct util<ulong> ;
template struct util<int> ;
template struct util<uint> ;
auto is_odd = [](int n) {return n%2==1;};


void testLambada() {
	   using namespace std;

	   // Create a list of integers with a few initial elements.
	   list<int> numbers;
	   numbers.push_back(13);
	   numbers.push_back(17);
	   numbers.push_back(42);
	   numbers.push_back(46);
	   numbers.push_back(99);

	   // Use the find_if function and a lambda expression to find the
	   // first even number in the list.
	   const list<int>::const_iterator result =
	      find_if(numbers.begin(), numbers.end(), is_odd);
	         //[](int n) { return (n % 2) == 0; });

	   // Print the result.
	   if (result != numbers.end())
	   {
	       cout << "The first even number in the list is "
	            << (*result)
	            << "."
	            << endl;
	   }
	   else
	   {
	       cout << "The list contains no even numbers."
	            << endl;
	   }

}

/*

class A {
public : virtual void foo(int) final { std::cout << "A.foo" << std::endl;}
};


class B : public A{
public : virtual void foo(int) override { std::cout << "A.foo" << std::endl;}
};
*/
typedef float (*MyFuncPtrType)(int, const char *);

float phoo(int x, const char* y) {
	printf("phoo x %d, y %s\n",x,y);
	return 1.0f/x;
}
float bphoo(int x, const char* y) {
	printf("bphoo x %d, y %s\n",x,y);
	return 20.f * x;
}

template <typename T, MyFuncPtrType func, int StateDim> void doSumpin() {
	(*func)(StateDim, b_util::unmangl(typeid(func).name()).c_str() );
}

int somethd() {
	doSumpin<int, bphoo, 2>();
	doSumpin<double, phoo, 55>();
	return 0;
}
