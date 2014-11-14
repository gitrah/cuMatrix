/*
 * cu_util.cc
 *
 *  Created on: Jul 19, 2014
 *      Author: reid
 */


#include <list>
#include <algorithm>
#include <iostream>
#include <iostream>
#include <string>
#include <algorithm>
#include <iterator>
#include <assert.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <csignal>
#include <ctime>
#include <sys/time.h>

#include "debug.h"
#include "cu_util.h"
#include "caps.h"
#include "MatrixExceptions.h"

/*

template<typename T> string cu_util<T>::pdm(const DMatrix<T>& md) {
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

template<typename T> int cu_util<T>::release(map<string, CuMatrix<T>*>& theMap) {
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

template<typename T> void cu_util<T>::parseDataLine(string line, T* elementData,
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
template<typename T> void cu_util<T>::parseCsvDataLine(const CuMatrix<T>* x,
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

template<typename T> map<string, CuMatrix<T>*> cu_util<T>::parseOctaveDataFile(
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

template<typename T> map<string, CuMatrix<T>*> cu_util<T>::parseCsvDataFile(
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

template int cu_util<CuMatrix<float> >::deletePtrMap<string>(
		map<string, CuMatrix<float>*>& m);
template int cu_util<CuMatrix<double> >::deletePtrMap<string>(
		map<string, CuMatrix<double>*>& m);
template int cu_util<CuMatrix<ulong> >::deletePtrMap<string>(
		map<string, CuMatrix<ulong>*>& m);
*/
