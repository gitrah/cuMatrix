/*
 * CuCuMatrixFile.cc
 *
 *  Created on: Dec 27, 2013
 *      Author: reid
 */

#include "CuMatrix.h"
#include <cstdlib>
#include <octave/oct.h>
#ifdef CuMatrix_UseOmp
#include <omp.h>
#endif

using std::ofstream;
using std::ios;

template<typename T> __host__ void CuMatrix<T>::toFile( const char* fileName ) const {
	toFile(fileName, *this);
}

template void CuMatrix<float>::toFile(const char*) const;
template void CuMatrix<double>::toFile(const char*) const;
template void CuMatrix<ulong>::toFile(const char*) const;

template<typename T> void CuMatrix<T>::toFile(const char* fileName, const CuMatrix<T>& o )  {
	ofstream ofs(fileName, ios::binary);
	const CuMatrix<T>* ptr = &o;
	ofs.write((char *)ptr, sizeof(o));
	if(o.elements) {
		uint l = o.m * o.n;
		ofs.write((char*)o.elements, l*sizeof(T));
		outln("wrote " << l << " elements");
	}
	ofs.close();
}

string my_itoa(int value, int base = 10){

	int i = 30;

	string buf = "";

	for(; value && i ; --i, value /= base) buf = "0123456789abcdef"[value % base] + buf;
	return buf;
}

template void CuMatrix<float>::toOctaveFile(const char*, const CuMatrix<float>& o);
template void CuMatrix<double>::toOctaveFile(const char*, const CuMatrix<double>& o);
template void CuMatrix<ulong>::toOctaveFile(const char*, const CuMatrix<ulong>& o);
template<typename T> void CuMatrix<T>::toOctaveFile(const char* name, const CuMatrix<T>& o )  {
	assert(o.synchedQ());
	string ext = ".txt";
	string fileName = name + ext;
	ofstream ofs;
	ofs.open(fileName.c_str());
	string nameStr = "# name: ";
	string typeStr = "\n# type: matrix\n";
	ofs << nameStr;
	ofs << name;
	ofs << typeStr;
	ofs << "# rows: ";
	ofs << my_itoa(o.m);
	ofs << "\n# columns: ";
	ofs << my_itoa(o.n);
	ofs << "\n";
	for(int i = 0; i < o.m; i++) {
		ofs << o.rowStr(i) ;
	}
	ofs.close();
	outln("wrote " << o.m << " rows");
}


template <typename T> CuMatrix<T> CuMatrix<T>::fromFile(const char* fileName) {
	string in;
	outln(" CuMatrix<T>::fromFile(" << fileName << ")");
	ifstream ifs(fileName, ios::binary);
	CuMatrix<T> tmp;
	ifs.read((char *)&tmp, sizeof(tmp));
	outln("read in temp " << tmp.toShortString());
	if(tmp.elements ) {
		outln("erasing stale pointer " << tmp.elements);
		tmp.elements = null;
	}
	if(tmp.tiler.hasDmemQ()) {
		tmp.tiler.clear();
	}
	CuMatrix<T> res(tmp.m, tmp.n,true,false);
	uint l = res.m * res.n;
	ifs.read((char *)res.elements, l * sizeof(T));
	outln("read " << l << " elements");
	ifs.close();
	res.invalidateDevice();
	return res;
}
template CuMatrix<float> CuMatrix<float>::fromFile(const char*);
template CuMatrix<double> CuMatrix<double>::fromFile(const char*);

template <typename T> vector< CuMatrix<T> > CuMatrix<T>::fromFileN(const char* fileName) {
	string in;
	outln(" CuMatrix<T>::fromFileN(const char* fileName)");
	ifstream ifs(fileName, ios::binary);
	CuMatrix<T> tmp;
	vector< CuMatrix<T> > ret;
	do{
		ifs.read((char *)&tmp, sizeof(tmp));
		CuMatrix<T> res(tmp.m, tmp.n, true,false);
		uint l = res.m * res.n;
		ifs.read((char *)res.elements, l * sizeof(T));
		outln("read " << l << " elements");
		ret.push_back(res);
		outln("ret now has " << ret.size());

	} while(ifs.peek() != EOF);
	ifs.close();
	return ret;
}

template vector< CuMatrix<float> > CuMatrix<float>::fromFileN(const char*);
template vector< CuMatrix<double> > CuMatrix<double>::fromFileN(const char*);


template<typename T> void CuMatrix<T>::parseDataLine(string line, T* elementData,
		int currRow, int rows, int cols, bool colMajor) {
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
template<typename T> void CuMatrix<T>::parseCsvDataLine(const CuMatrix<T>* x,
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

const string Name("name:");
const string Type("type:");
const string Rows("rows:");
const string Columns("columns:");

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
void addOctaveObjectDim3(map<string, dim3>& theMap,
		typename map<string, dim3>::iterator& it, string name,
		const OctaveType& elementType, dim3& currMat) {
	switch (elementType) {
	case scalar:
	case matrix: {
		outln(
				"adding " << ot2str(elementType).c_str() << " '" << name.c_str() << "' of dims " << currMat.x << ", " << currMat.y << ", " << currMat.z);
		//outln( "created matrix of dims " << m.getRows() << ", " << m.getCols());
		theMap.insert(it, pair<string, dim3>(name, currMat));
		break;
	}
	default: {
		outln("received unknown type for " << name.c_str());
		break;
	}
	}
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

template map<string, CuMatrix<float>*> CuMatrix<float>::parseOctaveDataFile(const char*, bool, bool);
template map<string, CuMatrix<double>*> CuMatrix<double>::parseOctaveDataFile(const char*, bool, bool);
template map<string, CuMatrix<ulong>*> CuMatrix<ulong>::parseOctaveDataFile(const char*, bool, bool);

template<typename T> map<string, CuMatrix<T>*> CuMatrix<T>::parseOctaveDataFile(
		const char * path, bool colMajor, bool matrixOwnsBuffer) {
#ifdef CuMatrix_UseOmp
	outln(	" reading " << path );
#else
	outln(	"thread id " << omp_get_thread_num() << " reading " << path );
#endif
	map<string, CuMatrix<T>*> dataMap;
	typedef typename map<string, CuMatrix<T>*>::iterator iterator;
	iterator it = dataMap.begin();
	OctaveType elementType = unknown;
	unsigned long idx = string::npos;
	int rows = 0, cols = 0, currRow = 0;
	uint rowTick = 0;
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
					if(checkDebug(debugFile))outln(
							"adding " << name.c_str() << " took " << b_util::diffclock(clock(),lastTime) << "s");
					if(checkDebug(debugFile))outln("now " << dataMap.size() << " elements in dataMap");
					currRow = 0;
				}
				name = trim_copy(line.substr(idx + Name.size()));
				if(checkDebug(debugFile))outln("found " << name.c_str());
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
					if(checkDebug(debugFile))outln(
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
						if(checkDebug(debugFile))outln(name.c_str() << " has rows " << rows);
						rowTick = rows / 100;
						if(rowTick > 1000) {
							rowTick = rowTick - (rowTick % 1000);
						} else if(rowTick > 100) {
							rowTick = rowTick - (rowTick % 100);
						} else if(rowTick > 50) {
							rowTick = rowTick - (rowTick % 50);
						} else if(rowTick > 10){
							rowTick = 10;
						} else {
							rowTick = 1;
						}
						rowsSet = true;
					} else {
						idx = line.find(Columns);
						if (idx != string::npos) {
							stringstream(
									trim_copy(
											line.substr(idx + Columns.size())))
									>> cols;
							if(checkDebug(debugFile))outln(name.c_str() << " has cols " << cols);
							colsSet = true;
						}
					}
					if (rowsSet && colsSet) {
						if(checkDebug(debugFile))outln(
								"creating buffer for " << name.c_str() << " of size " << (rows * cols));
						currMat = new CuMatrix<T>(rows, cols, matrixOwnsBuffer,true);
						currMat->invalidateDevice();
						if(checkDebug(debugFile))outln("got currMat " << currMat->toShortString());
						if(checkDebug(debugFile))outln("got buffer " << currMat->elements);
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
				CuMatrix<T>::parseDataLine(line, currMat->elements, currRow, rows,
						cols, colMajor);
				currRow++;
				if(checkDebug(debugFile))
					if ((currRow % rowTick) == 0) {
						clock_t now = clock();
						if (lastChunk != 0) {
							deltaMs = b_util::diffclock(now, lastChunk);
							if(checkDebug(debugFile))outln(
									"on " << currRow << "/" << rows << " at " << (deltaMs / 100) << " ms/row");
						} else {
							if(checkDebug(debugFile))outln("on " << currRow << "/" << rows);
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
				if(checkDebug(debugFile))outln("unknown type " << elementType);
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

template CuMatrix<float>& CuMatrix<float>::getMatrix(std::map<std::string, CuMatrix<float>*, std::less<std::string>, std::allocator<std::pair<std::string const, CuMatrix<float>*> > >, const char*);
template CuMatrix<double>& CuMatrix<double>::getMatrix(std::map<std::string, CuMatrix<double>*, std::less<std::string>, std::allocator<std::pair<std::string const, CuMatrix<double>*> > >, const char*);
template CuMatrix<ulong>& CuMatrix<ulong>::getMatrix(std::map<std::string, CuMatrix<ulong>*, std::less<std::string>, std::allocator<std::pair<std::string const, CuMatrix<unsigned long>*> > >, const char*);


template<typename T> map<string,dim3> CuMatrix<T>::parseOctaveMatrixSizes(const char * path ) {
	outln("parsing path " << path );
	map<string, dim3> dataMap;
	typedef typename map<string, dim3>::iterator iterator;
	iterator it = dataMap.begin();
	OctaveType elementType = unknown;
	unsigned long idx = string::npos;
	//int rows = 0, cols = 0,
	int currRow = 0;
	uint rowTick = 0;
	clock_t lastChunk = 0, lastTime = 0;
	string name = "";
	double deltaMs = 0;
	dim3 currMat;
	bool colsSet = false, rowsSet = false;
	vector<string> lines = b_util::readLines(path);
	outln("path " << path << ", had " << lines.size() << " lines");

	vector<string>::iterator itLines = lines.begin();
	while (itLines != lines.end()) {
		string line = trim_copy(*itLines);
		if (line.find("#") == 0) { // start of var declaration
			idx = line.find(Name);
			if (idx != string::npos) {
				if (name.length() > 0 && elementType != unknown) {
					// add previous object
					addOctaveObjectDim3(dataMap, it, name, elementType, currMat);
					if(checkDebug(debugFile))outln(
							"adding " << name.c_str() << " took " << b_util::diffclock(clock(),lastTime) << "s");
					if(checkDebug(debugFile))outln("now " << dataMap.size() << " elements in dataMap");
					currRow = 0;
				}
				name = trim_copy(line.substr(idx + Name.size()));
				if(checkDebug(debugFile))outln("found " << name.c_str());
				lastTime = clock();
				currMat.x = 0; currMat.y= 0; currMat.z= 0;
				colsSet = rowsSet = false;
				elementType = unknown;
				currMat = dim3(0,0,0);
			} else {
				idx = line.find(Type);
				if (idx != string::npos) {
					elementType = str2OctaveType(
							trim_copy(line.substr(idx + Type.size())));
					if(checkDebug(debugFile))outln(
							name.c_str() << " has type " << ot2str(elementType).c_str());
					if (elementType == scalar) {
						currMat.x=0;currMat.y=0; currMat.z=0;
						currMat.x = 1; currMat.y= 1; currMat.z= 1;
					}
				} else {
					idx = line.find(Rows);
					if (idx != string::npos) {
						stringstream(trim_copy(line.substr(idx + Rows.size())))
								>> currMat.x;
						if(checkDebug(debugFile))outln(name.c_str() << " has rows " << currMat.x);
						rowTick = currMat.x / 100;
						if(rowTick > 1000) {
							rowTick = rowTick - (rowTick % 1000);
						} else if(rowTick > 100) {
							rowTick = rowTick - (rowTick % 100);
						} else if(rowTick > 50) {
							rowTick = rowTick - (rowTick % 50);
						} else if(rowTick > 10){
							rowTick = 10;
						} else {
							rowTick = 1;
						}
						rowsSet = true;
					} else {
						idx = line.find(Columns);
						if (idx != string::npos) {
							stringstream(
									trim_copy(
											line.substr(idx + Columns.size())))
									>> currMat.y;
							currMat.z = currMat.y;
							if(checkDebug(debugFile))outln(name.c_str() << " has cols " << currMat.y);
							colsSet = true;
						}
					}
					if (rowsSet && colsSet) {
						if(checkDebug(debugFile))outln("got currMat " << b_util::pd3( currMat));
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
				// it's a data line (row), so just skip until next matrix etc is declared
				currRow++;
				if(checkDebug(debugFile))
					if ((currRow % rowTick) == 0) {
						clock_t now = clock();
						if (lastChunk != 0) {
							deltaMs = b_util::diffclock(now, lastChunk);
							if(checkDebug(debugFile))outln(
									"on " << currRow << "/" << currMat.x << " at " << (deltaMs / 100) << " ms/row");
						} else {
							if(checkDebug(debugFile))outln("on " << currRow << "/" << currMat.x);
						}
						lastChunk = now;
					}
				break;
			}
			case scalar: {
				//stringstream(trim_copy(line)) >> currMat->elements[0];
				outln("skipping scalar type ");
				break;
			}

			default:
				if(checkDebug(debugFile))outln("unknown type " << elementType);
				break;
			}
		}
		//outln("next");
		itLines++;
	}

	// add last obj
	if (!name.empty() && elementType != unknown) {
		addOctaveObjectDim3(dataMap, it, name, elementType, currMat);
	}

	return dataMap;

}

template map<string,dim3> CuMatrix<float>::parseOctaveMatrixSizes(const char * path ) ;
template map<string,dim3> CuMatrix<double>::parseOctaveMatrixSizes(const char * path ) ;
template map<string,dim3> CuMatrix<ulong>::parseOctaveMatrixSizes(const char * path ) ;

template<typename T> CuMatrix<T>& CuMatrix<T>::getMatrix(std::map<std::string, CuMatrix<T>*> map, const char* key) {
	assert(key != null);
	CuMatrix<T>* refPtr = map[key];
	assert(refPtr != null);
	return *refPtr;
}
template map<string, CuMatrix<float>*> CuMatrix<float>::parseCsvDataFile(const char*, const char*, bool, bool, bool);
template map<string, CuMatrix<double>*> CuMatrix<double>::parseCsvDataFile(const char*, const char*, bool, bool, bool);
template map<string, CuMatrix<ulong>*> CuMatrix<ulong>::parseCsvDataFile(const char*, const char*, bool, bool, bool);

template<typename T> map<string, CuMatrix<T>*> CuMatrix<T>::parseCsvDataFile(
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
	int cols = 0;//int rows = 0, cols = 0, currRow = 0;

	//clock_t lastChunk = 0, lastTime = 0;
	string name = "";
	//double deltaMs = 0;
	//bool colsSet = false, rowsSet = false;
	if(checkDebug(debugFile))outln("reading " << path);
	vector<string> lines = b_util::readLines(path);
	int lineCount = lines.size();
	if(checkDebug(debugFile))outln(
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
	if(checkDebug(debugFile))outln("mat " << mat->toShortString());
	while (itLines != lines.end()) {

		string line = trim_copy(*itLines);
		//const char* cline = line.c_str();
		if (!line.empty()) {
			//outln("expected data line");
			switch (elementType) {
			case matrix: {
				// it's a data line (row)
				CuMatrix<T>::parseCsvDataLine(mat, currLine, line, sepChars);
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
template<typename T> int CuMatrix<T>::releaseFromMap(map<string, CuMatrix<T>*>& theMap) {
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




